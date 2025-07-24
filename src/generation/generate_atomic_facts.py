import os
os.environ['VLLM_USE_V1'] = '0'  # logit processor 사용 목적
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from tqdm import tqdm
import argparse
from utils.utils import (
    MODEL_MAPPING, setup_logging, save_json,
    load_prompt, load_environment
)
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from etc.blocker_numpy import blocker
from pydantic import BaseModel
import torch
import pandas as pd
import json

class AtomicFacts(BaseModel):
    atomic_facts: list[str]


def load_df(file_path, prepare_gold_facts):
    df = pd.read_json(file_path)
    ids = list(zip(df['q_id'], df['a_id']))
    questions = df["preprocessed_question"]
    if prepare_gold_facts:
        answers = df['preprocessed_answer']
    else:
        answers = df['generated_answer']
    return ids, questions, answers

def load_model_and_tokenizer(model_name):
    model_path = MODEL_MAPPING[model_name]
    
    # vLLM 모델 로드
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.99,
        max_model_len=4096,
    )
    
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return llm, tokenizer

def prepare_prompts(questions, answers, tokenizer, env):
    base_user_prompt = load_prompt(env["user_atomic_prompt_path"])
    prompts = []

    for question, answer in zip(questions, answers):
        user_prompt = base_user_prompt.format(question=question, answer=answer)
        messages = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts

def generate_atomic_facts(
    llm, tokenizer, ids, questions, answers,
    env, output_path, logger
):
    results = []
    prompts = prepare_prompts(
        questions,
        answers,
        tokenizer, 
        env
    )
    
    fallback_output = {
        "atomic_facts": []
    }
    
    schema = {
        "type": "object",
        "properties": {
            "atomic_facts": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["atomic_facts"],
        "additionalProperties": False
    }
    
    guided = GuidedDecodingParams(
        json=schema,
        disable_any_whitespace=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        seed=42,
        max_tokens=2048,
        guided_decoding=guided,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None
    )
    
    # 중국어 토큰 생성 방지
    def _logits_processor(input_ids, logits):
        return blocker(tokenizer, input_ids, logits)
    sampling_params.logits_processors=[_logits_processor]
    
    batch_size = 100
    for i in tqdm(range(0, len(prompts), batch_size), desc="atomic facts 생성 중"):
        batch_prompts = prompts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_questions = questions[i : i + batch_size]
        batch_answers = answers[i : i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        for j, output_item in enumerate(outputs):
            current_id = batch_ids[j]
            json_string = output_item.outputs[0].text.strip()
            try:
                parsed_data = json.loads(json_string)
                atomic_facts = parsed_data["atomic_facts"]
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print(f"원본(json_string[:100]): \n{json_string[:100]}")
                print(f"원본(json_string[-100:]): \n{json_string[-100:]}")
                parsed_data = fallback_output
                atomic_facts = parsed_data["atomic_facts"]
            result = {
                "q_id": current_id[0],
                "a_id": current_id[1],
                "question": batch_questions[j],
                "answer": batch_answers[j],
                "atomic_facts": atomic_facts
            }
            results.append(result)
        save_json(results, output_path)
        
    logger.info("atomic facts 생성 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate atomic facts")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gpt-4o-mini")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--answer_type", type=str, choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--prepare_gold_facts", action="store_true")
    parser.add_argument("--judge_model_name", type=str, default="exaone-3.5-32b")
    args = parser.parse_args()
    
    env = load_environment()
    
    if args.prepare_gold_facts:
        input_path = env["test_data_path"]
        output_path = env["gold_facts_path"]
    else:
        if args.use_finetuned_model:
            endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}.json"
        else:
            endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}.json"
        input_path = os.path.join(env["generated_answers_dir"], endpoint)
        output_path = os.path.join(env['atomic_facts_dir'], endpoint)
    
    logger = setup_logging()
    logger.info(f"JUDGE MODEL NAME: {args.judge_model_name}")
    logger.info(f"PREPARE GOLD FACTS: {args.prepare_gold_facts}")
    logger.info(f"INPUT PATH: {input_path}")
    logger.info(f"OUTPUT PATH: {output_path}")
    
    llm, tokenizer = load_model_and_tokenizer(args.judge_model_name)
    ids, questions, answers = load_df(input_path, args.prepare_gold_facts)
    
    generate_atomic_facts(
        llm, tokenizer, ids, questions, answers,
        env, output_path, logger
    )