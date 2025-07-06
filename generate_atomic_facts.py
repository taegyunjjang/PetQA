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
    load_prompt, load_results, load_environment
)
from vllm import LLM, SamplingParams
from etc.blocker_numpy import blocker
import torch
import pandas as pd


def load_df(file_path, prepare_gold_facts):
    df = pd.read_json(file_path)
    ids = list(zip(df['q_id'], df['a_id']))
    if prepare_gold_facts:
        answers = df['preprocessed_answer']
    else:
        answers = df['generated_answer']
    return ids, answers

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

def prepare_prompts(answers, start_idx, tokenizer, env, ids):
    system_prompt = load_prompt(env["system_atomic_prompt_path"])
    base_user_prompt = load_prompt(env["user_atomic_prompt_path"])
    
    ids_to_process = ids[start_idx:]
    prompts = []
    data_to_process = answers[start_idx:].reset_index(drop=True)
    
    for item in data_to_process:
        user_prompt = base_user_prompt.format(sentence=item)
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(prompt)
    
    return ids_to_process, prompts, data_to_process.tolist()

def parse_atomic_facts_from_output(output, logger):
    lines = output.strip().split('\n')
    atomic_facts = []

    start_index = 0
    if lines and lines[0].startswith("문장:"):
        start_index = 1

    for i in range(start_index, len(lines)):
        line = lines[i].strip()
        if line.startswith('- '):
            fact = line[2:].strip()
            if fact and fact[-1] != '.':
                fact += '.'
            atomic_facts.append(fact)
        elif line:
            logger.info(f"- 로 시작하지 않는 줄: {line}")
            pass
            
    return atomic_facts

def generate_atomic_facts(
    llm, tokenizer, ids, answers, results, start_idx, 
    env, output_path, logger
):
    ids_to_process, prompts, data_to_process = prepare_prompts(
        answers,
        start_idx, 
        tokenizer, 
        env,
        ids
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2048,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None
    )
    
    # 중국어 토큰 생성 방지
    def _logits_processor(input_ids, logits):
        return blocker(tokenizer, input_ids, logits)
    sampling_params.logits_processors=[_logits_processor]
    
    batch_size = 100
    for i in tqdm(range(0, len(prompts), batch_size), desc="atomic facts 생성 중"):
        batch_prompts = prompts[i : i + batch_size]
        batch_ids = ids_to_process[i : i + batch_size]
        batch_data = data_to_process[i : i + batch_size]

        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        for j, output_item in enumerate(outputs):
            current_id = batch_ids[j]
            item = batch_data[j]
            raw_output = output_item.outputs[0].text.strip()
            atomic_facts = parse_atomic_facts_from_output(raw_output, logger)
            
            result = {
                "q_id": current_id[0],
                "a_id": current_id[1],
                "sentence": item,
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
    parser.add_argument("--judge_model_name", type=str, default="exaone-3.5-32b", choices=list(MODEL_MAPPING.keys()))
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
    ids, answers = load_df(input_path, args.prepare_gold_facts)
    results, start_idx = load_results(output_path)
    
    generate_atomic_facts(
        llm, tokenizer, ids, answers, results, start_idx, 
        env, output_path, logger
    )