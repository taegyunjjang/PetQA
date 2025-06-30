import os
os.environ['VLLM_USE_V1'] = '0'  # logit processor 사용 목적
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import pandas as pd
from tqdm import tqdm
import argparse
from utils.utils import (
    MODEL_MAPPING, setup_logging, save_json, load_json,
    load_prompt, load_results, load_environment
)
from vllm import LLM, SamplingParams
from etc.blocker_numpy import blocker
import torch


def load_model_and_tokenizer(model_path):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.99,
    )
    
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return llm, tokenizer

def load_fewshot_examples(env):
    fewshot_df = pd.read_json(env["fewshot_examples_path"])
    fewshot_map = {row["q_id"]: row for _, row in fewshot_df.iterrows()}
    return fewshot_map

def build_fewshot_examples(sample, shot, input_format):
    examples = ""
    for i in range(int(shot)):
        title = sample["similar_questions"][i]["title"]
        content = sample["similar_questions"][i]["content"]
        question = sample["similar_questions"][i]["preprocessed_question"]
        answer = sample["similar_questions"][i]["preprocessed_answer"]
        
        if input_format == "raw":
            examples += f"제목: {title}\n본문: {content}\n답변: {answer}\n\n"
        else:
            examples += f"질문: {question}\n답변: {answer}\n\n"
    return examples.strip()

def get_prompts(test_data, start_idx, tokenizer, shot, fewshot_map, input_format, env):
    prompts = []
    data_to_process = test_data[start_idx:]
    
    if shot == "0":
        system_prompt = load_prompt(env["system_zeroshot_prompt_path"])
    else:
        base_system_prompt = load_prompt(env["system_fewshot_prompt_path"])
        
    if input_format == "raw":
        base_user_prompt = load_prompt(env["user_raw_input_prompt_path"])
    else:
        base_user_prompt = load_prompt(env["user_processed_input_prompt_path"])
    
    for item in data_to_process:
        if shot == "0":
            pass
        else:
            sample = fewshot_map.get(item["q_id"])
            fewshot_examples = build_fewshot_examples(sample, shot, input_format)
            system_prompt = base_system_prompt.format(fewshot_examples=fewshot_examples)
            
        if input_format == "raw":
            user_prompt = base_user_prompt.format(title=item["title"], content=item["content"])
        else:
            user_prompt = base_user_prompt.format(question=item["preprocessed_question"])
            
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
    
    return prompts, data_to_process 

def generate_answers_sequential(
    llm, tokenizer, test_data, results, start_idx, 
    env, shot, input_format, output_path, logger
):
    fewshot_map = None
    if shot != "0":
        logger.info("fewshot 프롬프트 생성 중")
        fewshot_map = load_fewshot_examples(env)
        
    prompts, data_to_process = get_prompts(
        test_data, 
        start_idx, 
        tokenizer, 
        shot,
        fewshot_map,
        input_format, 
        env
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None
    )
    
    # 중국어 토큰 생성 방지
    def _logits_processor(input_ids, logits):
        return blocker(tokenizer, input_ids, logits)
    sampling_params.logits_processors=[_logits_processor]
    
    for (prompt, item) in tqdm(zip(prompts, data_to_process), total=len(prompts), desc="답변 생성 중"):
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        generated_answer = outputs[0].outputs[0].text.strip()
        item["generated_answer"] = generated_answer
        results.append(item)
        save_json(results, output_path)

    logger.info("답변 생성 완료")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="답변 생성")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="exaone-3.5-7.8b")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--answer_type", type=str, choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--use_finetuned_model", action="store_true")
    args = parser.parse_args()
    
    env = load_environment()    
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"SHOT: {args.shot}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"USE FINETUNED MODEL: {args.use_finetuned_model}")
    
    if args.use_finetuned_model:
        output_path = os.path.join(env["generated_answers_dir"],
                               f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}.json")
        checkpoint_dir = os.path.join(env["checkpoint_dir"],
                                      f"{args.model_name}_{args.input_format}_{args.answer_type}")
        best_model_dir = os.path.join(checkpoint_dir, "best_model")
        llm, tokenizer = load_model_and_tokenizer(best_model_dir)
    else:
        output_path = os.path.join(env["generated_answers_dir"],
                               f"output_{args.model_name}_{args.shot}_{args.input_format}.json")
        llm, tokenizer = load_model_and_tokenizer(MODEL_MAPPING[args.model_name])
        
    results, start_idx = load_results(output_path)
    test_data = load_json(env["test_data_path"])
    
    generate_answers_sequential(
        llm, tokenizer, test_data, results, start_idx, 
        env, args.shot, args.input_format, output_path, logger
    )