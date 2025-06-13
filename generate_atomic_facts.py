import json
import argparse
import pandas as pd
import os
os.environ['VLLM_USE_V1'] = '0'  # logit processor 사용 목적
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
import logging
import re
from blocker_numpy import blocker


MODEL_MAPPING = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "hcx-seed-3b": "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    
    # judge model
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "exaone-3.5-32b": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
}

def setup_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

def load_environment(args):
    MODEL_NAME = args.model_name
    SHOT = args.shot
    USE_RAW_FORMAT = args.use_raw_format
    USE_FINETUNED_MODEL = args.use_finetuned_model
    EXPERT_TYPE = args.expert_type
    ANIMAL_TYPE = args.animal_type
    
    
    SUFFIX = "raw" if USE_RAW_FORMAT else "preprocessed"
    FT = "_petqa" if USE_FINETUNED_MODEL else ""
    
    
    input_path = f"data/TEST/{EXPERT_TYPE}/{ANIMAL_TYPE}/output_{MODEL_NAME}{FT}_{SHOT}_{SUFFIX}.json"
    if args.prepare_gold_facts:
        input_path = f"data/TEST/{EXPERT_TYPE}/{ANIMAL_TYPE}/{EXPERT_TYPE}_{ANIMAL_TYPE}.json"
    output_path = f"data/TEST/{EXPERT_TYPE}/{ANIMAL_TYPE}/atomic_facts/{MODEL_NAME}{FT}_{SHOT}_{SUFFIX}.json"
    if args.prepare_gold_facts:
        output_path = f"data/TEST/{EXPERT_TYPE}/{ANIMAL_TYPE}/atomic_facts/gold_facts.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    
    system_prompt_path = f"prompt/system_atomic.txt"
    user_prompt_path = f"prompt/user_atomic.txt"
    
    return {
        "input_path": input_path,
        "output_path": output_path,
        "system_prompt_path": system_prompt_path,
        "user_prompt_path": user_prompt_path,
    }

def load_df(file_path, prepare_gold_facts):
    df = pd.read_json(file_path)
    ids = df['id']
    answers = df['preprocessed_answer'] if prepare_gold_facts else df['generated_answer']
    return ids, answers

def load_prompt(prompt_path):
    with open(prompt_path, encoding="utf-8") as file:
        return file.read()

def get_prompts(env):
    system_prompt = load_prompt(env["system_prompt_path"])
    user_prompt = load_prompt(env["user_prompt_path"])
    return system_prompt, user_prompt

def load_results(env, logger):
    output_path = env["output_path"]
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        logger.info(f"{start_idx}개까지 처리됨. 이어서 시작")
    else:
        results = []
        start_idx = 0
        logger.info("새로 시작")
    return results, start_idx

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
    system_prompt, base_user_prompt = get_prompts(env)
    
    ids_to_process = ids[start_idx:]
    prompts = []
    data_to_process = answers[start_idx:]
    
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
    
    return ids_to_process, prompts, data_to_process

def parse_atomic_facts_from_output(output):
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
            print(f"- 로 시작하지 않는 줄: {line}")
            pass
            
    return atomic_facts
    

def generate_atomic_facts(
    llm, tokenizer, ids, answers, results, start_idx, 
    env, logger
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
    
    for (current_id, prompt, item) in tqdm(zip(ids_to_process, prompts, data_to_process), 
                                           total=len(prompts), desc="atomic facts 생성 중"):
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        raw_output = outputs[0].outputs[0].text.strip()
        atomic_facts = parse_atomic_facts_from_output(raw_output)
        
        result = {
            "id": int(current_id),
            "sentence": item,
            "atomic_facts": atomic_facts
        }
        results.append(result)
        
        with open(env["output_path"], "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("atomic facts 생성 완료")

def main(args):
    logger = setup_logging()
    
    env = load_environment(args)
    
    if args.prepare_gold_facts:
        logger.info(f"PREPARE GOLD FACTS: {args.prepare_gold_facts}")
        logger.info(f"GOLD FACTS PATH: {env['output_path']}")
        
    else:
        logger.info(f"PREPARE GOLD FACTS: {args.prepare_gold_facts}")
        logger.info(f"INPUT PATH: {env['input_path']}")
        logger.info(f"OUTPUT PATH: {env['output_path']}")
    
    llm, tokenizer = load_model_and_tokenizer(args.judge_model_name)
    ids, answers = load_df(env["input_path"], args.prepare_gold_facts)
    results, start_idx = load_results(env, logger)
    
    generate_atomic_facts(
        llm, tokenizer, ids, answers, results, start_idx, 
        env, logger
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate atomic facts")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gpt-4o-mini")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_raw_format", action="store_true")
    parser.add_argument("--prepare_gold_facts", action="store_true")
    parser.add_argument("--expert_type", choices=["expert", "nonexpert"], default="expert")
    parser.add_argument("--animal_type", choices=["cat", "dog"], default="dog")
    parser.add_argument("--judge_model_name", type=str, default="exaone-3.5-32b", choices=list(MODEL_MAPPING.keys()))
    args = parser.parse_args()
    
    main(args)