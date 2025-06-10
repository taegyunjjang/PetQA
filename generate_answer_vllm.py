"""vLLM을 활용하여 답변 생성"""

import json
import os
os.environ['VLLM_USE_V1'] = '0'  # logit processor 사용 목적
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import logging

from blocker_numpy import blocker


MODEL_MAPPING = {
    "gemma-3-4b": "google/gemma-3-4b-it",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "hcx-seed-3b": "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
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

def load_environment(model_name, shot, use_raw_format, use_fintuned_model):
    suffix = "raw" if use_raw_format else "preprocessed"
    ft = "_petqa" if use_fintuned_model else ""
    
    fintuned_path = f'data/outputs/{model_name}_petqa_{suffix}/best_model'
    test_data_path = f"data/training/test.json"
    output_path = f"data/eval/output_{model_name}{ft}_{shot}_{suffix}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    shot_type = "zeroshot" if shot == "0" else "fewshot"
    system_prompt_path = f"prompt/system_{shot_type}.txt"
    user_prompt_path = f"prompt/user_{suffix}.txt"
    
    return {
        "fintuned_path": fintuned_path,
        "test_data_path": test_data_path,
        "output_path": output_path,
        "system_prompt_path": system_prompt_path,
        "user_prompt_path": user_prompt_path,
    }

def load_prompt(prompt_path):
    with open(prompt_path, encoding="utf-8") as file:
        return file.read()
  
def get_prompts(env):
    system_prompt = load_prompt(env["system_prompt_path"])
    user_prompt = load_prompt(env["user_prompt_path"])
    return system_prompt, user_prompt
  
def load_model_and_tokenizer(model_name, use_finetuned_model, fintuned_path):
    model_path = fintuned_path if use_finetuned_model else MODEL_MAPPING[model_name]
    
    # vLLM 모델 로드
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return llm, tokenizer

def load_test_data(test_data_path):
    with open(test_data_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_results(output_path):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"{start_idx}개까지 처리됨. 이어서 시작")
    else:
        results = []
        start_idx = 0
        print("새로 시작")
    return results, start_idx

def prepare_prompts(test_data, start_idx, tokenizer, use_raw_format, env):
    base_system_prompt, base_user_prompt = get_prompts(env)
    
    prompts = []
    data_to_process = test_data[start_idx:]
    
    for item in data_to_process:
        if use_raw_format:
            user_prompt = base_user_prompt.format(title=item["title"], content=item["content"])
        else:
            user_prompt = base_user_prompt.format(question=item["preprocessed_question"])
            
        messages = [
            {"role": "system", "content": base_system_prompt},
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
    env, use_raw_format, use_finetuned_model, logger
):
    prompts, data_to_process = prepare_prompts(
        test_data, 
        start_idx, 
        tokenizer, 
        use_raw_format, 
        env
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2048,
        repetition_penalty=1.4 if use_finetuned_model else 1.0,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None
    )
    
    # 중국어 토큰 생성 방지
    def _logits_processor(input_ids, logits):
        return blocker(tokenizer, input_ids, logits)
    sampling_params.logits_processors=[_logits_processor]
    
    for (prompt, item) in tqdm(zip(prompts, data_to_process), total=len(prompts), desc="답변 생성 중"):
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        generated_answer = outputs[0].outputs[0].text.strip()
        
        result = {
            "id": item["id"],
            "title": item["title"],
            "content": item["content"],
            "answer": item["answer"],
            "preprocessed_question": item["preprocessed_question"],
            "preprocessed_answer": item["preprocessed_answer"],
            "answer_date": item["answer_date"],
            "generated_answer": generated_answer
        }
        results.append(result)
        with open(env["output_path"], "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info("답변 생성 완료")
    
def main(model_name, shot, use_raw_format, use_finetuned_model):
    logger = setup_logging()
    logger.info(f"MODEL NAME: {model_name}")
    logger.info(f"SHOT: {shot}")
    logger.info(f"USE RAW FORMAT: {use_raw_format}")
    logger.info(f"USE FINETUNED MODEL: {use_finetuned_model}")
    
    env = load_environment(model_name, shot, use_raw_format, use_finetuned_model)    
    llm, tokenizer = load_model_and_tokenizer(model_name, use_finetuned_model, env["fintuned_path"])
    test_data = load_test_data(env["test_data_path"])
    results, start_idx = load_results(env["output_path"])
    
    generate_answers_sequential(
        llm, tokenizer, test_data, results, start_idx, 
        env, use_raw_format, use_finetuned_model, logger
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="답변 생성")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--shot", type=str, required=True, choices=["0", "1", "3", "6"])
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    main(args.model_name, args.shot, args.use_raw_format, args.use_finetuned_model)
