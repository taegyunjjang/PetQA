"""vLLM을 활용하여 답변 생성"""

import json
import os
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import re

from etc.blocker_numpy import blocker


MODEL_MAPPING = {
    "gemma-3-4b": "google/gemma-3-4b-it",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
}

def load_environment(model_name, use_fintuned_model, shot, use_raw_format):
    suffix = "_raw" if use_raw_format else ""
    ft = "_petqa" if use_fintuned_model else ""
    
    fintuned_path = f'data/outputs/{model_name}_petqa{suffix}/best_model'
    test_data_path = f"data/training/test.json"
    output_path = f"data/eval/output_{model_name}{ft}_{shot}{suffix}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    return {
        "fintuned_path": fintuned_path,
        "test_data_path": test_data_path,
        "output_path": output_path
    }

def load_prompt(file_path):
    with open(file_path, encoding="utf-8") as file:
        return file.read()
        
def parse_model_name(model_name):
    if model_name.startswith("gemma"):
        return "gemma"
    elif model_name.startswith("qwen"):
        return "qwen"
    elif model_name.startswith("exaone"):
        return "exaone"
  
def get_prompts(shot, use_raw_format):
    suffix = "_raw" if use_raw_format else ""
    shot_type = "zeroshot" if shot == "0" else "fewshot"
        
    system_prompt_path = f"prompt/generate_answer/system_{shot_type}.txt"
    user_prompt_path = f"prompt/generate_answer/user{suffix}.txt"
    
    system_prompt = load_prompt(system_prompt_path)
    user_prompt = load_prompt(user_prompt_path)
    return system_prompt, user_prompt
  
def load_model_and_tokenizer(model_name, use_finetuned_model, fintuned_path):
    if use_finetuned_model:
        model_path = fintuned_path
        print(f"Fine-tuned 모델 사용: {model_path}")
    else:
        model_path = MODEL_MAPPING[model_name]
        print(f"Base 모델 사용: {model_path}")
    
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

def prepare_prompts(test_data, start_idx, tokenizer, use_raw_format, shot):
    base_system_prompt, base_user_prompt = get_prompts(shot, use_raw_format)
    
    prompts = []
    data_to_process = test_data[start_idx:]
    
    for item in data_to_process:
        if use_raw_format:
            user_prompt = base_user_prompt.replace("{title}", item["title"]).replace("{content}", item["content"])
        else:
            user_prompt = base_user_prompt.replace("{question}", item["preprocessed_question"])
            
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

def extract_json_from_output(output):
    try:
        # ```json ... ``` block
        json_block_match = re.search(r"```json\s*({.*?})\s*```", output, re.DOTALL)
        if not json_block_match:
            # plain JSON
            json_block_match = re.search(r"({\s*\"답변\"\s*:\s*\".*?\".*?})", output, re.DOTALL)

        if json_block_match:
            answer_json_str = json_block_match.group(1)
        else:
            print(f"[Warning] JSON not found in output:\n{output}")
            answer_json_str = ""
            
    except Exception as e:
        print(f"[Error] Exception during regex JSON extraction: {e}")
        answer_json_str = ""
    
    return answer_json_str

def process_model_output(raw_output):
    # 괄호 개수 일치 보정
    open_braces = raw_output.count("{")
    close_braces = raw_output.count("}")
    if open_braces > close_braces:
        raw_output += "}" * (open_braces - close_braces)
    elif close_braces > open_braces:
        raw_output = "{" * (close_braces - open_braces) + raw_output

    # 비표준 escape 제거
    cleaned_output = re.sub(r'\\([^"\\/bfnrtu])', r'\1', raw_output)
    
    # JSON 형식으로 변환
    try:
        parsed = json.loads(cleaned_output)
    except json.JSONDecodeError:
        print(f"JSONDecodeError: {cleaned_output}")
        parsed = {"답변": ""}
    return parsed

def generate_answers_sequential(
    llm, tokenizer, test_data, results, start_idx, 
    output_path, use_finetuned_model, use_raw_format, model_name, shot
):
    prompts, data_to_process = prepare_prompts(
        test_data, 
        start_idx, 
        tokenizer, 
        use_raw_format, 
        model_name, shot
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        repetition_penalty=1.2,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None
    )
    
    if parse_model_name(model_name) == "qwen":
        def logits_processor_wrapper(input_ids, logits):
            return blocker(tokenizer, input_ids, logits)
        sampling_params.logits_processors=[logits_processor_wrapper]
    
    for i, (prompt, item) in enumerate(tqdm(zip(prompts, data_to_process), total=len(prompts), desc="답변 생성 중")):
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        raw_output = outputs[0].outputs[0].text.strip()
        
        if not use_finetuned_model:
            extracted_output = extract_json_from_output(raw_output)
            processed_output = process_model_output(extracted_output)
            generated_answer = processed_output.get("답변", "")
        
        result = {
            "id": item["id"],
            "title": item["title"],
            "content": item["content"],
            "answer": item["answer"],
            "preprocessed_question": item["preprocessed_question"],
            "preprocessed_answer": item["preprocessed_answer"],
            "answer_date": item["answer_date"],
            "generated_answer": generated_answer if not use_finetuned_model else raw_output
        }
        
        results.append(result)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"결과가 {output_path}에 저장되었습니다.")
    print("답변 생성 완료")
    
def main(model_name, use_finetuned_model, shot, use_raw_format):
    env = load_environment(model_name, use_finetuned_model, shot, use_raw_format)    
    
    llm, tokenizer = load_model_and_tokenizer(model_name, use_finetuned_model, env["fintuned_path"])
    
    test_data = load_test_data(env["test_data_path"])
    
    results, start_idx = load_results(env["output_path"])
    
    generate_answers_sequential(
        llm, tokenizer, test_data, results, start_idx, 
        env["output_path"], use_finetuned_model, use_raw_format, model_name, shot
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="답변 생성")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--shot", type=str, required=True, choices=["0", "1", "3", "6"])
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    main(args.model_name, args.use_finetuned_model, args.shot, args.use_raw_format)
