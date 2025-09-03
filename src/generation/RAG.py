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
        gpu_memory_utilization=0.9,
    )
    
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return llm, tokenizer

def load_retrieved_samples(env):
    retrieved_samples = load_json(env["retrieved_paragraphs_path"])
    retrieved_samples_map = {item["q_id"]: item for item in retrieved_samples}
    return retrieved_samples_map

def build_rag_examples(sample, top_k):
    paragraphs = []
    for i in range(top_k):
        paragraphs.append(sample["retrieved_paragraphs"][i])
    
    return paragraphs

def get_prompts(test_data, start_idx, tokenizer, retrieved_samples, args, env):
    prompts = []
    data_to_process = test_data[start_idx:]
    
    system_prompt = load_prompt(env["system_rag_prompt_path"])        
    base_user_prompt = load_prompt(env["user_rag_prompt_path"])
    
    for item in data_to_process:
        sample = retrieved_samples.get(item["q_id"])
        paragraphs = build_rag_examples(sample, args.top_k)
        user_prompt = base_user_prompt.format(question=item["preprocessed_question"], paragraphs=paragraphs)
            
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

def generate_answers(
    llm, tokenizer, test_data, results, start_idx, 
    env, args, output_path, logger, batch_size
):
    retrieved_samples = load_retrieved_samples(env)
    all_prompts, all_data_to_process = get_prompts(
        test_data, 
        start_idx, 
        tokenizer,
        retrieved_samples,
        args,
        env
    )
    
    total_prompts_to_process = len(all_prompts)
    current_processed_count = len(results) - start_idx if len(results) > start_idx else 0
    num_batches = (total_prompts_to_process - current_processed_count + batch_size - 1) // batch_size
    
    for i in tqdm(range(current_processed_count, total_prompts_to_process, batch_size), total=num_batches, desc="Batch Generation"):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_data_to_process = all_data_to_process[i : i + batch_size]

        if not batch_prompts:
            continue # 빈 배치는 건너뛰기

        generated_answers = []
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=512,
            # repetition_penalty=1.2,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None
        )
        
        # 중국어 토큰 생성 방지
        def _logits_processor(input_ids, logits):
            return blocker(tokenizer, input_ids, logits)
        sampling_params.logits_processors=[_logits_processor]

        for i in range(0, len(batch_prompts), batch_size):
            prompts = batch_prompts[i:i + batch_size]
            
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False) # 내부 tqdm은 사용하지 않음
            
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                generated_answers.append(generated_text)
        
        for j, item_data in enumerate(batch_data_to_process):
            item_data["generated_answer"] = generated_answers[j]
            results.append(item_data)
        
        save_json(results, output_path)

    logger.info("배치 답변 생성 및 저장 완료")


 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="답변 생성")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gemma-3-4b")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_summarization", action="store_true")
    parser.add_argument("--training_type", type=str, choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()
    
    env = load_environment()    
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"TOP_K: {args.top_k}")
    
    if args.use_finetuned_model:
        logger.info(f"USE FINETUNED MODEL")
        output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.input_format}_{args.training_type}_RAG_{args.top_k}.json")
        model_path = os.path.join(env["checkpoint_dir"],
                                   f"{args.model_name}_{args.input_format}_{args.training_type}",
                                   "best_model")
        llm, tokenizer = load_model_and_tokenizer(model_path)
    else:
        output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.input_format}_RAG_{args.top_k}.json")
        if args.use_summarization:
            output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.input_format}_RAG_{args.top_k}_summarization.json")
        llm, tokenizer = load_model_and_tokenizer(MODEL_MAPPING[args.model_name])
        
    results, start_idx = load_results(output_path)
    test_data = load_json(env["summarization_test_data_path"])
    
    batch_size = 500
    generate_answers(
        llm, tokenizer, test_data, results, start_idx, 
        env, args, output_path, logger, batch_size
    )