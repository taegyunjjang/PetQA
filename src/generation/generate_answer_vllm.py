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

def load_fewshot_examples(env, fewshot_type):
    fewshot_df = pd.read_json(env["fewshot_examples_path"].replace(".json", f"_{fewshot_type}.json"))
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
    
    system_zeroshot_prompt = None
    if shot == "0":
        system_zeroshot_prompt = load_prompt(env["system_zeroshot_prompt_path"])
    
    base_system_prompt = None
    base_user_prompt = None
    if shot != "0":
        base_system_prompt = load_prompt(env["system_fewshot_prompt_path"])
        
    if input_format == "raw":
        base_user_prompt = load_prompt(env["user_raw_input_prompt_path"])
    else:
        base_user_prompt = load_prompt(env["user_processed_input_prompt_path"])
    
    for item in data_to_process:
        if shot == "0":
            system_prompt = system_zeroshot_prompt
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


def generate_answers(
    llm, tokenizer, test_data, results, start_idx, 
    env, args, output_path, logger, batch_size
):
    fewshot_map = None
    if args.shot != "0":
        logger.info("Few-shot 프롬프트 생성 중")
        fewshot_map = load_fewshot_examples(env, args.fewshot_type)
        
    all_prompts, all_data_to_process = get_prompts(
        test_data, 
        start_idx, 
        tokenizer, 
        args.shot,
        fewshot_map,
        args.input_format, 
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
            repetition_penalty=1.2 if args.use_finetuned_model else 1.0,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
            seed=42
        )
        
        sampling_params.logits_processors = []
        if args.model_name == "qwen-2.5-7b":
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
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--fewshot_type", type=str, choices=["baseline", "bert", "llm", "oracle"], default="baseline")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_dpo_model", action="store_true")
    parser.add_argument("--use_summarization", action="store_true")
    parser.add_argument("--training_type", type=str, choices=["E", "NE", "ALL"], default="ALL")
    args = parser.parse_args()
    
    env = load_environment()    
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"SHOT: {args.shot}")
    if args.shot != "0":
        logger.info(f"FEWSHOT TYPE: {args.fewshot_type}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"USE SUMMARIZATION: {args.use_summarization}")
    
    if args.use_finetuned_model:
        logger.info(f"USE FINETUNED MODEL")
        logger.info(f"TRAINING TYPE: {args.training_type}")
        output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}.json")
        model_path = os.path.join(env["checkpoint_dir"],
                                   f"{args.model_name}_{args.input_format}_{args.training_type}",
                                   "best_model")
        if args.use_summarization:
            output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}_summarization.json")
            model_path = os.path.join(env["checkpoint_dir"],
                                   f"{args.model_name}_{args.input_format}_{args.training_type}_summarization",
                                   "best_model")
        llm, tokenizer = load_model_and_tokenizer(model_path)
    elif args.use_dpo_model:
        logger.info(f"USE DPO MODEL")
        logger.info(f"TRAINING TYPE: {args.training_type}")
        output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}_DPO.json")
        model_path = os.path.join(env["checkpoint_dir"],
                                   f"{args.model_name}_{args.input_format}_{args.training_type}_DPO",
                                   "merged_model")
        llm, tokenizer = load_model_and_tokenizer(model_path)
    else:
        output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.shot}_{args.input_format}.json")
        if args.use_summarization:
            output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.shot}_{args.input_format}_summarization.json")
        if args.shot != "0":
            output_path = output_path.replace(".json", f"_{args.fewshot_type}.json")
        llm, tokenizer = load_model_and_tokenizer(MODEL_MAPPING[args.model_name])
        
    results, start_idx = load_results(output_path)
    test_data = load_json(env["summarization_test_data_path"])
    
    batch_size = 500
    generate_answers(
        llm, tokenizer, test_data, results, start_idx, 
        env, args, output_path, logger, batch_size
    )