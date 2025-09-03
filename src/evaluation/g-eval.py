import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import (
    MODEL_MAPPING, setup_logging, save_json, load_json, format_time,
    load_prompt, load_results, load_environment
)

import json
import argparse
import time
import math
from dotenv import load_dotenv
from tqdm import tqdm
from copy import deepcopy
from pydantic import BaseModel
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
def get_prompts(env, item):
    base_system_prompt = load_prompt(env["g_eval_prompt_path"])
    system_prompt = base_system_prompt.format(
        question=item["preprocessed_question"],
        reference_answer=item["summarized_answer"],
        generated_answer=item["generated_answer"]
    )
    return system_prompt

def parse_response(response):
    try:
        content = response.choices[0].message.content.strip()
        score = int(content)
        if 1 <= score <= 5:
            return score
        else:
            return None
    except ValueError:
        return None

def call_openai_api(model_name, system_prompt):
    _response = client.chat.completions.create(
        model=MODEL_MAPPING[model_name],
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0,
    )
    score = parse_response(_response)
    return score

def run_openai_single_api(data_to_process, results, output_path, env, args, logger):
    logger.info("Generating single input...")
    for item in tqdm(data_to_process, total=len(data_to_process)):
        system_prompt = get_prompts(env, item)
        score = call_openai_api(args.judge_model_name, system_prompt)
        results.append({
            "q_id": item["q_id"],
            "a_id": item["a_id"],
            "question": item["preprocessed_question"],
            "reference_answer": item["summarized_answer"],
            "generated_answer": item["generated_answer"],
            "score": score
        })
        save_json(results, output_path)
    return results


def run_openai_batch_api(data_to_process, results, output_path, env, args, logger):
    start_time = time.time()
    logger.info("Generating batch input...")
    init_template = {
        "custom_id": None,
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_MAPPING[args.judge_model_name], 
            "messages": [],
            "temperature": 0
        }
    }
    
    batches = []
    for item in data_to_process:
        system_prompt = get_prompts(env, item)
        batch_request_template = deepcopy(init_template)
        batch_request_template["custom_id"] = f"{item['q_id']}_{item['a_id']}"
        batch_request_template["body"]["messages"].append({"role": "system", "content": system_prompt})
        batches.append(batch_request_template)
    
    endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_summarization"
    batch_input_path = env["batch_input_path"].replace(".jsonl", f"_{endpoint}.jsonl")
    with open(batch_input_path, "w") as f:
        for batch in batches:
            f.write(json.dumps(batch, ensure_ascii=False) + "\n")
            
    logger.info("Uploading batch input to OpenAI server...")
    batch_input_file = client.files.create(
        file=open(batch_input_path, "rb"),
        purpose="batch"
    )
    
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "generating G-Eval score"}
    )
    logger.info(f"Batch job created: {batch_job.id}")
    
    status_transition = False
    try:
        while True:
            batch = client.batches.retrieve(batch_job.id)
            
            if batch.status == "validating":
                logger.info("Batch is being validated...")
                
            if not status_transition and batch.status == "in_progress":
                logger.info("Batch is in progress...")
                status_transition = True
                
            if batch.status == "completed":
                logger.info("Batch is completed successfully!")
                break
            
            if batch.status == "failed":
                logger.error("Batch failed.")
                logger.error(batch.incomplete_details)
                break
            
            # 일부 실패 시 (status는 completed지만 error_file_id가 존재)
            if batch.error_file_id:
                logger.error("Batch failed partially.")
                error_file = client.files.content(batch.error_file_id)
                with open(env["batch_error_path"], "w", encoding="utf-8") as f:
                    f.write(error_file.decode("utf-8"))
                logger.error(error_file)
            
            time.sleep(60)
        
        file_response = client.files.content(batch.output_file_id).content
        batch_output_path = env["batch_output_path"].replace(".jsonl", f"_{endpoint}.jsonl")
        with open(batch_output_path, "wb") as f:
            f.write(file_response)
        
        scores_dict = {}
        with open(batch_output_path, "r") as f:
            for line in f:
                data = json.loads(line)
                custom_id = data["custom_id"]
                content = data["response"]["body"]["choices"][0]["message"]["content"].strip()
                scores_dict[custom_id] = parse_response(content)
        
        for item in data_to_process:
            cid = f"{item['q_id']}_{item['a_id']}"
            score = scores_dict[cid]
            results.append({
                "q_id": item["q_id"],
                "a_id": item["a_id"],
                "question": item["preprocessed_question"],
                "reference_answer": item["summarized_answer"],
                "generated_answer": item["generated_answer"],
                "score": score
            })
            
        save_json(results, output_path)
        return results
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.info(f"Time taken: {format_time(time.time() - start_time)}")
    

def calculate_score(endpoint, results):
    scores = [item["score"] for item in results if item["score"] is not None]
    avg_score = sum(scores) / len(scores)
    print("===" * 30)
    print(f"ID: {endpoint}")
    print(f"Average Score: {avg_score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV2R Score Evaluator")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gemma-3-4b")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--fewshot_type", type=str, choices=["baseline", "bert", "llm", "oracle"], default="baseline")
    parser.add_argument("--training_type", choices=["E", "NE", "ALL", "ORACLE"], default="ALL")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--use_summarization", action="store_true")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_dpo_model", action="store_true")
    parser.add_argument("--use_rag_model", action="store_true")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="batch")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4o")
    parser.add_argument("--sample_size", type=int, default=5)
    args = parser.parse_args()
    
    start_time = time.time()
    
    env = load_environment()
    logger = setup_logging()
    
    if args.use_finetuned_model:
        endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}.json"
        if args.use_summarization:
            endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}_summarization.json"
        input_path = os.path.join(env["generated_answers_dir"], endpoint)
    elif args.use_dpo_model:
        endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}_DPO.json"
        input_path = os.path.join(env["generated_answers_dir"], endpoint)
    elif args.use_rag_model:
        endpoint = f"output_{args.model_name}_{args.input_format}_RAG_{args.top_k}.json"
        input_path = os.path.join(env["generated_answers_dir"], endpoint)
    else:
        endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}.json"
        if args.use_summarization:
            endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_summarization.json"
        if args.shot != "0":
            endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.fewshot_type}.json"
        input_path = os.path.join(env["generated_answers_dir"], endpoint)
    
    output_path = os.path.join(env["g_eval_dir"], endpoint)
    results, start_idx = load_results(output_path)
    
    logger.info(f"INPUT PATH: {input_path}")
    logger.info(f"OUTPUT PATH: {output_path}")
    logger.info(f"MODE: {args.mode}")
    logger.info(f"SAMPLE SIZE: {args.sample_size}")
    logger.info(f"JUDGE MODEL NAME: {args.judge_model_name}")
    
    data = load_json(input_path)
    
    total_data = data[:args.sample_size]
    data_to_process = total_data[start_idx:]
    
    if args.mode == "single":
        results_with_score = run_openai_single_api(data_to_process, results, output_path, env, args, logger)
    elif args.mode == "batch":
        results_with_score = run_openai_batch_api(data_to_process, results, output_path, env, args, logger)
    
    calculate_score(endpoint, results_with_score)
    
    logger.info(f"TOTAL TIME: {format_time(time.time() - start_time)}")
    logger.info(f"G-Eval Evaluation Completed")