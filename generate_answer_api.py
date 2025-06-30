import pandas as pd
from tqdm import tqdm
import argparse
import json
import time
from copy import deepcopy
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import (
    MODEL_MAPPING, format_time, setup_logging, save_json, 
    load_json, load_prompt, load_results, load_environment
)

from dotenv import load_dotenv
import openai


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def parse_model_name(model_name):
    if model_name.startswith("gpt"):
        return "gpt"
    elif model_name.startswith("claude"):
        return "claude"
    elif model_name.startswith("gemini"):
        return "gemini"
  
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

def get_prompts(env, item, shot, fewshot_map, input_format):
    # system prompt
    if shot == "0":
        system_prompt = load_prompt(env["system_zeroshot_prompt_path"])
    else:
        sample = fewshot_map.get(item["q_id"])
        fewshot_examples = build_fewshot_examples(sample, shot, input_format)
        base_system_prompt = load_prompt(env["system_fewshot_prompt_path"])
        system_prompt = base_system_prompt.format(fewshot_examples=fewshot_examples)
    
    # user prompt
    if input_format == "raw":
        base_user_prompt = load_prompt(env["user_raw_input_prompt_path"])
        user_prompt = base_user_prompt.format(title=item["title"], content=item["content"])
    else:
        base_user_prompt = load_prompt(env["user_processed_input_prompt_path"])
        user_prompt = base_user_prompt.format(question=item["preprocessed_question"])
        
    return system_prompt, user_prompt

def get_model_processor(model_name):
    model_family = parse_model_name(model_name)
    if model_family == "gpt":
        return lambda *args: call_gpt_api(client, model_name, *args)

def call_gpt_api(client, model_name, *args):
    system_prompt, user_prompt = args
    response = client.chat.completions.create(
        model=MODEL_MAPPING[model_name],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        max_tokens=512,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def generate_answers_batch(
    args,
    data_to_process,
    fewshot_map,
    output_path,
    env,
    logger
):
    start_time = time.time()
    logger.info("Generating batch input...")
    init_template = {
        "custom_id": None,
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_MAPPING[args.model_name], 
            "messages": [],
            "max_tokens": 512,
            "temperature": 0
        }
    }
    
    batches = []
    for item in data_to_process:
        system_prompt, user_prompt = get_prompts(env, item, args.shot, fewshot_map, args.input_format)
        batch_request_template = deepcopy(init_template)
        batch_request_template["custom_id"] = f"{item['q_id']}_{item['a_id']}"
        batch_request_template["body"]["messages"].append({"role": "system", "content": system_prompt})
        batch_request_template["body"]["messages"].append({"role": "user", "content": user_prompt})
        batches.append(batch_request_template)
    
    batch_input_path = env["batch_input_path"].replace(".jsonl", f"_{args.shot}_{args.input_format}.jsonl")
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
        metadata={"description": "generating answer"}
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
                with open(env["batch_error_path"], "wb") as f:
                    f.write(error_file)
                logger.error(error_file)
            
            time.sleep(60)
        
        file_response = client.files.content(batch.output_file_id).content
        batch_output_path = env["batch_output_path"].replace(".jsonl", f"_{args.shot}_{args.input_format}.jsonl")
        with open(batch_output_path, "wb") as f:
            f.write(file_response)
        
        contents = []
        with open(batch_output_path, "r") as f:
            for line in f:
                data = json.loads(line)
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                contents.append(content)
                
        for item, content in zip(data_to_process, contents):
            item["generated_answer"] = content
            
        save_json(data_to_process, output_path)
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.info(f"Time taken: {format_time(time.time() - start_time)}")

def generate_answers_sequential(
    args,
    data_to_process,
    results,
    fewshot_map,
    output_path,
    env
):
    model_processor = get_model_processor(args.model_name)
    
    for item in tqdm(data_to_process, total=len(data_to_process), desc="답변 생성 중"):
        system_prompt, user_prompt = get_prompts(env, item, args.shot, fewshot_map, args.input_format)
        generated_answer = model_processor(system_prompt, user_prompt)
        item['generated_answer'] = generated_answer
        results.append(item)
        save_json(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="답변 생성")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gpt-4o-mini")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--use_batch_api", action="store_true")
    args = parser.parse_args()
    
    env = load_environment()
    output_path = os.path.join(env["generated_answers_dir"],
                               f"output_{args.model_name}_{args.shot}_{args.input_format}.json")
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"SHOT: {args.shot}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"USE BATCH API: {args.use_batch_api}")
    logger.info(f"OUTPUT PATH: {output_path}")
    
    fewshot_map = None
    if args.shot != "0":
        fewshot_map = load_fewshot_examples(env)
    
    test_data = load_json(env["test_data_path"])
    results, start_idx = load_results(output_path)
    data_to_process = test_data[start_idx:]
    
    if args.use_batch_api:
        generate_answers_batch(
            args,
            data_to_process,
            fewshot_map,
            output_path,
            env,
            logger
        )
    else:
        generate_answers_sequential(
            args,
            data_to_process,
            results,
            fewshot_map,
            output_path,
            env
        )
    
    logger.info("Generate answers completed.")