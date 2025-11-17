import time
import re
import json
from copy import deepcopy
import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import (
    format_time, setup_logging, save_json, load_json, MODEL_MAPPING,
    load_prompt, load_environment
)
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)



def parse_cleaned_qna(cleaned_qna_str, id, logger):
    try:
        cleaned_qna_dict = json.loads(cleaned_qna_str)
        return {
            "preprocessed_question": cleaned_qna_dict.get('preprocessed_question', ''),
            "preprocessed_answer": cleaned_qna_dict.get('preprocessed_answer', '')
        }
    except json.JSONDecodeError as e:
        if logger:
            logger.warning(f"[JSONDecodeError] id: {id} - 응답 파싱 실패\n내용: {cleaned_qna_str}\n에러: {e}")
        return None
    except Exception as e:
        if logger:
            logger.warning(f"[UnknownError] id: {id} - 예기치 않은 오류 발생\n내용: {cleaned_qna_str}\n에러: {e}")
        return None


def run_openai_batch_api(client, data_to_process, base_prompt, env, logger):
    start_time = time.time()
    logger.info("Generating batch input...")
    MODEL_NAME = "gpt-4o-mini"
    init_template = {
        "custom_id": None,
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_MAPPING[MODEL_NAME], 
            "messages": [],
            "temperature": 0,
            "seed": 42
        }
    }
    
    batches = []
    for item in data_to_process:
        user_prompt = base_prompt.format(question=item["title"] + "\n" + item["content"], answer=item["answer"])
        batch_request_template = deepcopy(init_template)
        batch_request_template["custom_id"] = f"{item['id']}"
        batch_request_template["body"]["messages"].append({"role": "user", "content": user_prompt})
        batches.append(batch_request_template)
    
    batch_input_path = env["batch_input_path"].replace(".jsonl", f"batch_input_filtering.jsonl")
    os.makedirs(os.path.dirname(batch_input_path), exist_ok=True)
    with open(batch_input_path, "w", encoding="utf-8") as f:
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
        metadata={"description": "filtering data"}
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
        batch_output_path = env["batch_output_path"].replace(".jsonl", f"batch_output_filtering.jsonl")
        with open(batch_output_path, "wb") as f:
            f.write(file_response)
        
        contents = []
        with open(batch_output_path, "r") as f:
            for line in f:
                data = json.loads(line)
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                contents.append(content)
        return contents
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.info(f"Time taken: {format_time(time.time() - start_time)}")
        
        
def filtering(data_to_process, base_prompt, env, logger):
    logger.info(f"필터링할 데이터: {len(data_to_process)}개")
    contents = run_openai_batch_api(client, data_to_process, base_prompt, env, logger)
    for item, content in zip(data_to_process, contents):
        item["is_relevant"] = content.lower()
    save_json(data_to_process, env["filtering_all_results_path"])
    irrelevant_data = [item for item in data_to_process if item['is_relevant'] == "false"]
    save_json(irrelevant_data, env["irrelevant_data_path"])
    relevant_data = [item for item in data_to_process if item['is_relevant'] == "true"]
    save_json(relevant_data, env["filtered_data_path"])
    logger.info(f"필터링 완료")


def cleaning(data_to_process, base_prompt, env, logger):
    logger.info(f"클리닝할 데이터: {len(data_to_process)}개")
    contents = run_openai_batch_api(client, data_to_process, base_prompt, env, logger)
    parsed_contents = [parse_cleaned_qna(content, item["id"], logger) for item, content in zip(data_to_process, contents)]
    for item, parsed_content in zip(data_to_process, parsed_contents):
        item["preprocessed_question"] = parsed_content["preprocessed_question"]
        item["preprocessed_answer"] = parsed_content["preprocessed_answer"]
        item.pop("is_relevant", None)
    save_json(data_to_process, env["cleaned_data_path"])
    logger.info(f"클리닝 완료")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["filtering", "cleaning"], required=True)
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    logger.info(f"MODE: {args.mode.upper()}")
    
    if args.mode == "filtering":
        base_prompt = load_prompt(env["filtering_prompt_path"])
        data_to_process = load_json(env["preprocessed_data_path"])
        filtering(data_to_process, base_prompt, env, logger)
        
    elif args.mode == "cleaning":
        base_prompt = load_prompt(env["cleaning_prompt_path"])
        data_to_process = load_json(env["filtered_data_path"])
        cleaning(data_to_process, base_prompt, env, logger)
