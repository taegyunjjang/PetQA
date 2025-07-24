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
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from google import genai
from google.genai import types
from google.genai import errors

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
google_client = genai.Client(api_key=GOOGLE_API_KEY)



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
        return lambda *args: call_openai_api(openai_client, model_name, *args)
    elif model_family == "claude":
        return lambda *args: call_anthropic_api(anthropic_client, model_name, *args)
    elif model_family == "gemini":
        return lambda *args: call_google_api(google_client, model_name, *args)

def call_openai_api(client, model_name, *args):
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

def call_anthropic_api(client, model_name, *args):
    system_prompt, user_prompt = args
    message = client.messages.create(
        model = MODEL_MAPPING[model_name],
        temperature = 0,
        max_tokens = 512,
        system = system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    return message.content[0].text

def call_google_api(client, model_name, *args):
    system_prompt, user_prompt = args
    
    max_retries = 10
    delay = 1
    for i in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_MAPPING[model_name],
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                    max_output_tokens=512,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                )
            )
            return response.text.strip()
        except errors.ServerError as e:
            if e.code == 503:
                print(f"Gemini API 503 에러 발생: {e.message}. {i+1}/{max_retries} 재시도. {delay}초 후 재시도...")
                time.sleep(delay)
                delay *= 2 
                if delay > 60: 
                    delay = 60
                else:
                    # 503이 아닌 다른 ServerError는 재시도하지 않고 바로 발생시킵니다.
                    print(f"Gemini API 서버 에러 발생: {e.message}")
                    raise 
        except Exception as e:
            # 예상치 못한 다른 모든 에러는 바로 발생시킵니다.
            print(f"예상치 못한 에러 발생: {e}")
            raise

def run_batch_openai_api(client, args, data_to_process, output_path, env, fewshot_map, logger):
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
    
    batch_input_path = env["batch_input_path"].replace(".jsonl", f"_{args.model_name}_{args.shot}_{args.input_format}.jsonl")
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
                with open(env["batch_error_path"], "w", encoding="utf-8") as f:
                    f.write(error_file.decode("utf-8"))
                logger.error(error_file)
            
            time.sleep(60)
        
        file_response = client.files.content(batch.output_file_id).content
        batch_output_path = env["batch_output_path"].replace(".jsonl", f"_{args.model_name}_{args.shot}_{args.input_format}.jsonl")
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

def run_batch_anthropic_api(client, args, data_to_process, output_path, env, fewshot_map, logger):
    start_time = time.time()
    
    batches = []
    for item in data_to_process:
        system_prompt, user_prompt = get_prompts(env, item, args.shot, fewshot_map, args.input_format)
            
        request = Request(
            custom_id=f"{item['q_id']}_{item['a_id']}",
            params=MessageCreateParamsNonStreaming(
                model=MODEL_MAPPING[args.model_name],
                temperature = 0,
                max_tokens=512,
                system = system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
        )
        batches.append(request)
    
    batch_input_path = env["batch_input_path"].replace(".jsonl", f"_{args.model_name}_{args.shot}_{args.input_format}.jsonl")
    with open(batch_input_path, "w", encoding="utf-8") as f:
        for batch in batches:
            f.write(json.dumps(batch, ensure_ascii=False) + "\n")
            
    message_batch = client.messages.batches.create(requests=batches)
    batch_id = message_batch.id
    logger.info(f"Batch job created: {batch_id}")
    
    try:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            
            if batch.processing_status == "canceling":
                logger.error("Request canceled.")
                break
            
            elif batch.processing_status == "ended":
                logger.info("Batch job completed.")
                break
            
            # 과도한 요청 방지
            time.sleep(60)
        
        contents = []
        for result in client.messages.batches.results(batch_id):
            match result.result.type:
                case "succeeded":
                    contents.append(result.result.message.content[0].text)
                case "errored":
                    if result.result.error.type == "invalid_request":
                        # 재전송하기 전에 요청 본문을 수정해야 함
                        logger.error(f"Validation error {result.custom_id}")
                    else:
                        # 요청을 직접 재시도할 수 있음
                        logger.error(f"Server error {result.custom_id}")
                    contents.append("")
                case "expired":
                    logger.error(f"Request expired {result.custom_id}")
                    contents.append("")
        
        for item, content in zip(data_to_process, contents):
            item["generated_answer"] = content
            
        save_json(data_to_process, output_path)

    except Exception as e:
        logger.error(f"Error: {e}")
        
    elapsed = time.time() - start_time
    logger.info(f"Time taken: {format_time(elapsed)}")


def generate_answers_batch(
    args,
    data_to_process,
    fewshot_map,
    output_path,
    env,
    logger
):
    if parse_model_name(args.model_name) == "gpt":
        run_batch_openai_api(openai_client, args, data_to_process, output_path, env, fewshot_map, logger)
    elif parse_model_name(args.model_name) == "claude":
        run_batch_anthropic_api(anthropic_client, args, data_to_process, output_path, env, fewshot_map, logger)
    
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