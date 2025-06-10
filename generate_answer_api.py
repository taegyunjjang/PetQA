import time
import os
import re
import pandas as pd
from tqdm import tqdm
import json
import argparse
from copy import deepcopy
import logging

from dotenv import load_dotenv
import openai
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

MAX_WAIT_TIME = 24 * 60 * 60
MODEL_MAPPING = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
}


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}시간 {minutes}분 {seconds}초"
    elif minutes > 0:
        return f"{minutes}분 {seconds}초"
    else:
        return f"{seconds}초"

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

def load_environment(model_name, shot, use_raw_format):
    suffix = "raw" if use_raw_format else "preprocessed"
    output_path = f"data/eval/output_{model_name}_{shot}_{suffix}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    shot_type = "zeroshot" if shot == "0" else "fewshot"
    system_prompt_path = f"prompt/system_{shot_type}.txt"
    user_prompt_path = f"prompt/user_{suffix}.txt"
    
    fewshot_examples_path = "data/training/fewshot_examples.json"
    
    batch_input_file = f"data/batch_input_{model_name}_{shot}_{suffix}.jsonl"
    batch_error_file = f"data/batch_error_file_{model_name}_{shot}_{suffix}.jsonl"
    batch_output_file = f"data/batch_output_{model_name}_{shot}_{suffix}.jsonl"
    
    return {
        "output_path": output_path,
        "system_prompt_path": system_prompt_path,
        "user_prompt_path": user_prompt_path,
        "fewshot_examples_path": fewshot_examples_path,
        "batch_input_file": batch_input_file,
        "batch_error_file": batch_error_file,
        "batch_output_file": batch_output_file
    }

def load_text(file_path):
    with open(file_path, encoding="utf-8") as file:
        return file.read()

def load_base_prompts(env):
    base_system_prompt = load_text(env["system_prompt_path"])
    base_user_prompt = load_text(env["user_prompt_path"])
    return base_system_prompt, base_user_prompt

def parse_model_name(model_name):
    if model_name.startswith("gpt"):
        return "gpt"
    elif model_name.startswith("claude"):
        return "claude"
    elif model_name.startswith("gemini"):
        return "gemini"
  
def load_fewshot_examples(env):
    fewshot_df = pd.read_json(env["fewshot_examples_path"])
    fewshot_map = {row["id"]: row for _, row in fewshot_df.iterrows()}
    return fewshot_map

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

def build_fewshot_examples(sample, shot, use_raw_format):
    examples = ""
    for i in range(int(shot)):
        title = sample["similar_questions"][i]["title"]
        content = sample["similar_questions"][i]["content"]
        question = sample["similar_questions"][i]["preprocessed_question"]
        answer = sample["similar_questions"][i]["preprocessed_answer"]
        
        if use_raw_format:
            examples += f"제목: {title}\n본문: {content}\n답변: {answer}\n\n"
        else:
            examples += f"질문: {question}\n답변: {answer}\n\n"
    return examples.strip()

def get_prompts(base_system_prompt, base_user_prompt,
                env, row, shot, use_raw_format):
    # system prompt
    if shot == "0":
        system_prompt = base_system_prompt
    else:
        fewshot_map = load_fewshot_examples(env)
        id = row["id"]
        sample = fewshot_map.get(id)
        fewshot_examples = build_fewshot_examples(sample, shot, use_raw_format)
        system_prompt = base_system_prompt.format(fewshot_examples=fewshot_examples)
    
    # user prompt
    if use_raw_format:
        user_prompt = base_user_prompt.format(title=row["title"], content=row["content"])
    else:
        user_prompt = base_user_prompt.format(question=row["preprocessed_question"])
        
    return system_prompt, user_prompt

def get_model_processor(model_name):
    model_family = parse_model_name(model_name)
    if model_family == "gpt":
        return lambda *args: generate_gpt_answer(openai_client, model_name, *args)
    elif model_family == "claude":
        return lambda *args: generate_claude_answer(anthropic_client, model_name, *args)
    elif model_family == "gemini":
        return lambda *args: generate_gemini_answer(model_name, *args)

def generate_gpt_answer(client, model_name, *args):
    system_prompt, user_prompt = args
    response = client.chat.completions.create(
        model=MODEL_MAPPING[model_name],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        max_tokens=2048,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def generate_claude_answer(client, model_name, *args):
    system_prompt, user_prompt = args
    message = client.messages.create(
        model = MODEL_MAPPING[model_name],
        temperature = 0,
        max_tokens = 2048,
        system = system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    return message.content[0].text

def generate_gemini_answer(model_name, *args):
    system_prompt, user_prompt = args
    model = genai.GenerativeModel(
        model_name=MODEL_MAPPING[model_name], 
        system_instruction=system_prompt
    )
    
    generation_config = {
        "temperature": 0, 
        "max_output_tokens": 2048,
    }
    
    response = model.generate_content(
        contents=[
            {"role": "user", "parts": [user_prompt]}
        ],
        generation_config=generation_config
    )
    return response.text

def run_batch_gpt_pipeline(df, model_name, client, 
                           base_system_prompt, base_user_prompt,
                           env, shot, use_raw_format, logger):
    logger.info("배치 작업을 위한 입력 파일 생성 중...")
    start_time = time.time()
    
    init_template = {
        "custom_id": None,
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_MAPPING[model_name], 
            "messages":[],
            "max_tokens": 2048,
            "temperature": 0
        }
    }
    
    batches = []
    for _, row in df.iterrows():
        system_prompt, user_prompt = get_prompts(base_system_prompt, base_user_prompt,
                                                 env, row, shot, use_raw_format)
        
        temp = deepcopy(init_template)
        temp["custom_id"] = str(row["id"])
        temp["body"]["messages"].append({"role": "system", "content": system_prompt})
        temp["body"]["messages"].append({"role": "user", "content": user_prompt})
        batches.append(temp)
        
    with open(env["batch_input_file"], "w", encoding="utf-8") as file:
        for item in batches:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info("OpenAI 서버에 입력 파일 업로드 중...")
    batch_input_file = client.files.create(
        file=open(batch_input_file, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "generating answer"
        }
    )
    logger.info(f"ID: {batch_job.id}")
    
    status_transition = False
    try:
        while True:
            batch = client.batches.retrieve(batch_job.id)
            
            if batch.status == "validating":
                logger.info("유효성 검사 중...")
                
            if not status_transition and batch.status == "in_progress":
                logger.info("배치 작업 진행 중...")
                status_transition = True
                
            if batch.status == "completed":
                logger.info("배치 작업이 성공적으로 완료되었습니다.")
                break
            
            if time.time() - start_time > MAX_WAIT_TIME:
                logger.error(f"최대 대기 시간을 초과했습니다.")
                break
            
            if batch.status == "failed":
                logger.error("배치 작업이 실패했습니다.")
                logger.error(batch.incomplete_details)
                break
            
            # 일부 실패 시 (status는 completed지만 error_file_id가 존재)
            if batch.error_file_id:
                logger.error("일부 배치 작업이 실패했습니다.")
                error_file = client.files.content(batch.error_file_id)
                with open(env["batch_error_file"], "wb") as f:
                    f.write(error_file)
                logger.error(error_file)
            
            # 과도한 요청 방지
            time.sleep(60)
        
        output_file_id = batch.output_file_id
        file_response = client.files.content(output_file_id).content
        
        with open(env["batch_output_file"], "wb") as f:
            f.write(file_response)
        
        contents = []
        with open(env["batch_output_file"], "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                cleaned_content = re.sub(r"^\s*(#{1,6})?\s*답변\s*[:：]?\s*\n*", "", content.strip(), flags=re.IGNORECASE)
                contents.append(cleaned_content)
                
        df['generated_answer'] = contents
        df.to_json(env["output_path"], orient='records', force_ascii=False, indent=4) 
        logger.info(f"{env['output_path']}")
            
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        
    elapsed = time.time() - start_time
    logger.info(f"배치 작업 총 소요 시간: {format_time(elapsed)}")
        
def run_batch_claude_pipeline(df, model_name, client, 
                              base_system_prompt, base_user_prompt,
                              env, shot, use_raw_format, logger):
    logger.info("배치 작업을 위한 요청 준비 중...")
    start_time = time.time()
    
    batches = []
    for _, row in df.iterrows():
        system_prompt, user_prompt = get_prompts(base_system_prompt, base_user_prompt,
                                                 env, row, shot, use_raw_format)
            
        request = Request(
            custom_id=row["id"],
            params=MessageCreateParamsNonStreaming(
                model=MODEL_MAPPING[model_name],
                temperature = 0,
                max_tokens=2048,
                system = system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
        )
        batches.append(request)
    
    with open(env["batch_input_file"], "w", encoding="utf-8") as file:
        for item in batches:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + "\n")
            
    message_batch = client.messages.batches.create(requests=batches)
    batch_id = message_batch.id
    logger.info(f"ID: {batch_id}")
    logger.info("배치 작업 진행 중...")
    
    try:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            
            if batch.processing_status == "canceling":
                logger.error("요청이 취소되었습니다.")
                break
            
            elif batch.processing_status == "ended":
                logger.info("배치 작업이 완료되었습니다.")
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
        
        df["generated_answer"] = contents
        df.to_json(env["output_path"], orient="records", force_ascii=False, indent=4)
        logger.info(f"{env['output_path']}")
            
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        
    elapsed = time.time() - start_time
    logger.info(f"배치 작업 총 소요 시간: {format_time(elapsed)}")

def main(df, model_name, shot, use_raw_format, use_batch_api):
    logger = setup_logging()
    logger.info(f"MODEL NAME: {model_name}")
    logger.info(f"SHOT: {shot}")
    logger.info(f"USE RAW FORMAT: {use_raw_format}")
    logger.info(f"USE BATCH API: {use_batch_api}")
    
    env = load_environment(model_name, shot, use_raw_format)
    base_system_prompt, base_user_prompt = load_base_prompts(env)
    model_family = parse_model_name(model_name)
    
    if use_batch_api:
        if model_family == "gpt":
            run_batch_gpt_pipeline(df, model_name, openai_client, 
                                   base_system_prompt, base_user_prompt, 
                                   env, shot, use_raw_format, logger)
        elif model_family == "claude":
            run_batch_claude_pipeline(df, model_name, anthropic_client, 
                                      base_system_prompt, base_user_prompt, 
                                      env, shot, use_raw_format, logger)
        elif model_family == "gemini":
            pass
    else:
        results, start_idx = load_results(env["output_path"])
        model_processor = get_model_processor(model_name)
        
        for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx, desc="답변 생성 중"):
            system_prompt, user_prompt = get_prompts(base_system_prompt, base_user_prompt,
                                                     env, row, shot, use_raw_format)
            
            generated_answer = model_processor(system_prompt, user_prompt)
            df.loc[i, 'generated_answer'] = generated_answer
            
            results.append(df.loc[i].to_dict())
            with open(env["output_path"], "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        
    logger.info("답변 생성 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="답변 생성")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--shot", type=str, required=True, choices=["0", "1", "3", "6"])
    parser.add_argument("--use_batch_api", action="store_true")
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    df = pd.read_json("data/training/test.json")
    
    main(df, args.model_name, args.shot, args.use_raw_format, args.use_batch_api)
    