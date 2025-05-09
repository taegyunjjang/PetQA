import time
import os
import re
import pandas as pd
from tqdm import tqdm
import json
import argparse
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from dotenv import load_dotenv
import openai
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import google.generativeai as genai


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

cache_dir = "./models"
MAX_WAIT_TIME = 24 * 60 * 60
MODEL_NAME_TO_API_ID = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
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

def load_prompt(file_path):
    with open(file_path, encoding="utf-8") as file:
        return file.read()

def parse_model_name(model_name):
    if model_name.startswith("gpt"):
        return "gpt"
    elif model_name.startswith("claude"):
        return "claude"
    elif model_name.startswith("gemini"):
        return "gemini"
    elif model_name.startswith("exaone"):
        return "exaone"
  
def get_prompts(model_name, shot, use_raw_format):
    model_family = parse_model_name(model_name)
    if shot == "0":
        shot = "zeroshot"
    else:
        shot = "fewshot"
        
    system_prompt_path = f"prompt/generate_answer/system_{shot}.txt"
    user_prompt_path = f"prompt/generate_answer/user.txt"
    
    if model_family == "claude":
        user_prompt_path = f"prompt/generate_answer/user_{model_family}.txt"
    if use_raw_format:
        user_prompt_path = user_prompt_path.replace(".txt", "_raw.txt")
    
    system_prompt = load_prompt(system_prompt_path)
    user_prompt = load_prompt(user_prompt_path)
    return system_prompt, user_prompt

def load_hf_model_and_tokenizer(model_name, cache_dir):
    model_id = MODEL_NAME_TO_API_ID[model_name]
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    return model, tokenizer

def build_fewshot_examples(sample, shot, use_raw_format):
    examples = ""
    for i in range(int(shot)):
        if use_raw_format:
            title = sample['similar_questions'][i]['title']
            content = sample['similar_questions'][i]['content']
            answer = sample['similar_questions'][i]['answer']
            examples += f"제목: {title}\n본문: {content}\n답변: {answer}\n\n"
        else:
            question = sample['similar_questions'][i]['question']
            answer = sample['similar_questions'][i]['answer']
            examples += f"질문: {question}\n답변: {answer}\n\n"
    return examples.strip()

def get_model_processor(model_name):
    model_family = parse_model_name(model_name)
    if model_family == "gpt":
        return lambda *args: generate_gpt_answer(openai_client, model_name, *args)
    elif model_family == "claude":
        return lambda *args: generate_claude_answer(anthropic_client, model_name, *args)
    elif model_family == "gemini":
        return lambda *args: generate_gemini_answer(model_name, *args)
    elif model_family == "exaone":
        model, tokenizer = load_hf_model_and_tokenizer(model_name, cache_dir)
        return lambda *args: generate_exaone_answer(model, tokenizer, *args)

def process_model_output(raw_output):
    # 괄호 개수 일치 보정
    open_braces = raw_output.count("{")
    close_braces = raw_output.count("}")
    if open_braces > close_braces:
        raw_output += "}" * (open_braces - close_braces)
    elif close_braces > open_braces:
        raw_output = "{" * (close_braces - open_braces) + raw_output

    # JSON 형식으로 변환
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        print(f"JSONDecodeError: {raw_output}")
        parsed = {"답변": ""}
    return parsed
    
def generate_gpt_answer(client, model_name, *args):
    system_prompt, user_prompt = args
    response = client.chat.completions.create(
        model=MODEL_NAME_TO_API_ID[model_name],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        max_tokens=2048,
        temperature=0,
        seed=42
    )
    return response.choices[0].message.content.strip()

def generate_claude_answer(client, model_name, *args):
    system_prompt, user_prompt = args
    message = client.messages.create(
        model = MODEL_NAME_TO_API_ID[model_name],
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
        model_name=MODEL_NAME_TO_API_ID[model_name], 
        system_instruction=system_prompt
    )
    
    generation_config = {
        "temperature": 0, 
        "max_output_tokens": 2048,
        "response_mime_type": "application/json"
    }
    
    response = model.generate_content(
        contents=[
            {"role": "user", "parts": [user_prompt]}
        ],
        generation_config=generation_config
    )
    return response.text
    
def generate_exaone_answer(model, tokenizer, *args):
    system_prompt, user_prompt = args
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    output = model.generate(
        input_ids.to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=2048,
        do_sample=False
    )
    
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=False)

    start_token = "[|assistant|]"
    end_token = "[|endofturn|]"
    
    # 답변 추출
    if start_token in decoded_output:
        answer = decoded_output.split(start_token, 1)[1].strip()
        if end_token in answer:
            return answer.split(end_token, 1)[0].strip()
        else:
            return answer.strip()
    else:
        return decoded_output.strip()

def run_batch_gpt_pipeline(df, model_name, client, base_system_prompt, base_user_prompt, shot, fewshot_map, use_raw_format):
    print("배치 작업을 위한 입력 파일 생성 중...")
    start_time = time.time()
    
    init_template = {
        "custom_id": None,
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_NAME_TO_API_ID[model_name], 
            "messages":[],
            "max_tokens": 2048,
            "temperature": 0
        }
    }
    
    batches = []
    def _prepare_batch_input(row, shot, use_raw_format):
        id = row["id"]
        if shot == "0":
            system_prompt = base_system_prompt
        else:
            sample = fewshot_map.get(id)
            if sample is None:
                raise ValueError(f"[Fewshot Error] No fewshot example for id: {id}")
            
            fewshot_examples = build_fewshot_examples(sample, shot, use_raw_format)
            system_prompt = base_system_prompt.replace("{fewshot_examples}", fewshot_examples)
        
        if use_raw_format:
            user_prompt = base_user_prompt.replace("{title}", row["title"]).replace("{content}", row["content"])
        else:
            user_prompt = base_user_prompt.replace("{question}", row["question"])
        
        temp = deepcopy(init_template)
        temp['custom_id'] = str(id)
        temp['body']['messages'].append({"role": "system", "content": system_prompt})
        temp['body']['messages'].append({"role": "user", "content": user_prompt})
        batches.append(temp)
        
    for _, row in df.iterrows():
        _prepare_batch_input(row, shot, use_raw_format)
        
    batch_input_file = f"data/batch_input_{model_name}_{shot}.jsonl"
    if use_raw_format:
        batch_input_file = batch_input_file.replace(".jsonl", "_raw.jsonl")
        
    with open(batch_input_file, 'w', encoding='utf-8') as file:
        for item in batches:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("OpenAI 서버에 입력 파일 업로드 중...")
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
    print(f"ID: {batch_job.id}")
    
    status_transition = False
    try:
        while True:
            batch = client.batches.retrieve(batch_job.id)
            
            if batch.status == "validating":
                print("유효성 검사 중...")
                
            if not status_transition and batch.status == "in_progress":
                print("배치 작업 진행 중...")
                status_transition = True
                
            if batch.status == "completed":
                print("배치 작업이 성공적으로 완료되었습니다.")
                break
            
            if time.time() - start_time > MAX_WAIT_TIME:
                print(f"최대 대기 시간: {MAX_WAIT_TIME//3600}시간을 초과했습니다.")
                break
            
            if batch.status == "failed":
                print("배치 작업이 실패했습니다.")
                print(batch.incomplete_details)
                break
            
            # 일부 실패 시 (status는 completed지만 error_file_id가 존재)
            if batch.error_file_id:
                print("일부 배치 작업이 실패했습니다.")
                error_file = client.files.content(batch.error_file_id)
                with open("error_file.jsonl", "wb") as f:
                    f.write(error_file)
                print(error_file)
            
            # 과도한 요청 방지
            time.sleep(60)
        
        output_file_id = batch.output_file_id
        file_response = client.files.content(output_file_id).content
        
        batch_output_file = f"data/batch_output_{model_name}_{shot}.jsonl"
        if use_raw_format:
            batch_output_file = batch_output_file.replace(".jsonl", "_raw.jsonl")
            
        with open(batch_output_file, "wb") as f:
            f.write(file_response)
        
        contents = []
        with open(batch_output_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                cleaned_content = re.sub(r"^\s*(#{1,6})?\s*답변\s*[:：]?\s*\n*", "", content.strip(), flags=re.IGNORECASE)
                contents.append(cleaned_content)
                
        df['generated_answer'] = contents
        batch_output_file = f"data/eval/output_{model_name}_{shot}.json"
        if use_raw_format:
            batch_output_file = batch_output_file.replace(".json", "_raw.json")
            
        df.to_json(batch_output_file, orient='records', force_ascii=False, indent=4) 
        print(f"{batch_output_file}")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        
    elapsed = time.time() - start_time
    print(f"배치 작업 총 소요 시간: {format_time(elapsed)}")
        
def run_batch_claude_pipeline(df, model_name, client, base_system_prompt, base_user_prompt, shot, fewshot_map, use_raw_format):
    print("배치 작업을 위한 요청 준비 중...")
    start_time = time.time()
    
    batches = []
    def _prepare_batch_input(row, shot, use_raw_format):
        id = row["id"]
        if shot == "0":
            system_prompt = base_system_prompt
        else:
            sample = fewshot_map.get(id)
            if sample is None:
                raise ValueError(f"[Fewshot Error] No fewshot example for id: {id}")
            
            fewshot_examples = build_fewshot_examples(sample, shot, use_raw_format)
            system_prompt = base_system_prompt.replace("{fewshot_examples}", fewshot_examples)
            
        if use_raw_format:
            user_prompt = base_user_prompt.replace("{title}", row["title"]).replace("{content}", row["content"])
        else:
            user_prompt = base_user_prompt.replace("{question}", row["question"])
            
        request = Request(
            custom_id=id,
            params=MessageCreateParamsNonStreaming(
                model=MODEL_NAME_TO_API_ID[model_name],
                temperature = 0,
                max_tokens=2048,
                system = system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
        )
        batches.append(request)
            
    for _, row in df.iterrows():
        _prepare_batch_input(row, shot, use_raw_format)
    
    # batch_input_file = f"data/batch_input_{model_name}_{shot}.jsonl"
    # if use_raw_format:
    #     batch_input_file = batch_input_file.replace(".jsonl", "_raw.jsonl")
    
    # with open(batch_input_file, 'w', encoding='utf-8') as file:
    #     for item in batches:
    #         json_string = json.dumps(item, ensure_ascii=False)
    #         file.write(json_string + '\n')
            
    message_batch = client.messages.batches.create(requests=batches)
    batch_id = message_batch.id
    print(f"ID: {batch_id}")
    print("배치 작업 진행 중...")
    
    try:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            
            if batch.processing_status == "canceling":
                print("요청이 취소되었습니다.")
                break
            
            elif batch.processing_status == "ended":
                print("배치 작업이 완료되었습니다.")
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
                        print(f"Validation error {result.custom_id}")
                    else:
                        # 요청을 직접 재시도할 수 있음
                        print(f"Server error {result.custom_id}")
                    contents.append("")
                case "expired":
                    print(f"Request expired {result.custom_id}")
                    contents.append("")
        
        df['generated_answer'] = contents
        batch_output_file = f"data/eval/output_{model_name}_{shot}.json"
        if use_raw_format:
            batch_output_file = batch_output_file.replace(".json", "_raw.json")
        
        df.to_json(batch_output_file, orient='records', force_ascii=False, indent=4)
        print(f"{batch_output_file}")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        
    elapsed = time.time() - start_time
    print(f"배치 작업 총 소요 시간: {format_time(elapsed)}")


def process_with_model(df, model_name, shot, use_batch_api, use_raw_format):
    print(f"MODEL NAME: {model_name}")
    print(f"SHOT: {shot}")
    print(f"USE BATCH API: {use_batch_api}")
    print(f"USE RAW FORMAT: {use_raw_format}")
    
    base_system_prompt, base_user_prompt = get_prompts(model_name, shot, use_raw_format)
    if use_raw_format:
        fewshot_examples_path = "data/training/fewshot_examples_raw.json"
    else:
        fewshot_examples_path = "data/training/fewshot_examples.json"
        
    fewshot_df = pd.read_json(fewshot_examples_path)
    fewshot_map = {row['id']: row for _, row in fewshot_df.iterrows()}
    
    model_family = parse_model_name(model_name)
    
    if use_batch_api:
        if model_family == "gpt":
            run_batch_gpt_pipeline(df, model_name, openai_client, base_system_prompt, base_user_prompt, shot, fewshot_map, use_raw_format)
        elif model_family == "claude":
            run_batch_claude_pipeline(df, model_name, anthropic_client, base_system_prompt, base_user_prompt, shot, fewshot_map, use_raw_format)
        elif model_family == "gemini":
            pass
    else:
        df_len = len(df)
        generated_data = []
        start_idx = 0
        output_path = f"data/eval/output_{model_name}_{shot}.json"
        if use_raw_format:
            output_path = output_path.replace(".json", "_raw.json")
        
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                generated_data = json.load(f)
            start_idx = len(generated_data)
            print(f"{start_idx}개까지 처리됨. 이어서 시작")
        else:
            print("새로 시작")
        
        model_processor = get_model_processor(model_name)
        
        for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=df_len - start_idx, desc="답변 생성 중"):
            id = row['id']
            if shot == "0":
                system_prompt = base_system_prompt
            else:
                sample = fewshot_map.get(id)
                if sample is None:
                    raise ValueError(f"[Fewshot Error] No fewshot example for id: {id}")
                
                fewshot_examples = build_fewshot_examples(sample, shot, use_raw_format)
                system_prompt = base_system_prompt.replace("{fewshot_examples}", fewshot_examples)
                
            if use_raw_format:
                user_prompt = base_user_prompt.replace("{title}", row["title"]).replace("{content}", row["content"])
            else:
                user_prompt = base_user_prompt.replace("{question}", row["question"])
            
            raw_output = model_processor(system_prompt, user_prompt)
            processed_output = process_model_output(raw_output)
            
            df.loc[i, 'generated_answer'] = processed_output.get("답변", "")
            generated_data.append(df.loc[i].to_dict())
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(generated_data, f, ensure_ascii=False, indent=4)
        print(f"{output_path}")
        
    print("답변 생성 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="답변 생성")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_NAME_TO_API_ID.keys()))
    parser.add_argument("--shot", type=str, required=True, choices=["0", "1", "3", "6"])
    parser.add_argument("--use_batch_api", action="store_true")
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    if args.use_raw_format:
        df = pd.read_json("data/training/test_data_raw.json")
    else:
        df = pd.read_json("data/training/test_data.json")
    
    process_with_model(df, args.model_name, args.shot, args.use_batch_api, args.use_raw_format)
    