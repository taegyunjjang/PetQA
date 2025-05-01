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
  
def get_prompts(model_name, shot):
    model_family = parse_model_name(model_name)
    system_prompt_path = f"prompt/generating_answer_{shot}_system.txt"
    user_prompt_path = f"prompt/generating_answer_{model_family}_user.txt"
    system_prompt = load_prompt(system_prompt_path)
    base_user_prompt = load_prompt(user_prompt_path)
    return system_prompt, base_user_prompt

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

def generate_gpt_answer(client, model_name, question, system_prompt, base_user_prompt):
    user_prompt = base_user_prompt.replace("{question}", question)
    response = client.chat.completions.create(
        model=MODEL_NAME_TO_API_ID[model_name],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        temperature=0,
        seed=42
    )
    return response.choices[0].message.content.strip()

def generate_claude_answer(client, model_name, question, system_prompt, base_user_prompt):
    user_prompt = base_user_prompt.replace("{question}", question)
    
    message = client.messages.create(
        model = MODEL_NAME_TO_API_ID[model_name],
        temperature = 0,
        max_tokens = 1024,
        system = system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return message.content[0].text

def generate_gemini_answer(model_name, question, system_prompt, base_user_prompt):
    user_prompt = base_user_prompt.replace("{question}", question)
    
    model = genai.GenerativeModel(
        model_name=MODEL_NAME_TO_API_ID[model_name], 
        system_instruction=system_prompt
    )
    
    generation_config = {
        "temperature": 0
    }
    
    response = model.generate_content(
        contents=[
            {"role": "user", "parts": [user_prompt]}
        ],
        generation_config=generation_config
    )
    
    return response.text
    
def generate_exaone_answer(model, tokenizer, question, system_prompt, base_user_prompt):
    user_prompt = base_user_prompt.replace("{question}", question)

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
        max_new_tokens=1024,
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

def run_batch_gpt_pipeline(df, model_name, client, system_prompt, base_user_prompt, shot):
    print("배치 작업을 위한 입력 파일 생성 중...")
    init_template = {
        "custom_id": None,
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {"model": MODEL_NAME_TO_API_ID[model_name], 
                "messages":[
                    {"role": "system", "content": system_prompt},
                    ],
                "max_tokens": 1024
                }
        }
    
    batches = []
    def _prepare_batch_input(question, i):
        user_prompt = base_user_prompt.replace("{question}", question)
        temp = deepcopy(init_template)
        temp['custom_id'] = f'{i}'
        temp['body']['messages'].append({"role": "user", "content": user_prompt})
        batches.append(temp)
        
    for _, row in df.iterrows():
        _prepare_batch_input(row['question'], row['id'])
        
    batch_input_file = f"data/batch_input_{model_name}_{shot}.jsonl"
    with open(batch_input_file, 'w', encoding='utf-8') as file:
        for item in batches:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + '\n')
    
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
    
    start_time = time.time()
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
        with open(batch_output_file, "wb") as f:
            f.write(file_response)
        
        contents = []
        with open(batch_output_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                contents.append(content)
                
        df['generated_answer'] = contents
        batch_output_file = f"data/eval/output_{model_name}_{shot}.json"
        df.to_json(batch_output_file, orient='records', force_ascii=False, indent=4) 
        print(f"{batch_output_file}")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        
    elapsed = time.time() - start_time
    print(f"배치 작업 총 소요 시간: {format_time(elapsed)}")
        
def run_batch_claude_pipeline(df, model_name, client, system_prompt, base_user_prompt, shot):
    print("배치 작업을 위한 요청 준비 중...")
    start_time = time.time()
    
    batches = []
    for _, row in df.iterrows():
        user_prompt = base_user_prompt.replace("{question}", row['question'])
        
        request = Request(
            custom_id=f"{row['id']}",
            params=MessageCreateParamsNonStreaming(
                model=MODEL_NAME_TO_API_ID[model_name],
                temperature = 0,
                max_tokens=1024,
                system = system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
        )
        batches.append(request)
    
    # batch_input_file = f"data/batch_input_{model_name}.jsonl"
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
        df.to_json(batch_output_file, orient='records', force_ascii=False, indent=4)
        print(f"{batch_output_file}")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        
    elapsed = time.time() - start_time
    print(f"배치 작업 총 소요 시간: {format_time(elapsed)}")


def process_with_model(df, model_name, use_batch_api, shot):
    print(f"모델명: {model_name}")
    
    system_prompt, base_user_prompt = get_prompts(model_name, shot)
    
    if use_batch_api:
        if model_name.startswith("gpt"):
            run_batch_gpt_pipeline(df, model_name, openai_client, system_prompt, base_user_prompt, shot)
        elif model_name.startswith("claude"):
            run_batch_claude_pipeline(df, model_name, anthropic_client, system_prompt, base_user_prompt, shot)
        elif model_name.startswith("gemini"):
            pass
    else:
        df_len = len(df)
        generated_data = []
        start_idx = 0
        output_path = f"data/eval/output_{model_name}_{shot}.json"
        
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                generated_data = json.load(f)
            start_idx = len(generated_data)
            print(f"{start_idx}개까지 처리됨. 이어서 시작")
        else:
            print("새로 시작")
        
        if model_name.startswith("gpt"):
            model_processor = lambda question: generate_gpt_answer(openai_client, model_name, question, system_prompt, base_user_prompt)
        elif model_name.startswith("claude"):
            model_processor = lambda question: generate_claude_answer(anthropic_client, model_name, question, system_prompt, base_user_prompt)
        elif model_name.startswith("gemini"):
            model_processor = lambda question: generate_gemini_answer(model_name, question, system_prompt, base_user_prompt)
        elif model_name.startswith("exaone"):
            model, tokenizer = load_hf_model_and_tokenizer(model_name, cache_dir)
            model_processor = lambda question: generate_exaone_answer(model, tokenizer, question, system_prompt, base_user_prompt)
        
        for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=df_len - start_idx, desc="답변 생성 중"):
            raw_output = model_processor(row['question'])
            cleaned_output = re.sub(r"^\s*(#{1,6})?\s*답변\s*[:：]?\s*\n*", "", raw_output.strip(), flags=re.IGNORECASE)

            df.loc[i, 'generated_answer'] = cleaned_output
            generated_data.append(df.loc[i].to_dict())
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(generated_data, f, ensure_ascii=False, indent=4)
        print(f"{output_path}")
        
    print("답변 생성 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 id 입력")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_NAME_TO_API_ID.keys()))
    parser.add_argument("--shot", type=str, required=True, choices=["0shot", "1shot", "2shot", "3shot"])
    parser.add_argument("--use_batch_api", action="store_true")
    args = parser.parse_args()
    
    df = pd.read_json("data/training/test_data.json")
    
    process_with_model(df, model_name=args.model_name, use_batch_api=args.use_batch_api, shot=args.shot)
    