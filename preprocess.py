import pandas as pd
import time
import re
import json
from tqdm import tqdm
from copy import deepcopy
import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import (
    MODEL_MAPPING, format_time, setup_logging, save_json, load_json,
    load_results, load_prompt, load_environment
)

import openai
from dotenv import load_dotenv
import tiktoken

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def num_tokens_from_messages(messages, model="gpt-4o-mini"):
    """
    메시지 목록에서 토큰 수를 반환합니다.
    (OpenAI Cookbook 예제 기반)
    """
    encoding = tiktoken.encoding_for_model(model)
    
    tokens_per_message = 3
    tokens_per_name = 1
        
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def call_openai_api(messages, model_name="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=MODEL_MAPPING[model_name],
        messages=messages,
        seed=42,
        response_format={
            "type": "json_schema", 
            "json_schema": {
                "name": "preprocess_question_and_answers",
                "schema": {
                    "type": "object",
                    "properties": {
                        "preprocessed_question": {
                            "type": "string"
                        },
                        "preprocessed_answers": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["preprocessed_question", "preprocessed_answers"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )
    json_string = response.choices[0].message.content.strip()
    
    prompt_tokens_used = response.usage.prompt_tokens
    completion_tokens_used = response.usage.completion_tokens
    total_tokens_used = response.usage.total_tokens

    print(f"API 응답 - 실제 사용된 프롬프트 토큰: {prompt_tokens_used} 토큰")
    print(f"API 응답 - 실제 사용된 완료 토큰: {completion_tokens_used} 토큰")
    print(f"API 응답 - 총 사용된 토큰: {total_tokens_used} 토큰")
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Received content: {json_string}")
        # 에러 발생 시 적절한 에러 처리 또는 기본값 반환
        return {"preprocessed_question": "", "preprocessed_answers": []}

def remove_common_greetings(text):
    if not isinstance(text, str):
        return ''
    
    # 유니코드 특수문자(제로폭 공백 등) 제거
    text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)
    
    patterns = [
        # 시작 문구
        r'^안녕하세요.*?입니다\.?\s*',
        
        # 끝 문구
        r'\s*감사합니다\.?\s*$',
        r'\s*고맙습니다\.?\s*$',
        r'\s*안녕히\s*계세요\.?\s*$',
        r'\s*안녕히\s*가세요\.?\s*$',
        r'\s*수고하세요\.?\s*$',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.UNICODE).strip()
    
    return text

def filtering(df, env, logger):
    start_time = time.time()
    filtered_data, start_idx = load_results(env["filtered_data_path"])
    system_prompt = load_prompt(env["system_filtering_prompt_path"])
    base_user_prompt = load_prompt(env["user_preprocessing_prompt_path"])
    
    df_to_process = df.iloc[start_idx:]
    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Filtering"):
        title = row.get('title', '')
        content = row.get('content', '')
        answer = row.get('answer', '')
        user_prompt = base_user_prompt.format(title=title, content=content, answer=answer)
        
        is_relevant = call_openai_api(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        current_item = row.fillna("").to_dict()
        current_item['is_relevant'] = is_relevant
        filtered_data.append(current_item)
        save_json(filtered_data, env["filtered_data_path"])

    relevant_count = sum(1 for item in filtered_data if item.get('is_relevant') == 'True')
    print(f"총 원본 데이터 개수: {len(df)}개")
    print(f"필터링된 데이터 개수: {len(df) - relevant_count}개")
    print(f"클리닝할 데이터 개수: {relevant_count}개")
    elapsed = time.time() - start_time
    print(f"필터링 총 소요 시간: {format_time(elapsed)}")
    logger.info(f"필터링 완료")

def cleaning(df, env, logger):
    start_time = time.time()
    cleaned_data, start_idx = load_results(env["cleaned_data_path"])
    system_prompt = load_prompt(env["system_cleaning_prompt_path"])
    base_user_prompt = load_prompt(env["user_preprocessing_prompt_path"])
    
    relevant_items_to_process = df.iloc[start_idx:].to_dict('records')
    for item in tqdm(relevant_items_to_process, total=len(relevant_items_to_process), desc="Cleaning"):
        title = item.get('title', '')
        content = item.get('content', '')
        raw_answers = item.get('answers', [])
        answer_texts = [ans['answer'] for ans in raw_answers]
        formatted_answers = "\n".join([f"- {i+1}. {ans_text}" for i, ans_text in enumerate(answer_texts)])

        user_prompt = base_user_prompt.format(title=title, content=content, answers=formatted_answers)
        cleaned_qna = call_openai_api(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        # print(cleaned_qna)
        preprocessed_question = cleaned_qna.get('preprocessed_question', '')
        preprocessed_answers = cleaned_qna.get('preprocessed_answers', [])


        cleaned_item = deepcopy(item)
        cleaned_item['preprocessed_question'] = preprocessed_question
        cleaned_item['preprocessed_answers'] = preprocessed_answers
        cleaned_data.append(cleaned_item)
        save_json(cleaned_data, env["cleaned_data_path"])
    
    elapsed = time.time() - start_time
    print(f"클리닝 총 소요 시간: {format_time(elapsed)}")
    logger.info(f"클리닝 완료")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="cleaning", choices=["filtering", "cleaning"])
    args = parser.parse_args()
    
    env = load_environment()
    df = pd.read_json(env["raw_data_path"], encoding='utf-8')
    logger = setup_logging()
    logger.info(f"MODE: {args.mode}")
    logger.info(f"RAW DATA PATH: {env['raw_data_path']}")
    
    if args.mode == "filtering":
        logger.info(f"FILTERED DATA PATH: {env['filtered_data_path']}")
        filtering(df, env, logger)
        
    elif args.mode == "cleaning":
        logger.info(f"CLEANED DATA PATH: {env['cleaned_data_path']}")
        cleaning(df, env, logger)
