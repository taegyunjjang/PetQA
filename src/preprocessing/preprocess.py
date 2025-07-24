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
    format_time, setup_logging, save_json, load_json,
    load_prompt, load_environment
)

from google import genai
from google.genai import types
from google.genai import errors

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)


def load_results(output_path, mode):
    if os.path.exists(output_path):
        if mode == "filtering":
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            last_item = results[-1]
            start_idx = last_item['q_id'] + 1
            print(f"{start_idx}개까지 처리됨. 이어서 시작")
        elif mode == "cleaning":
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            start_idx = len(results)
            print(f"{start_idx}개까지 처리됨. 이어서 시작")
    else:
        results = []
        start_idx = 0
        print("새로 시작")
    return results, start_idx

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

def parse_cleaned_qna(cleaned_qna_str, q_id, a_id, logger):
    try:
        cleaned_qna_dict = json.loads(cleaned_qna_str)
        return {
            "preprocessed_question": cleaned_qna_dict.get('preprocessed_question', ''),
            "preprocessed_answer": cleaned_qna_dict.get('preprocessed_answer', '')
        }
    except json.JSONDecodeError as e:
        if logger:
            logger.warning(f"[JSONDecodeError] q_id: {q_id}, a_id: {a_id} - 응답 파싱 실패\n내용: {cleaned_qna_str}\n에러: {e}")
        return None
    except Exception as e:
        if logger:
            logger.warning(f"[UnknownError] q_id: {q_id}, a_id: {a_id} - 예기치 않은 오류 발생\n내용: {cleaned_qna_str}\n에러: {e}")
        return None

class Preprocess(BaseModel):
    preprocessed_question: str
    preprocessed_answer: str

def call_gemini_api(mode, system_prompt, user_prompt):
    if mode == "filtering":
        config = types.GenerateContentConfig(
            system_instruction = system_prompt,
            temperature=0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )
    elif mode == "cleaning":
        config = types.GenerateContentConfig(
            system_instruction = system_prompt,
            temperature=0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            response_mime_type="application/json",
            response_schema=Preprocess
        )
    
    max_retries = 10
    delay = 1
    for i in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=user_prompt,
                config=config
            )
            return response.text.strip()
        except errors.ServerError as e:
            if e.code == 503: # ServerError 객체는 status_code 속성을 가집니다.
                print(f"Gemini API 503 에러 발생: {e.message}. {i+1}/{max_retries} 재시도. {delay:.1f}초 후 재시도...")
                time.sleep(delay)
                delay *= 2 
                if delay > 60: 
                    delay = 60
            else:
                # 503이 아닌 다른 ServerError는 재시도하지 않고 바로 발생시킵니다.
                print(f"Gemini API 서버 에러 발생 (Code: {e.status_code}): {e.message}")
                raise 
        except Exception as e:
            # 예상치 못한 다른 모든 에러는 바로 발생시킵니다.
            print(f"예상치 못한 에러 발생: {e}")
            raise

def filtering(mode, raw_data_list, env, logger):
    start_time = time.time()
    filtering_all_results, start_idx = load_results(env["filtering_all_results_path"], mode)
    system_prompt = load_prompt(env["system_filtering_prompt_path"])
    base_user_prompt = load_prompt(env["user_preprocessing_prompt_path"])
    
    raw_data_to_process = raw_data_list[start_idx:]
    total_data_count = len(raw_data_to_process)
    print(f"필터링할 데이터: {total_data_count}개")
    
    for item in tqdm(raw_data_to_process, total=total_data_count, desc="Filtering"):
        q_id = item.get('q_id', '')
        title = item.get('title', '')
        content = item.get('content', '')
        answers = item.get('answers', [])
        question_date = item.get('question_date', '')
        animal_type = item.get('animal_type', '')
        
        for a in answers:
            if a.get("selected"):
                a_id = a.get('a_id', '')
                answer_type = a.get('answer_type', '')
                answer = a.get('answer', '')
                answer_date = a.get('answer_date', '')
                
                user_prompt = base_user_prompt.format(title=title, content=content, answer=answer)
                is_relevant = call_gemini_api(mode, system_prompt, user_prompt)
                
                time.sleep(0.2)  # 지연 시간 추가
                
                filtering_result = {
                    "q_id": q_id,
                    "title": title,
                    "content": content,
                    "answer": answer,
                    "a_id": a_id,
                    "answer_type": answer_type,
                    "question_date": question_date,
                    "answer_date": answer_date,
                    "animal_type": animal_type,
                    "is_relevant": is_relevant.lower()
                }
                filtering_all_results.append(filtering_result)
                save_json(filtering_all_results, env["filtering_all_results_path"])

    irrelevant_data = [item for item in filtering_all_results if item['is_relevant'] == "false"]
    save_json(irrelevant_data, env["irrelevant_data_path"])
    relevant_data = [item for item in filtering_all_results if item['is_relevant'] == "true"]
    save_json(relevant_data, env["filtered_data_path"])
    
    elapsed = time.time() - start_time
    print(f"필터링 총 소요 시간: {format_time(elapsed)}")
    logger.info(f"필터링 완료")

def cleaning(mode, filtered_data_list, env, logger, batch_size):
    start_time = time.time()
    cleaned_data, start_idx = load_results(env["cleaned_data_path"], mode)
    system_prompt = load_prompt(env["system_cleaning_prompt_path"])
    base_user_prompt = load_prompt(env["user_preprocessing_prompt_path"])
    
    filtered_data_to_process = filtered_data_list[start_idx:]
    total_data_count = len(filtered_data_to_process)
    print(f"클리닝할 데이터: {total_data_count}개")
    
    current_batch = []
    for i, item in enumerate(tqdm(filtered_data_to_process, total=total_data_count, desc="Cleaning")):
        q_id = item.get('q_id', '')
        title = item.get('title', '')
        content = item.get('content', '')
        answer = remove_common_greetings(item.get('answer', ''))
        a_id = item.get('a_id', '')
        answer_type = item.get('answer_type', '')
        question_date = item.get('question_date', '')
        answer_date = item.get('answer_date', '')
        animal_type = item.get('animal_type', '')

        user_prompt = base_user_prompt.format(title=title, content=content, answer=answer)
        cleaned_qna_str = call_gemini_api(mode, system_prompt, user_prompt)
        parsed_result = parse_cleaned_qna(cleaned_qna_str, q_id, a_id, logger)
        
        if parsed_result is None:
            continue
        
        preprocessed_question = parsed_result['preprocessed_question']
        preprocessed_answer = parsed_result['preprocessed_answer']

        cleaned_item = {
            "q_id": q_id,
            "title": title,
            "content": content,
            "answer": answer,
            "a_id": a_id,
            "answer_type": answer_type,
            "question_date": question_date,
            "answer_date": answer_date,
            "animal_type": animal_type,
            "preprocessed_question": preprocessed_question,
            "preprocessed_answer": preprocessed_answer
        }
        current_batch.append(cleaned_item)
        
        if (i + 1) & batch_size == 0 or (i + 1) == total_data_count:
            cleaned_data.extend(current_batch)
            save_json(cleaned_data, env["cleaned_data_path"])
            current_batch = []
    
    elapsed = time.time() - start_time
    print(f"클리닝 총 소요 시간: {format_time(elapsed)}")
    logger.info(f"클리닝 완료")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["filtering", "cleaning"], required=True)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    logger.info(f"MODE: {args.mode}")
    
    if args.mode == "filtering":
        raw_data_list = load_json(env["raw_data_path"])
        logger.info(f"RAW DATA PATH: {env['raw_data_path']}")
        filtering(args.mode, raw_data_list, env, logger)
        
    elif args.mode == "cleaning":
        filtered_data_list = load_json(env["filtered_data_path"])
        logger.info(f"FILTERED DATA PATH: {env['filtered_data_path']}")
        cleaning(args.mode, filtered_data_list, env, logger, args.batch_size)
