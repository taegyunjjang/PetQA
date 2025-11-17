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
from pydantic import BaseModel

class Keywords(BaseModel):
    keywords: list[str]

from dotenv import load_dotenv
import openai
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
train_path = "/home/work/factchecking/PetQA/data/processed/train.json"
validation_path = "/home/work/factchecking/PetQA/data/processed/validation.json"
test_path = "/home/work/factchecking/PetQA/data/processed/test.json"

with open(train_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)
with open(validation_path, 'r', encoding='utf-8') as f:
    validation_data = json.load(f)
with open(test_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
    
data = train_data + validation_data + test_data

BASE_PROMPT = """
    주어진 텍스트는 반려동물(개, 고양이) 의료 상담 관련 질의응답 데이터셋의 샘플입니다.
    당신의 역할은 텍스트에서 한국어 위키피디아 검색을 위한 명사 중심 키워드를 최대 10개까지 추출하는 것입니다.
    아래의 예시를 참고하여 키워드만 출력하고, 불필요한 문장은 포함하지 마세요.
    
    - 가능한 한 구체적인 의학/생리/질병/약물 관련 명사를 포함하세요.
    - 반려동물 품종(예: 말티즈, 페르시안)도 포함합니다.
    - 겹치는 개념어(예: 암, 림프암)는 더 구체적인 용어(림프암)를 우선합니다.
    - 질병명, 검사명, 약물명, 해부학적 부위, 관련 생리현상, 의료행위 등이 주요 타깃입니다.

    ### 예시
    질문: 5살 말티즈를 키우는데, 얼마 전 목 한쪽에 혹이 만져져 병원에서 세포 검사 결과 세포 분열 양상을 보여 암이라고 했습니다. 추가 검사를 위해 2주를 기다리는 중인데, 혹이 빠르게 커졌다가 3일 후부터는 크기가 유지되더니 어제는 거의 반 크기로 줄었습니다. 그리고 밤에 피를 토해 이불에 피가 묻어 있었고, 이후 혹 크기가 더 줄었습니다. 병원에서는 90% 암이라고 했다가 100%라고 하는데, 혹 크기가 급격히 변하는 것이 암이 아닐 가능성이 있을까요?
    답변: FNA 세포학 검사를 통해 90~100% 림프암이 의심되며, 초기 약물로 스테로이드가 사용되었을 가능성이 있습니다. 혹의 크기가 줄어드는 것은 스테로이드의 효과일 수 있으며, 구토와 같은 부작용도 나타날 수 있습니다. 현재 세포학 검사는 진행 중이므로 확진이 나오기 전까지 기다리는 것이 좋습니다. 항암 치료에 대한 정보도 미리 확인하는 것이 필요합니다. 종양이 아니기를 바랍니다.
    출력: ["강아지", "말티즈", "FNA", "세포학 검사", "림프암", "스테로이드", "항암 치료"]
    
    질문: {question}
    답변: {reference_answer}
    출력:
    """

def call_openai_api(client, user_prompt, model_name="gpt-4o-mini"):
    response = client.chat.completions.parse(
        model=MODEL_MAPPING[model_name],
        messages=[{"role": "user", "content": user_prompt}],
        response_format=Keywords,
        max_tokens=512,
        temperature=0,
        seed=42
    )
    return response.choices[0].message.content.strip()

output_path = "./keywords.json"
keywords_list, start_idx = load_results(output_path)
data_to_process = data[start_idx:]
total_data_count = len(data_to_process)

for item in tqdm(data_to_process, total=total_data_count, desc="Extracting keywords"):
    question = item["question"]
    reference_answer = item["reference_answer"]
    user_prompt = BASE_PROMPT.format(question=question, reference_answer=reference_answer)
    keywords = call_openai_api(openai_client, user_prompt)
    try:
        parsed = json.loads(keywords)
        if isinstance(parsed, dict) and "keywords" in parsed:
            parsed = parsed["keywords"]
        # 문자열이나 리스트 외의 값이 오면 문자열로 감싸기
        elif not isinstance(parsed, list):
            parsed = [str(parsed)]
    except json.JSONDecodeError:
        # JSON이 아닐 경우: 콤마 구분 리스트로 대체
        parsed = [kw.strip() for kw in keywords.strip("[]").replace('"', "").split(",") if kw.strip()]

    keywords_list.append({
        "q_id": item["q_id"],
        "a_id": item["a_id"],
        "question": question,
        "reference_answer": reference_answer,
        "keywords": parsed
    })
    save_json(keywords_list, output_path)