import pandas as pd
import time
import re
import os
import json
from tqdm import tqdm
from copy import deepcopy

import openai
from dotenv import load_dotenv


MAX_WAIT_TIME = 24 * 60 * 60  

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


with open("prompt/filtering_system.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()
    
with open("prompt/filtering_user.txt", "r", encoding="utf-8") as file:
    base_user_prompt = file.read()
    
with open("prompt/cleaning_system.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()
    
with open("prompt/cleaning_user.txt", "r", encoding="utf-8") as file:
    base_user_prompt = file.read()


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    if hours > 0:
        return f"처리 시간: {hours}시간 {minutes}분 {seconds}초"
    elif minutes > 0:
        return f"처리 시간: {minutes}분 {seconds}초"
    else:
        return f"처리 시간: {seconds}초"

def prepare_qna_data(df):
    def _extract_text_data(df):
        extracted_df = df[(df['question_photo'] == "['사진 없음']") &
                        (df['answer_photo'] == "['사진 없음']") &
                        (df['question_video'] == '동영상 없음') &
                        (df['answer_video'] == '동영상 없음')]
        return extracted_df
    
    initial_row_count = len(df)
    df = _extract_text_data(df)
    extracted_row_count = len(df)
    df['id'] = list(range(len(df)))
    print(f"원본 데이터 크기: {initial_row_count}")
    print(f"추출된 데이터 크기: {extracted_row_count}")
    
    return df[['id', '제목', '본문', '답변', 'answer_date']]

def is_relevant_sample(title, text, answer):
    user_prompt = base_user_prompt.replace("{title}", title).replace("{text}", str(text)).replace("{answer}", answer)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        seed=42
    )
    
    return response.choices[0].message.content.strip()

def filtering(df):
    output_path = "data/filtered_data.json"
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            filtered_data = json.load(f)
        start_idx = len(filtered_data)
        print(f"{start_idx}개까지 처리됨. 이어서 시작")
    else:
        filtered_data = []
        start_idx = 0
        print("새로 시작")

    start_time = time.time()

    df_len = len(df)
    for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=df_len - start_idx, desc="Filtering"):
        df.loc[i, 'is_relevant'] = is_relevant_sample(row['제목'], row['본문'], row['답변'])
        filtered_data.append(df.loc[i].fillna("").to_dict())
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    initial_row_count = len(df)
    df = df[df['is_relevant'] == 'True']
    print(f"필터링된 데이터 개수: {initial_row_count - len(df)}개")
    print(f"클리닝할 데이터 개수: {len(df)}개")
    
    elapsed = time.time() - start_time
    print(f"필터링 총 소요 시간: {format_time(elapsed)}")
    
    print(f"필터링 완료: {output_path}")
    
    return df

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

def run_batch_pipeline(df):
    print("배치 작업을 위한 입력 파일 생성 중...")
    init_template = {
        "custom_id": None,
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-4o-mini", 
                "messages":[
                    {"role": "system", "content": system_prompt},
                    ],
                "max_tokens": 1000
                }
        }
    
    batches = []
    def _prepare_batch_input(title, text, answer, i):
        user_prompt = base_user_prompt.replace("{title}", title).replace("{text}", str(text)).replace("{answer}", answer)
        temp = deepcopy(init_template)
        temp['custom_id'] = f'{i}'
        temp['body']['messages'].append({"role": "user", "content": user_prompt})
        batches.append(temp)
        
    for _, row in df.iterrows():
        _prepare_batch_input(row['제목'], row['본문'], row['답변'], row['id'])
        
    batch_input_file = "data/batch_input.jsonl"
    with open(batch_input_file, 'w') as file:
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
            "description": "preprocessing"
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
            time.sleep(30)
        
        output_file_id = batch.output_file_id
        file_response = client.files.content(output_file_id).content
        
        batch_output_file = "data/batch_output_filtering.jsonl"
        with open(batch_output_file, "wb") as f:
            f.write(file_response)
            
    except Exception as e:
        print(f"오류 발생: {e}")
        
    elapsed = time.time() - start_time
    print(f"배치 작업 총 소요 시간: {format_time(elapsed)}")
        
    return batch_output_file

def cleaning(df):
    df['답변'] = df['답변'].apply(remove_common_greetings)
    
    # Batch API 활용
    batch_output_file = run_batch_pipeline(df)
    
    cleaned_questions = []
    cleaned_answers = []
    
    with open(batch_output_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            content = data["response"]["body"]["choices"][0]["message"]["content"]
            
            question_match = re.search(r"클리닝된 질문:\s*(.+)", content)
            answer_match = re.search(r"클리닝된 답변:\s*(.+)", content)
            
            cleaned_question = question_match.group(1).strip() if question_match else ""
            cleaned_question = "" if pd.isna(cleaned_question) else cleaned_question
            cleaned_answer = answer_match.group(1).strip() if answer_match else ""
            cleaned_answer = "" if pd.isna(cleaned_answer) else cleaned_answer
            
            cleaned_questions.append(cleaned_question)
            cleaned_answers.append(cleaned_answer)

    df['cleaned_question'] = cleaned_questions
    df['cleaned_answer'] = cleaned_answers

    cleaned_data_output_path = "data/cleaned_data.json"
    df.to_json(cleaned_data_output_path, orient='records', force_ascii=False, indent=4)
    print(f"클리닝 완료: {cleaned_data_output_path}")
    
    df['question'] = df['cleaned_question']
    df['answer'] = df['cleaned_answer']
    df = df[['id', 'question', 'answer', 'answer_date']]
    df = df[~((df['question'] == "") | (df['answer'] == ""))]
    preprocessed_df = df[['id', 'question', 'answer', 'answer_date']]

    preprocessed_data_output_path = "data/preprocessed_data.json"
    preprocessed_df.to_json(preprocessed_data_output_path, orient='records', force_ascii=False, indent=4)
    print(f"전처리 완료: {preprocessed_data_output_path}")
    
def preprocess():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists("prompt"):
        os.makedirs("prompt")

    raw_data_path = "data/med_expert.csv"
    df = pd.read_csv(raw_data_path, encoding='utf-8')
    
    extracted_df = prepare_qna_data(df)
    filtered_df = filtering(extracted_df)
    
    ''' 클리닝만 실행 '''
    # filtered_data_path = "data/filtered_data.json"
    # filtered_df = pd.read_json(filtered_data_path, encoding='utf-8')
    # filtered_df = filtered_df[filtered_df['is_relevant'] == 'True']
    
    cleaning(filtered_df)
    

if __name__ == "__main__":
    preprocess()

