"""judge LLM이 평가할 json 파일을 생성"""
import os
import re
import json
import argparse
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import openai


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

MODEL_MAPPING = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct"
}


def load_prompt(file_path):
    with open(file_path, encoding="utf-8") as file:
        return file.read()

def get_evaluation_prompts(use_raw_format):
    prompt_dir = "prompt/evaluation"
    prompts = {
        "score_system": load_prompt(f"{prompt_dir}/system_score.txt"),
        "pairwise_system": load_prompt(f"{prompt_dir}/system_pairwise.txt")
    }
    
    if use_raw_format:
        prompts["score_user"] = load_prompt(f"{prompt_dir}/user_score_raw.txt")
        prompts["pairwise_user"] = load_prompt(f"{prompt_dir}/user_pairwise_raw.txt")
    else:
        prompts["score_user"] = load_prompt(f"{prompt_dir}/user_score.txt")
        prompts["pairwise_user"] = load_prompt(f"{prompt_dir}/user_pairwise.txt")
    return prompts

def query_llm(system_prompt, user_prompt, model_name="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=MODEL_MAPPING[model_name],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        seed=42
    )
    return response.choices[0].message.content.strip()

def parse_json_safely(json_str):
    try:
        # 괄호 개수 일치 보정
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        if open_braces > close_braces:
            json_str += "}" * (open_braces - close_braces)
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"JSONDecodeError: {json_str}")
        return {
            "factuality": "",
            "completeness": "",
            "coherence": ""
        }

def get_file_paths(model_name, shot, use_raw_format):
    suffix = "_raw" if use_raw_format else ""
    input_path = f"data/eval/output_{model_name}_{shot}{suffix}.json"
    score_path = f"data/llm_eval/score_results_{model_name}_{shot}{suffix}.json"
    pairwise_path = f"data/llm_eval/pairwise_results_{model_name}_{shot}{suffix}.json"
    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    os.makedirs(os.path.dirname(pairwise_path), exist_ok=True)
    
    paths = {
        "input": input_path,
        "score": score_path,
        "pairwise": pairwise_path
    }
    return paths

def load_existing_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def evaluate_response(evaluation_type, row_data, prompts, output_path, use_raw_format, results=None):
    if results is None:
        results = []
    
    if use_raw_format:
        title = row_data.get("title", "")
        content = row_data.get("content", "")
        gold_answer = row_data.get("preprocessed_answer", "")
        generated_answer = row_data.get("generated_answer", "")
    else:
        question = row_data.get("preprocessed_question", "")
        gold_answer = row_data.get("preprocessed_answer", "")
        generated_answer = row_data.get("generated_answer", "")
        
    if evaluation_type == "score":
        system_prompt = prompts["score_system"]
        base_user_prompt = prompts["score_user"]
        
        if use_raw_format:
            user_prompt = base_user_prompt.replace("{title}", title).replace("{content}", content).replace("{gold_answer}", gold_answer).replace("{generated_answer}", generated_answer)
        else:
            user_prompt = base_user_prompt.replace("{question}", question).replace("{gold_answer}", gold_answer).replace("{generated_answer}", generated_answer)
    else:
        system_prompt = prompts["pairwise_system"]
        base_user_prompt = prompts["pairwise_user"]
        
        if use_raw_format:
            user_prompt = base_user_prompt.replace("{title}", title).replace("{content}", content).replace("{response_a}", gold_answer).replace("{response_b}", generated_answer)
        else:
            user_prompt = base_user_prompt.replace("{question}", question).replace("{response_a}", gold_answer).replace("{response_b}", generated_answer)
    
    result = query_llm(system_prompt, user_prompt)
    parsed = parse_json_safely(result)
    
    evaluation_result = {
        "id": row_data.get("id", None),
        "factuality": parsed.get("factuality", ""),
        "completeness": parsed.get("completeness", ""),
        "coherence": parsed.get("coherence", "")
    }
    
    if use_raw_format:
        evaluation_result = {
            "id": row_data.get("id", None),
            "title": title,
            "content": content,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "factuality": parsed.get("factuality", ""),
            "completeness": parsed.get("completeness", ""),
            "coherence": parsed.get("coherence", "")
        }
    else:
        evaluation_result = {
            "id": row_data.get("id", None),
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "factuality": parsed.get("factuality", ""),
            "completeness": parsed.get("completeness", ""),
            "coherence": parsed.get("coherence", "")
        }
    
    results.append(evaluation_result)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    return results

def generate_evaluations(input_path, score_output_path, pairwise_output_path, use_raw_format):
    df = pd.read_json(input_path).head(200)
    df_len = len(df)
    print(f"평가 파일: {input_path}")
    print(f"평가 데이터 수: {df_len}")
    
    prompts = get_evaluation_prompts(use_raw_format)
    
    score_results = load_existing_results(score_output_path)
    start_idx = len(score_results)
    print(f"score 평가:\n{start_idx}개까지 처리됨. {'이어서 시작' if start_idx > 0 else '새로 시작'}")
    
    for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=df_len - start_idx, desc="평가 진행중"):
        row_dict = row.to_dict()
        score_results = evaluate_response(
            "score",
            row_dict,
            prompts,
            score_output_path,
            use_raw_format,
            score_results
        )
    
    pairwise_results = load_existing_results(pairwise_output_path)
    start_idx = len(pairwise_results)
    print(f"pairwise 평가:\n{start_idx}개까지 처리됨. {'이어서 시작' if start_idx > 0 else '새로 시작'}")
    
    for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=df_len - start_idx, desc="평가 진행중"):
        row_dict = row.to_dict()
        pairwise_results = evaluate_response(
            "pairwise",
            row_dict,
            prompts,
            pairwise_output_path,
            use_raw_format,
            pairwise_results
        )
        
    print("평가 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--shot", type=str, required=True, choices=["0", "1", "3", "6"])
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    paths = get_file_paths(args.model_name, args.shot, args.use_raw_format)
    
    generate_evaluations(
        paths["input"], 
        paths["score"], 
        paths["pairwise"], 
        args.use_raw_format
    )
    