import os
import pandas as pd
import argparse

import torch.nn.functional as F

from rouge import Rouge
from bert_score import score
from sentence_transformers import SentenceTransformer, util

from dotenv import load_dotenv
load_dotenv()
cache_dir = "./models"

import openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


MODEL_NAME_TO_API_ID = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
}

with open("prompt/evaluation_system.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

with open("prompt/evaluation_user.txt", "r", encoding="utf-8") as file:
    base_user_prompt = file.read()
    
    
def compute_avg_rougeL_f1(gold_answers, generated_answers):
    rouge = Rouge()
    scores = [
        rouge.get_scores(g, p)[0]['rouge-l']['f']
        for g, p in zip(gold_answers, generated_answers)
    ]
    avg_rougeL_f1 = sum(scores) / len(scores)
    print(f"Avg ROUGE-L F1: {avg_rougeL_f1:.2f}")
    
def compute_avg_bertscore_f1(gold_answers, generated_answers):
    _, _, F1 = score(
        generated_answers,
        gold_answers,
        lang="ko"  # 'bert-base-multilingual-cased'
    )
    avg_bertscore_f1 = F1.mean().item()
    print(f"Avg BERTScore F1: {avg_bertscore_f1:.2f}")
    
def compute_avg_llm_similarity(gold_answers, generated_answers):
    def _llm_as_a_judge(text1, text2):
        user_prompt = base_user_prompt.replace("{text1}", text1).replace("{text2}", text2)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}],
            seed=42
        )
        return float(response.choices[0].message.content.strip())
    
    similarity_scores = []
    for i in range(len(gold_answers)):
        score = _llm_as_a_judge(gold_answers[i], generated_answers[i])
        similarity_scores.append(score)
    avg_similarity_score = sum(similarity_scores) / len(similarity_scores)
    print(f"LLM-based Avg Similarity: {avg_similarity_score:.2f}")

def compute_avg_sbert_similiarity(model_name, gold_answers, generated_answers):
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    gold_embeddings = model.encode(gold_answers)
    generated_embeddings = model.encode(generated_answers)
    cosine_scores = util.pytorch_cos_sim(gold_embeddings, generated_embeddings)
    avg_similarity_score = cosine_scores.mean().item()
    print(f"SBERT-based Avg Similarity: {avg_similarity_score:.2f}")

def load_answer(file_path):
    df = pd.read_json(file_path)
    gold_answers = df['answer']
    generated_answers = df['generated_answer']
    return gold_answers, generated_answers


def evaluate_answer(file_path):
    gold_answers, generated_answers = load_answer(file_path)
    compute_avg_rougeL_f1(gold_answers, generated_answers)
    compute_avg_bertscore_f1(list(gold_answers), list(generated_answers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="평가용 데이터 입력")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_NAME_TO_API_ID.keys()))
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    file_path = f"data/eval/output_{args.model_name}_0shot"
    if args.use_raw_format:
        file_path += "_raw"
    file_path += ".json"
    print(f"평가 파일: {file_path}")
    
    evaluate_answer(file_path)
    