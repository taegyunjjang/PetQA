import time
import os
import pandas as pd
import argparse

import torch.nn.functional as F

from rouge import Rouge
from bert_score import score
from BARTScore.bart_score import BARTScorer
from sentence_transformers import SentenceTransformer, util

from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
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

def get_prompts():
    system_score_prompt_path = f"prompt/evaluation/system_score.txt"
    user_score_prompt_path = f"prompt/evaluation/user_score.txt"
    system_pairwise_prompt_path = f"prompt/evaluation/system_pairwise.txt"
    user_pairwise_prompt_path = f"prompt/evaluation/user_pairwise.txt"
    
    system_score_prompt = load_prompt(system_score_prompt_path)
    user_score_prompt = load_prompt(user_score_prompt_path)
    system_pairwise_prompt = load_prompt(system_pairwise_prompt_path)
    user_pairwise_prompt = load_prompt(user_pairwise_prompt_path)
    return system_score_prompt, user_score_prompt, system_pairwise_prompt, user_pairwise_prompt
    
def compute_avg_rougeL_f1(gold_answers, generated_answers):
    rouge = Rouge()
    scores = []
    for g, p in zip(gold_answers, generated_answers):
        if p == "":
            scores.append(0)
        else:
            scores.append(rouge.get_scores(g, p)[0]['rouge-l']['f'])
    
    avg_rougeL_f1 = sum(scores) / len(scores)
    print(f"Avg ROUGE-L F1: {avg_rougeL_f1:.3f}")
    
def compute_avg_bertscore_f1(gold_answers, generated_answers):
    valid_pairs = [(g, p) for g, p in zip(gold_answers, generated_answers) if p != ""]
        
    valid_gold, valid_gen = zip(*valid_pairs)
    _, _, F1 = score(
        valid_gen,
        valid_gold,
        lang="ko"  # 'bert-base-multilingual-cased'
    )
    
    empty_count = sum(1 for p in generated_answers if p == "")
    total_scores = F1.tolist() + [0] * empty_count
    
    avg_bertscore_f1 = sum(total_scores) / len(generated_answers)
    print(f"Avg BERTScore F1: {avg_bertscore_f1:.3f}")
    
def compute_avg_bartscore(gold_answers, generated_answers):
    model_path = "models/bart_score.pth"
    bart_scorer = BARTScorer(checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path=model_path)
    
    valid_pairs = [(g, p) for g, p in zip(gold_answers, generated_answers) if p != ""]
    
    valid_gold, valid_gen = zip(*valid_pairs)
    
    precision = bart_scorer.score(valid_gold, valid_gen)
    recall = bart_scorer.score(valid_gen, valid_gold)
    
    def _normalize_bart_score(score):
        min_score = min(score)
        max_score = max(score)
        return [(s - min_score) / (max_score - min_score) if max_score != min_score else 0 for s in score]
    
    normalized_precision = _normalize_bart_score(precision)
    normalized_recall = _normalize_bart_score(recall)
    
    empty_count = sum(1 for p in generated_answers if p == "")
    total_precision = normalized_precision + [0] * empty_count
    total_recall = normalized_recall + [0] * empty_count
    
    avg_precision = sum(total_precision) / len(generated_answers)
    avg_recall = sum(total_recall) / len(generated_answers)
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    print(f"Avg BARTScore F1: {avg_f1:.3f}")

def load_answer(file_path):
    df = pd.read_json(file_path)
    gold_answers = df['answer']
    generated_answers = df['generated_answer']
    return gold_answers, generated_answers

def evaluate_answer(file_path):
    print(f"평가 파일: {file_path}")
    start_time = time.time()
    
    gold_answers, generated_answers = load_answer(file_path)
    print(f"평가 개수: {len(gold_answers)}건")
    compute_avg_rougeL_f1(gold_answers, generated_answers)
    compute_avg_bertscore_f1(list(gold_answers), list(generated_answers))
    compute_avg_bartscore(list(gold_answers), list(generated_answers))
    
    elapsed = time.time() - start_time
    print(f"평가 작업 총 소요 시간: {format_time(elapsed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="평가용 데이터 입력")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_NAME_TO_API_ID.keys()))
    parser.add_argument("--shot", type=str, required=True, choices=["0", "1", "3", "6"])
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    file_path = f"data/eval/output_{args.model_name}_{args.shot}.json"
    if args.use_raw_format:
        file_path = f"data/eval/output_{args.model_name}_{args.shot}_raw.json"
    
    evaluate_answer(file_path)
    