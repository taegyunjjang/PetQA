from colorama import Fore, Back, Style
import logging
import time
import pandas as pd
import argparse


MODEL_MAPPING = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
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

def compute_avg_rougeL(gold_answers, generated_answers):
    from rouge import Rouge
    rouge = Rouge()
    scores = []
    for g, p in zip(gold_answers, generated_answers):
        if p == "":
            scores.append(0)
        else:
            if len(g.split()) > 1000 or len(p.split()) > 1000:
                print(f"Gold or Predicted answer too long...{g}")
                scores.append(0)
            scores.append(rouge.get_scores(g, p)[0]['rouge-l']['f'])
    
    avg_rougeL_f1 = sum(scores) / len(scores)
    print(f"Avg ROUGE-L F1: {Fore.RED}{avg_rougeL_f1:.3f}{Style.RESET_ALL}")
    
def compute_avg_bertscore(gold_answers, generated_answers):
    from bert_score import score
    valid_pairs = [(g, p) for g, p in zip(gold_answers, generated_answers) if p != ""]
        
    valid_gold, valid_gen = zip(*valid_pairs)
    _, _, F1 = score(
        valid_gen,
        valid_gold,
        device="cuda:1",
        lang="ko"  # 다국어 모델: "bert-base-multilingual-cased"
    )
    
    empty_count = sum(1 for p in generated_answers if p == "")
    total_scores = F1.tolist() + [0] * empty_count
    
    avg_bertscore_f1 = sum(total_scores) / len(generated_answers)
    print(f"Avg BERTScore F1: {Fore.RED}{avg_bertscore_f1:.3f}{Style.RESET_ALL}")
    
def load_answer(file_path):
    df = pd.read_json(file_path)
    gold_answers = df['preprocessed_answer']
    generated_answers = df['generated_answer']
    return gold_answers, generated_answers

def evaluate_answer(file_path):
    print(f"평가 파일: {file_path}")
    start_time = time.time()
    
    gold_answers, generated_answers = load_answer(file_path)
    print(f"평가 개수: {len(gold_answers)}건")
    
    compute_avg_rougeL(gold_answers, generated_answers)
    compute_avg_bertscore(list(gold_answers), list(generated_answers))
    
    elapsed = time.time() - start_time
    print(f"평가 작업 총 소요 시간: {format_time(elapsed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="평가용 데이터 입력")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--shot", type=str, required=True, choices=["0", "1", "3", "6"])
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    ft = "_petqa" if args.use_finetuned_model else ""
    suffix = "_raw" if args.use_raw_format else ""
    file_path = f"data/eval/output_{args.model_name}{ft}_{args.shot}{suffix}.json"
    
    evaluate_answer(file_path)
    