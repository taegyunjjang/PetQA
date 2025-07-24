from colorama import Fore, Style
import time
import pandas as pd
import argparse
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.setrecursionlimit(10000)

from utils.utils import (
    MODEL_MAPPING, format_time, setup_logging,
    load_environment, save_json
)


def load_data(file_path):
    try:
        df = pd.read_json(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        sys.exit(1)
    return df

def load_results(output_path):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []
    return results

def compute_avg_rougeL(generated_answers, gold_answers):
    from rouge import Rouge
    from konlpy.tag import Okt
    okt = Okt()
    rouge = Rouge()
    
    # 형태소 단위로 계산
    generated_answers_morphs = [" ".join(okt.morphs(g)) for g in generated_answers]
    gold_answers_morphs = [" ".join(okt.morphs(p)) for p in gold_answers]

    scores = []
    for g, p in zip(generated_answers_morphs, gold_answers_morphs):
        scores.append(rouge.get_scores(g, p)[0]['rouge-l']['f'])
    
    avg_rougeL_f1 = sum(scores) / len(scores)
    print(f"Avg ROUGE-L F1: {Fore.RED}{avg_rougeL_f1:.3f}{Style.RESET_ALL}")
    return avg_rougeL_f1
    
def compute_avg_bertscore(generated_answers, gold_answers):
    from bert_score import score
    valid_pairs = [(g, p) for g, p in zip(generated_answers, gold_answers)]
        
    valid_gen, valid_gold = zip(*valid_pairs)
    _, _, F1 = score(
        valid_gen,
        valid_gold, 
        lang="ko"  # 다국어 모델: "bert-base-multilingual-cased"
    )
    
    total_scores = F1.tolist()
    avg_bertscore_f1 = sum(total_scores) / len(total_scores)
    print(f"Avg BERTScore F1: {Fore.RED}{avg_bertscore_f1:.3f}{Style.RESET_ALL}")
    return avg_bertscore_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="평가용 데이터 입력")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="exaone-3.5-7.8b")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--answer_type", type=str, choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_dpo_model", action="store_true")
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"SHOT: {args.shot}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"ANSWER TYPE: {args.answer_type}")
    logger.info(f"USE FINETUNED MODEL: {args.use_finetuned_model}")
    logger.info(f"USE DPO MODEL: {args.use_dpo_model}")

    if args.use_finetuned_model:
        file_path = os.path.join(env["generated_answers_dir"],
                                 f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}.json")
    elif args.use_dpo_model:
        file_path = os.path.join(env["generated_answers_dir"],
                                 f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}_DPO.json")
    else:
        file_path = os.path.join(env["generated_answers_dir"],
                                 f"output_{args.model_name}_{args.shot}_{args.input_format}.json")
    
    start_time = time.time()
    df = load_data(file_path)
    logger.info(f"TOTAL SAMPLE COUNT: {len(df):,}")
    print("--------------------------------")
    
    output_path = "./rouge_bertscore_results.json"
    results = load_results(output_path)
    
    if args.use_finetuned_model:
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}"
    else:
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}"
    endpoint_data = {"id": endpoint}
    
    unique_categories = [("dog", "expert"), ("dog", "nonexpert"), ("cat", "expert"), ("cat", "nonexpert")]
    for animal_type, answer_type in unique_categories:
        filtered_df = df[(df['animal_type'] == animal_type) & (df['answer_type'] == answer_type)]
            
        category = f"{animal_type}-{answer_type}"
        print(f"Evaluating: {category} (Sample count: {len(filtered_df):,})")
        
        generated_answers = filtered_df['generated_answer']
        gold_answers = filtered_df['preprocessed_answer']

        avg_rougeL_f1 = compute_avg_rougeL(generated_answers, gold_answers)
        avg_bertscore_f1 = compute_avg_bertscore(list(generated_answers), list(gold_answers))
        
        endpoint_data[category] = {
            "rougeL_f1": avg_rougeL_f1,
            "bertscore_f1": avg_bertscore_f1
        }
    results.append(endpoint_data)
    save_json(results, output_path)
    
    elapsed = time.time() - start_time
    print(f"TOTAL TIME: {format_time(elapsed)}")
    print("--------------------------------")