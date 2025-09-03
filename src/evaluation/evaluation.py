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
    print(f"Avg ROUGE: {Fore.RED}{avg_rougeL_f1:.3f}{Style.RESET_ALL}")
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
    print(f"Avg BERTScore: {Fore.RED}{avg_bertscore_f1:.3f}{Style.RESET_ALL}")
    return avg_bertscore_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="평가용 데이터 입력")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gemma-3-4b")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--fewshot_type", type=str, choices=["baseline", "bert", "llm", "oracle"], default="oracle")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--training_type", type=str, choices=["E", "NE", "ALL", "ORACLE"], default="ALL")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_summarization", action="store_true")
    parser.add_argument("--use_dpo_model", action="store_true")
    parser.add_argument("--use_rag_model", action="store_true")
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"SHOT: {args.shot}")
    if args.shot != "0":
        logger.info(f"FEWSHOT TYPE: {args.fewshot_type}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"USE SUMMARIZATION: {args.use_summarization}")
    
    if args.use_finetuned_model:
        logger.info(f"USE FINETUNED MODEL")
        logger.info(f"TRAINING TYPE: {args.training_type}")
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}"
        if args.use_summarization:
            endpoint = f"{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}_summarization"
        file_path = os.path.join(env["generated_answers_dir"],
                                 f"output_{endpoint}.json")
    elif args.use_dpo_model:
        logger.info(f"USE DPO MODEL")
        logger.info(f"TRAINING TYPE: {args.training_type}")
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}_DPO"
        file_path = os.path.join(env["generated_answers_dir"],
                                 f"output_{endpoint}.json")
    else:
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}"
        if args.use_summarization:
            endpoint = f"{endpoint}_summarization"
        if args.shot != "0":
            endpoint = f"{endpoint}_{args.fewshot_type}"
        if args.use_rag_model:
            logger.info(f"USE RAG MODEL")
            logger.info(f"TOP_K: {args.top_k}")
            endpoint = f"{args.model_name}_{args.input_format}_RAG_{args.top_k}"
        file_path = os.path.join(env["generated_answers_dir"],
                                 f"output_{endpoint}.json")
            
    start_time = time.time()
    df = load_data(file_path)
    logger.info(f"TEST DATA SAMPLE COUNT: {len(df):,}")
    
    output_path = "./rouge_bertscore_results.json"
    results = load_results(output_path)
    
    endpoint_data = {"id": endpoint}
    
    question_types = {
        "expert": "E", 
        "nonexpert": "NE"
    }

    all_scores = {}
    total_count = 0

    for question_type, type_name in question_types.items():
        print(f"Evaluating: {type_name}")
        filtered_df = df[df['answer_type'] == question_type]

        generated_answers = filtered_df['generated_answer']
        if args.use_summarization:
            gold_answers = filtered_df['summarized_answer']
        else:
            gold_answers = filtered_df['preprocessed_answer']
        
        assert len(generated_answers) == len(gold_answers)

        avg_rougeL_f1 = compute_avg_rougeL(generated_answers, gold_answers)
        avg_bertscore_f1 = compute_avg_bertscore(list(generated_answers), list(gold_answers))

        count = len(filtered_df)
        all_scores[type_name] = {
            "ROUGE": avg_rougeL_f1,
            "BERTScore": avg_bertscore_f1,
            "count": count
        }

        endpoint_data[type_name] = {
            "ROUGE": avg_rougeL_f1,
            "BERTScore": avg_bertscore_f1
        }
        total_count += count

    print("Evaluating: ALL")

    weighted_rouge = sum(all_scores[t]["ROUGE"] * all_scores[t]["count"] for t in all_scores) / total_count
    weighted_bertscore = sum(all_scores[t]["BERTScore"] * all_scores[t]["count"] for t in all_scores) / total_count

    endpoint_data["ALL"] = {
        "ROUGE": weighted_rouge,
        "BERTScore": weighted_bertscore
    }
    
    print(f"Avg ROUGE: {Fore.RED}{weighted_rouge:.3f}{Style.RESET_ALL}")
    print(f"Avg BERTScore: {Fore.RED}{weighted_bertscore:.3f}{Style.RESET_ALL}")

    results.append(endpoint_data)
    save_json(results, output_path)
    
    elapsed = time.time() - start_time
    print(f"TOTAL TIME: {format_time(elapsed)}")
    print("--------------------------------")