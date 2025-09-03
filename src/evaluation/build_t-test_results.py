from colorama import Fore, Style
import time
import pandas as pd
import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    df = pd.read_json(file_path)
    return df

def load_results(output_path):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []
    return results

def compute_rougeL(generated_answers, gold_answers):
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
    return scores
    
def compute_bertscore(generated_answers, gold_answers):
    from bert_score import score
    valid_pairs = [(g, p) for g, p in zip(generated_answers, gold_answers)]
        
    valid_gen, valid_gold = zip(*valid_pairs)
    _, _, F1 = score(
        valid_gen,
        valid_gold, 
        lang="ko"  # 다국어 모델: "bert-base-multilingual-cased"
    )
    
    total_scores = F1.tolist()
    return total_scores

def compute_factuality(data, answer_type):
    f1_scores = []
    for item in data:
        if item["answer_type"] == answer_type:
            precision = item["support_predicted_answer"] / item["facts_count_predicted_answer"] if item["facts_count_predicted_answer"] > 0 else 0
            recall = item["support_reference_answer"] / item["facts_count_reference_answer"] if item["facts_count_reference_answer"] > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            f1_scores.append(f1_score)
    return f1_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired t-test results builder")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gpt-4o-mini")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    
    endpoint = f"{args.model_name}_0_{args.input_format}"
    file_path = os.path.join(env["generated_answers_dir"], f"output_{endpoint}.json")
    
    start_time = time.time()
    df = load_data(file_path)
    logger.info(f"TOTAL SAMPLE COUNT: {len(df):,}")
    print("--------------------------------")
    
    output_path = "./t-test_results.json"
    results = load_results(output_path)
    
    result = {"id": endpoint}
    
    atomic_facts_path = os.path.join(env["atomic_facts_dir"], "output_" + endpoint + ".json")
    with open(atomic_facts_path, "r", encoding="utf-8") as f:
        atomic_facts_data = json.load(f)
    
    answer_types = ["expert", "nonexpert"]
    for answer_type in answer_types:
        filtered_df = df[(df['answer_type'] == answer_type)]
        print(f"Evaluating: {answer_type} (Sample count: {len(filtered_df):,})")
        
        generated_answers = filtered_df['generated_answer']
        gold_answers = filtered_df['preprocessed_answer']

        rougeL_f1_scores = compute_rougeL(generated_answers, gold_answers)
        bertscore_f1_scores = compute_bertscore(list(generated_answers), list(gold_answers))
        factuality_f1_scores = compute_factuality(atomic_facts_data, answer_type)
        
        result[answer_type] = {
            "ROUGE": rougeL_f1_scores,
            "BERTScore": bertscore_f1_scores,
            "Factuality": factuality_f1_scores
        }
    results.append(result)
    save_json(results, output_path)
    
    elapsed = time.time() - start_time
    print(f"TOTAL TIME: {format_time(elapsed)}")
    print("--------------------------------")