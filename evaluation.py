from colorama import Fore, Style
import time
import pandas as pd
import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.setrecursionlimit(10000)

from utils.utils import (
    MODEL_MAPPING, format_time, setup_logging,
    load_environment
)


def load_data(file_path):
    try:
        df = pd.read_json(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        sys.exit(1)
    return df

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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="평가용 데이터 입력")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gpt-4o-mini")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    args = parser.parse_args()
    
    env = load_environment()
    file_path = os.path.join(env["generated_answers_dir"],
                               f"output_{args.model_name}_{args.shot}_{args.input_format}.json")
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"SHOT: {args.shot}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"FILE PATH: {file_path}")
    
    start_time = time.time()
    df = load_data(file_path)
    logger.info(f"TOTAL SAMPLE COUNT: {len(df):,}")
    print("--------------------------------")
    
    unique_categories = [("dog", "expert"), ("dog", "nonexpert"), ("cat", "expert"), ("cat", "nonexpert")]
    for animal_type, answer_type in unique_categories:
        filtered_df = df[(df['animal_type'] == animal_type) & (df['answer_type'] == answer_type)]
            
        print(f"Evaluating: {animal_type}-{answer_type} (Sample count: {len(filtered_df):,})")
        
        generated_answers = filtered_df['generated_answer']
        gold_answers = filtered_df['preprocessed_answer']

        compute_avg_rougeL(generated_answers, gold_answers)
        compute_avg_bertscore(list(generated_answers), list(gold_answers))
    
    elapsed = time.time() - start_time
    print(f"TOTAL TIME: {format_time(elapsed)}")
    print("--------------------------------")