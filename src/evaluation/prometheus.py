# Absolute Grading: Outputs score of 1 to 5
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT
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

from utils.utils import (
    MODEL_MAPPING, format_time, setup_logging,
    load_environment, save_json, load_prompt
)

from rubric import COMPLETENESS_RUBRIC, COHERENCE_RUBRIC, HELPFULNESS_RUBRIC, HARMLESSNESS_RUBRIC


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

def prometheus_eval(judge, instructions, responses, reference_answers,
                    rubric_name, rubric_template):
    _, scores = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        rubric=rubric_template,
        reference_answers=reference_answers
    )
    avg_score = sum(scores) / len(scores)
    print(f"Avg Prometheus score for {rubric_name}: {Fore.RED}{avg_score:.3f}{Style.RESET_ALL}")
    return avg_score

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
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}"
    elif args.use_dpo_model:
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}_DPO"
    else:
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}.json"
    file_path = os.path.join(env["generated_answers_dir"], f"output_{endpoint}.json")
    
    start_time = time.time()
    df = load_data(file_path)
    logger.info(f"TOTAL SAMPLE COUNT: {len(df):,}")
    print("--------------------------------")
    
    prometheus_model = VLLM(
        model="Unbabel/M-Prometheus-7B",
        tensor_parallel_size=2
    )
    judge = PrometheusEval(model=prometheus_model, absolute_grade_template=ABSOLUTE_PROMPT)

    unique_categories = [("dog", "expert"), ("dog", "nonexpert"), ("cat", "expert"), ("cat", "nonexpert")]
    
    output_path = "./prometheus_results.json"
    results = load_results(output_path)
    system_prompt = load_prompt(env["system_prometheus_path"])
    
    endpoint_data = {"id": endpoint}
    
    rubrics_to_evaluate = {
        # "completeness": COMPLETENESS_RUBRIC,
        # "coherence": COHERENCE_RUBRIC,
        "helpfulness": HELPFULNESS_RUBRIC,
        "harmlessness": HARMLESSNESS_RUBRIC
    }
    
    for animal_type, answer_type in unique_categories:
        filtered_df = df[(df['animal_type'] == animal_type) & (df['answer_type'] == answer_type)]
        
        category = f"{animal_type}-{answer_type}"
        print(f"Evaluating: {category} (Sample count: {len(filtered_df):,})")
        
        instructions = [system_prompt.format(question=question) 
                        for question in filtered_df['preprocessed_question'].tolist()]
        responses = filtered_df['generated_answer'].tolist()
        reference_answers = filtered_df['preprocessed_answer'].tolist()
        
        category_scores = {}
        for rubric_name, rubric_template in rubrics_to_evaluate.items():
            avg_score = prometheus_eval(judge, instructions, responses, reference_answers,
                                        rubric_name, rubric_template)
            category_scores[rubric_name] = avg_score
        
        endpoint_data[category] = category_scores
        
    results.append(endpoint_data)
    save_json(results, output_path)

    
    elapsed = time.time() - start_time
    print(f"TOTAL TIME: {format_time(elapsed)}")
    print("--------------------------------")