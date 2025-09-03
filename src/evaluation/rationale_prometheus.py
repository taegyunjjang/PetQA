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

from rubric import *


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

def prometheus_eval(judge, instructions, responses, reference_answers, rubric_template):
    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        rubric=rubric_template,
        reference_answers=reference_answers
    )
    return feedbacks, scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="평가용 데이터 입력")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="qwen-2.5-7b")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--fewshot_type", type=str, choices=["baseline", "bert", "llm", "oracle"], default="baseline")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--training_type", type=str, choices=["E", "NE", "ALL", "ORACLE"], default="ALL")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_dpo_model", action="store_true")
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"SHOT: {args.shot}")
    if args.shot != "0":
        logger.info(f"FEWSHOT TYPE: {args.fewshot_type}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    
    if args.use_finetuned_model:
        logger.info(f"USE FINETUNED MODEL")
        logger.info(f"TRAINING TYPE: {args.training_type}")
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}"
    elif args.use_dpo_model:
        logger.info(f"USE DPO MODEL")
        logger.info(f"TRAINING TYPE: {args.training_type}")
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}_DPO"
    else:
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}"
        if args.shot != "0":
            endpoint = f"{endpoint}_{args.fewshot_type}"
    file_path = os.path.join(env["generated_answers_dir"], f"output_{endpoint}.json")
    
    model_name = "Unbabel/M-Prometheus-7B"
    logger.info(f"MODEL NAME: {model_name}")
    
    start_time = time.time()
    df = load_data(file_path)
    logger.info(f"TOTAL SAMPLE COUNT: {len(df):,}")
    print("--------------------------------")
    
    prometheus_model = VLLM(
        model=model_name,
        tensor_parallel_size=2
    )
    judge = PrometheusEval(model=prometheus_model, absolute_grade_template=ABSOLUTE_PROMPT)

    question_types = {
        "expert": "E", 
        "nonexpert": "NE"
    }
    
    total_count = 0
    
    output_path = "./rationale_gemma_0_3.json"
    results = load_results(output_path)
    system_prompt = load_prompt(env["system_prometheus_path"])
    
    endpoint_data = {"id": endpoint}
    
    rubrics_to_evaluate = {
        "coherence": COHERENCE_RUBRIC,
        "helpfulness": HELPFULNESS_RUBRIC,
        "harmlessness": HARMLESSNESS_RUBRIC
    }
    
    for question_type, type_name in question_types.items():
        all_scores = {}
        filtered_df = df[df['answer_type'] == question_type]
        
        sample_size = 10
        filtered_df = filtered_df.head(sample_size)
        
        questions = filtered_df['preprocessed_question'].tolist()
        instructions = [system_prompt.format(question=question) for question in questions]
        responses = filtered_df['generated_answer'].tolist()
        reference_answers = filtered_df['preprocessed_answer'].tolist()
        
        for rubric_name, rubric_template in rubrics_to_evaluate.items():
            feedbacks, scores = prometheus_eval(judge, instructions, responses, reference_answers, rubric_template)
            
            if rubric_name not in all_scores:
                all_scores[rubric_name] = []
            
            for i in range(sample_size):
                all_scores[rubric_name].append({
                    "question": questions[i],
                    "reference_answer": reference_answers[i],
                    "generated_answer": responses[i],
                    "score": scores[i],
                    "feedback": feedbacks[i]
                })
        
        endpoint_data[type_name] = all_scores
 
    results.append(endpoint_data)
    save_json(results, output_path)

    elapsed = time.time() - start_time
    print(f"TOTAL TIME: {format_time(elapsed)}")
    print("--------------------------------")