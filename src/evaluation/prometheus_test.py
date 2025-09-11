from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT
from colorama import Fore, Style
import time
import pandas as pd
import argparse
import json
from tqdm import tqdm
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import (
    MODEL_MAPPING, format_time, setup_logging,
    load_environment, load_json, save_json, load_prompt
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

def run_evaluation(data_to_process, output_path, system_prompt, rubrics_to_evaluate, judge):
    q_ids = [item['q_id'] for item in data_to_process]
    a_ids = [item['a_id'] for item in data_to_process]
    answer_types = [item['answer_type'] for item in data_to_process]
    questions = [item['preprocessed_question'] for item in data_to_process]
    instructions = [system_prompt.format(question=item['preprocessed_question']) for item in data_to_process]
    responses = [item['generated_answer'] for item in data_to_process]
    reference_answers = [item['summarized_answer'] for item in data_to_process]
    
    results = []
    for i in range(len(data_to_process)):
        results.append({
            "q_id": q_ids[i],
            "a_id": a_ids[i],
            "answer_type": answer_types[i],
            "question": questions[i],
            "reference_answer": reference_answers[i],
            "generated_answer": responses[i],
        })
    
    for rubric_name, rubric_template in rubrics_to_evaluate.items():
        feedbacks, scores = prometheus_eval(judge, instructions, responses, reference_answers, rubric_template)
        
        for i in range(len(data_to_process)):
            results[i][f"{rubric_name}_score"] = scores[i]
            results[i][f"{rubric_name}_feedback"] = feedbacks[i]
    
    save_json(results, output_path)
    return results

def print_scores(results, rubrics_to_evaluate):
    question_types = {
        "expert": "E", 
        "nonexpert": "NE"
    }
    for question_type, type_name in question_types.items():
        print(f"====== Evaluating {type_name} ======")
        filtered_results = [result for result in results if result["answer_type"] == question_type]
        for rubric_name, _ in rubrics_to_evaluate.items():
            scores = [result[f"{rubric_name}_score"] for result in filtered_results]
            avg_score = sum(scores) / len(scores)
            print(f"  {rubric_name}: {Fore.RED}{avg_score:.3f}{Style.RESET_ALL}")
    
    print(f"====== Evaluating ALL ======")
    for rubric_name in rubrics_to_evaluate.keys():
        all_scores = [result[f"{rubric_name}_score"] for result in results]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"  {rubric_name}: {Fore.RED}{avg_score:.3f}{Style.RESET_ALL}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="평가용 데이터 입력")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="qwen-2.5-7b")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--fewshot_type", type=str, choices=["baseline", "bert", "llm", "oracle"], default="oracle")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--training_type", type=str, choices=["E", "NE", "ALL", "ORACLE"], default="ALL")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_dpo_model", action="store_true")
    parser.add_argument("--use_rag_model", action="store_true")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--sample_size", type=int, default=500)
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
    elif args.use_rag_model:
        logger.info(f"USE RAG MODEL")
        logger.info(f"TOP_K: {args.top_k}")
        endpoint = f"{args.model_name}_{args.input_format}_RAG_{args.top_k}"
    else:
        endpoint = f"{args.model_name}_{args.shot}_{args.input_format}"
        if args.shot != "0":
            endpoint = f"{endpoint}_{args.fewshot_type}"
            
    endpoint = endpoint + "_summarization"
    input_path = os.path.join(env["generated_answers_dir"], f"output_{endpoint}.json")
    
    logger.info(f"INPUT PATH: {input_path}")
    
    JUDGE_MODEL = "Unbabel/M-Prometheus-7B"
    logger.info(f"JUDGE MODEL: {JUDGE_MODEL}")
    
    start_time = time.time()
    
    data = load_json(input_path)
    data_to_process = data
    
    logger.info(f"TOTAL SAMPLE COUNT: {len(data_to_process):,}")
    print("--------------------------------")
    
    prometheus_model = VLLM(
        model=JUDGE_MODEL,
        tensor_parallel_size=2,
        seed=42,
        gpu_memory_utilization=0.96
    )
    judge = PrometheusEval(model=prometheus_model, absolute_grade_template=ABSOLUTE_PROMPT)
    output_path = f"./prometheus_results/{endpoint}.json"
    system_prompt = load_prompt(env["system_prometheus_path"])
    rubrics_to_evaluate = {
        # "coherence": COHERENCE_RUBRIC,
        "helpfulness": HELPFULNESS_RUBRIC,
        "harmlessness": HARMLESSNESS_RUBRIC
    }
    results = run_evaluation(data_to_process, output_path, system_prompt, rubrics_to_evaluate, judge)
    
    print("--------------------------------")
    print(f"ID: {endpoint}")
    print_scores(results, rubrics_to_evaluate)
    elapsed = time.time() - start_time
    print(f"TOTAL TIME: {format_time(elapsed)}")
    print("--------------------------------")