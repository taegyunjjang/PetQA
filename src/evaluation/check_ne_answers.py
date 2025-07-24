# Absolute Grading: Outputs score of 1 to 5
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT
from colorama import Fore, Style
import time
import pandas as pd
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import (
    format_time, setup_logging,
    load_environment, load_prompt
)

from rubric import COHERENCE_RUBRIC, HELPFULNESS_RUBRIC, HARMLESSNESS_RUBRIC


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

def prometheus_eval_pairwise(judge, instructions, chosen_responses, rejected_responses,
                              rubric_name, rubric_template):
    # 채택된 답변 평가
    _, chosen_scores = judge.absolute_grade(
        instructions=instructions,
        responses=chosen_responses,
        rubric=rubric_template,
    )
    
    # 거절된 답변 평가
    _, rejected_scores = judge.absolute_grade(
        instructions=instructions,
        responses=rejected_responses,
        rubric=rubric_template,
    )
    
    NE = sum(chosen_scores) / len(chosen_scores)
    E = sum(rejected_scores) / len(rejected_scores)
    
    print("--------------------------------")
    print(f"{Fore.RED}{rubric_name.upper()}{Style.RESET_ALL}")
    print(f"NE: {NE}")
    print(f"E: {E}")
    print()

if __name__ == "__main__":
    env = load_environment()
    logger = setup_logging()
    
    file_path = "/home/work/factchecking/PetQA/data/interim/ne_chosen_e_rejected.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    start_time = time.time()
    
    
    prometheus_model = VLLM(
        model="Unbabel/M-Prometheus-7B",
        tensor_parallel_size=2
    )
    judge = PrometheusEval(model=prometheus_model, absolute_grade_template=ABSOLUTE_PROMPT)

    system_prompt = load_prompt(env["system_prometheus_path"])
    
    
    rubrics_to_evaluate = {
        "coherence": COHERENCE_RUBRIC,
        "helpfulness": HELPFULNESS_RUBRIC,
        "harmlessness": HARMLESSNESS_RUBRIC
    }
    
    chosen_responses = [item["chosen"]["preprocessed_answer"] for item in data]
    rejected_responses = [item["rejected"]["preprocessed_answer"] for item in data]
    instructions = [system_prompt.format(question=item["preprocessed_question"]) for item in data]

    for rubric_name, rubric_template in rubrics_to_evaluate.items():
        prometheus_eval_pairwise(
            judge,
            instructions=instructions,
            chosen_responses=chosen_responses,
            rejected_responses=rejected_responses,
            rubric_name=rubric_name,
            rubric_template=rubric_template
        )
        
