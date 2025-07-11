import json
import torch
import time
from colorama import Fore, Style
import argparse
from tqdm import tqdm
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import (
    MODEL_MAPPING, load_environment, load_prompt, setup_logging, save_json
)

from dotenv import load_dotenv
import openai
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def load_data(file_path):
    questions = []
    pred_answers = []
    ref_answers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = data[:10]
        for item in data:
            questions.append(item.get("preprocessed_question", ""))
            pred_answers.append(item.get("generated_answer", ""))
            ref_answers.append(item.get("preprocessed_answer", ""))
    return questions, pred_answers, ref_answers

class EV2REvaluator:
    def __init__(self, judge_model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MAX_RETRIES = 10
        self.SEED = 42
        self.TEMPERATURE = 0
        self.MAX_TOKENS = 2048
        
        self.judge_model_name = MODEL_MAPPING[judge_model_name]
        
        
        env = load_environment()
        self.prompt = load_prompt(env["user_ev2r_prompt_path"])
        
        
    def query_gpt_api(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model=self.judge_model_name,
            messages=messages,
            max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE,
            seed=self.SEED,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ev2r_score",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "facts in predicted answer": {"type": "string"},
                            "fact check predicted answer": {"type": "string"},
                            "facts count predicted answer": {"type": "number"},
                            "support predicted answer": {"type": "number"},
                            "facts in reference answer": {"type": "string"},
                            "fact check reference answer": {"type": "string"},
                            "facts count reference answer": {"type": "number"},
                            "support reference answer": {"type": "number"},
                        },
                        "required": ["facts in predicted answer", "fact check predicted answer", "facts count predicted answer", "support predicted answer", "facts in reference answer", "fact check reference answer", "facts count reference answer", "support reference answer"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        json_string = response.choices[0].message.content.strip()
        try:
            parsed_data = json.loads(json_string)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print(f"원본 LLM 출력 (부분): \n{json_string[:500]}...")
            return None
    
        
    def prompt_api_model(self, questions, pred_answers, ref_answers):
        responses = []
        
        for _, (question, pred_answer, ref_answer) in tqdm(enumerate(zip(questions, pred_answers, ref_answers)), 
                                                           total=len(questions), desc="Ev2R Score Evaluation"):
            prompt = self.prompt.format(question=question, ref_answer=ref_answer, pred_answer=pred_answer)
            response = self.query_gpt_api(prompt)
            responses.append(response)
            save_json(responses, "./response_results.json")
            
            # attempt = 0
            # while attempt < self.MAX_RETRIES:
            #     try:
            #         response = self.query_gpt_api(prompt)
            #         if response:
            #             responses.append(response)
            #             save_json(responses, "./response_results.json")
            #             break
            #         else:
            #             raise Exception("API 응답이 유효하지 않습니다.")
            #     except:
            #         attempt += 1
            #         wait_time = 10 ** attempt
            #         print(f"Request timed out. Retrying in {wait_time} seconds...")
            #         time.sleep(wait_time)
                    
        return responses
    
    def calculate_f1_score(self, responses):
        f1_scores = []
        for response in responses:
            precision = response["support predicted answer"] / response["facts count predicted answer"] if response["facts count predicted answer"] > 0 else 0
            recall = response["support reference answer"] / response["facts count reference answer"] if response["facts count reference answer"] > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            f1_scores.append(f1_score)
        return sum(f1_scores) / len(f1_scores)
        
    def evaluate(self, questions, pred_answers, ref_answers):
        responses = self.prompt_api_model(questions, pred_answers, ref_answers)
        return self.calculate_f1_score(responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV2R Score Evaluator")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gpt-4o-mini")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--answer_type", choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_dpo_model", action="store_true")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--batch_size", type=int, default=500)
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    
    if args.use_finetuned_model:
        output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}.json")
    elif args.use_dpo_model:
        output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}_DPO.json")
    else:
        output_path = os.path.join(env["generated_answers_dir"],
                                   f"output_{args.model_name}_{args.shot}_{args.input_format}.json")
    
    # pred_answers, ref_answers
    questions, pred_answers, ref_answers = load_data(output_path)
    
    # EV2R_f1_score
    EV2R_scorer = EV2REvaluator(args.judge_model_name)
    f1_score = EV2R_scorer.evaluate(questions, pred_answers, ref_answers)
    print(f"Ev2R F1 Score: {Fore.RED}{f1_score}{Style.RESET_ALL}")
