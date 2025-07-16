import json
import torch
from colorama import Fore, Style
import argparse
from tqdm import tqdm
import os
os.environ['VLLM_USE_V1'] = '0'  # logit processor 사용 목적
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import (
    MODEL_MAPPING, save_json, setup_logging,
    load_environment, load_prompt, load_results
)

from dotenv import load_dotenv
import openai
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from etc.blocker_numpy import blocker
from pydantic import BaseModel

def load_data(file_path):
    q_ids = []
    a_ids = []
    answer_types = []
    animal_types = []
    questions = []
    pred_answers = []
    ref_answers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            q_ids.append(item.get("q_id", ""))
            a_ids.append(item.get("a_id", ""))
            answer_types.append(item.get("answer_type", ""))
            animal_types.append(item.get("animal_type", ""))
            questions.append(item.get("preprocessed_question", ""))
            pred_answers.append(item.get("generated_answer", ""))
            ref_answers.append(item.get("preprocessed_answer", ""))
    return q_ids, a_ids, answer_types, animal_types, questions, pred_answers, ref_answers

class EV2RScore(BaseModel):
    facts_in_predicted_answer: str
    fact_check_predicted_answer: str
    facts_count_predicted_answer: int
    support_predicted_answer: int
    facts_in_reference_answer: str
    fact_check_reference_answer: str
    facts_count_reference_answer: int
    support_reference_answer: int

class EV2REvaluator:
    def __init__(self, judge_model_name, output_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MAX_RETRIES = 10
        self.SEED = 42
        self.TEMPERATURE = 0
        self.MAX_TOKENS = 2048
        self.judge_model_name = judge_model_name.split("-")[0]
        self.judge_model = MODEL_MAPPING[judge_model_name]
        self.output_path = output_path
        
        self.results, self.start_idx = load_results(output_path)
        
        if self.judge_model_name.startswith("qwen") or self.judge_model_name.startswith("exaone"):
            self.model_type = "open" 
            self.batch_size = 20
            self.llm = LLM(
                model=self.judge_model,
                tensor_parallel_size=2,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                max_model_len = 5000,
                gpu_memory_utilization=0.99
            )
            
            self.tokenizer = self.llm.get_tokenizer()
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        else:
            self.model_type = "closed"
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
        env = load_environment()
        self.base_prompt = load_prompt(env["user_ev2r_prompt_path"])
        
        
    def query_openai_api(self, prompt):
        fallback_output = {
            "facts_in_predicted_answer": "",
            "fact_check_predicted_answer": "",
            "facts_count_predicted_answer": 0,
            "support_predicted_answer": 0,
            "facts_in_reference_answer": "",
            "fact_check_reference_answer": "",
            "facts_count_reference_answer": 0,
            "support_reference_answer": 0
        }
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        for attempt in range(self.MAX_RETRIES):
            response = self.client.chat.completions.create(
                model=self.judge_model,
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
                                "facts_in_predicted_answer": {"type": "string"},
                                "fact_check_predicted_answer": {"type": "string"},
                                "facts_count_predicted_answer": {"type": "number"},
                                "support_predicted_answer": {"type": "number"},
                                "facts_in_reference_answer": {"type": "string"},
                                "fact_check_reference_answer": {"type": "string"},
                                "facts_count_reference_answer": {"type": "number"},
                                "support_reference_answer": {"type": "number"},
                            },
                            "required": [
                                "facts_in_predicted_answer", 
                                "fact_check_predicted_answer", 
                                "facts_count_predicted_answer", 
                                "support_predicted_answer", 
                                "facts_in_reference_answer", 
                                "fact_check_reference_answer", 
                                "facts_count_reference_answer", 
                                "support_reference_answer"
                            ],
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
                print(f"[시도 {attempt+1}] JSON 파싱 오류: {e}")
                print(f"[시도 {attempt+1}] 원본(json_string[:100]): \n{json_string[:100]}")
                print(f"[시도 {attempt+1}] 원본(json_string[-100:]): \n{json_string[-100:]}")
        
        print(f"모든 재시도 실패, 기본 출력 반환")
        return fallback_output
    
    def query_vllm_api(self, batch_prompts):
        responses = []
        
        fallback_output = {
            "facts_in_predicted_answer": "",
            "fact_check_predicted_answer": "",
            "facts_count_predicted_answer": 0,
            "support_predicted_answer": 0,
            "facts_in_reference_answer": "",
            "fact_check_reference_answer": "",
            "facts_count_reference_answer": 0,
            "support_reference_answer": 0
        }
        
        # schema = EV2RScore.model_json_schema()
        schema = {
            "type": "object",
            "properties": {
                "facts_in_predicted_answer": {"type": "string"},
                "fact_check_predicted_answer": {"type": "string"},
                "facts_count_predicted_answer": {"type": "number"},
                "support_predicted_answer": {"type": "number"},
                "facts_in_reference_answer": {"type": "string"},
                "fact_check_reference_answer": {"type": "string"},
                "facts_count_reference_answer": {"type": "number"},
                "support_reference_answer": {"type": "number"},
            },
            "required": [
                "facts_in_predicted_answer", 
                "fact_check_predicted_answer", 
                "facts_count_predicted_answer", 
                "support_predicted_answer", 
                "facts_in_reference_answer", 
                "fact_check_reference_answer", 
                "facts_count_reference_answer", 
                "support_reference_answer"
            ],
            "additionalProperties": False # 스키마에 정의되지 않은 추가 속성 허용 안 함
        }
        guided = GuidedDecodingParams(
            json=schema,
            disable_any_whitespace=True,
        )
        
        sampling_params = SamplingParams(
            temperature=self.TEMPERATURE,
            seed = self.SEED,
            max_tokens=self.MAX_TOKENS,
            guided_decoding=guided,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None
        )
            # 중국어 토큰 생성 방지
        def _logits_processor(input_ids, logits):
            return blocker(self.tokenizer, input_ids, logits)
        sampling_params.logits_processors=[_logits_processor]
        
        outputs = self.llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        for output in outputs:
            json_string = output.outputs[0].text.strip()
            try:
                parsed_data = json.loads(json_string)
                responses.append(parsed_data)
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print(f"원본(json_string[:100]): \n{json_string[:100]}")
                print(f"원본(json_string[-100:]): \n{json_string[-100:]}")
                responses.append(fallback_output)
        
        return responses
        
    def get_prompts(self, questions, pred_answers, ref_answers):
        prompts = []
        
        for _, (question, pred_answer, ref_answer) in enumerate(zip(questions, pred_answers, ref_answers)):
            user_prompt = self.base_prompt.format(question=question, ref_answer=ref_answer, pred_answer=pred_answer)
            messages = [
                {"role": "user", "content": user_prompt}
            ]
        
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        return prompts
        
    def prompt_api_model(self, q_ids, a_ids, answer_types, animal_types, questions, pred_answers, ref_answers):
        responses = []
        if self.model_type == "open":
            prompts = self.get_prompts(questions, pred_answers, ref_answers)
            num_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
            for i in tqdm(range(0, len(prompts), self.batch_size), total=num_batches, desc="Ev2R Score Evaluation"):
                batch_prompts = prompts[i : i + self.batch_size]
                batch_q_ids = q_ids[i : i + self.batch_size]
                batch_a_ids = a_ids[i : i + self.batch_size]
                batch_answer_types = answer_types[i : i + self.batch_size]
                batch_animal_types = animal_types[i : i + self.batch_size]
                
                batch_responses = self.query_vllm_api(batch_prompts)
                for j, response_data in enumerate(batch_responses):
                    combined_data = {
                        "q_id": batch_q_ids[j],
                        "a_id": batch_a_ids[j],
                        "answer_type": batch_answer_types[j],
                        "animal_type": batch_animal_types[j],
                        **response_data
                    }
                    responses.append(combined_data)
                save_json(responses, self.output_path)
                
            return responses
        else:
            for idx, (question, pred_answer, ref_answer) in tqdm(enumerate(zip(questions, pred_answers, ref_answers)), 
                                                            total=len(questions), desc="Ev2R Score Evaluation"):
                prompt = self.base_prompt.format(question=question, ref_answer=ref_answer, pred_answer=pred_answer)
                response = self.query_openai_api(prompt)
                combined_data = {
                    "q_id": q_ids[idx],
                    "a_id": a_ids[idx],
                    "answer_type": answer_types[idx],
                    "animal_type": animal_types[idx],
                    **response
                }
                responses.append(combined_data)
                save_json(responses, self.output_path)
                        
            return responses
    
    def calculate_f1_score(self, responses):
        unique_categories = [("dog", "expert"), ("dog", "nonexpert"), ("cat", "expert"), ("cat", "nonexpert")]
        for animal_type, answer_type in unique_categories:
            category = f"{animal_type}-{answer_type}"
            print(f"Evaluating: {category}")
            
            filtered_responses = [response for response in responses 
                                  if response["animal_type"] == animal_type and response["answer_type"] == answer_type]
            
            category_f1_scores = []
            for response in filtered_responses:
                precision = response["support_predicted_answer"] / response["facts_count_predicted_answer"] if response["facts_count_predicted_answer"] > 0 else 0
                recall = response["support_reference_answer"] / response["facts_count_reference_answer"] if response["facts_count_reference_answer"] > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                category_f1_scores.append(f1_score)
            avg_f1_score = (sum(category_f1_scores) / len(category_f1_scores))
            print(f"Average F1 Score: {Fore.RED}{avg_f1_score}{Style.RESET_ALL}")
            
    def evaluate(self, q_ids, a_ids, answer_types, animal_types, questions, pred_answers, ref_answers):
        responses = self.prompt_api_model(q_ids, a_ids, answer_types, animal_types, questions, pred_answers, ref_answers)
        self.calculate_f1_score(responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV2R Score Evaluator")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="exaone-3.5-7.8b")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--answer_type", choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_dpo_model", action="store_true")
    parser.add_argument("--judge_model_name", type=str, default="exaone-3.5-32b")
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    
    if args.use_finetuned_model:
        endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}.json"
        input_path = os.path.join(env["generated_answers_dir"], endpoint)
    elif args.use_dpo_model:
        endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.answer_type}_DPO.json"
        input_path = os.path.join(env["generated_answers_dir"], endpoint)
    else:
        endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}.json"
        input_path = os.path.join(env["generated_answers_dir"], endpoint)
    output_path = os.path.join(env["atomic_facts_dir"], endpoint)
    
    logger.info(f"INPUT PATH: {input_path}")
    
    q_ids, a_ids, answer_types, animal_types, questions, pred_answers, ref_answers = load_data(input_path)
    
    EV2R_scorer = EV2REvaluator(args.judge_model_name, output_path)
    EV2R_scorer.evaluate(q_ids, a_ids, answer_types, animal_types, questions, pred_answers, ref_answers)
    
    
