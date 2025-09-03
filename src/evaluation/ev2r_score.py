import json
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
    MODEL_MAPPING, load_json, save_json, setup_logging,
    load_environment, load_prompt, load_results
)

from dotenv import load_dotenv
load_dotenv()

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class EV2REvaluator:
    def __init__(self, judge_model_name, input_path, output_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MAX_RETRIES = 10
        self.SEED = 42
        self.TEMPERATURE = 0
        self.MAX_TOKENS = 2048
        self.judge_model = MODEL_MAPPING[judge_model_name]
        self.input_path = input_path
        self.output_path = output_path
        
        self.results, self.start_idx = load_results(output_path)
        
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
        
        env = load_environment()
        self.base_prompt = load_prompt(env["user_ev2r_prompt_path"])
        
        
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
            "additionalProperties": False
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
        
    def prompt_api_model(self, data):
        data = data[self.start_idx:]
        questions = [item["preprocessed_question"] for item in data]
        pred_answers = [item["generated_answer"] for item in data]
        ref_answers = [item["summarized_answer"] for item in data]
        q_ids = [item["q_id"] for item in data]
        a_ids = [item["a_id"] for item in data]
        answer_types = [item["answer_type"] for item in data]
        animal_types = [item["animal_type"] for item in data]
        
        prompts = self.get_prompts(questions, pred_answers, ref_answers)
        num_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(prompts), self.batch_size), total=num_batches, desc="Ev2R Score Evaluation"):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_q_ids = q_ids[i : i + self.batch_size]
            batch_a_ids = a_ids[i : i + self.batch_size]
            batch_answer_types = answer_types[i : i + self.batch_size]
            batch_animal_types = animal_types[i : i + self.batch_size]
            
            batch_results = self.query_vllm_api(batch_prompts)
            for j, result in enumerate(batch_results):
                combined_data = {
                    "q_id": batch_q_ids[j],
                    "a_id": batch_a_ids[j],
                    "answer_type": batch_answer_types[j],
                    "animal_type": batch_animal_types[j],
                    **result
                }
                self.results.append(combined_data)
            save_json(self.results, self.output_path)
            
        return self.results
    
    def calculate_f1_score(self, results):
        question_types = {
            "expert": "E", 
            "nonexpert": "NE"
        }
        
        category_f1_scores = {}
        for question_type, type_name in question_types.items():
            
            filtered_results = [result for result in results if result["answer_type"] == question_type]
            
            f1_scores = []
            for result in filtered_results:
                precision = result["support_predicted_answer"] / result["facts_count_predicted_answer"] if result["facts_count_predicted_answer"] > 0 else 0
                recall = result["support_reference_answer"] / result["facts_count_reference_answer"] if result["facts_count_reference_answer"] > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                f1_scores.append(f1_score)

            category_f1_scores[type_name] = {
                "F1": sum(f1_scores) / len(f1_scores),
                "count": len(filtered_results)
            }
        
        category_f1_scores["ALL"] = {
            "F1": sum(category_f1_scores[type_name]["F1"] * category_f1_scores[type_name]["count"] for type_name in category_f1_scores) / sum(category_f1_scores[type_name]["count"] for type_name in category_f1_scores),
            "count": sum(category_f1_scores[type_name]["count"] for type_name in category_f1_scores)
        }
        
        for type_name, data_info in category_f1_scores.items():
            print(f"Evaluating: {type_name}")
            print(f"Average F1 Score: {Fore.RED}{data_info['F1']:0.3f}{Style.RESET_ALL}")
            
    def evaluate(self, data):
        results = self.prompt_api_model(data)
        self.calculate_f1_score(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV2R Score Evaluator")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gemma-3-4b")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--fewshot_type", type=str, choices=["baseline", "bert", "llm", "oracle"], default="oracle")
    parser.add_argument("--training_type", choices=["E", "NE", "ALL", "ORACLE"], default="ALL")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--use_summarization", action="store_true")
    parser.add_argument("--use_dpo_model", action="store_true")
    parser.add_argument("--use_rag_model", action="store_true")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--judge_model_name", type=str, default="exaone-3.5-32b")
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    
    if args.use_finetuned_model:
        endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}.json"
    elif args.use_dpo_model:
        endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.training_type}_DPO.json"
    elif args.use_rag_model:
        endpoint = f"output_{args.model_name}_{args.input_format}_RAG_{args.top_k}.json"
    else:
        endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}.json"
        if args.shot != "0":
            endpoint = f"output_{args.model_name}_{args.shot}_{args.input_format}_{args.fewshot_type}.json"
    
    if args.use_summarization:
        endpoint = endpoint.replace(".json", "_summarization.json")
        
    input_path = os.path.join(env["generated_answers_dir"], endpoint)
    output_path = os.path.join(env["atomic_facts_dir"], endpoint)
    
    logger.info(f"INPUT PATH: {input_path}")
    logger.info(f"OUTPUT PATH: {output_path}")
    
    data = load_json(input_path)
    
    EV2R_scorer = EV2REvaluator(args.judge_model_name, input_path, output_path)
    EV2R_scorer.evaluate(data)
    
    logger.info(f"{endpoint}")
    logger.info("Ev2R Evaluation Completed")
