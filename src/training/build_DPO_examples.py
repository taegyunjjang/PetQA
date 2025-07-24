import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # GPU 메모리 할당 효율성을 높여서 메모리 단편화로 인한 OOM 발생을 줄임
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import (
    setup_logging, load_json, save_json,
    load_prompt, load_environment
)

from collections import defaultdict
from datasets import load_dataset


def load_datasets(data_files, logger):
    logger.info("Loading datasets...")
    dataset = load_dataset("json", data_files=data_files)
    
    train_data = dataset["train"]
    validation_data = dataset["validation"]
    logger.info(f"Train data size: {len(train_data)}")
    logger.info(f"Validation data size: {len(validation_data)}")
    
    return train_data, validation_data

def load_prompts(env):
    system_prompt = load_prompt(env["system_zeroshot_prompt_path"])
    base_user_processed_prompt = load_prompt(env["user_processed_input_prompt_path"])
    base_user_raw_prompt = load_prompt(env["user_raw_input_prompt_path"])
    
    return {
        "system": system_prompt,
        "base_user_processed": base_user_processed_prompt,
        "base_user_raw": base_user_raw_prompt,
    }

def filter_mixed_answer_samples(data):
    q_id_to_answer_types = defaultdict(set)
    for item in data:
        q_id = item["q_id"]
        answer_type = item["answer_type"]
        q_id_to_answer_types[q_id].add(answer_type)
        
    mixed_q_ids = [q_id for q_id, types in q_id_to_answer_types.items() 
               if "expert" in types and "nonexpert" in types]
    
    mixed_samples = [item for item in data if item["q_id"] in mixed_q_ids]
    return mixed_samples

def build_dpo_examples(samples, prompts, env, logger, data_type):
    q_id_data = defaultdict(lambda: {
        "prompt": {"preprocessed": "", "raw": ""}, 
        "expert_answers": [], 
        "nonexpert_answers": []
    })
    
    system_prompt = prompts["system"]
    base_user_processed_prompt = prompts["base_user_processed"]
    base_user_raw_prompt = prompts["base_user_raw"]
    
    for sample in samples:
        q_id = sample["q_id"]
        
        user_processed_prompt = base_user_processed_prompt.format(question=sample['preprocessed_question'])
        user_raw_prompt = base_user_raw_prompt.format(title=sample['title'], content=sample['content'])
        
        q_id_data[q_id]["prompt"]["preprocessed"] = system_prompt + "\n\n" + user_processed_prompt
        q_id_data[q_id]["prompt"]["raw"] = system_prompt + "\n\n" + user_raw_prompt
        
        if sample["answer_type"] == "expert":
            q_id_data[q_id]["expert_answers"].append(sample["preprocessed_answer"])
        elif sample["answer_type"] == "nonexpert":
            q_id_data[q_id]["nonexpert_answers"].append(sample["preprocessed_answer"])

    dpo_samples = []
    for q_id, info in q_id_data.items():
        prompt = info["prompt"]
        expert_answers = info["expert_answers"]
        nonexpert_answers = info["nonexpert_answers"]

        for expert_answer in expert_answers:
            for nonexpert_answer in nonexpert_answers:
                dpo_samples.append({
                    "q_id": q_id,
                    "prompt": prompt,
                    "chosen": expert_answer,
                    "rejected": nonexpert_answer
                })
                
    save_dpo_examples(dpo_samples, env, logger, data_type)

def save_dpo_examples(dpo_samples, env, logger, data_type):
    logger.info(f"DPO {data_type} data size: {len(dpo_samples)}")
    output_path = env[f"dpo_{data_type}_data_path"]
    save_json(dpo_samples, output_path)
    logger.info(f"{output_path} saved")


if __name__ == "__main__":
    env = load_environment()
    logger = setup_logging()
    
    train_data, validation_data = load_datasets(env["data_files"], logger)
    
    prompts = load_prompts(env)
    
    mixed_train_samples = filter_mixed_answer_samples(train_data)
    mixed_validation_samples = filter_mixed_answer_samples(validation_data)
    
    build_dpo_examples(mixed_train_samples, prompts, env, logger, "train")
    build_dpo_examples(mixed_validation_samples, prompts, env, logger, "validation")
                