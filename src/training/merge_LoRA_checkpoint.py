# merge_best_model.py
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils.utils import MODEL_MAPPING, setup_logging, load_environment
import json
import glob

def find_best_checkpoint(checkpoint_dir, logger):
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        best_model_checkpoint = trainer_state.get('best_model_checkpoint')
        if best_model_checkpoint:
            logger.info(f"Found best model checkpoint: {best_model_checkpoint}")
            return best_model_checkpoint
    
    # trainer_state.json이 없으면 가장 마지막 checkpoint 사용
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
        logger.info(f"Using latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    raise ValueError(f"No checkpoints found in {checkpoint_dir}")

def merge_lora_model(base_model_name, checkpoint_path, output_dir, logger):
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAPPING[base_model_name],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[base_model_name])
    tokenizer.model_max_length = 2048
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading LoRA adapter from: {checkpoint_path}")
    
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    logger.info("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Model merging completed!")
    return merged_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA with base model")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="qwen-2.5-7b")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--answer_type", type=str, choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--checkpoint_path", type=str)
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    
    checkpoint_dir = os.path.join(env["checkpoint_dir"],
                                  f"{args.model_name}_{args.input_format}_{args.answer_type}_LoRA")
    best_model_dir = os.path.join(checkpoint_dir, "best_model")
    
    if args.checkpoint_path:
        best_checkpoint = args.checkpoint_path
    else:
        best_checkpoint = find_best_checkpoint(checkpoint_dir, logger)
    
    # LoRA 모델 병합
    merged_model = merge_lora_model(
        args.model_name, 
        best_checkpoint, 
        best_model_dir, 
        logger
    )
    
    logger.info("All operations completed successfully!")