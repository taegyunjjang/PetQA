import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # GPU 메모리 할당 효율성을 높여서 메모리 단편화로 인한 OOM 발생을 줄임
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import (
    MODEL_MAPPING, setup_logging, format_time,
    load_json, load_environment, 
)

import torch
import wandb
import time
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import gc

from dotenv import load_dotenv
load_dotenv()


def load_datasets(env, logger):
    logger.info("Loading datasets...")
    def _return_prompt_and_responses(item):
        return {
            "prompt": item["preprocessed_question"],
            "chosen": item["chosen"]["preprocessed_answer"],
            "rejected": item["rejected"]["preprocessed_answer"]
        }
        
    train_data = load_json(env["dpo_train_data_path"])
    validation_data = load_json(env["dpo_validation_data_path"])
    
    logger.info(f"Train data size: {len(train_data)}")
    logger.info(f"Validation data size: {len(validation_data)}")
    
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)
    
    return train_dataset.map(_return_prompt_and_responses), validation_dataset.map(_return_prompt_and_responses)
    
def load_model_tokenizer(model_name, model_path, logger):
    logger.info("Loading model and tokenizer...")
    
    model_load_kwargs = {
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
    }
    
    if model_name == "gemma-3-4b":
        logger.info("Setting 'attn_implementation' to 'eager' for gemma-3-4b.")
        model_load_kwargs['attn_implementation'] = 'eager'
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_load_kwargs,
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_load_kwargs,
    )
    
    if model_name == "gemma-3-4b":
        logger.info("Disabling cache for DataParallel compatibility.")
        model.config.use_cache = False
        ref_model.config.use_cache = False
        
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.model_max_length = 1536

    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, ref_model, tokenizer    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM DPO Training")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="exaone-3.5-7.8b")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--answer_type", type=str, choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()
    
    start_time = time.time()
    env = load_environment()
    
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"ANSWER TYPE: {args.answer_type}")
    logger.info(f"BETA: {args.beta}")
    logger.info(f"LEARNING RATE: {args.lr}")
    
    wandb_run_name = f"{args.model_name}_{args.input_format}_{args.answer_type}_DPO"
    output_dir = os.path.join(env["checkpoint_dir"], wandb_run_name)
    os.environ["WANDB_DIR"] = output_dir
    wandb.init(
        project="PetQA",
        entity="petqa",
        name=wandb_run_name
    )
    
    train_dataset, validation_dataset = load_datasets(env, logger)
    
    model_path = os.path.join(env["checkpoint_dir"], 
                              f"{args.model_name}_{args.input_format}_{args.answer_type}",
                              "best_model")
    model, ref_model, tokenizer = load_model_tokenizer(args.model_name, model_path, logger)
        
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        logging_steps=500,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=1536,  # prompt + completion
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        bf16=True,
        learning_rate=args.lr,
        beta=args.beta,
        report_to="wandb",
    )
    
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer
    )
    
    logger.info("Training...")
    dpo_trainer.train()
        
    logger.info(f" Merging LoRA adapter and base model...")
    merged_model = model.merge_and_unload()
    merged_model_path = os.path.join(output_dir, "merged_model")
    
    logger.info(f"Saving merged model...")
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    
    logger.info(f"DPO process completed: {format_time(time.time() - start_time)}")