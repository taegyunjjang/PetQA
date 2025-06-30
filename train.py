import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # GPU 메모리 할당 효율성을 높여서 메모리 단편화로 인한 OOM 발생을 줄임
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import (
    MODEL_MAPPING, setup_logging, format_time,
    load_prompt, load_config, load_environment
)

import torch
import wandb
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

from dotenv import load_dotenv
load_dotenv()


def load_datasets(data_files, answer_type, logger):
    logger.info("Loading datasets...")
    logger.info(f"Answer type: {answer_type}")
    dataset = load_dataset("json", data_files=data_files)
    
    if answer_type == "E":
        dataset = dataset.filter(lambda x: x["answer_type"] == "expert")
    elif answer_type == "NE":
        dataset = dataset.filter(lambda x: x["answer_type"] == "nonexpert")
    else:
        pass
    
    train_data = dataset["train"]
    validation_data = dataset["validation"]
    
    return train_data, validation_data

def load_model_tokenizer(model_name, logger):
    logger.info("Loading model and tokenizer...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    if model_name == "gemma-3-4b":
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAPPING[model_name],
        **model_kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_MAPPING[model_name],
    )
    
    tokenizer.model_max_length = 2048  # 원본 train set 받으면 다시 계산해야함
    
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def create_prompt_function(tokenizer, env, input_format):
    system_prompt = load_prompt(env["system_zeroshot_prompt_path"])
    def generate_prompt(data):
        if input_format == "raw":
            base_user_prompt = load_prompt(env["user_raw_input_prompt_path"])
            user_prompt = base_user_prompt.format(title=data['title'], content=data['content'])
        else:
            base_user_prompt = load_prompt(env["user_processed_input_prompt_path"])
            user_prompt = base_user_prompt.format(question=data['preprocessed_question'])
        
        answer = data['preprocessed_answer']
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt
    
    return generate_prompt

def setup_training_args(output_dir, config, train_data):
    config_training_args = config["training_args"]
    
    num_train_epochs = config_training_args["num_train_epochs"]
    per_device_train_batch_size = config_training_args["per_device_train_batch_size"]
    gradient_accumulation_steps = config_training_args["gradient_accumulation_steps"]
    steps_per_epoch = len(train_data) // (per_device_train_batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(0.1 * total_steps)

    final_training_args = config_training_args.copy()
    final_training_args["output_dir"] = output_dir
    final_training_args["warmup_steps"] = warmup_steps
    
    return TrainingArguments(**final_training_args)

def train_model(model, train_data, validation_data, training_args, formatting_func, logger,
                resume_from_checkpoint=None):
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        args=training_args,
        formatting_func=formatting_func,
    )
    
    if resume_from_checkpoint:
        logger.info("Resuming training...")
    else:
        logger.info("Starting training...")
        
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM SFT")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="exaone-3.5-7.8b")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--answer_type", type=str, choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    args = parser.parse_args()
    
    start_time = time.time()
    env = load_environment()
    logger = setup_logging()
    
    wandb_run_name = f"{args.model_name}_{args.input_format}_{args.answer_type}"
    output_dir = os.path.join(env["checkpoint_dir"], wandb_run_name)
    os.environ["WANDB_DIR"] = output_dir
    wandb.init(
        project="PetQA",
        entity="petqa",
        name=wandb_run_name
    )
    
    train_data, validation_data = load_datasets(env["data_files"], args.answer_type, logger)
    model, tokenizer = load_model_tokenizer(args.model_name, logger)
    generate_prompt = create_prompt_function(tokenizer, env, args.input_format)
    config = load_config(env["config_path"])
    training_args = setup_training_args(output_dir, config, train_data)
    
    train_model(
        model, 
        train_data, 
        validation_data, 
        training_args, 
        generate_prompt,
        logger,
        args.resume_from_checkpoint
    )
    
    wandb.finish()
    logger.info(f"Train process completed: {format_time(time.time() - start_time)}")