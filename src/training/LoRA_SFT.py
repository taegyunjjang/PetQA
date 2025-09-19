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

from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dotenv import load_dotenv
load_dotenv()


def setup_distributed():
    if 'RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return local_rank, world_size, rank
    else:
        return None, 1, 0

def load_datasets(data_files, answer_type, logger):
    logger.info("Loading datasets...")
    dataset = load_dataset("json", data_files=data_files)
    
    if answer_type == "E":
        dataset = dataset.filter(lambda x: x["answer_type"] == "expert")
    elif answer_type == "NE":
        dataset = dataset.filter(lambda x: x["answer_type"] == "nonexpert")
    else:
        pass
    
    train_data = dataset["train"]
    validation_data = dataset["validation"]
    
    logger.info(f"Train data length: {len(train_data)}")
    logger.info(f"Validation data length: {len(validation_data)}")
    
    return train_data, validation_data

def setup_lora_config(logger):
    logger.info("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return lora_config

def load_model_tokenizer(model_name, lora_config, local_rank, logger):
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
    
    # model = model.to(torch.device(f"cuda:{local_rank}"))
    # model = DDP(model, device_ids=[local_rank])
    
    # logger.info("Applying LoRA configuration...")
    # model = get_peft_model(model, lora_config)
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_MAPPING[model_name],
    )
    
    tokenizer.model_max_length = 2048
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def create_prompt_function(tokenizer, env, logger):
    logger.info("Creating prompt function...")
    system_prompt = load_prompt(env["system_zeroshot_prompt_path"])
    def generate_prompt(data):
        base_user_prompt = load_prompt(env["user_processed_input_prompt_path"])
        user_prompt = base_user_prompt.format(question=data['preprocessed_question'])
        
        answer = data['summarized_answer']
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt
    
    return generate_prompt

def setup_training_args(output_dir, config, train_data, logger):
    logger.info("Setting up training arguments...")
    config_training_args = config["training_args"]
    
    config_training_args["ddp_find_unused_parameters"] = False
    config_training_args["dataloader_num_workers"] = 2
    
    # config_training_args["load_best_model_at_end"] = True
    
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

def train_model(model, train_data, validation_data, training_args, formatting_func, lora_config, logger,
                resume_from_checkpoint=None):
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        args=training_args,
        formatting_func=formatting_func,
        peft_config=lora_config,
    )
    
    if resume_from_checkpoint:
        logger.info("Resuming training...")
    else:
        logger.info("Starting training...")
        
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer

def merge_best_model(trainer, model_name, output_dir, logger):
    logger.info("Starting model merging process...")
    
    if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
        best_checkpoint = trainer.state.best_model_checkpoint
        logger.info(f"Using best model checkpoint: {best_checkpoint}")
    else:
        best_checkpoint = trainer.args.output_dir
        logger.info(f"Using final checkpoint: {best_checkpoint}")
    
    try:
        logger.info(f"Loading base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_MAPPING[model_name],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        logger.info("Loading LoRA adapter...")
        peft_model = PeftModel.from_pretrained(base_model, best_checkpoint)
        
        logger.info("Merging LoRA weights...")
        merged_model = peft_model.merge_and_unload()
        
        merged_output_dir = os.path.join(output_dir, "best_model")
        os.makedirs(merged_output_dir, exist_ok=True)
        
        logger.info(f"Saving merged model to: {merged_output_dir}")
        merged_model.save_pretrained(merged_output_dir)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[model_name])
        tokenizer.save_pretrained(merged_output_dir)
        
        logger.info("Model merging completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model merging: {str(e)}")
        logger.info("Continuing without merging...")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM SFT")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="qwen-2.5-7b")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--answer_type", type=str, choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    args = parser.parse_args()
    
    start_time = time.time()
    env = load_environment()
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"ANSWER TYPE: {args.answer_type}")
    
    
    wandb_run_name = f"{args.model_name}_{args.input_format}_{args.answer_type}_LoRA"
    output_dir = os.path.join(env["checkpoint_dir"], wandb_run_name)
    os.environ["WANDB_DIR"] = output_dir
    
    local_rank, world_size, rank = setup_distributed()
    is_main_process = rank == 0
    
    if is_main_process:
        wandb.init(project="PetQA", entity="petqa", name=wandb_run_name)
    
    data_files = env["summarization_data_files"]
        
    train_data, validation_data = load_datasets(data_files, args.answer_type, logger)
    lora_config = setup_lora_config(logger)
    model, tokenizer = load_model_tokenizer(args.model_name, lora_config, local_rank, logger)
    generate_prompt = create_prompt_function(tokenizer, env, logger)
    config = load_config(env["config_path"])
    training_args = setup_training_args(output_dir, config, train_data, logger)
    
    trainer = train_model(
        model, 
        train_data, 
        validation_data, 
        training_args, 
        generate_prompt,
        lora_config,
        logger,
        args.resume_from_checkpoint
    )
    
    if is_main_process:
        merge_best_model(trainer, args.model_name, output_dir, logger)
        
    # DDP cleanup
    if local_rank is not None:
        dist.destroy_process_group()
        
    if is_main_process:
        wandb.finish()
        
    logger.info(f"SFT process completed: {format_time(time.time() - start_time)}")