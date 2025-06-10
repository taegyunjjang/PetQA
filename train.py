import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # GPU 메모리 할당 효율성을 높여서 메모리 단편화로 인한 OOM 발생을 줄임

import torch
import logging
import wandb
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from datasets import load_dataset
from dotenv import load_dotenv
from trl import SFTTrainer


MODEL_MAPPING = {
    "gemma-3-4b": "google/gemma-3-4b-it",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "hcx-seed-3b": "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
}

def setup_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

def load_environment(model_name, use_raw_format):
    load_dotenv()
    
    suffix = "raw" if use_raw_format else "preprocessed"
    output_dir = "./data/outputs"
    train_dir = "./data/training"
    checkpoint_dir = os.path.join(output_dir, f"{model_name}_petqa_{suffix}")
    
    os.environ["WANDB_DIR"] = checkpoint_dir
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    train_path = os.path.join(train_dir, f"train.json")
    validation_path = os.path.join(train_dir, f"validation.json")
    
    data_files = {
        "train": train_path,
        "validation": validation_path
    }
    
    return {
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "data_files": data_files
    }

def load_datasets(data_files, logger):
    logger.info("데이터셋 로드 중...")
    dataset = load_dataset("json", data_files=data_files)
    
    train_data = dataset["train"]
    validation_data = dataset["validation"]
    
    return train_data, validation_data

def load_model_tokenizer(model_name, logger):
    logger.info("모델 및 토크나이저 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAPPING[model_name],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_MAPPING[model_name],
    )
    
    tokenizer.model_max_length = 4096
    
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def create_prompt_function(tokenizer, use_raw_format):
    def generate_prompt(data):
        system_prompt = "당신은 반려동물(개, 고양이) 의료 상담 전문 수의사입니다.\n당신의 역할은 개(강아지), 고양이 관련 의료 질문에 대해 유용하고, 완전하며, 전문적인 지식에 기반한 정확한 답변을 하는 것입니다.\n질문에 대해 차근차근 생각하며 답변하고, 답변 문장의 개수는 최대 5개를 넘지 마세요."
        
        if use_raw_format:
            question = f"제목: {data['title']}\n본문: {data['content']}\n답변: " if data['content'].strip() else f"제목: {data['title']}\n답변: "
        else:
            question = f"질문: {data['preprocessed_question']}\n답변: "
        
        answer = data['preprocessed_answer']
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt
    
    return generate_prompt

def setup_training_args(checkpoint_dir, train_data):
    num_train_epochs = 3
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 4

    steps_per_epoch = len(train_data) // (per_device_train_batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(0.1 * total_steps)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        warmup_steps=warmup_steps,
        learning_rate=5e-5,
        weight_decay=0.01,
        bf16=True,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to="wandb",
        save_total_limit=3,
    )

    return training_args

def train_model(model, train_data, validation_data, training_args, formatting_func, logger):
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        args=training_args,
        formatting_func=formatting_func,
    )
    
    logger.info("학습 시작...")
    trainer.train()
    logger.info("학습 완료")
    
def main(model_name, use_raw_format):
    logger = setup_logging()
    
    env = load_environment(model_name, use_raw_format)
    
    wandb.init(
        project="PetQA",
        entity="petqa",
        name=f"{model_name}{'_raw' if use_raw_format else ''}",
        config={
            "model_name": MODEL_MAPPING[model_name],
            "use_raw_format": use_raw_format,
            "learning_rate": 5e-5,
            "num_train_epochs": 3,
            "batch_size": 1,
            "gradient_accumulation_steps": 4
        }
    )
    
    train_data, validation_data = load_datasets(env["data_files"], logger)
    
    model, tokenizer = load_model_tokenizer(model_name, logger)
    
    generate_prompt = create_prompt_function(tokenizer, use_raw_format)
    
    training_args = setup_training_args(env["checkpoint_dir"], train_data)
    
    train_model(
        model, 
        train_data, 
        validation_data, 
        training_args, 
        generate_prompt,
        logger
    )
    logger.info("프로세스 완료")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 훈련")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    main(args.model_name, args.use_raw_format)