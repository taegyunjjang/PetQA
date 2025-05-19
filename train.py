import os
import torch
import logging
import wandb
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from dotenv import load_dotenv
from trl import SFTTrainer


MODEL_MAPPING = {
    "gemma-3-4b": "google/gemma-3-4b-it",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_environment(model_name, use_raw_format):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    suffix = "_raw" if use_raw_format else ""
    cache_dir = "./models"
    output_dir = "./data/outputs"
    final_model_dir = f"./data/outputs/{model_name}{suffix}"

    train_path = f"./data/training/train_data{suffix}.json"
    validation_path = f"./data/training/validation_data{suffix}.json"
    
    data_files = {
        "train": train_path,
        "validation": validation_path
    }
    
    return {
        "hf_token": hf_token,
        "cache_dir": cache_dir,
        "output_dir": output_dir,
        "final_model_dir": final_model_dir,
        "data_files": data_files
    }

def load_datasets(data_files, logger):
    logger.info("데이터셋 로드 중...")
    dataset = load_dataset("json", data_files=data_files)
    
    train_data = dataset["train"]
    validation_data = dataset["validation"]
    
    logger.info(f"학습 데이터 크기: {len(train_data)}")
    logger.info(f"검증 데이터 크기: {len(validation_data)}")
    
    return train_data, validation_data

def setup_model_config():
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    return lora_config, bnb_config

def load_model_tokenizer(model_name, bnb_config, cache_dir, logger):
    logger.info("모델 및 토크나이저 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAPPING[model_name],
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
        cache_dir=cache_dir
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_MAPPING[model_name],
        cache_dir=cache_dir,
    )
    
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.model_max_length = model.config.text_config.max_position_embeddings
    
    return model, tokenizer

def create_prompt_function(tokenizer, use_raw_format):
    def generate_prompt(data):
        system_prompt = "당신은 반려동물(개, 고양이) 의료 상담 전문 수의사입니다.\n당신의 역할은 개(강아지), 고양이 관련 의료 질문에 대해 유용하고, 완전하며, 전문적인 지식에 기반한 정확한 답변을 하는 것입니다.\n답변 외의 문장은 포함하지 마세요."
        
        if use_raw_format:
            questions = [
                f"{title}\n\n{content}" if content.strip() else title
                for title, content in zip(data['title'], data['content'])
            ]
        else:
            questions = data['question']
        
        prompts = []
        for question, answer in zip(questions, data['answer']):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            prompts.append(prompt)
            
        return prompts
    
    return generate_prompt

def setup_training_args(output_dir, train_data, use_raw_format):
    num_train_epochs = 3
    steps_per_epoch = len(train_data) // (1 * 4)  # batch_size * gradient_accumulation_steps
    total_steps = steps_per_epoch * num_train_epochs
    suffix = "_raw" if use_raw_format else ""
    
    training_args = TrainingArguments(
        output_dir=f"{output_dir}{suffix}",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        warmup_steps=int(0.1 * total_steps),
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to="wandb",
        save_total_limit=3,
    )
    
    return training_args

def train_model(model, train_data, validation_data, training_args, lora_config, formatting_func, logger):
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        args=training_args,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    
    logger.info("학습 시작...")
    trainer.train()
    logger.info("학습 완료")
    
    return trainer

def save_best_model(trainer, output_dir, tokenizer, use_raw_format):
    suffix = "_raw" if use_raw_format else ""
    best_model_path = os.path.join(output_dir, f"best_model{suffix}")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    return best_model_path

def merge_and_save_model(model_name, best_model_path, cache_dir, final_model_dir, tokenizer, logger):
    logger.info("LoRA 어댑터와 기본 모델 병합 중...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAPPING[model_name],
        device_map="cpu",
        cache_dir=cache_dir
    )
    
    peft_model = PeftModel.from_pretrained(
        base_model,
        best_model_path,
        device_map="cpu",
        cache_dir=cache_dir
    )
    
    merged_model = peft_model.merge_and_unload()
    
    logger.info(f"최종 병합 모델 저장 중: {final_model_dir}")
    merged_model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

def main(model_name, use_raw_format):
    logger = setup_logging()
    
    wandb.init(
        project="PetQA",
        entity="petqa",
        config={
            "model_name": MODEL_MAPPING[model_name],
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "batch_size": 1,
            "gradient_accumulation_steps": 4
        }
    )
    
    env = load_environment(model_name, use_raw_format)
    
    train_data, validation_data = load_datasets(env["data_files"], logger)
    
    lora_config, bnb_config = setup_model_config()
    
    model, tokenizer = load_model_tokenizer(model_name, bnb_config, env["cache_dir"], logger)
    
    generate_prompt = create_prompt_function(tokenizer, use_raw_format)
    
    training_args = setup_training_args(env["output_dir"], train_data, use_raw_format)
    
    trainer = train_model(
        model, 
        train_data, 
        validation_data, 
        training_args, 
        lora_config, 
        generate_prompt,
        logger
    )
    
    best_model_path = save_best_model(trainer, env["output_dir"], tokenizer, use_raw_format)
    
    merge_and_save_model(
        model_name,
        best_model_path,
        env["cache_dir"],
        env["final_model_dir"],
        tokenizer,
        logger
    )
    
    logger.info("프로세스 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 훈련")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    main(args.model_name, args.use_raw_format)
