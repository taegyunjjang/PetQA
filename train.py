import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_environment(model_name, use_raw_format):
    load_dotenv()
    
    suffix = "_raw" if use_raw_format else ""
    output_dir = "./data/outputs"
    train_dir = "./data/training"
    checkpoint_dir = os.path.join(output_dir, f"{model_name}_petqa{suffix}")
    
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
    
    eval_sample_size = 1
    validation_data = validation_data.select(range(eval_sample_size))
    
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
        system_prompt = "당신은 반려동물(개, 고양이) 의료 상담 전문 수의사입니다.\n당신의 역할은 개(강아지), 고양이 관련 의료 질문에 대해 유용하고, 완전하며, 전문적인 지식에 기반한 정확한 답변을 하는 것입니다.\n답변 외의 문장은 포함하지 마세요."
        
        if use_raw_format:
            question = f"{data['title']}\n\n{data['content']}" if data['content'].strip() else data['title']
        else:
            question = data['preprocessed_question']
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
        fp16=False,
        bf16=True,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
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
    
    return trainer

def cleanup_memory_and_load_best_model(trainer, checkpoint_dir, tokenizer, logger):
    """메모리를 정리하고 best model을 안전하게 로드"""
    
    # 1. 현재 모델과 옵티마이저를 메모리에서 해제
    logger.info("메모리 정리 중...")
    
    # 모델 참조 해제
    del trainer.model
    del trainer.optimizer
    if hasattr(trainer, 'lr_scheduler'):
        del trainer.lr_scheduler
    
    # CUDA 캐시 정리
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    logger.info("VRAM 정리 완료")
    
    # 2. 최고 성능 체크포인트 찾기
    checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoint_dirs:
        logger.warning("체크포인트를 찾을 수 없습니다.")
        return None
    
    # trainer_state.json에서 eval_loss 기반으로 최고 성능 체크포인트 찾기
    best_checkpoint = None
    best_eval_loss = float('inf')
    
    for checkpoint_name in checkpoint_dirs:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        
        if os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                
                log_history = trainer_state.get('log_history', [])
                eval_losses = [entry.get('eval_loss') for entry in log_history if 'eval_loss' in entry]
                
                if eval_losses:
                    current_eval_loss = eval_losses[-1]
                    logger.info(f"{checkpoint_name}: eval_loss = {current_eval_loss:.4f}")
                    
                    if current_eval_loss < best_eval_loss:
                        best_eval_loss = current_eval_loss
                        best_checkpoint = checkpoint_path
            except Exception as e:
                logger.warning(f"{checkpoint_name} 읽기 실패: {e}")
    
    if not best_checkpoint:
        # 최신 체크포인트 사용
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
        best_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
        logger.info(f"eval_loss를 찾을 수 없어 최신 체크포인트 사용: {latest_checkpoint}")
    else:
        logger.info(f"최고 성능 체크포인트: {os.path.basename(best_checkpoint)} (eval_loss: {best_eval_loss:.4f})")
    
    # 3. 토크나이저를 best checkpoint에 저장
    tokenizer.save_pretrained(best_checkpoint)
    
    # 4. DeepSpeed 체크포인트를 PyTorch 모델로 변환
    try:
        import deepspeed
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        import json
        
        logger.info(f"DeepSpeed 체크포인트를 PyTorch 모델로 변환 중: {best_checkpoint}")
        
        # 변환된 모델 저장 경로
        converted_model_path = os.path.join(checkpoint_dir, "best_model")
        os.makedirs(converted_model_path, exist_ok=True)
        
        # ZeRO-3 체크포인트를 FP32 state dict로 변환 (메모리 효율적)
        state_dict = get_fp32_state_dict_from_zero_checkpoint(best_checkpoint)
        
        # state_dict를 PyTorch 형식으로 저장
        torch.save(state_dict, os.path.join(converted_model_path, "pytorch_model.bin"))
        
        # 토크나이저와 config 복사
        tokenizer.save_pretrained(converted_model_path)
        
        import shutil
        config_src = os.path.join(best_checkpoint, "config.json")
        config_dst = os.path.join(converted_model_path, "config.json")
        if os.path.exists(config_src):
            shutil.copy2(config_src, config_dst)
        
        logger.info(f"변환된 모델 저장 완료: {converted_model_path}")
        
        return converted_model_path
        
    except Exception as e:
        logger.warning(f"DeepSpeed 체크포인트 변환 실패: {e}")
        logger.info("수동으로 convert_deepspeed_checkpoint.py를 사용하여 변환하세요.")
        logger.info(f"명령어: python convert_deepspeed_checkpoint.py --checkpoint_dir {best_checkpoint} --output_dir {checkpoint_dir}/best_model_converted")
        return best_checkpoint

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
    
    trainer = train_model(
        model, 
        train_data, 
        validation_data, 
        training_args, 
        generate_prompt,
        logger
    )
    
    best_model_path = cleanup_memory_and_load_best_model(trainer, env["checkpoint_dir"], tokenizer, logger)
    print(f"best_model_path: {best_model_path}")
    logger.info("프로세스 완료")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 훈련")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    main(args.model_name, args.use_raw_format)