import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import (
    MODEL_MAPPING, setup_logging, load_environment, load_prompt
)

import argparse
import shutil
import json
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset


def load_datasets(data_files, answer_type):
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

def find_best_checkpoint(checkpoint_dir, use_latest_checkpoint, logger):
    logger.info(f"Finding best checkpoint in {checkpoint_dir}")
    checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    
    if not checkpoint_dirs:
        logger.warning("Checkpoint not found.")
        return None
    
    if use_latest_checkpoint:
        latest_checkpoint = checkpoint_dirs[-1]
        best_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        return best_checkpoint_path
    else:
        best_checkpoint_path = None
        best_eval_loss = float('inf')
        
        for checkpoint_name in checkpoint_dirs:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
            
            if os.path.exists(trainer_state_path):
                try:
                    with open(trainer_state_path, 'r', encoding='utf-8') as f:
                        trainer_state = json.load(f)
                    
                    log_history = trainer_state.get('log_history', [])
                    eval_losses = [entry.get('eval_loss') for entry in log_history if 'eval_loss' in entry]
                    
                    if eval_losses:
                        current_eval_loss = eval_losses[-1]  # 마지막 eval_loss 사용
                        
                        logger.info(f"{checkpoint_name}: eval_loss = {current_eval_loss:.4f}")
                        
                        if current_eval_loss < best_eval_loss:
                            best_eval_loss = current_eval_loss
                            best_checkpoint_path = checkpoint_path
                            
                except Exception as e:
                    logger.warning(f"{checkpoint_name} trainer_state.json 읽기 실패: {e}")
        return best_checkpoint_path

def convert_checkpoint(checkpoint_path, output_dir, model_name):
    logger.info(f"Converting checkpoint: {checkpoint_path} to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    try:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
        config = AutoConfig.from_pretrained(MODEL_MAPPING[model_name], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            None, # 가중치는 로드하지 않고 아키텍처만 로드
            config=config,
            state_dict=state_dict, # 변환된 state_dict를 직접 전달
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[model_name])
        
        model.save_pretrained(output_dir, safe_serialization=True) # safe_serialization=True로 safetensors 강제
        tokenizer.save_pretrained(output_dir)
        
        config_src = os.path.join(checkpoint_path, "config.json")
        config_dst = os.path.join(output_dir, "config.json")
        if os.path.exists(config_src):
            shutil.copy2(config_src, config_dst)
            logger.info(f"Config 파일 복사 완료: {config_dst}")
        else:
            logger.warning(f"config.json을 찾을 수 없습니다: {config_src}")
        
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt"
        ]
        
        copied_files = []
        for file_name in tokenizer_files:
            src_path = os.path.join(checkpoint_path, file_name)
            if os.path.exists(src_path):
                dst_path = os.path.join(output_dir, file_name)
                shutil.copy2(src_path, dst_path)
                copied_files.append(file_name)
        
        if copied_files:
            logger.info(f"토크나이저 파일 복사 완료: {', '.join(copied_files)}")
        else:
            logger.warning("토크나이저 파일을 찾을 수 없습니다.")
        
        gen_config_src = os.path.join(checkpoint_path, "generation_config.json")
        if os.path.exists(gen_config_src):
            gen_config_dst = os.path.join(output_dir, "generation_config.json")
            shutil.copy2(gen_config_src, gen_config_dst)
            logger.info("generation_config.json 복사 완료")
        
        logger.info(f"변환 완료: {output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"변환 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_model(args, output_dir, logger):
    logger.info(f"Verifying model: {output_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        output_dir, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[args.model_name])
    model.eval()
    
    train_data, _ = load_datasets(env["data_files"], args.answer_type)
    system_prompt = load_prompt(env["system_zeroshot_prompt_path"])
    
    samples = 5
    for i in range(samples):
        print(f"\n----- Sample {i+1} -----")
        sample = train_data[i]
        if args.input_format == "preprocessed":
            base_user_prompt = load_prompt(env["user_processed_input_prompt_path"])
            question = sample["preprocessed_question"]
            print(f"Question: {question}")
            user_prompt = base_user_prompt.format(question=question)
        else:
            base_user_prompt = load_prompt(env["user_raw_input_prompt_path"])
            title = sample["title"]
            content = sample["content"]
            print(f"Title: {title}")
            print(f"Content: {content}")
            user_prompt = base_user_prompt.format(title=title, content=content)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids.to(model.device),
                max_new_tokens=512,
                do_sample=False,
            )
            
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        print(f"Gold Answer: {sample["preprocessed_answer"]}")
        print(f"Generated Answer: {generated_answer}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSpeed 체크포인트를 PyTorch 모델로 변환")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="exaone-3.5-7.8b")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--answer_type", type=str, choices=["E", "NE", "ALL"], default="ALL")
    parser.add_argument("--use_latest_checkpoint", action="store_true")
    args = parser.parse_args()
    
    env = load_environment()
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"ANSWER TYPE: {args.answer_type}")
    logger.info(f"USE LATEST CHECKPOINT: {args.use_latest_checkpoint}")
    
    checkpoint_dir = os.path.join(env["checkpoint_dir"],
                                  f"{args.model_name}_{args.input_format}_{args.answer_type}")
    best_model_dir = os.path.join(checkpoint_dir, "best_model")
    
    
    checkpoint_path = find_best_checkpoint(checkpoint_dir, args.use_latest_checkpoint, logger)
    success = convert_checkpoint(checkpoint_path, best_model_dir, args.model_name)
    if success:
        verify_model(args, best_model_dir, logger)