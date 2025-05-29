#!/usr/bin/env python3
"""
DeepSpeed ZeRO-3 체크포인트 -> PyTorch 모델
"""

import os
import argparse
import torch
import shutil
import json
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MODEL_MAPPING = {
    "gemma-3-4b": "google/gemma-3-4b-it",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
}


def find_best_checkpoint(checkpoint_dir):
    """
    trainer_state.json의 eval_loss를 기반으로 최고 성능 체크포인트 찾기
    """
    logger.info(f"최고 성능 체크포인트 찾는 중: {checkpoint_dir}")
    
    # 체크포인트 디렉토리 목록 가져오기
    try:
        checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    except FileNotFoundError:
        logger.error(f"체크포인트 디렉토리가 존재하지 않습니다: {checkpoint_dir}")
        return None
    
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
                with open(trainer_state_path, 'r', encoding='utf-8') as f:
                    trainer_state = json.load(f)
                
                log_history = trainer_state.get('log_history', [])
                eval_losses = [entry.get('eval_loss') for entry in log_history if 'eval_loss' in entry]
                
                if eval_losses:
                    current_eval_loss = eval_losses[-1]  # 마지막 eval_loss 사용
                    
                    logger.info(f"{checkpoint_name}: eval_loss = {current_eval_loss:.4f}")
                    
                    if current_eval_loss < best_eval_loss:
                        best_eval_loss = current_eval_loss
                        best_checkpoint = checkpoint_path
                        
            except Exception as e:
                logger.warning(f"{checkpoint_name} trainer_state.json 읽기 실패: {e}")
    
    if not best_checkpoint:
        # eval_loss를 찾을 수 없는 경우 최신 체크포인트 사용
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
        best_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
        logger.info(f"eval_loss를 찾을 수 없어 최신 체크포인트 사용: {latest_checkpoint}")
    else:
        checkpoint_name = os.path.basename(best_checkpoint)
        logger.info(f"최고 성능 체크포인트: {checkpoint_name} (eval_loss: {best_eval_loss:.4f})")
    
    return best_checkpoint

def convert_checkpoint(checkpoint_path, output_dir):
    """DeepSpeed 체크포인트를 PyTorch 모델로 변환"""
    
    logger.info(f"DeepSpeed 체크포인트를 PyTorch 모델로 변환 중...")
    logger.info(f"입력: {checkpoint_path}")
    logger.info(f"출력: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # ZeRO-3 체크포인트를 FP32 state dict로 변환
        logger.info("DeepSpeed 체크포인트를 로딩 중...")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
        
        # PyTorch 모델 파일로 저장
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        
        # config.json 복사
        config_src = os.path.join(checkpoint_path, "config.json")
        config_dst = os.path.join(output_dir, "config.json")
        if os.path.exists(config_src):
            shutil.copy2(config_src, config_dst)
            logger.info(f"Config 파일 복사 완료: {config_dst}")
        else:
            logger.warning(f"config.json을 찾을 수 없습니다: {config_src}")
        
        # 토크나이저 파일들 복사
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
        
        # generation_config.json 복사
        gen_config_src = os.path.join(checkpoint_path, "generation_config.json")
        if os.path.exists(gen_config_src):
            gen_config_dst = os.path.join(output_dir, "generation_config.json")
            shutil.copy2(gen_config_src, gen_config_dst)
            logger.info("generation_config.json 복사 완료")
        
        # chat_template.jinja 복사
        chat_template_src = os.path.join(checkpoint_path, "chat_template.jinja")
        if os.path.exists(chat_template_src):
            chat_template_dst = os.path.join(output_dir, "chat_template.jinja")
            shutil.copy2(chat_template_src, chat_template_dst)
            logger.info("chat_template.jinja 복사 완료")
        else:
            logger.warning(f"chat_template.jinja을 찾을 수 없습니다: {chat_template_src}")

        logger.info(f"변환 완료: {output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"변환 실패: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        return False

def main():
    parser = argparse.ArgumentParser(description="DeepSpeed 체크포인트를 PyTorch 모델로 변환")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--use_raw_format", action="store_true")
    parser.add_argument("--auto_find_latest", action="store_true",
                       help="checkpoint_dir에서 가장 최신 체크포인트 자동 찾기")
    
    args = parser.parse_args()
    
    suffix = "_raw" if args.use_raw_format else ""
    checkpoint_dir = f"data/outputs/{args.model_name}_petqa{suffix}"
    output_dir = f"data/outputs/{args.model_name}_petqa{suffix}/best_model"
    
    checkpoint_path = find_best_checkpoint(checkpoint_dir)
    
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트 디렉토리가 존재하지 않습니다: {checkpoint_path}")
        return
    
    success = convert_checkpoint(checkpoint_path, output_dir)
    if success:
        print("변환이 성공적으로 완료되었습니다!")
    else:
        print("변환 중 오류가 발생했습니다.")


if __name__ == "__main__":
    main() 