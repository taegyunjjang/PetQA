import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from colorama import Fore, Style
import argparse

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}시간 {minutes}분 {seconds}초"
    elif minutes > 0:
        return f"{minutes}분 {seconds}초"
    else:
        return f"{seconds}초"

MODEL_MAPPING = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "hcx-seed-3b": "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    
    # judge model
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "exaone-3.5-32b": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=list(MODEL_MAPPING.keys()), default="gpt-4o-mini")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--use_raw_format", action="store_true")
    parser.add_argument("--expert_type", choices=["expert", "nonexpert"], default="nonexpert")
    parser.add_argument("--animal_type", choices=["cat", "dog"], default="dog")
    args = parser.parse_args()

suffix = "raw" if args.use_raw_format else "preprocessed"
generated_facts_path = f"data/TEST/{args.expert_type}/{args.animal_type}/atomic_facts/{args.model_name}_{args.shot}_{suffix}.json"
gold_facts_path = f"data/TEST/{args.expert_type}/{args.animal_type}/atomic_facts/gold_facts.json"
print(f"평가 파일: {generated_facts_path}")

start_total_time = time.time()

MODEL_NAME = 'klue/bert-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_sentence_embedding(text):
    if not text:
        return np.zeros(model.config.hidden_size)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


gold_data_by_id = {}
with open(gold_facts_path, 'r', encoding='utf-8') as f:
    gold_list = json.load(f)
    for entry in gold_list:
        gold_data_by_id[entry["id"]] = entry.get("atomic_facts", [])

generated_data_by_id = {}
with open(generated_facts_path, 'r', encoding='utf-8') as f:
    generated_list = json.load(f)
    for entry in generated_list:
        generated_data_by_id[entry["id"]] = entry.get("atomic_facts", [])


threshold = 0.8
all_f1_scores = []
processed_ids = sorted(list(set(gold_data_by_id.keys()) & set(generated_data_by_id.keys())))
print(f"샘플 개수: {len(processed_ids)}")

for doc_id in processed_ids:
    gold_atomics_for_doc = gold_data_by_id.get(doc_id, [])
    generated_atomics_for_doc = generated_data_by_id.get(doc_id, [])

    gold_embeddings = [get_sentence_embedding(atomic)[0] for atomic in gold_atomics_for_doc]
    generated_embeddings = [get_sentence_embedding(atomic)[0] for atomic in generated_atomics_for_doc]

    # NumPy 배열로 변환 (빈 리스트 처리)
    gold_embeddings_np = np.array(gold_embeddings) if gold_embeddings else np.empty((0, model.config.hidden_size))
    generated_embeddings_np = np.array(generated_embeddings) if generated_embeddings else np.empty((0, model.config.hidden_size))

    recall_count = 0
    if len(gold_atomics_for_doc) > 0 and len(generated_embeddings_np) > 0:
        for gold_emb in gold_embeddings_np:
            similarities = cosine_similarity(gold_emb.reshape(1, -1), generated_embeddings_np)[0]
            if np.max(similarities) >= threshold:
                recall_count += 1
        recall = recall_count / len(gold_atomics_for_doc)
    else:
        recall = 0

    precision_count = 0
    if len(generated_atomics_for_doc) > 0 and len(gold_embeddings_np) > 0:
        for gen_emb in generated_embeddings_np:
            similarities = cosine_similarity(gen_emb.reshape(1, -1), gold_embeddings_np)[0]
            if np.max(similarities) >= threshold:
                precision_count += 1
        precision = precision_count / len(generated_atomics_for_doc)
    else:
        precision = 0

    if precision + recall == 0:
        f1_score_for_doc = 0
    else:
        f1_score_for_doc = 2 * (precision * recall) / (precision + recall)
    
    all_f1_scores.append(f1_score_for_doc)


final_avg_f1_score = np.mean(all_f1_scores) if all_f1_scores else 0


print(f"Avg Factuality F1: {Fore.RED}{final_avg_f1_score:.3f}{Style.RESET_ALL}")

end_total_time = time.time()
print(f"평가 작업 총 소요 시간: {format_time(end_total_time - start_total_time)}")
print("--------------------------------")