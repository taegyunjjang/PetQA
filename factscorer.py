import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from colorama import Fore, Style
import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import (
    MODEL_MAPPING, format_time,
    load_environment
)


class FactualityEvaluator:
    def __init__(self, embedding_model_name='klue/bert-base'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name)
        self.model.eval()
        self.model.to(self.device)

    def get_sentence_embedding(self, text):
        if not text:
            return np.zeros(self.model.config.hidden_size)
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def load_facts(self, file_path):
        data_by_id = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            fact_list = json.load(f)
            for entry in fact_list:
                data_by_id[entry["id"]] = entry.get("atomic_facts", [])
        return data_by_id

    def calculate_f1_score(self, gold_atomics, generated_atomics, threshold=0.8):
        gold_embeddings = [self.get_sentence_embedding(atomic)[0] for atomic in gold_atomics]
        generated_embeddings = [self.get_sentence_embedding(atomic)[0] for atomic in generated_atomics]

        gold_embeddings_np = np.array(gold_embeddings) if gold_embeddings else np.empty((0, self.model.config.hidden_size))
        generated_embeddings_np = np.array(generated_embeddings) if generated_embeddings else np.empty((0, self.model.config.hidden_size))

        recall_count = 0
        if len(gold_atomics) > 0 and len(generated_embeddings_np) > 0:
            for gold_emb in gold_embeddings_np:
                similarities = cosine_similarity(gold_emb.reshape(1, -1), generated_embeddings_np)[0]
                if np.max(similarities) >= threshold:
                    recall_count += 1
            recall = recall_count / len(gold_atomics)
        else:
            recall = 0

        precision_count = 0
        if len(generated_atomics) > 0 and len(gold_embeddings_np) > 0:
            for gen_emb in generated_embeddings_np:
                similarities = cosine_similarity(gen_emb.reshape(1, -1), gold_embeddings_np)[0]
                if np.max(similarities) >= threshold:
                    precision_count += 1
            precision = precision_count / len(generated_atomics)
        else:
            precision = 0

        if precision + recall == 0:
            f1_score_for_doc = 0
        else:
            f1_score_for_doc = 2 * (precision * recall) / (precision + recall)
        
        return f1_score_for_doc

    def evaluate(self, generated_facts_path, gold_facts_path, threshold=0.8):
        """
        주어진 경로의 파일들을 기반으로 전체 평가를 수행하고 평균 F1 점수를 반환합니다.
        """
        print(f"평가 파일: {generated_facts_path}")

        gold_data_by_id = self.load_facts(gold_facts_path)
        generated_data_by_id = self.load_facts(generated_facts_path)

        all_f1_scores = []
        processed_ids = sorted(list(set(gold_data_by_id.keys()) & set(generated_data_by_id.keys())))
        print(f"샘플 개수: {len(processed_ids)}")

        for doc_id in processed_ids:
            gold_atomics_for_doc = gold_data_by_id.get(doc_id, [])
            generated_atomics_for_doc = generated_data_by_id.get(doc_id, [])
            
            f1_score = self.calculate_f1_score(gold_atomics_for_doc, generated_atomics_for_doc, threshold)
            all_f1_scores.append(f1_score)
            
        final_avg_f1_score = np.mean(all_f1_scores) if all_f1_scores else 0
        return final_avg_f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="factuality 평가")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gpt-4o-mini")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--answer_type", choices=["expert", "nonexpert"], default="expert")
    parser.add_argument("--animal_type", choices=["cat", "dog"], default="dog")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    parser.add_argument("--use_finetuned_model", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    env = load_environment()
    generated_facts_path = os.path.join(env['atomic_facts_dir'], 
                                        f"{args.model_name}_{args.shot}_{args.input_format}.json")
    gold_facts_path = env['gold_facts_path']

    start_time = time.time()
    evaluator = FactualityEvaluator(embedding_model_name='klue/bert-base')
    final_score = evaluator.evaluate(generated_facts_path, gold_facts_path, args.threshold)

    print(f"Avg Factuality F1: {Fore.RED}{final_score:.3f}{Style.RESET_ALL}")
    print(f"평가 작업 총 소요 시간: {format_time(time.time() - start_time)}")
    print("--------------------------------")