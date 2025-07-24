"""python=3.9"""
import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import (
    setup_logging, load_environment, load_json, save_json
)


def extract_preprocessed_data(data):
    extracted_data = {
        'q_ids': [],
        'titles': [],
        'contents': [],
        'a_ids': [],
        'answer_types': [],
        'animal_types': [],
        'preprocessed_questions': [],
        'preprocessed_answers': []
    }
    
    for item in data:
        extracted_data['q_ids'].append(item['q_id'])
        extracted_data['titles'].append(item['title'])
        extracted_data['contents'].append(item['content'])
        extracted_data['a_ids'].append(item['a_id'])
        extracted_data['answer_types'].append(item['answer_type'])
        extracted_data['animal_types'].append(item['animal_type'])
        extracted_data['preprocessed_questions'].append(item['preprocessed_question'])
        extracted_data['preprocessed_answers'].append(item['preprocessed_answer'])
        
    return extracted_data

class SentenceEmbedder:
    def __init__(self, model_name="klue/roberta-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embeddings(self, sentences, batch_size=32):
        all_embeddings = []
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="임베딩 생성 중"):
            batch = sentences[i:i+batch_size]
            
            encoded_input = self.tokenizer(
                batch,
                padding=True,
                truncation=True,  # 긴 텍스트 처리를 위해 truncation 활성화
                max_length=512,   # 최대 길이 설정
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # 평균 풀링으로 문장 임베딩 추출
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            all_embeddings.append(sentence_embeddings.cpu().numpy())
        
        # 모든 배치 결합
        all_embeddings = np.vstack(all_embeddings)
        
        return all_embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product 기반 유사도 검색
    index.add(embeddings)  # 인덱스에 벡터 추가
    
    return index

def find_similar_questions(test_embeddings, train_embeddings, extracted_train_data, k=6):
    index = build_faiss_index(train_embeddings)
    scores, indices = index.search(test_embeddings, k)
    
    results = []
    for _, (idx_list, score_list) in enumerate(zip(indices, scores)):
        similar_items = []
        for _, (idx, score) in enumerate(zip(idx_list, score_list)):
            item = {
                "q_id": extracted_train_data['q_ids'][idx],
                "title": extracted_train_data['titles'][idx],
                "content": extracted_train_data['contents'][idx],
                "a_id": extracted_train_data['a_ids'][idx],
                "answer_type": extracted_train_data['answer_types'][idx],
                "animal_type": extracted_train_data['animal_types'][idx],
                "preprocessed_question": extracted_train_data['preprocessed_questions'][idx],
                "preprocessed_answer": extracted_train_data['preprocessed_answers'][idx],
                "similarity_score": float(score)
            }
            similar_items.append(item)
                
        results.append(similar_items)
    
    return results


if __name__ == "__main__":
    env = load_environment()
    logger = setup_logging()
    train_path = env['data_files']['train']
    test_path = env['data_files']['test']
    
    train_data = load_json(train_path)
    test_data = load_json(test_path)
    
    extracted_train_data = extract_preprocessed_data(train_data)
    extracted_test_data = extract_preprocessed_data(test_data)
    
    logger.info(f"Train questions: {len(extracted_train_data['preprocessed_questions']):,}")
    logger.info(f"Test questions: {len(extracted_test_data['preprocessed_questions']):,}")
    
    embedder = SentenceEmbedder()
    logger.info("Train embeddings generation")
    train_embeddings = embedder.get_embeddings(extracted_train_data['preprocessed_questions'])
    logger.info("Test embeddings generation")
    test_embeddings = embedder.get_embeddings(extracted_test_data['preprocessed_questions'])
    
    logger.info("Find similart questions")
    similar_questions = find_similar_questions(
        test_embeddings,
        train_embeddings, 
        extracted_train_data,
        k=6  # 6-shot
    )
    
    processed_results = []
    for i, similar in enumerate(similar_questions):
        result = {
            "q_id": extracted_test_data['q_ids'][i],
            "title": extracted_test_data['titles'][i],
            "content": extracted_test_data['contents'][i],
            "a_id": extracted_test_data['a_ids'][i],
            "answer_type": extracted_test_data['answer_types'][i],
            "animal_type": extracted_test_data['animal_types'][i],
            "preprocessed_question": extracted_test_data['preprocessed_questions'][i],
            "preprocessed_answer": extracted_test_data['preprocessed_answers'][i],
            "similar_questions": similar
        }
        processed_results.append(result)
    
    save_json(processed_results, env["fewshot_examples_path"])
    logger.info(f"Few-shot examples saved: {env['fewshot_examples_path']}")
    
