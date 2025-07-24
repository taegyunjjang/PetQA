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

def categorize_data(data):
    """
    데이터를 animal_type과 answer_type에 따라 분류하고, 각 카테고리별로 preprocessed_questions 리스트를 추출합니다.
    예: { 'dog_expert': {'data': [...], 'preprocessed_questions': [...]}, ... }
    """
    categorized_data = {
        'dog_expert': {'data': [], 'preprocessed_questions': []},
        'dog_nonexpert': {'data': [], 'preprocessed_questions': []},
        'cat_expert': {'data': [], 'preprocessed_questions': []},
        'cat_nonexpert': {'data': [], 'preprocessed_questions': []}
    }
    for item in data:
        key = f"{item['animal_type']}_{item['answer_type']}"
        if key in categorized_data:
            categorized_data[key]['data'].append(item)
            categorized_data[key]['preprocessed_questions'].append(item['preprocessed_question'])
    return categorized_data

def find_similar_questions_by_category(test_sample_embedding, 
                                       categorized_train_embeddings, categorized_train_data_raw, 
                                       test_animal_type, test_answer_type, k=6):
    """
    test_sample의 animal_type과 answer_type에 맞는 train 데이터 내에서 유사한 질문을 찾습니다.
    """
    category_key = f"{test_animal_type}_{test_answer_type}"
    
    if category_key not in categorized_train_embeddings or categorized_train_embeddings[category_key]['count'] == 0:
        # 해당 카테고리에 train 데이터가 없는 경우 처리
        return []
    
    train_embeddings_in_category = categorized_train_embeddings[category_key]['embeddings']
    train_data_in_category = categorized_train_data_raw[category_key]['data'] # 원본 데이터를 사용
    
    index = build_faiss_index(train_embeddings_in_category)
    scores, indices = index.search(np.array([test_sample_embedding]), k)
    
    similar_items = []
    for idx, score in zip(indices[0], scores[0]):
        # 원본 데이터에서 해당 인덱스의 아이템을 가져와서 구조를 맞춤
        item = train_data_in_category[idx]
        similar_item_dict = {
            "q_id": item['q_id'],
            "title": item['title'],
            "content": item['content'],
            "a_id": item['a_id'],
            "answer_type": item['answer_type'],
            "animal_type": item['animal_type'],
            "preprocessed_question": item['preprocessed_question'],
            "preprocessed_answer": item['preprocessed_answer'],
            "similarity_score": float(score)
        }
        similar_items.append(similar_item_dict)
    
    return similar_items


if __name__ == "__main__":
    env = load_environment()
    logger = setup_logging()
    train_path = env['data_files']['train']
    test_path = env['data_files']['test']
    
    train_data = load_json(train_path)
    test_data = load_json(test_path)
    
    embedder = SentenceEmbedder()

    # 1. Train 데이터를 카테고리별로 분류하고 임베딩 생성
    logger.info("Categorizing train data and generating embeddings...")
    categorized_train_data_raw = categorize_data(train_data)
    categorized_train_embeddings = {}

    for category, data_info in categorized_train_data_raw.items():
        if data_info['data']:
            categorized_train_embeddings[category] = {
                'embeddings': embedder.get_embeddings(data_info['preprocessed_questions']),
                'count': len(data_info['preprocessed_questions'])
            }
            logger.info(f"Train questions in category '{category}': {categorized_train_embeddings[category]['count']:,}")
        else:
            logger.info(f"No train data in category '{category}'.")
            categorized_train_embeddings[category] = {'embeddings': np.array([]).reshape(0, embedder.model.config.hidden_size), 'count': 0}


    # 2. Test 데이터 임베딩 생성
    logger.info("Test embeddings generation")
    test_preprocessed_questions = [item['preprocessed_question'] for item in test_data]
    test_embeddings = embedder.get_embeddings(test_preprocessed_questions)
    logger.info(f"Test questions: {len(test_data):,}")

    # 3. 각 Test 샘플에 대해 해당 카테고리 내에서 유사한 질문 찾기
    logger.info("Finding similar questions by category for each test sample")
    processed_results = []
    for i in tqdm(range(len(test_data)), desc="Test 샘플별 유사 질문 찾기"):
        test_sample = test_data[i]
        test_sample_embedding = test_embeddings[i]
        
        similar_questions = find_similar_questions_by_category(
            test_sample_embedding,
            categorized_train_embeddings,
            categorized_train_data_raw, # 원본 분류된 데이터를 넘겨줌
            test_sample['animal_type'],
            test_sample['answer_type'],
            k=6  # 6-shot
        )
        
        # test_data의 원본 구조를 그대로 유지
        result = {
            "q_id": test_sample['q_id'],
            "title": test_sample['title'],
            "content": test_sample['content'],
            "a_id": test_sample['a_id'],
            "answer_type": test_sample['answer_type'],
            "animal_type": test_sample['animal_type'],
            "preprocessed_question": test_sample['preprocessed_question'],
            "preprocessed_answer": test_sample['preprocessed_answer'],
            "similar_questions": similar_questions
        }
        processed_results.append(result)
    
    save_json(processed_results, env["fewshot_examples_path"])
    logger.info(f"Few-shot examples saved: {env['fewshot_examples_path']}")