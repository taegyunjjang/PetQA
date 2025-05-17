"""python=3.9"""
import json
import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os
import argparse


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_preprocessed_data(data):
    extracted_data = {
        'ids': [],
        'titles': [],
        'contents': [],
        'answers': [],
        'preprocessed_questions': [],
        'preprocessed_answers': []
    }
    
    for item in data:
        extracted_data['ids'].append(item['id'])
        extracted_data['titles'].append(item['title'])
        extracted_data['contents'].append(item['content'])
        extracted_data['answers'].append(item['answer'])
        extracted_data['preprocessed_questions'].append(item['preprocessed_question'])
        extracted_data['preprocessed_answers'].append(item['preprocessed_answer'])
        
    return extracted_data

class SentenceEmbedder:
    def __init__(self, model_name="klue/roberta-base"):
        """모델과 토크나이저를 초기화합니다."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        """토큰 임베딩의 평균을 계산합니다."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embeddings(self, sentences, batch_size=32):
        """문장 리스트의 임베딩을 계산합니다."""
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
    """Faiss 인덱스를 구축합니다."""
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
                "id": extracted_train_data['ids'][idx],
                "title": extracted_train_data['titles'][idx],
                "content": extracted_train_data['contents'][idx],
                "answer": extracted_train_data['answers'][idx],
                "question": extracted_train_data['preprocessed_questions'][idx],
                "similarity_score": float(score)
            }
            similar_items.append(item)
                
        results.append(similar_items)
    
    return results

def create_id_to_data_mapping(data):
    """ID를 키로 하는 데이터 매핑을 생성합니다."""
    id_to_data = {}
    for item in data:
        id_to_data[item['id']] = item
    return id_to_data

def map_fewshot_examples_with_raw_data(fewshot_examples, train_raw_data, test_raw_data):
    """기존 fewshot 예제에 raw 데이터를 매핑합니다."""
    train_id_to_data = create_id_to_data_mapping(train_raw_data)
    test_id_to_data = create_id_to_data_mapping(test_raw_data)
    
    updated_results = []
    
    for example in fewshot_examples:
        test_id = example['id']
        if test_id in test_id_to_data:
            test_item = test_id_to_data[test_id]
            
            result = {
                "id": test_id,
                "title": test_item.get('title', ''),
                "content": test_item.get('content', ''),
                "answer": test_item.get('answer', ''),
                "similar_questions": []
            }
            
            # 유사한 질문들에 대해 raw 데이터 매핑
            for similar in example['similar_questions']:
                similar_id = similar['id']
                train_item = train_id_to_data[similar_id]
                result['similar_questions'].append({
                    "id": similar_id,
                    "title": train_item.get('title', ''),
                    "content": train_item.get('content', ''),
                    "answer": train_item.get('answer', ''),
                    "similarity_score": similar['similarity_score']
                })
            
            updated_results.append(result)
    
    return updated_results

def main(train_path, test_path, processed_output_path):
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    extracted_train_data = extract_preprocessed_data(train_data)
    extracted_test_data = extract_preprocessed_data(test_data)
    
    print(f"훈련 질문 수: {len(extracted_train_data['preprocessed_questions'])}")
    print(f"테스트 질문 수: {len(extracted_test_data['preprocessed_questions'])}")
    
    embedder = SentenceEmbedder()
    print("훈련 데이터 임베딩 생성 중...")
    train_embeddings = embedder.get_embeddings(extracted_train_data['preprocessed_questions'])
    print("테스트 데이터 임베딩 생성 중...")
    test_embeddings = embedder.get_embeddings(extracted_test_data['preprocessed_questions'])
    
    print("유사한 질문 검색 중...")
    similar_questions = find_similar_questions(
        test_embeddings,
        train_embeddings, 
        extracted_train_data,
        k=6,
    )
    
    processed_results = []
    for i, similar in enumerate(similar_questions):
        result = {
            "id": extracted_test_data['ids'][i],
            "title": extracted_test_data['titles'][i],
            "content": extracted_test_data['contents'][i],
            "answer": extracted_test_data['answers'][i],
            "preprocessed_question": extracted_test_data['preprocessed_questions'][i],
            "preprocessed_answer": extracted_test_data['preprocessed_answers'][i],
            "similar_questions": similar
        }
        processed_results.append(result)
    
    os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)
    with open(processed_output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=4)
        
    
    print(f"Few-shot 예제 저장 완료:")
    print(f"{processed_output_path}")


if __name__ == "__main__":
    train_path = "data/training/train.json"
    test_path = "data/training/test.json"
    output_path = "data/training/fewshot_examples_with_raw_text.json"
    
    main(train_path, test_path, output_path)
    
