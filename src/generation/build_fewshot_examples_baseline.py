import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import (
    setup_logging, load_environment, load_json, save_json
)


class SentenceEmbedder:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences, batch_size=32):
        all_embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size), desc="Embedding generation"):
            batch = sentences[i:i+batch_size]
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(sentence_embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

def categorize_data(data):
    categorized = defaultdict(lambda: {'data': [], 'preprocessed_questions': []})
    for item in data:
        categorized['all']['data'].append(item)
        categorized['all']['preprocessed_questions'].append(item['preprocessed_question'])
    return dict(categorized)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product 기반
    if embeddings.shape[0] > 0:
        index.add(embeddings)
    return index

def find_similar_samples(test_sample_embedding, categorized_train_embeddings, categorized_train_data_raw, k=6):
    train_embeddings_in_category = categorized_train_embeddings['all']['embeddings']
    train_data_in_category = categorized_train_data_raw['all']['data']

    index = build_faiss_index(train_embeddings_in_category)
    scores, indices = index.search(np.array([test_sample_embedding]), k)

    similar_items = []
    for idx, score in zip(indices[0], scores[0]):
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
    
    model_name = "klue/roberta-large"
    embedder = SentenceEmbedder(model_name)
    
    logger.info("Categorizing train data and generating embeddings...")
    categorized_train_data_raw = categorize_data(train_data)
    categorized_train_embeddings = {}
    
    for question_type, data_info in categorized_train_data_raw.items():
        embeddings = embedder.get_embeddings(data_info['preprocessed_questions'])
        categorized_train_embeddings[question_type] = {
            'embeddings': embeddings,
            'count': len(embeddings)
        }
        logger.info(f"Training data '{question_type}': {len(embeddings):,}")
    
    logger.info("Processing test data...")
    test_questions = [item['preprocessed_question'] for item in test_data]
    test_embeddings = embedder.get_embeddings(test_questions)
    
    processed_results = []
    for i, test_sample in enumerate(tqdm(test_data, desc="Processing test samples")):
        test_sample = test_data[i]
        test_sample_embedding = test_embeddings[i]
        
        similar_questions = find_similar_samples(
            test_sample_embedding,
            categorized_train_embeddings,
            categorized_train_data_raw,
            k=6
        )
        
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
    
    output_path = env["fewshot_examples_path"].replace(".json", "_baseline.json")
    save_json(processed_results, output_path)
    logger.info(f"Random few-shot examples from entire train set saved: {output_path}")