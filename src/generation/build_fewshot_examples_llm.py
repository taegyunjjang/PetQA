import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import os
import sys
import openai
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

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
        question_type = item['answer_type']
        categorized[question_type]['data'].append(item)
        categorized[question_type]['preprocessed_questions'].append(item['preprocessed_question'])
    return dict(categorized)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    if embeddings.shape[0] > 0:
        index.add(embeddings)
    return index

def find_similar_samples(test_sample_embedding, predicted_question_type, categorized_train_embeddings, categorized_train_data_raw, k=6):
    train_embeddings_in_category = categorized_train_embeddings[predicted_question_type]['embeddings']
    train_data_in_category = categorized_train_data_raw[predicted_question_type]['data']

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

def get_gpt_predictions(client, questions, logger):
    predictions = []
    system_prompt = """
    당신의 역할은 주어진 질문이 전문가 답변을 요구하는 질문인지, 일반인 답변을 요구하는 질문인지 분류하는 것입니다.
    전문가 답변을 요구하는 질문은 전문적인 지식을 요구하며, 일반인 답변을 요구하는 질문은 실질적인 해결책을 요구합니다.
    응답은 반드시 'expert' 또는 'nonexpert' 중 하나여야 합니다.
    
    ### 예시
    질문: 강아지가 푸들인데 지금 6개월 되었습니다. 다른 이빨은 다 유치가 빠졌는데 송곳니만 아직 안 빠졌습니다. 흔들림도 전혀 없고, 영구치가 바로 옆에 또 나고 있는데 어떻게 해야 하나요?
    출력: expert
    
    질문: 약 1달 전에 길고양이를 데려와 키우고 있습니다. 어미가 없어 보여 데려왔는데, 적응할수록 입질이 심해집니다. 처음에는 약해서 괜찮았지만, 이제는 아파서 밀쳐내면 더 심해집니다. 특히 밤에 잘 때 심해서 잠을 못 자 이층침대를 사려고 하는데, 아빠는 몇 개월 있으면 소용없어질 거라고 합니다. 성묘가 되면 입질이 덜할까요?
    출력: nonexpert
    """
    
    for question in tqdm(questions, desc="Classifying question types..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"질문: {question}\n출력: "}
                ],
                temperature=0,
                max_tokens=10
            )
            label = response.choices[0].message.content.strip().lower()
            if label not in ['expert', 'nonexpert']:
                label = 'nonexpert'
            predictions.append(label)
        except Exception as e:
            logger.error(f"Error classifying question '{question}': {e}")
            predictions.append('nonexpert') 
            
    return predictions


if __name__ == "__main__":
    env = load_environment()
    logger = setup_logging()

    train_path = env['data_files']['train']
    test_path = env['data_files']['test']

    train_data = load_json(train_path)
    test_data = load_json(test_path)

    model_name = "klue/roberta-large"
    embedder = SentenceEmbedder(model_name)

    logger.info("Categorizing and generating embeddings for training data...")
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
    processed_results = []
    
    test_questions = [item['preprocessed_question'] for item in test_data]
    test_embeddings = embedder.get_embeddings(test_questions)

    predicted_labels_path = "./predicted_labels_by_llm.json"
    if os.path.exists(predicted_labels_path):
        logger.info("Loading pre-generated prediction labels...")
        predicted_labels = load_json(predicted_labels_path)
        logger.info(f"Loaded {len(predicted_labels)} prediction labels")
    else:
        logger.info("Generating prediction labels...")
        predicted_labels = get_gpt_predictions(client, test_questions, logger)
        save_json(predicted_labels, predicted_labels_path)
        logger.info(f"Saved {len(predicted_labels)} prediction labels to '{predicted_labels_path}'")
    
    for i, test_sample in enumerate(tqdm(test_data, desc="Finding similar samples...")):
        predicted_question_type = predicted_labels[i]
        test_sample_embedding = test_embeddings[i]
        similar_samples = find_similar_samples(
            test_sample_embedding,
            predicted_question_type,
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
            "similar_questions": similar_samples
        }
        processed_results.append(result)
        
    output_path = env["fewshot_examples_path"].replace(".json", "_llm.json")
    save_json(processed_results, output_path)
    logger.info(f"Few-shot examples saved to: {output_path}")