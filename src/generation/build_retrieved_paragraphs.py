import numpy as np
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import (
    setup_logging, load_environment, load_json, save_json
)


def retrieve_documents(query, embed_model, index, docs, top_k):
    q_emb = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    D, I = index.search(q_emb, top_k)
    return [docs[i] for i in I[0]]


if __name__ == "__main__":
    env = load_environment()
    logger = setup_logging()
    
    index = faiss.read_index("vector_db.index")
    docs = np.load("docs.npy", allow_pickle=True)

    model_name = "Qwen/Qwen3-Embedding-4B"
    embed_model = SentenceTransformer(model_name)
    
    logger.info("Processing test data...")
    test_path = env['data_files']['test']
    test_data = load_json(test_path)
    
    TOP_K = 6
    
    processed_results = []
    for item in tqdm(test_data, desc="Processing test samples"):
        query = item['preprocessed_question']
        retrieved_paragraphs = retrieve_documents(query, embed_model, index, docs, TOP_K)
        
        result = {
            "q_id": item['q_id'],
            "title": item['title'],
            "content": item['content'],
            "a_id": item['a_id'],
            "answer_type": item['answer_type'],
            "animal_type": item['animal_type'],
            "preprocessed_question": item['preprocessed_question'],
            "preprocessed_answer": item['preprocessed_answer'],
            "retrieved_paragraphs": retrieved_paragraphs
        }
        processed_results.append(result)
    
    output_path = env["retrieved_paragraphs_path"]
    save_json(processed_results, output_path)
    logger.info(f"Retrieved paragraphs saved: {output_path}")