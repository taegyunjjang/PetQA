from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import setup_logging


def load_and_split(file_path, chunk_size=500, overlap=50):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

logger = setup_logging()

textbook_path = "./textbook.txt"
docs = load_and_split(textbook_path)

logger.info(f"총 {len(docs)} 개의 chunk 생성 완료.")

model_id = "Qwen/Qwen3-Embedding-4B"
embed_model = SentenceTransformer(model_id)

doc_embeddings = embed_model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
logger.info(f"임베딩 완료: {doc_embeddings.shape}")

d = doc_embeddings.shape[1]  # 임베딩 차원
index = faiss.IndexFlatL2(d)  # 인덱스 생성
index.add(doc_embeddings)  # 임베딩 벡터 추가

logger.info("FAISS DB 구축 완료")

faiss.write_index(index, "vector_db.index")
np.save("docs.npy", np.array(docs))

logger.info("저장 완료: vector_db.index, docs.npy")