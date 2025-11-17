import sys
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import setup_logging, load_json


def split_context_into_chunks(context, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(context):
        end = min(len(context), start + chunk_size)
        chunks.append(context[start:end])
        start += chunk_size - overlap
    return chunks

def prepare_documents(knowledge_base, chunk_size=500, overlap=50):
    chunks = []
    
    for entry in tqdm(knowledge_base, desc="Processing documents"):
        context = entry.get("context", "")
        if not context:
            continue
        
        context_chunks = split_context_into_chunks(context, chunk_size, overlap)
        for chunk in context_chunks:
            chunks.append(chunk)
    return chunks

def build_vector_db(
    knowledge_base_path,
    output_dir,
    model_id="Qwen/Qwen3-Embedding-4B",
    chunk_size=500,
    overlap=50,
    batch_size=32
):
    logger = setup_logging()
    os.makedirs(output_dir, exist_ok=True)
    knowledge_base = load_json(knowledge_base_path)
    logger.info(f"Loaded {len(knowledge_base)} Wikipedia documents")
    
    chunks = prepare_documents(knowledge_base, chunk_size, overlap)
    logger.info(f"Total {len(chunks)} chunks created")
    
    embed_model = SentenceTransformer(model_id)
    doc_embeddings = embed_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size
    )
    logger.info(f"Embedding completed: {doc_embeddings.shape}")
    
    d = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(doc_embeddings)
    logger.info(f"FAISS index created: {index.ntotal} vectors")
    
    index_path = os.path.join(output_dir, "vector_db.index")
    chunks_path = os.path.join(output_dir, "chunks.npy")
    
    faiss.write_index(index, index_path)
    np.save(chunks_path, np.array(chunks, dtype=object))
    
    logger.info(f"Saved successfully:")
    logger.info(f"  - FAISS index: {index_path}")
    logger.info(f"  - Chunks: {chunks_path}")
    
    return index, chunks


if __name__ == "__main__":
    index, chunks = build_vector_db(
        knowledge_base_path="./kowiki_knowledge_base.json",
        output_dir="./vector_db",
        chunk_size=500,
        overlap=50
    )
    
    print("Vector database built successfully")