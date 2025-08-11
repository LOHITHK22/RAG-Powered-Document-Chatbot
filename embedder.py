import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    return model.encode(chunks, show_progress_bar=True)

def save_faiss_index(vectors, chunks, index_path="index/faiss.index"):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, index_path)

    with open(index_path + ".pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_faiss_index(index_path="index/faiss.index"):
    index = faiss.read_index(index_path)
    with open(index_path + ".pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
