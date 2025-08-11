from embedder import model, load_faiss_index

def retrieve_top_chunks(query, k=3, index_path="index/faiss.index"):
    index, chunks = load_faiss_index(index_path)
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "score": float(distances[0][rank])  # FAISS L2 distance (lower = better)
        })
    return results
