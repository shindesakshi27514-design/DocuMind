import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# A compact, widely used sentence-transformer model. The first run downloads it.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vector_store(chunks: list[dict]) -> dict:
    if not chunks:
        raise ValueError("There is no document text to index.")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [item["text"] for item in chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return {"index": index, "model": model, "chunks": chunks}


def search_vector_store(store: dict, query: str, k: int = 4) -> list[dict]:
    if not query.strip():
        return []
    vector = store["model"].encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    count = min(k, len(store["chunks"]))
    scores, indices = store["index"].search(vector, count)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            item = dict(store["chunks"][int(idx)])
            item["score"] = float(score)
            results.append(item)
    return results
