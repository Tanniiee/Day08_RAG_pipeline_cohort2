"""
Task 5 — Semantic Search Module.

Dùng embedding model từ Task 4 để embed query, sau đó query ChromaDB
bằng cosine similarity để lấy top_k chunks liên quan nhất.
"""

from sentence_transformers import SentenceTransformer
from src.task4_chunking_indexing import EMBEDDING_MODEL, get_collection

# Load model 1 lần, tái sử dụng cho mọi query
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng cosine similarity trên ChromaDB.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    # Bước 1: Embed query bằng cùng model Task 4
    model = _get_model()
    query_embedding = model.encode(query).tolist()

    # Bước 2: Query ChromaDB (cosine similarity)
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "content": doc,
            "score": round(1 - dist, 4),
            "metadata": meta,
        })

    # Đảm bảo sorted descending theo score
    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "ca sĩ bị bắt vì sử dụng ma tuý",
        "cai nghiện bắt buộc theo luật",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = semantic_search(q, top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] ({r['metadata'].get('source','?')}) {r['content'][:80]}...")
