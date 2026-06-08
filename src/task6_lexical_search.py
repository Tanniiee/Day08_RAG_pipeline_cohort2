"""
Task 6 — Lexical Search Module (BM25).

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn từ phổ biến
    - Length normalization: document dài không được ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * tf(qi,d)*(k1+1) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
      với k1=1.5 (term saturation), b=0.75 (length normalization)

Cài đặt:
    pip install rank-bm25
"""

import numpy as np
from rank_bm25 import BM25Okapi
from src.task4_chunking_indexing import load_documents, chunk_documents

# =============================================================================
# Build index từ corpus (lazy load — chỉ build 1 lần)
# =============================================================================

_bm25: BM25Okapi | None = None
_corpus: list[dict] = []


def _load_corpus() -> list[dict]:
    """Load và chunk toàn bộ documents từ data/standardized/."""
    docs = load_documents()
    return chunk_documents(docs)


def _get_bm25() -> tuple[BM25Okapi, list[dict]]:
    """Lazy-build BM25 index. Trả về (bm25, corpus)."""
    global _bm25, _corpus
    if _bm25 is None:
        _corpus = _load_corpus()
        # Tokenize: lowercase + split (đủ dùng cho cả tiếng Việt có dấu cách)
        tokenized = [doc["content"].lower().split() for doc in _corpus]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, _corpus


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus cho trước.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    Returns:
        BM25Okapi instance
    """
    tokenized = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25Okapi.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending. Chỉ trả về chunks có score > 0.
    """
    bm25, corpus = _get_bm25()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)  # numpy array, len = len(corpus)

    # Lấy top_k indices có score cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx]["metadata"],
        })

    return results  # đã sorted descending


if __name__ == "__main__":
    queries = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "ca sĩ bị bắt sử dụng ma tuý",
        "cai nghiện bắt buộc",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = lexical_search(q, top_k=3)
        if not results:
            print("  (không có kết quả)")
        for r in results:
            print(f"  [{r['score']:.3f}] ({r['metadata'].get('source','?')}) {r['content'][:80]}...")
