"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF)
    3. Rerank
    4. Nếu top semantic score < threshold → fallback sang PageIndex
    5. Return top_k results

Lưu ý về scoring:
    - RRF score tự nhiên nhỏ (~0.016–0.033), KHÔNG phải cosine similarity
    - Threshold nên dùng semantic score (cosine sim, range 0–1), không dùng RRF score
    - Default threshold=0.3 → fallback nếu semantic top-1 < 0.3 (query không liên quan)
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Threshold dùng SEMANTIC score (cosine sim 0–1), không phải RRF score
SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"   # "rrf" (không cần API) | "cross_encoder" (cần Jina key)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense (cosine sim 0–1)
          ├→ Lexical Search  → sparse (BM25 score)
          ├→ RRF Merge       → merged (RRF score ~0.016–0.033)
          ├→ Rerank          → final_results
          └→ If semantic_top_score < threshold → PageIndex fallback

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng cosine similarity tối thiểu (0–1)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'hybrid'|'pageindex'}
    """
    # Step 1: Semantic + lexical search
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Lấy best semantic score để quyết định fallback
    # (cosine sim 0–1, có ý nghĩa hơn RRF score để so threshold)
    best_semantic_score = dense_results[0]["score"] if dense_results else 0.0

    # Step 2: Merge bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # Đảm bảo source được gán
    for item in final_results:
        if "source" not in item:
            item["source"] = "hybrid"

    # Step 4: Check threshold → fallback PageIndex
    # Dùng semantic score (cosine sim) chứ không phải RRF score
    if not final_results or best_semantic_score < score_threshold:
        print(
            f"  ⚠ Semantic score ({best_semantic_score:.3f}) < threshold ({score_threshold}). "
            f"Fallback → PageIndex"
        )
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception as e:
            print(f"  ✗ PageIndex fallback lỗi: {e}")

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] {r['content'][:80]}...")
