"""
Task 7 — Reranking Module.

Phương pháp: RRF (Reciprocal Rank Fusion) + BM25 re-scoring
    - Lý do chọn RRF: không cần API key, không cần download model nặng,
      kết quả tốt theo paper Cormack et al. 2009.
    - RRF gộp 2 ranking: (1) original score từ retrieval, (2) BM25 re-score candidates
    - Formula: RRF(d) = Σ 1 / (k + rank_r(d)), k=60

Tùy chọn nâng cao (có API key):
    - Cross-encoder Jina: rerank_cross_encoder() với JINA_API_KEY
    - Cross-encoder local: sentence-transformers CrossEncoder
"""

import os
import numpy as np
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()  # đọc JINA_API_KEY từ .env tự động


# =============================================================================
# RRF (mặc định — không cần API key)
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60)
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)
    return results


def rerank_bm25_rescore(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Re-score candidates bằng BM25 trực tiếp trên tập candidates (không phải corpus).
    Kết hợp với original score bằng RRF.
    """
    if not candidates:
        return []

    # BM25 re-score trên tập candidates
    tokenized = [c["content"].lower().split() for c in candidates]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Tạo 2 ranked lists: (1) original score, (2) BM25 re-score
    original_ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)

    bm25_ranked = [
        {**candidates[i], "score": float(bm25_scores[i])}
        for i in np.argsort(bm25_scores)[::-1]
    ]

    return rerank_rrf([original_ranked, bm25_ranked], top_k=top_k)


# =============================================================================
# Cross-encoder (cần API key — optional)
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank bằng Jina Reranker API.
    Cần JINA_API_KEY trong environment variable.
    """
    import requests

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise EnvironmentError("JINA_API_KEY chưa được set. Dùng method='rrf' thay thế.")

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()

    results = []
    for r in response.json()["results"]:
        item = candidates[r["index"]].copy()
        item["score"] = round(r["relevance_score"], 4)
        results.append(item)
    return results


# =============================================================================
# MMR (Maximal Marginal Relevance)
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    Chọn candidates vừa relevant vừa diverse.

    Yêu cầu: candidates phải có key 'embedding'.
    """
    if not candidates:
        return []

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_score = None, float("-inf")

        for idx in remaining:
            emb = candidates[idx].get("embedding", [])
            if not emb:
                continue
            relevance = cosine_sim(query_embedding, emb)
            max_sim = max(
                (cosine_sim(emb, candidates[s].get("embedding", [])) for s in selected_indices),
                default=0.0,
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, idx

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [
        {**candidates[i], "score": round(float(candidates[i].get("score", 0)), 4)}
        for i in selected_indices
    ]


# =============================================================================
# Unified interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query     : Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k     : Số kết quả sau rerank
        method    : "rrf" (default) | "cross_encoder" | "mmr"
    """
    if not candidates:
        return []

    if method == "rrf":
        return rerank_bm25_rescore(query, candidates, top_k)
    elif method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise ValueError("MMR cần query_embedding — gọi rerank_mmr() trực tiếp.")
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý tại bãi biển", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ chất cấm", "score": 0.6, "metadata": {}},
        {"content": "Python programming tutorial", "score": 0.3, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=3)
    print("Reranked results:")
    for r in results:
        print(f"  [{r['score']:.6f}] {r['content']}")
