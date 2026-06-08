"""
HyDE — Hypothetical Document Embeddings

Thay vì embed raw query, ta:
1. Dùng LLM sinh ra một đoạn văn giả định chứa câu trả lời
2. Embed đoạn văn đó để tìm kiếm
3. Điểm mấu chốt: embedding của "câu trả lời giả định" gần với embedding
   của tài liệu thật hơn là embedding của câu hỏi ngắn.

Paper gốc: Gao et al. 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"
https://arxiv.org/abs/2212.10496
"""

import os


HYDE_SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam về phòng chống ma tuý.
Hãy viết một đoạn văn ngắn (3-4 câu) như thể đây là nội dung trích từ văn bản pháp luật
hoặc bài báo có chứa thông tin trả lời cho câu hỏi bên dưới.
Chỉ viết đoạn văn đó, không thêm giải thích hay tiêu đề."""


def generate_hypothetical_document(query: str, openai_key: str | None = None) -> str:
    """
    Sinh ra một tài liệu giả định (hypothetical document) từ query.

    Args:
        query: Câu hỏi người dùng
        openai_key: OpenAI API key (lấy từ env nếu None)

    Returns:
        Chuỗi văn bản giả định. Nếu lỗi, trả về query gốc (graceful fallback).
    """
    key = openai_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return query  # fallback: dùng query gốc

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.4,
            max_tokens=250,
        )
        hyp_doc = resp.choices[0].message.content.strip()
        return hyp_doc if hyp_doc else query
    except Exception:
        return query  # graceful fallback


def hyde_retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3,
                  openai_key: str | None = None) -> list[dict]:
    """
    HyDE retrieval: sinh hypothetical document → embed → semantic search.

    So sánh với retrieval thường:
      - Thường: embed(query) → search
      - HyDE:   embed(LLM("viết tài liệu trả lời: " + query)) → search

    Lợi ích: hypothetical doc có cùng "phong cách" với corpus (văn xuôi pháp luật)
    nên embedding gần hơn với tài liệu thật.
    """
    hyp_doc = generate_hypothetical_document(query, openai_key)

    # Dùng task9 nhưng truyền hypothetical doc thay query
    from src.task9_retrieval_pipeline import retrieve
    chunks = retrieve(hyp_doc, top_k=top_k, score_threshold=score_threshold)
    # Gắn tag để UI biết đây là HyDE
    for c in chunks:
        c["hyde_query"] = hyp_doc[:120] + "..." if len(hyp_doc) > 120 else hyp_doc
    return chunks
