"""
Task 8 — PageIndex Vectorless RAG.

PageIndex hoạt động khác hoàn toàn so với vector search:
    - Không dùng embedding hay vector similarity
    - Structural understanding của document (headings, tables, lists)
    - Chỉ hỗ trợ file PDF
    - Flow: submit_document(pdf) → submit_query → get_retrieval

API flow:
    client = PageIndexClient(api_key)
    doc    = client.submit_document(file_path)   → {"doc_id": ...}
    query  = client.submit_query(doc_id, query)  → {"retrieval_id": ...}
    result = client.get_retrieval(retrieval_id)  → {...}
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

# PageIndex chỉ nhận PDF → chỉ upload PDF gốc từ data/landing/legal/
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
CACHE_FILE = Path(__file__).parent.parent / "data" / ".pageindex_doc_ids"


def _get_client():
    from pageindex import PageIndexClient
    if not PAGEINDEX_API_KEY:
        raise EnvironmentError(
            "PAGEINDEX_API_KEY chưa set. Đăng ký tại https://pageindex.ai và thêm vào .env"
        )
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> list[str]:
    """
    Upload các file PDF pháp luật lên PageIndex.
    Trả về list doc_id. Kết quả được cache để tránh upload lại.
    """
    # Đọc cache nếu có
    if CACHE_FILE.exists():
        doc_ids = CACHE_FILE.read_text().strip().splitlines()
        doc_ids = [d for d in doc_ids if d.strip()]
        if doc_ids:
            print(f"  ✓ Dùng {len(doc_ids)} doc đã upload (cache)")
            return doc_ids

    client = _get_client()
    doc_ids = []

    # Chỉ upload PDF (PageIndex không hỗ trợ DOCX/MD)
    pdf_files = sorted(LANDING_LEGAL_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"  ⚠ Không tìm thấy PDF trong {LANDING_LEGAL_DIR}")
        return []

    for pdf_file in pdf_files:
        print(f"  Uploading: {pdf_file.name}")
        try:
            result = client.submit_document(file_path=str(pdf_file))
            doc_id = result.get("doc_id") or result.get("id") or str(result)
            doc_ids.append(doc_id)
            print(f"    ✓ doc_id: {doc_id}")
        except Exception as e:
            print(f"    ✗ Lỗi: {e}")

    # Lưu cache
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text("\n".join(doc_ids))
    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex (chỉ với file PDF pháp luật).
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    client = _get_client()
    doc_ids = upload_documents()

    if not doc_ids:
        return []

    all_results = []

    for doc_id in doc_ids:
        try:
            # Bước 1: Submit query
            query_result = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = (
                query_result.get("retrieval_id")
                or query_result.get("id")
                or str(query_result)
            )

            # Bước 2: Poll cho đến khi có kết quả (tối đa 30s)
            result = None
            for _ in range(15):
                result = client.get_retrieval(retrieval_id=retrieval_id)
                status = result.get("status", "")
                if status in ("completed", "done", "ready") or result.get("results"):
                    break
                time.sleep(2)

            if not result:
                continue

            # Bước 3: Parse kết quả
            items = result.get("results") or result.get("chunks") or []
            for item in items[:top_k]:
                content = item.get("text") or item.get("content") or str(item)
                score = float(item.get("score", 1.0))
                all_results.append({
                    "content": content,
                    "score": score,
                    "metadata": {"doc_id": doc_id, **item.get("metadata", {})},
                    "source": "pageindex",
                })

        except Exception as e:
            print(f"  ✗ Query lỗi (doc_id={doc_id}): {e}")
            continue

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
    else:
        print("Test PageIndex search:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        if results:
            for r in results:
                print(f"[{r['score']:.3f}] {r['content'][:100]}...")
        else:
            print("(không có kết quả)")
