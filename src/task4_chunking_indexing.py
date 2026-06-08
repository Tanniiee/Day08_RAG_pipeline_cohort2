"""
Task 4 — Chunking & Indexing vào Vector Store.

Lựa chọn kỹ thuật:
    Chunking: RecursiveCharacterTextSplitter
        - Lý do: An toàn nhất cho văn bản hỗn hợp (pháp luật + tin tức).
          Tự động fallback qua nhiều separator (\n\n → \n → . → space),
          đảm bảo không bao giờ cắt giữa câu nếu có thể tránh.
        - chunk_size=800: đủ context cho 1 đoạn văn pháp luật (thường 2-3 điều khoản)
        - chunk_overlap=100: giữ ngữ cảnh liên đoạn, tránh mất thông tin ở ranh giới

    Embedding: paraphrase-multilingual-MiniLM-L12-v2 (384 dim)
        - Lý do: Multilingual (hỗ trợ tiếng Việt), nhẹ (384 dim), chạy CPU được.
          Tốt hơn all-MiniLM-L6-v2 cho tiếng Việt, nhẹ hơn bge-m3 (1024 dim).

    Vector Store: ChromaDB (local persistence)
        - Lý do: Không cần Docker, lưu local, hỗ trợ cả vector search + metadata filter.
          Phù hợp cho môi trường dev/học tập.

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"

# =============================================================================
# CONFIGURATION
# =============================================================================

# chunk_size=800: đủ context cho 1 đoạn văn pháp luật VN (thường 2-3 điều khoản)
CHUNK_SIZE = 800

# chunk_overlap=100: ~12% overlap — giữ ngữ cảnh liên đoạn, tránh mất thông tin ở ranh giới
CHUNK_OVERLAP = 100

CHUNKING_METHOD = "recursive"  # RecursiveCharacterTextSplitter

# paraphrase-multilingual-MiniLM-L12-v2: multilingual, 384 dim, chạy CPU, tốt cho tiếng Việt
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "drug_law_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file),
            }
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Separator ưu tiên: đoạn văn → dòng → câu → từ → ký tự
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "total_chunks": len(splits),
                }
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng paraphrase-multilingual-MiniLM-L12-v2 (384 dim).

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    print(f"  Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB (local persistence tại data/chroma_db/).
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Xóa collection cũ nếu có (để re-index sạch)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    # Insert theo batch (ChromaDB giới hạn batch size)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"  Collection '{COLLECTION_NAME}': {collection.count()} items")


def get_collection():
    """Helper: lấy ChromaDB collection đã index (dùng cho Task 5)."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  VectorDB : {VECTOR_STORE} → {CHROMA_DIR}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    for d in docs:
        print(f"  - [{d['metadata']['type']}] {d['metadata']['source']}")

    chunks = chunk_documents(docs)
    print(f"\n✓ Created {len(chunks)} chunks")

    print("\n→ Embedding...")
    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    print("\n→ Indexing to ChromaDB...")
    index_to_vectorstore(chunks)
    print("✓ Done!")


if __name__ == "__main__":
    run_pipeline()
