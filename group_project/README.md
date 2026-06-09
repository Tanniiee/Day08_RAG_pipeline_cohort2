# Bài Tập Nhóm — RAG Chatbot & Evaluation Pipeline

## Tổng Quan

Xây dựng **RAG Chatbot** trả lời câu hỏi về **pháp luật ma tuý** và **tin tức nghệ sĩ liên quan**, kèm **Evaluation Pipeline** sử dụng RAGAS.
---
## **Thành viên nhóm:**
- Lâm Văn Tài - 2A202600938
- Đặng Thị Thu Thảo - 2A202600685
- Hà Kế Trung Đức - 2A202600594
- Nguyễn Đăng Huy - 2A202600641

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Pipeline                            │
│                                                             │
│  User Query                                                 │
│      │                                                      │
│      ├──► Semantic Search (OpenAI text-embedding-3-small)   │
│      │         ChromaDB · 260 chunks                        │
│      │                                                      │
│      ├──► Lexical Search (BM25 · rank-bm25)                 │
│      │                                                      │
│      ├──► RRF Merge (Reciprocal Rank Fusion)                │
│      │                                                      │
│      ├──► Reranking (Jina Reranker v2 · multilingual)       │
│      │                                                      │
│      └──► Generation (GPT-4o-mini · citation prompt)        │
│                                                             │
│  Fallback: PageIndex Vectorless RAG (score < 0.3)          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Data Sources                               │
│  📄 Luật Phòng chống ma tuý 2021 (73luat.pdf)              │
│  ⚖️  Bộ luật Hình sự – Chương XX (DOCX)                    │
│  📋 Nghị định 57/2022/NĐ-CP (57-cp.signed.pdf)             │
│  📰 5 bài báo VnExpress (JSON → Markdown)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Sản Phẩm

### 1. RAG Chatbot (Streamlit)

**Tính năng:**
- Giao diện chat thân thiện
- Trả lời có citation `[Nguồn, Điều X]`
- Topic guard — từ chối câu hỏi ngoài chủ đề
- Conversation memory (follow-up questions)
- Hiển thị source documents đã dùng
- 3 câu hỏi gợi ý nhanh

**Chạy:**
```bash
streamlit run app.py
```

---

### 2. RAG Evaluation Pipeline (RAGAS)

**Framework:** [RAGAS](https://github.com/explodinggradients/ragas)

**Golden Dataset:** `group_project/evaluation/golden_dataset.json`
- 17 câu hỏi (8 pháp luật, 5 tin tức, 4 hỗn hợp)
- Mỗi câu có `expected_answer` và `expected_context`

**Metrics đánh giá:**

| Metric | Mô tả |
|--------|-------|
| **Faithfulness** | Câu trả lời có bám sát context không? (không hallucinate) |
| **Answer Relevancy** | Câu trả lời có trả lời đúng câu hỏi không? |
| **Context Recall** | Retriever có lấy đủ evidence cần thiết không? |
| **Context Precision** | Trong context lấy về, bao nhiêu % thực sự hữu ích? |

**So sánh A/B:**

| | Config A | Config B |
|--|----------|----------|
| Retrieval | Hybrid (Semantic + BM25) | Hybrid (Semantic + BM25) |
| Reranking | ✅ Jina Cross-encoder | ❌ Không |
| LLM | GPT-4o-mini | GPT-4o-mini |

**Chạy evaluation:**
```bash
pip install ragas datasets
python group_project/evaluation/eval_pipeline.py
```

**Kết quả:** `group_project/evaluation/results.md`

---

## Cấu Trúc Files

```
group_project/
├── README.md                          ← File này
└── evaluation/
    ├── golden_dataset.json            ← 17 cặp Q&A chuẩn
    ├── eval_pipeline.py               ← Script RAGAS evaluation
    └── results.md                     ← Bảng điểm + phân tích (auto-generated)
```

---

## Hướng Dẫn Cài Đặt & Chạy

### Cài đặt
```bash
pip install -r requirements.txt
pip install ragas datasets
```

### Tạo file .env
```bash
cp .env.example .env
# Điền OPENAI_API_KEY và JINA_API_KEY vào .env
```

### Bước 1: Chuẩn bị dữ liệu (nếu chưa có)
```bash
python src/task2_crawl_news.py      # Crawl 5 bài báo
python src/task3_convert_markdown.py # Convert sang markdown
python src/task4_chunking_indexing.py # Chunk + embed + index
```

### Bước 2: Chạy Chatbot
```bash
streamlit run app.py
# Mở http://localhost:8501
```

### Bước 3: Chạy Evaluation
```bash
python group_project/evaluation/eval_pipeline.py
# Kết quả lưu tại group_project/evaluation/results.md
```

---

## Stack Kỹ Thuật

| Component | Công nghệ |
|-----------|-----------|
| Embedding | OpenAI `text-embedding-3-small` (1536 dim) |
| Vector Store | ChromaDB (local, persistent) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (500/50) |
| Lexical Search | BM25 (`rank-bm25`) |
| Merge | RRF (Reciprocal Rank Fusion, k=60) |
| Reranking | Jina `jina-reranker-v2-base-multilingual` |
| LLM | OpenAI `gpt-4o-mini` (temp=0.2, top_p=0.9) |
| Fallback | PageIndex Vectorless RAG |
| UI | Streamlit |
| Evaluation | RAGAS |

---

## Phân Công Công Việc

| Thành viên | | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Tài, Thảo, Huy, Đức| | Task 1-3: Thu thập & xử lý dữ liệu | ✅ |
| Tài, Thảo, Huy, Đức | | Task 4-6: Chunking, Indexing, Search | ✅ |
|Tài, Thảo, Huy, Đức |   | Task 7-8: Reranking, PageIndex | ✅ |
| Tài, Thảo, Huy, Đức | | Task 9-10: Pipeline, Generation | ✅ |
| Tài, Thảo, Huy, Đức| | Chatbot UI + Evaluation | ✅ |

---

## Lưu Ý

> Giữ lại repo này nếu học **Track 3 Giai đoạn 2** — sẽ phát triển tiếp lên **Knowledge Graph** để xử lý câu hỏi phức tạp hơn.
