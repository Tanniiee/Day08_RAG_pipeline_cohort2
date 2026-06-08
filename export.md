# Báo Cáo Lab 8 — RAG Pipeline
**VinUniversity AICB-P1 Cohort 2 | Ngày 8/15**

---

## PHẦN I — BÁO CÁO CÁ NHÂN (Tasks 1–10)

### Task 1 — Thu Thập Văn Bản Pháp Luật ✅

**Mục tiêu:** Thu thập tối thiểu 3 văn bản pháp luật về phòng chống ma tuý.

**Kết quả — 3 văn bản đã tải về `data/landing/legal/`:**

| File | Văn bản |
|------|---------|
| `luat-phong-chong-ma-tuy-2021.pdf` | Luật Phòng, chống ma tuý 2021 (Luật số 73/2021/QH15) |
| `nghi-dinh-105-2021.pdf` | Nghị định 105/2021/NĐ-CP hướng dẫn thi hành |
| `cac-toi-pham-ve-ma-tuy-chuong-xx.docx` | Bộ luật Hình sự 2015 (sửa đổi 2017) — Chương XX |

**Code:** `src/task1_collect_legal_docs.py`

---

### Task 2 — Crawl Bài Báo ✅

**Mục tiêu:** Crawl tối thiểu 5 bài báo về nghệ sĩ Việt Nam liên quan ma tuý.

**Thư viện sử dụng:** Crawl4AI — tự động render JavaScript, trích xuất Markdown sạch.

**Kết quả — 5 bài báo đã crawl về `data/landing/news/`:**

| File | Nguồn |
|------|-------|
| `01_anh-em-ca-si-chi-dan-...json` | VnExpress — Ca sĩ Chí Dân |
| `02_ca-si-miu-le-bi-bat-...json` | VnExpress — Ca sĩ Miu Lê |
| `03_ca-si-long-nhat-...json` | VnExpress — Long Nhật, Sơn Ngọc Minh |
| `04_nguoi-mau-andrea-aybar-...json` | VnExpress — Người mẫu Andrea Aybar |
| `05_rapper-binh-gold-...json` | VnExpress — Rapper Bình Gold |

Mỗi file JSON lưu metadata đầy đủ: `url`, `title`, `date_crawled`, `content` (dạng Markdown).

**Code:** `src/task2_crawl_news.py`

---

### Task 3 — Convert Sang Markdown ✅

**Mục tiêu:** Dùng MarkItDown (Microsoft) convert PDF/DOCX/JSON → Markdown chuẩn.

**Kết quả:** 8 file Markdown trong `data/standardized/`:
- `legal/`: 3 file (luat-phong-chong-ma-tuy-2021.md, nghi-dinh-105-2021.md, cac-toi-pham-ve-ma-tuy-chuong-xx.md)
- `news/`: 5 file tương ứng 5 bài báo

**Lý do chọn MarkItDown:** Hỗ trợ đa định dạng (PDF, DOCX, HTML, JSON), output Markdown bảo toàn cấu trúc văn bản (heading, table, list).

**Code:** `src/task3_convert_markdown.py`

---

### Task 4 — Chunking & Indexing ✅

**Mục tiêu:** Chunk documents và index vào vector store.

**Lựa chọn kỹ thuật:**

| Thành phần | Lựa chọn | Lý do |
|-----------|----------|-------|
| **Chunking** | `RecursiveCharacterTextSplitter` | Fallback qua nhiều separator, không cắt giữa câu |
| **chunk_size** | 800 tokens | Đủ cho 2-3 điều khoản pháp luật |
| **chunk_overlap** | 100 tokens (~12%) | Giữ ngữ cảnh liên đoạn |
| **Embedding model** | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual, hỗ trợ tiếng Việt, 384 dim, chạy CPU |
| **Vector Store** | ChromaDB (local persistence) | Không cần Docker, hỗ trợ metadata filter |

**Code:** `src/task4_chunking_indexing.py`

---

### Task 5 — Semantic Search ✅

**Mục tiêu:** Tìm kiếm ngữ nghĩa (dense retrieval) trên ChromaDB.

**Cách hoạt động:**
1. Embed query bằng cùng model Task 4 (`paraphrase-multilingual-MiniLM-L12-v2`)
2. Query ChromaDB bằng cosine similarity
3. Trả về top_k chunks sorted descending theo score

**Interface:**
```python
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    # Returns: [{'content': str, 'score': float (0-1), 'metadata': dict}]
```

Score là cosine similarity (0–1), càng gần 1 càng liên quan.

**Code:** `src/task5_semantic_search.py`

---

### Task 6 — Lexical Search (BM25) ✅

**Mục tiêu:** Tìm kiếm theo từ khoá (sparse retrieval) dùng BM25.

**Công thức BM25:**

$$\text{score}(q, d) = \sum_{i} \text{IDF}(q_i) \cdot \frac{tf(q_i, d) \cdot (k_1 + 1)}{tf(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

**Các tham số:**
- `k1 = 1.5` — term saturation: kiểm soát mức độ ảnh hưởng của term frequency
- `b = 0.75` — length normalization: tránh ưu tiên document dài

**So sánh BM25 vs TF-IDF:**

| Tiêu chí | TF-IDF | BM25 |
|---------|--------|------|
| Term frequency | Tuyến tính | Bão hoà (saturated) — tránh spam từ khoá |
| Length norm | Chia cho độ dài | Smoothing parameter b — điều chỉnh được |
| IDF | log(N/df) | log((N-df+0.5)/(df+0.5)) — chính xác hơn với corpus nhỏ |
| Hiệu suất | Tốt | Tốt hơn ~5-15% trên benchmark TREC |

BM25 vượt trội hơn TF-IDF vì term frequency được "bão hoà" — từ xuất hiện 10 lần không có điểm gấp 10 lần từ xuất hiện 1 lần.

**Interface:**
```python
def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    # Returns: [{'content': str, 'score': float, 'metadata': dict}]
```

**Code:** `src/task6_lexical_search.py`

---

### Task 7 — Reranking ✅

**Mục tiêu:** Chấm lại độ liên quan của kết quả retrieval để lọc noise.

**Phương pháp chọn: RRF (Reciprocal Rank Fusion)**

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}, \quad k = 60$$

**Lý do chọn RRF thay vì cross-encoder:**
- Không cần API key hay download model nặng
- Hoạt động trên mọi thiết bị (CPU-only)
- Kết quả tốt theo paper Cormack et al. 2009
- Phù hợp với môi trường học tập

**Pipeline reranking:**
1. Lấy kết quả từ semantic search (sorted by cosine sim)
2. Lấy kết quả từ BM25 re-score trên tập candidates
3. RRF merge 2 ranking lists → kết quả cuối

**Code:** `src/task7_reranking.py`

---

### Task 8 — PageIndex Vectorless RAG ✅

**Mục tiêu:** Implement fallback retrieval dùng PageIndex (không dùng vector/embedding).

**Cách PageIndex hoạt động — khác hoàn toàn vector search:**
- Không dùng embedding hay cosine similarity
- Hiểu cấu trúc tài liệu: heading, table, list, section
- Thích hợp cho văn bản pháp luật có cấu trúc rõ ràng (điều, khoản, điểm)

**API flow:**
```python
client = PageIndexClient(api_key)
doc    = client.submit_document(file_path)   # upload PDF
query  = client.submit_query(doc_id, query)  # gửi câu hỏi
result = client.get_retrieval(retrieval_id)  # lấy kết quả
```

**Vai trò trong pipeline:** Fallback khi hybrid search không đủ evidence (score < threshold).

**Code:** `src/task8_pageindex_vectorless.py`

---

### Task 9 — Retrieval Pipeline Hoàn Chỉnh ✅

**Mục tiêu:** Kết hợp Task 5–8 thành một pipeline thống nhất với fallback logic.

**Kiến trúc:**

```
Query
  ├→ Semantic Search (cosine sim, ChromaDB)
  ├→ Lexical Search  (BM25, rank-bm25)
  ├→ RRF Merge
  ├→ Rerank
  └→ Nếu top semantic score < 0.3 → PageIndex Fallback
```

**Lưu ý về scoring:**
- Semantic score: cosine similarity (0–1) → dùng để quyết định fallback
- BM25 score: giá trị tuyệt đối → chỉ dùng để rank
- RRF score: ~0.016–0.033 → chỉ dùng để rank, không dùng cho threshold

**Interface:**
```python
def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3) -> list[dict]:
    # 1. semantic_search + lexical_search
    # 2. RRF merge
    # 3. Rerank
    # 4. Fallback PageIndex nếu top semantic score < threshold
    # 5. Return top_k results
```

**Code:** `src/task9_retrieval_pipeline.py`

---

### Task 10 — Generation Có Citation ✅

**Mục tiêu:** Inject context vào prompt, yêu cầu LLM trả lời có citation.

**Cấu hình LLM:**

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `top_k` (chunks) | 5 | Đủ evidence, không gây lost in the middle |
| `top_p` | 0.9 | Đủ diverse, không quá random |
| `temperature` | 0.3 | Factual RAG cần ít sáng tạo |
| Model | GPT-4o-mini | Cân bằng chất lượng / chi phí |

**Document Reordering (tránh Lost in the Middle):**

LLM nhớ thông tin ở đầu và cuối tốt hơn giữa prompt. Strategy:
```
Input (sorted by score): [1, 2, 3, 4, 5]
Output (reordered):      [1, 3, 5, 4, 2]
```
Chunk quan trọng nhất → đầu; quan trọng nhì → cuối; kém quan trọng → giữa.

**System Prompt:** Cho phép dùng kiến thức nền (Bộ luật Hình sự 2015, Luật 2021, Nghị định 105) khi context không đủ, nhưng phải ghi rõ `(thông tin từ kiến thức nền)` để người dùng phân biệt với thông tin từ context được index.

**Code:** `src/task10_generation.py`

---

## PHẦN II — BÁO CÁO NHÓM

### Sản phẩm 1: RAG Chatbot ✅

**Giao diện:** Streamlit (`group_project/chatbot.py`)
**Deploy:** Hugging Face Spaces — Docker SDK
**URL:** https://huggingface.co/spaces/TannieNe/druglaw-rag-chatbot

**Tính năng:**

| Tính năng | Trạng thái | Mô tả |
|-----------|-----------|-------|
| Giao diện chat | ✅ | Streamlit chat_message, lịch sử cuộc trò chuyện |
| Citation | ✅ | Hiển thị nguồn `[Tên nguồn, Năm]` trong câu trả lời |
| Conversation Memory | ✅ | Nhớ lịch sử 10 tin nhắn gần nhất để trả lời follow-up |
| Source display | ✅ | Hiển thị chunks với score, source, loại tài liệu |
| Tunable params | ✅ | Sidebar: top_k, score threshold |
| HyDE toggle | ✅ | Checkbox bật/tắt HyDE retrieval |

**Kiến trúc chatbot:**
```
User Query
    ↓
[Conversation Memory] → Query enrichment (nếu có lịch sử)
    ↓
[HyDE] (nếu bật) → Sinh hypothetical document → Embed
    ↓
[Retrieval Pipeline] (Task 9) → top_k chunks
    ↓
[Generation] (Task 10) → Answer + Citations
    ↓
Streamlit UI → Display answer + sources
```

---

### Sản phẩm 2: RAG Evaluation Pipeline ✅

**Framework:** RAGAS

**Golden Dataset:** `group_project/evaluation/golden_dataset.json` — 17 cặp Q&A về pháp luật ma tuý, bao gồm:
- Câu hỏi về điều khoản cụ thể (Điều 248, 249, 250...)
- Câu hỏi về hình phạt theo loại và lượng ma tuý
- Câu hỏi về tin tức nghệ sĩ liên quan

**Kết quả Evaluation — A/B Comparison:**

| Metric | Config A: Hybrid + Rerank | Config B: Dense-only | Chênh lệch |
|--------|--------------------------|---------------------|------------|
| **Faithfulness** | **0.8758** | 0.5474 | +0.33 ↑ |
| **Answer Relevancy** | **0.6802** | 0.3987 | +0.28 ↑ |
| **Context Recall** | **0.5922** | 0.5784 | +0.01 ↑ |
| **Context Precision** | **0.7958** | 0.6556 | +0.14 ↑ |
| **Overall** | **0.736** | 0.545 | **+0.19 ↑** |

**Phân tích:**
- Config A (Hybrid+Rerank) vượt Config B trên cả 4 metrics, đặc biệt Faithfulness (+33%) và Answer Relevancy (+28%)
- Context Recall thấp (0.59) → retriever chưa lấy đủ evidence cho một số câu hỏi chi tiết về điều khoản — cải tiến bằng cách tăng chunk_overlap hoặc dùng HyDE
- Faithfulness cao (0.88) → câu trả lời bám sát context, ít hallucination

**Code:** `group_project/evaluation/eval_pipeline.py`

---

## PHẦN III — BONUS POINTS (20/20 điểm)

### Bonus 1: Conversation Memory — 3 điểm ✅

Chatbot nhớ lịch sử 10 tin nhắn gần nhất. Khi có follow-up question ("còn điều đó thì sao?"), hệ thống tự động làm giàu query bằng lịch sử hội thoại trước khi gửi vào retrieval pipeline, giúp chatbot hiểu ngữ cảnh câu hỏi.

**Implementation:** `group_project/chatbot.py` — hàm `answer_with_memory()` với `st.session_state.messages`

---

### Bonus 2: UI/UX với Source & Score Display — 3 điểm ✅

Mỗi câu trả lời hiển thị:
- Badge loại tài liệu: `📚 Pháp luật` / `📰 Tin tức`
- Số chunks sử dụng, badge HyDE nếu đang bật
- Expander "Nguồn tham khảo" với từng chunk: nội dung tóm tắt, score, source file
- Score hiển thị dạng progress bar và số thập phân

---

### Bonus 3: HyDE (Hypothetical Document Embeddings) — 5 điểm ✅

**Paper gốc:** Gao et al. 2022 — *"Precise Zero-Shot Dense Retrieval without Relevance Labels"* (https://arxiv.org/abs/2212.10496)

**Nguyên lý:**

```
Query ngắn                                →  Embedding gần với câu hỏi
Hypothetical document (LLM-generated)    →  Embedding gần với tài liệu thật
```

Query ngắn ("Điều 248 quy định gì?") có embedding khác xa tài liệu pháp luật dài. HyDE giải quyết bằng cách:
1. Dùng LLM (GPT-4o-mini) sinh đoạn văn giả định như thể đây là trích dẫn từ văn bản pháp luật
2. Embed đoạn văn đó thay vì embed câu hỏi gốc
3. Embedding của "câu trả lời giả định" gần hơn nhiều với embedding của tài liệu thật

**Graceful fallback:** Nếu không có API key, tự động fallback về query gốc — chatbot không bị crash.

**Implementation:** `src/hyde.py` + toggle trong sidebar chatbot

---

### Bonus 4: Giải Thích Lexical Search (BM25 vs TF-IDF) — 5 điểm ✅

Sidebar chatbot có expander "📚 Về BM25 & Lexical Search" giải thích:

- Công thức BM25 đầy đủ với giải thích từng thành phần
- So sánh BM25 vs TF-IDF: term saturation, length normalization, IDF cải tiến
- Lý do BM25 tốt hơn TF-IDF cho văn bản pháp luật (tránh spam điều khoản lặp)
- Hybrid search = dense (semantic) + sparse (BM25) → bổ trợ nhau

---

### Bonus 5: Deploy Online — 4 điểm ✅

**Platform:** Hugging Face Spaces (Docker SDK)
**URL:** https://huggingface.co/spaces/TannieNe/druglaw-rag-chatbot

**Cách deploy:**
- Docker image `python:3.11-slim`, expose port 7860
- Upload qua `huggingface_hub` Python API (xử lý binary files: `.sqlite3`, `.pdf`, `.bin`)
- OPENAI_API_KEY set qua HF Spaces Secrets (không hard-code)
- Build thành công: Docker image push ~34.7s, tất cả packages cài đặt trong ~107s

**File deploy:** `group_project/Dockerfile`, `group_project/requirements.txt`, `group_project/DEPLOY.md`

---

## TỔNG KẾT ĐIỂM

| Hạng mục | Điểm tối đa | Trạng thái |
|----------|------------|-----------|
| Tasks 1–10 (cá nhân) | — | ✅ Hoàn thành |
| RAG Chatbot (nhóm) | — | ✅ Hoàn thành |
| RAG Evaluation (nhóm) | — | ✅ Hoàn thành |
| Conversation Memory | 3 | ✅ |
| UI/UX Source Display | 3 | ✅ |
| HyDE | 5 | ✅ |
| Lexical Search Explanation | 5 | ✅ |
| Deploy Online | 4 | ✅ |
| **Tổng Bonus** | **20** | **✅ 20/20** |

---

## CẤU TRÚC THƯ MỤC

```
Day08_RAG_pipeline_cohort2/
├── src/
│   ├── task1_collect_legal_docs.py     ← Thu thập văn bản pháp luật
│   ├── task2_crawl_news.py             ← Crawl bài báo (Crawl4AI)
│   ├── task3_convert_markdown.py       ← Convert PDF/DOCX → Markdown
│   ├── task4_chunking_indexing.py      ← Chunking + ChromaDB indexing
│   ├── task5_semantic_search.py        ← Dense retrieval (cosine sim)
│   ├── task6_lexical_search.py         ← Sparse retrieval (BM25)
│   ├── task7_reranking.py              ← RRF reranking
│   ├── task8_pageindex_vectorless.py   ← PageIndex fallback
│   ├── task9_retrieval_pipeline.py     ← Hybrid pipeline + fallback
│   ├── task10_generation.py            ← Generation + citation
│   └── hyde.py                         ← HyDE implementation
├── data/
│   ├── landing/legal/                  ← 3 văn bản pháp luật gốc
│   ├── landing/news/                   ← 5 bài báo crawled (JSON)
│   ├── standardized/                   ← 8 file Markdown
│   └── chroma_db/                      ← Vector store
├── group_project/
│   ├── chatbot.py                      ← Streamlit chatbot
│   ├── Dockerfile                      ← HF Spaces deployment
│   ├── requirements.txt                ← Dependencies
│   ├── DEPLOY.md                       ← Hướng dẫn deploy
│   └── evaluation/
│       ├── golden_dataset.json         ← 17 cặp Q&A
│       ├── eval_pipeline.py            ← RAGAS evaluation
│       └── results.md                  ← Kết quả A/B comparison
└── export.md                           ← File báo cáo này
```
