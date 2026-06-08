# Hướng Dẫn Deploy lên Hugging Face Spaces (Docker)

> HF Spaces đã bỏ SDK Streamlit riêng. Dùng **Docker** để deploy Streamlit app.

## 1. Tạo Space

1. Vào https://huggingface.co/spaces → **Create new Space**
2. Chọn:
   - **Space name**: `druglaw-rag-chatbot`
   - **SDK**: **Docker**
   - **Visibility**: Public
3. Click **Create Space**

---

## 2. Chuẩn bị repo

```bash
git clone https://huggingface.co/spaces/<username>/druglaw-rag-chatbot
cd druglaw-rag-chatbot
```

Copy các file vào Space repo:

```bash
# Copy code chatbot
cp /path/to/Day08_RAG_pipeline_cohort2/group_project/chatbot.py .
cp /path/to/Day08_RAG_pipeline_cohort2/group_project/Dockerfile .
cp /path/to/Day08_RAG_pipeline_cohort2/group_project/requirements_hf.txt requirements.txt

# Copy src
cp -r /path/to/Day08_RAG_pipeline_cohort2/src .

# Copy data đã index (quan trọng!)
cp -r /path/to/Day08_RAG_pipeline_cohort2/data .
```

Cấu trúc thư mục Space:
```
├── Dockerfile              ← HF Docker dùng file này
├── chatbot.py              ← Streamlit app
├── requirements.txt
├── src/
│   ├── task9_retrieval_pipeline.py
│   ├── task10_generation.py
│   ├── hyde.py
│   └── ...
└── data/
    └── chroma_db/          ← vector store đã index
```

---

## 3. Set Secrets (API Keys)

Trong HF Space → **Settings** → **Variables and secrets** → **New secret**:

| Name | Value |
|------|-------|
| `OPENAI_API_KEY` | sk-... |

> Docker app trên HF tự inject secrets thành env vars — code đọc qua `os.getenv("OPENAI_API_KEY")` là chạy được.

---

## 4. Push và deploy

```bash
git add .
git commit -m "Deploy DrugLaw RAG Chatbot (Docker + Streamlit)"
git push
```

HF tự build Docker image và deploy. Theo dõi tại tab **Logs**.
Build lần đầu mất ~3–5 phút.

---

## 5. Kiểm tra

URL: `https://huggingface.co/spaces/<username>/druglaw-rag-chatbot`

---

## Lưu ý

- `data/chroma_db` phải được commit vào repo (kiểm tra `.gitignore` không exclude thư mục này)
- HF free tier: 2 vCPU + 16 GB RAM — đủ chạy pipeline
- Cold start: lần đầu load embedding model mất ~30s
