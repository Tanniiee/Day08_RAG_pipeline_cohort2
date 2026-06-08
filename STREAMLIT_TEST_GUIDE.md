# 🧪 Kịch bản test Streamlit App — RAG Pipeline Day 8

Chạy app: `streamlit run app.py`  
URL: http://localhost:8501

---

## ✅ CHECKLIST NHANH (chạy trước)

Khi app load xong, kiểm tra sidebar:
- [ ] Module Status: tất cả 6 task hiện `✅`
- [ ] Không có lỗi đỏ nào trên màn hình

---

## TC-01 · Kiểm tra giao diện cơ bản

| Bước | Thao tác | Kết quả mong đợi |
|------|----------|-----------------|
| 1 | Mở http://localhost:8501 | App load, không lỗi |
| 2 | Xem sidebar | 6 slider/control hiển thị đầy đủ |
| 3 | Xem Module Status | 6 task đều `✅` |
| 4 | Để trống query, nhấn nút Run | Nút bị disabled (không click được) |

---

## TC-02 · Query hỏi về pháp luật

**Sidebar settings:** top_k=5, threshold=0.3, rrf, reranking ON

**Query:** `Hình phạt cho tội tàng trữ trái phép chất ma tuý Điều 248`

Kết quả mong đợi:
- [ ] Step 1 Semantic: hiện kết quả, score ~0.5–0.8, source file là `luat` hoặc `cac-toi-pham`
- [ ] Step 2 Lexical: hiện kết quả, score ~10–15, có "Điều 248" trong content
- [ ] Step 3 RRF Merge: hiện merged results
- [ ] Step 4 Rerank: **KHÔNG** hiện warning `⚠ fallback`; source type = `legal`
- [ ] Full Pipeline: bar chart hiển thị 5 cột, top result chứa "tàng trữ" hoặc "Điều 248"
- [ ] Summary metrics: Semantic top-1 > 0.5, Final source = `hybrid`

---

## TC-03 · Query hỏi về tin tức nghệ sĩ

**Query:** `Ca sĩ Miu Lê bị bắt vì ma tuý ở đảo Cát Bà`

Kết quả mong đợi:
- [ ] Semantic top-1 chứa "Miu Lê" trong content
- [ ] Source type = `news` trong top 3
- [ ] Full Pipeline trả source = `hybrid`, không fallback PageIndex
- [ ] Expander #1 có nội dung về vụ bắt giữ

---

## TC-04 · Điều chỉnh top_k

**Query:** `ma tuý`

| top_k | Kết quả mong đợi |
|-------|-----------------|
| 1 | Bar chart 1 cột, 1 expander kết quả |
| 5 | Bar chart 5 cột |
| 10 | Bar chart 10 cột |
| 20 | Bar chart ≤20 cột (tùy số chunks trong DB) |

- [ ] Số expander kết quả khớp với top_k đã chọn

---

## TC-05 · Điều chỉnh Score Threshold

**Query:** `Hình phạt tàng trữ ma tuý`

| Threshold | Kết quả mong đợi |
|-----------|-----------------|
| 0.0 | Không bao giờ fallback PageIndex |
| 0.3 | Không fallback (semantic ~0.57 > 0.3) |
| 0.99 | `⚠ Semantic score (0.xxx) < threshold (0.99)` → fallback PageIndex |

- [ ] Khi threshold=0.99: hiện warning, thử gọi PageIndex

---

## TC-06 · Tắt Reranking

**Sidebar:** Toggle `Bật Reranking` → OFF  
**Query:** `cai nghiện bắt buộc`

- [ ] Step 4 Rerank biến mất khỏi giao diện
- [ ] Full Pipeline vẫn trả kết quả (dùng RRF merged trực tiếp)
- [ ] Kết quả có thể khác so với khi reranking ON

---

## TC-07 · Ẩn/hiện từng bước

**Sidebar:** Bỏ tick `Hiển thị Semantic Search`  
**Query:** `ma tuý`

- [ ] Section "Step 1 — Semantic Search" biến mất
- [ ] Các step khác vẫn hiển thị
- [ ] Full Pipeline vẫn chạy bình thường

Thử tắt thêm `Hiển thị Lexical Search`:
- [ ] Step 2 cũng biến mất
- [ ] Full Pipeline không bị ảnh hưởng

---

## TC-08 · Query chọn từ dropdown mẫu

**Thao tác:** Sidebar → chọn `"Hoặc chọn mẫu"` → chọn `"Rapper Bình Gold dương tính với ma tuý"`

- [ ] Text input tự điền query
- [ ] Nhấn Run → kết quả liên quan đến Bình Gold

---

## TC-09 · Query không liên quan (stress test)

**Query:** `python programming tutorial`

Kết quả mong đợi:
- [ ] Semantic score thấp (~0.2–0.4)
- [ ] Content trả về không liên quan, nhưng app KHÔNG crash
- [ ] Nếu score < 0.3 → warning fallback PageIndex hiển thị

---

## TC-10 · Generation (nếu có OPENAI_API_KEY)

**Sidebar:** Tick `Bật Generation`  
**Query:** `Hình phạt tội tàng trữ ma tuý theo Điều 248 là gì?`

- [ ] Section "Generation với Citation" xuất hiện
- [ ] Answer có chứa `[Nguồn...]` hoặc tên file trong ngoặc
- [ ] "Sources used" expander có ≥ 1 chunk
- [ ] `retrieval_source` = `hybrid`

**Nếu không có API key:**
- [ ] Hiện message "OPENAI_API_KEY chưa set" thay vì crash

---

## TC-11 · Kiểm tra hiệu năng (tốc độ)

**Query:** `Luật phòng chống ma tuý 2021`  
Chạy 3 lần liên tiếp:

| Lần | Step 1 Semantic | Step 2 Lexical | Full Pipeline |
|-----|----------------|---------------|---------------|
| 1 (cold) | ~10–15s (load model) | ~0.5s | ~11s |
| 2 | < 0.1s | < 0.05s | < 0.2s |
| 3 | < 0.1s | < 0.05s | < 0.2s |

- [ ] Lần 2+ nhanh hơn đáng kể (model đã load vào cache)

---

## 🔴 Các lỗi phổ biến & cách fix

| Lỗi hiện trong app | Nguyên nhân | Fix |
|---------------------|-------------|-----|
| `ModuleNotFoundError: chromadb` | Chưa index | Chạy `python -m src.task4_chunking_indexing` |
| `Collection drug_law_docs does not exist` | ChromaDB trống | Chạy task4 |
| `PAGEINDEX_API_KEY chưa set` | Thiếu .env | Thêm key vào `.env` |
| `Failed to submit retrieval: processing` | PDF chưa index xong | Đợi 30s rồi thử lại |
| App trắng/không load | Port bị chiếm | Thử `streamlit run app.py --server.port 8502` |
