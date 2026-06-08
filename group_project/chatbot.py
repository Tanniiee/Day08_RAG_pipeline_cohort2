"""
Group Project — RAG Chatbot
Giao diện chat với citation, conversation memory, source documents.

Chạy: streamlit run group_project/chatbot.py
"""

import sys
import os
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DrugLaw RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ DrugLaw Chatbot")
    st.caption("Hỏi đáp về pháp luật ma tuý và tin tức liên quan")
    st.divider()

    st.subheader("⚙️ Cài đặt")
    top_k = st.slider("Số nguồn tham khảo (top_k)", 1, 10, 5)
    score_threshold = st.slider("Ngưỡng độ liên quan", 0.0, 1.0, 0.3, 0.05)
    show_sources = st.checkbox("Hiển thị nguồn tham khảo", value=True)
    show_scores = st.checkbox("Hiển thị điểm relevance", value=False)
    use_hyde = st.checkbox(
        "🧪 HyDE",
        value=False,
        help="Hypothetical Document Embeddings: sinh tài liệu giả định từ câu hỏi trước khi tìm kiếm. Tốt hơn với câu hỏi ngắn/trừu tượng.",
    )

    st.divider()
    st.subheader("💬 Conversation")
    st.caption(f"Lịch sử: {len(st.session_state.get('messages', []))} tin nhắn")

    if st.button("🗑 Xoá lịch sử", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_history = []
        st.rerun()

    st.divider()
    st.subheader("📌 Câu hỏi gợi ý")
    suggestions = [
        "Điều 248 quy định hình phạt tội tàng trữ ma tuý thế nào?",
        "Ca sĩ nào bị bắt vì liên quan ma tuý?",
        "Cai nghiện bắt buộc theo Nghị định 105 kéo dài bao lâu?",
        "Tội sản xuất ma tuý bị phạt ra sao?",
        "Rapper Bình Gold dính líu ma tuý thế nào?",
    ]
    for s in suggestions:
        if st.button(s[:40] + "...", key=s, use_container_width=True):
            st.session_state.pending_query = s

    st.divider()
    with st.expander("📖 Lexical Search: TF-IDF vs BM25"):
        st.markdown("""
**TF-IDF** *(Task 4 — dùng trong pipeline này)*

Điểm của từ `t` trong tài liệu `d`:

`score = TF(t,d) × IDF(t)`

- **TF** = tần suất từ trong tài liệu
- **IDF** = log(N / df(t)) — phạt từ phổ biến
- **Hạn chế**: không tính độ dài tài liệu → tài liệu dài được lợi

---

**BM25** *(Task 6 — Okapi BM25)*

`score = IDF × TF × (k₁+1) / (TF + k₁(1 - b + b·|d|/avgdl))`

- **k₁** (~1.2–2.0): điều chỉnh mức bão hoà TF
- **b** (~0.75): hệ số chuẩn hoá độ dài
- **avgdl**: độ dài tài liệu trung bình
- TF **bão hoà**: thêm 100 lần từ không tăng điểm tuyến tính
- Chuẩn hoá độ dài tài liệu → **công bằng hơn TF-IDF**

---

**Pipeline này dùng cả hai:**
Dense (ChromaDB) + BM25 → RRF Merge → Rerank
        """)
    with st.expander("🔬 HyDE hoạt động thế nào?"):
        st.markdown("""
**Vấn đề của query embedding:**
Câu hỏi ngắn ("Điều 248?") và tài liệu dài có embedding xa nhau trong không gian vector.

**HyDE giải quyết:**
1. LLM sinh tài liệu giả định: *"Điều 248 BLHS quy định tội tàng trữ ma tuý có hình phạt..."*
2. Embed tài liệu giả định → gần corpus hơn
3. Dùng embedding đó để tìm kiếm

**Kết quả:** Retrieval tốt hơn với câu hỏi ngắn, trừu tượng hoặc follow-up.
        """)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_history" not in st.session_state:
    st.session_state.sources_history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ──────────────────────────────────────────────
# MAIN CHAT UI
# ──────────────────────────────────────────────
st.title("⚖️ DrugLaw RAG Chatbot")
st.caption("Chatbot tra cứu pháp luật phòng chống ma tuý — VinUniversity AICB-P1 Cohort 2")

# Render lịch sử chat
assistant_idx = 0  # đếm riêng index assistant message để map đúng sources_history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Hiển thị sources nếu có và là assistant message
        if msg["role"] == "assistant":
            src_idx = assistant_idx
            assistant_idx += 1
            if show_sources and src_idx < len(st.session_state.sources_history):
                sources = st.session_state.sources_history[src_idx]
            else:
                sources = []
            if sources:
                with st.expander(f"📚 {len(sources)} nguồn tham khảo", expanded=False):
                    for j, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        fname = meta.get("source", "?")
                        stype = meta.get("type", "")
                        score = src.get("score", 0)
                        label = f"**[{j}]** `{fname}`"
                        if stype:
                            label += f" · *{stype}*"
                        if show_scores:
                            label += f" · score `{score:.4f}`"
                        st.markdown(label)
                        st.caption(src.get("content", "")[:200] + "...")

# ──────────────────────────────────────────────
# PIPELINE LOADER (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang khởi động pipeline...")
def load_pipeline():
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation, reorder_for_llm, format_context
    return retrieve, generate_with_citation, reorder_for_llm, format_context

def build_context_from_history(messages: list[dict], window: int = 3) -> str:
    """Tóm tắt lịch sử hội thoại gần nhất để thêm vào prompt."""
    recent = messages[-(window * 2):]  # window cặp Q/A
    parts = []
    for m in recent:
        role = "Người dùng" if m["role"] == "user" else "Trợ lý"
        parts.append(f"{role}: {m['content'][:200]}")
    return "\n".join(parts) if parts else ""

# ──────────────────────────────────────────────
# SAFETY FILTER
# ──────────────────────────────────────────────

_SAFE_REFUSAL = (
    "⚠️ **Tôi không thể hỗ trợ yêu cầu này.**\n\n"
    "Chatbot này chỉ cung cấp thông tin **pháp lý** về phòng chống ma tuý "
    "(hình phạt, quy trình cai nghiện, quy định pháp luật) và **tin tức** liên quan.\n\n"
    "Tôi không cung cấp hướng dẫn mua bán, chế tạo, sử dụng ma tuý "
    "hoặc bất kỳ hành vi vi phạm pháp luật nào.\n\n"
    "Nếu bạn hoặc người thân cần hỗ trợ cai nghiện, hãy liên hệ:\n"
    "- **Đường dây hỗ trợ cai nghiện**: 1800 599 920 (miễn phí)\n"
    "- **Cơ sở cai nghiện bắt buộc** tại địa phương"
)

# Dùng keyword normalized (bỏ dấu) để tránh miss accent variants
# Mỗi rule là list các keyword — TẤT CẢ phải xuất hiện trong câu (AND logic)
_DANGEROUS_RULES: list[list[str]] = [
    # Mua/bán/tìm nguồn
    ["mua", "ma tuy"],
    ["ban", "ma tuy"],
    ["kiem", "ma tuy"],
    ["lay", "ma tuy"],
    ["tim nguon", "ma tuy"],
    ["mua", "heroin"],
    ["mua", "cocaine"],
    ["mua", "can sa"],
    ["mua", "thuoc lac"],
    ["dealer"],
    ["lien he", "ma tuy"],
    ["so dien thoai", "ma tuy"],
    # Chế tạo / tổng hợp
    ["cach lam", "ma tuy"],
    ["cong thuc", "ma tuy"],
    ["tong hop", "ma tuy"],
    ["che bien", "ma tuy"],
    ["san xuat", "ma tuy"],
    ["nau", "heroin"],
    ["nau", "ma tuy"],
    ["trong", "thuoc phien"],
    ["chiet xuat", "ma tuy"],
    # Hướng dẫn sử dụng / liều lượng
    ["cach dung", "ma tuy"],
    ["cach su dung", "ma tuy"],
    ["lieu luong", "ma tuy"],
    ["lieu", "heroin"],
    ["inject", "heroin"],
    ["chich", "ma tuy"],
    ["hut", "ma tuy"],
    ["phe", "ma tuy"],
    ["suong", "ma tuy"],
    # Trốn tránh pháp luật / xét nghiệm
    ["qua mat", "xet nghiem"],
    ["am tinh", "xet nghiem"],
    ["khong bi phat hien", "ma tuy"],
    ["tron", "cong an"],
    # Buôn lậu / vận chuyển trái phép
    ["van chuyen lau"],
    ["buon lau", "ma tuy"],
    ["giau", "ma tuy"],
    ["che giau", "ma tuy"],
    # Rửa tiền
    ["rua tien", "ma tuy"],
]


def _normalize_vi(text: str) -> str:
    """Bỏ dấu tiếng Việt để match robust hơn."""
    import unicodedata
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# Whitelist: nếu câu hỏi chứa từ khóa pháp lý → pass dù có từ nhạy cảm
# Ví dụ: "Tội sản xuất ma tuý bị phạt ra sao?" là câu hỏi pháp lý hợp lệ
_LEGAL_INTENT_KEYWORDS = [
    "toi ", "hinh phat", "bi phat", "xu ly", "phap luat", "phap ly",
    "quy dinh", "dieu ", "khoan ", "bo luat", "luat ", "nghi dinh",
    "truy to", "xu phat", "canh cao", "canh cao", "thu tuc", "co so phap ly",
    "bi bat", "bi xu", "nguoi pham toi", "hanh vi", "toi danh",
    "khai niem", "dinh nghia", "theo quy dinh",
]


def safety_check(query: str) -> tuple[bool, str]:
    """
    Kiểm tra câu hỏi có vi phạm an toàn không.
    1. Nếu chứa từ khóa pháp lý rõ ràng → pass (câu hỏi luật)
    2. Nếu match dangerous rule → block
    Returns: (is_safe, refusal_message)
    """
    q_norm = _normalize_vi(query)

    # Bước 1: whitelist — câu hỏi về luật/hình phạt luôn được phép
    if any(kw in q_norm for kw in _LEGAL_INTENT_KEYWORDS):
        return True, ""

    # Bước 2: kiểm tra dangerous patterns
    for rule in _DANGEROUS_RULES:
        if all(kw in q_norm for kw in rule):
            return False, _SAFE_REFUSAL

    return True, ""


def answer_with_memory(query: str, history: list[dict], top_k: int, threshold: float,
                       use_hyde: bool = False) -> dict:
    """
    Trả lời câu hỏi có tính đến lịch sử hội thoại.
    Nếu query là follow-up (ngắn, dùng đại từ), ghép context lịch sử vào query.
    """
    retrieve, generate_with_citation, reorder_for_llm, format_context = load_pipeline()

    # Phát hiện follow-up question
    follow_up_signals = ["nó", "họ", "điều đó", "vụ này", "trường hợp đó", "còn", "thế còn", "vậy"]
    is_followup = len(query.split()) < 8 or any(s in query.lower() for s in follow_up_signals)

    enriched_query = query
    if is_followup and history:
        history_ctx = build_context_from_history(history, window=2)
        enriched_query = f"[Ngữ cảnh hội thoại trước:\n{history_ctx}\n]\n\nCâu hỏi hiện tại: {query}"

    # Retrieve — HyDE hoặc thường
    if use_hyde:
        from src.hyde import hyde_retrieve
        openai_key = os.getenv("OPENAI_API_KEY")
        chunks = hyde_retrieve(enriched_query, top_k=top_k, score_threshold=threshold,
                               openai_key=openai_key)
    else:
        from src.task9_retrieval_pipeline import retrieve as _retrieve
        import src.task9_retrieval_pipeline as t9
        t9.SCORE_THRESHOLD = threshold
        chunks = _retrieve(enriched_query, top_k=top_k, score_threshold=threshold)

    if not chunks:
        return {
            "answer": "Tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Build context + generate
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        # Fallback: trả lời dựa trên context không qua LLM
        answer = "⚠️ Chưa có OPENAI_API_KEY. Context đã tìm được:\n\n"
        for i, c in enumerate(chunks[:3], 1):
            src = c.get("metadata", {}).get("source", "?")
            answer += f"**[{i}] {src}:**\n{c['content'][:300]}...\n\n"
        return {"answer": answer, "sources": chunks, "retrieval_source": chunks[0].get("source", "hybrid")}

    try:
        from openai import OpenAI
        from src.task10_generation import SYSTEM_PROMPT, TEMPERATURE, TOP_P
        client = OpenAI(api_key=openai_key)

        # Thêm lịch sử vào messages nếu có
        messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in history[-4:]:  # 2 cặp Q/A gần nhất
            messages_payload.append({"role": m["role"], "content": m["content"][:500]})
        messages_payload.append({
            "role": "user",
            "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}"
        })

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_payload,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"Lỗi khi gọi LLM: {e}\n\n**Context tìm được:**\n{context[:600]}..."

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }

# ──────────────────────────────────────────────
# INPUT & RESPONSE
# ──────────────────────────────────────────────
# Luôn render st.chat_input để khung chat không biến mất
chat_typed = st.chat_input("Hỏi về pháp luật ma tuý hoặc tin tức liên quan...")

# Ưu tiên: suggestion button → chat input
if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    user_input = chat_typed

if user_input:
    # Hiển thị tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Safety check — từ chối ngay nếu query vi phạm
    is_safe, refusal_msg = safety_check(user_input)
    if not is_safe:
        with st.chat_message("assistant"):
            st.warning(refusal_msg)
        st.session_state.messages.append({"role": "assistant", "content": refusal_msg})
        st.session_state.sources_history.append([])
        st.stop()

    # Sinh câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu và tổng hợp..."):
            try:
                result = answer_with_memory(
                    user_input,
                    st.session_state.messages[:-1],  # lịch sử không gồm câu vừa hỏi
                    top_k=top_k,
                    threshold=score_threshold,
                    use_hyde=use_hyde,
                )
                answer = result["answer"]
                sources = result["sources"]
                ret_src = result["retrieval_source"]
            except Exception as e:
                import traceback
                answer = f"❌ Lỗi pipeline: {e}"
                sources = []
                ret_src = "error"
                st.code(traceback.format_exc())

        st.markdown(answer)

        # Badge nguồn
        badge = "🔵 hybrid" if ret_src == "hybrid" else "🟡 pageindex" if ret_src == "pageindex" else "⚪ none"
        hyde_tag = " · 🧪 HyDE" if use_hyde else ""
        st.caption(f"Nguồn retrieval: {badge} · {len(sources)} chunks{hyde_tag}")
        # Hiển thị hypothetical document khi dùng HyDE
        if use_hyde and sources and sources[0].get("hyde_query"):
            with st.expander("🧪 Hypothetical document (HyDE)", expanded=False):
                st.caption(sources[0]["hyde_query"])

        # Source documents
        if show_sources and sources:
            with st.expander(f"📚 {len(sources)} nguồn tham khảo", expanded=False):
                for j, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    fname = meta.get("source", "?")
                    stype = meta.get("type", "")
                    score = src.get("score", 0)
                    label = f"**[{j}]** `{fname}`"
                    if stype:
                        label += f" · *{stype}*"
                    if show_scores:
                        label += f" · score `{score:.4f}`"
                    st.markdown(label)
                    st.caption(src.get("content", "")[:200] + "...")

    # Lưu vào session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    # Lưu sources theo index của cặp Q/A (assistant turn index // 2)
    st.session_state.sources_history.append(sources)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption("VinUniversity AICB-P1 Cohort 2 · Day 8 RAG Pipeline · Group Project")
