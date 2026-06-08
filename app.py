"""
Streamlit RAG Pipeline Demo — Day 8 Lab
Test toàn bộ flow: Semantic → Lexical → RRF → Rerank → PageIndex → Generation
"""

import sys
import time
import traceback
from pathlib import Path

import streamlit as st

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="RAG Pipeline Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 RAG Pipeline — Day 8 Lab")
st.caption("Test toàn bộ flow: Semantic → Lexical → RRF Merge → Rerank → PageIndex Fallback → Generation")

# ============================================================
# SIDEBAR — CONTROLS
# ============================================================
with st.sidebar:
    st.header("⚙️ Cấu hình Pipeline")

    top_k = st.slider("top_k (số kết quả)", min_value=1, max_value=20, value=5, step=1)

    score_threshold = st.slider(
        "Score threshold (fallback PageIndex nếu thấp hơn)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="Nếu best score < threshold → tự động fallback sang PageIndex"
    )

    rerank_method = st.selectbox(
        "Rerank method",
        options=["rrf", "cross_encoder"],
        index=0,
        help="rrf: không cần API key | cross_encoder: cần JINA_API_KEY"
    )

    use_reranking = st.toggle("Bật Reranking", value=True)

    st.divider()
    st.subheader("📋 Chạy từng bước riêng")
    show_semantic = st.checkbox("Hiển thị Semantic Search", value=True)
    show_lexical = st.checkbox("Hiển thị Lexical Search", value=True)
    show_merged = st.checkbox("Hiển thị sau RRF Merge", value=True)
    show_reranked = st.checkbox("Hiển thị sau Rerank", value=True)

    st.divider()
    st.subheader("🤖 Generation")
    enable_generation = st.checkbox("Bật Generation (cần OPENAI_API_KEY)", value=False)

    st.divider()
    # Status check
    st.subheader("📦 Module Status")
    modules = {
        "task5 semantic": "src.task5_semantic_search",
        "task6 lexical": "src.task6_lexical_search",
        "task7 rerank": "src.task7_reranking",
        "task8 pageindex": "src.task8_pageindex_vectorless",
        "task9 retrieve": "src.task9_retrieval_pipeline",
        "task10 generation": "src.task10_generation",
    }
    for label, mod in modules.items():
        try:
            __import__(mod)
            st.success(f"✅ {label}")
        except Exception as e:
            st.error(f"❌ {label}: {str(e)[:40]}")

# ============================================================
# QUERY INPUT
# ============================================================
st.subheader("💬 Câu truy vấn")

sample_queries = [
    "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
    "Ca sĩ nào bị bắt vì sử dụng ma tuý",
    "Cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021",
    "Nghị định 105 quy định gì về cơ sở cai nghiện",
    "Rapper Binz bị bắt vì ma tuý",
]

col_query, col_sample = st.columns([3, 1])
with col_sample:
    selected = st.selectbox("Hoặc chọn mẫu:", ["(tự nhập)"] + sample_queries, label_visibility="collapsed")

with col_query:
    if selected != "(tự nhập)":
        query = st.text_input("Nhập câu truy vấn:", value=selected)
    else:
        query = st.text_input("Nhập câu truy vấn:", placeholder="Ví dụ: hình phạt tàng trữ ma tuý...")

run_btn = st.button("🚀 Chạy Pipeline", type="primary", disabled=not query.strip())

# ============================================================
# HELPER
# ============================================================
def render_results(results: list[dict], title: str, color: str = "blue"):
    if not results:
        st.warning("Không có kết quả.")
        return
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        source = r.get("metadata", {}).get("source", r.get("source", "?"))
        doc_type = r.get("metadata", {}).get("type", "")
        content = r.get("content", "")
        label = f"**#{i}** · score `{score:.4f}` · `{source}`"
        if doc_type:
            label += f" · *{doc_type}*"
        with st.expander(label, expanded=(i == 1)):
            st.write(content)

def time_it(fn, *args, **kwargs):
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    return result, elapsed

# ============================================================
# PIPELINE EXECUTION
# ============================================================
if run_btn and query.strip():
    st.divider()
    st.subheader(f"🔎 Query: *{query}*")

    # ------ STEP 1: Semantic Search ------
    if show_semantic:
        with st.expander("🧠 Step 1 — Semantic Search (Dense)", expanded=True):
            try:
                from src.task5_semantic_search import semantic_search
                with st.spinner("Embedding query & searching ChromaDB..."):
                    sem_results, t = time_it(semantic_search, query, top_k=top_k * 2)
                st.caption(f"⏱ {t:.2f}s · {len(sem_results)} results")
                render_results(sem_results, "Semantic")
            except Exception as e:
                st.error(f"Lỗi: {e}")
                st.code(traceback.format_exc())

    # ------ STEP 2: Lexical Search ------
    if show_lexical:
        with st.expander("📝 Step 2 — Lexical Search (BM25)", expanded=True):
            try:
                from src.task6_lexical_search import lexical_search
                with st.spinner("BM25 scoring..."):
                    lex_results, t = time_it(lexical_search, query, top_k=top_k * 2)
                st.caption(f"⏱ {t:.2f}s · {len(lex_results)} results")
                render_results(lex_results, "Lexical")
            except Exception as e:
                st.error(f"Lỗi: {e}")
                st.code(traceback.format_exc())

    # ------ STEP 3: RRF Merge ------
    if show_merged:
        with st.expander("🔀 Step 3 — RRF Merge (Dense + Sparse)", expanded=True):
            try:
                from src.task5_semantic_search import semantic_search
                from src.task6_lexical_search import lexical_search
                from src.task7_reranking import rerank_rrf

                dense = semantic_search(query, top_k=top_k * 2)
                sparse = lexical_search(query, top_k=top_k * 2)
                merged, t = time_it(rerank_rrf, [dense, sparse], top_k=top_k * 2)
                st.caption(f"⏱ {t:.3f}s · {len(merged)} results after RRF")
                render_results(merged[:top_k], "RRF Merged")
            except Exception as e:
                st.error(f"Lỗi: {e}")
                st.code(traceback.format_exc())

    # ------ STEP 4: Rerank ------
    if show_reranked and use_reranking:
        with st.expander(f"🏆 Step 4 — Rerank ({rerank_method})", expanded=True):
            try:
                from src.task5_semantic_search import semantic_search
                from src.task6_lexical_search import lexical_search
                from src.task7_reranking import rerank, rerank_rrf

                dense = semantic_search(query, top_k=top_k * 2)
                sparse = lexical_search(query, top_k=top_k * 2)
                # Lưu semantic score để check threshold (cosine sim 0–1)
                best_semantic = dense[0]["score"] if dense else 0.0

                merged = rerank_rrf([dense, sparse], top_k=top_k * 2)
                for item in merged:
                    item["source"] = "hybrid"

                try:
                    reranked, t = time_it(rerank, query, merged, top_k, rerank_method)
                except Exception as rerank_err:
                    if "403" in str(rerank_err) or "Forbidden" in str(rerank_err):
                        st.warning(f"⚠️ `cross_encoder` lỗi 403 (Jina API key không hợp lệ) → tự động fallback về `rrf`")
                    else:
                        st.warning(f"⚠️ `{rerank_method}` lỗi: {rerank_err} → fallback về `rrf`")
                    reranked, t = time_it(rerank, query, merged, top_k, "rrf")
                st.caption(f"⏱ {t:.3f}s · {len(reranked)} results")

                if reranked:
                    # Dùng semantic score (cosine sim) để check threshold — nhất quán với task9
                    st.info(f"📊 Semantic top-1 score: `{best_semantic:.4f}` (cosine sim, dùng để so threshold)")
                    if best_semantic < score_threshold:
                        st.warning(f"⚠️ Semantic score `{best_semantic:.4f}` < threshold `{score_threshold}` → sẽ fallback PageIndex")
                    else:
                        st.success(f"✅ Semantic score `{best_semantic:.4f}` ≥ threshold `{score_threshold}` → hybrid OK")

                render_results(reranked, "Reranked")
            except Exception as e:
                st.error(f"Lỗi: {e}")
                st.code(traceback.format_exc())

    # ------ STEP 5: Full Pipeline ------
    st.divider()
    st.subheader("🎯 Full Pipeline — retrieve()")
    try:
        import src.task9_retrieval_pipeline as t9
        # Nếu cross_encoder không có API key → tự fallback về rrf
        effective_method = rerank_method
        try:
            import os
            if rerank_method == "cross_encoder" and not os.getenv("JINA_API_KEY"):
                effective_method = "rrf"
                st.info("ℹ️ JINA_API_KEY chưa set → dùng `rrf` thay cho `cross_encoder`")
        except Exception:
            pass
        t9.RERANK_METHOD = effective_method

        from src.task9_retrieval_pipeline import retrieve
        with st.spinner("Running full pipeline..."):
            final_results, t = time_it(
                retrieve, query,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=use_reranking,
            )

        sources_used = set(r.get("source", "?") for r in final_results)
        st.caption(f"⏱ {t:.2f}s · {len(final_results)} results · sources: {sources_used}")

        # Score chart
        if final_results:
            import pandas as pd
            df = pd.DataFrame([
                {
                    "rank": i + 1,
                    "score": r["score"],
                    "source_file": r.get("metadata", {}).get("source", "?")[:30],
                    "type": r.get("metadata", {}).get("type", r.get("source", "?")),
                }
                for i, r in enumerate(final_results)
            ])
            st.bar_chart(df.set_index("rank")["score"])

        render_results(final_results, "Final Results")

    except Exception as e:
        st.error(f"Lỗi pipeline: {e}")
        st.code(traceback.format_exc())

    # ------ STEP 6: Generation ------
    if enable_generation:
        st.divider()
        st.subheader("✍️ Generation với Citation")
        try:
            from src.task10_generation import generate_with_citation, reorder_for_llm, format_context
            with st.spinner("Generating answer..."):
                gen_result, t = time_it(generate_with_citation, query, top_k)

            st.caption(f"⏱ {t:.2f}s · via {gen_result.get('retrieval_source', '?')}")
            st.markdown("**Answer:**")
            st.markdown(gen_result.get("answer", "(no answer)"))

            with st.expander("📚 Sources used"):
                for i, chunk in enumerate(gen_result.get("sources", []), 1):
                    st.write(f"**[{i}]** `{chunk.get('metadata', {}).get('source', '?')}` — {chunk.get('content', '')[:150]}...")

        except Exception as e:
            st.error(f"Lỗi generation: {e}")
            st.code(traceback.format_exc())

    # ------ SUMMARY TABLE ------
    st.divider()
    st.subheader("📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    try:
        from src.task5_semantic_search import semantic_search
        from src.task6_lexical_search import lexical_search
        s = semantic_search(query, top_k=1)
        l = lexical_search(query, top_k=1)
        col1.metric("Semantic top-1", f"{s[0]['score']:.4f}" if s else "N/A")
        col2.metric("Lexical top-1", f"{l[0]['score']:.4f}" if l else "N/A")
    except:
        pass
    try:
        from src.task9_retrieval_pipeline import retrieve
        fr = retrieve(query, top_k=top_k, score_threshold=score_threshold)
        col3.metric("Final top-1", f"{fr[0]['score']:.4f}" if fr else "N/A")
        col4.metric("Source", fr[0].get("source", "?") if fr else "N/A")
    except:
        pass

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("VinUniversity AICB-P1 Cohort 2 · Day 8 RAG Pipeline Lab")
