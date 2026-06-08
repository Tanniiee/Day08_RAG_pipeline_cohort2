"""
Kịch bản test toàn bộ RAG Pipeline — Day 8 Lab
Chạy: python test_scenarios.py
"""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
BLUE  = "\033[94m"
BOLD  = "\033[1m"
RESET = "\033[0m"

passed = failed = skipped = 0

def ok(msg):
    global passed; passed += 1
    print(f"  {GREEN}✅ PASS{RESET} {msg}")

def fail(msg, detail=""):
    global failed; failed += 1
    print(f"  {RED}❌ FAIL{RESET} {msg}")
    if detail:
        print(f"       {RED}{detail}{RESET}")

def skip(msg, reason=""):
    global skipped; skipped += 1
    print(f"  {YELLOW}⏭  SKIP{RESET} {msg} ({reason})")

def section(title):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")

def run(label, fn, *args, **kwargs):
    try:
        t0 = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        return result, elapsed
    except Exception as e:
        fail(label, str(e))
        return None, 0


# ================================================================
# SCENARIO 1: Data sanity check
# ================================================================
section("SC-01 · Data Sanity Check")

legal_dir = Path("data/standardized/legal")
news_dir  = Path("data/standardized/news")

legal_files = list(legal_dir.glob("*.md")) if legal_dir.exists() else []
news_files  = list(news_dir.glob("*.md"))  if news_dir.exists()  else []

if len(legal_files) >= 2:
    ok(f"Legal markdown: {len(legal_files)} files")
else:
    fail("Legal markdown files < 2", f"Found: {len(legal_files)}")

if len(news_files) >= 5:
    ok(f"News markdown: {len(news_files)} files")
else:
    fail("News markdown files < 5", f"Found: {len(news_files)}")

for f in legal_files + news_files:
    size = len(f.read_text(encoding="utf-8"))
    if size > 200:
        ok(f"{f.name}: {size} chars")
    else:
        fail(f"{f.name} quá ngắn", f"{size} chars < 200")


# ================================================================
# SCENARIO 2: Semantic Search
# ================================================================
section("SC-02 · Semantic Search (Task 5)")

try:
    from src.task5_semantic_search import semantic_search

    TEST_CASES = [
        # (query, expected_keyword_in_top1_content)
        ("hình phạt tàng trữ ma tuý", ["Điều", "tàng trữ", "ma tuý", "tội"]),
        ("ca sĩ bị bắt sử dụng ma tuý", ["ma túy", "ma tuý", "bắt", "ca sĩ"]),
        ("cai nghiện bắt buộc", ["cai nghiện", "bắt buộc"]),
    ]

    for query, keywords in TEST_CASES:
        results, t = run(query, semantic_search, query, top_k=5)
        if results is None:
            continue
        if not results:
            fail(f"Semantic [{query[:30]}] — trả về rỗng")
            continue

        # Check schema
        r = results[0]
        if not all(k in r for k in ["content", "score", "metadata"]):
            fail(f"Semantic schema thiếu key", str(r.keys()))
        elif not (0 <= r["score"] <= 1):
            fail(f"Score ngoài [0,1]", str(r["score"]))
        elif results != sorted(results, key=lambda x: x["score"], reverse=True):
            fail(f"Kết quả không sorted descending")
        elif len(results) > 5:
            fail(f"Trả về nhiều hơn top_k=5")
        else:
            found = any(
                any(kw.lower() in r["content"].lower() for kw in keywords)
                for r in results[:3]
            )
            if found:
                ok(f"Semantic [{query[:35]}] → {r['score']:.3f} ({t:.2f}s)")
            else:
                fail(f"Semantic [{query[:35]}] — top 3 không chứa keyword", str(keywords))

except ImportError as e:
    skip("task5_semantic_search", str(e))


# ================================================================
# SCENARIO 3: Lexical Search
# ================================================================
section("SC-03 · Lexical Search / BM25 (Task 6)")

try:
    from src.task6_lexical_search import lexical_search

    TEST_CASES = [
        ("Điều 248 tàng trữ trái phép chất ma tuý", ["Điều 248", "tàng trữ"]),
        ("ca sĩ Chi Dân bị bắt", ["Chi Dân", "bắt"]),
        ("Miu Lê ma tuý", ["Miu Lê", "ma túy", "ma tuý"]),
    ]

    for query, keywords in TEST_CASES:
        results, t = run(query, lexical_search, query, top_k=5)
        if results is None:
            continue
        if not results:
            # BM25 có thể trả rỗng nếu score = 0
            skip(f"Lexical [{query[:30]}]", "score=0 với query này")
            continue

        found = any(
            any(kw.lower() in r["content"].lower() for kw in keywords)
            for r in results[:3]
        )
        if found:
            ok(f"Lexical [{query[:35]}] → {results[0]['score']:.3f} ({t:.2f}s)")
        else:
            fail(f"Lexical [{query[:35]}] — top 3 không chứa keyword")

except ImportError as e:
    skip("task6_lexical_search", str(e))


# ================================================================
# SCENARIO 4: Reranking — RRF
# ================================================================
section("SC-04 · Reranking — RRF (Task 7)")

try:
    from src.task7_reranking import rerank, rerank_rrf

    # Dummy candidates
    candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý bị phạt tù 2-7 năm",
         "score": 0.8, "metadata": {"source": "luat.md"}},
        {"content": "Ca sĩ Chi Dân bị bắt vì sử dụng ma tuý trong nhà riêng",
         "score": 0.7, "metadata": {"source": "news.md"}},
        {"content": "Python tutorial for beginners — not relevant",
         "score": 0.2, "metadata": {"source": "other.md"}},
        {"content": "Điều 249: Tội sản xuất trái phép chất ma tuý",
         "score": 0.6, "metadata": {"source": "luat.md"}},
        {"content": "Hình phạt tù chung thân cho tội mua bán ma tuý số lượng lớn",
         "score": 0.55, "metadata": {"source": "luat.md"}},
    ]

    # Test rerank_rrf
    merged, t = run("RRF merge", rerank_rrf, [candidates[:3], candidates[2:]], top_k=3)
    if merged is not None:
        if len(merged) <= 3:
            ok(f"rerank_rrf → {len(merged)} results, top score={merged[0]['score']:.6f} ({t:.3f}s)")
        else:
            fail("rerank_rrf trả nhiều hơn top_k=3")

    # Test rerank với method="rrf"
    result, t = run("rerank(rrf)", rerank, "hình phạt tàng trữ ma tuý", candidates, 3, "rrf")
    if result is not None:
        if result[0]["content"] != candidates[2]["content"]:  # Python tutorial không nên đứng đầu
            ok(f"rerank(rrf) lọc đúng — top: '{result[0]['content'][:40]}...' ({t:.3f}s)")
        else:
            fail("rerank(rrf) — Python tutorial đứng đầu, ranking sai")

    # Test top_k respected
    result2, _ = run("rerank top_k", rerank, "ma tuý", candidates, 2, "rrf")
    if result2 is not None:
        if len(result2) <= 2:
            ok(f"rerank top_k=2 → {len(result2)} results")
        else:
            fail("rerank không respect top_k=2", f"got {len(result2)}")

except ImportError as e:
    skip("task7_reranking", str(e))


# ================================================================
# SCENARIO 5: PageIndex
# ================================================================
section("SC-05 · PageIndex Vectorless (Task 8)")

try:
    from src.task8_pageindex_vectorless import pageindex_search
    import os

    if not os.getenv("PAGEINDEX_API_KEY"):
        skip("pageindex_search", "PAGEINDEX_API_KEY chưa set")
    else:
        results, t = run("pageindex_search", pageindex_search, "ma tuý", top_k=3)
        if results is not None:
            if all(r.get("source") == "pageindex" for r in results):
                ok(f"pageindex_search → source='pageindex' ✓ ({t:.2f}s)")
            else:
                fail("pageindex_search — source không phải 'pageindex'")

except ImportError as e:
    skip("task8_pageindex_vectorless", str(e))


# ================================================================
# SCENARIO 6: Full Retrieval Pipeline
# ================================================================
section("SC-06 · Full Retrieval Pipeline (Task 9)")

try:
    from src.task9_retrieval_pipeline import retrieve

    PIPELINE_TESTS = [
        # (query, expected_source, check_content_keyword)
        ("hình phạt tội tàng trữ ma tuý Điều 248",        "hybrid", ["Điều", "ma tuý", "tàng"]),
        ("ca sĩ Miu Lê bị bắt vì ma tuý ở đảo Cát Bà",   "hybrid", ["Miu Lê", "ma túy", "bắt"]),
        ("rapper Bình Gold dương tính ma tuý",              "hybrid", ["Bình Gold", "ma túy"]),
        ("nghị định 105 cơ sở cai nghiện bắt buộc",       "hybrid", ["cai nghiện", "bắt buộc"]),
    ]

    for query, expected_source, keywords in PIPELINE_TESTS:
        results, t = run(query[:40], retrieve, query, top_k=5, score_threshold=0.3)
        if results is None:
            continue

        if not isinstance(results, list):
            fail(f"retrieve không trả list", type(results).__name__)
            continue
        if len(results) > 5:
            fail(f"retrieve trả > top_k=5", f"got {len(results)}")
            continue
        if not results:
            fail(f"retrieve trả rỗng cho '{query[:40]}'")
            continue

        # Schema check
        for r in results:
            for key in ["content", "score", "source"]:
                if key not in r:
                    fail(f"Thiếu key '{key}'", str(r.keys()))
                    break
            if r["source"] not in ("hybrid", "pageindex"):
                fail(f"source không hợp lệ: {r['source']}")

        found = any(
            any(kw.lower() in r["content"].lower() for kw in keywords)
            for r in results[:3]
        )
        if found:
            ok(f"retrieve [{query[:35]}] → {len(results)} results, top={results[0]['score']:.4f} ({t:.2f}s)")
        else:
            fail(f"retrieve [{query[:35]}] — top 3 không chứa keyword", str(keywords))

    # Test fallback: ngưỡng cao → fallback PageIndex (hoặc trả kết quả hybrid nếu không có key)
    results_fb, t = run("fallback test", retrieve, "hình phạt", top_k=3, score_threshold=0.99)
    if results_fb is not None:
        ok(f"Fallback (threshold=0.99) không crash → {len(results_fb)} results ({t:.2f}s)")

    # Test top_k respected
    results_k, _ = run("top_k=2 test", retrieve, "ma tuý", top_k=2)
    if results_k is not None:
        if len(results_k) <= 2:
            ok(f"retrieve top_k=2 → {len(results_k)} results")
        else:
            fail("retrieve không respect top_k=2", f"got {len(results_k)}")

except ImportError as e:
    skip("task9_retrieval_pipeline", str(e))


# ================================================================
# SCENARIO 7: Generation functions
# ================================================================
section("SC-07 · Generation Functions (Task 10)")

try:
    from src.task10_generation import reorder_for_llm, format_context, generate_with_citation
    import os

    # Test reorder_for_llm
    chunks = [{"content": f"Chunk {i}", "score": 1.0 - i*0.1, "metadata": {}} for i in range(5)]
    reordered, t = run("reorder_for_llm", reorder_for_llm, chunks)
    if reordered is not None:
        if len(reordered) == 5:
            ok(f"reorder_for_llm giữ đủ {len(reordered)} chunks ({t:.4f}s)")
        else:
            fail(f"reorder_for_llm mất chunk", f"got {len(reordered)}, expected 5")

        # Verify reorder thực sự thay đổi thứ tự
        if reordered != chunks:
            ok("reorder_for_llm thay đổi thứ tự (lost-in-middle strategy)")
        else:
            fail("reorder_for_llm không thay đổi thứ tự")

    # Test reorder với <= 2 chunks (edge case)
    small = [{"content": "A", "score": 0.9, "metadata": {}}, {"content": "B", "score": 0.8, "metadata": {}}]
    reordered_small, _ = run("reorder edge case (<=2)", reorder_for_llm, small)
    if reordered_small is not None:
        ok(f"reorder_for_llm edge case (2 chunks) → {len(reordered_small)} chunks")

    # Test format_context
    context, t = run("format_context", format_context, chunks[:3])
    if context is not None:
        if "Document" in context and "Source" in context:
            ok(f"format_context có 'Document' và 'Source' labels ({t:.4f}s)")
        else:
            fail("format_context thiếu Document/Source label", context[:100])
        if len(context) > 50:
            ok(f"format_context có nội dung ({len(context)} chars)")
        else:
            fail("format_context quá ngắn", f"{len(context)} chars")

    # Test generate_with_citation (không cần API key — sẽ trả fallback message)
    result, t = run("generate_with_citation", generate_with_citation, "hình phạt tàng trữ ma tuý", 3)
    if result is not None:
        if all(k in result for k in ["answer", "sources", "retrieval_source"]):
            ok(f"generate_with_citation trả đúng schema ({t:.2f}s)")
        else:
            fail("generate_with_citation thiếu key", str(result.keys()))

        if result["retrieval_source"] in ("hybrid", "pageindex", "none"):
            ok(f"retrieval_source hợp lệ: '{result['retrieval_source']}'")
        else:
            fail("retrieval_source không hợp lệ", result["retrieval_source"])

        if isinstance(result["sources"], list):
            ok(f"sources là list ({len(result['sources'])} chunks)")
        else:
            fail("sources không phải list")

        if result.get("answer"):
            ok(f"answer không rỗng ({len(result['answer'])} chars)")
        else:
            fail("answer rỗng")

except ImportError as e:
    skip("task10_generation", str(e))


# ================================================================
# SCENARIO 8: End-to-End — 3 câu hỏi thực tế
# ================================================================
section("SC-08 · End-to-End (3 câu hỏi thực tế)")

E2E_QUERIES = [
    {
        "q": "Điều 248 Bộ luật Hình sự quy định tội tàng trữ ma tuý thế nào?",
        "expect_in": ["Điều 248", "tàng trữ", "ma tuý", "phạt tù"],
        "source_type": "legal",
    },
    {
        "q": "Ca sĩ nào trong ngành giải trí bị bắt vì liên quan ma tuý?",
        "expect_in": ["ca sĩ", "bắt", "ma túy", "ma tuý"],
        "source_type": "news",
    },
    {
        "q": "Cai nghiện bắt buộc kéo dài bao nhiêu tháng theo Nghị định 105?",
        "expect_in": ["cai nghiện", "tháng", "bắt buộc"],
        "source_type": "legal",
    },
]

try:
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import format_context, reorder_for_llm

    for case in E2E_QUERIES:
        t0 = time.time()
        results = retrieve(case["q"], top_k=5)
        chunks  = reorder_for_llm(results)
        context = format_context(chunks)
        elapsed = time.time() - t0

        # Check nội dung liên quan
        all_content = " ".join(r["content"].lower() for r in results[:3])
        found_kw = [kw for kw in case["expect_in"] if kw.lower() in all_content]

        if len(found_kw) >= 1:
            ok(f"E2E [{case['q'][:45]}...] → {len(results)} chunks, keywords: {found_kw} ({elapsed:.2f}s)")
        else:
            fail(f"E2E [{case['q'][:45]}...] — không tìm thấy keywords", str(case["expect_in"]))

        # Check source type trong top results
        types = [r.get("metadata", {}).get("type", "") for r in results[:3]]
        if case["source_type"] in types:
            ok(f"  Source type '{case['source_type']}' xuất hiện trong top 3")
        else:
            skip(f"  Source type '{case['source_type']}'", f"got {types}")

except ImportError as e:
    skip("E2E test", str(e))
except Exception as e:
    fail("E2E test crashed", str(e))
    traceback.print_exc()


# ================================================================
# FINAL SUMMARY
# ================================================================
total = passed + failed + skipped
print(f"\n{BOLD}{'='*60}")
print(f"  TỔNG KẾT: {passed}/{total} PASS · {failed} FAIL · {skipped} SKIP")
print(f"{'='*60}{RESET}")
if failed == 0:
    print(f"{GREEN}{BOLD}  🎉 Tất cả test pass!{RESET}")
else:
    print(f"{RED}{BOLD}  ⚠️  {failed} test thất bại — kiểm tra lại.{RESET}")
