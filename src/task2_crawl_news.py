"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI để tự động render JS và lấy nội dung.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    crawl4ai-setup   # cài Playwright browsers (chỉ cần chạy 1 lần)
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài báo về nghệ sĩ VN liên quan ma tuý
ARTICLE_URLS = [
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-cung-tro-ly-lam-tiec-ma-tuy-trong-can-ho-cao-cap-5059429.html",
    "https://vnexpress.net/rapper-binh-gold-tiep-tuc-duong-tinh-voi-ma-tuy-lai-cuop-taxi-4919259.html",
]

# CSS selector để chỉ lấy nội dung bài báo (bỏ nav/footer/ads)
# Key: domain, Value: selector phần thân bài
SITE_SELECTORS = {
    "vnexpress.net": "article.sidebar, .fck_detail, .content-detail",
    "tuoitre.vn": ".detail-content, .main-content-body",
    "thanhnien.vn": ".detail-content, #abody",
}


def slugify(url: str) -> str:
    """Tạo tên file an toàn từ URL."""
    name = re.sub(r"https?://[^/]+/", "", url)
    name = re.sub(r"[^a-z0-9]+", "-", name.lower())
    return name[:80].strip("-")


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo bằng Crawl4AI và trả về dict chứa metadata + content.

    Crawl4AI tự động:
    - Render JavaScript (dùng Playwright headless browser)
    - Chuyển HTML → Markdown sạch (loại nav, ads, footer)
    - Trích metadata title, description, ...

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str,        # ISO 8601
            "content_markdown": str,    # nội dung bài báo dạng Markdown
            "word_count": int
        }
    """
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

    from urllib.parse import urlparse
    domain = urlparse(url).netloc.replace("www.", "")
    css_selector = SITE_SELECTORS.get(domain)

    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(
        word_count_threshold=50,
        remove_overlay_elements=True,
        exclude_social_media_links=True,
        excluded_tags=["nav", "footer", "aside", "header"],
        css_selector=css_selector,       # chỉ lấy phần thân bài báo
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

    if not result.success:
        raise RuntimeError(f"Crawl thất bại: {url} — {result.error_message}")

    title = (result.metadata or {}).get("title", "").strip() or url
    # crawl4ai >= 0.4: result.markdown là MarkdownGenerationResult, không còn là string
    md = result.markdown
    if hasattr(md, "fit_markdown"):
        content = md.fit_markdown or md.raw_markdown or ""
    else:
        content = str(md) if md else ""

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
        "word_count": len(content.split()),
    }


async def crawl_all(urls: list[str] = None, delay: float = 1.5):
    """
    Crawl toàn bộ bài báo trong danh sách, lưu mỗi bài thành 1 file JSON.

    Args:
        urls:  danh sách URL (mặc định dùng ARTICLE_URLS)
        delay: giây nghỉ giữa các request để tránh bị block
    """
    setup_directory()
    urls = urls or ARTICLE_URLS

    success, failed = 0, []

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Crawling: {url}")
        try:
            article = await crawl_article(url)

            filename = f"{i:02d}_{slugify(url)}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  ✓ Saved: {filepath.name}  ({article['word_count']} words)")
            success += 1

        except Exception as e:
            print(f"  ✗ Lỗi: {e}")
            failed.append({"url": url, "error": str(e)})

        if i < len(urls):
            await asyncio.sleep(delay)   # polite delay

    print(f"\n✅ Hoàn thành: {success}/{len(urls)} bài")
    if failed:
        print(f"❌ Thất bại ({len(failed)} bài):")
        for f in failed:
            print(f"  - {f['url']}: {f['error']}")

    return success, failed


# ---------------------------------------------------------------------------
# Cách dùng thêm URL bên ngoài script:
#
#   from src.task2_crawl_news import crawl_all
#   asyncio.run(crawl_all(["https://vnexpress.net/bai-bao-1.html", ...]))
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(crawl_all())
