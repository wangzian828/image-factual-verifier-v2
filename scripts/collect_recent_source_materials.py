from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from src.storage import data_path


load_dotenv()


DEFAULT_OUTPUT_DIR = data_path(
    "artifacts/source_materials/recent_official",
    "data/source_materials/recent_official",
)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 30


@dataclass(frozen=True)
class FeedConfig:
    source_id: str
    feed_url: str
    source_type: str
    license_note: str
    page_strategy: str = "generic"


FEEDS: List[FeedConfig] = [
    FeedConfig(
        source_id="nasa_iotd",
        feed_url="https://www.nasa.gov/feeds/iotd-feed/",
        source_type="official",
        license_note="NASA imagery is generally available for informational and educational use; review item-specific restrictions.",
        page_strategy="nasa",
    ),
    FeedConfig(
        source_id="nasa_main",
        feed_url="https://www.nasa.gov/feed/",
        source_type="official",
        license_note="NASA imagery is generally available for informational and educational use; review item-specific restrictions.",
        page_strategy="nasa",
    ),
    FeedConfig(
        source_id="nasa_photojournal",
        feed_url="https://science.nasa.gov/feed/photojournal/latest-content/",
        source_type="official",
        license_note="NASA imagery is generally available for informational and educational use; review item-specific restrictions.",
        page_strategy="nasa",
    ),
    FeedConfig(
        source_id="esa_space",
        feed_url="https://www.esa.int/rssfeed/Our_Activities/Space_News",
        source_type="official",
        license_note="ESA media may be reused according to ESA terms; review item-specific restrictions before redistribution.",
        page_strategy="esa",
    ),
    FeedConfig(
        source_id="esa_earth",
        feed_url="https://www.esa.int/rssfeed/Our_Activities/Observing_the_Earth",
        source_type="official",
        license_note="ESA media may be reused according to ESA terms; review item-specific restrictions before redistribution.",
        page_strategy="esa",
    ),
    FeedConfig(
        source_id="noaa_main",
        feed_url="https://www.noaa.gov/rss.xml",
        source_type="official",
        license_note="NOAA images are generally not copyrighted, but confirm any third-party restrictions noted on the item.",
        page_strategy="noaa",
    ),
    FeedConfig(
        source_id="cdc_newsroom",
        feed_url="https://tools.cdc.gov/api/v2/resources/media/132608.rss",
        source_type="official",
        license_note="CDC content is generally public domain unless otherwise noted; confirm any third-party restrictions on specific assets.",
        page_strategy="generic",
    ),
    FeedConfig(
        source_id="ec_press",
        feed_url="https://ec.europa.eu/commission/presscorner/api/rss?language=en",
        source_type="official",
        license_note="European Commission press materials may have reuse conditions; review item-specific restrictions before redistribution.",
        page_strategy="generic",
    ),
    FeedConfig(
        source_id="un_news",
        feed_url="https://news.un.org/feed/subscribe/en/news/all/rss.xml",
        source_type="official",
        license_note="UN News content may have reuse conditions; review the source page before redistribution.",
        page_strategy="generic",
    ),
    FeedConfig(
        source_id="apple_newsroom",
        feed_url="https://www.apple.com/newsroom/rss-feed.rss",
        source_type="brand_official",
        license_note="Apple Newsroom images and materials are subject to Apple media usage terms.",
        page_strategy="generic",
    ),
    FeedConfig(
        source_id="nhc_atlantic",
        feed_url="https://www.nhc.noaa.gov/index-at.xml",
        source_type="official",
        license_note="NOAA/NHC content is generally public domain unless otherwise noted.",
        page_strategy="generic",
    ),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch collect recent official source materials with text and images."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store metadata, HTML snapshots, and images.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="Keep items published within this many days from now.",
    )
    parser.add_argument(
        "--per-feed-limit",
        type=int,
        default=12,
        help="Maximum number of recent items to keep from each feed.",
    )
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated source ids to include, or 'all'.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Collect metadata and HTML only, without downloading images.",
    )
    return parser.parse_args()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    if not text:
        return None

    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass

    iso_text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clean_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    return text


def _slugify(value: str, fallback: str = "item") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return slug or fallback


def _hash_text(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return _clean_text(soup.get_text(" ", strip=True))


def _extract_first_image_from_html(html: str) -> Optional[str]:
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    img = soup.find("img")
    if not img:
        return None
    src = (img.get("src") or "").strip()
    return src or None


def _iter_feeds(selected_sources: str) -> Iterable[FeedConfig]:
    selected = {part.strip() for part in selected_sources.split(",") if part.strip()}
    if not selected or selected == {"all"}:
        yield from FEEDS
        return

    for config in FEEDS:
        if config.source_id in selected:
            yield config


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


def _parse_feed_items(content: bytes) -> List[BeautifulSoup]:
    soup = BeautifulSoup(content, "xml")
    items = list(soup.find_all("item"))
    if items:
        return items
    return list(soup.find_all("entry"))


def _feed_node_text(node: BeautifulSoup, *names: str) -> str:
    for name in names:
        tag = node.find(name)
        if tag:
            if name == "link" and tag.get("href"):
                return _clean_text(tag.get("href"))
            text = tag.get_text(" ", strip=True)
            if text:
                return _clean_text(text)
    return ""


def _feed_image_url(node: BeautifulSoup, description_html: str) -> Optional[str]:
    media_content = node.find("content")
    if media_content and media_content.get("url"):
        return media_content.get("url", "").strip() or None

    media_thumbnail = node.find("thumbnail")
    if media_thumbnail and media_thumbnail.get("url"):
        return media_thumbnail.get("url", "").strip() or None

    link_tag = node.find("link", rel="enclosure")
    if link_tag and link_tag.get("href"):
        href = link_tag.get("href", "").strip()
        if href:
            return href

    return _extract_first_image_from_html(description_html)


def _safe_filename_from_url(url: str, fallback_stem: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff"}:
        suffix = ".jpg"
    return f"{fallback_stem}{suffix}"


def _collect_page_payload(
    session: requests.Session,
    config: FeedConfig,
    link: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "page_fetch_status": "skipped",
        "page_status_code": None,
        "final_url": link,
        "raw_html": None,
        "page_text": "",
        "image_url": None,
    }

    try:
        response = session.get(link, timeout=DEFAULT_TIMEOUT)
    except Exception as exc:
        result["page_fetch_status"] = "error"
        result["fetch_error"] = repr(exc)
        return result

    result["page_status_code"] = response.status_code
    result["final_url"] = response.url

    if response.status_code != 200:
        result["page_fetch_status"] = "http_error"
        return result

    result["page_fetch_status"] = "ok"
    result["raw_html"] = response.text

    soup = BeautifulSoup(response.text, "html.parser")
    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        result["image_url"] = og_image["content"].strip()

    if config.page_strategy == "nasa":
        text = _extract_text_from_nasa_page(soup)
    elif config.page_strategy == "esa":
        text = _extract_text_from_esa_page(soup)
    else:
        text = _extract_text_generic(soup)

    result["page_text"] = text
    return result


def _extract_text_generic(soup: BeautifulSoup) -> str:
    paragraphs: List[str] = []
    seen = set()
    for tag in soup.find_all("p"):
        text = _clean_text(tag.get_text(" ", strip=True))
        if len(text) < 40:
            continue
        if text in seen:
            continue
        seen.add(text)
        paragraphs.append(text)
    return "\n\n".join(paragraphs[:12])


def _extract_text_from_nasa_page(soup: BeautifulSoup) -> str:
    blocked_prefixes = (
        "Read More",
        "NASA explores the unknown",
        "For media inquiries",
    )
    paragraphs: List[str] = []
    seen = set()
    for tag in soup.find_all("p"):
        text = _clean_text(tag.get_text(" ", strip=True))
        if len(text) < 40:
            continue
        if text.startswith(blocked_prefixes):
            continue
        if text in seen:
            continue
        seen.add(text)
        paragraphs.append(text)
    return "\n\n".join(paragraphs[:12])


def _extract_text_from_esa_page(soup: BeautifulSoup) -> str:
    blocked_substrings = (
        "Thank you for liking",
        "You have already liked this page",
        "Share this",
    )
    paragraphs: List[str] = []
    seen = set()
    for tag in soup.find_all("p"):
        text = _clean_text(tag.get_text(" ", strip=True))
        if len(text) < 30:
            continue
        if any(token in text for token in blocked_substrings):
            continue
        if text in seen:
            continue
        seen.add(text)
        paragraphs.append(text)
    return "\n\n".join(paragraphs[:12])


def _download_image(
    session: requests.Session,
    image_url: str,
    destination: Path,
) -> Optional[int]:
    try:
        response = session.get(image_url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
    except Exception:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return len(response.content)


def collect_recent_materials(
    output_dir: Path,
    *,
    days_back: int,
    per_feed_limit: int,
    selected_sources: str,
    skip_images: bool,
) -> Dict[str, Any]:
    now_utc = _utc_now()
    cutoff = now_utc - timedelta(days=days_back)
    session = _build_session()

    metadata_dir = output_dir / "metadata"
    html_dir = output_dir / "raw_html"
    image_dir = output_dir / "images"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    seen_links = set()

    for config in _iter_feeds(selected_sources):
        response = session.get(config.feed_url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        items = _parse_feed_items(response.content)

        kept_for_feed = 0
        for item in items:
            if kept_for_feed >= per_feed_limit:
                break

            title = _feed_node_text(item, "title")
            link = _feed_node_text(item, "link", "id", "guid")
            if not title or not link or link in seen_links:
                continue

            date_text = _feed_node_text(item, "pubDate", "updated", "published", "dc:date")

            published_at = _parse_datetime(date_text)
            if not published_at or published_at < cutoff or published_at > now_utc:
                continue

            description_html = _feed_node_text(item, "description", "summary", "content", "content:encoded")

            description_text = _strip_html(description_html)
            feed_image_url = _feed_image_url(item, description_html)
            page_payload = _collect_page_payload(session, config, link)

            sample_id = (
                f"{config.source_id}_{published_at.strftime('%Y%m%d')}_"
                f"{_slugify(title, fallback='item')[:48]}"
            )
            if len(sample_id) > 96:
                sample_id = f"{sample_id[:85]}-{_hash_text(link)}"

            html_path = metadata_dir / f"{sample_id}.json"
            raw_html_path = html_dir / f"{sample_id}.html"
            if page_payload["raw_html"]:
                raw_html_path.write_text(page_payload["raw_html"], encoding="utf-8")
                raw_html_rel = str(raw_html_path)
            else:
                raw_html_rel = None

            image_url = page_payload.get("image_url") or feed_image_url
            local_image_path = None
            image_bytes = None
            if image_url and not skip_images:
                image_name = _safe_filename_from_url(image_url, sample_id)
                image_path = image_dir / image_name
                image_bytes = _download_image(session, image_url, image_path)
                if image_bytes is not None:
                    local_image_path = str(image_path)

            page_text = page_payload.get("page_text") or ""
            if not page_text:
                page_text = description_text

            record = {
                "sample_id": sample_id,
                "title": title,
                "source_id": config.source_id,
                "source_type": config.source_type,
                "license_note": config.license_note,
                "source_url": link,
                "final_url": page_payload.get("final_url", link),
                "feed_url": config.feed_url,
                "publish_date": published_at.isoformat(),
                "collected_at": now_utc.isoformat(),
                "feed_description_text": description_text,
                "page_text": page_text,
                "image_url": image_url,
                "local_image_path": local_image_path,
                "raw_html_path": raw_html_rel,
                "page_fetch_status": page_payload.get("page_fetch_status"),
                "page_status_code": page_payload.get("page_status_code"),
                "tags": [config.source_id, "recent_official"],
            }
            html_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_rows.append(record)
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "source_id": config.source_id,
                    "publish_date": published_at.isoformat(),
                    "title": title,
                    "source_url": link,
                    "image_downloaded": bool(local_image_path),
                    "page_fetch_status": page_payload.get("page_fetch_status"),
                }
            )
            kept_for_feed += 1
            seen_links.add(link)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "generated_at": now_utc.isoformat(),
        "days_back": days_back,
        "per_feed_limit": per_feed_limit,
        "selected_sources": selected_sources,
        "num_items": len(manifest_rows),
        "items": summary_rows,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "num_items": len(manifest_rows),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
    }


def main() -> None:
    args = _parse_args()
    report = collect_recent_materials(
        Path(args.output_dir),
        days_back=args.days_back,
        per_feed_limit=args.per_feed_limit,
        selected_sources=args.sources,
        skip_images=args.skip_images,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
