from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


SERPER_TEXT_ENDPOINT = "https://google.serper.dev/search"
SERPER_IMAGE_ENDPOINT = "https://google.serper.dev/images"
SERPER_LENS_ENDPOINT = "https://google.serper.dev/lens"


def _get_proxies() -> Optional[Dict[str, str]]:
    """Get proxy settings from environment."""
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
    )
    return {"http": proxy, "https": proxy} if proxy else None


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text or "")


def default_gl(query: str, gl: Optional[str]) -> str:
    if gl:
        return gl
    return "cn" if contains_cjk(query) else "us"


def default_hl(query: str, gl: Optional[str], hl: Optional[str]) -> str:
    if hl:
        return hl
    return "zh-cn" if default_gl(query, gl) == "cn" else "en"


@dataclass
class SerperTextSearchClient:
    api_key: Optional[str] = None
    endpoint: str = SERPER_TEXT_ENDPOINT
    timeout: int = 20
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY_ID")
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def search(self, query: str, *, top_k: int = 10, gl: Optional[str] = None, hl: Optional[str] = None, time_range: Optional[str] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY is not set. Put it in your environment or .env before running search.")

        country = default_gl(query, gl)
        language = default_hl(query, country, hl)
        payload = {"q": query, "gl": country, "hl": language, "num": top_k}
        # time_range: "qdr:d" (day), "qdr:w" (week), "qdr:m" (month), "qdr:y" (year)
        if time_range:
            payload["tbs"] = time_range
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        raw = self._post_json(payload, headers)
        organic = raw.get("organic", [])
        results = [self._normalize_result(query, item, idx + 1) for idx, item in enumerate(organic[:top_k])]
        return {
            "query": query,
            "provider": "serper",
            "gl": country,
            "hl": language,
            "results": results,
            "answer_box": raw.get("answerBox"),
            "knowledge_graph": raw.get("knowledgeGraph"),
            "credits_used": raw.get("credits"),
        }

    def _post_json(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_session().post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    proxies=_get_proxies(),
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("SerperTextSearchClient failed without a captured exception.")

    @staticmethod
    def _normalize_result(query: str, item: Dict[str, Any], rank: int) -> Dict[str, Any]:
        return {
            "query": query,
            "rank": rank,
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "source": item.get("source", ""),
            "date": item.get("date"),
        }


@dataclass
class SerperImageSearchClient:
    api_key: Optional[str] = None
    endpoint: str = SERPER_IMAGE_ENDPOINT
    timeout: int = 20
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY_ID")
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def search(self, query: str, *, top_k: int = 5, gl: Optional[str] = None, hl: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY is not set. Add it to the environment before using image search.")

        country = default_gl(query, gl)
        language = default_hl(query, country, hl)
        payload = {"q": query, "gl": country, "hl": language, "num": top_k}
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        raw = self._post_json(payload, headers)
        images = raw.get("images", [])
        return [self._normalize_result(query, item, idx + 1) for idx, item in enumerate(images[:top_k])]

    def _post_json(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_session().post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    proxies=_get_proxies(),
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("SerperImageSearchClient failed without a captured exception.")

    @staticmethod
    def _normalize_result(query: str, item: Dict[str, Any], rank: int) -> Dict[str, Any]:
        return {
            "query": query,
            "rank": rank,
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "image_url": item.get("imageUrl", ""),
            "source": item.get("source", ""),
        }


@dataclass
class SerperLensSearchClient:
    api_key: Optional[str] = None
    endpoint: str = SERPER_LENS_ENDPOINT
    timeout: int = 25
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY_ID")
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def search(self, image_url: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY is not set. Add it to the environment before using lens search.")
        if not image_url:
            raise ValueError("image_url is required for lens search.")

        payload = {"url": image_url}
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        raw = self._post_json(payload, headers)
        organic = raw.get("organic", [])
        return [self._normalize_result(image_url, item, idx + 1) for idx, item in enumerate(organic[:top_k])]

    def _post_json(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_session().post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    proxies=_get_proxies(),
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("SerperLensSearchClient failed without a captured exception.")

    @staticmethod
    def _normalize_result(image_url: str, item: Dict[str, Any], rank: int) -> Dict[str, Any]:
        return {
            "query": image_url,
            "rank": rank,
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "image_url": item.get("imageUrl", "") or item.get("thumbnailUrl", ""),
            "source": item.get("source", ""),
            "snippet": item.get("snippet", ""),
            "provider": "serper_lens",
        }
