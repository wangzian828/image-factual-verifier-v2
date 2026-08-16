"""Baidu general OCR client with in-memory access-token caching."""

from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests


DEFAULT_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
DEFAULT_OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
DEFAULT_TIMEOUT_SECONDS = 30.0
TOKEN_CACHE_SKEW_SECONDS = 60.0

_TOKEN_CACHE: dict[tuple[str, str, str], tuple[str, float]] = {}
_TOKEN_CACHE_LOCK = threading.RLock()


class BaiduOCRError(RuntimeError):
    """A bounded Baidu OCR provider failure."""

    def __init__(self, message: str, *, request_count: int = 0):
        super().__init__(message)
        self.request_count = max(0, int(request_count))


@dataclass
class BaiduOCRClient:
    """Call Baidu's general OCR endpoint without persisting credentials."""

    api_key: str = ""
    secret_key: str = ""
    token_url: str = ""
    ocr_url: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    session: Any = field(default=None, repr=False)
    _session_local: threading.local = field(
        default_factory=threading.local,
        init=False,
        repr=False,
    )

    def _api_key(self) -> str:
        return (self.api_key or os.getenv("BAIDU_OCR_API_KEY", "")).strip()

    def _secret_key(self) -> str:
        return (self.secret_key or os.getenv("BAIDU_OCR_SECRET_KEY", "")).strip()

    def _token_url(self) -> str:
        return (
            self.token_url.strip()
            or os.getenv("BAIDU_OCR_TOKEN_URL", "").strip()
            or DEFAULT_TOKEN_URL
        )

    def _ocr_url(self) -> str:
        return (
            self.ocr_url.strip()
            or os.getenv("BAIDU_OCR_URL", "").strip()
            or DEFAULT_OCR_URL
        )

    def _timeout(self) -> float:
        raw = os.getenv(
            "BAIDU_OCR_TIMEOUT_SECONDS",
            str(self.timeout_seconds),
        ).strip()
        try:
            return max(1.0, float(raw))
        except ValueError:
            return DEFAULT_TIMEOUT_SECONDS

    def _session(self) -> Any:
        if self.session is not None:
            return self.session
        current = getattr(self._session_local, "session", None)
        if current is None:
            current = requests.Session()
            self._session_local.session = current
        return current

    @staticmethod
    def _payload(response: Any) -> Dict[str, Any]:
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise BaiduOCRError(
                "Baidu OCR returned non-JSON response",
                request_count=1,
            ) from exc
        if not isinstance(payload, dict):
            raise BaiduOCRError(
                "Baidu OCR returned a non-object JSON response",
                request_count=1,
            )
        return payload

    @staticmethod
    def _provider_error(payload: Dict[str, Any], *, stage: str) -> BaiduOCRError:
        code = payload.get("error_code", payload.get("error"))
        message = str(
            payload.get("error_msg")
            or payload.get("error_description")
            or "unknown provider error"
        ).strip()
        suffix = f" ({code})" if code not in (None, "") else ""
        return BaiduOCRError(
            f"Baidu {stage} failed{suffix}: {message[:300]}",
            request_count=1,
        )

    def get_access_token(self) -> tuple[str, bool]:
        """Return ``(token, cache_hit)`` without exposing the token in errors."""

        direct_token = os.getenv("BAIDU_OCR_ACCESS_TOKEN", "").strip()
        if direct_token:
            return direct_token, True

        api_key = self._api_key()
        secret_key = self._secret_key()
        if not api_key or not secret_key:
            raise BaiduOCRError(
                "BAIDU_OCR_API_KEY and BAIDU_OCR_SECRET_KEY are required"
            )

        cache_key = (api_key, secret_key, self._token_url())
        now = time.monotonic()
        with _TOKEN_CACHE_LOCK:
            cached = _TOKEN_CACHE.get(cache_key)
            if cached and cached[1] > now:
                return cached[0], True

            try:
                response = self._session().post(
                    self._token_url(),
                    params={
                        "grant_type": "client_credentials",
                        "client_id": api_key,
                        "client_secret": secret_key,
                    },
                    timeout=self._timeout(),
                )
            except requests.RequestException as exc:
                raise BaiduOCRError(
                    f"Baidu token request failed: {type(exc).__name__}",
                    request_count=1,
                ) from exc

            payload = self._payload(response)
            if not bool(getattr(response, "ok", False)):
                raise self._provider_error(payload, stage="token request")
            if payload.get("error") or payload.get("error_code"):
                raise self._provider_error(payload, stage="token request")

            token = str(payload.get("access_token", "")).strip()
            if not token:
                raise BaiduOCRError(
                    "Baidu token response lacks access_token",
                    request_count=1,
                )
            try:
                expires_in = float(payload.get("expires_in", 0))
            except (TypeError, ValueError):
                expires_in = 0.0
            ttl = max(60.0, expires_in - TOKEN_CACHE_SKEW_SECONDS)
            _TOKEN_CACHE[cache_key] = (token, now + ttl)
            return token, False

    def recognize(
        self,
        image_bytes: bytes,
        *,
        access_token: str,
        language_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not image_bytes:
            raise BaiduOCRError("Baidu OCR input image is empty")
        try:
            response = self._session().post(
                self._ocr_url(),
                params={"access_token": access_token},
                data={
                    "image": base64.b64encode(image_bytes).decode("ascii"),
                    "language_type": (
                        language_type
                        or os.getenv("BAIDU_OCR_LANGUAGE_TYPE", "CHN_ENG")
                    ),
                    "detect_direction": os.getenv(
                        "BAIDU_OCR_DETECT_DIRECTION",
                        "false",
                    ),
                    "detect_language": os.getenv(
                        "BAIDU_OCR_DETECT_LANGUAGE",
                        "false",
                    ),
                    "paragraph": os.getenv("BAIDU_OCR_PARAGRAPH", "false"),
                    "vertexes_location": "true",
                    "probability": "true",
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=self._timeout(),
            )
        except requests.RequestException as exc:
            raise BaiduOCRError(
                f"Baidu OCR request failed: {type(exc).__name__}",
                request_count=1,
            ) from exc

        payload = self._payload(response)
        if not bool(getattr(response, "ok", False)):
            raise self._provider_error(payload, stage="OCR request")
        if payload.get("error") or payload.get("error_code"):
            raise self._provider_error(payload, stage="OCR request")
        if not isinstance(payload.get("words_result", []), list):
            raise BaiduOCRError(
                "Baidu OCR response has invalid words_result",
                request_count=1,
            )
        return payload
