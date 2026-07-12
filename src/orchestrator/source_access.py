"""Evaluation-only controls for preventing benchmark search-time contamination."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from src.orchestrator.source_provenance import (
    canonicalize_url,
    domain_matches,
    registered_domain,
)


POLICY_SCHEMA_VERSION = "source-access-policy-v1"
_WAYBACK_TARGET = re.compile(r"/web/(?:[^/]+/)?(https?://.+)$", re.IGNORECASE)

FACT_CHECK_DOMAIN_MARKERS = (
    "factcheck",
    "fact-check",
    "factcrescendo",
    "factly",
    "factraker",
    "fullfact",
    "leadstories",
    "checkyourfact",
    "newschecker",
    "vishvasnews",
    "youturn",
    "altnews",
    "boomlive",
    "logicallyfacts",
    "politifact",
    "snopes",
    "misbar",
    "verafiles",
    "thip.media",
    "healthfeedback",
    "sochfactcheck",
    "check4spam",
    "fakenews.pl",
)


def url_variants(value: str) -> tuple[str, ...]:
    """Return canonical outer and embedded archive URLs for policy matching."""

    raw = unquote(str(value or "").strip())
    if not raw:
        return ()
    candidates = [raw]
    try:
        parsed = urlsplit(raw)
    except ValueError:
        parsed = None
    if parsed is not None and domain_matches(parsed.hostname or "", "web.archive.org"):
        match = _WAYBACK_TARGET.search(parsed.path)
        if match:
            candidates.append(match.group(1))

    canonical: list[str] = []
    for candidate in candidates:
        normalized = canonicalize_url(candidate)
        if normalized and normalized not in canonical:
            canonical.append(normalized)
    return tuple(canonical)


def _hostname(value: str) -> str:
    try:
        return (urlsplit(value).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


@dataclass(frozen=True)
class SourceAccessPolicy:
    """A retrieval policy kept outside the model-visible verification case."""

    policy_id: str = "product-open-web"
    excluded_domains: frozenset[str] = field(default_factory=frozenset)
    excluded_urls: frozenset[str] = field(default_factory=frozenset)

    @property
    def active(self) -> bool:
        return bool(self.excluded_domains or self.excluded_urls)

    @property
    def cache_partition(self) -> str:
        payload = {
            "schema_version": POLICY_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "excluded_domains": sorted(self.excluded_domains),
            "excluded_urls": sorted(self.excluded_urls),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"{self.policy_id}:{digest[:16]}"

    def allows(self, value: str) -> bool:
        variants = url_variants(value)
        if not variants:
            return True
        for variant in variants:
            if variant in self.excluded_urls:
                return False
            hostname = _hostname(variant)
            if any(domain_matches(hostname, domain) for domain in self.excluded_domains):
                return False
        return True

    def blocked_query_reference(self, query: str) -> str:
        """Return a forbidden domain explicitly named in a search query."""

        if not self.active:
            return ""
        text = unquote(str(query or "")).lower()
        for domain in sorted(self.excluded_domains, key=len, reverse=True):
            pattern = rf"(?<![a-z0-9.-])(?:[a-z0-9-]+\.)*{re.escape(domain)}(?![a-z0-9.-])"
            if re.search(pattern, text):
                return domain
        return ""

    def filter_rows(
        self,
        rows: Any,
        *,
        url_fields: Sequence[str] = ("url", "link"),
        image_url_fields: Sequence[str] = ("image_url", "imageUrl", "thumbnailUrl"),
    ) -> tuple[list[dict[str, Any]], int]:
        """Remove complete provider rows before any title/snippet is consumed."""

        if not isinstance(rows, list):
            return [], 0
        allowed: list[dict[str, Any]] = []
        blocked = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            urls = [str(row.get(field, "")).strip() for field in (*url_fields, *image_url_fields)]
            urls = [url for url in urls if url]
            if urls and any(not self.allows(url) for url in urls):
                blocked += 1
                continue
            allowed.append(dict(row))
        return allowed, blocked

    def sanitize_payload(self, value: Any) -> tuple[Any, int]:
        """Recursively remove URL-bearing objects that violate this policy."""

        if not self.active:
            return value, 0
        if isinstance(value, list):
            output = []
            blocked = 0
            for item in value:
                sanitized, count = self.sanitize_payload(item)
                blocked += count
                if sanitized is not None:
                    output.append(sanitized)
            return output, blocked
        if not isinstance(value, dict):
            return value, 0

        url_keys = {
            "url",
            "link",
            "selected_url",
            "reference_url",
            "reference_image_url",
            "image_url",
            "imageUrl",
            "thumbnailUrl",
            "candidate_url",
        }
        urls = [str(value.get(key, "")).strip() for key in url_keys if value.get(key)]
        if urls and any(not self.allows(url) for url in urls):
            return None, 1

        output: dict[str, Any] = {}
        blocked = 0
        for key, item in value.items():
            if key in {"candidate_page_urls", "reference_image_candidates"} and isinstance(item, list):
                filtered = [str(url) for url in item if self.allows(str(url))]
                blocked += len(item) - len(filtered)
                output[key] = filtered
                continue
            sanitized, count = self.sanitize_payload(item)
            blocked += count
            if sanitized is not None:
                output[key] = sanitized
        return output, blocked

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "excluded_domains": sorted(self.excluded_domains),
            "excluded_urls": sorted(self.excluded_urls),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceAccessPolicy":
        schema = str(payload.get("schema_version", "")).strip()
        if schema and schema != POLICY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported source access policy schema: {schema}")
        domains = {
            _normalize_domain(str(item))
            for item in payload.get("excluded_domains", []) or []
            if str(item).strip()
        }
        urls = {
            variant
            for item in payload.get("excluded_urls", []) or []
            for variant in url_variants(str(item))
        }
        return cls(
            policy_id=str(payload.get("policy_id", "benchmark-evaluation")).strip()
            or "benchmark-evaluation",
            excluded_domains=frozenset(item for item in domains if item),
            excluded_urls=frozenset(urls),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SourceAccessPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Source access policy must be a JSON object.")
        return cls.from_dict(payload)


def benchmark_source_access_policy(
    source_urls: Iterable[str],
    *,
    policy_id: str = "benchmark-evaluation",
) -> SourceAccessPolicy:
    """Build a benchmark-wide deny policy solely from provenance URLs."""

    variants = {
        variant
        for source_url in source_urls
        for variant in url_variants(str(source_url))
    }
    candidate_hosts = {
        _hostname(variant)
        for variant in variants
        if _hostname(variant) and not domain_matches(_hostname(variant), "web.archive.org")
    }
    domains = {
        scope
        for hostname in candidate_hosts
        if (scope := _fact_check_domain_scope(hostname))
    }
    return SourceAccessPolicy(
        policy_id=policy_id,
        excluded_domains=frozenset(item for item in domains if item),
        excluded_urls=frozenset(variants),
    )


def benchmark_policy_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy_id: str = "benchmark-evaluation",
) -> SourceAccessPolicy:
    return benchmark_source_access_policy(
        (str(row.get("source_article_url", "")) for row in rows),
        policy_id=policy_id,
    )


def _normalize_domain(value: str) -> str:
    text = str(value or "").lower().strip().rstrip(".")
    if not text:
        return ""
    if "://" in text:
        return _hostname(canonicalize_url(text))
    return text.split("/", 1)[0].split(":", 1)[0].rstrip(".")


def _fact_check_domain_scope(hostname: str) -> str:
    host = _normalize_domain(hostname)
    domain = registered_domain(host)
    if any(marker in domain for marker in FACT_CHECK_DOMAIN_MARKERS):
        return domain
    if any(marker in host for marker in FACT_CHECK_DOMAIN_MARKERS):
        return host
    return ""
