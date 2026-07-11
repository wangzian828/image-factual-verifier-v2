from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from src.integrations.gemini import GeminiInteractionsClient, extract_images


load_dotenv()


DEFAULT_PROVIDER = "gemini"
DEFAULT_NECODEX_ENDPOINT = "https://fast.sbbbbbbbbb.xyz/v1/images/generations"
DEFAULT_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_NECODEX_MODEL = "gpt-image-2"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-image"


def _load_items(input_file: Path) -> List[Dict[str, Any]]:
    payload = json.loads(input_file.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise TypeError("Prompt input must be a JSON object or list.")


def _detect_image_extension(image_bytes: bytes, mime_type: Optional[str] = None) -> str:
    normalized_mime = (mime_type or "").lower().strip()
    if normalized_mime == "image/png":
        return ".png"
    if normalized_mime in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if normalized_mime == "image/webp":
        return ".webp"
    if normalized_mime == "image/gif":
        return ".gif"

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    return ".png"


def _extract_openai_style_image(payload: Dict[str, Any]) -> Tuple[bytes, Optional[str]]:
    data_items = payload.get("data", [])
    if not data_items:
        raise ValueError("Response missing data field.")

    item = data_items[0]
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"]), item.get("mime_type")
    if item.get("b64"):
        return base64.b64decode(item["b64"]), item.get("mime_type")
    raise ValueError("OpenAI-compatible response item does not contain b64_json or b64.")


def _extract_gemini_image(payload: Dict[str, Any]) -> Tuple[bytes, Optional[str]]:
    for item in extract_images(payload):
        for field in ("data", "base64", "b64", "b64_json"):
            if item.get(field):
                return base64.b64decode(item[field]), item.get("mime_type")

        uri = str(item.get("uri", "")).strip()
        if uri:
            response = requests.get(uri, timeout=180)
            response.raise_for_status()
            response_mime = response.headers.get("Content-Type", "").split(";", 1)[0].strip()
            return response.content, response_mime or item.get("mime_type")

    raise ValueError("Gemini response does not contain an image item with data, base64, or uri.")


def _resolve_size_for_gemini(size: str) -> Dict[str, str]:
    try:
        width_str, height_str = size.lower().split("x", 1)
        width = int(width_str)
        height = int(height_str)
    except Exception as exc:
        raise ValueError(f"Unsupported size format for Gemini: {size}") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size: {size}")

    ratio_map = {
        (1, 1): "1:1",
        (4, 3): "4:3",
        (3, 4): "3:4",
        (16, 9): "16:9",
        (9, 16): "9:16",
        (3, 2): "3:2",
        (2, 3): "2:3",
    }
    gcd = _gcd(width, height)
    key = (width // gcd, height // gcd)
    aspect_ratio = ratio_map.get(key)
    if not aspect_ratio:
        raise ValueError(
            f"Gemini image generation currently supports common aspect ratios only; got {size}."
        )

    max_edge = max(width, height)
    if max_edge <= 1024:
        image_size = "1K"
    elif max_edge <= 2048:
        image_size = "2K"
    else:
        raise ValueError(
            f"Gemini image generation size too large for this script mapping: {size}."
        )
    return {"aspect_ratio": aspect_ratio, "image_size": image_size}


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a) or 1


def _request_necodex_image(api_key: str, model: str, prompt: str, size: str) -> Dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        os.getenv("NECODEX_IMAGE_API_URL", DEFAULT_NECODEX_ENDPOINT),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "prompt": prompt,
            "size": size,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def _request_gemini_image(model: str, prompt: str, size: str) -> Dict[str, Any]:
    size_spec = _resolve_size_for_gemini(size)
    async def _generate() -> Dict[str, Any]:
        async with GeminiInteractionsClient(
            base_url=os.getenv("GEMINI_IMAGE_API_URL", DEFAULT_GEMINI_ENDPOINT),
            timeout=180.0,
            max_retries=0,
        ) as client:
            return await client.generate_image(
                model=model,
                prompt=prompt,
                aspect_ratio=size_spec["aspect_ratio"],
                image_size=size_spec["image_size"],
                background=False,
                store=True,
            )

    return asyncio.run(_generate())


def _request_image(provider: str, api_key: str, model: str, prompt: str, size: str) -> Dict[str, Any]:
    if provider == "necodex":
        return _request_necodex_image(api_key=api_key, model=model, prompt=prompt, size=size)
    if provider == "gemini":
        return _request_gemini_image(model=model, prompt=prompt, size=size)
    raise ValueError(f"Unsupported provider: {provider}")


def _extract_image(provider: str, payload: Dict[str, Any]) -> Tuple[bytes, Optional[str]]:
    if provider == "necodex":
        return _extract_openai_style_image(payload)
    if provider == "gemini":
        return _extract_gemini_image(payload)
    raise ValueError(f"Unsupported provider: {provider}")


def _resolve_api_key(provider: str) -> str:
    if provider == "necodex":
        api_key = os.getenv("NECODEX_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("NECODEX_API_KEY is required in the environment.")
        return api_key
    if provider == "gemini":
        api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required in the environment.")
        return api_key
    raise ValueError(f"Unsupported provider: {provider}")


def generate_samples(
    input_path: str,
    output_dir: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
    size: str = "1024x1024",
    skip_existing: bool = True,
    max_retries: int = 3,
    retry_sleep: float = 3.0,
) -> Dict[str, Any]:
    provider = provider.strip().lower()
    api_key = _resolve_api_key(provider)
    effective_model = model or (
        DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_NECODEX_MODEL
    )

    input_file = Path(input_path)
    root = Path(output_dir)
    image_dir = root / "images"
    response_dir = root / "responses"
    manifest_path = root / "manifest.jsonl"
    errors_path = root / "errors.jsonl"

    items = _load_items(input_file)
    image_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        default_id_prefix = "gemini_image" if provider == "gemini" else "gpt_image"
        sample_id = str(item.get("sample_id", f"{default_id_prefix}_{index:06d}"))
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"{sample_id} missing prompt.")

        existing_image = next(image_dir.glob(f"{sample_id}.*"), None)
        existing_response = response_dir / f"{sample_id}.json"
        if skip_existing and existing_image and existing_response.exists():
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "provider": provider,
                    "prompt": prompt,
                    "model": effective_model,
                    "size": size,
                    "image_path": str(existing_image),
                    "response_path": str(existing_response),
                    "metadata": item.get("metadata", {}),
                    "status": "existing",
                }
            )
            continue

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                payload = _request_image(
                    provider=provider,
                    api_key=api_key,
                    model=effective_model,
                    prompt=prompt,
                    size=size,
                )
                image_bytes, mime_type = _extract_image(provider, payload)
                image_ext = _detect_image_extension(image_bytes, mime_type)

                image_path = image_dir / f"{sample_id}{image_ext}"
                image_path.write_bytes(image_bytes)

                response_path = response_dir / f"{sample_id}.json"
                response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "provider": provider,
                        "prompt": prompt,
                        "model": effective_model,
                        "size": size,
                        "image_path": str(image_path),
                        "response_path": str(response_path),
                        "metadata": item.get("metadata", {}),
                        "status": "generated",
                    }
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(retry_sleep * (attempt + 1))

        if last_error is not None:
            error_rows.append(
                {
                    "sample_id": sample_id,
                    "provider": provider,
                    "model": effective_model,
                    "size": size,
                    "prompt": prompt,
                    "metadata": item.get("metadata", {}),
                    "error": repr(last_error),
                }
            )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with errors_path.open("w", encoding="utf-8") as handle:
        for row in error_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "input_path": str(input_file),
        "output_dir": str(root),
        "provider": provider,
        "model": effective_model,
        "num_samples": len(manifest_rows),
        "num_errors": len(error_rows),
        "manifest_path": str(manifest_path),
        "errors_path": str(errors_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dataset images with Gemini Interactions by default, or an "
            "explicitly selected compatible provider."
        )
    )
    parser.add_argument("--input", required=True, help="Path to JSON prompt list.")
    parser.add_argument("--output-dir", required=True, help="Directory to store generated images.")
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["necodex", "gemini"],
        help="Image generation provider.",
    )
    parser.add_argument("--model", default=None, help="Image model name.")
    parser.add_argument("--size", default="1024x1024", help="Image size, e.g. 1024x1024.")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip samples whose image and response already exist.")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", help="Regenerate even if output files already exist.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample on transient failures.")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Base retry sleep in seconds.")
    args = parser.parse_args()

    report = generate_samples(
        input_path=args.input,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        size=args.size,
        skip_existing=args.skip_existing,
        max_retries=max(0, args.max_retries),
        retry_sleep=max(0.0, args.retry_sleep),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
