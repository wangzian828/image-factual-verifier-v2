"""Bounded, uncached paired latency probe; never changes live PSD outputs."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path('/volume/ybo/wza')
CODE = ROOT / 'training-artifacts/psd-flash-high-20260917-v55/code'
SEARCH = ROOT / 'runs/psd-production-round1-20260917-v1/search-gemini37-flash-high'
OUTPUT = ROOT / 'runs/psd-flash-latency-probe-20260917'
sys.path[:0] = [str(CODE), str(CODE / 'training')]

from dotenv import load_dotenv
from src.integrations.gemini import GeminiInteractionsClient, extract_text
from ifv_training.psd_gemini_judge import (
    LOCALIZE_PROMPT, LOCALIZE_SCHEMA, review_images, trace_steps,
)


def load(path):
    return json.loads(path.read_text())


def safe_error(exc):
    result = {'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
    try:
        data = json.loads(getattr(exc, 'response_body', '{}')).get('error', {})
        if isinstance(data, dict):
            message = str(data.get('message', ''))
            for name in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
                secret = os.environ.get(name)
                if secret:
                    message = message.replace(secret, '[REDACTED]')
            message = re.sub(r'(?i)(key|token|authorization)\s*[=:]\s*[^\s&,]+', r'\1=[REDACTED]', message)
            result.update(provider_code=data.get('code'), provider_status=data.get('status'), message=message[:2000])
    except (ValueError, TypeError, AttributeError):
        pass
    return result


def emit(result):
    line = json.dumps(result, ensure_ascii=False)
    with (OUTPUT / 'results.jsonl').open('a') as stream:
        stream.write(line + '\n')
    print(line, flush=True)


async def probe(model, kind, payload, schema=None):
    started = time.monotonic()
    result = {'model': model, 'kind': kind, 'time': time.time(), 'thinking': 'high', 'max_retries': 0}
    emit({'event': 'request_started', **result})
    try:
        async with GeminiInteractionsClient(timeout=150, max_retries=0) as client:
            kwargs = {'response_format': {'type': 'text', 'mime_type': 'application/json', 'schema': schema}} if schema else {}
            response = await client.create(
                model=model, input=payload, generation_config={'thinking_level': 'high', 'max_output_tokens': 8192},
                store=True, **kwargs,
            )
        result.update(status=response.get('status'), usage=response.get('usage'), returned_model=response.get('model'))
        text = extract_text(response)
        if schema:
            try:
                value = json.loads(text)
                result['required_fields_present'] = isinstance(value, dict) and set(schema['required']).issubset(value)
            except ValueError:
                result['required_fields_present'] = False
        else:
            result['ok_response'] = text.strip() == 'OK'
    except Exception as exc:
        result.update(safe_error(exc))
    result['elapsed_seconds'] = round(time.monotonic() - started, 2)
    emit({'event': 'request_finished', **result})


async def main(models, case_key=None):
    os.umask(0o077)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / 'private/runtime.env', override=True)
    # Read/prepare once, so both models see identical text, archived images and settings.
    if case_key is not None:
        if not re.fullmatch(r'[0-9a-f]{16}', case_key):
            raise ValueError('case key must be the 16-character frozen input directory name')
        error_path = SEARCH / 'case-errors' / (case_key + '.json')
    else:
        error_path = next(path for path in sorted((SEARCH / 'case-errors').glob('*.json'))
                          if load(path).get('http_status') == 429)
    inputs = SEARCH / 'case-inputs' / error_path.stem
    candidate, gold = load(inputs / 'candidate.json'), load(inputs / 'gold.json')
    trace = load(ROOT / 'runs/psd-production400x8-20260917-v6/episodes' / candidate['source']['source_trace_path'])
    manifest = ROOT / 'runs/psd-pilot400-preparation-20260915-v1/runtime-release/runtime_input/cases.jsonl'
    case = next(row for line in manifest.read_text().splitlines()
                if (row := json.loads(line))['case_id'] == candidate['case_id'])
    images, media = review_images({'source': trace}, image_path=manifest.parent / case['image_path'])
    packet = {'source_steps': trace_steps(trace), 'private_reference': gold, 'media': media}
    text = LOCALIZE_PROMPT + '\nMATERIAL:\n' + json.dumps(packet, ensure_ascii=False)
    payload = [{'type': 'text', 'text': text}, *images]
    emit({'event': 'prepared', 'case_id': candidate['case_id'], 'models': models, 'text_chars': len(text),
          'image_count': sum(block.get('type') == 'image' for block in images),
          'note': 'same archived PSD localization input; no production cache or target writes'})
    await asyncio.gather(*(probe(model, 'tiny', 'Return only OK') for model in models))
    await asyncio.gather(*(probe(model, 'psd_localization', payload, LOCALIZE_SCHEMA) for model in models))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+',
                        choices=['gemini-3.5-flash', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
                        default=['gemini-3.7-flash', 'gemini-3.8-flash'])
    parser.add_argument('--case-key')
    args = parser.parse_args()
    if len(args.models) > 2 or len(set(args.models)) != len(args.models):
        parser.error('select one or two distinct models to keep the probe bounded')
    asyncio.run(main(args.models, args.case_key))
