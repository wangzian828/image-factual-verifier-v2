"""Bounded full-episode recovery, never answer-conditioned resampling.

Only exceptions marked at the policy-model transport/numerical boundary are
retryable. Tool failures, bad answers, length limits and format errors are not.
The frozen Agent and the gateway's no-POST-replay contract stay unchanged.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import httpx

from .psd_repair_storage import load_bound, save_bound
from .io import load_json, sha256_file

VERSION = "ifv-psd-infrastructure-retry-v2"
MAX_ATTEMPTS = 3  # Initial attempt plus at most two complete reruns.

NONFINITE_SERIALIZATION_MESSAGES = frozenset(
    "Out of range float values are not JSON compliant: " + value
    for value in ("nan", "inf", "-inf"))


def is_nonfinite_serialization_response(status, payload):
    """Narrow provider error, not a generic HTTP400 or generated error text."""
    error = payload.get("error") if isinstance(payload, dict) else None
    return (status == 400 and isinstance(error, dict)
            and error.get("type") == "BadRequestError" and error.get("code") == 400
            and error.get("message") in NONFINITE_SERIALIZATION_MESSAGES)


class PolicyInfrastructureFailure(RuntimeError):
    """Trusted local marker; never infer this from model/tool-generated text."""


class InfrastructureRetriesExhausted(RuntimeError):
    pass


def recovery_attempt_budget(root, inputs, default=MAX_ATTEMPTS, *, allowance=None):
    """A one-time, evidence-bound extension after an actual service intervention.

    Only a new ledger bank may carry this receipt. The original three failures
    stay charged, and a five-attempt bank can never authorize another extension.
    No checker/answer quality enters this decision.
    """
    path = Path(root) / 'recovery-allowance.json'
    if allowance is None and not path.exists():
        return default
    receipt = load_json(path) if allowance is None else allowance
    if (default != 3 or receipt.get('schema_version') != 'ifv-psd-numerical-recovery-allowance-v1'
            or receipt.get('max_attempts') != 5 or receipt.get('inputs') != inputs):
        raise ValueError('Invalid explicit numerical recovery allowance')
    source = Path(receipt['source_ledger'])
    if source.resolve() == (Path(root) / 'retry-state.json').resolve():
        raise ValueError('Recovery must preserve a separate original ledger')
    if sha256_file(source) != receipt['source_ledger_sha256']:
        raise ValueError('Original retry ledger changed')
    old = load_json(source)
    old_identity = old['identity']
    state = load_bound(source, identity=old_identity)
    if old_identity['inputs'] != inputs or old_identity['max_attempts'] != 3:
        raise ValueError('Cannot extend another recovery budget')
    attempts = state['attempts']
    if len(attempts) != 3 or any(a['status'] != 'infrastructure_failed' for a in attempts):
        raise ValueError('Recovery requires three documented infrastructure failures')
    if any(a.get('reason') != 'model_http_400_nonfinite_serialization' for a in attempts):
        raise ValueError('This recovery receipt only covers the diagnosed numerical error')
    evidence = receipt.get('evidence', {})
    if set(evidence) != {'intervention', 'replay'}:
        raise ValueError('Need both intervention and post-intervention replay evidence')
    for item in evidence.values():
        if sha256_file(Path(item['path'])) != item['sha256']:
            raise ValueError('Recovery evidence changed')
    intervention = load_json(Path(evidence['intervention']['path']))
    replay = load_json(Path(evidence['replay']['path']))
    if intervention.get('phase') != 'ready_for_replay' or not replay.get('diagnostic_only'):
        raise ValueError('Intervention has not passed readiness/replay')
    rows = replay.get('results', [])
    expected = {(gpu, source_index, repetition) for gpu in range(4)
                for source_index in range(4) for repetition in range(4)}
    actual = {(r.get('gpu'), r.get('source_index'), r.get('repetition')) for r in rows}
    if (replay.get('completed') != 64 or len(rows) != 64 or actual != expected
            or any(r.get('status') != 'completed'
                   or r.get('finish_reason') not in {'stop', 'tool_calls'} for r in rows)):
        raise ValueError('Four-replica post-intervention replay gate is incomplete or failed')
    return 5


def _transport_reason(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            try:
                payload = error.response.json()
            except (ValueError, UnicodeError):
                payload = None
            if is_nonfinite_serialization_response(status, payload):
                return "model_http_400_nonfinite_serialization"
            return f"model_http_{status}" if status in {408, 429, 500, 502, 503, 504} else None
        if isinstance(error, (httpx.TimeoutException, TimeoutError)):
            return "model_request_timeout"
        if isinstance(error, (httpx.NetworkError, httpx.RemoteProtocolError)):
            return "model_transport_failure"
        if type(error) is RuntimeError and str(error) == "Chat Completions returned an empty response body.":
            return "model_empty_http_body"
        error = error.__cause__
    return None


def validate_generated_probabilities(payload):
    """Selected tokens must be finite; masked-out top-k -Inf is legal.

    Stock grammar-masked probabilities are diagnostics, NOT PSD teacher targets.
    Do not inspect arbitrary text for 'NaN', or ban token 0 (a legitimate '!').
    """
    for choice in payload.get("choices", []):
        logprobs = choice.get("logprobs") or {}
        if not isinstance(logprobs, dict):
            continue
        for entry in logprobs.get("content", []):
            value = entry.get("logprob")
            if value is None or not math.isfinite(float(value)):
                raise PolicyInfrastructureFailure("model_nonfinite_selected_logprob")
            for candidate in entry.get("top_logprobs", []):
                value = candidate.get("logprob")
                if value is None or math.isnan(float(value)) or float(value) == math.inf:
                    raise PolicyInfrastructureFailure("model_nonfinite_candidate_logprob")
        for value in logprobs.get("token_logprobs", []):
            if value is None or not math.isfinite(float(value)):
                raise PolicyInfrastructureFailure("model_nonfinite_selected_logprob")


def guard_policy_backend(backend):
    """Instance-only wrapper: identify failures BEFORE actions execute."""
    if getattr(backend, "_psd_infrastructure_guard", False):
        return
    # Retrying the entire Agent must not multiply hidden per-request retries.
    if getattr(backend, "max_retries", 0) != 0:
        raise ValueError("PSD full-episode recovery requires model request retries=0")
    original = backend.get_response

    def quarantine(payload):
        from src.orchestrator.runtime_events import current_case_runtime_store
        from src.redaction import sanitize_for_persistence
        runtime = current_case_runtime_store()
        if runtime is not None:
            artifact = runtime.artifacts.put_text(
                json.dumps(sanitize_for_persistence(payload), ensure_ascii=False),
                media_type="application/json", suffix=".json",
                metadata={"kind": "psd_quarantined_model_response"})
            runtime.append_event("psd_numerical_response_quarantined", {"artifact": artifact})

    # The native parser may reject an unusable response before returning an
    # LLMResponse. Validate the policy HTTP response before that parser as well;
    # do not turn an ordinary empty/malformed model action into a retry.
    get_client = getattr(backend, "_get_shared_client", None)
    if get_client is not None:
        class PolicyClient:
            def __init__(self, client):
                self.client = client

            async def post(self, *args, **kwargs):
                response = await self.client.post(*args, **kwargs)
                try:
                    payload = response.json()
                except (ValueError, UnicodeError):
                    return response
                if is_nonfinite_serialization_response(response.status_code, payload):
                    quarantine(payload)
                    raise PolicyInfrastructureFailure("model_http_400_nonfinite_serialization")
                if response.is_success and isinstance(payload, dict):
                    try:
                        validate_generated_probabilities(payload)
                    except PolicyInfrastructureFailure:
                        quarantine(payload)
                        raise
                return response

            def __getattr__(self, name):
                return getattr(self.client, name)

        backend._get_shared_client = lambda: PolicyClient(get_client())

    async def guarded(*args, **kwargs):
        try:
            response = await original(*args, **kwargs)
        except Exception as error:
            reason = _transport_reason(error)
            if reason is None:
                raise
            raise PolicyInfrastructureFailure(reason) from error
        payload = response.raw if isinstance(response.raw, dict) else {}
        try:
            validate_generated_probabilities(payload)
        except PolicyInfrastructureFailure:
            quarantine(payload)
            raise
        return response

    backend.get_response = guarded
    backend._psd_infrastructure_guard = True


async def retry_episode(*, root, identity, generate, max_attempts=MAX_ATTEMPTS,
                        sleep=asyncio.sleep):
    """generate(attempt_directory) returns a JSON-serializable result.

    Persist the budget and successful result before returning. Crashes with an
    uncertain in-flight attempt require inspection, not an unbounded new budget.
    Gold/checker outcomes are deliberately absent from this decision function.
    """
    from .psd_repair_search import search_lock
    if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
        raise ValueError("PSD infrastructure max_attempts must be 1..5")
    root = Path(root)
    max_attempts = recovery_attempt_budget(root, identity, max_attempts)
    binding = {"version": VERSION, "inputs": identity, "max_attempts": max_attempts}
    marker, result_path = root / "retry-state.json", root / "result.json"
    with search_lock(root):
        state = load_bound(marker, identity=binding) if marker.exists() else {"attempts": []}
        if result_path.exists():
            return load_bound(result_path, identity=binding)
        attempts = state["attempts"]
        if attempts and attempts[-1]["status"] != "infrastructure_failed":
            raise RuntimeError("PSD previous attempt is unresolved; inspect it before redispatch")
        while len(attempts) < max_attempts:
            if attempts:
                await sleep(min(5 * 2 ** (len(attempts) - 1), 30))
            directory = root / f"attempt-{len(attempts) + 1:03d}"
            directory.mkdir(exist_ok=False)
            row = {"index": len(attempts) + 1, "directory": str(directory), "status": "running"}
            attempts.append(row)
            save_bound(marker, identity=binding, payload=state)
            try:
                result = await generate(directory)
            except PolicyInfrastructureFailure as error:
                row.update(status="infrastructure_failed", reason=str(error))
                save_bound(marker, identity=binding, payload=state)
                continue
            except BaseException as error:
                row.update(status="interrupted" if isinstance(error, asyncio.CancelledError)
                           else "nonretryable_error", error_type=type(error).__name__)
                save_bound(marker, identity=binding, payload=state)
                raise
            save_bound(result_path, identity=binding, payload=result)
            row["status"] = "completed"
            save_bound(marker, identity=binding, payload=state)
            return result
        raise InfrastructureRetriesExhausted(
            f"PSD infrastructure retry budget exhausted ({max_attempts} attempts); slot remains pending")
