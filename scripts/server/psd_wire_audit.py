"""Separate native policy acceptance from recorded auxiliary HTTP retries."""
from collections import Counter
import gzip
import hashlib
import json
import re


def audit_wire_archive(root, *, tokenizer, perception_prompt):
    """Fail closed on policy/capture errors; retain auxiliary transport evidence.

    A successful identical auxiliary request is corroborating evidence, NOT a
    claim that it is the same logical retry. Canonical tool outcomes are audited
    separately. No failed ticket or failed tool observation is removed.
    """
    counts = Counter()
    successes, auxiliary_failures = {}, []
    end = tokenizer.token_to_id('</think>')
    if end is None:
        raise ValueError('Missing thinking terminator')
    maximum = 0
    for ticket in sorted(root.iterdir()):
        def body(side):
            raw = gzip.decompress((ticket / (side + '.json.gz')).read_bytes())
            meta = json.loads((ticket / (side + '-meta.json')).read_text())
            if len(raw) != meta['bytes'] or hashlib.sha256(raw).hexdigest() != meta['sha256']:
                raise ValueError('Wire content hash mismatch')
            return json.loads(raw), meta

        request, _ = body('request')
        messages = request.get('messages', [])
        # Frozen OpenAIVisionClient also serves crop/relation/anomaly/reference
        # tools. Their two-message JSON/non-thinking envelope is distinct from
        # every captured native ReAct/Judgment decision (checked in the trace).
        auxiliary = bool(len(messages) == 2 and messages[0].get('role') == 'system'
                         and isinstance(messages[0].get('content'), str)
                         and messages[1].get('role') == 'user'
                         and isinstance(messages[1].get('content'), list)
                         and any(p.get('type') == 'image_url' for p in messages[1]['content'])
                         and request.get('chat_template_kwargs') == {'enable_thinking': False}
                         and request.get('response_format', {}).get('type') in ('json_schema', 'json_object')
                         and request.get('temperature') == 0
                         and not request.get('tools') and not request.get('logprobs')
                         and not request.get('tool_choice'))
        if not auxiliary and (request.get('logprobs') is not True or request.get('top_logprobs') != 20):
            raise ValueError('Unknown or uncaptured request in policy archive')
        # The gateway deliberately injects a fresh APC-isolation nonce. It is
        # not model input. Ignore ONLY that validated field when matching the
        # same generation payload; temperature/tools/images/etc. stay bound.
        comparable = dict(request)
        salt = comparable.pop('cache_salt', None)
        if salt is not None and not re.fullmatch(r'ifv-psd-isolated-[0-9a-f]{32}', salt):
            raise ValueError('Unknown cache isolation nonce')
        digest = hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()
        counts['requests'] += 1
        error_path = ticket / 'error.json'
        if error_path.exists():
            error = json.loads(error_path.read_text())
            if (not auxiliary or messages[0]['content'] != perception_prompt
                    or error.get('kind') != 'request_cancelled_or_transport_failed'
                    or (ticket / 'response-meta.json').exists()):
                raise ValueError('Native policy or capture failure requires inspection')
            auxiliary_failures.append({'ticket': ticket.name, 'request_fingerprint_without_cache_salt': digest,
                                       'error': error})
            counts['auxiliary_transport_failures'] += 1
            continue
        response, meta = body('response')
        if meta['status_code'] != 200 or not response.get('choices'):
            raise ValueError('Incomplete or non-200 response')
        for choice in response['choices']:
            ids = choice.get('token_ids') or []
            if choice.get('finish_reason') not in ('stop', 'tool_calls') or not ids:
                raise ValueError('Truncated or missing raw token response')
            maximum = max(maximum, len(ids))
            if request.get('tool_choice') == 'required':
                if end not in ids or choice['finish_reason'] != 'tool_calls':
                    raise ValueError('Required action did not finish after thinking')
                calls = json.loads(tokenizer.decode(ids[ids.index(end) + 1:], skip_special_tokens=True).strip())
                if not isinstance(calls, list) or len(calls) != 1:
                    raise ValueError('Required action is not one native call')
                counts['required_single_calls'] += 1
        counts['responses'] += 1
        if auxiliary:
            counts['auxiliary_json_responses'] += 1
            successes.setdefault(digest, []).append(ticket.name)
    for failure in auxiliary_failures:
        peers = successes.get(failure['request_fingerprint_without_cache_salt'], [])
        if not peers:
            raise ValueError('Auxiliary failure has no successful identical request; inspect tool outcomes')
        failure['identical_request_successes'] = peers
    return {**dict(counts), 'max_output_tokens': maximum,
            'auxiliary_failure_evidence': auxiliary_failures}
