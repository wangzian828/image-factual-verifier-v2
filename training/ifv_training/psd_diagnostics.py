"""Private, bounded failure snapshots without exception text or arbitrary locals.

These are diagnostic artifacts, never accepted trajectories or training targets.
Keep insertion order and raw model-visible strings for exact-token diagnosis.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import gzip
import hashlib
import json
from pathlib import Path
import time
import uuid


_CAPTURE_LOCALS = {'position', 'hint', 'steps', 'unhinted_prefix', 'system_instruction',
    'request_id', 'request', 'snapshot', 'messages', 'prefix', 'teacher_ids', 'student_ids', 'capture'}
_RUNTIME_LOCALS = {'teacher_steps', 'teacher_history', 'student_history', 'local_targets',
    'used_positions', 'runtime_state', 'source_prefix', 'site', 'parsed_judgment',
    'judgment_steps', 'position', 'teacher_complete'}


def _value(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Do not repr() providers, clients, HTTP requests or arbitrary objects.
    return {'omitted_object_type': type(value).__name__}


def exception_snapshot(error):
    frames, selected = [], []
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        module = frame.f_globals.get('__name__', '')
        function = frame.f_code.co_name
        record = {'file': frame.f_code.co_filename, 'function': function, 'line': trace.tb_lineno}
        frames.append(record)
        allowed = set()
        if module == 'ifv_training.psd_slate' and function == 'capture_target':
            allowed = _CAPTURE_LOCALS
        elif module == 'ifv_training.psd_repair_runtime':
            if function == 'run_hinted_episode':
                allowed = _RUNTIME_LOCALS
            elif function == '_remove_local_hint':
                allowed = {'history', 'unhinted_prefix', 'hint', 'index'}
        if allowed:
            selected.append({**record, 'locals': {key: _value(frame.f_locals[key])
                for key in sorted(allowed) if key in frame.f_locals}})
        trace = trace.tb_next
    return {'schema_version': 'ifv-psd-private-exception-v1', 'error_type': type(error).__name__,
            'frames': frames, 'selected_model_context': selected, 'time': time.time(),
            'exception_message_omitted': True, 'not_a_training_target': True}


def persist_exception(directory, error, *, maximum_bytes=64 * 1024**2):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = exception_snapshot(error)
    body = json.dumps(snapshot, ensure_ascii=False, separators=(',', ':')).encode()
    if len(body) > maximum_bytes:
        snapshot.pop('selected_model_context')
        snapshot['context_omitted_due_to_limit'] = True
        snapshot['original_bytes'] = len(body)
        body = json.dumps(snapshot, ensure_ascii=False).encode()
    path = directory / (str(time.time_ns()) + '-' + uuid.uuid4().hex + '.json.gz')
    with path.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', compresslevel=1, mtime=0) as compressed:
            compressed.write(body)
    return {'path': str(path), 'sha256': hashlib.sha256(body).hexdigest(),
            'error_type': type(error).__name__, 'bytes': len(body),
            'context_omitted_due_to_limit': snapshot.get('context_omitted_due_to_limit', False)}
