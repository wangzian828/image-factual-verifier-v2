"""Bounded, opt-in server-local wire archive for isolated PSD diagnosis.

Never stores HTTP headers or modifies provider request/response bodies.
Disabled unless a new capture directory is explicitly configured.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid


class WireCapture:
    def __init__(self, root, *, max_bytes=2*1024**3, max_request_bytes=64*1024**2,
                 max_response_bytes=8*1024**2, min_free_bytes=2*1024**3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.max_bytes = max_bytes
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.min_free_bytes = min_free_bytes
        self.reserved = 0
        self.committed = 0
        self.outstanding = {}
        self.lock = threading.Lock()
        self.started = self.completed = self.failed = 0

    @classmethod
    def from_env(cls):
        value = os.environ.get('PSD_WIRE_CAPTURE_DIR', '').strip()
        if not value:
            return None
        path = Path(value).resolve()
        path.relative_to(Path('/volume/ybo/wza').resolve())
        response_limit = int(os.environ.get('PSD_WIRE_MAX_RESPONSE_BYTES', str(8*1024**2)))
        if not 1 <= response_limit <= 64*1024**2:
            raise ValueError('PSD wire response allowance must be at most 64 MiB')
        return cls(path, max_response_bytes=response_limit)

    def _json(self, directory, name, value):
        with (directory / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)

    def _body(self, directory, name, body):
        with (directory / name).open('xb') as stream:
            with gzip.GzipFile(fileobj=stream, mode='wb', compresslevel=1, mtime=0) as compressed:
                compressed.write(body)
        return {'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body),
                'gzip_bytes': (directory / name).stat().st_size}

    def begin(self, body):
        # Bound gzip expansion and metadata before dispatch. On completion,
        # retain actual disk usage, not an unused 8 MiB allowance forever.
        bound = lambda size: size + (size // 16384 + 1) * 16 + 128
        required = bound(len(body)) + bound(self.max_response_bytes) + 16384
        key = str(time.time_ns()) + '-' + uuid.uuid4().hex
        directory = self.root / key
        with self.lock:
            if len(body) > self.max_request_bytes or self.reserved + required > self.max_bytes:
                raise ValueError('PSD diagnostic capture budget exhausted before dispatch')
            if shutil.disk_usage(self.root).free < self.min_free_bytes + required:
                raise ValueError('PSD diagnostic capture minimum free space reached')
            self.reserved += required
            self.outstanding[directory] = required
            self.started += 1
        directory.mkdir()
        binding = self._body(directory, 'request.json.gz', body)
        self._json(directory, 'request-meta.json', {'started_at':time.time(), **binding})
        return directory

    def _settle(self, ticket):
        with self.lock:
            allowance = self.outstanding.pop(ticket, None)
            if allowance is None:
                return
            actual = sum(path.stat().st_size for path in ticket.iterdir() if path.is_file())
            self.reserved += actual - allowance
            self.committed += actual

    def response(self, ticket, body, *, status_code, replica):
        if len(body) > self.max_response_bytes:
            self.error(ticket, 'response_capture_limit', status_code=status_code, replica=replica)
            return False
        binding = self._body(ticket, 'response.json.gz', body)
        self._json(ticket, 'response-meta.json', {'ended_at':time.time(), 'status_code':status_code,
                                                  'replica':replica, **binding})
        self.completed += 1
        self._settle(ticket)
        return True

    def error(self, ticket, kind, **metadata):
        self._json(ticket, 'error.json', {'ended_at':time.time(), 'kind':kind, **metadata})
        self.failed += 1
        self._settle(ticket)

    def status(self):
        return {'enabled':True,'root':str(self.root),'started':self.started,'completed':self.completed,
                'failed':self.failed,'reserved_bytes':self.reserved,'max_bytes':self.max_bytes,
                'committed_bytes':self.committed,'outstanding_requests':len(self.outstanding),
                'budget_accounting':'compressed_completed_plus_bounded_inflight'}
