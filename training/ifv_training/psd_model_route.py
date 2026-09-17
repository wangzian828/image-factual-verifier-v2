"""Explicit, scoped model handoff for uncached PSD external requests only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def uncached_model(model, cache_dir):
    path = os.environ.get('IFV_PSD_EXTERNAL_ROUTE')
    if not path:
        return model, None
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != os.environ.get('IFV_PSD_EXTERNAL_ROUTE_SHA256'):
        raise ValueError('PSD external model route changed after launch')
    route = json.loads(raw)
    if (route.get('schema_version') != 'ifv-psd-external-model-route-v1'
            or route.get('preserve_completed_responses') is not True
            or route.get('from_model') != 'gemini-3.7-flash'
            or route.get('to_model') != 'gemini-3.6-flash'
            or route.get('thinking_level') != 'high'
            or not Path(str(route.get('search_root', ''))).is_absolute()):
        raise ValueError('Invalid explicit PSD external model route')
    if model != route['from_model'] or cache_dir is None:
        return model, None
    root = Path(route['search_root']).resolve()
    if not Path(cache_dir).resolve().is_relative_to(root):
        return model, None
    return route['to_model'], {'path': str(Path(path).resolve()), 'sha256': digest,
                              'from_model': model, 'to_model': route['to_model']}
