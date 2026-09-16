"""Enforce the already-requested single-action contract during JSON decoding."""
from __future__ import annotations

import copy
import json


def single_call_schema(schema):
    value = json.loads(schema) if isinstance(schema, str) else copy.deepcopy(schema)
    if not isinstance(value, dict) or value.get('type') != 'array' or value.get('minItems') != 1:
        raise ValueError('Unexpected required-tool schema; refusing silent protocol drift')
    if 'items' not in value or value.get('maxItems', 1) != 1:
        raise ValueError('Unexpected tool-array limit')
    value['maxItems'] = 1
    return value
