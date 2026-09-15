"""CPU-only per-request thinking-budget state for the isolated PSD server."""
from __future__ import annotations


def thinking_budget(params):
    extra = params.extra_args or {}
    value = extra.get('ifv_thinking_budget')
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= 8192:
        raise ValueError('ifv_thinking_budget must be an integer in [1, 8192]')
    if params.max_tokens is None or params.max_tokens <= value + 1:
        raise ValueError('Output budget must leave room for </think> and an answer')
    return value


class ThinkingState:
    def __init__(self, budget, end_token, output_tokens):
        self.budget = budget
        self.end_token = end_token
        self.output_tokens = output_tokens  # vLLM-owned live list; never copy or edit.
        self.cursor = 0
        self.reasoning_tokens = 0
        self.closed = False

    def needs_close(self):
        if len(self.output_tokens) < self.cursor:
            raise ValueError('Output token list shrank; unsupported decoding mode')
        for token in self.output_tokens[self.cursor:]:
            if not self.closed:
                if token == self.end_token:
                    self.closed = True
                else:
                    self.reasoning_tokens += 1
        self.cursor = len(self.output_tokens)
        return not self.closed and self.reasoning_tokens >= self.budget


def update_slots(slots, batch_update, factory):
    """Apply the public v0.18.1 remove/add/move contract, including slot reuse."""
    if batch_update is None:
        return
    for index in batch_update.removed:
        slots.pop(index, None)
    for index, params, prompt, output in batch_update.added:
        state = factory(params, prompt, output)
        if state is None:
            slots.pop(index, None)
        else:
            slots[index] = state
    for source, destination, direction in batch_update.moved:
        if source == destination:
            continue
        old_source = slots.pop(source, None)
        old_destination = slots.pop(destination, None)
        if old_source is not None:
            slots[destination] = old_source
        if direction.name == 'SWAP' and old_destination is not None:
            slots[source] = old_destination
    if any(index < 0 or index >= batch_update.batch_size for index in slots):
        raise ValueError('Thinking state no longer matches active batch slots')
