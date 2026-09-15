"""Opt-in vLLM 0.18.1 plugin. Do not load into the frozen evaluation service."""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from vllm.v1.sample.logits_processor import LogitsProcessor

from scripts.server.psd_thinking_state import ThinkingState, thinking_budget, update_slots


class PSDThinkingBudget(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params):
        thinking_budget(sampling_params)

    def __init__(self, vllm_config, device, is_pin_memory):
        if vllm_config.speculative_config is not None:
            raise ValueError('PSD thinking plugin has not been validated with speculative decoding')
        model = Path(vllm_config.model_config.tokenizer)
        tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
        self.end_token = tokenizer.token_to_id('</think>')
        start_token = tokenizer.token_to_id('<think>')
        if self.end_token is None or start_token is None:
            raise ValueError('Model tokenizer is missing thinking delimiters')
        for text, token in [('</think>', self.end_token), ('<think>', start_token)]:
            if tokenizer.encode(text, add_special_tokens=False).ids != [token]:
                raise ValueError('Thinking delimiters must each be one token')
        self.enabled_suffix = tokenizer.encode(
            '<|im_start|>assistant\n<think>\n', add_special_tokens=False).ids
        self.disabled_suffix = tokenizer.encode(
            '<|im_start|>assistant\n<think>\n\n</think>\n\n',
            add_special_tokens=False).ids
        self.slots = {}

    def is_argmax_invariant(self):
        return False  # Greedy requests must also receive the budget boundary.

    def _new_state(self, params, prompt, output):
        budget = thinking_budget(params)
        if budget is None:
            return None
        if prompt is None:
            raise ValueError('PSD thinking plugin requires exact prompt token IDs')
        if prompt[-len(self.disabled_suffix):] == self.disabled_suffix:
            return None
        if prompt[-len(self.enabled_suffix):] != self.enabled_suffix:
            raise ValueError('Unexpected assistant generation prefix; refusing ineffective budget')
        return ThinkingState(budget, self.end_token, output)

    def update_state(self, batch_update):
        update_slots(self.slots, batch_update, self._new_state)

    def apply(self, logits):
        for row, state in self.slots.items():
            if state.needs_close():
                if self.end_token >= logits.shape[-1]:
                    raise ValueError('Tokenizer and model vocabulary do not match')
                # Close reasoning, NOT EOS. Once emitted, all answer logits are untouched.
                logits[row, :] = -float('inf')
                logits[row, self.end_token] = 0.0
        return logits
