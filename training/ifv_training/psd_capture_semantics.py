"""PSD-only metadata adapter; the frozen Agent's actions/prompts stay unchanged."""
RAW_POLICY_LOGPROBS = 'pre_grammar_unprocessed_v1'


def install_capture_semantics():
    from src.orchestrator import llm_backend
    if getattr(llm_backend.extract_policy_token_capture, '_psd_semantics', False):
        return
    original = llm_backend.extract_policy_token_capture

    def extract(payload, **kwargs):
        capture = original(payload, **kwargs)
        marker = payload.get('ifv_policy_logprobs')
        if marker == RAW_POLICY_LOGPROBS:
            capture['teacher_logprob_semantics'] = marker
        return capture

    extract._psd_semantics = True
    llm_backend.extract_policy_token_capture = extract
