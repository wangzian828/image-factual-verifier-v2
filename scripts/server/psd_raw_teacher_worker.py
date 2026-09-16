"""Isolated eager PSD worker with pre-grammar self-teacher probabilities."""
from vllm.v1.worker.gpu_worker import Worker
from scripts.server.psd_raw_logprobs import install_raw_policy_logprobs


class RawTeacherWorker(Worker):
    def load_model(self):
        super().load_model()
        if not self.model_config.enforce_eager:
            raise ValueError('Raw PSD capture has only been validated in eager mode')
        if self.vllm_config.speculative_config is not None:
            raise ValueError('Raw PSD capture does not support speculative decoding')
        import vllm.v1.worker.gpu_model_runner as runner
        install_raw_policy_logprobs(runner, self.model_runner.sampler)
