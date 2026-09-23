"""ms-swift external plugin entrypoint for IFV privileged self-distillation."""

from ifv_training.psd_ms_swift import install_ms_swift_psd_plugin
from ifv_training.psd_fsdp_resume import install_fsdp_resume_bridge
from ifv_training.psd_deterministic_sampler import install_deterministic_psd_sampler
from ifv_training.psd_fast_length import install_fast_psd_length
from ifv_training.psd_kernel_trace import install_fla_tuning_trace


install_ms_swift_psd_plugin()
install_fsdp_resume_bridge()
install_deterministic_psd_sampler()
install_fast_psd_length()
install_fla_tuning_trace()
