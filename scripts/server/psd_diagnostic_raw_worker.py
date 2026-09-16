"""Diagnostic-only worker: capture first bad prefill, never clamp or hide NaNs."""
import os
from scripts.server.psd_raw_teacher_worker import RawTeacherWorker
from psd_nan_boundary_capture import install_prefill_observer


class DiagnosticRawWorker(RawTeacherWorker):
    def load_model(self):
        super().load_model()
        install_prefill_observer(self, os.environ['PSD_PREFILL_DIAGNOSTIC_DIR'])
