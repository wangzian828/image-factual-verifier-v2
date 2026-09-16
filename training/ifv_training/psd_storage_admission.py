"""Fail-closed collection admission without discarding slots on slow scans."""
import asyncio
import shutil
import subprocess
import time


def measure_run_storage(run):
    measured = subprocess.run(['du', '-s', '-B1', str(run)], capture_output=True,
                              text=True, check=True, timeout=180)
    return int(measured.stdout.split()[0]), shutil.disk_usage(run).free


class StorageAdmission:
    def __init__(self, *, run, ceiling, save, measure=measure_run_storage,
                 clock=time.time, sleep=asyncio.sleep):
        self.run, self.ceiling, self.save = run, ceiling, save
        self.measure, self.clock, self.sleep = measure, clock, sleep
        self.state = {'checked_at': 0, 'admission_open': False}
        self.lock = None

    async def __call__(self):
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            while True:
                if self.state['admission_open'] and self.clock()-self.state['checked_at'] < 60:
                    return
                try:
                    used, free = await asyncio.to_thread(self.measure, self.run)
                except (subprocess.SubprocessError, OSError, ValueError) as error:
                    # Do not let gather(return_exceptions=True) turn an admission
                    # measurement failure into a missing sampling slot. No model
                    # attempt starts or is charged while measurement is unknown.
                    self.state.update(admission_open=False, measurement_status='retrying',
                                      measurement_error=type(error).__name__,
                                      last_error_at=self.clock())
                    self.save(self.run/'storage.json', self.state)
                    await self.sleep(10)
                    continue
                self.state = dict(checked_at=self.clock(), run_bytes=used,
                                  shared_free_bytes=free, personal_quota_known=False,
                                  ceiling_bytes=self.ceiling, measurement_status='complete',
                                  admission_open=used < self.ceiling and free >= 16*1024**3)
                self.save(self.run/'storage.json', self.state)
                if self.state['admission_open']:
                    return
                # A real space hold also waits without consuming/skipping slots.
                # Monitoring can inspect the durable state and resolve it safely.
                await self.sleep(60)
