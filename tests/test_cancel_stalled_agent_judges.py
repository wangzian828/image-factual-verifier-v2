import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.server.cancel_stalled_agent_judges import select_targets, save_new


class CancellationTests(unittest.TestCase):
    def test_only_active_owned_jobs(self):
        rows = [{'batch_name': name} for name in ['a', 'b', 'c', 'd']]
        states = dict(zip(['a', 'b', 'c', 'd'], ['JOB_STATE_RUNNING', 'JOB_STATE_SUCCEEDED',
                         'JOB_STATE_CANCELLED', 'JOB_STATE_PENDING']))
        self.assertEqual(select_targets(rows, states), [rows[0], rows[3]])

    def test_identity_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            select_targets([{'batch_name': 'a'}], {'other': 'JOB_STATE_RUNNING'})
        with self.assertRaises(ValueError):
            select_targets([{'batch_name': 'a'}] * 2, {'a': 'JOB_STATE_RUNNING'})

    def test_receipts_never_overwritten(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'receipt.json'
            save_new(path, {'original': True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_new(path, {'replacement': True})
            self.assertEqual(before, path.read_bytes())
