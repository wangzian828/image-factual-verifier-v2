import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.server.repair_agent_judge_tail import classify_batch_item, validate_history
from scripts.server.run_realtime_agent_judges import Runner, retry_delay, valid_output
from src.eval.private_gold_judge_contract import PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA as SCHEMA

GOOD = {'quality_bucket': 'strong', 'fact_alignment': 'same_fact',
        'reason_quality': 'decisive_and_grounded', 'failure_modes': [], 'explanation': 'x'}


class TailRepairTests(unittest.TestCase):
    def test_only_explicit_cancelled_items_transferred(self):
        self.assertEqual(classify_batch_item({'status':'completed','judge_output':GOOD}, valid_output, SCHEMA),'retain')
        self.assertEqual(classify_batch_item({'status':'error','error_type':'BatchItemError',
            'error':'code=1 message=\'Request was cancelled because the batch was cancelled.\''}, valid_output, SCHEMA),'transfer')
        for row in [{'status':'running'}, {'status':'error','error_type':'BatchItemError','error':'HTTP503'},
                    {'status':'completed','judge_output':{}}, {'status':'error','error':'cancelled because the batch was cancelled'}]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                classify_batch_item(row,valid_output,SCHEMA)

    def history(self, root, number=6):
        for n in range(1,number+1):
            (root/f'attempt-{n:02d}.json').write_text(json.dumps({'attempt':n,'state':'provider_rejected','http_code':503}))
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.json')}

    def test_extension_preserves_original_receipts(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);bound=self.history(root)
            validate_history(root,base_attempts=6,limit=8,bound_hashes=bound)
            (root/'attempt-07.json').write_text('{"attempt":7,"state":"provider_rejected","http_code":503}')
            validate_history(root,base_attempts=6,limit=8,bound_hashes=bound)
            (root/'attempt-01.json').write_text('{}')
            with self.assertRaises(ValueError):validate_history(root,base_attempts=6,limit=8,bound_hashes=bound)

    def test_ambiguous_attempt_not_replayed(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);bound=self.history(root)
            (root/'attempt-07.json').write_text('{"attempt":7,"state":"in_flight"}')
            with self.assertRaises(ValueError):validate_history(root,base_attempts=6,limit=8,bound_hashes=bound)
            (root/'attempt-07.response.json').write_text('{}')
            validate_history(root,base_attempts=6,limit=8,bound_hashes=bound)

    def test_attempt_gaps_and_extra_budget_rejected(self):
        for n in [8,9]:
            with TemporaryDirectory() as temp:
                root=Path(temp);bound=self.history(root)
                if n==9:self.history(root,9)
                else:(root/'attempt-08.json').write_text('{"attempt":8,"state":"provider_rejected","http_code":503}')
                with self.assertRaises(ValueError):validate_history(root,base_attempts=6,limit=8,bound_hashes=bound)

    def test_default_budget_unchanged(self):
        self.assertIsNone(retry_delay(6,503))
        self.assertIsNotNone(retry_delay(6,503,limit=8))
        self.assertIsNotNone(retry_delay(7,503,limit=8))
        self.assertIsNone(retry_delay(8,503,limit=8))
        self.assertIsNone(retry_delay(7,None,limit=8))

    def test_gif_fails_without_api_or_changing_denominator(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            runner=Runner.__new__(Runner)
            runner.ownership={'journal_sha256':'J','sources':{'q':{'batch':[],'realtime':['ok','gif']}}}
            runner.terminal_failed_cases={'q':['gif']}
            runner.attempt_limits={}
            self.assertEqual(runner.run_one('q','gif'),'terminal_input_failure')
            saved={}
            def save(path,value):saved[str(path)]=value
            runner.f=SimpleNamespace(JOURNAL=root/'journal',digest=lambda p:'J',rows=lambda p:[],MODEL='unchanged',save=save)
            runner.root=root
            runner.record={'sources':{'q':{'sha256':'Q'}}}
            runner.schema=SCHEMA
            runner.candidates={'q':{'ok':{},'gif':{}}}
            runner.directory=lambda name,case:root/case
            runner.collect=SimpleNamespace(_category=lambda row:'correct_point_with_strong_evidence',_write_gzip_jsonl=lambda p,r:None)
            (root/'ok').mkdir()
            (root/'ok/result.json').write_text(json.dumps({'case_id':'ok','source_name':'q','source_sha256':'Q','judge_output':GOOD}))
            s=runner.merge()['q']
            self.assertEqual(s['valid_judges'],1)
            self.assertEqual(s['terminal_failed_judges'],1)
            self.assertEqual(s['reported_denominator'],1527)
            self.assertTrue(s['finished'])
            self.assertFalse(s['complete'])
            self.assertAlmostEqual(s['sesr_reported_percent'],100/1527)
