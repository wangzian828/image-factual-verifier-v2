import copy
import json
import unittest

from scripts.server.run_realtime_agent_judges import audit_text, check_intents, partition, retry_delay, valid_output
from src.eval.private_gold_judge_contract import PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA as SCHEMA


class RealtimeJudgeTests(unittest.TestCase):
    def setUp(self):
        self.candidates = {'q': {'a': {}, 'b': {}}, 'p': {'a': {}, 'b': {}}}
        self.record = {'sources': {'q': {'sha256': 'Q'}, 'p': {'sha256': 'P'}}}
        self.job = {'source_name': 'q', 'source_sha256': 'Q', 'case_ids': ['a'],
                    'sample_count': 1, 'display_name': 'd', 'batch_name': 'batch'}
        self.output = {'quality_bucket': 'strong', 'fact_alignment': 'same_fact',
                       'reason_quality': 'decisive_and_grounded', 'failure_modes': [], 'explanation': 'x'}

    def test_exclusive_partition_by_source(self):
        result = partition(self.candidates, [self.job], self.record)
        self.assertEqual(result['q'], {'batch': ['a'], 'realtime': ['b']})
        self.assertEqual(result['p'], {'batch': [], 'realtime': ['a', 'b']})

    def test_duplicate_membership_rejected(self):
        with self.assertRaises(ValueError):
            partition(self.candidates, [self.job, self.job], self.record)

    def test_changed_hash_rejected(self):
        with self.assertRaises(ValueError):
            partition(self.candidates, [{**self.job, 'source_sha256': 'wrong'}], self.record)

    def test_unknown_case_rejected(self):
        with self.assertRaises(ValueError):
            partition(self.candidates, [{**self.job, 'case_ids': ['unknown']}], self.record)

    def test_duplicate_case_rejected(self):
        with self.assertRaises(ValueError):
            partition(self.candidates, [{**self.job, 'case_ids': ['a', 'a'], 'sample_count': 2}], self.record)

    def test_ambiguous_create_rejected(self):
        with self.assertRaises(ValueError):
            check_intents([{'display_name': 'unresolved', 'state': 'create_inflight'}], [self.job])

    def test_accepted_or_rejected_intents_allowed(self):
        check_intents([{'display_name': 'd', 'state': 'create_inflight'},
                       {'display_name': 'rejected', 'state': 'server_rejected_429'}], [self.job])

    def test_text_matches_batch_exactly(self):
        candidate = {'gold': {'private': '事实'}, 'candidate_answer': {'verdict': 'fake'},
                     'candidate_output': {'raw_observations': [{'error': 'search failure'}]}}
        expected = 'PROMPT\n\nAUDIT INPUT:\n' + json.dumps({
            'private_gold': candidate['gold'], 'candidate_material': {'mode': 'agent_trace',
            'candidate_answer': candidate['candidate_answer'],
            'supporting_trace_material': candidate['candidate_output']}}, ensure_ascii=False, indent=2)
        self.assertEqual(audit_text(candidate, 'PROMPT'), expected)

    def test_valid_schema_allows_low_quality(self):
        self.assertTrue(valid_output(self.output, SCHEMA))
        self.output.update(quality_bucket='rejected', reason_quality='unsupported')
        self.assertTrue(valid_output(self.output, SCHEMA))

    def test_invalid_schema_rejected(self):
        for field, value in [('quality_bucket', 'invented'), ('failure_modes', ['x'] * 13),
                             ('failure_modes', [1]), ('explanation', None)]:
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(self.output)
                changed[field] = value
                self.assertFalse(valid_output(changed, SCHEMA))
        self.assertFalse(valid_output({**self.output, 'unexpected': 1}, SCHEMA))
        self.assertFalse(valid_output({}, SCHEMA))

    def test_bounded_retries(self):
        self.assertGreaterEqual(retry_delay(1, 429), 15)
        self.assertLessEqual(retry_delay(5, 503), 305)
        for code in [400, 401, 403, None, 429]:
            self.assertIsNone(retry_delay(6, code))
        self.assertIsNone(retry_delay(1, None))
        self.assertIsNone(retry_delay(1, 400))


if __name__ == '__main__':
    unittest.main()
