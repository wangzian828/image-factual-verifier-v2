import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.server.psd_wire_capture import WireCapture


class WireCaptureTests(unittest.TestCase):
    def capture(self, root, **kwargs):
        return WireCapture(root, min_free_bytes=0, **kwargs)

    def test_exact_bytes_retained_with_sha(self):
        with TemporaryDirectory() as temp:
            capture=self.capture(Path(temp)/'wire')
            request=b'{ "messages": ["exact  whitespace"] }'
            response=b'{"choices":[{"finish_reason":"length","message":{"content":null}}]}'
            ticket=capture.begin(request)
            self.assertTrue(capture.response(ticket,response,status_code=200,replica='http://127.0.0.1:19004'))
            self.assertEqual(gzip.decompress((ticket/'request.json.gz').read_bytes()),request)
            self.assertEqual(gzip.decompress((ticket/'response.json.gz').read_bytes()),response)
            self.assertEqual(json.loads((ticket/'response-meta.json').read_text())['sha256'],hashlib.sha256(response).hexdigest())

    def test_limits_reject_before_dispatch_without_eviction(self):
        with TemporaryDirectory() as temp:
            capture=self.capture(Path(temp)/'wire',max_bytes=4200,max_request_bytes=10,max_response_bytes=100)
            with self.assertRaises(ValueError):capture.begin(b'x'*11)
            ticket=capture.begin(b'abc')
            with self.assertRaises(ValueError):capture.begin(b'abc')
            self.assertTrue((ticket/'request.json.gz').exists())

    def test_response_limit_and_error_receipts(self):
        with TemporaryDirectory() as temp:
            capture=self.capture(Path(temp)/'wire',max_response_bytes=2)
            ticket=capture.begin(b'{}')
            self.assertFalse(capture.response(ticket,b'too large',status_code=200,replica='local'))
            self.assertEqual(json.loads((ticket/'error.json').read_text())['kind'],'response_capture_limit')

    def test_no_reuse_or_overwrite(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)/'wire';capture=self.capture(root)
            with self.assertRaises(FileExistsError):self.capture(root)
            ticket=capture.begin(b'{}');capture.response(ticket,b'{}',status_code=200,replica='local')
            with self.assertRaises(FileExistsError):capture.response(ticket,b'changed',status_code=200,replica='local')
