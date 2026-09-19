from ifv_training.io import iter_jsonl, load_jsonl, write_jsonl


def test_jsonl_gzip_round_trip_is_direct_and_deterministic(tmp_path):
    path = tmp_path / "ledger.jsonl.gz"
    rows = [{"capture": [1, 2, 3] * 10_000}, {"accepted": True}]
    write_jsonl(path, rows)
    first = path.read_bytes()
    assert first[:2] == b"\x1f\x8b"
    assert load_jsonl(path) == rows
    assert list(iter_jsonl(path)) == rows
    write_jsonl(path, rows)
    assert path.read_bytes() == first
