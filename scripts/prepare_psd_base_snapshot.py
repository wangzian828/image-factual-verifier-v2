"""Hash an existing pretrained policy without claiming a local training run."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from ifv_training.checkpoints import build_checkpoint_manifest, build_serving_profile
from ifv_training.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--tool-call-parser", required=True)
    parser.add_argument("--reasoning-parser", required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=20) as response:
        models = json.load(response)["data"]
    served = [item for item in models if item["id"] == args.model_id]
    if len(served) != 1 or Path(served[0]["root"]).resolve() != args.model_dir.resolve():
        raise ValueError("served model does not match pretrained policy path")
    if int(served[0]["max_model_len"]) != 131072:
        raise ValueError("PSD policy no longer supports the requested 128K cap")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = args.output_dir / "pretrained-provenance.json"
    write_json(provenance, {"dataset_version": None, "local_training_performed": False,
        "source": "existing pretrained model; original training dataset/revision not attested",
        "artifact_identity": "exact model-file SHA256 values in checkpoint manifest"})
    manifest_path = args.output_dir / "checkpoint-manifest.json"
    manifest = build_checkpoint_manifest(checkpoint_dir=args.model_dir,
        dataset_manifest_path=provenance, output_path=manifest_path,
        base_model_id=str(args.model_dir.resolve()), model_revision="unknown; artifact hashes authoritative",
        processor_revision="artifact hashes authoritative", method="pretrained", framework_version="not_applicable")
    manifest["framework"] = {"name": "pretrained_snapshot", "version": None}
    write_json(manifest_path, manifest)
    profile = build_serving_profile(output_path=args.output_dir / "serving-profile.json",
        profile_id=args.model_id, model_path=str(args.model_dir.resolve()), engine="vllm",
        port=args.port, tensor_parallel_size=1, dtype=args.dtype, context_length=131072,
        tool_call_parser=args.tool_call_parser, reasoning_parser=args.reasoning_parser,
        thinking_enabled=True, checkpoint_manifest_path=manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "model_files": len(manifest["artifacts"]),
                      "profile_id": profile["profile_id"]}))


if __name__ == "__main__":
    main()
