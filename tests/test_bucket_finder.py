import json
import subprocess
import sys
from pathlib import Path


def test_propose_candidate_buckets_applies_economic_constraints() -> None:
    script_path = Path(__file__).resolve().parents[1] / "stat_arb_bucket_finder.py"
    code = f"""
import importlib.util, json, numpy as np, sys
spec = importlib.util.spec_from_file_location("stat_arb_bucket_finder", r"{script_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
symbols = ["AMD", "NVDA", "SMH", "SOXX", "XLE"]
embeddings = np.array([
    [1.0, 0.0],
    [0.99, 0.01],
    [0.98, 0.02],
    [0.97, 0.03],
    [0.0, 1.0],
], dtype=float)
buckets = module.propose_candidate_buckets(
    asset_embeddings=embeddings,
    symbols=symbols,
    bucket_sizes=[3],
    top_k_neighbors=4,
    max_candidates_per_anchor=10,
)
print(json.dumps(buckets))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    buckets = [tuple(bucket) for bucket in json.loads(result.stdout)]

    assert ("AMD", "NVDA", "SMH") in buckets
    assert ("AMD", "SMH", "SOXX") not in buckets
    assert all("XLE" not in bucket for bucket in buckets)
