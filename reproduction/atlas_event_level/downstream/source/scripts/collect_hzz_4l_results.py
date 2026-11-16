#!/usr/bin/env python3
"""Collect the uniquely completed 4l fits; never accept inclusive-run predictions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def run_keys():
    return [f"{family}_{condition}_{'s' if condition == 'scratch' else 'p'}{seed}"
            for family in ("q1_projected", "q4", "q8", "continuous_flat", "hierarchical_kin_aux")
            for condition in ("finetune", "scratch", "frozen") for seed in (42, 43, 44)]


def collect(gcloud, prefix, root, key, membership_hash):
    route = f"{prefix}/{key}"
    scan = subprocess.run([gcloud, "storage", "ls", f"{route}/cluster_*/completion.json"],
                          capture_output=True, text=True, timeout=120)
    candidates = [line.strip() for line in scan.stdout.splitlines() if line.strip()]
    if scan.returncode or len(candidates) != 1:
        raise ValueError(f"{key}: require exactly one completed run, got {len(candidates)}")
    if not re.fullmatch(re.escape(route) + r"/cluster_[0-9]+_proc_[0-9]+/completion\.json", candidates[0]):
        raise ValueError(f"{key}: malformed completion object path")
    base = candidates[0].rsplit("/", 1)[0]
    target = root / "runs" / key
    partial = target.with_name(target.name + ".partial")
    if target.exists():
        raise FileExistsError(target)
    evaluation = partial / "test_evaluation"
    evaluation.mkdir(parents=True, exist_ok=False)
    for subdir, names in (("", ("completion.json", "job_status.txt")),
                          ("test_evaluation/", ("test_predictions.npz", "evaluation_summary.json"))):
        subprocess.run([gcloud, "storage", "cp", *[f"{base}/{subdir}{name}" for name in names],
                        str(partial / subdir) + "/"], check=True, timeout=300)
    state = dict(line.split("=", 1) for line in (partial / "job_status.txt").read_text().splitlines() if "=" in line)
    completion = json.loads((partial / "completion.json").read_text())
    summary_path = evaluation / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text())
    if (state.get("exit_code") != "0" or completion.get("passed") is not True or
        completion.get("run_key") != key or summary.get("run_key") != key or
        summary.get("membership_sha256") != membership_hash or summary.get("test_events") != 74921 or
        completion.get("evaluation_summary_sha256") != sha(summary_path) or
        summary.get("prediction_sha256") != sha(evaluation / "test_predictions.npz")):
        raise ValueError(f"{key}: failed status or mismatched run, cohort, or prediction checksums")
    partial.rename(target)
    return dict(run_key=key, result_prefix=base, evaluation_summary_sha256=sha(target / "test_evaluation" / summary_path.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-prefix", required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--gcloud", default="gcloud")
    args = parser.parse_args()
    prefix = args.results_prefix.rstrip("/")
    if not prefix.startswith("gs://") or any(char in prefix for char in "*?[]"):
        parser.error("--results-prefix must be a single GCS campaign prefix")
    membership_hash = sha(args.study_dir / "selection" / "membership.npz")
    keys = run_keys()
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(lambda key: collect(args.gcloud, prefix, args.study_dir, key, membership_hash), keys))
    with (args.study_dir / "collection_receipt.json").open("x") as stream:
        json.dump(dict(passed=True, membership_sha256=membership_hash, runs=receipts), stream, indent=2)
    print(f"PASS: {len(receipts)} completed 4l runs collected", flush=True)


if __name__ == "__main__":
    main()
