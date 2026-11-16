#!/usr/bin/env python3
"""Collect all 45 exact Paper-0 core results with byte and cohort verification."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess

from collect_hzz_4l_results import run_keys, sha
from hzz_4l_core_cohort import SELECTION_ID, TEST_EVENTS, validate_selection


def collect(gcloud, prefix, root, key, membership_hash):
    route = f"{prefix}/{key}"
    scan = subprocess.run([gcloud, "storage", "ls", f"{route}/cluster_*/completion.json"],
                          capture_output=True, text=True, timeout=120, check=True)
    matches = scan.stdout.split()
    if len(matches) != 1 or not re.fullmatch(re.escape(route) + r"/cluster_[0-9]+_proc_[0-9]+/completion\.json", matches[0]):
        raise ValueError(f"{key}: require exactly one completed run under this campaign")
    remote = matches[0].rsplit("/", 1)[0]
    target = root / "runs" / key
    partial = target.with_name(target.name + ".partial")
    if target.exists():
        raise FileExistsError(target)
    evaluation = partial / "test_evaluation"
    evaluation.mkdir(parents=True, exist_ok=False)
    for subdir, files in (("", ("completion.json", "job_status.txt")),
                          ("test_evaluation/", ("test_predictions.npz", "evaluation_summary.json"))):
        subprocess.run([gcloud, "storage", "cp", *[f"{remote}/{subdir}{name}" for name in files],
                        str(partial / subdir) + "/"], check=True, timeout=300)
    status = dict(line.split("=", 1) for line in (partial / "job_status.txt").read_text().splitlines() if "=" in line)
    done = json.loads((partial / "completion.json").read_text())
    summary = evaluation / "evaluation_summary.json"
    report = json.loads(summary.read_text())
    digest = sha(summary)
    if (status.get("exit_code") != "0" or done.get("passed") is not True
        or done.get("run_key") != key or report.get("run_key") != key
        or report.get("membership_sha256") != membership_hash or report.get("test_events") != TEST_EVENTS
        or report.get("protocol", {}).get("cohort") != SELECTION_ID
        or done.get("evaluation_summary_sha256") != digest
        or report.get("prediction_sha256") != sha(evaluation / "test_predictions.npz")):
        raise ValueError(f"{key}: failed run, wrong cohort, or wrong result checksum")
    partial.rename(target)
    return dict(run_key=key, result_prefix=remote, evaluation_summary_sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-prefix", required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--gcloud", default="gcloud")
    args = parser.parse_args()
    prefix = args.results_prefix.rstrip("/")
    if not prefix.startswith("gs://") or any(c in prefix for c in "*?[]") or not prefix.endswith("/hzz_4l_core_multiclass_v1"):
        parser.error("Use the single Paper-0 core campaign prefix")
    selection, _ = validate_selection(args.study_dir / "selection")
    membership = selection["membership_sha256"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        runs = list(pool.map(lambda key: collect(args.gcloud, prefix, args.study_dir, key, membership), run_keys()))
    with (args.study_dir / "collection_receipt.json").open("x") as f:
        json.dump(dict(passed=True, selection_id=SELECTION_ID, membership_sha256=membership, runs=runs), f, indent=2)
    print(f"PASS: {len(runs)} matched Paper-0 core results collected", flush=True)


if __name__ == "__main__":
    main()
