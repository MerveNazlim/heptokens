#!/usr/bin/env python3
"""Check or explicitly queue the sealed core-cohort DAG once on gcpsub01."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

CAMPAIGN = "hzz_4l_core_multiclass_v1"
SELECTION = "hzz_4l_paper0_core_v1"
LAUNCH = "hzz_4l_core_multiclass_launch.json"
WORKER = "amscgpu_hzz_4l_core_multiclass.sh"
DAG = "amscgpu_hzz_4l_core_multiclass.dag"
SUB = "amscgpu_hzz_4l_core_multiclass_fit.sub"
PLOT_SUB = "amscgpu_hzz_4l_core_multiclass_plots.sub"
KEY = Path("/usatlas/u/magaras/condor_google/fccstorageuser.json")
PREFIX = "gs://fcc-teststorage/bucket_results/foundation_pretrain"
FAMILIES = {"q1_projected": "q1", "q4": "q4", "q8": "q8", "continuous_flat": "continuous",
            "hierarchical_kin_aux": "continuous"}


def run(command):
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if process.returncode:
        raise RuntimeError(f"{command[0]} exited {process.returncode}: {process.stderr.strip()}")
    return process.stdout.strip()


def check_dag(text, plan):
    desired = {
        f"{family}_{condition}_{'s' if condition == 'scratch' else 'p'}{seed}": kind
        for family, kind in FAMILIES.items() for condition in ("finetune", "scratch", "frozen")
        for seed in (42, 43, 44)
    }
    run_keys = plan["run_keys"]
    if len(run_keys) != 45 or set(run_keys) != set(desired):
        raise ValueError("Require the original five families, three conditions, and three seeds")
    nodes = [f"FIT_{i:02d}" for i in range(len(run_keys))]
    exact = [f"JOB PLOTS {PLOT_SUB}"]
    for key, node in zip(run_keys, nodes):
        kind = desired[key]
        digest = plan["inputs"][kind]["sha256"]
        exact += [f"JOB {node} {SUB}", f'VARS {node} RunKey="{key}" DataKind="{kind}" DataSha="{digest}"']
    exact += [f"PARENT {' '.join(nodes)} CHILD PLOTS"]
    if text.splitlines() != exact:
        raise ValueError("DAG does not declare exactly 45 independent fits feeding the single plot child")


def check_local(root):
    if Path.cwd().resolve() != root:
        raise ValueError(f"Run from the staged launch directory: {root}")
    launch = root / LAUNCH
    if launch.is_symlink():
        raise ValueError("Launch receipt cannot be a symlink")
    plan = json.loads(launch.read_text())
    if (plan["campaign"] != CAMPAIGN or plan["selection_id"] != SELECTION or plan["credential_path"] != str(KEY)
            or plan["dag"] != DAG or plan["data_prefix"] != f"{PREFIX}/downstream_data/{CAMPAIGN}"
            or plan["result_prefix"] != f"{PREFIX}/downstream/{CAMPAIGN}"
            or (plan["token_fits"], plan["continuous_fits"], plan["continuous_preparation_jobs"], plan["plot_jobs"])
                != (27, 18, 0, 1)):
        raise ValueError("Unexpected core campaign or job inventory")
    if set(plan["launch_files"]) != {WORKER, Path(__file__).name, DAG, SUB, PLOT_SUB}:
        raise ValueError("Unexpected launch file inventory")
    for name, digest in plan["launch_files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed or missing launch file: {name}")
    if not KEY.is_file() or KEY.stat().st_size <= 0:
        raise ValueError("Existing submit-side service key unavailable")
    if set(plan["inputs"]) != {"common", *FAMILIES.values()}:
        raise ValueError("Missing or extra sealed data archives")
    for kind, spec in {"runtime": plan["runtime"], **plan["inputs"]}.items():
        if (spec["archive"] != f"{kind}.tar.gz" or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])
                or type(spec["bytes"]) is not int or spec["bytes"] <= 0):
            raise ValueError(f"Malformed archive pin: {kind}")
    check_dag((root / DAG).read_text(), plan)
    for name, gpus, cpus in ((SUB, 1, 8), (PLOT_SUB, 0, 2)):
        text = (root / name).read_text().splitlines()
        required = {f"executable = {WORKER}", "+amscgpu_job = true", f"request_gpus = {gpus}",
                    f"request_cpus = {cpus}", f"transfer_input_files = {KEY}", "queue 1"}
        if not required.issubset(text):
            raise ValueError(f"Wrong resource or transfer configuration: {name}")
    return plan


def check(root, attempt):
    plan = check_local(root)
    if attempt.exists() or attempt.is_symlink():
        raise ValueError(f"Campaign already has a submission attempt receipt: {attempt}")
    where = f'HzzFourLeptonCampaign == "{CAMPAIGN}" || JobBatchName == "{CAMPAIGN}"'
    for tool in ("condor_q", "condor_history"):
        command = [tool, "-constraint", where] + (["-limit", "1"] if tool == "condor_history" else [])
        existing = run(command + ["-af", "ClusterId", "ProcId", "JobStatus"])
        if existing:
            raise ValueError(f"Existing core campaign in {tool}: {existing}")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    attempt = KEY.parent / (CAMPAIGN + "_submission.json")
    if args.check_only:
        plan = check(root, attempt)
        print("READY: 45 independent one-GPU fits; one CPU plot child; no preparation or smoke jobs. No jobs submitted.")
        print("Pinned runtime:", plan["runtime"]["sha256"])
        return
    fd = os.open(KEY.parent / ("." + CAMPAIGN + "_submit.lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Unsafe submit lock")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = check(root, attempt)
        receipt = dict(campaign=CAMPAIGN, stage="attempt_started", directory=str(root),
                       runtime_sha256=plan["runtime"]["sha256"], membership_sha256=plan["membership_sha256"])
        with attempt.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        output = run(["condor_submit_dag", "-batch-name", CAMPAIGN, plan["dag"]])
        match = re.search(r"submitted to cluster (\d+)\.", output)
        if not match:
            raise RuntimeError("Submission response unrecognized; inspect Condor before retrying:\n" + output)
        receipt.update(stage="submitted", dag_cluster=int(match.group(1)))
        attempt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(output)
        print("Submission receipt:", attempt)


if __name__ == "__main__":
    main()
