#!/usr/bin/env python3
"""Check or explicitly queue the three sealed Paper0 binary jobs; no gcloud here."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tempfile

CAMPAIGN = "hzz_paper0_q1_4l_v1"
LAUNCH = "hzz_paper0_q1_4l_launch.json"
WORKER = "amscgpu_hzz_paper0_q1_4l.sh"
SUBMIT = "amscgpu_hzz_paper0_q1_4l.sub"
GUARD = "submit_hzz_paper0_q1_4l.py"
KEY = Path("/usatlas/u/magaras/condor_google/fccstorageuser.json")
ARM_DATASET = {"scratch": "q1", "finetune": "q1", "continuous_finetune": "continuous"}
ARM_SOURCE_JOB = {"scratch": None, "finetune": "130.0", "continuous_finetune": "133.0"}
PREFIX = "gs://fcc-teststorage/bucket_results/foundation_pretrain/downstream_data/" + CAMPAIGN


def run(args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


def check_local(root):
    if Path.cwd().resolve() != root:
        raise ValueError(f"Run this launcher from its staged directory: {root}")
    receipt_path = root / LAUNCH
    if receipt_path.is_symlink():
        raise ValueError("Launch receipt cannot be a symlink")
    plan = json.loads(receipt_path.read_text())
    if (plan["campaign"] != CAMPAIGN or plan["arms"] != list(ARM_DATASET) or plan["submit"] != SUBMIT
            or plan["credential_path"] != str(KEY) or plan["pretraining_jobs"] != ARM_SOURCE_JOB
            or plan["identity_overlap_detected"] is not False):
        raise ValueError("Unexpected campaign, arm, credential path, or pretraining source")
    if set(plan["files"]) != {WORKER, SUBMIT, GUARD}:
        raise ValueError("Unexpected launch file inventory")
    for name, digest in plan["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed or missing launch file: {name}")
    if not KEY.is_file() or KEY.stat().st_size <= 0:
        raise ValueError("Missing existing submit-side worker credential")
    archives = plan["cloud_archives"]
    if set(archives) != {"runtime.tar.gz", "q1.tar.gz", "continuous.tar.gz"}:
        raise ValueError("Unexpected cloud archive inventory")
    for name, item in archives.items():
        if item["uri"] != f"{PREFIX}/{name}" or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError(f"Unpinned or misplaced cloud archive: {name}")
    if not re.fullmatch(r"[0-9a-f]{64}", plan["membership_sha256"]):
        raise ValueError("Invalid common membership checksum")
    if set(plan["datasets"]) != set(ARM_DATASET.values()):
        raise ValueError("Unexpected prepared representation inventory")
    for kind, dataset in plan["datasets"].items():
        name = kind + ".tar.gz"
        if (dataset["archive"] != name or dataset["archive_sha256"] != archives[name]["sha256"]
                or not re.fullmatch(r"[0-9a-f]{64}", dataset["manifest_sha256"])):
            raise ValueError(f"Unexpected representation pin: {kind}")
    if any(plan["datasets"]["q1"][field] == plan["datasets"]["continuous"][field]
           for field in ("archive_sha256", "manifest_sha256")):
        raise ValueError("Q1 and continuous must use different prepared representations")
    return plan


def check_existing(attempt):
    if attempt.exists() or attempt.is_symlink():
        raise ValueError(f"Existing submission receipt; refusing duplicates: {attempt}")
    where = f'HzzPaper0Campaign == "{CAMPAIGN}" || regexp("(^|/){WORKER[:-3]}[.]sh$", Cmd)'
    for tool in ("condor_q", "condor_history"):
        args = [tool, "-constraint", where] + (["-limit", "1"] if tool == "condor_history" else [])
        old = run(args + ["-af", "ClusterId", "ProcId", "JobStatus"])
        if old:
            raise ValueError(f"Existing campaign in {tool}; refusing duplicates: {old}")


def read_ads(text):
    """Support regular ClassAds and HTCondor's common-ad/proc-delta format."""
    blocks, current = [], {}
    for line in text.splitlines():
        value = line.strip()
        if not value or value in ("[", "]"):
            if current:
                blocks.append(current)
                current = {}
            continue
        if value.startswith("#"):
            continue
        pair = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.*?)\s*;?", value)
        if not pair:
            raise ValueError("Malformed dry-run ClassAd")
        key, val = pair[1].lower(), pair[2]
        if key in current:
            if "procid" not in current:
                raise ValueError("Duplicate common dry-run ClassAd field")
            blocks.append(current)
            current = {}
        current[key] = val
    if current:
        blocks.append(current)
    common, result = {}, []
    for block in blocks:
        if "procid" not in block:
            if result or set(common) & set(block):
                raise ValueError("Malformed common dry-run ClassAd")
            common.update(block)
            continue
        ad = dict(common)
        ad.update(block)
        result.append(ad)
        if len(result) == 1:
            common = ad
    return result


def inspect_dry_run(text, plan):
    ads = read_ads(text)
    if len(ads) != len(ARM_DATASET):
        raise ValueError("Expected exactly three independent Condor processes")
    for i, (arm, kind) in enumerate(ARM_DATASET.items()):
        ad = ads[i]
        args = shlex.split(json.loads(ad["args"]))
        data = plan["datasets"][kind]
        if (ad["procid"] != str(i) or ad["requestgpus"] != "1" or ad["requestcpus"] != "8"
                or ad["amscgpu_job"].lower() != "true" or Path(json.loads(ad["cmd"])).name != WORKER
                or len(args) != 3 or args[1:] != [str(i), arm] or json.loads(ad["hzzpaper0arm"]) != arm
                or json.loads(ad["hzzpaper0campaign"]) != CAMPAIGN
                or json.loads(ad["hzzpaper0manifestsha"]) != data["manifest_sha256"]
                or json.loads(ad["hzzpaper0membershipsha"]) != plan["membership_sha256"]):
            raise ValueError(f"Unexpected effective job or resources for {arm}")
        env = json.loads(ad.get("environment", ad.get("env")))
        expected = {"RUNTIME": plan["cloud_archives"]["runtime.tar.gz"]["sha256"],
                    "DATA": data["archive_sha256"], "MANIFEST": data["manifest_sha256"],
                    "MEMBERSHIP": plan["membership_sha256"]}
        for name, digest in expected.items():
            if re.findall(fr"(?:^|[;\s])EXPECTED_{name}_SHA256=([^;\s]+)", env) != [digest]:
                raise ValueError(f"Dry-run {name} checksum differs for {arm}")


def check(root, attempt):
    plan = check_local(root)
    check_existing(attempt)
    with tempfile.TemporaryDirectory(prefix="hzz-paper0-condor-check-") as temporary:
        ad = Path(temporary) / "classads.txt"
        run(["condor_submit", "-dry-run", str(ad), SUBMIT])
        inspect_dry_run(ad.read_text(), plan)
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
        print("READY: three independent one-GPU jobs (Q1 scratch, Q1 fine-tune, continuous-flat fine-tune). No jobs submitted.")
        print("Pinned common event membership:", plan["membership_sha256"])
        return
    fd = os.open(KEY.parent / ("." + CAMPAIGN + "_submit.lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Unsafe submit lock")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = check(root, attempt)
        state = dict(campaign=CAMPAIGN, status="submission_intent", directory=str(root),
                     datasets=plan["datasets"], membership_sha256=plan["membership_sha256"],
                     cloud_archives=plan["cloud_archives"])
        with attempt.open("x") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        output = run(["condor_submit", "-terse", SUBMIT])
        result = re.fullmatch(r"(\d+)\.0\s*-\s*\1\.2", output)
        if not result:
            raise RuntimeError("Unexpected submission response; inspect Condor before any retry: " + output)
        state.update(status="submitted", jobs={arm: f"{result[1]}.{i}" for i, arm in enumerate(ARM_DATASET)})
        attempt.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state["jobs"], indent=2))


if __name__ == "__main__":
    main()
