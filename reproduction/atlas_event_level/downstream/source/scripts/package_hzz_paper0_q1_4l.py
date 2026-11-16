#!/usr/bin/env python3
"""Seal the three Paper0 binary arms and both matched data representations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import re
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = "hzz_paper0_q1_4l_v1"
BUCKET = "gs://fcc-teststorage/bucket_results/foundation_pretrain"
BASE = ROOT / "condor/heptokens-canonical-multiclass-runtime-v1.tar.gz"
BASE_SHA = "9c1f3cc713910d50160b13de459e525c80371c8fe026171d5e88d1c334613d31"
PLAN_SHA = "6279af55912e85d0b8341b6364783a41b5304ee8ef0c1ad62f774bcf514119ce"
TRAINER = "scripts/train_atlas_paper0_q1_4l.py"
WORKER = "amscgpu_hzz_paper0_q1_4l.sh"
SUBMIT = "amscgpu_hzz_paper0_q1_4l.sub"
GUARD = "submit_hzz_paper0_q1_4l.py"
RECEIPT = "hzz_paper0_q1_4l_launch.json"
LAUNCH = "heptokens-hzz-paper0-three-arm-launch-v1.tar.gz"
CREDENTIAL = "/usatlas/u/magaras/condor_google/fccstorageuser.json"
ARM_DATASET = {"scratch": "q1", "finetune": "q1", "continuous_finetune": "continuous"}
ARM_SOURCE_JOB = {"scratch": None, "finetune": "130.0", "continuous_finetune": "133.0"}
SELECTION = "paper0_atlas_hzz_cms_core_q1_v1"
SPLITS = ("train", "val", "test")
CLASSES = ("background", "signal")


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def make_runtime(destination):
    if digest(BASE) != BASE_SHA:
        raise ValueError("Frozen canonical multiclass runtime SHA256 differs")
    trainer = ROOT / TRAINER
    if not trainer.is_file() or trainer.is_symlink() or not trainer.stat().st_size:
        raise ValueError("Missing Paper0 trainer; build only after the exact trainer is ready")
    compile(trainer.read_bytes(), str(trainer), "exec")
    seen = set()
    with tarfile.open(BASE, "r:gz") as source, tarfile.open(destination, "w:gz") as out:
        for info in source:
            name = PurePosixPath(info.name)
            if (not name.parts or name.is_absolute() or ".." in name.parts or name in seen
                    or name == PurePosixPath(TRAINER) or not (info.isfile() or info.isdir())
                    or name.name == "fccstorageuser.json"):
                raise ValueError(f"Unexpected canonical runtime archive member: {info.name}")
            seen.add(name)
            stream = source.extractfile(info) if info.isfile() else None
            if name.as_posix() == "multiclass_controls_plan.json":
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != PLAN_SHA:
                    raise ValueError("Pretraining source plan differs from the frozen original")
                stream = source.extractfile(info)
            out.addfile(info, stream)
        for required in ("multiclass_controls_plan.json", "condor/pixi-matched-multiclass.toml",
                         "condor/pixi-matched-multiclass.lock"):
            if PurePosixPath(required) not in seen:
                raise ValueError(f"Canonical runtime lacks required member: {required}")
        out.add(trainer, arcname=TRAINER, recursive=False)


def make_data(prepared, destination):
    manifest = prepared / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("Expected the prepared Paper0 data manifest.json")
    data = json.loads(manifest.read_text())
    if (data["selection_id"] != SELECTION or data["n_classes"] != 2 or data["token_shape"] != [256, 1]
            or data["identity_overlap_detected"] is not False or type(data["file_overlap_detected"]) is not bool):
        raise ValueError("Expected the event-disjoint selected binary cohort")
    membership = prepared / "membership.npz"
    if not membership.is_file() or membership.is_symlink() or digest(membership) != data["membership_sha256"]:
        raise ValueError("Selected event membership does not match the manifest")
    if set(data["split_counts"]) != set(SPLITS):
        raise ValueError("Incorrect split inventory")
    for split, counts in data["split_counts"].items():
        if (set(counts) != {*CLASSES, "total"} or any(type(counts[c]) is not int or counts[c] <= 0 for c in CLASSES)
                or counts["total"] != sum(counts[c] for c in CLASSES)):
            raise ValueError(f"Invalid class counts: {split}")
    train = data["split_counts"]["train"]
    if any(not math.isclose(data["recommended_cross_entropy_weights"][name], train["total"] / (2 * train[name]),
                            rel_tol=1e-10) for name in CLASSES):
        raise ValueError("Prepared class weights differ from inverse training frequencies")
    paths = sorted(prepared.rglob("*"))
    dirs = {PurePosixPath(split) for split in SPLITS}
    dirs.update(PurePosixPath(split, name) for split in SPLITS for name in CLASSES)
    shards = {}
    for path in paths:
        relative = PurePosixPath(path.relative_to(prepared).as_posix())
        if path.is_symlink():
            raise ValueError(f"Unsafe dataset archive member: {path}")
        if path.is_dir() and relative in dirs:
            continue
        if path.is_file() and path in (manifest, membership):
            continue
        if (not path.is_file() or len(relative.parts) != 3 or relative.parent not in dirs
                or not re.fullmatch(r"part-\d+\.parquet", relative.name)):
            raise ValueError(f"Unexpected dataset archive member: {path}")
        shards[relative.as_posix()] = digest(path)
    if shards != data["shards"] or {PurePosixPath(p).parent for p in shards} != {d for d in dirs if len(d.parts) == 2}:
        raise ValueError("Incorrect selected Parquet shard inventory or checksum")
    with tarfile.open(destination, "w:gz") as out:
        out.add(prepared, arcname="prepared", recursive=False)
        for path in paths:
            out.add(path, arcname="prepared/" + path.relative_to(prepared).as_posix(), recursive=False)
    return digest(manifest), data


def check_common_cohort(q1, continuous, q1_manifest_sha):
    if "continuous_preparation" in q1 or not isinstance(continuous.get("continuous_preparation"), dict):
        raise ValueError("Expected separate Q1 and continuous prepared representations")
    provenance = continuous["continuous_preparation"]
    if (provenance.get("source_selected_q1_manifest_sha256") != q1_manifest_sha
            or provenance.get("source_selected_q1_shards") != q1["shards"]):
        raise ValueError("Continuous data were not built from this exact selected Q1 manifest")
    if provenance.get("no_refit") is not True:
        raise ValueError("Continuous data must use the original frozen feature preprocessing")
    for key in ("selection", "selection_id", "membership_sha256", "split_counts", "source_files_by_split",
                "recommended_cross_entropy_weights", "identity_overlap_detected", "file_overlap_detected"):
        if q1[key] != continuous[key]:
            raise ValueError(f"Q1 and continuous cohorts differ: {key}")
    if set(q1["shards"]) != set(continuous["shards"]) or q1["shards"] == continuous["shards"]:
        raise ValueError("Continuous representation does not have independently materialized matched shards")


def launch_text(runtime_sha, datasets, membership_sha):
    rows = "\n".join(f"{arm} {datasets[kind]['archive_sha256']} {datasets[kind]['manifest_sha256']}"
                     for arm, kind in ARM_DATASET.items())
    return f'''universe = vanilla
executable = {WORKER}
arguments = $(ClusterId) $(ProcId) $(Arm)
environment = "EXPECTED_RUNTIME_SHA256={runtime_sha} EXPECTED_DATA_SHA256=$(DataSha) EXPECTED_MANIFEST_SHA256=$(ManifestSha) EXPECTED_MEMBERSHIP_SHA256={membership_sha}"
+amscgpu_job = true
+HzzPaper0Campaign = "{CAMPAIGN}"
+HzzPaper0Arm = "$(Arm)"
+HzzPaper0ManifestSHA = "$(ManifestSha)"
+HzzPaper0MembershipSHA = "{membership_sha}"
request_gpus = 1
request_cpus = 8
request_memory = 48GB
request_disk = 80GB
output = /dev/null
error = /dev/null
log = {CAMPAIGN}.$(ClusterId).log
notification = Never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {CREDENTIAL}
transfer_output_files =
queue Arm, DataSha, ManifestSha from (
{rows}
)
'''


def package(prepared, continuous, output):
    inputs = {"q1": prepared, "continuous": continuous}
    if any(p.is_symlink() or not p.is_dir() for p in inputs.values()):
        raise ValueError("Select the completed data directories, not symlinks")
    inputs = {kind: path.resolve() for kind, path in inputs.items()}
    output = output.resolve()
    if output.exists() or any(output == p or p in output.parents for p in inputs.values()):
        raise ValueError("Use a new output directory outside both prepared representations")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hzz-paper0-pack-", dir=output.parent) as scratch:
        stage = Path(scratch) / "sealed"
        stage.mkdir()
        runtime = stage / "runtime.tar.gz"
        make_runtime(runtime)
        runtime_sha = digest(runtime)
        datasets, manifests = {}, {}
        for kind, prepared in inputs.items():
            archive = stage / (kind + ".tar.gz")
            manifest_sha, manifests[kind] = make_data(prepared, archive)
            datasets[kind] = dict(archive=archive.name, archive_sha256=digest(archive), manifest_sha256=manifest_sha)
        cohort = manifests["q1"]
        check_common_cohort(cohort, manifests["continuous"], datasets["q1"]["manifest_sha256"])
        for name in (WORKER, GUARD):
            shutil.copy2(ROOT / "condor" / name, stage / name)
            (stage / name).chmod(0o755)
        (stage / SUBMIT).write_text(launch_text(runtime_sha, datasets, cohort["membership_sha256"]))
        target = f"{BUCKET}/downstream_data/{CAMPAIGN}"
        receipt = dict(campaign=CAMPAIGN, arms=list(ARM_DATASET), credential_path=CREDENTIAL, submit=SUBMIT,
                       datasets=datasets, membership_sha256=cohort["membership_sha256"], base_runtime_sha256=BASE_SHA,
                       selection_id=cohort["selection_id"], file_overlap_detected=cohort["file_overlap_detected"],
                       identity_overlap_detected=cohort["identity_overlap_detected"], split_counts=cohort["split_counts"],
                       pretraining_jobs=ARM_SOURCE_JOB, source_plan_sha256=PLAN_SHA,
                       files={name: digest(stage / name) for name in (WORKER, GUARD, SUBMIT)},
                       cloud_archives={name: {"uri": f"{target}/{name}", "sha256": sha}
                                       for name, sha in [(runtime.name, runtime_sha)] + [
                                           (d["archive"], d["archive_sha256"]) for d in datasets.values()]})
        (stage / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        with tarfile.open(stage / LAUNCH, "w:gz") as archive:
            for name in (WORKER, GUARD, SUBMIT, RECEIPT):
                archive.add(stage / name, arcname=name, recursive=False)
        os.rename(stage, output)
    print(json.dumps({"output": str(output), **receipt, "launch_sha256": digest(output / LAUNCH)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True, help="selected original Q1 data")
    parser.add_argument("--continuous-dir", type=Path, required=True, help="same selected events with continuous features")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    package(args.prepared_dir, args.continuous_dir, args.output_dir)


if __name__ == "__main__":
    main()
