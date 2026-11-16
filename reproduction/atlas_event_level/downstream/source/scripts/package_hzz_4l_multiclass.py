#!/usr/bin/env python3
"""Seal the isolated ≥4-lepton multiclass worker, runtime and DAG launch files."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

from package_continuous_code_control import pack, read_archive, sha

ROOT = Path(__file__).resolve().parents[1]
CONDOR = ROOT / "condor"
BASE = CONDOR / "heptokens-canonical-multiclass-runtime-v1.tar.gz"
BASE_SHA = "9c1f3cc713910d50160b13de459e525c80371c8fe026171d5e88d1c334613d31"
BASE_PLAN = CONDOR / "multiclass_controls_plan.json"
PREFIX = "gs://fcc-teststorage/bucket_results/foundation_pretrain"
DATA_URI = f"{PREFIX}/downstream_data/hzz_4l_multiclass_v1"
RESULT_URI = f"{PREFIX}/downstream/hzz_4l_multiclass_v1"
KEY_PATH = "/usatlas/u/magaras/condor_google/fccstorageuser.json"
WORKER = "amscgpu_hzz_4l_multiclass.sh"
DAG = "amscgpu_hzz_4l_multiclass.dag"
FAMILIES = dict(q1_projected="q1", q4="q4", q8="q8", continuous_flat="continuous",
                hierarchical_kin_aux="continuous")
PAYLOAD = (
    "scripts/prepare_hzz_4l_multiclass.py",
    "scripts/hzz_4l_multiclass_study.py",
    "scripts/collect_hzz_4l_results.py",
    "scripts/plot_hzz_4l_multiclass.py",
    "scripts/plot_multiclass_conditions.py",
    "scripts/plot_multiclass_seed_uncertainty.py",
    "scripts/plot_multiclass_studies.py",
    "configs/plots/hzz_4l_multiclass.json",
)
SUBS = dict(prepare="amscgpu_hzz_4l_multiclass_prepare.sub",
            token="amscgpu_hzz_4l_multiclass_token.sub",
            continuous="amscgpu_hzz_4l_multiclass_continuous.sub",
            plots="amscgpu_hzz_4l_multiclass_plots.sub")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def make_plan(base):
    plan = json.loads(base)
    if set(plan["representations"]) != set(FAMILIES) or len(plan["sources"]) != 15:
        raise ValueError("Expected the original five paper representations and fifteen pretrained sources")
    runs = {}
    for family, kind in FAMILIES.items():
        rep = plan["representations"][family]
        if rep["data_kind"] != kind:
            raise ValueError("Original representation changed: " + family)
        for condition in ("finetune", "scratch", "frozen"):
            for seed in (42, 43, 44):
                scratch = condition == "scratch"
                suffix = f"{'s' if scratch else 'p'}{seed}"
                key = f"{family}_{condition}_{suffix}"
                source_key = None if scratch else f"{family}_p{seed}"
                if not scratch:
                    src = plan["sources"][source_key]
                    if (src["model_family"] != family or src["pretrain_seed"] != seed
                            or src["pretraining_configuration_id"] != rep["configuration_id"]):
                        raise ValueError("Pretrained source identity drift: " + source_key)
                    for field in ("checkpoint", "pretrain_config", "pretrain_status",
                                  "pretrain_completion", "pretraining_vocabulary_shard"):
                        uri, digest = src.get(field), src.get(field + "_sha256")
                        if field in ("pretrain_completion", "pretraining_vocabulary_shard") and not (uri or digest):
                            continue
                        if not (uri and uri.startswith("gs://fcc-teststorage/") and not any(c.isspace() for c in uri)
                                and digest and re.fullmatch(r"[a-f0-9]{64}", digest)):
                            raise ValueError(f"Unpinned source {source_key}/{field}")
                runs[key] = dict(model_family=family, condition=condition,
                                 training_seed=seed if scratch else 42,
                                 pretrain_seed=None if scratch else seed, source_key=source_key)
    plan["runs"] = runs
    return plan


def submit(stage, runtime_sha, common_sha, membership_sha):
    common = (f"EXPECTED_RUNTIME_SHA256={runtime_sha} EXPECTED_COMMON_SHA256={common_sha} "
              f"EXPECTED_MEMBERSHIP_SHA256={membership_sha}")
    memory, cpus, disk = {"prepare": (64, 8, 100), "token": (48, 8, 80),
                          "continuous": (48, 8, 80), "plots": (16, 2, 35)}[stage]
    xfer_in, xfer_out = KEY_PATH, ""
    if stage == "prepare":
        mode, run, kind = "prepare", "prepare_continuous", "continuous"
        common += " TRANSFER_PREP_RECEIPT=1"
        xfer_out = "continuous_ready.json"
    elif stage == "plots":
        mode, run, kind = "plot", "analysis", "common"
    else:
        mode, run, kind = "train", "$(RunKey)", "$(DataKind)"
        if stage == "continuous":
            common += " USE_TRANSFERRED_PREP_RECEIPT=1"
            xfer_in += ",continuous_ready.json"
        else:
            common += " EXPECTED_DATA_SHA256=$(DataSha)"
    return f'''universe = vanilla
executable = {WORKER}
arguments = $(ClusterId) $(ProcId) {mode} {run} {kind}
environment = "{common}"
+amscgpu_job = true
+HzzFourLeptonCampaign = "hzz_4l_multiclass_v1"
+HzzFourLeptonRun = "{run}"
+HzzFourLeptonStage = "{stage}"
+HzzFourLeptonRuntimeSha = "{runtime_sha}"
request_gpus = {int(mode == 'train')}
request_cpus = {cpus}
request_memory = {memory}GB
request_disk = {disk}GB
output = /dev/null
error = /dev/null
log = hzz_4l_multiclass.$(ClusterId).log
notification = Never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {xfer_in}
transfer_output_files = {xfer_out}
queue 1
'''.encode()


def make_dag(plan, data_hashes):
    rows = [f"JOB PREPARE_CONTINUOUS {SUBS['prepare']}", f"JOB PLOTS {SUBS['plots']}"]
    token, continuous = [], []
    for i, (key, run) in enumerate(plan["runs"].items()):
        kind = FAMILIES[run["model_family"]]
        node = f"FIT_{i:02d}"
        if kind == "continuous":
            continuous.append(node)
            rows += [f"JOB {node} {SUBS['continuous']}",
                     f'VARS {node} RunKey="{key}" DataKind="{kind}"']
        else:
            token.append(node)
            rows += [f"JOB {node} {SUBS['token']}",
                     f'VARS {node} RunKey="{key}" DataKind="{kind}" DataSha="{data_hashes[kind]}"']
    if (len(token), len(continuous)) != (27, 18):
        raise ValueError("Wrong DAG fit inventory")
    rows += [f"PARENT PREPARE_CONTINUOUS CHILD {' '.join(continuous)}",
             f"PARENT {' '.join(token)} CHILD PLOTS",
             f"PARENT {' '.join(continuous)} CHILD PLOTS"]
    return ("\n".join(rows) + "\n").encode()


def stage_payload(output, common, representations):
    # No submit-side tarball except a tiny launch bundle. Dataset tarballs are
    # produced separately from selected Parquet views and uploaded to GCS.
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing launch output: " + str(output))
    with tarfile.open(common, "r:gz") as tar:
        membership = sha(tar.extractfile("selection/membership.npz").read())
        selection = json.load(tar.extractfile("selection/selection_receipt.json"))
        tar.getmember("canonical_index/index_manifest.json")
        if membership != selection.get("membership_sha256") or not selection.get("passed"):
            raise ValueError("Invalid membership in the common bundle")
    data = dict(common=dict(archive="common.tar.gz", sha256=sha_file(common), bytes=common.stat().st_size))
    for kind, path in representations.items():
        with tarfile.open(path, "r:gz") as tar:
            tar.getmember(f"{kind}/manifest.json")
            view = json.load(tar.extractfile(f"{kind}/selection_view.json"))
            if (not view.get("passed") or view.get("membership_sha256") != membership
                    or view.get("representation") != kind):
                raise ValueError("Selected archive does not match common membership: " + kind)
        data[kind] = dict(archive=f"{kind}.tar.gz", sha256=sha_file(path), bytes=path.stat().st_size)

    runtime_members = read_archive(BASE, BASE_SHA)
    original_hashes = {p: sha(v) for p, v in runtime_members.items()}
    base_plan = BASE_PLAN.read_bytes()
    if runtime_members.get("multiclass_controls_plan.json") != base_plan:
        raise ValueError("Working multiclass control plan differs from the frozen archive")
    plan = make_plan(base_plan)
    added = {name: (ROOT / name).read_bytes() for name in PAYLOAD}
    added["hzz_4l_multiclass_plan.json"] = json_bytes(plan)
    if set(added) & set(runtime_members):
        raise ValueError("New runtime payload would overwrite frozen original code")
    runtime_members.update(added)
    runtime_members["hzz_4l_runtime_provenance.json"] = json_bytes(dict(
        original_runtime_sha256=BASE_SHA, original_members_sha256=original_hashes,
        added_members_sha256={p: sha(v) for p, v in added.items()}, changed_original_members=[]))
    runtime = pack(runtime_members)
    runtime_sha = sha(runtime)
    artifacts = {WORKER: (CONDOR / WORKER).read_bytes()}
    artifacts.update({name: submit(stage, runtime_sha, data["common"]["sha256"], membership)
                      for stage, name in SUBS.items()})
    artifacts[DAG] = make_dag(plan, {kind: data[kind]["sha256"] for kind in representations})
    artifacts["submit_hzz_4l_multiclass.py"] = (CONDOR / "submit_hzz_4l_multiclass.py").read_bytes()
    receipt = dict(format_version=1, campaign="hzz_4l_multiclass_v1", data_prefix=DATA_URI,
                   result_prefix=RESULT_URI, credential_path=KEY_PATH, dag=DAG,
                   runtime=dict(archive="runtime.tar.gz", sha256=runtime_sha, bytes=len(runtime)),
                   original_runtime_sha256=BASE_SHA, membership_sha256=membership, inputs=data,
                   run_keys=list(plan["runs"]), token_fits=27, continuous_fits=18,
                   continuous_preparation_jobs=1, plot_jobs=1,
                   launch_files={name: sha(value) for name, value in artifacts.items()})
    artifacts["hzz_4l_multiclass_launch.json"] = json_bytes(receipt)
    output.mkdir(parents=True)
    (output / "runtime.tar.gz").write_bytes(runtime)
    for name, value in artifacts.items():
        path = output / name
        path.write_bytes(value)
        if name == WORKER:
            path.chmod(0o755)
    (output / "heptokens-hzz-4l-launch-v1.tar.gz").write_bytes(pack(artifacts))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "common", "q1", "q4", "q8"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_payload(args.output_dir, args.common, {k: getattr(args, k) for k in ("q1", "q4", "q8")}), indent=2))


if __name__ == "__main__":
    main()
