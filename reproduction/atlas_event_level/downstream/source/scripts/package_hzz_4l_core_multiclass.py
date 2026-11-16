#!/usr/bin/env python3
"""Seal the 45 independent Paper0-core multiclass fits and their plot child."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from package_continuous_code_control import pack, read_archive, sha
from package_hzz_4l_multiclass import FAMILIES, make_plan, sha_file

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "hzz_4l_core_multiclass_v1"
SELECTION = "hzz_4l_paper0_core_v1"
PREFIX = "gs://fcc-teststorage/bucket_results/foundation_pretrain"
DATA_URI = f"{PREFIX}/downstream_data/{CAMPAIGN}"
RESULT_URI = f"{PREFIX}/downstream/{CAMPAIGN}"
BASE = ROOT / "output/hzz_4l_multiclass_v1/launch/runtime.tar.gz"
BASE_SHA = "4ff2583fbe46c885c0b3ae808336fecc0f78aa69312e30ed897d55fc6c673bd0"
SOURCE_SHA = "19cdf0a7afe372d9269df1213474e5945196c63c34b8b864b447b60197f8ba3a"
KEY = "/usatlas/u/magaras/condor_google/fccstorageuser.json"
WORKER = "amscgpu_hzz_4l_core_multiclass.sh"
GUARD = "submit_hzz_4l_core_multiclass.py"
DAG = "amscgpu_hzz_4l_core_multiclass.dag"
SUB = "amscgpu_hzz_4l_core_multiclass_fit.sub"
PLOT_SUB = "amscgpu_hzz_4l_core_multiclass_plots.sub"
RECEIPT = "hzz_4l_core_multiclass_launch.json"
BUNDLE = "heptokens-hzz-4l-core-multiclass-launch-v1.tar.gz"
KINDS = ("q1", "q4", "q8", "continuous")
NAMES = ("ggf", "vbf", "vh", "tth", "zz_continuum")
TOTALS = dict(train=300279, val=64264, test=64403)
PAYLOAD = (
    "scripts/hzz_4l_core_cohort.py",
    "scripts/hzz_4l_core_multiclass_study.py",
    "scripts/plot_hzz_4l_core_multiclass.py",
    "scripts/collect_hzz_4l_core_results.py",
    "configs/plots/hzz_4l_core_multiclass.json",
)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def read_json(path):
    return json.loads(path.read_text())


def make_runtime():
    members = read_archive(BASE, BASE_SHA)
    plan = make_plan(members["multiclass_controls_plan.json"])
    if json.loads(members["hzz_4l_multiclass_plan.json"]) != plan or len(plan["runs"]) != 45:
        raise ValueError("Frozen plan differs from the original 45 fit inventory")
    if set(members) & set(PAYLOAD):
        raise ValueError("Core runtime must be additive to the frozen comparison runtime")
    extra = {name: (ROOT / name).read_bytes() for name in PAYLOAD}
    for name, content in extra.items():
        if name.endswith(".py"):
            compile(content, name, "exec")
        else:
            json.loads(content)
    members.update(extra)
    members["hzz_4l_core_runtime_provenance.json"] = json_bytes(dict(
        original_runtime_sha256=BASE_SHA, original_plan_sha256=sha(members["hzz_4l_multiclass_plan.json"]),
        added_members_sha256={name: sha(content) for name, content in extra.items()}, changed_original_members=[]))
    return pack(members), plan


def exact_files(directory, names):
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"Missing selected payload or symlink: {directory}")
    paths = {}
    for path in directory.rglob("*"):
        rel = path.relative_to(directory).as_posix()
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError(f"Unsafe archive member: {path}")
        if path.is_file():
            paths[rel] = path
    if set(paths) != set(names):
        raise ValueError(f"Payload inventory differs in {directory}: extra={sorted(set(paths)-set(names))}, "
                         f"missing={sorted(set(names)-set(paths))}")
    return paths


def make_archive(destination, files):
    with destination.open("xb") as handle, gzip.GzipFile(fileobj=handle, filename="", mtime=0, mode="wb",
                                                           compresslevel=4) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, path in sorted(files.items()):
                rel = PurePosixPath(name)
                if rel.is_absolute() or not rel.parts or ".." in rel.parts or rel.name == "fccstorageuser.json":
                    raise ValueError(f"Unsafe selected archive name: {name}")
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Unsafe selected archive source: {path}")
                entry = tarfile.TarInfo(name)
                entry.size = path.stat().st_size
                entry.mode = 0o644
                with path.open("rb") as stream:
                    archive.addfile(entry, stream)


def collect_common(study, manifests):
    selected, index = study / "selection", study / "canonical_index"
    selection_files = exact_files(selected, ("selection_receipt.json", "membership.npz", "source_manifest.json"))
    sel = read_json(selected / "selection_receipt.json")
    membership = sha_file(selected / "membership.npz")
    receipt_sha = sha_file(selected / "selection_receipt.json")
    if (sel["passed"] is not True or sel["selection_id"] != SELECTION or tuple(sel["class_names"]) != NAMES
            or sel["membership_sha256"] != membership or sel["source_manifest_sha256"] != SOURCE_SHA
            or sha_file(selected / "source_manifest.json") != SOURCE_SHA
            or {key: value["total"] for key, value in sel["split_counts"].items()} != TOTALS):
        raise ValueError("Selection is not the expected five-class Paper0-core cohort")
    schedule_specs = {(seed, epoch): f"train_seed{seed}_epoch{epoch:02d}"
                      for seed in (42, 43, 44) for epoch in range(10)}
    index_files = exact_files(index, ["index_manifest.json", "schedule_campaign.json"] +
                              [split + ".npz" for split in TOTALS] +
                              [f"schedules/{stem}.{suffix}" for stem in schedule_specs.values()
                               for suffix in ("npz", "json")])
    index_receipt = read_json(index / "index_manifest.json")
    campaign = read_json(index / "schedule_campaign.json")
    q8_manifest_sha = sha_file(manifests["q8"])
    batches, dropped = divmod(TOTALS["train"], 256)
    if (campaign["passed"] is not True or campaign["selection_id"] != SELECTION
            or campaign["selection_receipt_sha256"] != receipt_sha or campaign["membership_sha256"] != membership
            or campaign["split_counts"] != sel["split_counts"]
            or campaign["selected_q8_manifest_sha256"] != q8_manifest_sha
            or index_receipt["source_manifest_sha256"] != q8_manifest_sha
            or tuple(index_receipt["class_names"]) != NAMES
            or campaign["index_manifest_sha256"] != sha_file(index / "index_manifest.json")
            or campaign["index_identity_sha256"] != index_receipt["index_identity_sha256"]
            or campaign["updates_per_epoch"] != batches or campaign["dropped_rows_per_epoch"] != dropped
            or campaign["full_optimizer_steps"] != batches * 10
            or campaign["settings"] != dict(batch_size=256, drop_last=True, epochs=10, logical_chunk_size=128,
                                            seeds=[42, 43, 44], shuffle_buffer_size=8192)):
        raise ValueError("Canonical schedule was not built for the selected core cohort and frozen protocol")
    if set(index_receipt["splits"]) != set(TOTALS):
        raise ValueError("Canonical index has wrong splits")
    for split, total in TOTALS.items():
        spec = index_receipt["splits"][split]
        if spec["file"] != split + ".npz" or spec["rows"] != total or spec["file_sha256"] != sha_file(index / spec["file"]):
            raise ValueError(f"Canonical index split changed: {split}")
    schedule_rows = {(row["seed"], row["epoch"]): row for row in campaign["schedules"]}
    if len(campaign["schedules"]) != 30 or set(schedule_rows) != set(schedule_specs):
        raise ValueError("Wrong canonical schedule inventory")
    for pair, stem in schedule_specs.items():
        row = schedule_rows[pair]
        if ({name: value for name, value in row.items() if name != "file"}
                != read_json(index / "schedules" / (stem + ".json"))
                or row["file"] != stem + ".npz" or row["full_rows"] != TOTALS["train"]
                or row["index_identity_sha256"] != campaign["index_identity_sha256"]
                or row["batches"] != batches or row["dropped_rows"] != dropped
                or row["archive_sha256"] != sha_file(index / "schedules" / row["file"])):
            raise ValueError(f"Canonical schedule changed: {stem}")
    files = {f"selection/{name}": path for name, path in selection_files.items()}
    files.update({f"canonical_index/{name}": path for name, path in index_files.items()})
    return files, sel, receipt_sha, campaign


def collect_representation(directory, kind, selection, receipt_sha):
    view = read_json(directory / "selection_view.json")
    if (view["passed"] is not True or view["representation"] != kind or view["selection_id"] != SELECTION
            or view["membership_sha256"] != selection["membership_sha256"]
            or view["membership_identity_sha256"] != selection["membership_identity_sha256"]
            or view["selection_receipt_sha256"] != receipt_sha or view["split_counts"] != selection["split_counts"]
            or view["recommended_cross_entropy_weights"] != selection["recommended_cross_entropy_weights"]
            or view["manifest_sha256"] != sha_file(directory / "manifest.json")
            or view["parent_manifest_sha256"] != sha_file(directory / "parent_manifest.json")):
        raise ValueError(f"Representation receipt differs from common cohort: {kind}")
    names = {name for name, shard in view["shards"].items() if shard["rows"] > 0} | {
        "manifest.json", "parent_manifest.json", "selection_view.json"}
    files = exact_files(directory, names)
    counts = {split: dict.fromkeys(NAMES, 0) for split in TOTALS}
    for name, shard in view["shards"].items():
        rel = PurePosixPath(name)
        if (len(rel.parts) != 3 or rel.parts[0] not in TOTALS or rel.parts[1] not in NAMES
                or rel.suffix != ".parquet" or type(shard["rows"]) is not int or shard["rows"] < 0
                or (shard["rows"] > 0 and shard["sha256"] != sha_file(files[name]))
                or (shard["rows"] == 0 and shard["sha256"] is not None)):
            raise ValueError(f"Unpinned selected shard: {kind}/{name}")
        counts[rel.parts[0]][rel.parts[1]] += shard["rows"]
    for split, spec in counts.items():
        if {**spec, "total": sum(spec.values())} != selection["split_counts"][split]:
            raise ValueError(f"Selected representation shard counts differ: {kind}/{split}")
    return {f"{kind}/{name}": path for name, path in files.items()}


def make_submit(plot, runtime_sha, common_sha, membership):
    env = (f"EXPECTED_RUNTIME_SHA256={runtime_sha} EXPECTED_COMMON_SHA256={common_sha} "
           f"EXPECTED_MEMBERSHIP_SHA256={membership} EXPECTED_SOURCE_MANIFEST_SHA256={SOURCE_SHA}")
    if plot:
        mode, run, kind, gpus, cpus, memory, disk = "plot", "analysis", "common", 0, 2, 16, 35
    else:
        mode, run, kind, gpus, cpus, memory, disk = "train", "$(RunKey)", "$(DataKind)", 1, 8, 48, 80
        env += " EXPECTED_DATA_SHA256=$(DataSha)"
    return f'''universe = vanilla
executable = {WORKER}
arguments = $(ClusterId) $(ProcId) {mode} {run} {kind}
environment = "{env}"
+amscgpu_job = true
+HzzFourLeptonCampaign = "{CAMPAIGN}"
+HzzFourLeptonRun = "{run}"
+HzzFourLeptonStage = "{mode}"
+HzzFourLeptonRuntimeSha = "{runtime_sha}"
request_gpus = {gpus}
request_cpus = {cpus}
request_memory = {memory}GB
request_disk = {disk}GB
output = /dev/null
error = /dev/null
log = {CAMPAIGN}.$(ClusterId).log
notification = Never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {KEY}
transfer_output_files =
queue 1
'''.encode()


def make_dag(plan, hashes):
    rows, nodes = [f"JOB PLOTS {PLOT_SUB}"], []
    for i, (key, run) in enumerate(plan["runs"].items()):
        node = f"FIT_{i:02d}"
        kind = FAMILIES[run["model_family"]]
        nodes.append(node)
        rows += [f"JOB {node} {SUB}", f'VARS {node} RunKey="{key}" DataKind="{kind}" DataSha="{hashes[kind]}"']
    if len(nodes) != 45:
        raise ValueError("Require exactly 45 independent fit nodes")
    rows.append(f"PARENT {' '.join(nodes)} CHILD PLOTS")
    return ("\n".join(rows) + "\n").encode()


def stage(study, output):
    study, output = study.resolve(), output.resolve()
    if output.exists() or output == study or study in output.parents:
        raise ValueError("Choose a new launch directory outside the selected study data")
    runtime, plan = make_runtime()
    common, selection, receipt_sha, schedule = collect_common(study, {kind: study / kind / "manifest.json" for kind in KINDS})
    archive_files = {"common": common}
    archive_files.update({kind: collect_representation(study / kind, kind, selection, receipt_sha) for kind in KINDS})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hzz-4l-core-launch-", dir=output.parent) as temporary:
        target = Path(temporary) / "sealed"
        target.mkdir()
        (target / "runtime.tar.gz").write_bytes(runtime)
        data = {}
        for kind, files in archive_files.items():
            path = target / (kind + ".tar.gz")
            make_archive(path, files)
            data[kind] = dict(archive=path.name, sha256=sha_file(path), bytes=path.stat().st_size)
        runtime_sha, common_sha, membership = sha(runtime), data["common"]["sha256"], selection["membership_sha256"]
        files = {name: (ROOT / "condor" / name).read_bytes() for name in (WORKER, GUARD)}
        files.update({SUB: make_submit(False, runtime_sha, common_sha, membership),
                      PLOT_SUB: make_submit(True, runtime_sha, common_sha, membership),
                      DAG: make_dag(plan, {kind: data[kind]["sha256"] for kind in KINDS})})
        receipt = dict(format_version=1, campaign=CAMPAIGN, selection_id=SELECTION, data_prefix=DATA_URI,
                       result_prefix=RESULT_URI, credential_path=KEY, dag=DAG, inputs=data,
                       runtime=dict(archive="runtime.tar.gz", sha256=runtime_sha, bytes=len(runtime)),
                       original_runtime_sha256=BASE_SHA, membership_sha256=membership,
                       selection_receipt_sha256=receipt_sha, source_manifest_sha256=SOURCE_SHA,
                       split_counts=selection["split_counts"], canonical_index_sha256=schedule["index_identity_sha256"],
                       run_keys=list(plan["runs"]), token_fits=27, continuous_fits=18,
                       continuous_preparation_jobs=0, plot_jobs=1, launch_files={name: sha(raw) for name, raw in files.items()})
        files[RECEIPT] = json_bytes(receipt)
        for name, raw in files.items():
            path = target / name
            path.write_bytes(raw)
            if name == WORKER:
                path.chmod(0o755)
        (target / BUNDLE).write_bytes(pack(files))
        os.rename(target, output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.study_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
