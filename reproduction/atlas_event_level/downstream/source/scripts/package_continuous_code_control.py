#!/usr/bin/env python3
"""Build one isolated, complete runtime and six-GPU smoke-gated launch bundle.

Standard-library only: no Pixi, cloud CLI or ML installation needed to build.
Frozen reference code is merged into a NEW archive; originals are not edited.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CONDOR = ROOT / "condor"
ARCHIVE = "heptokens-continuous-code-control-runtime-v1.tar.gz"
BUNDLE = "heptokens-continuous-code-control-launch-v1.tar.gz"
WORKER = "amscgpu_continuous_code_control_6gpu.sh"
SUBMIT = "amscgpu_continuous_code_control_6gpu.sub"
CHECKER = "submit_continuous_code_control.py"
PLAN = "continuous_code_control_launch.json"
PAYLOAD = (
    "src/heptokens/models/foundation_continuous_code_control.py",
    "src/heptokens/data/continuous_code_control.py",
    "configs/model/foundation_continuous_code_control.yaml",
    "configs/datamodule/continuous_code_control.yaml",
    "configs/callbacks/continuous_code_control.yaml",
    "scripts/continuous_code_control_checks.py",
    "scripts/check_continuous_code_control_ddp.py",
    "tests/test_continuous_code_control.py",
)


def sha(data): return hashlib.sha256(data).hexdigest()


def pack(files):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for name, content in sorted(files.items()):
                entry = tarfile.TarInfo(name)
                entry.size = len(content)
                entry.mode = 0o755 if name.endswith(".sh") else 0o644
                archive.addfile(entry, io.BytesIO(content))
    return buffer.getvalue()


def read_archive(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected: raise ValueError(f"Frozen reference archive changed: {path}")
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isdir(): continue
            if not member.isfile() or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if Path(name).name == "fccstorageuser.json":
                raise ValueError("Credentials must not enter a runtime archive")
            result[name] = archive.extractfile(member).read()
    return result


def main():
    sources = {
        "base": (ROOT / "heptokens-google-data-grl-b300.ab256.tar.gz",
            "ab256c9feb6aab2195570a67f5ede231f95ed5bd26dad329407b6a2b8f88d558"),
        "reference_214_overlay": (CONDOR / "heptokens-q8-prefix-study-update.tar.gz",
            "a98bef05f942bff80e913aa6ce38544a33f27fa91479f7299e478fbd752b1c44"),
    }
    payload = {}
    for path, expected in sources.values(): payload.update(read_archive(path, expected))
    payload.update({name: (ROOT / name).read_bytes() for name in PAYLOAD})
    payload["continuous_code_control_source_manifest.json"] = json.dumps(dict(
        sources={name: expected for name, (_, expected) in sources.items()},
        new_files={name: sha(payload[name]) for name in PAYLOAD},
        note="Common model/loader code frozen from successful reference 214; separate runtime only"), indent=2).encode()
    runtime = pack(payload)
    submit = f'''universe = vanilla
executable = {WORKER}
arguments = $(ClusterId) $(ProcId)
environment = "EXPECTED_ARCHIVE_SHA256={sha(runtime)} MODE=smoke_then_full"
+amscgpu_job = true
request_gpus = 6
request_cpus = 48
request_memory = 96GB
request_disk = 200GB
output = logs/amscgpu_continuous_code_control.$(ClusterId).$(ProcId).out
error = logs/amscgpu_continuous_code_control.$(ClusterId).$(ProcId).err
log = logs/amscgpu_continuous_code_control.$(ClusterId).$(ProcId).log
stream_output = True
stream_error = True
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {ARCHIVE},fccstorageuser.json,pixi-continuous-code-control.toml,pixi-continuous-code-control.lock,continuous-code-control-nccl.py
transfer_output_files =
queue 1
'''.encode()
    files = {ARCHIVE: runtime, SUBMIT: submit, WORKER: (CONDOR / WORKER).read_bytes(),
             CHECKER: (CONDOR / CHECKER).read_bytes()}
    for alias, source, expected in (
        ("pixi-continuous-code-control.toml", "pixi-hier-kin-aux-seed42.toml", "30dddb84cca51abb684936c23cf677f04698eba96503dd89d823775aa922e752"),
        ("pixi-continuous-code-control.lock", "pixi-hier-kin-aux-seed42.lock", "ba154627f585bf1013dbfbca459589a4a78fd04a3d004f9b84a6b74717507aa9"),
        ("continuous-code-control-nccl.py", "b300-hier-kin-aux-nccl.py", "1a6e7315774b528e8aa84be66a9e7cbba2c9e98cebd29c52a4ba08e390c15ef7"),
    ):
        raw = (CONDOR / source).read_bytes()
        if sha(raw) != expected: raise ValueError(f"Known working runtime input changed: {source}")
        files[alias] = raw
    files[PLAN] = json.dumps(dict(submit_file=SUBMIT, reference_job="214.0",
        devices=6, batch_size=512, global_batch_size=3072, full_epochs=10,
        files={name: sha(raw) for name, raw in files.items()},
        stages=["paired_data_checks", "200_step_smoke", "fresh_10_epoch_pretraining"],
        outputs="gs://fcc-teststorage/bucket_results/foundation_pretrain/data_grl_data16_200m_v1_pretrain/continuous_code_control_v1"), indent=2).encode()
    for name, raw in files.items():
        (CONDOR / name).write_bytes(raw)
        if name.endswith(".sh"): (CONDOR / name).chmod(0o755)
    bundle = pack(files)
    (CONDOR / BUNDLE).write_bytes(bundle)
    print(f"Bundle: {CONDOR / BUNDLE}\nSHA256: {sha(bundle)}\nRuntime SHA256: {sha(runtime)}")
    print("Prepared only. One six-GPU job: validated 200-step smoke -> fresh 10 epochs; no fine-tunes submitted.")


if __name__ == "__main__": main()
