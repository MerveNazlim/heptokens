#!/usr/bin/env python3
"""Package the six FINAL Zephyr Q4 tokenizers, not the cloud Stage-1 scans.

CPU-only. Sources are never changed; no credentials/H5/Parquets are packaged.
The preprocessing hashes are pinned to the original Q1/Q8 export bundle
61690be5b54807855f1bf0c8e736c8029ef881abf2a4e97e8a07321057b6a514.
Run gcloud separately as magaras, NEVER through sudo.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

import torch

ARCHIVE_NAME = "final-q4-data-grl-tokenizer-artifacts.tar.gz"
BUNDLE_ROOT = "data_grl_q4_tokenizer_artifacts"
SPECS = {
    "electrons": (8192, "electrons_log_standard.joblib"),
    "muons": (8192, "muons_log_standard.joblib"),
    "photons": (8192, "photons_log_standard.joblib"),
    "jets": (16384, "jets_log_standard.joblib"),
    "taus": (8192, "taus_log_standard.joblib"),
    "tracks": (16384, "tracks_log_standard_no_ndoflog.joblib"),
}
PREPROCESS_SHA256 = {
    "electrons": "3545d07fb897abd03b402a42b1e36ae7c7a1210f35b1cc4227ba4aa9e347d04a",
    "jets": "2a98f09134db923fde152500a8ed3fbc12b45a0ef9dcd0dae08ccb69cf18b492",
    "muons": "3bac37a3524c855fa4716a34be39f0fd96c2d4e451fab3c1f7be14a2401cbffd",
    "photons": "7f69cec8b9454c3090d425d8259ebfb1bb79de505e923bdf26629730b5829943",
    "taus": "ff5336eeea7f6dc75a0bf80683157b958bb1cbe65720b606e2b1248e1b4942c1",
    "tracks": "c03c0a11fb85acaadf519e5ed885136f0658fa5edbed781e6706ba38df2b36ad",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_checkpoint(path, object_name, codebook_size, *, require_finished=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hp = checkpoint["hyper_parameters"]
    expected = dict(num_quantizers=4, codebook_dim=16, codebook_size=codebook_size)
    for key, value in expected.items():
        if hp.get(key) != value:
            raise ValueError(f"{object_name}: {path}: {key}={hp.get(key)!r}; expected {value}")
    if int(checkpoint.get("global_step", 0)) <= 0:
        raise ValueError(f"{object_name}: checkpoint has no optimizer steps")
    if require_finished and checkpoint.get("epoch") != 19:
        raise ValueError(f"{object_name}: final last.ckpt epoch={checkpoint.get('epoch')!r}; expected 19 (20 completed epochs)")
    state = checkpoint["state_dict"]
    levels = {int(m.group(1)) for key in state
              if (m := re.match(r"vector_quantization\.layers\.(\d+)\.", key))}
    if levels != {0, 1, 2, 3}:
        raise ValueError(f"{object_name}: actual state does not contain exactly four quantizers: {levels}")
    for q in range(4):
        prefix = f"vector_quantization.layers.{q}."
        keys = [key for key in (prefix + "_codebook.embed", prefix + "embed") if key in state]
        if len(keys) != 1:
            raise ValueError(f"{object_name}: missing/ambiguous codebook {q}")
        key = keys[0]
        expected_shape = (1, codebook_size, 16) if key.endswith("_codebook.embed") else (16, codebook_size)
        if tuple(state[key].shape) != expected_shape:
            raise ValueError(f"{object_name}: wrong codebook tensor shape at {key}")
    for key, value in state.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError(f"{object_name}: non-finite weights at {key}")
    return dict(epoch=int(checkpoint["epoch"]), global_step=int(checkpoint["global_step"]), **expected)


def audit_sources(results_dir):
    results_dir = Path(results_dir)
    run_base = results_dir / "atlas_object_final_tokenizers_new_mcdata"
    prep_base = results_dir / "preprocessing/atlas_object_final_tokenizers_new_mcdata"
    files, records = {}, {}
    for name, (k, preprocessor_name) in SPECS.items():
        run = run_base / f"{name}_full_dim16_cb{k}_q4_e20_new_mcdata"
        last = run / "checkpoints/last.ckpt"
        best = run / "checkpoints/best.ckpt"
        preprocessor = prep_base / preprocessor_name
        for required in (last, preprocessor):
            if not required.is_file() or required.stat().st_size == 0:
                raise FileNotFoundError(f"Required final Q4 artifact missing: {required}")
        completion = check_checkpoint(last, name, k, require_finished=True)
        selected = best if best.is_file() else last
        selected_info = check_checkpoint(selected, name, k)
        preprocessor_hash = sha256(preprocessor)
        if preprocessor_hash != PREPROCESS_SHA256[name]:
            raise ValueError(f"{name}: preprocessor differs from the original Q1/Q8 export; STOP, do not refit or silently replace it")
        files[f"q4/{name}.ckpt"] = selected
        files[f"preprocessing/{name}.joblib"] = preprocessor
        records[name] = dict(source_run=str(run), checkpoint=str(selected),
                             checkpoint_sha256=sha256(selected), preprocessor=str(preprocessor),
                             preprocessor_sha256=preprocessor_hash,
                             completion=completion, selected_checkpoint=selected_info)
        print(f"PASS {name}: final Q4/dim16/cb{k}; last epoch=19; selected={selected.name}; matching Q1/Q8 preprocessing", flush=True)
    return files, records


def package(results_dir, output_dir):
    output_dir = Path(output_dir)
    archive_path = output_dir / ARCHIVE_NAME
    checksum_path = output_dir / (ARCHIVE_NAME + ".sha256")
    if archive_path.exists() or checksum_path.exists():
        raise FileExistsError("Bundle already exists; keep it unchanged rather than packaging/uploading twice")
    files, records = audit_sources(results_dir)  # Audit ALL six before writing anything.
    manifest = dict(format_version=1, purpose="final Q4 tokenisation of the matched data-GRL campaign",
                    selection="best.ckpt when available, else last.ckpt; verified final last.ckpt epoch=19",
                    tokenizer_training_sample="final new MC+data", source_results=str(results_dir),
                    num_quantizers=4, codebook_dim=16, objects=records,
                    reference_q1_q8_bundle_sha256="61690be5b54807855f1bf0c8e736c8029ef881abf2a4e97e8a07321057b6a514")
    sums = {name: sha256(path) for name, path in files.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".q4-bundle-", suffix=".tar.gz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for name, path in sorted(files.items()):
                # Dereference an explicitly chosen checkpoint, but package no directories/symlinks.
                entry = tarfile.TarInfo(f"{BUNDLE_ROOT}/{name}")
                entry.size, entry.mode = path.stat().st_size, 0o644
                with path.open("rb") as source:
                    archive.addfile(entry, source)
            for name, content in (
                ("manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()),
                ("SHA256SUMS", "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())).encode()),
            ):
                entry = tarfile.TarInfo(f"{BUNDLE_ROOT}/{name}")
                entry.size, entry.mode = len(content), 0o644
                archive.addfile(entry, io.BytesIO(content))
        # Reject a checkpoint/preprocessor changed while it was being bundled.
        with tarfile.open(temporary, "r:gz") as archive:
            for name, expected_hash in sums.items():
                digest = hashlib.sha256()
                with archive.extractfile(f"{BUNDLE_ROOT}/{name}") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_hash:
                    raise ValueError(f"Source changed while packaging: {name}")
        digest = sha256(temporary)
        os.link(temporary, archive_path)  # Exclusive; never overwrite another bundle.
        with checksum_path.open("x") as stream:
            stream.write(f"{digest}  {ARCHIVE_NAME}\n")
        print(f"\nBundle: {archive_path}\nSHA256: {digest}", flush=True)
        print("All six final Q4 checkpoints and original fitted preprocessing verified. No training or upload was started.")
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    package(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
