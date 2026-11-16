#!/usr/bin/env python3
"""Check the integrity and syntax of the frozen, source-only reproductions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


ROOT = Path(__file__).resolve().parent
PROFILES = ("pretraining", "downstream")
SHA256 = re.compile(r"[0-9a-f]{64}")
CREDENTIAL_MARKERS = re.compile(
    rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"
    rb"|[\"']private_key[\"']\s*:"
    rb"|gh[pousr]_[A-Za-z0-9]{30,}"
    rb"|github_pat_[A-Za-z0-9_]{30,}"
    rb"|AIza[0-9A-Za-z_-]{35}"
)
NON_SOURCE_SUFFIXES = {".ckpt", ".h5", ".hdf5", ".joblib", ".key", ".npy", ".npz",
                       ".parquet", ".pem", ".pkl", ".pt", ".pth", ".root", ".tar",
                       ".gz", ".zip", ".pyc"}


def check_profile(root: Path, *, syntax: bool) -> int:
    manifests = sorted(root.rglob("source_provenance.json"))
    if not manifests:
        raise FileNotFoundError(f"No source inventories under {root}")
    files = {}
    for path in manifests:
        if path.resolve() != path.absolute():
            raise ValueError(f"Unexpected provenance symlink: {path}")
        entries = json.loads(path.read_text(encoding="utf-8"))["files"]
        if not isinstance(entries, dict) or not entries or set(files) & set(entries):
            raise ValueError(f"{path}: empty or overlapping source inventory")
        files.update(entries)
    for name, entry in sorted(files.items()):
        rel = PurePosixPath(name)
        if (not rel.parts or rel.is_absolute() or ".." in rel.parts
                or rel.as_posix() != name or rel.suffix in NON_SOURCE_SUFFIXES
                or rel.name in {"fccstorageuser.json", ".env", "credentials.json"}):
            raise ValueError(f"Unexpected source path: {name}")
        file = root / name
        if not file.is_file() or file.resolve() != file.absolute():
            raise ValueError(f"Missing source file or symlink: {file}")
        digest = entry["sha256"]
        if not isinstance(digest, str) or not SHA256.fullmatch(digest) or not entry.get("source"):
            raise ValueError(f"Invalid source provenance: {file}")
        payload = file.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Source checksum differs: {file}")
        if CREDENTIAL_MARKERS.search(payload) or b"\0" in payload:
            raise ValueError(f"Unexpected credential marker or binary file: {file}")
        if syntax and file.suffix == ".py":
            compile(payload, str(file), "exec")
        if syntax and file.suffix == ".sh":
            subprocess.run(["bash", "-n", str(file)], check=True)

    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() or p.is_symlink()}
    metadata = {"README.md", "HISTORICAL_SOURCES.md"} | {
        p.relative_to(root).as_posix() for p in manifests
    }
    if actual - set(files) - metadata:
        raise ValueError(f"Untracked profile files: {sorted(actual - set(files) - metadata)}")
    for name in metadata:
        file = root / name
        if file.is_symlink() or (file.is_file() and CREDENTIAL_MARKERS.search(file.read_bytes())):
            raise ValueError(f"Unexpected symlink or credential marker: {file}")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syntax", action="store_true", help="Also compile Python and run bash -n")
    args = parser.parse_args()
    for profile in PROFILES:
        count = check_profile(ROOT / profile, syntax=args.syntax)
        print(f"PASS {profile}: {count} frozen source files")


if __name__ == "__main__":
    main()
