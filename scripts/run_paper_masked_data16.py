#!/usr/bin/env python3
"""Run Figure 6 locally on explicitly selected, downloaded data16 validation shards."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

OBJECTS = ("jets", "electrons", "muons", "photons", "taus", "tracks")
PART = "part-group-00000-00000.parquet"
CACHE = Path("/home/magaras/.cache/rattler/cache/pkgs")
PYARROW_PACKAGE = "pyarrow-core-16.1.0-py311hf1d6e26_2_cpu"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_hashes(bundle):
    if not bundle.is_file():
        raise FileNotFoundError(
            f"Original local tokenizer export bundle missing: {bundle}. "
            "Use --tokenizer-bundle for its actual location; do not rebuild it."
        )
    with tarfile.open(bundle, "r:gz") as archive:
        stream = archive.extractfile("data_grl_tokenizer_artifacts/SHA256SUMS")
        if stream is None:
            raise ValueError("Tokenizer bundle has no SHA256SUMS file")
        records = {}
        for line in stream.read().decode().splitlines():
            digest, name = line.split(maxsplit=1)
            name = name.removeprefix("*").removeprefix("./")
            if (
                name in records
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"Invalid export checksum record: {line}")
            records[name] = digest
    return records


def tokenizer_paths(root, q, hashes):
    base = root / "results/atlas_object_final_tokenizers_new_mcdata"
    prep = root / "results/preprocessing/atlas_object_final_tokenizers_new_mcdata"
    result = {}
    for obj in OBJECTS:
        k = 16384 if q == 1 else (4096 if obj in ("taus", "tracks") else 2048)
        run = base / f"{obj}_full_dim8_cb{k}_q{q}_e20_new_mcdata"
        expected = hashes[f"q{q}/{obj}.ckpt"]
        # Match the export's content, not a best/last or latest-file heuristic.
        matches = [
            path
            for name in ("best.ckpt", "last.ckpt")
            if (path := run / "checkpoints" / name).is_file() and sha256(path) == expected
        ]
        if not matches:
            raise ValueError(f"No checkpoint in {run} matches the original Q{q} export")
        if not (run / "full_config.yaml").is_file():
            raise FileNotFoundError(run / "full_config.yaml")
        name = "tracks_log_standard_no_ndoflog" if obj == "tracks" else f"{obj}_log_standard"
        joblib = prep / f"{name}.joblib"
        if sha256(joblib) != hashes[f"preprocessing/{obj}.joblib"]:
            raise ValueError(f"Preprocessor differs from tokenizer export: {joblib}")
        if not joblib.with_suffix(".json").is_file():
            raise FileNotFoundError(joblib.with_suffix(".json"))
        result[obj] = matches[0]
    return result


def environment(root, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    paths = [str(root / "src")]
    site = CACHE / PYARROW_PACKAGE / "lib/python3.11/site-packages"
    if (site / "pyarrow").is_dir():
        paths.append(str(site))
        libraries = sorted(str(p) for p in CACHE.glob("*/lib") if p.is_dir())
        env["LD_LIBRARY_PATH"] = ":".join(
            libraries + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
        )
    env["PYTHONPATH"] = ":".join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def commands(
    root,
    figure,
    qs,
    mappings,
    unit,
    *,
    prepared=None,
    n_events=50_000,
    max_objects=10_000,
    with_original=False,
    postprocess_only=False,
):
    python = str(root / ".pixi/envs/default/bin/python")
    script = str(root / "scripts/paper_masked_object_reconstruction.py")
    evaluations, plots = [], []
    prepared = prepared or figure / "prepared"
    for q in qs:
        output = str(figure / f"q{q}")
        checkpoint = (
            root
            / "results/atlas_event_foundation_pretrain_data16_200m"
            / f"q{q}_b448_8gpu_e10_seed42/checkpoints/last.ckpt"
        )
        evaluate = [
            python,
            script,
            "evaluate",
            "--evaluation-dir",
            output,
            "--checkpoint",
            str(checkpoint),
            "--prepared-dir",
            str(prepared / f"q{q}"),
            "--split",
            "val",
            "--quantizers",
            str(q),
            "--device",
            "cuda",
            "--input-momentum-unit",
            unit,
            "--sample-label",
            "Collision data | data16 GRL validation subset",
            "--n-events",
            str(n_events),
            "--max-objects-per-type",
            str(max_objects),
            "--mask-prob",
            "0.15",
            "--seed",
            "42",
            "--batch-size",
            "8",
            "--confirm-tokenizer-mapping",
        ]
        if not postprocess_only:
            for obj in OBJECTS:
                evaluate += ["--tokenizer-checkpoint", f"{obj}={mappings[q][obj]}"]
            evaluations.append(evaluate)
        summary = [python, script, "summarize", "--evaluation-dir", output]
        if q == 1:
            summary += ["--binning-from", str(figure / "q8")]
        flags = ["--with-original"] if with_original else []
        summary += flags
        plots.extend(
            [
                summary,
                [python, script, "plot", "--evaluation-dir", output, "--y-scale", "log", *flags],
            ]
        )
    audit = [paired_command(root, figure)] if 1 in qs else []
    if with_original:
        # Attachment includes the paired-cache audit before reading original inputs.
        audit = [
            [
                python,
                script,
                "attach-originals",
                "--q8-dir",
                str(figure / "q8"),
                "--q1-dir",
                str(figure / "q1"),
                "--continuous-dir",
                str(prepared / "continuous/val"),
            ]
        ]
    return evaluations + audit + plots


def paired_command(root, figure):
    return [
        str(root / ".pixi/envs/default/bin/python"),
        str(root / "scripts/paper_masked_object_reconstruction.py"),
        "audit-pair",
        "--q8-dir",
        str(figure / "q8"),
        "--q1-dir",
        str(figure / "q1"),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/magaras/heptok_fork/heptokens"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--runs", choices=("both", "q8", "q1"), default="both")
    parser.add_argument("--input-momentum-unit", choices=("GeV", "MeV"), default="GeV")
    parser.add_argument("--tokenizer-bundle", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        help="Explicit parent of q8/q1 outputs; defaults to the existing Figure 6 directory",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help="Parent of paired q8/val, q1/val, continuous/val; defaults to the existing Figure 6 prepared directory",
    )
    parser.add_argument("--n-events", type=int, default=50_000)
    parser.add_argument("--max-objects-per-type", type=int, default=10_000)
    parser.add_argument(
        "--all-prepared-shards",
        action="store_true",
        help="Explicitly allow multiple downloaded, paired shards (no downloads)",
    )
    parser.add_argument(
        "--with-original", action="store_true", help="Add aligned original inputs to both Q plots"
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Use saved predictions only; no model loading or inference",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Compare existing Q8/Q1 caches on CPU; no inference, downloads or writes",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Plot paired Q1/Q8 binned errors and residuals against cached originals; no inference",
    )
    parser.add_argument("--comparison-bins", type=int, default=8)
    parser.add_argument("--comparison-min-count", type=int, default=50)
    parser.add_argument("--comparison-binning", choices=("quantile", "linear"), default="quantile")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        help="Defaults to paired_comparison under the existing Figure 6 root",
    )
    parser.add_argument("--show-tokenizer-reference", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    default_figure = root / "results/paper/figure6_data16_200m_validation_v1"
    figure = (args.evaluation_root or default_figure).expanduser().resolve()
    prepared = (args.prepared_root or default_figure / "prepared").expanduser().resolve()
    if args.n_events <= 0 or args.max_objects_per_type <= 0:
        parser.error("Event and per-object limits must be positive")
    if args.with_original and args.runs != "both":
        parser.error("--with-original requires --runs both for the paired input audit")
    if args.postprocess_only and args.runs != "both":
        parser.error("--postprocess-only requires --runs both")
    if args.compare_only:
        if args.runs != "both" or args.audit_only or args.postprocess_only or args.with_original:
            parser.error("Use --compare-only separately, after --postprocess-only --with-original")
        if args.comparison_bins <= 0 or args.comparison_min_count <= 0:
            parser.error("Comparison bins and minimum counts must be positive")
        command = [
            str(root / ".pixi/envs/default/bin/python"),
            str(root / "scripts/paper_masked_object_reconstruction.py"),
            "compare",
            "--q8-dir",
            str(figure / "q8"),
            "--q1-dir",
            str(figure / "q1"),
            "--output-dir",
            str((args.comparison_dir or figure / "paired_comparison").resolve()),
            "--n-bins",
            str(args.comparison_bins),
            "--min-bin-count",
            str(args.comparison_min_count),
            "--binning",
            args.comparison_binning,
        ]
        if args.show_tokenizer_reference:
            command.append("--show-tokenizer-reference")
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, env=environment(root, args.gpu), check=True)
        return
    if args.audit_only:
        subprocess.run(
            paired_command(root, figure), cwd=root, env=environment(root, args.gpu), check=True
        )
        return
    if args.postprocess_only:
        jobs = commands(
            root,
            figure,
            (8, 1),
            {},
            args.input_momentum_unit,
            prepared=prepared,
            with_original=args.with_original,
            postprocess_only=True,
        )
        for command in jobs:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=root, env=environment(root, args.gpu), check=True)
        return
    bundle = (
        args.tokenizer_bundle
        or root / "results/data_grl_tokenizer_artifacts/data-grl-q1-q8-tokenizer-artifacts.tar.gz"
    )
    qs = (8, 1) if args.runs == "both" else (int(args.runs[1:]),)
    required = [
        root / ".pixi/envs/default/bin/python",
        root / "scripts/paper_masked_object_reconstruction.py",
    ]
    if qs == (1,):
        required.append(figure / "q8/masked_prediction_summary.json")
    for q in qs:
        required.extend(
            [
                root
                / "results/atlas_event_foundation_pretrain_data16_200m"
                / f"q{q}_b448_8gpu_e10_seed42/checkpoints/last.ckpt",
            ]
        )
        shards = sorted((prepared / f"q{q}" / "val").glob("*.parquet"))
        if not args.all_prepared_shards and shards != [prepared / f"q{q}" / "val" / PART]:
            raise ValueError(f"Q{q}: expected only the verified 50,000-row validation shard")
        if not shards:
            raise FileNotFoundError(f"No Q{q} validation shards in {prepared}")
    if not args.all_prepared_shards and args.n_events > 50_000:
        parser.error(
            "More than 50,000 events requires --all-prepared-shards and additional paired inputs"
        )
    if args.with_original:
        groups = [
            sorted(path.name for path in (prepared / rep / "val").glob("*.parquet"))
            for rep in ("q8", "q1", "continuous")
        ]
        if not groups[0] or groups[0] != groups[1] or groups[0] != groups[2]:
            raise ValueError(
                "Download the same validation shards for q8, q1 and continuous before inference"
            )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = export_hashes(bundle.expanduser().resolve())
    mappings = {q: tokenizer_paths(root, q, hashes) for q in qs}
    env = environment(root, args.gpu)
    print(
        f"Figure 6 | GPU {args.gpu} | Q sequence: {qs} | physical units: {args.input_momentum_unit}",
        flush=True,
    )
    print(f"Prepared files: {prepared}; outputs: {figure}", flush=True)
    print(
        f"Requested events: {args.n_events:,}; masked-object cap/type: {args.max_objects_per_type:,} (not guaranteed counts)",
        flush=True,
    )
    print(f"Tokenizer identities checked against local export bundle: {bundle}", flush=True)
    preflight = [
        str(root / ".pixi/envs/default/bin/python"),
        str(root / "scripts/paper_masked_object_reconstruction.py"),
        "check-inputs",
        "--prepared-root",
        str(prepared),
        "--n-events",
        str(args.n_events),
    ]
    jobs = ([preflight] if args.all_prepared_shards else []) + commands(
        root,
        figure,
        qs,
        mappings,
        args.input_momentum_unit,
        prepared=prepared,
        n_events=args.n_events,
        max_objects=args.max_objects_per_type,
        with_original=args.with_original,
    )
    for command in jobs:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, env=env, check=True)
    print(
        "Dry run complete; no inference performed." if args.dry_run else "Figure 6 plots complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
