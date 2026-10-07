#!/usr/bin/env python3
"""Audit MC-only and MC+data tokenizer runs for apples-to-apples comparisons.

The input files are expected to differ. Output/logging paths are bookkeeping.
Every other configuration difference is reported, with preprocessing and
checkpoint completion called out separately because both affect physics
reconstruction comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf


DEFAULT_RUNS = OrderedDict(
    [
        (
            "jets",
            {
                "mc_only": "results/atlas_event_tokenizers_1606_jets_logstd_capacity_scan/"
                "jets_logstd_dim8_cb4096_q4",
                "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/"
                "jets_logstd_dim8_cb4096_q4",
            },
        ),
        (
            "electrons",
            {
                "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/"
                "electrons_logstd_dim8_cb4096_q4",
                "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/"
                "electrons_logstd_dim8_cb4096_q4",
            },
        ),
        (
            "muons",
            {
                "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/"
                "muons_logstd_dim8_cb4096_q4",
                "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/"
                "muons_logstd_dim8_cb4096_q4",
            },
        ),
        (
            "photons",
            {
                "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/"
                "photons_logstd_dim8_cb4096_q4",
                "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/"
                "photons_logstd_dim8_cb4096_q4",
            },
        ),
        (
            "taus",
            {
                "mc_only": "results/atlas_event_tokenizers_1806_nonjet_logstd_capacity_scan/"
                "taus_logstd_dim8_cb8192_q4",
                "mcdata": "results/atlas_event_tokenizers_0107_logstd_mc_realdata/"
                "taus_logstd_dim8_cb8192_q4",
            },
        ),
    ]
)


BOOKKEEPING_EXACT = {
    "full_path",
    "output_dir",
    "project_name",
    "network_name",
    "logger.id",
    "logger.name",
    "logger.project",
    "logger.save_dir",
}
BOOKKEEPING_SUFFIXES = (".dirpath",)
INTENDED_DATA_KEYS = {
    "datamodule.data_paths",
    "datamodule.data_domains",
}
RUNTIME_KEYS = {
    "datamodule.num_workers",
    "datamodule.pin_memory",
    "datamodule.persistent_workers",
    "datamodule.multiprocessing_context",
}
PREPROCESSOR_SUFFIX = "transforms.preprocess.cst_fn.filename"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paired MC-only and MC+data tokenizer run configurations."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing results/ (default: current directory).",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="OBJECT=MC_ONLY_RUN,MC_DATA_RUN",
        help="Override the built-in pairs; repeat once per object.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/fair_mc_vs_mcdata_eval/config_audit",
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Do not inspect epoch/global_step in checkpoint files.",
    )
    return parser.parse_args()


def parse_pairs(values: list[str]) -> OrderedDict[str, dict[str, str]]:
    if not values:
        return DEFAULT_RUNS
    pairs: OrderedDict[str, dict[str, str]] = OrderedDict()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected OBJECT=MC_ONLY_RUN,MC_DATA_RUN, got {value!r}")
        object_name, paths = value.split("=", 1)
        split_paths = paths.split(",", 1)
        if len(split_paths) != 2:
            raise ValueError(f"Expected two comma-separated run paths, got {paths!r}")
        pairs[object_name.strip()] = {
            "mc_only": split_paths[0].strip(),
            "mcdata": split_paths[1].strip(),
        }
    return pairs


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(item, path))
        return out
    # Keep lists atomic. In particular, expanding data_paths produces thousands
    # of unhelpful per-index differences.
    return {prefix: value}


def load_config(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_dir / "full_config.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected mapping in {path}")
    return cfg, flatten(cfg)


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return left == right


def classify(path: str) -> str:
    if path in BOOKKEEPING_EXACT or path.endswith(BOOKKEEPING_SUFFIXES):
        return "bookkeeping"
    if path in INTENDED_DATA_KEYS:
        return "input_data"
    if path.endswith(PREPROCESSOR_SUFFIX):
        return "preprocessing"
    if path in RUNTIME_KEYS:
        return "runtime"
    return "critical"


def summarize_value(path: str, value: Any) -> Any:
    if path in INTENDED_DATA_KEYS and isinstance(value, list):
        first = value[0] if value else None
        last = value[-1] if value else None
        return {"count": len(value), "first": first, "last": last}
    text = repr(value)
    return value if len(text) <= 300 else text[:297] + "..."


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_transformer_state(value: Any) -> dict[str, np.ndarray]:
    """Collect transformer settings and fitted attributes for comparison."""
    state: dict[str, np.ndarray] = {}
    seen: set[int] = set()

    def visit(item: Any, prefix: str) -> None:
        item_id = id(item)
        if item_id in seen:
            return
        if isinstance(item, (str, bytes, int, float, bool, type(None))):
            return
        seen.add(item_id)

        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{prefix}.{key}" if prefix else str(key))
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{prefix}[{index}]")
            return

        attrs = getattr(item, "__dict__", None)
        if not isinstance(attrs, dict):
            return
        for name, child in attrs.items():
            child_path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (np.ndarray, np.number, str, int, float, bool)):
                try:
                    state[child_path] = np.asarray(child)
                except Exception:
                    pass
            elif isinstance(child, (dict, list, tuple)) or hasattr(child, "__dict__"):
                visit(child, child_path)

    visit(value, "transformer")
    return state


def compare_preprocessors(left_path: Path, right_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mc_only_path": str(left_path),
        "mcdata_path": str(right_path),
        "mc_only_exists": left_path.exists(),
        "mcdata_exists": right_path.exists(),
        "file_sha256_equal": False,
        "learned_state_equal": None,
        "learned_state_differences": [],
    }
    left_hash = sha256(left_path)
    right_hash = sha256(right_path)
    result["mc_only_sha256"] = left_hash
    result["mcdata_sha256"] = right_hash
    result["file_sha256_equal"] = left_hash is not None and left_hash == right_hash
    if not left_path.exists() or not right_path.exists():
        return result

    try:
        left_state = collect_transformer_state(joblib.load(left_path))
        right_state = collect_transformer_state(joblib.load(right_path))
        differences = []
        for key in sorted(set(left_state) | set(right_state)):
            if key not in left_state or key not in right_state:
                differences.append({"attribute": key, "reason": "missing on one side"})
                continue
            left = left_state[key]
            right = right_state[key]
            if left.shape != right.shape:
                differences.append(
                    {"attribute": key, "left_shape": left.shape, "right_shape": right.shape}
                )
                continue
            numeric = np.issubdtype(left.dtype, np.number) and np.issubdtype(
                right.dtype, np.number
            )
            arrays_equal = (
                np.array_equal(left, right, equal_nan=True)
                if numeric
                else np.array_equal(left, right)
            )
            if not arrays_equal:
                max_abs = None
                if numeric and left.size:
                    max_abs = float(np.nanmax(np.abs(left.astype(float) - right.astype(float))))
                differences.append({"attribute": key, "max_abs_difference": max_abs})
        result["learned_state_equal"] = not differences
        result["learned_state_differences"] = differences
    except Exception as error:
        result["learned_state_error"] = f"{type(error).__name__}: {error}"
    return result


def find_checkpoint(run_dir: Path, name: str) -> Path | None:
    path = run_dir / "checkpoints" / name
    return path if path.exists() else None


def checkpoint_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return {
            "path": str(path),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
        }
    except Exception as error:
        return {"path": str(path), "error": f"{type(error).__name__}: {error}"}


def audit_pair(
    object_name: str,
    left_dir: Path,
    right_dir: Path,
    inspect_checkpoints: bool,
) -> dict[str, Any]:
    _, left = load_config(left_dir)
    _, right = load_config(right_dir)
    differences: dict[str, list[dict[str, Any]]] = {
        "critical": [],
        "preprocessing": [],
        "runtime": [],
        "input_data": [],
        "bookkeeping": [],
    }
    for path in sorted(set(left) | set(right)):
        left_value = left.get(path, "<MISSING>")
        right_value = right.get(path, "<MISSING>")
        if values_equal(left_value, right_value):
            continue
        category = classify(path)
        differences[category].append(
            {
                "key": path,
                "mc_only": summarize_value(path, left_value),
                "mcdata": summarize_value(path, right_value),
            }
        )

    preprocess = None
    preprocessor_diffs = differences["preprocessing"]
    if preprocessor_diffs:
        left_name = left.get(f"datamodule.{PREPROCESSOR_SUFFIX}")
        right_name = right.get(f"datamodule.{PREPROCESSOR_SUFFIX}")
        if isinstance(left_name, str) and isinstance(right_name, str):
            preprocess = compare_preprocessors(Path(left_name), Path(right_name))

    checkpoints = None
    if inspect_checkpoints:
        checkpoints = {
            "mc_only": {
                "best": checkpoint_metadata(find_checkpoint(left_dir, "best.ckpt")),
                "last": checkpoint_metadata(find_checkpoint(left_dir, "last.ckpt")),
            },
            "mcdata": {
                "best": checkpoint_metadata(find_checkpoint(right_dir, "best.ckpt")),
                "last": checkpoint_metadata(find_checkpoint(right_dir, "last.ckpt")),
            },
        }

    apples_to_apples = not differences["critical"] and (
        preprocess is None or preprocess.get("learned_state_equal") is True
    )
    return {
        "object": object_name,
        "mc_only_run": str(left_dir),
        "mcdata_run": str(right_dir),
        "apples_to_apples_except_input_data": apples_to_apples,
        "differences": differences,
        "preprocessor_comparison": preprocess,
        "checkpoints": checkpoints,
    }


def markdown_report(results: list[dict[str, Any]]) -> str:
    lines = ["# MC-only vs MC+data configuration audit", ""]
    for result in results:
        name = result["object"]
        passed = result["apples_to_apples_except_input_data"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"**Verdict:** {'PASS' if passed else 'REVIEW REQUIRED'}",
                "",
            ]
        )
        differences = result["differences"]
        for category, title in [
            ("critical", "Comparison-critical differences"),
            ("preprocessing", "Preprocessing paths"),
            ("runtime", "Runtime differences"),
            ("input_data", "Expected input-data differences"),
        ]:
            rows = differences[category]
            lines.append(f"### {title}")
            lines.append("")
            if not rows:
                lines.extend(["None.", ""])
                continue
            lines.extend(["| key | MC-only | MC+data |", "|---|---|---|"])
            for row in rows:
                left = str(row["mc_only"]).replace("|", "\\|")
                right = str(row["mcdata"]).replace("|", "\\|")
                lines.append(f"| `{row['key']}` | `{left}` | `{right}` |")
            lines.append("")

        preprocessor = result["preprocessor_comparison"]
        if preprocessor is not None:
            lines.extend(
                [
                    "### Preprocessor content",
                    "",
                    f"- Exact file hash equal: **{preprocessor['file_sha256_equal']}**",
                    f"- Learned fitted state equal: **{preprocessor['learned_state_equal']}**",
                    f"- Learned-state differences: "
                    f"**{len(preprocessor['learned_state_differences'])}**",
                    "",
                ]
            )

        checkpoints = result["checkpoints"]
        if checkpoints is not None:
            lines.extend(
                [
                    "### Checkpoint completion",
                    "",
                    "| run | checkpoint | epoch | global step |",
                    "|---|---|---:|---:|",
                ]
            )
            for side in ["mc_only", "mcdata"]:
                for checkpoint_name in ["best", "last"]:
                    metadata = checkpoints[side][checkpoint_name]
                    epoch = metadata.get("epoch") if metadata else "missing"
                    step = metadata.get("global_step") if metadata else "missing"
                    lines.append(f"| {side} | {checkpoint_name} | {epoch} | {step} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_pairs(args.pair)

    results = []
    for object_name, pair in pairs.items():
        left = (root / pair["mc_only"]).resolve()
        right = (root / pair["mcdata"]).resolve()
        print(f"Auditing {object_name} ...")
        result = audit_pair(object_name, left, right, not args.skip_checkpoints)
        results.append(result)
        counts = {
            category: len(rows) for category, rows in result["differences"].items()
        }
        verdict = "PASS" if result["apples_to_apples_except_input_data"] else "REVIEW"
        print(
            f"  {verdict}: critical={counts['critical']}, "
            f"preprocessing={counts['preprocessing']}, runtime={counts['runtime']}"
        )

    json_path = output_dir / "config_audit.json"
    markdown_path = output_dir / "config_audit.md"
    json_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    markdown_path.write_text(markdown_report(results))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
