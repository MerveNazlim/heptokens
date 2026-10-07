#!/usr/bin/env python3
"""Explain zero and oversized bars in decoded-feature resolution output."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = {
    "parallel_greedy": 0,
    "autoregressive_greedy": 1,
    "parallel_sample": 2,
    "autoregressive_sample": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["electrons", "muons"],
        help="Object names to inspect (default: electrons muons).",
    )
    return parser.parse_args()


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def diagnosis(row: dict[str, str], panel_max: float) -> str:
    reference_iqr = abs(number(row, "reference_iqr"))
    residual_iqr = abs(number(row, "residual_iqr"))
    ratio = abs(number(row, "resolution_over_reference_iqr"))
    labels = []
    if reference_iqr <= 1e-12:
        labels.append("zero reference IQR; ratio unstable")
    if residual_iqr <= 1e-12:
        labels.append("zero residual IQR")
    elif panel_max > 0 and ratio / panel_max < 1e-3:
        labels.append("nonzero but hidden by panel scale")
    return "; ".join(labels) or "visible finite value"


def main() -> None:
    args = parse_args()
    with args.csv_path.expanduser().open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    requested = {name.lower() for name in args.objects}
    selected = [row for row in rows if row["object"].lower() in requested]
    if not selected:
        available = sorted({row["object"] for row in rows})
        raise SystemExit(f"No requested objects found. Available: {available}")

    by_object: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_object[row["object"]].append(row)

    for object_name, object_rows in by_object.items():
        panel_max = max(
            abs(number(row, "resolution_over_reference_iqr"))
            for row in object_rows
        )
        features = {name: index for index, name in enumerate(dict.fromkeys(
            row["feature"] for row in object_rows
        ))}
        object_rows.sort(
            key=lambda row: (
                features[row["feature"]],
                METHOD_ORDER.get(row["method"], 99),
            )
        )

        print(f"\n=== {object_name} (largest plotted ratio: {panel_max:.6g}) ===")
        print(
            f"{'feature':<20} {'method':<27} {'ref_IQR':>11} "
            f"{'resid_IQR':>11} {'median_AE':>11} {'ratio':>11}  diagnosis"
        )
        for row in object_rows:
            print(
                f"{row['feature']:<20} {row['method']:<27} "
                f"{number(row, 'reference_iqr'):>11.4e} "
                f"{number(row, 'residual_iqr'):>11.4e} "
                f"{number(row, 'median_absolute_error'):>11.4e} "
                f"{number(row, 'resolution_over_reference_iqr'):>11.4e}  "
                f"{diagnosis(row, panel_max)}"
            )


if __name__ == "__main__":
    main()
