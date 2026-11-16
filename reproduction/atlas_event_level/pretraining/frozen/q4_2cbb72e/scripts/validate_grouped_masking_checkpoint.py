#!/usr/bin/env python3
"""Reject incomplete or misconfigured masking-study runs before DAG successors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def validate(results: Path, probability: float, epochs: int, steps: int,
             global_batch: int, *, max_quantizers: int = 8,
             vocab_size: int = 131588, embedding_dim: int = 32) -> None:
    checkpoints = list(results.rglob("checkpoints/last.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected exactly one last.ckpt, found {len(checkpoints)}")
    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != epochs - 1 or checkpoint["global_step"] != steps:
        raise ValueError(f"Incomplete training: epoch={checkpoint['epoch']}, "
                         f"global_step={checkpoint['global_step']}; expected {epochs-1}, {steps}")
    parameters = checkpoint["hyper_parameters"]
    expected = dict(mask_prob=probability, ensure_masked_physical_object_per_event=True,
                    object_projection_mode="identity", max_quantizers=max_quantizers,
                    quantizer_embedding_dim=embedding_dim, hidden_dim=256, num_heads=8,
                    num_layers=4, max_seq_length=256, vocab_size=vocab_size,
                    quantizer_decoder="parallel", learning_rate=1e-4)
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise ValueError(f"Checkpoint {key}={parameters.get(key)!r}, expected {value!r}")
    vocabulary = parameters.get("token_vocabulary") or {}
    if (vocabulary.get("vocab_size") != vocab_size or
            vocabulary.get("max_quantizers") != max_quantizers or not vocabulary.get("objects")):
        raise ValueError("Checkpoint does not retain the resolved object vocabulary")
    if "token_vocabulary_path" in parameters:
        raise ValueError("Checkpoint depends on a temporary vocabulary path")
    if any("object_projection.weight" in name for name in checkpoint["state_dict"]):
        raise ValueError("Checkpoint unexpectedly has a learned object projection")
    del checkpoint
    audits = list(results.rglob("masking_audit.csv"))
    if len(audits) != 1:
        raise ValueError(f"Expected exactly one masking audit, found {len(audits)}")
    with audits[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    final_train = {}
    for row in rows:
        if float(row["mask_prob"]) != probability:
            raise ValueError("Audit masking probability disagrees with checkpoint")
        if int(row["physical_maskless_after"]) != 0:
            raise ValueError("An eligible event lacked a masked physical object")
        if row["split"] == "train":
            final_train[int(row["epoch"])] = row
    if set(final_train) != set(range(epochs)):
        raise ValueError("Masking audit does not cover every completed epoch")
    seen = sum(int(row["events"]) for row in final_train.values())
    if seen != steps * global_batch:
        raise ValueError(f"Training event count {seen} differs from {steps*global_batch}")
    zero_objects = sum(int(row["zero_physical_events"]) for row in final_train.values())
    forced = sum(int(row["forced_events"]) for row in final_train.values())
    print(f"PASS checkpoint epoch={epochs-1} global_step={steps}")
    print(f"PASS Q{max_quantizers} identity; mask_prob={probability}; physical-object guarantee enabled")
    print(f"PASS globally audited training examples={seen}; zero-object examples retained={zero_objects}")
    print(f"PASS uniformly forced events={forced}; eligible events without masked objects=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--mask-prob", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--expected-global-steps", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--max-quantizers", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=131588)
    parser.add_argument("--embedding-dim", type=int, default=32)
    args = parser.parse_args()
    validate(args.results_dir, args.mask_prob, args.epochs,
             args.expected_global_steps, args.global_batch_size,
             max_quantizers=args.max_quantizers, vocab_size=args.vocab_size,
             embedding_dim=args.embedding_dim)


if __name__ == "__main__":
    main()
