#!/usr/bin/env python3
"""Smoke-test raw, hierarchical, and decoded-Q8 continuous model paths."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from heptokens.data.benchmark_sequence import (
    CONTINUOUS_FEATURE_MASK_KEY,
    CONTINUOUS_FEATURES_KEY,
    LABELS_KEY,
    MASK_KEY,
    POSITION_ROLE_IDS_KEY,
    TYPE_IDS_KEY,
)
from heptokens.models.foundation_continuous import (
    LitContinuousCLSClassifier,
    LitContinuousMaskedSequenceModel,
)
from benchmark_tokenize_objects_to_grouped_parquet import assemble_grouped_rows


def make_schema() -> dict:
    return {
        "max_feature_dim": 6,
        "role_ids": {
            "padding": 0,
            "cls": 1,
            "event": 2,
            "object": 3,
            "separator": 4,
        },
        "event_inputs": [{"index": 0, "name": "mu"}],
        "objects": {
            "jets": {
                "type_id": 6,
                "feature_count": 6,
                "feature_names": ["pt", "eta", "phi", "mass", "tag1", "tag2"],
                "groups": {
                    "kinematics": [0, 1, 2, 3],
                    "tagging": [4, 5],
                },
            },
            "tracks": {
                "type_id": 9,
                "feature_count": 4,
                "feature_names": ["pt", "eta", "d0", "z0"],
                "groups": {"kinematics": [0, 1], "impact": [2, 3]},
            },
        },
    }


def make_batch() -> dict:
    batch_size, sequence_length, feature_dim = 3, 8, 6
    features = torch.randn(batch_size, sequence_length, feature_dim)
    feature_mask = torch.zeros_like(features, dtype=torch.bool)
    role_ids = torch.tensor([[1, 2, 3, 3, 4, 0, 0, 0]] * batch_size)
    type_ids = torch.tensor([[0, 10, 6, 9, 0, 0, 0, 0]] * batch_size)
    feature_mask[:, 1, :1] = True
    feature_mask[:, 2, :6] = True
    feature_mask[:, 3, :4] = True
    return {
        CONTINUOUS_FEATURES_KEY: features,
        CONTINUOUS_FEATURE_MASK_KEY: feature_mask,
        POSITION_ROLE_IDS_KEY: role_ids,
        TYPE_IDS_KEY: type_ids,
        MASK_KEY: role_ids.ne(0),
        LABELS_KEY: torch.tensor([0, 1, 0]),
    }


def check_decoded_export_alignment() -> None:
    vocabulary = {
        "max_quantizers": 2,
        "continuous_schema": {
            "max_feature_dim": 2,
            "feature_columns": {"decoded_q8": "decoded_continuous_features"},
        },
        "objects": {
            "jets": {
                "num_quantizers": 2,
                "quantizers": [
                    {"base": 100, "size": 8},
                    {"base": 108, "size": 8},
                ],
            }
        },
    }
    args = SimpleNamespace(
        max_seq_length=6,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        no_cls=False,
        no_event_token=False,
        no_separators=False,
        object_order=["jets"],
    )
    raw = torch.tensor([[[1.0, 2.0]]])
    decoded = torch.tensor([[[0.75, 1.5]]])
    assembled = assemble_grouped_rows(
        {
            "jets": (
                torch.tensor([[[2, 3]]]),
                torch.tensor([[True]]),
                raw,
                decoded,
            )
        },
        np.asarray([[50]], dtype=np.int64),
        1,
        vocabulary,
        args,
        event_values=np.asarray([[0.25]], dtype=np.float32),
    )
    extras = assembled[3]
    if not np.array_equal(extras["continuous_features"][0, 2, :2], [1.0, 2.0]):
        raise RuntimeError("Raw object features are misaligned in grouped export")
    if not np.array_equal(
        extras["decoded_continuous_features"][0, 2, :2], [0.75, 1.5]
    ):
        raise RuntimeError("Decoded-Q8 object features are misaligned in grouped export")
    if extras["continuous_features"][0, 1, 0] != extras[
        "decoded_continuous_features"
    ][0, 1, 0]:
        raise RuntimeError("Raw and decoded-Q8 event inputs differ")


def main() -> None:
    check_decoded_export_alignment()
    schema = make_schema()
    batch = make_batch()
    common = dict(
        continuous_schema=schema,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        inner_num_heads=4,
        inner_num_layers=1,
        max_seq_length=8,
        num_type_ids=12,
    )
    for architecture in ("flat", "hierarchical"):
        pretrain = LitContinuousMaskedSequenceModel(
            architecture=architecture, **common
        )
        loss = pretrain.training_step(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{architecture} pretraining produced non-finite loss")
        loss.backward()

        classifier = LitContinuousCLSClassifier(
            architecture=architecture, **common
        )
        logits = classifier(batch)
        if logits.shape != (3, 2) or not torch.isfinite(logits).all():
            raise RuntimeError(
                f"{architecture} classifier produced invalid logits {logits.shape}"
            )

    decoded_batch = dict(batch)
    decoded_batch[CONTINUOUS_FEATURES_KEY] = (
        batch[CONTINUOUS_FEATURES_KEY] * 0.75
    ).clone()
    decoded_pretrain = LitContinuousMaskedSequenceModel(
        architecture="flat", **common
    )
    decoded_loss = decoded_pretrain.training_step(decoded_batch)
    if not torch.isfinite(decoded_loss):
        raise RuntimeError("decoded-Q8 pretraining produced non-finite loss")
    decoded_classifier = LitContinuousCLSClassifier(
        architecture="flat", **common
    )
    decoded_logits = decoded_classifier(decoded_batch)
    if decoded_logits.shape != (3, 2) or not torch.isfinite(decoded_logits).all():
        raise RuntimeError("decoded-Q8 classifier produced invalid logits")
    print(
        "PASS: raw flat, hierarchical, and decoded-Q8 continuous foundation models"
    )


if __name__ == "__main__":
    main()
