#!/usr/bin/env python3
"""Smoke-test grouped classification streaming and masked-mean checkpoint loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from heptokens.data.sequence import (
    LABELS_KEY,
    MASK_KEY,
    TOKENS_KEY,
    TYPE_IDS_KEY,
)
from heptokens.data.token_parquet import (
    GroupedTokenParquetClassificationModule,
    StreamingTokenParquetDataset,
)
from heptokens.models.foundation_grouped import LitGroupedMaskedSequenceModel
from heptokens.models.foundation_grouped_mean_classifier import (
    LitGroupedMaskedMeanClassifier,
)


MODEL_KWARGS = {
    "hidden_dim": 8,
    "num_heads": 2,
    "num_layers": 1,
    "dropout": 0.0,
    "max_seq_length": 4,
    "vocab_size": 512,
    "max_quantizers": 2,
    "quantizer_embedding_dim": 4,
    "num_type_ids": 4,
    "pad_token_id": 0,
    "use_type_embedding": True,
    "use_position_embedding": True,
}


def write_grouped_parquet(path: Path, values: list[int], label: int) -> None:
    tokens = pa.array(
        [[[value, value + 100], [value + 200, 0]] for value in values]
    )
    table = pa.table(
        {
            "tokens": tokens,
            "mask": pa.array([[True, True] for _ in values]),
            "type_ids": pa.array([[1, 2] for _ in values]),
            "label": pa.array([label] * len(values), type=pa.int64()),
        }
    )
    pq.write_table(table, path, row_group_size=3)


def check_streaming(directory: Path) -> None:
    signal = directory / "signal.parquet"
    background = directory / "background.parquet"
    write_grouped_parquet(signal, list(range(4, 10)), 1)
    write_grouped_parquet(background, list(range(10, 16)), 0)
    dataset = StreamingTokenParquetDataset(
        parquet_files=[str(signal), str(background)],
        split_start=0.0,
        split_end=1.0,
        seed=42,
        stream_batch_size=2,
        shuffle=True,
        shuffle_buffer_size=4,
        label_column="label",
        require_labels=True,
    )
    samples = list(dataset)
    labels = sorted(int(sample[LABELS_KEY]) for sample in samples)
    assert len(samples) == 12
    assert labels == [0] * 6 + [1] * 6
    assert all(sample[TOKENS_KEY].shape == (2, 2) for sample in samples)


def check_datamodule(directory: Path) -> None:
    prepared = directory / "prepared"
    split_counts = {}
    for split_name in ("train", "val", "test"):
        for class_name, label, offset in (
            ("signal", 1, 4),
            ("background", 0, 20),
        ):
            class_dir = prepared / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            write_grouped_parquet(
                class_dir / "part-00000.parquet",
                list(range(offset, offset + 6)),
                label,
            )
        split_counts[split_name] = {"signal": 6, "background": 6, "total": 12}
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "identity_overlap_detected": False,
                "split_counts": split_counts,
            }
        )
    )
    datamodule = GroupedTokenParquetClassificationModule(
        prepared_dir=str(prepared),
        n_classes=2,
        num_workers=0,
        batch_size=4,
        pin_memory=False,
        stream_batch_size=2,
        shuffle_buffer_size=4,
        label_column="label",
    )
    batch = next(iter(datamodule.train_dataloader()))
    assert batch[TOKENS_KEY].shape == (4, 2, 2)
    assert batch[LABELS_KEY].shape == (4,)


def check_model(directory: Path) -> None:
    pretrained = LitGroupedMaskedSequenceModel(**MODEL_KWARGS)
    checkpoint = directory / "last.ckpt"
    torch.save({"state_dict": pretrained.state_dict()}, checkpoint)
    classifier = LitGroupedMaskedMeanClassifier(
        **MODEL_KWARGS,
        backbone_ckpt_path=str(checkpoint),
        freeze_backbone=True,
    )
    expected = pretrained.backbone.state_dict()
    observed = classifier.backbone.state_dict()
    assert expected.keys() == observed.keys()
    assert all(torch.equal(expected[key], observed[key]) for key in expected)
    assert all(not parameter.requires_grad for parameter in classifier.backbone.parameters())

    batch = {
        TOKENS_KEY: torch.tensor([[[4, 5], [6, 0], [0, 0], [0, 0]]]),
        MASK_KEY: torch.tensor([[True, True, False, False]]),
        TYPE_IDS_KEY: torch.tensor([[1, 2, 0, 0]]),
    }
    classifier.train()
    logits = classifier(batch)
    assert tuple(logits.shape) == (1, 2)
    assert classifier.n_classes == 2
    assert not classifier.backbone.training


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        check_streaming(root)
        check_datamodule(root)
        check_model(root)
    print("PASS: grouped datamodule, streaming labels, and masked-mean classifier")


if __name__ == "__main__":
    main()
