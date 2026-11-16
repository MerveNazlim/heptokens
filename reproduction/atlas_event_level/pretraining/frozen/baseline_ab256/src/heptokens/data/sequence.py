"""Canonical helpers for sequence-token batches.

Sequence batches use the same naming style as the rest of the repo:
``tokens``, ``mask``, optional ``type_ids``, and optional ``labels``.
"""
# TODO: Add validation for sequence batches, including shape checks,
# token range checks, and type-id consistency checks.

from __future__ import annotations

from collections.abc import Mapping

import torch

TOKENS_KEY = "tokens"
MASK_KEY = "mask"
TYPE_IDS_KEY = "type_ids"
LABELS_KEY = "labels"


def first_existing_column(
    columns: list[str],
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    """Return the first candidate present in a parquet/table schema."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise KeyError(f"Expected one of columns {candidates}, found {columns}")
    return None


def sequence_inputs(
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return ``tokens``, ``mask``, and optional ``type_ids`` from a sequence batch."""
    return batch[TOKENS_KEY], batch[MASK_KEY], batch.get(TYPE_IDS_KEY)


def sequence_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    """Move tensor values from a sequence batch to a device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
