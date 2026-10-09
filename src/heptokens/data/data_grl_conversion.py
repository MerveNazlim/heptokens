"""Reusable policy and validation for the data-GRL representation conversion.

This is an offline conversion module, not a Lightning DataModule.  It owns the
scientific invariants shared by local, Condor, and future workflow runners:
the tokenizer specifications, data-only input policy, and deterministic event
split.  Runners remain responsible for file staging and process lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Mapping

import numpy as np


DEFAULT_OBJECT_ORDER = (
    "electrons",
    "muons",
    "taus",
    "photons",
    "jets",
    "tracks",
)

DEFAULT_EVENT_TOKEN_INPUTS = (
    "common/event/mu",
    "common/met/pt",
    "common/met/phi",
    "common/met/sumet",
)

DEFAULT_Q1_TOKENIZER_SPEC = {
    object_name: (16_384, 1) for object_name in DEFAULT_OBJECT_ORDER
}

DEFAULT_Q8_TOKENIZER_SPEC = {
    "electrons": (2_048, 8),
    "muons": (2_048, 8),
    "taus": (4_096, 8),
    "photons": (2_048, 8),
    "jets": (2_048, 8),
    "tracks": (4_096, 8),
}


def stable_source_hash(value: str) -> np.uint64:
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, person=b"heptokens"
    ).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    mixed = values.astype(np.uint64, copy=True) + np.uint64(seed)
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return mixed


@dataclass(frozen=True)
class DataGrlConversionModule:
    """Scientific configuration shared by every H5→Parquet conversion runner."""

    datamodule_config: str = "configs/datamodule/atlas_event_object.yaml"
    seed: int = 42
    train_frac: float = 0.9
    batch_size: int = 1024
    max_seq_length: int = 256
    shard_rows: int = 50_000
    row_group_rows: int = 4096
    compression: str = "snappy"
    event_bins: int = 128
    object_order: tuple[str, ...] = DEFAULT_OBJECT_ORDER
    event_token_inputs: tuple[str, ...] = DEFAULT_EVENT_TOKEN_INPUTS
    event_token_ranges: tuple[str, ...] = ()
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    mask_token_id: int = 3
    q1_vocab_size: int = 98_820
    q8_vocab_size: int = 131_588
    require_real_data: bool = True
    q1_tokenizer_spec: Mapping[str, tuple[int, int]] = field(
        default_factory=lambda: dict(DEFAULT_Q1_TOKENIZER_SPEC)
    )
    q8_tokenizer_spec: Mapping[str, tuple[int, int]] = field(
        default_factory=lambda: dict(DEFAULT_Q8_TOKENIZER_SPEC)
    )

    def __post_init__(self) -> None:
        # Hydra/OmegaConf materializes YAML sequences and mappings as its own
        # container types.  Normalize them immediately so the rest of the
        # converter sees ordinary immutable tuples and dictionaries regardless
        # of whether this object came from YAML or direct Python construction.
        object.__setattr__(self, "object_order", tuple(self.object_order))
        object.__setattr__(
            self, "event_token_inputs", tuple(self.event_token_inputs)
        )
        object.__setattr__(
            self, "event_token_ranges", tuple(self.event_token_ranges)
        )
        object.__setattr__(
            self,
            "q1_tokenizer_spec",
            {
                str(name): tuple(int(item) for item in specification)
                for name, specification in self.q1_tokenizer_spec.items()
            },
        )
        object.__setattr__(
            self,
            "q8_tokenizer_spec",
            {
                str(name): tuple(int(item) for item in specification)
                for name, specification in self.q8_tokenizer_spec.items()
            },
        )
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError("train_frac must be between zero and one")
        for name in (
            "batch_size",
            "max_seq_length",
            "shard_rows",
            "row_group_rows",
            "event_bins",
            "q1_vocab_size",
            "q8_vocab_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if len(set(self.object_order)) != len(self.object_order):
            raise ValueError("object_order contains duplicates")
        expected_objects = set(DEFAULT_OBJECT_ORDER)
        if set(self.object_order) != expected_objects:
            raise ValueError(
                f"object_order={self.object_order!r} does not contain exactly "
                f"{sorted(expected_objects)!r}"
            )
        for label, specification in (
            ("q1", self.q1_tokenizer_spec),
            ("q8", self.q8_tokenizer_spec),
        ):
            if set(specification) != expected_objects:
                raise ValueError(
                    f"{label}_tokenizer_spec does not contain exactly "
                    f"{sorted(expected_objects)!r}"
                )
            for object_name, (codebook_size, num_quantizers) in specification.items():
                if codebook_size <= 0 or num_quantizers <= 0:
                    raise ValueError(
                        f"Invalid {label} tokenizer specification for "
                        f"{object_name}: {(codebook_size, num_quantizers)!r}"
                    )

    @property
    def split_method(self) -> str:
        return "blake2b(source_uri) xor event_index, then splitmix64"

    def train_mask(self, source_uri: str, event_indices: np.ndarray) -> np.ndarray:
        values = np.asarray(event_indices, dtype=np.uint64)
        source_hash = stable_source_hash(source_uri)
        threshold = np.uint64(int(self.train_frac * 2**64))
        return splitmix64(values ^ source_hash, self.seed) < threshold

    def validate_source_metadata(self, source_uri: str, metadata: Mapping) -> None:
        if self.require_real_data and bool(metadata.get("is_mc", False)):
            raise ValueError(f"Data-only campaign received MC input: {source_uri}")

    @staticmethod
    def _model_spec(models: Mapping) -> dict[str, tuple[int, int]]:
        return {
            object_name: (
                int(model.hparams.codebook_size),
                int(model.hparams.num_quantizers),
            )
            for object_name, model in models.items()
        }

    def validate_tokenizers(self, models: Mapping, representation: str) -> None:
        if representation == "q1":
            expected = dict(self.q1_tokenizer_spec)
        elif representation == "q8":
            expected = dict(self.q8_tokenizer_spec)
        else:
            raise ValueError(f"Unknown tokenizer representation: {representation}")
        observed = self._model_spec(models)
        if observed != expected:
            raise ValueError(
                f"{representation.upper()} tokenizer specification mismatch: "
                f"observed={observed!r} expected={expected!r}"
            )

    def manifest_settings(self) -> dict:
        return {
            "datamodule_config": self.datamodule_config,
            "seed": self.seed,
            "train_fraction_requested": self.train_frac,
            "split_method": self.split_method,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "shard_rows": self.shard_rows,
            "row_group_rows": self.row_group_rows,
            "compression": self.compression,
            "event_bins": self.event_bins,
            "object_order": list(self.object_order),
            "event_token_inputs": list(self.event_token_inputs),
            "event_token_ranges": list(self.event_token_ranges),
            "special_token_ids": {
                "pad": self.pad_token_id,
                "cls": self.cls_token_id,
                "sep": self.sep_token_id,
                "mask": self.mask_token_id,
            },
            "q1_vocab_size": self.q1_vocab_size,
            "q8_vocab_size": self.q8_vocab_size,
            "require_real_data": self.require_real_data,
            "q1_tokenizer_spec": {
                key: list(value) for key, value in self.q1_tokenizer_spec.items()
            },
            "q8_tokenizer_spec": {
                key: list(value) for key, value in self.q8_tokenizer_spec.items()
            },
        }
