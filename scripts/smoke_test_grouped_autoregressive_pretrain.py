#!/usr/bin/env python3
"""Exercise grouped teacher forcing, legal heads, and sequential generation."""

from __future__ import annotations

import torch

from heptokens.data.sequence import MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.models.foundation_grouped import LitGroupedMaskedSequenceModel


def vocabulary() -> dict:
    return {
        "vocab_size": 38,
        "includes_cls": True,
        "event_tokens": [{"base": 4, "size": 4, "type_id": 1}],
        "objects": {
            "electrons": {
                "type_id": 4,
                "quantizers": [
                    {"index": 0, "base": 8, "size": 5},
                    {"index": 1, "base": 13, "size": 5},
                    {"index": 2, "base": 18, "size": 5},
                ],
            },
            "muons": {
                "type_id": 5,
                "quantizers": [
                    {"index": 0, "base": 23, "size": 5},
                    {"index": 1, "base": 28, "size": 5},
                    {"index": 2, "base": 33, "size": 5},
                ],
            },
        },
    }


def make_model() -> LitGroupedMaskedSequenceModel:
    return LitGroupedMaskedSequenceModel(
        hidden_dim=24,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        max_seq_length=5,
        vocab_size=38,
        max_quantizers=3,
        quantizer_embedding_dim=8,
        num_type_ids=6,
        mask_token_id=3,
        pad_token_id=0,
        mask_prob=1.0,
        quantizer_decoder="autoregressive",
        token_vocabulary=vocabulary(),
    )


def make_batch() -> dict:
    return {
        TOKENS_KEY: torch.tensor(
            [
                [[1, 0, 0], [4, 0, 0], [8, 13, 18], [23, 28, 33], [2, 0, 0]],
                [[1, 0, 0], [7, 0, 0], [12, 17, 22], [27, 32, 37], [2, 0, 0]],
            ],
            dtype=torch.long,
        ),
        MASK_KEY: torch.ones(2, 5, dtype=torch.bool),
        TYPE_IDS_KEY: torch.tensor(
            [[0, 1, 4, 5, 0], [0, 1, 4, 5, 0]], dtype=torch.long
        ),
    }


def check_teacher_forcing(model: LitGroupedMaskedSequenceModel) -> None:
    decoder = model.model.autoregressive_decoder
    assert decoder is not None
    decoder.eval()
    context = torch.randn(2, 24)
    first = torch.tensor([[8, 13, 18], [9, 14, 19]])
    changed_q0 = first.clone()
    changed_q0[:, 0] += 1
    changed_q1 = first.clone()
    changed_q1[:, 1] += 1

    states = decoder._teacher_forced_states(
        context, first, model.backbone.token_embedding
    )
    q0_states = decoder._teacher_forced_states(
        context, changed_q0, model.backbone.token_embedding
    )
    q1_states = decoder._teacher_forced_states(
        context, changed_q1, model.backbone.token_embedding
    )

    assert torch.allclose(states[:, 0], q0_states[:, 0])
    assert not torch.allclose(states[:, 1], q0_states[:, 1])
    assert torch.allclose(states[:, 1], q1_states[:, 1])
    assert not torch.allclose(states[:, 2], q1_states[:, 2])


def check_generation(model: LitGroupedMaskedSequenceModel, batch: dict) -> None:
    tokens = batch[TOKENS_KEY].clone()
    mask = batch[MASK_KEY]
    type_ids = batch[TYPE_IDS_KEY]
    selected = torch.zeros_like(mask)
    selected[:, 2:4] = True
    valid = tokens.ne(0)
    tokens[selected.unsqueeze(-1) & valid] = 3
    generated = model.model.generate_masked_object_codes(
        tokens=tokens,
        mask=mask,
        type_ids=type_ids,
        masked_positions=selected,
    )
    selected_types = type_ids[selected]
    decoder = model.model.autoregressive_decoder
    assert decoder is not None
    for row, type_id in zip(generated, selected_types, strict=True):
        for quantizer, token in enumerate(row.tolist()):
            base, size = decoder.object_specs[(int(type_id), quantizer)]
            assert base <= token < base + size


def main() -> None:
    torch.manual_seed(42)
    model = make_model()
    batch = make_batch()
    loss = model._shared_step(batch, "train")
    assert loss.isfinite()
    loss.backward()
    decoder = model.model.autoregressive_decoder
    assert decoder is not None
    assert decoder.autoregressive_gru.weight_ih_l0.grad is not None
    assert model.model.mlm_head is None
    check_teacher_forcing(model)
    check_generation(model, batch)
    print("PASS: grouped autoregressive teacher forcing and legal sequential decoding")


if __name__ == "__main__":
    main()
