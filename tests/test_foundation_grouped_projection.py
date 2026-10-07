from __future__ import annotations

import unittest

import torch
from torch import nn

from heptokens.models.foundation_grouped import GroupedSequenceBackbone


def make_backbone(**overrides) -> GroupedSequenceBackbone:
    arguments = {
        "vocab_size": 32,
        "hidden_dim": 16,
        "max_quantizers": 1,
        "quantizer_embedding_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "max_seq_length": 8,
        "num_type_ids": 4,
        "pad_token_id": 0,
        "use_type_embedding": True,
        "use_position_embedding": True,
    }
    arguments.update(overrides)
    return GroupedSequenceBackbone(**arguments)


class GroupedProjectionTest(unittest.TestCase):
    def test_linear_projection_remains_default(self) -> None:
        backbone = make_backbone()
        self.assertIsInstance(backbone.object_projection, nn.Linear)

    def test_q1_direct_uses_identity_projection(self) -> None:
        backbone = make_backbone(object_projection_mode="identity")
        self.assertIsInstance(backbone.object_projection, nn.Identity)
        tokens = torch.tensor([[[1], [4], [5], [0]]])
        mask = torch.tensor([[True, True, True, False]])
        type_ids = torch.tensor([[0, 1, 1, 0]])
        output = backbone(tokens, mask, type_ids)
        self.assertEqual(tuple(output.shape), (1, 4, 16))

    def test_q8_direct_uses_identity_projection(self) -> None:
        backbone = make_backbone(
            max_quantizers=8,
            quantizer_embedding_dim=2,
            object_projection_mode="identity",
        )
        self.assertIsInstance(backbone.object_projection, nn.Identity)

    def test_identity_rejects_mismatched_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "Identity object projection"):
            make_backbone(
                quantizer_embedding_dim=4,
                object_projection_mode="identity",
            )

    def test_unknown_projection_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "object_projection_mode"):
            make_backbone(object_projection_mode="unsupported")


if __name__ == "__main__":
    unittest.main()
