"""Regression tests for Zephyr's Q0-only initialization and EMA reset fix."""

import unittest

import torch

from heptokens.models.vq_vae import LitVqVae


def make_model(*, initialize=False, reset=False):
    return LitVqVae(
        encoder=torch.nn.Identity,
        decoder=torch.nn.Identity,
        codebook_size=8,
        codebook_dim=3,
        num_quantizers=2,
        data_codebook_init=initialize,
        data_codebook_init_samples=16,
        data_codebook_init_quantizers=[0],
        dead_code_reset=reset,
        dead_code_reset_quantizers=[0],
    )


class TestCodebookInitialization(unittest.TestCase):
    def test_reset_updates_transposed_embeddings_ema_and_counts(self):
        model = make_model()
        layer = model.vector_quantization.layers[0]
        before_embed = layer.embed.clone()
        before_ema = layer.embed_avg.clone()
        indices = torch.tensor([1, 3])
        replacements = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        model._reset_layer_codes(layer, indices, replacements)
        torch.testing.assert_close(layer.embed[:, indices].T, replacements)
        torch.testing.assert_close(layer.embed_avg[:, indices].T, replacements)
        torch.testing.assert_close(layer.cluster_size[indices], torch.ones(2))
        other = torch.tensor([0, 2, 4, 5, 6, 7])
        torch.testing.assert_close(layer.embed[:, other], before_embed[:, other])
        torch.testing.assert_close(layer.embed_avg[:, other], before_ema[:, other])

    def test_q0_initialization_keeps_training_rng_and_q1_unchanged(self):
        model = make_model(initialize=True)
        second = model.vector_quantization.layers[1]
        before_second = [x.clone() for x in (second.embed, second.embed_avg, second.cluster_size)]
        samples = torch.arange(90, dtype=torch.float32).reshape(30, 1, 3)
        mask = torch.ones(30, 1, dtype=torch.bool)
        mask[::3] = False
        before_rng = torch.get_rng_state().clone()
        model._collect_data_codebook_init_samples(samples, {"mask": mask})
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(model._data_codebook_init_sample_count, 16)
        self.assertEqual(model._initialize_codebooks_from_data(), 8)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        first = model.vector_quantization.layers[0]
        torch.testing.assert_close(first.embed, first.embed_avg)
        torch.testing.assert_close(first.cluster_size, torch.ones(8))
        valid_samples = samples[mask]
        for code in first.embed.T:
            self.assertTrue(torch.any(torch.all(valid_samples == code, dim=-1)))
        for before, after in zip(before_second, (second.embed, second.embed_avg, second.cluster_size)):
            torch.testing.assert_close(before, after)

    def test_q0_reset_respects_cap_and_preserves_used_codes_and_q1(self):
        model = make_model(reset=True)
        first, second = model.vector_quantization.layers
        before_embed = first.embed.clone()
        before_second = [x.clone() for x in (second.embed, second.embed_avg, second.cluster_size)]
        model._dead_code_usage[0, :2] = 10
        model._dead_code_reset_sample = torch.arange(48, dtype=torch.float32).reshape(16, 3)
        replaced, _ = model._reset_dead_codes()
        self.assertEqual(replaced, 2)
        self.assertEqual(int((first.embed != before_embed).any(dim=0).sum()), 2)
        torch.testing.assert_close(first.embed[:, :2], before_embed[:, :2])
        for before, after in zip(before_second, (second.embed, second.embed_avg, second.cluster_size)):
            torch.testing.assert_close(before, after)


if __name__ == "__main__":
    unittest.main()
