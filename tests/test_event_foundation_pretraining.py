from pathlib import Path
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import torch
import torch.nn.functional as F

from heptokens.data.continuous_schema import ROLE_IDS, build_continuous_schema


def _config(experiment):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                "seed=43",
                "datamodule.prepared_dir=/example/prepared",
                "output_dir=/example/results",
            ],
        )


def _assert_executed_training_protocol(experiment, data_seed, step_logging):
    cfg = _config(experiment)
    assert cfg.precision == "medium"
    assert cfg.datamodule.seed == data_seed
    assert cfg.datamodule.batch_size == 448
    assert cfg.datamodule.num_workers == 3
    assert cfg.datamodule.stream_batch_size == 4096
    assert cfg.datamodule.shuffle_buffer_size == 8192
    assert cfg.trainer.devices == 8
    assert cfg.trainer.strategy == "ddp"
    assert cfg.trainer.max_epochs == 10
    assert cfg.trainer.limit_train_batches == 1.0
    assert cfg.trainer.limit_val_batches == 500
    assert cfg.trainer.val_check_interval == 1.0
    assert cfg.trainer.log_every_n_steps == step_logging
    assert cfg.model.mask_prob == 0.15
    assert cfg.model.hidden_dim == 256
    assert cfg.model.num_layers == 4
    assert cfg.model.num_heads == 8
    if experiment == "event_foundation_q8":
        assert cfg.datamodule.include_tokens
        assert cfg.model.vocab_size == 131588
        assert cfg.model.max_quantizers == 8
        assert cfg.model.quantizer_embedding_dim == 32
    else:
        assert not cfg.datamodule.include_tokens
        assert cfg.datamodule.require_continuous
        assert cfg.model.architecture == "flat"
        assert cfg.callbacks.periodic_checkpoint.every_n_train_steps == 2000


def _assert_parallel_q8_object_masking():
    model = instantiate(
        _config("event_foundation_q8").model,
        hidden_dim=16,
        vocab_size=20,
        num_heads=2,
        num_layers=1,
        mask_prob=0.0,
    ).eval()
    # An empty Bernoulli draw forces one position per minibatch, not per event.
    batch = {
        "tokens": torch.tensor([[[1] + [0] * 7, list(range(4, 12))]] * 2),
        "mask": torch.tensor([[1, 1], [1, 1]]),
        "type_ids": torch.tensor([[0, 8], [0, 8]]),
    }
    assert model.backbone.object_projection.in_features == 8 * 32
    with patch.object(model, "log"), patch.object(
        model.model, "masked_logits", wraps=model.model.masked_logits
    ) as predict:
        loss = model.training_step(batch)
    masked_tokens, mask, type_ids, positions = predict.call_args.args
    assert positions.tolist() == [[False, True], [False, False]]
    assert masked_tokens[0, 1].tolist() == [3] * 8
    assert torch.equal(masked_tokens[1], batch["tokens"][1])
    logits = model.model.masked_logits(masked_tokens, mask, type_ids, positions)
    assert logits.shape == (1, 8, 20)
    torch.testing.assert_close(loss, F.cross_entropy(logits[0], batch["tokens"][0, 1]))
    loss.backward()


def _assert_flat_continuous_loss_reduction():
    schema = build_continuous_schema(
        collections={"jets": {"inputs": ["common/jets/pt", "common/jets/eta"]}},
        object_order=["jets"],
        event_token_specs=[{"input": "common/event/mu", "range": [0.0, 100.0]}],
        type_ids={"jets": 8},
    )
    model = instantiate(
        _config("event_foundation_continuous").model,
        continuous_schema=schema,
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        mask_prob=1.0,
    ).eval()
    batch = {
        "continuous_features": torch.tensor([[[0.0, 0.0], [0.2, 0.0], [1.0, 3.0]]]),
        "continuous_feature_mask": torch.tensor([[[0, 0], [1, 0], [1, 1]]]),
        "mask": torch.tensor([[1, 1, 1]]),
        "type_ids": torch.tensor([[0, 0, 8]]),
        "position_role_ids": torch.tensor(
            [[ROLE_IDS["cls"], ROLE_IDS["event"], ROLE_IDS["object"]]]
        ),
    }
    assert model.backbone.inner_transformer is None
    with patch.object(model, "log"), patch.object(
        model.model, "reconstruct", wraps=model.model.reconstruct
    ) as reconstruct:
        loss = model.training_step(batch)
    args = reconstruct.call_args.kwargs
    assert args["masked_positions"].tolist() == [[False, True, True]]
    predicted = model.model.reconstruct(**args)
    features = batch["continuous_features"][args["masked_positions"]]
    squared_error = (predicted - features).square()
    expected_loss = (squared_error[0, 0] + squared_error[1].mean()) / 2
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()


class TestEventFoundationPretraining(unittest.TestCase):
    def test_q8_experiment_keeps_executed_training_protocol(self):
        _assert_executed_training_protocol("event_foundation_q8", 43, 50)

    def test_continuous_experiment_keeps_executed_training_protocol(self):
        _assert_executed_training_protocol("event_foundation_continuous", 42, 20)

    def test_parallel_q8_masks_one_object_and_predicts_eight_codes(self):
        _assert_parallel_q8_object_masking()

    def test_flat_continuous_loss_averages_features_then_positions(self):
        _assert_flat_continuous_loss_reduction()
