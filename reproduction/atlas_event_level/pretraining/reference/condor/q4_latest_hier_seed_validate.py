#!/usr/bin/env python3
"""Additional seed/epoch checks around the unchanged successful pretrain code.

Executed only on the GPU worker with its frozen project on PYTHONPATH. The
epoch count is derived from that project's real distributed sampler, not from
ceil(total_rows / global_batch), which is wrong for row-group streaming.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def epoch_plan(family, prepared, seed, batch, workers, devices, epochs):
    if family == "q4":
        from heptokens.data.token_parquet import TokenParquetPretrainModule
        extra = {}
    elif family == "hierarchical_kin_aux":
        from heptokens.data.benchmark_token_parquet import TokenParquetPretrainModule
        extra = dict(require_continuous=True, include_tokens=False,
                     continuous_feature_column="continuous_features")
    else:
        raise ValueError(f"Unknown family: {family}")
    module = TokenParquetPretrainModule(
        prepared_dir=str(prepared), seed=seed, batch_size=batch, num_workers=workers,
        n_classes=1, stream_batch_size=4096, shuffle_buffer_size=8192, **extra)
    quotas = module.train_set._distributed_worker_limits(devices, max(workers, 1))
    rank_steps = [sum(quotas[w * devices + r] // batch for w in range(max(workers, 1)))
                  for r in range(devices)]
    if not rank_steps or min(rank_steps) <= 0 or len(set(rank_steps)) != 1:
        raise ValueError(f"Unequal/empty DDP epoch: {rank_steps}")
    return dict(model_family=family, pretrain_seed=seed, fine_tuning_seed=42,
                batch_size_per_gpu=batch, num_devices=devices,
                global_batch_size=batch * devices, rank_steps=rank_steps,
                epochs=epochs, updates_per_epoch=rank_steps[0],
                expected_global_steps=rank_steps[0] * epochs)


def check_checkpoint(checkpoint, family, seed, *, epochs=None, steps=None):
    import torch
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = state.get("hyper_parameters", {})
    expected = dict(hidden_dim=256, num_heads=8, num_layers=4, max_seq_length=256,
                    mask_prob=0.15, learning_rate=1e-4)
    if family == "q4":
        expected.update(max_quantizers=4, vocab_size=262660,
                        quantizer_embedding_dim=64, object_projection_mode="identity",
                        quantizer_decoder="parallel",
                        ensure_masked_physical_object_per_event=True)
    else:
        expected.update(architecture="hierarchical", feature_grouping="kinematics_auxiliary_v1",
                        inner_num_heads=4, inner_num_layers=1)
    for name, value in expected.items():
        if hp.get(name) != value:
            raise ValueError(f"Checkpoint {name}={hp.get(name)!r}; expected {value!r}")
    # The module does not save the global RNG seed in hyper_parameters. The
    # run's resolved full_config.yaml is checked at completion below instead.
    if epochs is not None and state.get("epoch") != epochs - 1:
        raise ValueError(f"Incomplete epochs: {state.get('epoch')}; expected {epochs - 1}")
    if steps is not None and state.get("global_step") != steps:
        raise ValueError(f"Incomplete optimizer steps: {state.get('global_step')}; expected {steps}")
    for name, tensor in state.get("state_dict", {}).items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint parameter: {name}")
    return dict(epoch=state.get("epoch"), global_step=state.get("global_step"))


def completed(family, run_dir, output, seed, epoch_file):
    from omegaconf import OmegaConf
    plan = json.loads(Path(epoch_file).read_text())
    if plan["model_family"] != family or plan["pretrain_seed"] != seed:
        raise ValueError("Epoch plan belongs to a different family/seed")
    cfg = OmegaConf.load(run_dir / "full_config.yaml")
    expected = {"seed": seed, "datamodule.seed": seed, "datamodule.batch_size": 597,
                "datamodule.num_workers": 3, "trainer.devices": 6, "trainer.num_nodes": 1,
                "trainer.max_epochs": 10, "trainer.limit_train_batches": 1.0,
                "trainer.limit_val_batches": 500, "model.hidden_dim": 256,
                "model.num_layers": 4, "model.num_heads": 8,
                "model.optimizer.lr": 1e-4, "model.optimizer.weight_decay": 1e-2,
                "model.scheduler.warmup_epochs": 1, "model.scheduler.min_lr": 1e-6}
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f"Resolved configuration {key}={OmegaConf.select(cfg, key)!r}; expected {value!r}")
    if OmegaConf.select(cfg, "ckpt_path") not in (None, "null"):
        # Only the original hierarchical same-Condor-job auto-restart is allowed.
        if family != "hierarchical_kin_aux" or not str(cfg.ckpt_path).endswith("/resume/resume.ckpt"):
            raise ValueError("Unexpected external starting checkpoint")
    checkpoint = run_dir / "checkpoints/last.ckpt"
    observed = check_checkpoint(checkpoint, family, seed,
                                epochs=10, steps=plan["expected_global_steps"])
    def sha(path):
        result = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
        return result.hexdigest()
    report = dict(passed=True, checkpoint_sha256=sha(checkpoint),
                  full_config_sha256=sha(run_dir / "full_config.yaml"), **plan, **observed)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print("PASS completed seed-replicate pretraining", json.dumps(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    epoch = commands.add_parser("epoch-plan")
    epoch.add_argument("--prepared-dir", type=Path, required=True)
    epoch.add_argument("--output", type=Path, required=True)
    epoch.add_argument("--batch", type=int, default=597)
    epoch.add_argument("--workers", type=int, default=3)
    epoch.add_argument("--devices", type=int, default=6)
    epoch.add_argument("--epochs", type=int, default=10)
    final = commands.add_parser("completed")
    final.add_argument("--run-dir", type=Path, required=True)
    final.add_argument("--epoch-plan", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    for command in (epoch, final):
        command.add_argument("--family", choices=("q4", "hierarchical_kin_aux"), required=True)
        command.add_argument("--seed", type=int, choices=(43, 44), required=True)
    args = parser.parse_args(argv)
    if args.command == "epoch-plan":
        report = epoch_plan(args.family, args.prepared_dir, args.seed, args.batch,
                            args.workers, args.devices, args.epochs)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS actual distributed epoch plan", json.dumps(report))
    else:
        completed(args.family, args.run_dir, args.output, args.seed, args.epoch_plan)


if __name__ == "__main__":
    main()
