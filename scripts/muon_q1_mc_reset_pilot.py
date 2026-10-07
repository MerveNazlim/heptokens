#!/usr/bin/env python3
"""Run one matched MC-only muon pilot with periodic unused-code replacement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from lightning.pytorch import Callback
from omegaconf import OmegaConf

import muon_q1_codebook_control as history
import muon_q1_init_control as initialization
import muon_q1_mc_only_pilot as mc

ARM = "data_init"
AUDIT_TARGET = "muon_q1_mc_reset_pilot.ResetActionAudit"


class ResetActionAudit(Callback):
    """Observe already-logged reset actions without modifying the model or RNG."""

    def __init__(self):
        self.records = {}

    def capture(self, trainer, model):
        step = trainer.global_step
        interval = int(model.hparams.dead_code_reset_interval)
        if not step or step % interval or step in self.records:
            return
        keys = {
            "codes_reset": "train/codebook/dead_codes_reset",
            "unused_before_reset": "train/codebook/q0_dead_before_reset",
        }
        record = {"step": step, "epoch": trainer.current_epoch}
        for name, key in keys.items():
            value = trainer.callback_metrics.get(key)
            if value is None:
                raise RuntimeError(f"Reset action at step {step} was not logged: {key}")
            record[name] = int(value)
        self.records[step] = record

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # The model's batch-end reset runs after callback batch-end hooks.
        self.capture(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self.capture(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        self.capture(trainer, pl_module)
        if not self.records:
            raise RuntimeError("Pilot completed without reaching a periodic codebook reset")
        initialization.write_json(
            Path(trainer.default_root_dir) / "reset_actions.json",
            {"actions": [self.records[k] for k in sorted(self.records)]},
        )


def baseline_context(path, root):
    plan = mc.verify_plan(path)
    if plan["root"] != str(root) or plan["pilot_epochs"] != 3:
        raise ValueError("Use the prepared three-epoch MC train-fit pilot in its original checkout")
    cfg = initialization.load_config(plan["configs"][ARM]["path"])
    model = cfg["model"]
    if model["dead_code_reset"] or not model["data_codebook_init"]:
        raise ValueError("Reference must have initialization ON and periodic reset OFF")
    policy = (
        model["dead_code_reset_interval"],
        model["dead_code_reset_min_count"],
        model["dead_code_reset_max_fraction"],
        model["dead_code_reset_sample_size"],
    )
    if policy != (1000, 0, 0.25, 4096) or model.get("dead_code_reset_quantizers") not in (
        None,
        [0],
    ):
        raise ValueError("Unexpected saved reset policy; review before submitting")
    return plan, cfg


def reset_config(source, output):
    cfg = initialization.arm_config(source, output, ARM, pilot_epochs=3)
    cfg["datamodule"]["split_audit_path"] = str(output / ARM / "mc_membership.json")
    cfg["model"]["dead_code_reset"] = True
    cfg["callbacks"]["reset_action_audit"] = {"_target_": AUDIT_TARGET}
    return cfg


def preflight():
    """Check real-backend replacement and EMA state without physics inputs."""
    import torch
    from heptokens.models.vq_vae import LitVqVae

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = LitVqVae(
            encoder=lambda **kw: torch.nn.Identity(),
            decoder=lambda **kw: torch.nn.Identity(),
            codebook_size=16384,
            codebook_dim=8,
            num_quantizers=1,
            dead_code_reset=True,
        )
        layer = model.vector_quantization.layers[0]
        before, ema_before = layer.embed.clone(), layer.embed_avg.clone()
        model._dead_code_usage[0, :2] = 10
        model._dead_code_reset_sample = torch.arange(32, dtype=torch.float32).reshape(4, 8) + 10
        replaced, stats = model._reset_dead_codes()
        changed = (layer.embed != before).any(dim=0)
        assert replaced == int(changed.sum()) == 4096, "Wrong reset cap/backend update"
        assert stats[0] == {"dead_before_reset": 16382, "codes_reset": 4096}
        assert torch.equal(layer.embed[:, changed], layer.embed_avg[:, changed]), "Stale EMA sums"
        assert torch.equal(layer.embed[:, :2], before[:, :2]), "Used entries were reset"
        assert torch.equal(layer.embed_avg[:, :2], ema_before[:, :2])
        assert torch.equal(layer.cluster_size[changed], torch.ones(4096)), "Stale EMA counts"
        assert not bool(model._dead_code_usage.any()) and model._dead_code_reset_sample is None
        z, indices, loss = model.vector_quantization(torch.randn(4, 1, 8))
        assert indices.shape[-1] == 1 and torch.isfinite(z).all() and torch.isfinite(loss).all()
    print("PASS: Q1/cb16384 periodic-reset/EMA preflight (CPU, no physics inference)")


def verify_plan(path):
    plan = json.loads(path.read_text())
    if plan["kind"] != "muon_q1_mc_reset_pilot" or Path(plan["output"]) != path.parent.resolve():
        raise ValueError("Wrong or moved reset pilot plan")
    baseline, source = baseline_context(Path(plan["baseline_plan"]), Path(plan["root"]))
    for record in plan["inputs"] + plan["sources"] + list(plan["configs"].values()):
        initialization.verify_record(record)
    if plan["runtime"] != baseline["runtime"] or plan["runtime"] != initialization.runtime_record():
        raise ValueError("Reset/reference runtime differs")
    actual = initialization.load_config(plan["configs"][ARM]["path"])
    differences = initialization.diff_paths(reset_config(source, path.parent.resolve()), actual)
    if differences:
        raise ValueError(f"Unexpected reset-pilot settings: {differences}")
    return plan, baseline


def describe(plan):
    cfg = initialization.load_config(plan["configs"][ARM]["path"])
    print("Muon Q1 MC-only RESET: ONE fresh three-epoch pilot")
    print(f"  reset-OFF MC reference: {Path(plan['baseline_plan']).parent}")
    print("  initialization ON; periodic codebook reset ON (Q0 only)")
    print("  every 1000 steps; zero assignments in the window; cap=25% (4096 entries)")
    print(
        "  replacements sampled from recent valid training encoder outputs; not a full-codebook reset"
    )
    print("  same Q1/K=16384/dim8, widths=[128,256,512], batch=1024, seed=42")
    print("  same three-epoch budget, 20-epoch scheduler, MC membership/features/losses")
    print(f"  same MC-training-only joblib: {initialization.joblib_paths(cfg['datamodule'])}")
    print("  no refit or extra MC; can run alongside reference after its preprocessing fit")
    print(f"  output: {plan['output']}")


def prepare(args):
    root, output = args.root.resolve(), args.output.resolve()
    baseline_path = args.baseline.resolve() / "plan.json"
    path = output / "plan.json"
    if path.exists():
        plan, _ = verify_plan(path)
        if plan["root"] != str(root) or plan["baseline_plan"] != str(baseline_path):
            raise ValueError("Existing reset plan has different inputs")
        describe(plan)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output; nothing overwritten")
    baseline, source = baseline_context(baseline_path, root)
    original, _ = mc.baseline_context(Path(baseline["baseline_plan"]), root)
    for protected in (
        baseline_path.parent,
        Path(baseline["baseline_plan"]).parent,
        Path(original["reference"]).parent,
    ):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Keep reset output separate from the reference trainings")
    cfg = reset_config(source, output)
    preflight()
    (output / "configs").mkdir(parents=True)
    config_path = output / "configs/data_init.yaml"
    OmegaConf.save(OmegaConf.create(cfg), config_path)
    plan = {
        "version": 1,
        "kind": "muon_q1_mc_reset_pilot",
        "root": str(root),
        "output": str(output),
        "baseline_plan": str(baseline_path),
        "pilot_epochs": 3,
        "runtime": baseline["runtime"],
        "inputs": [initialization.file_record(baseline_path)],
        "sources": [initialization.file_record(root / "scripts/muon_q1_mc_reset_pilot.py")],
        "configs": {ARM: initialization.file_record(config_path)},
        "source_config_changes": initialization.diff_paths(source, cfg),
        "intervention": "model.dead_code_reset: false -> true (saved reset policy unchanged)",
        "notes": [
            "Reuse the reference MC-only training-fitted joblib; never refit/copy its data.",
            "Only periodic reset changes model behavior; audit callback only observes logged reset actions.",
            "Reference need not finish training before this pilot starts; preprocessing must finish first.",
            "Reset sampling consumes RNG; same seed does not imply identical shuffled training batches.",
            "Compare matched validation inputs/steps. Occupancy alone is not reconstruction quality.",
        ],
    }
    initialization.write_json(path, plan)
    describe(plan)
    print("Frozen reset plan; no job submitted, no H5 loading or preprocessing fit")


def wait_preprocessing(baseline, seconds):
    if seconds < 0:
        raise ValueError("Preprocessing wait must be nonnegative")
    deadline = time.monotonic() + seconds
    receipt = Path(baseline["preprocessing"]["receipt"])
    if not receipt.is_file():
        print(
            "Waiting for GPU3 reference's MC-only preprocessing fit; no GPU training yet",
            flush=True,
        )
    while not receipt.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "MC joblib not ready. Finish the reference preprocessing fit, then submit again."
            )
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    return mc.verify_preprocessing(baseline)


def run(args):
    import fcntl
    import torch

    output = args.output.resolve()
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, baseline = verify_plan(output / "plan.json")
        run_dir = output / ARM
        if (run_dir.exists() and any(run_dir.iterdir())) or (output / "data_init.log").exists():
            raise ValueError("Refusing to overwrite/resume reset pilot")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU using CUDA_VISIBLE_DEVICES")
        metadata = wait_preprocessing(baseline, args.wait_fit_seconds)
        initialization.write_json(
            output / "shared_preprocessing.json",
            {
                "metadata": metadata,
                "receipt": initialization.file_record(Path(baseline["preprocessing"]["receipt"])),
            },
        )
        preflight()
        cfg_path = Path(plan["configs"][ARM]["path"])
        cmd = [
            sys.executable,
            str(Path(plan["root"]) / "scripts/train.py"),
            "--config-path",
            str(cfg_path.parent),
            "--config-name",
            ARM,
        ]
        cfg = initialization.load_config(cfg_path)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(plan["root"]) / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["WANDB_MODE"] = "offline" if cfg["logger"].get("offline") else "online"
        initialization.write_json(
            output / "execution.json",
            {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": str(torch.cuda.get_device_properties(0)),
                "torch_cuda": torch.version.cuda,
            },
        )
        print(f"START reset-ON pilot: {' '.join(cmd)}", flush=True)
        with (output / "data_init.log").open("x") as log:
            subprocess.run(
                cmd, cwd=plan["root"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        summarize(args)


def summarize(args):
    output = args.output.resolve()
    plan, baseline = verify_plan(output / "plan.json")
    initialization.verify_pilot_completion(plan, ARM)
    shared = json.loads((output / "shared_preprocessing.json").read_text())
    initialization.verify_record(shared["receipt"])
    if shared["metadata"] != mc.verify_preprocessing(baseline):
        raise ValueError("Shared preprocessing changed during the reset pilot")
    membership = json.loads((output / ARM / "mc_membership.json").read_text())
    if membership["mc_files"] != baseline["mc_paths"] or (
        membership["partitions"]["train"]["ordered_compact_indices_sha256"]
        != shared["metadata"]["ordered_compact_train_indices_sha256"]
    ):
        raise ValueError("Reset training and shared preprocessing used different MC membership")
    rows, _ = history.read_history(output, 16384)
    actions = json.loads((output / ARM / "reset_actions.json").read_text())["actions"]
    if not actions or not any(r["codes_reset"] > 0 for r in actions):
        raise ValueError("No entries actually reset; not a successful reset intervention")
    candidates = [r for r in rows.values() if r["initialization_completed"]]
    if not candidates:
        raise ValueError("No post-initialization reset validation passes")
    best = min(candidates, key=lambda r: r["validation_metrics"]["val/total_loss"])
    compact = lambda r: {k: v for k, v in r.items() if k != "counts"}
    initialization.write_json(
        output / "pilot_summary.json",
        {
            "best_post_init_by_total_loss": compact(best),
            "mc_membership": membership,
            "reset_actions": actions,
            "validation_passes": [compact(rows[k]) for k in sorted(rows)],
        },
    )
    reference = Path(baseline["output"])
    if not (reference / ARM / "SUCCESS.txt").is_file():
        print(
            "Reset pilot complete. Matched comparison pending reference completion; run summarize later."
        )
        return
    initialization.verify_pilot_completion(baseline, ARM)
    reference_rows, _ = history.read_history(reference, 16384)
    if set(reference_rows) != set(rows):
        raise ValueError("Different validation epoch/step sets between reset OFF/ON")
    pairs = []
    for key in sorted(rows):
        one, two = reference_rows[key], rows[key]
        for field in ("input_tensor_sha256", "validation_batches"):
            if one[field] != two[field]:
                raise ValueError(f"Different ordered validation {field} at {key}")
        if one["codebook_metrics"][0]["objects"] != two["codebook_metrics"][0]["objects"]:
            raise ValueError(f"Different validation object counts at {key}")
        if one["initialization_completed"] and two["initialization_completed"]:
            pairs.append(
                {
                    "epoch": key[0],
                    "step": key[1],
                    "reset_off": compact(one),
                    "reset_on": compact(two),
                }
            )
    if not pairs:
        raise ValueError("No matched post-initialization validation passes")
    best_off = min(pairs, key=lambda r: r["reset_off"]["validation_metrics"]["val/total_loss"])
    best_on = min(pairs, key=lambda r: r["reset_on"]["validation_metrics"]["val/total_loss"])
    initialization.write_json(
        output / "comparison.json",
        {
            "paired_validation_passes": pairs,
            "at_reference_best": best_off,
            "best_reset_on": best_on,
            "final_matched_pass": pairs[-1],
            "reset_actions": actions,
            "note": "Identical MC validation and preprocessing; compare reconstruction errors as well as usage. "
            "Same seed does not fix training order after reset RNG draws. Three epochs are an early-behavior pilot.",
        },
    )
    print(
        f"Verified {len(pairs)} matched reset-OFF/ON validation passes: {output / 'comparison.json'}"
    )
    for arm in ("reset_off", "reset_on"):
        row = best_off[arm]
        print(
            f"At reference-best step {best_off['step']}: {arm} recon={row['validation_metrics']['val/recon_loss']:.6g}; usage={row['codebook_metrics'][0]}"
        )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=root / "results/atlas_muon_q1_mc_only_trainfit_pilot_e3_v1"
    )
    parser.add_argument("--wait-fit-seconds", type=int, default=3600)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
