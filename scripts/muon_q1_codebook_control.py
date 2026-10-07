#!/usr/bin/env python3
"""Run one smaller-codebook muon Q1 pilot against a completed data-init pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

import muon_q1_init_control as initialization

ARM = "data_init"


def read_history(output, codebook_size):
    run = output / ARM
    if not (run / "SUCCESS.txt").is_file():
        raise ValueError(f"Training has not completed: {run}")
    records = {}
    paths = sorted((run / "validation_usage").glob("*.json"))
    for path in paths:
        row = json.loads(path.read_text())
        key = (row["epoch"], row["step"])
        if key in records:
            raise ValueError(f"Duplicate validation epoch/step: {key}")
        counts, usage = row["counts"], row["codebook_metrics"]
        if len(counts) != 1 or len(counts[0]) != codebook_size or len(usage) != 1:
            raise ValueError(f"Wrong Q1/codebook capacity in {path}")
        if any(type(n) is not int or n < 0 for n in counts[0]):
            raise ValueError(f"Invalid code assignment counts in {path}")
        objects = sum(counts[0])
        used = sum(n > 0 for n in counts[0])
        if (
            not row.get("data_codebook_init")
            or objects <= 0
            or usage[0]["q"] != 0
            or usage[0]["objects"] != objects
            or usage[0]["used_codes"] != used
            or not math.isclose(usage[0]["used_fraction"], used / codebook_size)
        ):
            raise ValueError(f"Inconsistent data-init usage audit in {path}")
        for name in ("val/total_loss", "val/recon_loss"):
            if not math.isfinite(row["validation_metrics"][name]):
                raise ValueError(f"Missing/nonfinite validation loss in {path}")
        records[key] = row
    if not records or not records[max(records)]["initialization_completed"]:
        raise ValueError(f"Missing history or unfinished data initialization: {run}")
    return records, paths


def baseline_context(path, root=None):
    plan = initialization.verify_plan(path)
    if root is not None and plan["root"] != str(root):
        raise ValueError("Use the completed baseline's original checkout")
    if not 0 < plan.get("pilot_epochs", 0) < 20:
        raise ValueError("Baseline must be a completed short pilot with the 20-epoch schedule")
    initialization.validate_reference(initialization.load_config(plan["reference"]))
    initialization.verify_pilot_completion(plan, ARM)
    cfg = initialization.load_config(plan["configs"][ARM]["path"])
    if not cfg["model"].get("data_codebook_init"):
        raise ValueError("Compare against the completed data_init arm")
    history, paths = read_history(Path(plan["output"]), 16384)
    return plan, cfg, history, paths


def capacity_config(baseline, output, pilot_epochs, codebook_size):
    if type(codebook_size) is not int or not 8 < codebook_size < 16384:
        raise ValueError("Use a smaller codebook with 9-16383 entries; default is 1024")
    cfg = initialization.arm_config(baseline, output, ARM, pilot_epochs)
    cfg["model"]["codebook_size"] = codebook_size
    return cfg


def verify_capacity_config(baseline, cfg, output, pilot_epochs, codebook_size):
    expected = capacity_config(baseline, output, pilot_epochs, codebook_size)
    differences = initialization.diff_paths(expected, cfg)
    if differences:
        raise ValueError(f"Unexpected settings changed beyond codebook size: {differences}")


def preflight(codebook_size):
    """Check the actual smaller EMA codebook without loading physics inputs."""
    import torch
    from heptokens.models.vq_vae import LitVqVae

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = LitVqVae(
            encoder=lambda **kw: torch.nn.Identity(),
            decoder=lambda **kw: torch.nn.Identity(),
            codebook_size=codebook_size,
            codebook_dim=8,
            num_quantizers=1,
            data_codebook_init=True,
            data_codebook_init_samples=65536,
        )
        samples = torch.randn(65536, 1, 8)
        before = torch.get_rng_state().clone()
        model._collect_data_codebook_init_samples(samples, {"mask": torch.ones(65536, 1).bool()})
        assert model._initialize_codebooks_from_data() == codebook_size
        assert torch.equal(before, torch.get_rng_state()), "Initialization advanced the RNG"
        layer = model.vector_quantization.layers[0]
        assert layer.embed.shape == (8, codebook_size), "Unexpected quantizer backend layout"
        assert torch.equal(layer.embed, layer.embed_avg), "Stale EMA embedding buffer"
        assert torch.equal(layer.cluster_size, torch.ones(codebook_size)), "Stale EMA counts"
        quantized, indices, loss = model.vector_quantization(samples[:4])
        assert indices.shape == (1, 4, 1), "Unexpected quantizer index layout"
        assert torch.isfinite(quantized).all() and torch.isfinite(loss).all()
        assert torch.isfinite(layer.embed).all(), "Nonfinite EMA update"
    print(f"PASS: Q1 cb{codebook_size} initialization/EMA/RNG preflight (CPU, no inference)")


def verify_plan(path):
    plan = json.loads(path.read_text())
    if plan["kind"] != "muon_q1_codebook_control" or Path(plan["output"]) != path.parent.resolve():
        raise ValueError("Wrong or moved capacity-control plan")
    baseline, cfg, _, paths = baseline_context(Path(plan["baseline_plan"]), Path(plan["root"]))
    for record in plan["inputs"] + plan["sources"] + list(plan["configs"].values()):
        initialization.verify_record(record)
    if [str(p) for p in paths] != plan["baseline_history_paths"]:
        raise ValueError("Completed baseline validation history changed")
    if initialization.runtime_record() != plan["runtime"]:
        raise ValueError("Runtime changed; keep the completed baseline runtime")
    if plan["pilot_epochs"] != baseline["pilot_epochs"]:
        raise ValueError("Capacity pilot budget differs from the completed baseline")
    verify_capacity_config(
        cfg,
        initialization.load_config(plan["configs"][ARM]["path"]),
        path.parent.resolve(),
        plan["pilot_epochs"],
        plan["codebook_size"],
    )
    return plan


def describe(plan):
    cfg = initialization.load_config(plan["configs"][ARM]["path"])
    dm = cfg["datamodule"]
    print("Muon Q1 smaller-codebook control: ONE additional fresh training")
    print(f"  completed reference: {Path(plan['baseline_plan']).parent / ARM}")
    print(f"  K: 16384 -> {plan['codebook_size']}; Q=1; dim=8; widths=[128,256,512]")
    print(f"  epochs to run={plan['pilot_epochs']}; scheduler horizon=20")
    print(f"  batch={dm['batch_size']}; seed={cfg['seed']}; saved files={len(dm['data_paths'])}")
    print(
        f"  split: train={dm['train_frac']}, val={dm['val_frac']}, "
        f"test={dm['test_frac']}; split seed={dm['seed']}"
    )
    features = next(c["inputs"] for c in dm["object_collections"] if c["object_name"] == "muons")
    print(f"  features (saved order): {features}")
    print(f"  unchanged joblibs: {initialization.joblib_paths(dm)}")
    print(f"  validation limit: {cfg['trainer'].get('limit_val_batches', 1.0)} batches/fraction")
    print("  data initialization=true; pool=65536; dead-code reset=false")
    print("  all other training settings inherited; validation input hashes must match")
    print(f"  output: {plan['output']}")


def prepare(args):
    root, output = args.root.resolve(), args.output.resolve()
    baseline_path = args.baseline.resolve() / "plan.json"
    path = output / "plan.json"
    if path.exists():
        plan = verify_plan(path)
        if (
            plan["root"] != str(root)
            or plan["baseline_plan"] != str(baseline_path)
            or plan["codebook_size"] != args.codebook_size
        ):
            raise ValueError("Existing plan has different settings; choose a separate output")
        describe(plan)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty without a valid plan; nothing overwritten")
    baseline, source, _, history_paths = baseline_context(baseline_path, root)
    production = Path(baseline["reference"]).parent
    for protected in (baseline_path.parent, production):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Keep the new output separate from completed training directories")
    cfg = capacity_config(source, output, baseline["pilot_epochs"], args.codebook_size)
    verify_capacity_config(source, cfg, output, baseline["pilot_epochs"], args.codebook_size)
    completed = baseline_path.parent / ARM
    inputs = [
        initialization.file_record(p)
        for p in (
            baseline_path,
            completed / "SUCCESS.txt",
            completed / "pilot_completion.json",
            *history_paths,
        )
    ]
    sources = [initialization.file_record(root / "scripts/muon_q1_codebook_control.py")]
    preflight(args.codebook_size)
    (output / "configs").mkdir(parents=True)
    config_path = output / "configs" / f"{ARM}.yaml"
    OmegaConf.save(OmegaConf.create(cfg), config_path)
    plan = {
        "version": 1,
        "kind": "muon_q1_codebook_control",
        "root": str(root),
        "output": str(output),
        "baseline_plan": str(baseline_path),
        "baseline_history_paths": [str(p) for p in history_paths],
        "pilot_epochs": baseline["pilot_epochs"],
        "codebook_size": args.codebook_size,
        "inputs": inputs,
        "sources": sources,
        "runtime": baseline["runtime"],
        "configs": {ARM: initialization.file_record(config_path)},
        "intervention": f"model.codebook_size: 16384 -> {args.codebook_size}",
        "notes": [
            "Reuse the completed K16384 data_init arm; train only the smaller K arm.",
            "The baseline plan continues to verify original inputs, joblibs, code and runtime.",
            "Retain the completed pilot budget and original 20-epoch learning-rate schedule.",
            "Same seed and split; changing K can alter later RNG consumption and shuffled training order.",
            "Verify every paired validation input hash; no historical training-order audit is claimed.",
            "Select using post-initialization validation errors; physical-unit plots are a later step.",
        ],
    }
    initialization.write_json(path, plan)
    describe(plan)
    print(f"Frozen plan: {path}; no training submitted")


def run(args):
    import fcntl
    import torch

    output = args.output.resolve()
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify_plan(output / "plan.json")
        dest = output / ARM
        if dest.exists() and any(dest.iterdir()) or (output / f"{ARM}.log").exists():
            raise ValueError(f"Refusing to overwrite/resume existing training: {dest}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
        preflight(plan["codebook_size"])
        initialization.write_json(
            output / "execution.json",
            {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": str(torch.cuda.get_device_properties(0)),
                "torch_cuda": torch.version.cuda,
            },
        )
        cfg_path = Path(plan["configs"][ARM]["path"])
        cmd = [
            sys.executable,
            str(Path(plan["root"]) / "scripts/train.py"),
            "--config-path",
            str(cfg_path.parent),
            "--config-name",
            ARM,
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(plan["root"]) / "src") + os.pathsep + env.get("PYTHONPATH", "")
        cfg = initialization.load_config(cfg_path)
        env["WANDB_MODE"] = "offline" if cfg["logger"].get("offline") else "online"
        print(f"START cb{plan['codebook_size']}: {' '.join(cmd)}", flush=True)
        with (output / f"{ARM}.log").open("x") as log:
            subprocess.run(
                cmd, cwd=plan["root"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        initialization.verify_pilot_completion(plan, ARM)
        summarize(args)


def summarize(args):
    output = args.output.resolve()
    plan = verify_plan(output / "plan.json")
    _, _, reference, _ = baseline_context(Path(plan["baseline_plan"]))
    initialization.verify_pilot_completion(plan, ARM)
    small, _ = read_history(output, plan["codebook_size"])
    if reference.keys() != small.keys():
        raise ValueError("Validation epoch/step histories differ from the completed baseline")
    labels = ("cb16384", f"cb{plan['codebook_size']}")
    paired = []
    for key in sorted(reference):
        a, b = reference[key], small[key]
        if (
            a["input_tensor_sha256"] != b["input_tensor_sha256"]
            or a["validation_batches"] != b["validation_batches"]
            or a["codebook_metrics"][0]["objects"] != b["codebook_metrics"][0]["objects"]
        ):
            raise ValueError(f"Different ordered validation objects at epoch/step {key}")
        paired.append(
            {
                "epoch": key[0],
                "step": key[1],
                **{
                    label: {k: v for k, v in row.items() if k != "counts"}
                    for label, row in zip(labels, (a, b))
                },
            }
        )
    eligible = [p for p in paired if all(p[l]["initialization_completed"] for l in labels)]
    if not eligible:
        raise ValueError("No matching post-initialization validation passes")
    best = {
        label: min(eligible, key=lambda p: p[label]["validation_metrics"]["val/total_loss"])
        for label in labels
    }
    initialization.write_json(
        output / "comparison.json",
        {
            "baseline": str(Path(plan["baseline_plan"]).parent),
            "pilot_epochs": plan["pilot_epochs"],
            "codebook_sizes": [16384, plan["codebook_size"]],
            "paired_validation_passes": paired,
            "best_post_init_by_total_loss": {label: best[label][label] for label in labels},
            "at_reference_best": best[labels[0]],
            "final_matched_pass": eligible[-1],
            "note": "Compare reconstruction and per-feature errors in preprocessed units, not usage alone. "
            "Perplexity/K and used/K have different denominators; retain absolute counts/perplexity too.",
        },
    )
    print(f"Verified {len(paired)} matching validation passes: {output / 'comparison.json'}")
    for label in labels:
        row = best[label][label]
        print(
            f"{label}: best post-init val/total_loss at step {row['step']}; "
            f"recon={row['validation_metrics']['val/recon_loss']:.6g}; "
            f"usage={row['codebook_metrics'][0]}"
        )
    print(
        f"Also saved both configurations at reference-best step {best[labels[0]]['step']} and final step."
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=root / "results/atlas_muon_q1_initialization_pilot_e3_v1",
    )
    parser.add_argument("--codebook-size", type=int, default=1024)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
