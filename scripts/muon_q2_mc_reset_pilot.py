#!/usr/bin/env python3
"""Repeat the MC-only muon reset comparison with Q2 and 8192 entries per codebook."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

import muon_q1_codebook_control as history
import muon_q1_init_control as initialization
import muon_q1_mc_only_pilot as mc
import muon_q1_mc_reset_pilot as reset

ARMS = ("reset_off", "reset_on")
K, NQ, DIM, EPOCHS = 8192, 2, 8, 3
KIND = "muon_q2_cb8192_mc_reset_pilot"
SOURCE_FILES = (
    "scripts/muon_q2_mc_reset_pilot.py",
    "scripts/submit_muon_q2_mc_reset_pilot.sh",
    "scripts/muon_q1_codebook_control.py",
    "scripts/muon_q1_mc_reset_pilot.py",
)


def baseline_context(path, root):
    baseline, source = reset.baseline_context(path, root)
    initialization.verify_pilot_completion(baseline, mc.ARM)
    rows, paths = history.read_history(path.parent, 16384)
    metadata = mc.verify_preprocessing(baseline)
    membership = json.loads((path.parent / mc.ARM / "mc_membership.json").read_text())
    if membership["mc_files"] != baseline["mc_paths"] or (
        membership["partitions"]["train"]["ordered_compact_indices_sha256"]
        != metadata["ordered_compact_train_indices_sha256"]
    ):
        raise ValueError("Q1 baseline membership and shared MC preprocessing disagree")
    return baseline, source, rows, paths, metadata, membership


def arm_config(source, output, arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown reset arm: {arm}")
    cfg = initialization.arm_config(source, output, arm, pilot_epochs=EPOCHS)
    cfg["model"].update(
        num_quantizers=NQ,
        codebook_size=K,
        data_codebook_init=True,
        dead_code_reset=arm == "reset_on",
    )
    cfg["datamodule"]["split_audit_path"] = str(output / arm / "mc_membership.json")
    if arm == "reset_on":
        cfg["callbacks"]["reset_action_audit"] = {"_target_": reset.AUDIT_TARGET}
    return cfg


def preflight():
    """Exercise both real RVQ layers, while keeping initialization/reset on Q0."""
    import torch
    from heptokens.models.vq_vae import LitVqVae

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = LitVqVae(
            encoder=lambda **kw: torch.nn.Identity(),
            decoder=lambda **kw: torch.nn.Identity(),
            codebook_size=K,
            codebook_dim=DIM,
            num_quantizers=NQ,
            data_codebook_init=True,
            data_codebook_init_samples=65536,
            dead_code_reset=True,
        )
        q0, q1 = model.vector_quantization.layers
        q1_buffers = tuple(t.clone() for t in (q1.embed, q1.embed_avg, q1.cluster_size))
        samples = torch.randn(65536, 1, DIM)
        before = torch.get_rng_state().clone()
        model._collect_data_codebook_init_samples(samples, {"mask": torch.ones(65536, 1).bool()})
        assert model._initialize_codebooks_from_data() == K, "Initialize Q0 only"
        assert torch.equal(before, torch.get_rng_state()), "Initialization advanced the RNG"
        assert q0.embed.shape == (DIM, K) and torch.equal(q0.embed, q0.embed_avg)
        assert torch.equal(q0.cluster_size, torch.ones(K)), "Stale initialization EMA counts"
        model._dead_code_usage[0, :2] = 10
        model._dead_code_reset_sample = samples[:4, 0]
        q0_before = q0.embed.clone()
        replaced, stats = model._reset_dead_codes()
        changed = (q0.embed != q0_before).any(dim=0)
        assert replaced == int(changed.sum()) == K // 4, "Wrong 25% reset cap"
        assert stats == {0: {"dead_before_reset": K - 2, "codes_reset": K // 4}}
        assert torch.equal(q0.embed[:, :2], q0_before[:, :2]), "Used entries were reset"
        assert torch.equal(q0.embed[:, changed], q0.embed_avg[:, changed])
        assert torch.equal(q0.cluster_size[changed], torch.ones(K // 4))
        assert all(
            torch.equal(old, actual)
            for old, actual in zip(q1_buffers, (q1.embed, q1.embed_avg, q1.cluster_size))
        ), "Initialization/reset must not write raw encoder latents into Q1"
        assert not bool(model._dead_code_usage.any()) and model._dead_code_reset_sample is None
        quantized, indices, loss = model.vector_quantization(samples[:4])
        assert indices.shape == (NQ, 4, 1), "Wrong backend [Q, B, objects] index layout"
        assert indices.permute(1, 2, 0).shape == (4, 1, NQ)
        assert torch.isfinite(quantized).all() and torch.isfinite(loss).all()
        assert all(torch.isfinite(layer.embed).all() for layer in (q0, q1))
    print("PASS: Q2/cb8192 initialization/reset/EMA preflight (CPU, no physics inference)")


def verify_plan(path):
    path = path.resolve()
    plan = json.loads(path.read_text())
    if (
        plan.get("version") != 1
        or plan.get("kind") != KIND
        or Path(plan["output"]) != path.parent
        or plan.get("pilot_epochs") != EPOCHS
        or plan.get("num_quantizers") != NQ
        or plan.get("codebook_size") != K
        or set(plan["configs"]) != set(ARMS)
    ):
        raise ValueError("Wrong, changed or moved Q2 pilot plan")
    baseline, source, _, _, metadata, membership = baseline_context(
        Path(plan["baseline_plan"]), Path(plan["root"])
    )
    for record in plan["inputs"] + plan["sources"] + list(plan["configs"].values()):
        initialization.verify_record(record)
    if plan["runtime"] != baseline["runtime"] or plan["runtime"] != initialization.runtime_record():
        raise ValueError("Q2/reference runtime differs")
    if plan["shared_preprocessing"] != metadata or plan["membership"] != membership:
        raise ValueError("Shared MC preprocessing/membership changed")
    for arm in ARMS:
        cfg = initialization.load_config(plan["configs"][arm]["path"])
        differences = initialization.diff_paths(arm_config(source, path.parent, arm), cfg)
        if differences:
            raise ValueError(f"Unexpected Q2 {arm} settings: {differences}")
    return plan


def describe(plan):
    cfg = initialization.load_config(plan["configs"]["reset_off"]["path"])
    dm = cfg["datamodule"]
    print("Muon Q2 MC-only reset comparison: TWO fresh three-epoch trainings")
    print(f"  Q1 MC reference: {Path(plan['baseline_plan']).parent}")
    print("  capacity changes only: Q1 -> Q2; K16384 -> K8192 per codebook; dim8 unchanged")
    print("  widths=[128,256,512]; batch=1024; seed=42; original losses/optimizer unchanged")
    print(f"  epochs={EPOCHS}; scheduler horizon={cfg['trainer']['max_epochs']}")
    print(f"  saved scheduler: {cfg['model'].get('scheduler')}")
    print(f"  saved MC files={len(dm['data_paths'])}; same train/val/test event membership")
    print(f"  same MC-training-only joblib: {initialization.joblib_paths(dm)}")
    features = next(c["inputs"] for c in dm["object_collections"] if c["object_name"] == "muons")
    print(f"  same ordered features: {features}")
    print("  both arms: Q0 data initialization after 65,536 latents; Q1 learns residuals normally")
    print("  reset OFF versus ON for Q0 only, exactly as in the Q1 study")
    print("  reset every 1000 steps; zero-use entries; cap=25% (2048 of 8192); pool limit=4096")
    print("  no joblib refit, data rescan, extra MC, resume or changes to completed Q1 runs")
    print("  same total codebook rows as Q1, but nominal code budget increases from 14 to 26 bits")
    print(f"  output: {plan['output']}")


def prepare(args):
    root, output = args.root.resolve(), args.output.resolve()
    baseline_path = args.baseline.resolve() / "plan.json"
    path = output / "plan.json"
    if path.exists():
        plan = verify_plan(path)
        if plan["root"] != str(root) or plan["baseline_plan"] != str(baseline_path):
            raise ValueError("Existing Q2 plan has different inputs")
        describe(plan)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output; nothing overwritten")
    baseline, source, _, history_paths, metadata, membership = baseline_context(baseline_path, root)
    original, _ = mc.baseline_context(Path(baseline["baseline_plan"]), root)
    for protected in (
        baseline_path.parent,
        Path(baseline["baseline_plan"]).parent,
        Path(original["reference"]).parent,
    ):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Keep Q2 output separate from completed reference trainings")
    preflight()
    (output / "configs").mkdir(parents=True)
    configs = {}
    for arm in ARMS:
        config_path = output / "configs" / f"{arm}.yaml"
        OmegaConf.save(OmegaConf.create(arm_config(source, output, arm)), config_path)
        configs[arm] = initialization.file_record(config_path)
    spec = baseline["preprocessing"]
    inputs = (
        baseline_path,
        Path(spec["receipt"]),
        Path(spec["joblib"]),
        Path(spec["metadata"]),
        baseline_path.parent / mc.ARM / "SUCCESS.txt",
        baseline_path.parent / mc.ARM / "pilot_completion.json",
        baseline_path.parent / mc.ARM / "mc_membership.json",
        *history_paths,
    )
    plan = {
        "version": 1,
        "kind": KIND,
        "root": str(root),
        "output": str(output),
        "baseline_plan": str(baseline_path),
        "pilot_epochs": EPOCHS,
        "num_quantizers": NQ,
        "codebook_size": K,
        "runtime": baseline["runtime"],
        "inputs": [initialization.file_record(p) for p in inputs],
        "sources": [initialization.file_record(root / p) for p in SOURCE_FILES],
        "configs": configs,
        "shared_preprocessing": metadata,
        "membership": membership,
        "source_config_changes": {
            arm: initialization.diff_paths(source, arm_config(source, output, arm)) for arm in ARMS
        },
        "notes": [
            "Capacity variant changes both depth and K, not a depth-only ablation; 26 versus 14 nominal code bits.",
            "Q0-only data initialization/reset is inherited. Q1 uses the existing backend residual/EMA training.",
            "Same 25% reset cap: 2048 entries for K8192, not the old absolute cap of 4096.",
            "MC training-fitted joblib and exact saved membership reused, without loading H5 during preparation.",
            "Same seed does not guarantee the same shuffled training order after capacity/reset RNG changes.",
            "Compare matched validation input hashes/steps and both codebooks; three epochs are a pilot only.",
        ],
    }
    initialization.write_json(path, plan)
    describe(plan)
    print(f"Frozen plan: {path}; no training submitted")


def read_history(output, arm):
    run = output / arm
    if not (run / "SUCCESS.txt").is_file():
        raise ValueError(f"Training has not completed: {run}")
    rows = {}
    for path in sorted((run / "validation_usage").glob("*.json")):
        row = json.loads(path.read_text())
        key = (row["epoch"], row["step"])
        if key in rows:
            raise ValueError(f"Duplicate validation epoch/step: {key}")
        if len(row["counts"]) != NQ or len(row["codebook_metrics"]) != NQ:
            raise ValueError(f"Wrong Q2 capacity in {path}")
        for q, (counts, usage) in enumerate(zip(row["counts"], row["codebook_metrics"])):
            if len(counts) != K or any(type(n) is not int or n < 0 for n in counts):
                raise ValueError(f"Wrong codebook/counts in {path}")
            objects, used = sum(counts), sum(n > 0 for n in counts)
            if objects <= 0:
                raise ValueError(f"No validation objects in {path}")
            perplexity = math.exp(-sum((n / objects) * math.log(n / objects) for n in counts if n))
            expected = (q, objects, used, used / K, perplexity, perplexity / K)
            names = (
                "q",
                "objects",
                "used_codes",
                "used_fraction",
                "perplexity",
                "normalized_perplexity",
            )
            if any(not math.isclose(usage[name], value) for name, value in zip(names, expected)):
                raise ValueError(f"Inconsistent Q{q} usage audit in {path}")
        if row["codebook_metrics"][0]["objects"] != row["codebook_metrics"][1]["objects"]:
            raise ValueError(f"Different object counts between Q0/Q1 in {path}")
        if not row.get("data_codebook_init") or any(
            not math.isfinite(row["validation_metrics"][name])
            for name in ("val/total_loss", "val/recon_loss")
        ):
            raise ValueError(f"Invalid initialization/loss audit in {path}")
        rows[key] = row
    if not rows or not rows[max(rows)]["initialization_completed"]:
        raise ValueError(f"Missing history or unfinished data initialization: {run}")
    return rows


def summarize(args):
    output = args.output.resolve()
    plan = verify_plan(output / "plan.json")
    _, _, reference, _, _, membership = baseline_context(
        Path(plan["baseline_plan"]), Path(plan["root"])
    )
    histories = {}
    for arm in ARMS:
        if not (output / arm / "SUCCESS.txt").is_file():
            print(
                f"Matched Q2 comparison pending {arm} completion; run summarize after both finish"
            )
            return
        initialization.verify_pilot_completion(plan, arm)
        actual = json.loads((output / arm / "mc_membership.json").read_text())
        for key in ("mc_files", "seed", "split_fractions", "partitions"):
            if actual[key] != membership[key]:
                raise ValueError(f"Different MC {key} for {arm}")
        histories[arm] = read_history(output, arm)
    actions = json.loads((output / "reset_on/reset_actions.json").read_text())["actions"]
    if not actions or not any(r["codes_reset"] > 0 for r in actions):
        raise ValueError("No entries actually reset in the Q2 intervention")
    for action in actions:
        if action["step"] <= 0 or action["step"] % 1000 or not 0 <= action["codes_reset"] <= K // 4:
            raise ValueError("Reset actions do not follow the inherited Q0 policy")
    if set(reference) != set(histories["reset_off"]) or set(reference) != set(
        histories["reset_on"]
    ):
        raise ValueError("Different validation epoch/step sets between Q1 and Q2 arms")
    pairs = []
    compact = lambda row: {k: v for k, v in row.items() if k != "counts"}
    for key in sorted(reference):
        rows = [reference[key], *(histories[arm][key] for arm in ARMS)]
        for row in rows[1:]:
            for field in ("input_tensor_sha256", "validation_batches"):
                if row[field] != rows[0][field]:
                    raise ValueError(f"Different ordered validation {field} at {key}")
            if row["codebook_metrics"][0]["objects"] != rows[0]["codebook_metrics"][0]["objects"]:
                raise ValueError(f"Different validation object counts at {key}")
        if all(row["initialization_completed"] for row in rows):
            pairs.append(
                {
                    "epoch": key[0],
                    "step": key[1],
                    **{arm: compact(histories[arm][key]) for arm in ARMS},
                }
            )
    if not pairs:
        raise ValueError("No matched post-initialization validation passes")
    best_off = min(pairs, key=lambda row: row["reset_off"]["validation_metrics"]["val/total_loss"])
    best_on = min(pairs, key=lambda row: row["reset_on"]["validation_metrics"]["val/total_loss"])
    initialization.write_json(
        output / "comparison.json",
        {
            "paired_validation_passes": pairs,
            "at_reference_best": best_off,
            "best_reset_on": best_on,
            "final_matched_pass": pairs[-1],
            "reset_actions": actions,
            "q1_validation_inputs_verified": True,
            "note": "Q2/K8192 per stage; same MC preprocessing/validation as Q1. Q0-only init/reset. "
            "Best is selected by total loss, not necessarily reconstruction loss. Three-epoch pilot only.",
        },
    )
    print(f"Verified {len(pairs)} matched Q1/Q2 validation inputs: {output / 'comparison.json'}")
    for arm in ARMS:
        row = best_off[arm]
        print(
            f"At reset-OFF best step {best_off['step']}: {arm} recon={row['validation_metrics']['val/recon_loss']:.6g}"
        )
        for usage in row["codebook_metrics"]:
            print(f"  {usage}")


def run(args):
    import fcntl
    import torch

    output = args.output.resolve()
    arms = ARMS if args.arm == "both" else (args.arm,)
    # Per-arm locks also protect separately submitted GPU0/GPU3 workers.
    from contextlib import ExitStack

    with ExitStack() as stack:
        for arm in arms:
            lock = stack.enter_context((output / f"{arm}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify_plan(output / "plan.json")
        for arm in arms:
            dest = output / arm
            if (dest.exists() and any(dest.iterdir())) or (output / f"{arm}.log").exists():
                raise ValueError(f"Refusing to overwrite/resume {arm}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU using CUDA_VISIBLE_DEVICES")
        preflight()
        for arm in arms:
            verify_plan(output / "plan.json")
            initialization.write_json(
                output / f"execution_{arm}.json",
                {
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu": str(torch.cuda.get_device_properties(0)),
                    "torch_cuda": torch.version.cuda,
                },
            )
            cfg_path = Path(plan["configs"][arm]["path"])
            cfg = initialization.load_config(cfg_path)
            env = dict(os.environ)
            env["PYTHONPATH"] = (
                str(Path(plan["root"]) / "src") + os.pathsep + env.get("PYTHONPATH", "")
            )
            env["WANDB_MODE"] = "offline" if cfg["logger"].get("offline") else "online"
            cmd = [
                sys.executable,
                str(Path(plan["root"]) / "scripts/train.py"),
                "--config-path",
                str(cfg_path.parent),
                "--config-name",
                arm,
            ]
            print(f"START Q2 {arm}: {' '.join(cmd)}", flush=True)
            with (output / f"{arm}.log").open("x") as log:
                subprocess.run(
                    cmd, cwd=plan["root"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            if not (output / arm / "SUCCESS.txt").is_file():
                raise RuntimeError(f"{arm} exited without SUCCESS.txt")
            initialization.verify_pilot_completion(plan, arm)
            print(f"FINISHED Q2 {arm}", flush=True)
        summarize(args)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=root / "results/atlas_muon_q1_mc_only_trainfit_pilot_e3_v1"
    )
    parser.add_argument("--arm", choices=("both", *ARMS), default="both")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
