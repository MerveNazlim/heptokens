#!/usr/bin/env python3
"""Freeze and run two fresh, matched muon Q1 trainings from a saved full config."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

ARMS = ("control", "data_init")
AUDIT_TARGET = "heptokens.callbacks.tokenizer_usage.TokenizerUsageAudit"
SOURCE_FILES = (
    "scripts/muon_q1_init_control.py",
    "scripts/train.py",
    "src/heptokens/models/vq_vae.py",
    "src/heptokens/models/coders.py",
    "src/heptokens/models/utils.py",
    "src/heptokens/data/atlas_event_mappable.py",
    "src/heptokens/data/atlas_mappable.py",
    "src/heptokens/data/collation.py",
    "src/heptokens/utils/hydra.py",
    "src/heptokens/callbacks/tokenizer_usage.py",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path, *, hash_content=True):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    result = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_content:
        result["sha256"] = digest(path)
    return result


def verify_record(record):
    actual = file_record(record["path"], hash_content="sha256" in record)
    if actual != record:
        raise ValueError(f"Input changed after preparation: {record['path']}")


def runtime_record():
    import vector_quantize_pytorch

    package = Path(vector_quantize_pytorch.__file__).resolve().parent
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "lightning",
                "vector-quantize-pytorch",
                "numpy",
                "scikit-learn",
                "h5py",
                "joblib",
                "hydra-core",
                "omegaconf",
            )
        },
        "quantizer_source": {str(p): digest(p) for p in sorted(package.rglob("*.py"))},
    }


def load_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True, throw_on_missing=True)


def validate_reference(cfg):
    model, dm, trainer = cfg["model"], cfg["datamodule"], cfg["trainer"]
    if model["_target_"] != "heptokens.models.vq_vae.LitVqVae":
        raise ValueError("Expected the saved LitVqVae configuration")
    if (model["num_quantizers"], model["codebook_size"], model["codebook_dim"]) != (1, 16384, 8):
        raise ValueError("Reference must be full Q1, cb16384, dim8, not a Q8 prefix")
    if dm.get("object_type") != "muons" or dm.get("output_mode") != "object":
        raise ValueError("This experiment is muon-only")
    if dm["_target_"] != "heptokens.data.atlas_event_mappable.AtlasEventMapModule":
        raise ValueError("Expected the original eager global-split datamodule")
    if not dm.get("data_paths") or len(dm["data_paths"]) != len(set(dm["data_paths"])):
        raise ValueError("Reference needs a nonempty, unique, ordered input file list")
    if any(not Path(p).is_absolute() for p in dm["data_paths"]):
        raise ValueError("Reference H5 paths must be absolute; no path guessing")
    if model.get("data_codebook_init", False) or model.get("dead_code_reset", False):
        raise ValueError("Reference must have data initialization and dead-code reset disabled")
    if model.get("data_codebook_init_quantizers") not in (None, [0]):
        raise ValueError("Initialization must select Q0")
    if cfg.get("full_resume") or cfg.get("ckpt_path") or cfg.get("weight_ckpt_path"):
        raise ValueError("Reference must describe fresh training, not a resumed job")
    if cfg.get("compile"):
        raise ValueError("Compiled training is not supported by this control launcher")
    if trainer.get("devices") not in (1, [0]) or trainer.get("num_nodes", 1) != 1:
        raise ValueError("Both arms must use one device and one node")
    if trainer.get("accelerator") not in ("gpu", "cuda"):
        raise ValueError("Expected the saved single-GPU trainer configuration")
    if trainer.get("fast_dev_run"):
        raise ValueError("Reference is a smoke run")
    if int(model.get("data_codebook_init_samples", 65536)) != 65536:
        raise ValueError("Expected the agreed 65,536-latent initialization pool")
    for side in ("encoder", "decoder"):
        if model[side]["model"]["hidden_dims"] != [128, 256, 512]:
            raise ValueError(f"Unexpected {side} widths; review rather than silently change them")
    if trainer["max_epochs"] != 20 or dm["batch_size"] != 1024:
        raise ValueError("Expected the full 20-epoch, batch-1024 source run")


def joblib_paths(value):
    paths = []
    if isinstance(value, dict):
        if value.get("_target_") == "joblib.load":
            paths.append(value["filename"])
        for child in value.values():
            paths.extend(joblib_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.extend(joblib_paths(child))
    return paths


def arm_config(source, output, arm, pilot_epochs=0):
    """Change output identity and the explicit intervention; retain saved settings."""
    horizon = source["trainer"]["max_epochs"]
    if type(pilot_epochs) is not int or not 0 <= pilot_epochs < horizon:
        raise ValueError(f"pilot_epochs must be 0 (full run) or between 1 and {horizon - 1}")
    if pilot_epochs and (
        (source["trainer"].get("min_epochs") or 0) > pilot_epochs
        or (source["trainer"].get("min_steps") or 0) > 0
        or source["trainer"].get("max_steps", -1) not in (None, -1)
    ):
        raise ValueError("Trainer epoch/step constraints conflict with an exact epoch pilot")
    cfg = copy.deepcopy(source)
    run = output / arm
    cfg.update(
        project_name=output.name,
        network_name=arm,
        output_dir=str(output.parent),
        full_path=str(run),
        full_resume=False,
        ckpt_path=None,
        weight_ckpt_path=None,
    )
    cfg["trainer"]["default_root_dir"] = str(run)
    for callback in cfg["callbacks"].values():
        if isinstance(callback, dict) and callback.get("_target_", "").endswith("ModelCheckpoint"):
            callback["dirpath"] = str(run / "checkpoints")
    cfg["callbacks"]["tokenizer_usage_audit"] = {"_target_": AUDIT_TARGET}
    if pilot_epochs:
        cfg["callbacks"]["tokenizer_usage_audit"]["stop_after_epochs"] = pilot_epochs
    logger = cfg["logger"]
    if "WandbLogger" not in logger.get("_target_", ""):
        raise ValueError("Expected the saved WandB logger; refusing to guess its output paths")
    logger.update(save_dir=str(run), project=output.name, name=arm, id=None, resume=False)
    cfg["model"]["data_codebook_init"] = arm == "data_init"
    cfg["model"]["data_codebook_init_samples"] = 65536
    cfg["hydra"] = {"run": {"dir": str(run)}, "job": {"chdir": True}}
    return cfg


def diff_paths(one, two, prefix=""):
    if isinstance(one, dict) and isinstance(two, dict):
        result = []
        for key in sorted(set(one) | set(two)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in one or key not in two:
                result.append(path)
            else:
                result.extend(diff_paths(one[key], two[key], path))
        return result
    return [] if one == two else [prefix]


def verify_arms(configs):
    normalized = []
    for arm in ARMS:
        cfg = copy.deepcopy(configs[arm])
        cfg["network_name"] = "ARM"
        cfg["full_path"] = "RUN"
        cfg["trainer"]["default_root_dir"] = "RUN"
        cfg["logger"]["name"] = "ARM"
        cfg["logger"]["save_dir"] = "RUN"
        cfg["hydra"]["run"]["dir"] = "RUN"
        for cb in cfg["callbacks"].values():
            if isinstance(cb, dict) and cb.get("_target_", "").endswith("ModelCheckpoint"):
                cb["dirpath"] = "CHECKPOINTS"
        normalized.append(cfg)
    differences = diff_paths(*normalized)
    if differences != ["model.data_codebook_init"]:
        raise ValueError(f"Unexpected arm differences: {differences}")


def preflight():
    """Exercise real quantizer EMA buffers on CPU before allocating the H5 dataset."""
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
            data_codebook_init=True,
        )
        samples = torch.randn(65536, 1, 8)
        before = torch.get_rng_state().clone()
        model._collect_data_codebook_init_samples(samples, {"mask": torch.ones(65536, 1).bool()})
        assert model._initialize_codebooks_from_data() == 16384
        assert torch.equal(before, torch.get_rng_state()), "Initialization consumed the global RNG"
        layer = model.vector_quantization.layers[0]
        # The pinned backend uses transposed EMA sums. Fail if its contract changes.
        assert layer.embed.shape == (8, 16384), "Unexpected quantizer backend layout"
        assert torch.equal(layer.embed, layer.embed_avg), "Stale EMA embedding buffer"
        assert torch.equal(layer.cluster_size, torch.ones(16384)), "Stale EMA counts"
        quantized, indices, loss = model.vector_quantization(samples[:4])
        assert indices.shape == (1, 4, 1), "Unexpected quantizer index layout"
        assert torch.isfinite(quantized).all() and torch.isfinite(loss).all()
        assert torch.isfinite(layer.embed).all(), "Nonfinite embeddings after first EMA update"
    print("PASS: real-backend initialization/EMA/RNG preflight (CPU, no physics inference)")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_plan(path):
    plan = json.loads(path.read_text())
    if Path(plan["output"]) != path.parent.resolve():
        raise ValueError("Plan was moved; use its original output directory")
    for record in plan["inputs"] + plan["sources"] + list(plan["configs"].values()):
        verify_record(record)
    if runtime_record() != plan["runtime"]:
        raise ValueError("Runtime changed after preparation; do not mix experiment arms")
    configs = {arm: load_config(plan["configs"][arm]["path"]) for arm in ARMS}
    verify_arms(configs)
    for cfg in configs.values():
        stop = cfg["callbacks"]["tokenizer_usage_audit"].get("stop_after_epochs", 0)
        if stop != plan.get("pilot_epochs", 0) or cfg["trainer"]["max_epochs"] != 20:
            raise ValueError("Frozen pilot budget/schedule mismatch")
    return plan


def verify_pilot_completion(plan, arm):
    pilot_epochs = plan.get("pilot_epochs", 0)
    if not pilot_epochs:
        return
    run = Path(plan["output"]) / arm
    receipt = json.loads((run / "pilot_completion.json").read_text())
    if (
        receipt["completed_epochs"] != pilot_epochs
        or receipt["stop_after_epochs"] != pilot_epochs
        or receipt["schedule_max_epochs"] != 20
        or Path(receipt["checkpoint"]) != run / "checkpoints/pilot_end.ckpt"
        or not (run / "checkpoints/pilot_end.ckpt").is_file()
    ):
        raise ValueError(f"Incomplete or inconsistent pilot for {arm}")


def describe(plan):
    cfg = load_config(plan["configs"]["control"]["path"])
    dm = cfg["datamodule"]
    features = next(c["inputs"] for c in dm["object_collections"] if c["object_name"] == "muons")
    print("Muon Q1 initialization control: TWO fresh trainings, sequential on the same GPU")
    print(f"  reference: {plan['reference']}")
    print("  Q=1; K=16384; dim=8; encoder/decoder=[128,256,512]")
    epochs_to_run = plan.get("pilot_epochs") or cfg["trainer"]["max_epochs"]
    print(f"  epochs to run={epochs_to_run}; scheduler horizon={cfg['trainer']['max_epochs']}")
    print(f"  batch={dm['batch_size']}; seed={cfg['seed']}")
    print(f"  saved input files: {len(dm['data_paths'])}; no rescan or joblib refit")
    print(
        f"  split: train={dm.get('train_frac')}, val={dm.get('val_frac')}, "
        f"test={dm.get('test_frac')}; split seed={dm.get('seed')}"
    )
    print(f"  features (saved order): {features}")
    print(f"  joblibs: {joblib_paths(dm)}")
    print(f"  validation limit: {cfg['trainer'].get('limit_val_batches', 1.0)} batches/fraction")
    print("  control: data_codebook_init=false; data_init: true after 65,536 encoder outputs")
    print("  random sampled latents, NOT k-means; dead-code reset OFF in both")
    print(f"  feature loss weights: {cfg['model'].get('feature_loss_weights')}")
    print("  all optimizer/loss/split/preprocessing settings inherited from saved config")
    print(f"  output: {plan['output']}")


def prepare(args):
    output, root, reference = args.output.resolve(), args.root.resolve(), args.reference.resolve()
    pilot_epochs = getattr(args, "pilot_epochs", 0)
    path = output / "plan.json"
    if path.exists():
        plan = verify_plan(path)
        if plan["reference"] != str(reference) or plan["root"] != str(root):
            raise ValueError("Existing experiment uses a different reference/root")
        if plan.get("pilot_epochs", 0) != pilot_epochs:
            raise ValueError(
                "Existing experiment has a different epoch budget; use a separate output"
            )
        describe(plan)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty without a valid plan; nothing overwritten")
    source = load_config(reference)
    validate_reference(source)
    if output == reference.parent or reference.parent in output.parents:
        raise ValueError("Keep experimental outputs separate from the production run")
    joblibs = joblib_paths(source["datamodule"])
    if not joblibs or any(not Path(p).is_absolute() for p in joblibs):
        raise ValueError("Saved config needs explicit absolute joblib paths")
    inputs = [file_record(reference), file_record(reference.parent / "SUCCESS.txt")]
    for p in joblibs:
        inputs.append(file_record(p))
        metadata = Path(p).with_suffix(".json")
        if metadata.exists():
            inputs.append(file_record(metadata))
    inputs += [file_record(p, hash_content=False) for p in source["datamodule"]["data_paths"]]
    sources = [file_record(root / p) for p in SOURCE_FILES]
    configs = {arm: arm_config(source, output, arm, pilot_epochs) for arm in ARMS}
    verify_arms(configs)
    preflight()
    (output / "configs").mkdir(parents=True)
    config_records = {}
    for arm, cfg in configs.items():
        config_path = output / "configs" / f"{arm}.yaml"
        OmegaConf.save(OmegaConf.create(cfg), config_path)
        config_records[arm] = file_record(config_path)
    plan = {
        "version": 2,
        "root": str(root),
        "output": str(output),
        "reference": str(reference),
        "pilot_epochs": pilot_epochs,
        "inputs": inputs,
        "sources": sources,
        "runtime": runtime_record(),
        "configs": config_records,
        "intervention": "model.data_codebook_init: false -> true",
        "common_instrumentation": AUDIT_TARGET,
        "source_config_changes": {arm: diff_paths(source, cfg) for arm, cfg in configs.items()},
        "notes": [
            "Saved H5 order/splits/features/joblibs retained. H5 identity checked by path/size/mtime, not content hash.",
            "Two fresh trainings under current frozen code/runtime, not exact historical-code reproduction.",
            "Preprocessing is inherited unchanged; this is not a new file-disjoint holdout study.",
            "Validation usage covers the configured validation pass; never select using test performance.",
        ],
    }
    write_json(path, plan)
    describe(plan)
    print(f"Frozen plan: {path}; no training submitted")


def run(args):
    import fcntl
    import torch

    path = args.output.resolve() / "plan.json"
    with (path.parent / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify_plan(path)
        for arm in ARMS:
            dest = path.parent / arm
            if dest.exists() and any(dest.iterdir()):
                raise ValueError(f"Refusing to overwrite/resume existing arm: {dest}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
        preflight()
        write_json(
            path.parent / "execution.json",
            {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": str(torch.cuda.get_device_properties(0)),
                "torch_cuda": torch.version.cuda,
            },
        )
        for arm in ARMS:
            verify_plan(path)
            cfg_path = Path(plan["configs"][arm]["path"])
            cmd = [
                sys.executable,
                str(Path(plan["root"]) / "scripts/train.py"),
                "--config-path",
                str(cfg_path.parent),
                "--config-name",
                arm,
            ]
            print(f"START {arm}: {' '.join(cmd)}", flush=True)
            env = dict(os.environ)
            env["PYTHONPATH"] = (
                str(Path(plan["root"]) / "src") + os.pathsep + env.get("PYTHONPATH", "")
            )
            cfg = load_config(cfg_path)
            env["WANDB_MODE"] = "offline" if cfg["logger"].get("offline") else "online"
            with (path.parent / f"{arm}.log").open("x") as log:
                subprocess.run(
                    cmd, cwd=plan["root"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            if not (path.parent / arm / "SUCCESS.txt").is_file():
                raise RuntimeError(f"{arm} exited without SUCCESS.txt")
            verify_pilot_completion(plan, arm)
            print(f"FINISHED {arm}", flush=True)
        summarize(args)


def summarize(args):
    output = args.output.resolve()
    plan = verify_plan(output / "plan.json")
    records = {}
    for arm in ARMS:
        if not (output / arm / "SUCCESS.txt").is_file():
            raise ValueError(f"Training has not completed: {arm}")
        verify_pilot_completion(plan, arm)
        records[arm] = {}
        for path in sorted((output / arm / "validation_usage").glob("*.json")):
            row = json.loads(path.read_text())
            records[arm][(row["epoch"], row["step"])] = row
    if not records["control"] or records["control"].keys() != records["data_init"].keys():
        raise ValueError("Need completed matching validation histories from BOTH arms")
    if not list(records["data_init"].values())[-1]["initialization_completed"]:
        raise ValueError("Treatment did not initialize its codebook")
    paired = []
    for key, control in records["control"].items():
        treatment = records["data_init"][key]
        if control["input_tensor_sha256"] != treatment["input_tensor_sha256"]:
            raise ValueError(f"Different ordered validation inputs at epoch/step {key}")
        paired.append(
            {
                "epoch": key[0],
                "step": key[1],
                **{
                    arm: {k: v for k, v in records[arm][key].items() if k != "counts"}
                    for arm in ARMS
                },
            }
        )
    write_json(
        output / "comparison.json",
        {
            "reference": plan["reference"],
            "pilot_epochs": plan.get("pilot_epochs", 0),
            "paired_validation_passes": paired,
            "note": "Higher occupancy alone is not success. Compare val/recon_loss and all feature errors; "
            "then use matched validation objects for physical-unit residual/binned-pT plots.",
        },
    )
    print(
        f"Verified {len(paired)} matched validation passes. Summary: {output / 'comparison.json'}"
    )
    for arm in ARMS:
        best = min(records[arm].values(), key=lambda r: r["validation_metrics"]["val/total_loss"])
        print(
            f"{arm}: best val/total_loss at step {best['step']}; "
            f"recon={best['validation_metrics']['val/recon_loss']:.6g}; "
            f"usage={best['codebook_metrics'][0]}"
        )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pilot-epochs",
        type=int,
        default=0,
        help="Preparation only: stop both arms after N epochs, keeping the 20-epoch schedule; 0 runs all 20.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root
        / "results"
        / "atlas_object_final_tokenizers_new_mcdata"
        / "muons_full_dim8_cb16384_q1_e20_new_mcdata/full_config.yaml",
    )
    args = parser.parse_args()
    if args.command != "prepare" and args.pilot_epochs:
        parser.error("Set --pilot-epochs during prepare; run/summarize use the frozen plan budget")
    {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
