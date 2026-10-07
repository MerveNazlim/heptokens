#!/usr/bin/env python3
"""Plot Q2/K8192 reset OFF/ON on identical saved MC validation muons."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

import muon_q1_init_control as initialization
import muon_q2_mc_reset_pilot as pilot
import plot_muon_q1_mc_reset as shared

log = logging.getLogger(__name__)
ARMS, K, NQ = pilot.ARMS, pilot.K, pilot.NQ
CHECKPOINT = "pilot_end.ckpt"


def audit(args):
    root = args.reset_dir.resolve()
    plan = pilot.verify_plan(root / "plan.json")
    pilot.summarize(argparse.Namespace(output=root))
    baseline = pilot.mc.verify_plan(Path(plan["baseline_plan"]))
    fit = pilot.mc.verify_preprocessing(baseline)
    records = [initialization.file_record(root / "plan.json")]
    configs = {}
    for arm in ARMS:
        initialization.verify_pilot_completion(plan, arm)
        configs[arm] = initialization.load_config(plan["configs"][arm]["path"])
        paths = (
            Path(plan["configs"][arm]["path"]),
            root / arm / "SUCCESS.txt",
            root / arm / "pilot_completion.json",
            root / arm / "checkpoints" / CHECKPOINT,
            root / arm / "mc_membership.json",
            *sorted((root / arm / "validation_usage").glob("*.json")),
        )
        records.extend(initialization.file_record(path) for path in paths)
    records.append(initialization.file_record(root / "reset_on/reset_actions.json"))
    if initialization.diff_paths(configs["reset_off"]["model"], configs["reset_on"]["model"]) != [
        "dead_code_reset"
    ]:
        raise ValueError("Q2 model settings differ beyond reset OFF/ON")
    if initialization.diff_paths(
        configs["reset_off"]["datamodule"], configs["reset_on"]["datamodule"]
    ) != ["split_audit_path"]:
        raise ValueError("Q2 saved input preparation differs between arms")
    for key in ("joblib", "metadata", "receipt"):
        records.append(initialization.file_record(Path(baseline["preprocessing"][key])))
    if not args.input_momentum_unit:
        raise ValueError("Declare --input-momentum-unit GeV or MeV; no automatic unit guessing")
    return {
        "version": "muon-q2-cb8192-mc-reset-validation-end-e3-v1",
        "sources": records,
        "roots": {arm: str(root / arm) for arm in ARMS},
        "max_valid_objects": args.max_valid_objects,
        "batch_size": args.batch_size,
        "input_momentum_unit": args.input_momentum_unit,
        "checkpoint": CHECKPOINT,
        "num_quantizers": NQ,
        "codebook_size": K,
        "preprocessing": fit,
        "comparison": json.loads((root / "comparison.json").read_text()),
        "sample": "First N valid objects in original MC validation permutation order; no resplit",
        "note": "Identical saved MC validation membership and shared MC-training-fitted joblib. "
        "Q2/K8192 per stage, Q0-only initialization/reset; three-epoch pilot endpoints. "
        "Not file-disjoint holdout or independent final test evaluation.",
    }, configs


def validate_pairs(original, items, names):
    if original.ndim != 2 or original.shape[1] != len(names) or not len(original):
        raise ValueError("Empty or invalid original-object arrays")
    for arm in ARMS:
        x, y, codes = items[arm]
        if not np.array_equal(x, original, equal_nan=True) or y.shape != original.shape:
            raise ValueError(f"Original objects/order differ for {arm}")
        if (
            codes.shape != (len(original), NQ)
            or codes.dtype.kind not in "iu"
            or np.any((codes < 0) | (codes >= K))
        ):
            raise ValueError(f"Expected valid Q2/cb8192 indices for {arm}")


def read_pairs(out):
    receipt = json.loads((out / "paired_muons.json").read_text())
    for key in ("arrays", "evaluation", "membership"):
        initialization.verify_record(receipt[key])
    manifest = json.loads((out / "evaluation.json").read_text())
    if manifest.get("version") != "muon-q2-cb8192-mc-reset-validation-end-e3-v1" or (
        manifest.get("num_quantizers"),
        manifest.get("codebook_size"),
        manifest.get("checkpoint"),
    ) != (NQ, K, CHECKPOINT):
        raise ValueError("Not a Q2/K8192 endpoint cache; never reuse a Q1 cache")
    for record in manifest["sources"]:
        initialization.verify_record(record)
    with np.load(out / "paired_muons.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate_pairs(
        arrays["original"],
        {arm: (arrays["original"], arrays[arm], arrays[f"{arm}_indices"]) for arm in ARMS},
        arrays["feature_names"].tolist(),
    )
    if len(arrays["original"]) != receipt["objects"]:
        raise ValueError("Cache object count differs from receipt")
    return manifest, arrays


def evaluate(args):
    if args.max_valid_objects <= 0 or args.batch_size <= 0:
        raise ValueError("Object cap and evaluation batch size must be positive")
    manifest, configs = audit(args)
    out = args.output_dir.resolve()
    roots = [Path(path) for path in manifest["roots"].values()]
    if any(out == root or out in root.parents or root in out.parents for root in roots) or (
        out == args.reset_dir.resolve() or out in args.reset_dir.resolve().parents
    ):
        raise ValueError("Keep plotting output below figures/, separate from training arms")
    if any(
        out == Path(record["path"]) or out in Path(record["path"]).parents
        for record in manifest["sources"]
    ):
        raise ValueError("Plotting output overlaps audited inputs")
    manifest_path = out / "evaluation.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Evaluation settings or inputs changed; existing cache not overwritten"
            )
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Nonempty output without an evaluation audit; nothing overwritten")
    else:
        shared.reco.write_json(manifest_path, manifest)
    arrays_path = out / "paired_muons.npz"
    if arrays_path.exists() or arrays_path.with_suffix(".json").exists():
        read_pairs(out)
        log.info("Reusing paired Q2 cache; no inference")
        return
    import torch

    cfg = OmegaConf.create(configs["reset_off"])
    _, inverse = shared.diagnostics.transform_list_and_cst_fn_from_cfg(cfg)
    if inverse is None:
        raise ValueError("Missing inverse MC preprocessor")
    cfg.datamodule.split_audit_path = str(out / "evaluation_membership.json")
    cfg.datamodule.batch_size = args.batch_size
    cfg.datamodule.num_workers = 0
    cfg.datamodule.pin_memory = False
    if "persistent_workers" in cfg.datamodule:
        cfg.datamodule.persistent_workers = False
    if "multiprocessing_context" in cfg.datamodule:
        cfg.datamodule.multiprocessing_context = None
    log.info("Loading saved MC datamodule once; exact original global split, no rescan/refit")
    dm = hydra.utils.instantiate(cfg.datamodule)
    actual = json.loads((out / "evaluation_membership.json").read_text())
    for root in roots:
        shared.check_membership(actual, json.loads((root / "mc_membership.json").read_text()))
    loader = shared.diagnostics.dataloader_from_datamodule(dm, "val")
    names = [
        Path(path).name
        for path in next(
            c for c in cfg.datamodule.object_collections if c.object_name == "muons"
        ).inputs
    ]
    device = shared.diagnostics.choose_device(args.device)
    items = {}
    for arm in ARMS:
        run = Path(manifest["roots"][arm])
        checkpoint = run / "checkpoints" / CHECKPOINT
        log.info("Decoding Q2 %s endpoint on the same MC validation loader", shared.LABELS[arm])
        model = shared.diagnostics.load_analysis_model(run, str(checkpoint), device)
        if tuple(
            int(getattr(model.hparams, key))
            for key in ("num_quantizers", "codebook_size", "codebook_dim")
        ) != (NQ, K, pilot.DIM) or bool(model.hparams.dead_code_reset) != (arm == "reset_on"):
            raise ValueError(f"Wrong capacity/reset setting in checkpoint: {checkpoint}")
        if (
            not model.hparams.data_codebook_init
            or model.hparams.data_codebook_init_quantizers not in (None, [0])
            or model.hparams.dead_code_reset_quantizers not in (None, [0])
        ):
            raise ValueError(f"Wrong Q0-only initialization/reset policy: {checkpoint}")
        x, y, indices, _ = shared.diagnostics.collect_diagnostics_from_loader(
            model=model,
            loader=loader,
            cst_inverse_transformer=inverse,
            device=device,
            max_valid_objects=args.max_valid_objects,
        )
        items[arm] = (x, y, indices)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    original = items["reset_off"][0]
    validate_pairs(original, items, names)
    temporary = arrays_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        original=original,
        feature_names=np.asarray(names),
        **{arm: items[arm][1] for arm in ARMS},
        **{f"{arm}_indices": items[arm][2] for arm in ARMS},
    )
    temporary.replace(arrays_path)
    shared.reco.write_json(
        arrays_path.with_suffix(".json"),
        {
            "arrays": initialization.file_record(arrays_path),
            "evaluation": initialization.file_record(manifest_path),
            "membership": initialization.file_record(out / "evaluation_membership.json"),
            "objects": len(original),
        },
    )
    log.info(
        "VERIFIED: %s identical paired Q2 MC validation muons; cache=%s",
        f"{len(original):,}",
        arrays_path,
    )


def summaries(manifest, arrays, bins, min_count):
    validate_pairs(
        arrays["original"],
        {arm: (arrays["original"], arrays[arm], arrays[f"{arm}_indices"]) for arm in ARMS},
        arrays["feature_names"].tolist(),
    )
    summary = shared.feature_summaries(manifest, arrays, bins, min_count)
    usage = {}
    for arm in ARMS:
        indices = arrays[f"{arm}_indices"]
        counts = shared.diagnostics.codebook_counts(indices, K)
        stages = []
        for q in range(NQ):
            c = counts[q]
            p = c[c > 0] / c.sum()
            perplexity = float(np.exp(-np.sum(p * np.log(p))))
            stages.append(
                {
                    "q": q,
                    **shared.diagnostics.codebook_summary(c[None])["quantizer_0"],
                    "perplexity": perplexity,
                    "normalized_perplexity": perplexity / K,
                    "top10_percent": float(100 * np.sort(c)[-10:].sum() / c.sum()),
                    "sorted_counts": np.sort(c)[::-1].tolist(),
                }
            )
        pairs, pair_counts = np.unique(indices, axis=0, return_counts=True)
        probabilities = pair_counts / pair_counts.sum()
        usage[arm] = {
            "quantizers": stages,
            "distinct_code_pairs": len(pairs),
            "code_pair_perplexity": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        }
    return {**summary, "num_quantizers": NQ, "codebook_size": K, "codebooks": usage}


def render(summary, comparison, out):
    import matplotlib.pyplot as plt

    shared.render_reconstruction(
        summary,
        out,
        f"Muons | Q2, K=8192 | MC validation | end of epoch 3 | N={summary['objects']:,}",
    )
    with shared.paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
        x = np.arange(NQ)
        for ax, key, label, limit in (
            (axes[0, 0], "percent_used", "Used entries [%]", 100),
            (axes[0, 1], "normalized_perplexity", r"Normalized perplexity $\mathcal{P}/K$", 1),
        ):
            for i, arm in enumerate(ARMS):
                values = [row[key] for row in summary["codebooks"][arm]["quantizers"]]
                positions = x + (i - 0.5) * 0.34
                ax.bar(
                    positions,
                    values,
                    width=0.34,
                    color=shared.COLORS[arm],
                    label=shared.LABELS[arm],
                )
                for position, value in zip(positions, values):
                    ax.text(
                        position,
                        min(limit * 0.96, value + limit * 0.02),
                        f"{value:.4g}",
                        ha="center",
                        fontsize=8,
                    )
            ax.set_xticks(x, [f"Q{q}" for q in range(NQ)])
            ax.set_ylim(0, limit)
            ax.set_ylabel(label)
        for arm in ARMS:
            for stage in summary["codebooks"][arm]["quantizers"]:
                counts = np.asarray(stage["sorted_counts"])
                positive = counts[counts > 0]
                axes[1, 0].plot(
                    np.arange(1, len(positive) + 1),
                    positive / counts.sum(),
                    color=shared.COLORS[arm],
                    ls="-" if stage["q"] == 0 else "--",
                    label=f"{shared.LABELS[arm]}, Q{stage['q']}",
                )
            rows = comparison["paired_validation_passes"]
            axes[1, 1].plot(
                [row["step"] for row in rows],
                [row[arm]["validation_metrics"]["val/recon_loss"] for row in rows],
                color=shared.COLORS[arm],
                label=shared.LABELS[arm],
            )
        axes[1, 0].set(
            xscale="log",
            yscale="log",
            xlabel="Code rank (sorted independently)",
            ylabel="Assignment fraction",
        )
        axes[1, 1].set(xlabel="Training step", ylabel="Validation reconstruction loss")
        axes[0, 0].legend(frameon=False)
        axes[1, 0].legend(frameon=False, fontsize=8)
        axes[1, 1].legend(frameon=False)
        for ax in axes.flat:
            shared.style_axis(ax)
        shared.save_figure(fig, out / "muons_codebooks")
        plt.close(fig)


def plot(args):
    manifest, arrays = read_pairs(args.output_dir)
    summary = summaries(manifest, arrays, args.bins, args.min_bin_count)
    shared.reco.write_json(args.output_dir / "summary.json", summary)
    render(summary, manifest["comparison"], args.output_dir)
    pt = next(feature for feature in summary["features"] if feature["name"].lower() == "pt")
    for arm in ARMS:
        row = pt["rows"][arm]
        log.info(
            "%s: N=%s; pT R=%s%%; median |pT residual|=%s GeV; empty decoded pT bins=%s; distinct code pairs=%s",
            shared.LABELS[arm],
            f"{summary['objects']:,}",
            row.get("response_width_percent"),
            row["median_absolute_residual"],
            row["empty_decoded_bins"],
            summary["codebooks"][arm]["distinct_code_pairs"],
        )
        for stage in summary["codebooks"][arm]["quantizers"]:
            log.info(
                "  Q%s: used=%s/%s (%.3f%%); P/K=%.5f",
                stage["q"],
                stage["used_codes"],
                K,
                stage["percent_used"],
                stage["normalized_perplexity"],
            )
    (args.output_dir / "README.md").write_text(
        "# Muon Q2/K8192 MC-only reset comparison\n\n" + manifest["note"] + "\n\n"
        f"Both models use {manifest['checkpoint']}; initialization ON for Q0 only, reset OFF/ON for Q0 only. "
        "Q1 learns residuals with the unchanged backend. The saved MC datamodule is constructed once. "
        f"Both models decode exactly the same ordered {summary['objects']:,} validation muons, "
        "using the shared MC-training-fitted joblib. No refit, training, sample replacement or resplit.\n\n"
        "Original means detector-reconstructed input, inverse-transformed from the canonical batch, "
        "not generator truth. Histograms use the same original-derived bins and normalize by all "
        "finite pairs, including out-of-window objects. Ratios are decoded/original bin counts; "
        f"bins below {args.min_bin_count} original entries are omitted. Phi residuals wrap to [-pi,pi). "
        "Other residuals are decoded minus original. Binned median absolute errors use original pT. "
        "R_pT = 100 IQR(decoded/original pT)/median(decoded/original pT), using positive pT pairs.\n\n"
        "Each Q0/Q1 usage bar and normalized perplexity uses K=8192. Used means assigned at least "
        "once in this evaluation sample, not alive throughout training. Sorted occupancy uses "
        "solid lines for Q0 and dashed lines for Q1. Unique code pairs and pair perplexity describe "
        "combined assignment diversity, not guaranteed unique decoded values or reconstruction quality. "
        "Validation-loss curves come from matched training audits, not a new inference loss. "
        "Three-epoch pilot endpoints; no seed or uncertainty bands are implied.\n"
    )
    log.info(
        "Plots: %s/{muons_pt_comparison,muons_all_features,muons_codebooks}.{png,pdf}",
        args.output_dir,
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "plot"))
    parser.add_argument(
        "--reset-dir", type=Path, default=root / "results/atlas_muon_q2_cb8192_mc_reset_pilot_e3_v1"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-valid-objects", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--input-momentum-unit", choices=("GeV", "MeV"))
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--min-bin-count", type=int, default=20)
    args = parser.parse_args()
    args.reset_dir = args.reset_dir.resolve()
    args.output_dir = (args.output_dir or args.reset_dir / "figures/mc_validation_end_e3").resolve()
    if args.bins < 2 or args.min_bin_count < 1:
        parser.error("Use at least two bins and a positive minimum bin count")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "run":
        evaluate(args)
    plot(args)


if __name__ == "__main__":
    main()
