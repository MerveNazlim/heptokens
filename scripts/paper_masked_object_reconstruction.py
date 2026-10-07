#!/usr/bin/env python3
"""Figure 6: decode masked-object head predictions for Q1 or Q8, then plot caches.

The reference is the tokenizer decode of the true codes, NOT original features.
All paths are explicit. Evaluation, histogramming and rendering are separate.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import paper_tokenizer_capacity as capacity
import paper_tokenizer_reconstruction as reco
from paper_plot_style import OBJECTS, OBJECT_LABELS, paper_style, save_figure

log = logging.getLogger(__name__)
VERSION = "paper-masked-object-reconstruction-v1"
PLAN = "masked_prediction_plan.json"
SUMMARY = "masked_prediction_summary.json"
SUMMARY_ORIGINAL = "masked_prediction_summary_with_original.json"
LABELS = dict(
    reference_label="True-code decode",
    prediction_label="Predicted-code decode",
    ratio_label="Predicted /\nreference",
)


def assignments(values):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in OBJECTS or name in result:
            raise ValueError(f"Expected one OBJECT=CHECKPOINT per object, got {value!r}")
        result[name] = Path(path).expanduser().resolve()
    if set(result) != set(OBJECTS):
        raise ValueError(
            f"Supply all six tokenizer checkpoints; missing {set(OBJECTS) - set(result)}"
        )
    return result


def vocabulary_specs(vocabulary, quantizers):
    specs = {}
    type_ids, intervals = set(), []
    for obj in OBJECTS:
        entry = vocabulary["objects"][obj]
        levels = sorted(entry["quantizers"], key=lambda x: x["index"])
        if [s["index"] for s in levels] != list(range(quantizers)):
            raise ValueError(f"{obj}: vocabulary is not Q{quantizers}")
        type_id = int(entry["type_id"])
        if type_id in type_ids:
            raise ValueError("Ambiguous object type IDs")
        type_ids.add(type_id)
        for level in levels:
            low, size = int(level["base"]), int(level["size"])
            if low <= max(vocabulary["special_tokens"].values()) or size < 1:
                raise ValueError(f"Invalid code range for {obj}")
            if low + size > int(vocabulary["vocab_size"]):
                raise ValueError(f"Code range exceeds vocabulary for {obj}")
            intervals.append((low, low + size))
        specs[obj] = {"type_id": type_id, "levels": levels}
    ordered = sorted(intervals)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Overlapping object/quantizer code ranges")
    return specs


def tokenizer_record(obj, checkpoint, quantizers, levels=None):
    run = checkpoint.parent.parent
    cfg = OmegaConf.load(run / "full_config.yaml")
    if capacity.plain(cfg, "datamodule.object_type") != obj:
        raise ValueError(f"Wrong tokenizer object for {obj}: {run}")
    q = int(capacity.plain(cfg, "model.num_quantizers"))
    k = int(capacity.plain(cfg, "model.codebook_size"))
    if q != quantizers or (levels and any(int(s["size"]) != k for s in levels)):
        raise ValueError(f"{obj}: tokenizer q/K disagrees with the requested codes")
    matches = [
        c for c in capacity.plain(cfg, "datamodule.object_collections") if c["object_name"] == obj
    ]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous feature configuration for {obj}")
    inputs = list(matches[0]["inputs"])
    names = [Path(p).name for p in inputs]
    saved = capacity.plain(cfg, "datamodule.transforms.preprocess.cst_fn.filename")
    if not saved:
        raise ValueError(f"Missing saved preprocessor for {obj}; no substitution")
    prep = Path(saved).expanduser()
    if not prep.is_absolute():
        # The diagnostic helper resolves relative preprocessing paths from cwd.
        prep = prep.resolve()
    metadata = prep.with_suffix(".json")
    description = json.loads(metadata.read_text())
    if (
        description.get("object_type") != obj
        or description.get("feature_paths") != inputs
        or description.get("feature_names") != names
    ):
        raise ValueError(f"Preprocessor feature order/object mismatch: {metadata}")
    return {
        "q": q,
        "k": k,
        "d": int(capacity.plain(cfg, "model.codebook_dim")),
        "feature_names": names,
        "checkpoint": capacity.file_record(checkpoint, content_hash=True),
        "config": capacity.file_record(run / "full_config.yaml", content_hash=True),
        "preprocessor": capacity.file_record(prep, content_hash=True),
        "preprocessor_metadata": capacity.file_record(metadata, content_hash=True),
    }


def mask_positions(tokens, mask, types, specs, probability, generator):
    import torch

    valid = torch.zeros_like(mask, dtype=torch.bool)
    for spec in specs.values():
        valid |= types.eq(spec["type_id"])
    valid &= mask.bool()
    # CPU RNG and no forced per-batch mask: fixed row order gives batch-independent masks.
    chosen = torch.rand(valid.shape, generator=generator).to(valid.device) < probability
    return valid & chosen


def predict_codes(model, tokens, mask, types, selected, specs):
    """Use the trained head; never expose masked labels to prediction."""
    import torch

    corrupted = tokens.clone()
    corrupted[selected.unsqueeze(-1) & tokens.ne(model.pad_token_id)] = model.mask_token_id
    selected_types = types[selected]
    if model.model.quantizer_decoder == "autoregressive":
        predictions = model.model.generate_masked_object_codes(
            tokens=corrupted,
            mask=mask,
            type_ids=types,
            masked_positions=selected,
            sample=False,
        )
        return predictions, torch.zeros_like(predictions, dtype=torch.bool)
    logits = model.model.masked_logits(corrupted, mask, types, selected)
    predictions = torch.zeros(logits.shape[:2], dtype=torch.long, device=tokens.device)
    invalid_global = torch.zeros_like(predictions, dtype=torch.bool)
    global_codes = logits.argmax(dim=-1)
    for spec in specs.values():
        rows = selected_types.eq(spec["type_id"])
        for q, level in enumerate(spec["levels"]):
            base, size = int(level["base"]), int(level["size"])
            predictions[rows, q] = logits[rows, q, base : base + size].argmax(-1) + base
            invalid_global[rows, q] = (global_codes[rows, q] < base) | (
                global_codes[rows, q] >= base + size
            )
    return predictions, invalid_global


def local_codes(codes, levels):
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != len(levels):
        raise ValueError("Code tuple width does not match the tokenizer")
    result = codes.astype(np.int64, copy=True)
    for q, spec in enumerate(levels):
        result[:, q] -= int(spec["base"])
        if np.any((result[:, q] < 0) | (result[:, q] >= int(spec["size"]))):
            raise ValueError(f"Illegal code for quantizer {q}; refusing to clip or replace")
    return result


def verify_receipt(directory, plan, obj):
    path = directory / "decoded_arrays" / f"{obj}.npz"
    receipt = reco.read_sealed(path.with_suffix(".json"), "receipt_id")
    if receipt["plan_id"] != plan["plan_id"] or receipt["object"] != obj:
        raise ValueError(f"Wrong cached prediction provenance: {path}")
    if receipt["arrays"]["path"] != str(path.resolve()):
        raise ValueError(f"Moved prediction cache: {path}")
    capacity.verify_file(receipt["arrays"])
    return path, receipt


def load_pretrained_model(checkpoint):
    """Keep Lightning loading strict, including for newer Hydra checkpoint metadata."""
    from heptokens.models.foundation_grouped import LitGroupedMaskedSequenceModel

    try:
        return LitGroupedMaskedSequenceModel.load_from_checkpoint(
            checkpoint, map_location="cpu", strict=True
        )
    except ModuleNotFoundError as error:
        if error.name != "hydra._internal.target_policy":
            raise

    from lightning.pytorch.core.saving import _load_state
    from heptokens.models.foundation_grouped_cls_classifier import _load_checkpoint_state

    log.info(
        "Loading trusted pretrained checkpoint with existing Hydra compatibility: %s", checkpoint
    )
    state = _load_checkpoint_state(checkpoint)
    # Reuse the loaded state rather than rereading or rewriting this large checkpoint.
    # Lightning restores saved hyperparameters and the complete model, including its head.
    return _load_state(LitGroupedMaskedSequenceModel, state, strict=True)


def evaluate(args):
    import pyarrow.parquet as pq
    import torch
    from heptokens.data.sequence import MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
    from analyze_vqvae_tokenizer import load_analysis_model, transform_list_and_cst_fn_from_cfg
    from evaluate_grouped_quantizer_sampling import decode_in_batches
    from probe_grouped_autoregressive_quantizers import load_vocabulary, make_loader, parquet_files

    directory = args.evaluation_dir.resolve()
    checkpoints = assignments(args.tokenizer_checkpoint)
    paths = parquet_files(args.prepared_dir.resolve(), args.split)
    vocabulary = load_vocabulary(paths[0])
    for path in paths:
        if load_vocabulary(path) != vocabulary:
            raise ValueError(f"Mixed vocabulary in evaluation shards: {path}")
        if not {TOKENS_KEY, MASK_KEY, TYPE_IDS_KEY}.issubset(
            pq.ParquetFile(path).schema_arrow.names
        ):
            raise ValueError(f"Missing grouped sequence columns: {path}")
    specs = vocabulary_specs(vocabulary, args.quantizers)
    tokenizers = {
        obj: tokenizer_record(obj, checkpoints[obj], args.quantizers, specs[obj]["levels"])
        for obj in OBJECTS
    }
    plan = reco.seal(
        {
            "version": VERSION,
            "quantizers": args.quantizers,
            "checkpoint": capacity.file_record(args.checkpoint.resolve(), content_hash=True),
            "shards": [capacity.file_record(p) for p in paths],
            "split": args.split,
            "vocabulary": vocabulary,
            "tokenizers": tokenizers,
            "input_momentum_unit": args.input_momentum_unit,
            "sample_label": args.sample_label,
            "n_events": args.n_events,
            "max_objects_per_type": args.max_objects_per_type,
            "mask_probability": args.mask_prob,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "decode_batch_size": args.decode_batch_size,
            "torch_version": str(torch.__version__),
            "prediction_rule": "legal-range argmax / free-running autoregressive greedy",
            "tokenizer_mapping": "User-confirmed export checkpoints; vocabulary checks q/K but does not store checkpoint hashes",
            "selection_note": "Deterministic prefix of explicit shards; masked event-row and sequence-position identities saved. Cross-Q original event matching is not verified.",
            "evaluation_source_digest": capacity.digest_json(
                {
                    f.__name__: inspect.getsource(f)
                    for f in (
                        vocabulary_specs,
                        tokenizer_record,
                        mask_positions,
                        predict_codes,
                        local_codes,
                        load_pretrained_model,
                        evaluate,
                    )
                }
            ),
        },
        "plan_id",
    )
    if (directory / PLAN).exists():
        if reco.read_sealed(directory / PLAN, "plan_id") != plan:
            raise ValueError(
                "Evaluation inputs/settings changed; existing cache will not be overwritten"
            )
    else:
        if list((directory / "decoded_arrays").glob("*.npz")):
            raise ValueError(
                "Existing decoded arrays without this evaluation plan; use summarize --legacy-method to reuse them"
            )
    pending = []
    for obj in OBJECTS:
        path = directory / "decoded_arrays" / f"{obj}.npz"
        if path.exists() or path.with_suffix(".json").exists():
            verify_receipt(directory, plan, obj)
            log.info("Reusing verified predictions: %s", path)
        else:
            pending.append(obj)
    if not pending:
        return
    if args.cache_only:
        raise FileNotFoundError(f"Missing predictions for {pending}; inference forbidden")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = load_pretrained_model(args.checkpoint).eval().to(device)
    capacity.verify_file(plan["checkpoint"])
    if int(model.hparams.max_quantizers) != args.quantizers or int(model.hparams.vocab_size) != int(
        vocabulary["vocab_size"]
    ):
        raise ValueError("Pretraining checkpoint Q/vocabulary does not match the evaluation tokens")
    for key, actual in (("pad", model.pad_token_id), ("mask", model.mask_token_id)):
        if vocabulary["special_tokens"][key] != actual:
            raise ValueError(f"Checkpoint {key} ID disagrees with the vocabulary")
    saved_vocab = model.hparams.get("token_vocabulary")
    if saved_vocab and any(
        saved_vocab.get(k) != vocabulary.get(k)
        for k in ("objects", "special_tokens", "event_tokens")
    ):
        raise ValueError("Checkpoint token ranges differ from the shard vocabulary")
    if not (directory / PLAN).exists():
        reco.write_json(directory / PLAN, plan)
    loader = make_loader(
        paths,
        max_rows=args.n_events,
        batch_size=args.batch_size,
        stream_batch_size=1024,
        shuffle=False,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    stored = {
        obj: {"true_codes": [], "predicted_codes": [], "positions": [], "invalid_global": []}
        for obj in pending
    }
    counts = dict.fromkeys(pending, 0)
    row_offset = 0
    with torch.inference_mode():
        for batch in loader:
            tokens, mask, types = (
                batch[k].to(device) for k in (TOKENS_KEY, MASK_KEY, TYPE_IDS_KEY)
            )
            selected = mask_positions(tokens, mask, types, specs, args.mask_prob, generator)
            if selected.any():
                predictions, invalid = predict_codes(model, tokens, mask, types, selected, specs)
                labels, selected_types = tokens[selected], types[selected]
                locations = selected.nonzero().cpu().numpy()
                locations[:, 0] += row_offset
                for obj in pending:
                    rows = (
                        selected_types.eq(specs[obj]["type_id"])
                        .nonzero()
                        .flatten()[: max(0, args.max_objects_per_type - counts[obj])]
                    )
                    if not len(rows):
                        continue
                    stored[obj]["true_codes"].append(labels[rows].cpu().numpy())
                    stored[obj]["predicted_codes"].append(predictions[rows].cpu().numpy())
                    stored[obj]["invalid_global"].append(invalid[rows].cpu().numpy())
                    stored[obj]["positions"].append(locations[rows.cpu().numpy()])
                    counts[obj] += len(rows)
            row_offset += tokens.shape[0]
            if all(n >= args.max_objects_per_type for n in counts.values()):
                break
    decoder_kind = model.model.quantizer_decoder
    for shard in plan["shards"]:
        capacity.verify_file(shard)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if any(n == 0 for n in counts.values()):
        raise ValueError(
            f"No masked objects for some requested types: {counts}; no sample replacement"
        )
    for obj in pending:
        record = tokenizers[obj]
        for key in ("checkpoint", "config", "preprocessor", "preprocessor_metadata"):
            capacity.verify_file(record[key])
        arrays = {k: np.concatenate(v) for k, v in stored[obj].items()}
        true = local_codes(arrays["true_codes"], specs[obj]["levels"])
        predicted = local_codes(arrays["predicted_codes"], specs[obj]["levels"])
        cfg = OmegaConf.load(record["config"]["path"])
        _, inverse = transform_list_and_cst_fn_from_cfg(cfg)
        if inverse is None:
            raise ValueError(f"No inverse preprocessor for {obj}")
        tokenizer = load_analysis_model(
            checkpoints[obj].parent.parent, str(checkpoints[obj]), device
        )
        if (
            int(tokenizer.hparams.num_quantizers),
            int(tokenizer.hparams.codebook_size),
            int(tokenizer.hparams.codebook_dim),
        ) != (args.quantizers, record["k"], record["d"]):
            raise ValueError(f"Tokenizer checkpoint disagrees with its saved configuration: {obj}")
        with torch.inference_mode():
            for key, codes in (("reference", true), ("prediction", predicted)):
                arrays[key] = decode_in_batches(
                    tokenizer,
                    codes,
                    transformer=inverse,
                    device=device,
                    n_features=len(record["feature_names"]),
                    batch_size=args.decode_batch_size,
                )
        arrays["feature_names"] = np.asarray(record["feature_names"])
        path = directory / "decoded_arrays" / f"{obj}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        reco.write_json(
            path.with_suffix(".json"),
            reco.seal(
                {
                    "plan_id": plan["plan_id"],
                    "object": obj,
                    "decoder_kind": decoder_kind,
                    "masked_objects": counts[obj],
                    "events_scanned": row_offset,
                    "illegal_global_argmax_per_quantizer": arrays["invalid_global"]
                    .sum(axis=0)
                    .tolist(),
                    "arrays": capacity.file_record(path, content_hash=True),
                },
                "receipt_id",
            ),
        )
        log.info("Saved Q%s %s masked prediction decodes: %s", args.quantizers, obj, path)
        del tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


def read_pair(path, reference_key, prediction_key, expected_features):
    with np.load(path, allow_pickle=False) as loaded:
        names = loaded["feature_names"].tolist()
        x, y = loaded[reference_key], loaded[prediction_key]
    if names != expected_features or len(set(names)) != len(names):
        raise ValueError(f"Wrong feature names/order: {path}")
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != len(names) or not len(x):
        raise ValueError(f"Missing/misaligned/empty decoded pairs: {path}")
    if sum(name.lower() == "pt" for name in names) != 1:
        raise ValueError(f"Expected exactly one pT feature: {path}")
    return names, x, y


def audit_pair(args):
    """Verify cached Q8/Q1 object identities and feature exclusions, without inference."""
    import pyarrow.parquet as pq

    directories = (args.q8_dir.resolve(), args.q1_dir.resolve())
    plans = [reco.read_sealed(path / PLAN, "plan_id") for path in directories]
    for q, plan in zip((8, 1), plans):
        if plan["version"] != VERSION or plan["quantizers"] != q:
            raise ValueError(f"Expected native Q{q} evaluation plan")
    for key in (
        "split",
        "n_events",
        "max_objects_per_type",
        "mask_probability",
        "seed",
        "input_momentum_unit",
        "torch_version",
        "evaluation_source_digest",
    ):
        if plans[0][key] != plans[1][key]:
            raise ValueError(f"Q8/Q1 evaluation settings differ: {key}")
    for obj in OBJECTS:
        if (
            plans[0]["vocabulary"]["objects"][obj]["type_id"]
            != plans[1]["vocabulary"]["objects"][obj]["type_id"]
        ):
            raise ValueError(f"Q8/Q1 type IDs differ for {obj}")

    # Recheck the ordered source-event mapping, not just counts or RNG seeds.
    columns = ["source_file", "event_index", "mask", "type_ids"]

    def identity_batches(plan):
        remaining = plan["n_events"]
        for record in plan["shards"]:
            capacity.verify_file(record)
            if remaining <= 0:
                continue
            for batch in pq.ParquetFile(record["path"]).iter_batches(
                batch_size=1024, columns=columns
            ):
                count = min(remaining, batch.num_rows)
                yield batch.slice(0, count)
                remaining -= count
                if remaining <= 0:
                    break

    left, right = (iter(identity_batches(plan)) for plan in plans)
    a, b = next(left, None), next(right, None)
    events = 0
    while a is not None and b is not None:
        count = min(a.num_rows, b.num_rows)
        for column in columns:
            if not a.column(column).slice(0, count).equals(b.column(column).slice(0, count)):
                raise ValueError(f"Q8/Q1 source rows differ in {column} near event row {events}")
        events += count
        a = a.slice(count) if count < a.num_rows else next(left, None)
        b = b.slice(count) if count < b.num_rows else next(right, None)
    if a is not None or b is not None or events == 0:
        raise ValueError("Q8/Q1 event prefixes differ in length or are empty")

    for obj in OBJECTS:
        contents = []
        for directory, plan in zip(directories, plans):
            path, receipt = verify_receipt(directory, plan, obj)
            with np.load(path, allow_pickle=False) as arrays:
                positions = arrays["positions"]
                reference, prediction = arrays["reference"], arrays["prediction"]
                names = arrays["feature_names"].tolist()
            n = receipt["masked_objects"]
            if (
                positions.shape != (n, 2)
                or reference.shape != (n, len(names))
                or prediction.shape != reference.shape
            ):
                raise ValueError(f"{obj}: invalid cache shapes")
            if (
                not np.issubdtype(positions.dtype, np.integer)
                or n == 0
                or np.any(positions < 0)
                or np.any(positions[:, 0] >= events)
            ):
                raise ValueError(f"{obj}: invalid saved object positions")
            if len(np.unique(positions, axis=0)) != n:
                raise ValueError(f"{obj}: duplicate saved object positions")
            contents.append((positions, names, np.isfinite(reference) & np.isfinite(prediction)))
        positions, names, finite = contents[0]
        other_positions, other_names, other_finite = contents[1]
        if not np.array_equal(positions, other_positions):
            raise ValueError(
                f"{obj}: Q8/Q1 retained masked object positions differ ({len(positions)} versus {len(other_positions)})"
            )
        if names != other_names:
            raise ValueError(f"{obj}: Q8/Q1 feature lists differ")
        if not np.array_equal(finite, other_finite):
            differing = [
                name
                for i, name in enumerate(names)
                if not np.array_equal(finite[:, i], other_finite[:, i])
            ]
            raise ValueError(
                f"{obj}: Q8/Q1 finite-pair exclusions differ for {differing}; do not treat the current per-feature plots as matched"
            )
        pt = names.index(next(name for name in names if name.lower() == "pt"))
        log.info(
            "PAIRED OK: %s: %s masked objects; %s finite pT pairs",
            obj,
            len(positions),
            int(finite[:, pt].sum()),
        )
    log.info(
        "PAIRED OK: %s source events; all six saved masked-object selections and per-feature exclusions match. No inference performed.",
        events,
    )


def legacy_source(args):
    """Explicit, read-only import of the existing decoder-comparison NPZ schema."""
    if args.quantizers is None or not args.sample_label or not args.input_momentum_unit:
        raise ValueError(
            "Legacy arrays require --quantizers, --sample-label and --input-momentum-unit"
        )
    summary_path = args.evaluation_dir / "summary.json"
    metadata = json.loads(summary_path.read_text())
    missing = set(OBJECTS) - set(metadata.get("decoded", {}))
    if missing:
        raise ValueError(
            f"Legacy evaluation did not decode all six objects: missing {sorted(missing)}; no inference performed"
        )
    method = args.legacy_method
    prefix = method.split("_")[0]
    checkpoint = Path(metadata[f"{prefix}_checkpoint"])
    cfg_path = checkpoint.parent.parent / "full_config.yaml"
    cfg = OmegaConf.load(cfg_path)
    if int(capacity.plain(cfg, "model.max_quantizers")) != args.quantizers:
        raise ValueError("Legacy pretraining configuration does not match requested Q")
    tokenizers = {
        obj: tokenizer_record(
            obj, Path(metadata["decoded"][obj]["tokenizer_checkpoint"]), args.quantizers
        )
        for obj in OBJECTS
    }
    return {
        "quantizers": args.quantizers,
        "sample_label": args.sample_label,
        "input_momentum_unit": args.input_momentum_unit,
        "tokenizers": tokenizers,
        "checkpoint": capacity.file_record(checkpoint, content_hash=True),
        "config": capacity.file_record(cfg_path, content_hash=True),
        "evaluation_summary": capacity.file_record(summary_path, content_hash=True),
        "method": method,
        "metadata": metadata,
        "provenance_note": "Legacy arrays lack per-file receipts and event IDs. Current source hashes are frozen; historical provenance and cross-Q event matching are not verified.",
    }


def summarize(args):
    directory = args.evaluation_dir
    if args.legacy_method:
        source = legacy_source(args)
        reference_key = "truth_sample" if args.legacy_method.endswith("sample") else "truth_greedy"
        prediction_key = args.legacy_method
        records = [
            capacity.file_record(directory / "decoded_arrays" / f"{obj}.npz", content_hash=True)
            for obj in OBJECTS
        ]
    else:
        source = reco.read_sealed(directory / PLAN, "plan_id")
        if source["version"] != VERSION:
            raise ValueError("Not a masked-object prediction evaluation")
        records = [verify_receipt(directory, source, obj)[1]["arrays"] for obj in OBJECTS]
        reference_key, prediction_key = "reference", "prediction"
    settings = {
        "n_bins": args.n_bins,
        "min_ratio_count": args.min_ratio_count,
        "percentiles": [0.5, 99.5],
    }
    with_original = getattr(args, "with_original", False)
    originals, original_records = {}, {}
    if with_original:
        from paper_masked_originals import verify_original

        if args.legacy_method:
            raise ValueError("Original overlays require native saved object positions")
        for obj in OBJECTS:
            originals[obj], original_records[obj] = verify_original(directory, source, obj)
    binning = None
    binning_record = None
    if args.binning_from:
        path = args.binning_from / (SUMMARY_ORIGINAL if with_original else SUMMARY)
        binning = reco.read_sealed(path, "summary_id")
        if binning["version"] != VERSION or binning["statistics"] != settings:
            raise ValueError("Binning source has incompatible statistics or is not Figure 6")
        binning_record = capacity.file_record(path, content_hash=True)
    identity = capacity.digest_json(
        {
            "source": source,
            "arrays": records,
            "statistics": settings,
            "binning_source": binning_record,
            **({"original_inputs": original_records} if with_original else {}),
        }
    )
    destination = directory / (SUMMARY_ORIGINAL if with_original else SUMMARY)
    if destination.exists():
        cached = reco.read_sealed(destination, "summary_id")
        if cached["input_id"] != identity:
            raise ValueError(
                "Summary source/statistics changed; not overwriting the existing summary"
            )
        log.info("Reusing verified summary: %s", destination)
        return
    summaries = {}
    for obj, record in zip(OBJECTS, records):
        names, x, y = read_pair(
            Path(record["path"]),
            reference_key,
            prediction_key,
            source["tokenizers"][obj]["feature_names"],
        )
        features = reco.summarize_arrays(
            {"masked": {"original": x, "reconstruction": y}},
            {"statistics": settings, "input_momentum_unit": source["input_momentum_unit"]},
            {"feature_names": names},
        )
        if with_original and binning is None:
            for i, feature in enumerate(features):
                finite = np.isfinite(x[:, i]) & np.isfinite(y[:, i])
                edges = reco.bin_edges(
                    originals[obj][finite, i] * feature["scale"],
                    settings["n_bins"],
                    settings["percentiles"],
                    discrete=True,
                )
                feature["edges"] = edges.tolist()
                feature["domains"]["masked"] = reco.feature_summary(
                    x[:, i] * feature["scale"],
                    y[:, i] * feature["scale"],
                    edges,
                    np.asarray(feature["residual_edges"]),
                    settings["min_ratio_count"],
                    feature["name"],
                )
        if binning is not None:
            shared = binning["objects"][obj]["features"]
            if [f["name"] for f in shared] != names:
                raise ValueError(f"Q1/Q8 feature lists differ for {obj}; cannot share bins")
            for i, (feature, template) in enumerate(zip(features, shared)):
                if feature["unit"] != template["unit"]:
                    raise ValueError(f"Q1/Q8 feature units differ for {obj}")
                feature["edges"] = template["edges"]
                feature["residual_edges"] = template["residual_edges"]
                feature["domains"]["masked"] = reco.feature_summary(
                    x[:, i] * feature["scale"],
                    y[:, i] * feature["scale"],
                    np.asarray(feature["edges"]),
                    np.asarray(feature["residual_edges"]),
                    settings["min_ratio_count"],
                    feature["name"],
                )
        if any(not f["domains"]["masked"]["finite_pairs"] for f in features):
            raise ValueError(f"No finite paired values for a feature of {obj}")
        if with_original:
            for i, feature in enumerate(features):
                finite = np.isfinite(x[:, i]) & np.isfinite(y[:, i])
                values = originals[obj][finite, i] * feature["scale"]
                counts = np.histogram(values, bins=feature["edges"])[0]
                row = feature["domains"]["masked"]
                row["input_counts"] = counts.tolist()
                row["outside_input"] = int(len(values) - counts.sum())
        summaries[obj] = {"features": features}
    reco.write_json(
        destination,
        reco.seal(
            {
                "version": VERSION,
                "input_id": identity,
                "source": source,
                "arrays": records,
                "statistics": settings,
                "binning_source": binning_record,
                **({"original_inputs": original_records} if with_original else {}),
                "objects": summaries,
            },
            "summary_id",
        ),
    )
    log.info("Saved %s; cached arrays only, no inference", destination)


def render_pt(summary, *, log_y=True):
    import matplotlib.pyplot as plt

    source = summary["source"]
    fig = plt.figure(figsize=(12, 8.2))
    grid = fig.add_gridspec(
        2, 3, left=0.075, right=0.985, bottom=0.08, top=0.91, wspace=0.35, hspace=0.4
    )
    for i, (obj, label) in enumerate(zip(OBJECTS, OBJECT_LABELS)):
        feature = next(f for f in summary["objects"][obj]["features"] if f["name"].lower() == "pt")
        ax, ratio = reco.draw_distribution(
            fig,
            grid[i // 3, i % 3],
            feature,
            "masked",
            log_y=log_y,
            count_label="objects",
            **LABELS,
        )
        ratio.set_xlabel(r"$p_T$ [GeV]")
        count = feature["domains"]["masked"]["finite_pairs"]
        ax.set_title(f"({chr(97+i)}) {label}", loc="left")
        ax.text(
            0.97, 0.95, f"N = {count:,}", transform=ax.transAxes, ha="right", va="top", fontsize=8
        )
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
    fig.text(
        0.075, 0.975, f"Q{source['quantizers']} | {source['sample_label']}", va="top", fontsize=9
    )
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        ncol=len(labels),
        frameon=False,
        fontsize=9,
    )
    return fig


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_name = SUMMARY_ORIGINAL if getattr(args, "with_original", False) else SUMMARY
    summary = reco.read_sealed(args.evaluation_dir / summary_name, "summary_id")
    if summary["version"] != VERSION or set(summary["objects"]) != set(OBJECTS):
        raise ValueError("Expected all six masked-object summaries")
    source = summary["source"]
    q = source["quantizers"]
    if q not in (1, 8):
        raise ValueError("Figure 6 requires Q1 or Q8")
    for obj in OBJECTS:
        features = summary["objects"][obj]["features"]
        if sum(f["name"].lower() == "pt" for f in features) != 1:
            raise ValueError(f"Missing/ambiguous pT summary for {obj}")
        if any(not f["domains"]["masked"]["finite_pairs"] for f in features):
            raise ValueError(f"Empty masked-object feature summary for {obj}")
    destination = args.evaluation_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"figure6_q{q}"
    with paper_style():
        fig = render_pt(summary, log_y=args.y_scale == "log")
        save_figure(fig, destination / f"{stem}_pt_{args.y_scale}")
        plt.close(fig)
        if not args.pt_only:
            for obj in OBJECTS:
                label = dict(zip(OBJECTS, OBJECT_LABELS))[obj]
                fig = reco.render_triptych_page(
                    summary["objects"][obj]["features"],
                    obj,
                    "masked",
                    1,
                    1,
                    log_y=args.y_scale == "log",
                    count_label="objects",
                    header=f"{label} | Q{q} | {source['sample_label']}",
                    residual_label="Predicted - true-code decode\n",
                    reference_axis="True-code decoded ",
                    prediction_axis="Predicted-code decoded ",
                    show_absolute_error=True,
                    **LABELS,
                )
                save_figure(fig, destination / f"{stem}_{obj}_all_features_{args.y_scale}")
                plt.close(fig)
    rows = []
    for obj in OBJECTS:
        for feature in summary["objects"][obj]["features"]:
            row = feature["domains"]["masked"]
            rows.append(
                {
                    "object": obj,
                    "feature": feature["name"],
                    "unit": feature["unit"],
                    **{
                        k: row[k]
                        for k in (
                            "finite_pairs",
                            "nonfinite_pairs",
                            "median_residual",
                            "median_absolute_residual",
                            "residual_iqr",
                            "outside_original",
                            "outside_decoded",
                        )
                    },
                }
            )
    with (destination / f"{stem}_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / f"{stem}_caption.md").write_text(
        f"# Figure 6: Q{q} masked-object reconstruction\n\n"
        f"Evaluation sample: {source['sample_label']}. Reference: VQ/RVQ decoder applied to the "
        "true codes. Prediction: the same object decoder applied to codes predicted by the "
        "event-transformer's masked-object head. "
        + (
            "The additional Original curve uses aligned continuous inputs, inverse-preprocessed "
            "with the exact saved export joblib (subject to preprocessing/float precision), not VQ-decoded features. "
            "All three curves use identical masked objects and bins. "
            if "original_inputs" in summary
            else "No original detector-feature curve is included. "
        )
        + "The ratio, residual and density panels remain prediction versus true-code decode; "
        "they do not measure total error relative to original detector features. "
        "Every distribution, residual and paired density uses the same finite masked-object pairs. "
        "The parallel head is decoded by argmax restricted to each legal object/quantizer range; "
        "autoregressive greedy decoding is free-running, never teacher-forced. "
        "For legacy sampling imports the recorded method/temperature in the source manifest instead applies. "
        "All q codes of a selected object are hidden. Non-masked positions provide context. "
        "Histograms are normalized by total finite paired objects, including entries outside the "
        "displayed range. Ratios are predicted/true-code decoded bin counts, not per-object responses. "
        f"Bins with fewer than {summary['statistics']['min_ratio_count']} reference entries are suppressed. "
        + (
            "Edges use original-input 0.5--99.5 percentiles (small integer features use integer bins), "
            if "original_inputs" in summary
            else "Edges use true-code decoded 0.5--99.5 percentiles (small integer features use integer bins), "
        )
        + "or are explicitly reused from the other Q evaluation when binning_source is recorded. "
        "Residuals are predicted minus true-code decoded values, including unwrapped phi to match "
        "the tokenizer triptychs. Density axes have identical limits and a dashed equality line. "
        "Median residual, median absolute residual and residual IQR use all finite pairs. "
        "Each PDF contains one page with all features of that object. "
        "Q1 and Q8 have different tokenizer reference decodes; their separate residual widths alone "
        "do not rank end-to-end reconstruction. Cross-Q event/mask matching is not established by "
        "this plotting command, even if counts and RNG seeds agree. See the source manifest for "
        "checkpoint, sample, masking and tokenizer identities.\n"
    )
    reco.write_json(
        destination / f"{stem}_sources.json",
        {
            "summary": capacity.file_record(args.evaluation_dir / summary_name, content_hash=True),
            "source": source,
            "statistics": summary["statistics"],
            "binning_source": summary.get("binning_source"),
            "original_inputs": summary.get("original_inputs"),
            "y_scale": args.y_scale,
        },
    )
    log.info("Saved Q%s Figure 6 plots in %s; summaries only, no inference", q, destination)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser(
        "evaluate", help="Run one explicit pretrained head and decode all six objects"
    )
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--prepared-dir", type=Path, required=True)
    evaluation.add_argument("--split", choices=("val", "test"), required=True)
    evaluation.add_argument("--quantizers", type=int, choices=(1, 8), required=True)
    evaluation.add_argument(
        "--tokenizer-checkpoint", action="append", required=True, metavar="OBJECT=CKPT"
    )
    evaluation.add_argument(
        "--confirm-tokenizer-mapping",
        action="store_true",
        help="Confirm these exact tokenizers created the Parquet codes; vocabulary alone cannot prove checkpoint identity",
    )
    evaluation.add_argument("--input-momentum-unit", choices=("GeV", "MeV"), required=True)
    evaluation.add_argument("--sample-label", required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--n-events", type=int, default=50_000)
    evaluation.add_argument("--max-objects-per-type", type=int, default=10_000)
    evaluation.add_argument("--mask-prob", type=float, default=0.15)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--batch-size", type=int, default=8)
    evaluation.add_argument("--decode-batch-size", type=int, default=2048)
    evaluation.add_argument("--cache-only", action="store_true")
    summary = commands.add_parser(
        "summarize", help="Bin existing decoded arrays; never run inference"
    )
    summary.add_argument(
        "--legacy-method",
        choices=(
            "parallel_greedy",
            "autoregressive_greedy",
            "parallel_sample",
            "autoregressive_sample",
        ),
    )
    summary.add_argument("--quantizers", type=int, choices=(1, 8))
    summary.add_argument("--input-momentum-unit", choices=("GeV", "MeV"))
    summary.add_argument("--sample-label")
    summary.add_argument(
        "--binning-from",
        type=Path,
        help="Use the existing Figure 6 summary in this evaluation root for shared Q8/Q1 histogram edges",
    )
    summary.add_argument("--n-bins", type=int, default=60)
    summary.add_argument("--min-ratio-count", type=int, default=20)
    plotting = commands.add_parser(
        "plot", help="Read summaries only; pT grid plus one all-feature page per object"
    )
    plotting.add_argument("--y-scale", choices=("log", "linear"), default="log")
    plotting.add_argument("--pt-only", action="store_true")
    paired = commands.add_parser(
        "audit-pair",
        help="Check saved Q8/Q1 source events and retained masked objects; no inference or writes",
    )
    paired.add_argument("--q8-dir", type=Path, required=True)
    paired.add_argument("--q1-dir", type=Path, required=True)
    original = commands.add_parser(
        "attach-originals",
        help="Read aligned continuous inputs for existing Q8/Q1 caches; no GPU inference",
    )
    original.add_argument("--q8-dir", type=Path, required=True)
    original.add_argument("--q1-dir", type=Path, required=True)
    original.add_argument("--continuous-dir", type=Path, required=True)
    inputs = commands.add_parser(
        "check-inputs", help="Verify downloaded paired validation event prefixes before inference"
    )
    inputs.add_argument("--prepared-root", type=Path, required=True)
    inputs.add_argument("--n-events", type=int, required=True)
    comparison = commands.add_parser(
        "compare",
        help="Paired Q1/Q8 feature errors against the same original inputs; cached arrays only",
    )
    comparison.add_argument("--q8-dir", type=Path, required=True)
    comparison.add_argument("--q1-dir", type=Path, required=True)
    comparison.add_argument("--output-dir", type=Path, required=True)
    comparison.add_argument("--n-bins", type=int, default=8)
    comparison.add_argument("--min-bin-count", type=int, default=50)
    comparison.add_argument("--binning", choices=("quantile", "linear"), default="quantile")
    comparison.add_argument("--show-tokenizer-reference", action="store_true")
    for cmd in (summary, plotting):
        cmd.add_argument(
            "--with-original",
            action="store_true",
            help="Include verified original input distributions; keep prediction/reference ratios and residuals",
        )
    for cmd in (evaluation, summary, plotting):
        cmd.add_argument(
            "--evaluation-dir",
            type=Path,
            required=True,
            help="Reuse this evaluation root for caches, summary and figures; no automatic directory selection",
        )
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if not args.confirm_tokenizer_mapping:
            parser.error(
                "--confirm-tokenizer-mapping is required; do not substitute a tokenizer with the same q/K"
            )
        if not 0 < args.mask_prob <= 1:
            parser.error("--mask-prob must be in (0, 1]")
    if (
        args.command == "summarize"
        and not args.legacy_method
        and any(
            value is not None
            for value in (args.quantizers, args.sample_label, args.input_momentum_unit)
        )
    ):
        parser.error(
            "Native evaluation uses its frozen q/sample/unit; overrides apply only to --legacy-method"
        )
    for key in (
        "n_events",
        "max_objects_per_type",
        "batch_size",
        "decode_batch_size",
        "n_bins",
        "min_ratio_count",
        "min_bin_count",
    ):
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.command == "compare":
        from paper_masked_comparison import compare

        compare(args)
        return
    if args.command == "attach-originals":
        from paper_masked_originals import attach

        attach(args)
        return
    if args.command == "check-inputs":
        from paper_masked_originals import check_inputs

        check_inputs(args)
        return
    {"evaluate": evaluate, "summarize": summarize, "plot": plot, "audit-pair": audit_pair}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
