#!/usr/bin/env python3
"""Encode FINAL Q4 IDs from saved ORIGINAL preprocessed object features.

Preserve every event, split, position, label and non-token column. Never decode
Q8 IDs, truncate them, resplit events, refit preprocessing or use test labels in
tokenization. Refuse token-only datasets: those require the original H5 files.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from hybrid_q4_downstream import classes, sequence_numpy, sha256, SPLITS
import q4_data_grl_campaign as campaign
import tokenize_objects_to_grouped_parquet as grouped
import tokenize_objects_to_parquet_with_atlasopenmagic_metadata as flat
from package_final_q4_tokenizer_artifacts import PREPROCESS_SHA256


def validate_feature_schema(schema, models, collections, preprocessors):
    if schema.get("preprocessing", {}).get("objects") != "matching fitted VQ-VAE input transformer":
        raise ValueError("Not the saved original preprocessed features")
    campaign.validate_q4_feature_schema(models, collections, preprocessors)
    for name, model in models.items():
        spec = schema.get("objects", {}).get(name)
        if spec is None or spec["feature_names"] != list(model.feature_names):
            raise ValueError(f"{name}: saved continuous feature order differs from Q4 training")
        if spec["feature_paths"] != collections[name]["inputs"]:
            raise ValueError(f"{name}: saved feature paths differ from final Q4 schema")
        if spec["type_id"] != flat.TYPE_IDS[name]:
            raise ValueError(f"{name}: object type ID differs")


def make_vocabulary(source, models):
    specs = source["event_tokens"]
    if len({spec["size"] for spec in specs}) != 1:
        raise ValueError("Non-uniform event bins not supported")
    args = SimpleNamespace(**{f"{name}_token_id": int(value)
                             for name, value in source["special_tokens"].items()},
                           object_order=source["object_order"], event_bins=specs[0]["size"], no_cls=False)
    result = grouped.build_grouped_vocabulary(models, [s["input"] for s in specs],
                                              {s["input"]: tuple(s["range"]) for s in specs}, args)
    if result["event_tokens"] != specs or result["special_tokens"] != source["special_tokens"]:
        raise ValueError("Event/special token IDs would change")
    if result["vocab_size"] != campaign.VOCAB_SIZE or result["max_quantizers"] != 4:
        raise ValueError("Unexpected final Q4 vocabulary")
    if source.get("includes_cls") is not True or source.get("sequence_layout") != "one_position_per_object":
        raise ValueError("Expected grouped CLS sequences")
    if "continuous_schema" in source:
        result["continuous_schema"] = source["continuous_schema"]
    return result


def encode_batch(batch, models, vocabulary, schema, device, object_batch_size=1024):
    column = lambda key: batch.column(batch.schema.get_field_index(key))
    original = sequence_numpy(column("tokens"), np.int64)
    mask = sequence_numpy(column("mask"), bool)
    types = sequence_numpy(column("type_ids"), np.int64)
    features = sequence_numpy(column("continuous_features"), np.float32)
    feature_mask = sequence_numpy(column("continuous_feature_mask"), bool)
    if original.shape[1:] != (256, 8) or features.shape[:2] != mask.shape:
        raise ValueError("Unexpected source input shape")
    result = original[:, :, :4].copy()
    covered = np.zeros_like(mask)
    with torch.inference_mode():
        for name, model in models.items():
            selected = mask & (types == vocabulary["objects"][name]["type_id"])
            covered |= selected
            count = int(schema["objects"][name]["feature_count"])
            if not np.all(feature_mask[selected, :count]):
                raise ValueError(f"Missing original features for {name}")
            values = features[selected, :count]
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite features for {name}")
            local_codes = []
            for start in range(0, len(values), object_batch_size):
                chunk = torch.from_numpy(values[start:start + object_batch_size].copy())[None]
                indices = flat.encode_object_batch(model, chunk, torch.ones(chunk.shape[:2], dtype=torch.bool), device)
                local_codes.append(indices[0].numpy())
            if not local_codes:
                continue
            codes = np.concatenate(local_codes)
            specs = vocabulary["objects"][name]["quantizers"]
            if codes.shape != (int(selected.sum()), 4):
                raise ValueError(f"Wrong Q4 encoder output shape for {name}")
            for level, spec in enumerate(specs):
                if np.any(codes[:, level] < 0) or np.any(codes[:, level] >= spec["size"]):
                    raise ValueError("Out-of-range Q4 local code")
                codes[:, level] += spec["base"]
            result[selected] = codes
    # Nonphysical entries (CLS/context/SEP/padding) must not contain residual IDs.
    if np.any(original[~covered, 1:]):
        raise ValueError("Unexpected residual codes at nonphysical positions")
    array = pa.array(result.tolist(), type=pa.list_(pa.list_(pa.int64(), 4), 256))
    table = pa.Table.from_batches([batch])
    for name in ("tokens", "input_ids"):
        index = table.schema.get_field_index(name)
        if index >= 0:
            table = table.set_column(index, name, array)
    metadata = dict(table.schema.metadata or {})
    metadata[b"heptokens_token_vocabulary"] = json.dumps(vocabulary, sort_keys=True).encode()
    return table.replace_schema_metadata(metadata)


def prepare(args):
    if sha256(args.source_dir / "manifest.json") != args.source_manifest_sha256:
        raise ValueError("Source classification manifest checksum mismatch")
    if args.output_dir.exists() or args.output_dir.with_name(args.output_dir.name + ".partial").exists():
        raise FileExistsError("Refusing to overwrite existing/incomplete Q4 output")
    manifest = json.loads((args.source_dir / "manifest.json").read_text())
    if manifest.get("token_shape") != [256, 8] or manifest.get("identity_overlap_detected") is not False:
        raise ValueError("Source is not the matched prepared Q8 classification dataset")
    definitions = classes(manifest)
    config = flat.load_data_config(str(campaign.feature_config_path()))
    collections = flat.collection_map(config["object_collections"])
    device = torch.device(args.device)
    models = flat.load_models({name: str(args.bundle_root / "q4" / f"{name}.ckpt") for name in campaign.Q4_SIZES}, device)
    campaign.validate_q4_models(models)
    paths = {name: str(args.bundle_root / "preprocessing" / f"{name}.joblib") for name in models}
    if any(sha256(paths[name]) != PREPROCESS_SHA256[name] for name in models):
        raise ValueError("Final Q4 preprocessing differs from the matched Q1/Q8 preprocessing")
    preprocessors = flat.load_preprocessors(paths)
    temporary = args.output_dir.with_name(args.output_dir.name + ".partial")
    temporary.mkdir(parents=True)
    vocabulary, continuous_schema, source_vocab, inventory = None, None, None, {}
    for split in SPLITS:
        for name, _ in definitions:
            rows = 0
            for path in sorted((args.source_dir / split / name).glob("*.parquet")):
                parquet = pq.ParquetFile(path)
                metadata = parquet.schema_arrow.metadata or {}
                if not {"continuous_features", "continuous_feature_mask"}.issubset(parquet.schema_arrow.names):
                    raise ValueError("Token-only source: original H5/paired continuous features are required; cannot truncate Q8")
                schema = json.loads(metadata[b"heptokens_continuous_schema"])
                observed_vocab = json.loads(metadata[b"heptokens_token_vocabulary"])
                if vocabulary is None:
                    validate_feature_schema(schema, models, collections, preprocessors)
                    vocabulary, continuous_schema, source_vocab = make_vocabulary(observed_vocab, models), schema, observed_vocab
                if schema != continuous_schema or observed_vocab != source_vocab:
                    raise ValueError("Different source schemas/vocabularies")
                destination = temporary / path.relative_to(args.source_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with pq.ParquetWriter(destination, encode_batch(next(parquet.iter_batches(batch_size=1)),
                        models, vocabulary, schema, device).schema, compression="snappy") as writer:
                    for batch in parquet.iter_batches(batch_size=args.read_batch_size):
                        writer.write_table(encode_batch(batch, models, vocabulary, schema, device), row_group_size=4096)
                        rows += batch.num_rows
                inventory[str(destination.relative_to(temporary))] = dict(
                    rows=parquet.metadata.num_rows, sha256=sha256(destination), source_sha256=sha256(path))
                print(f"Encoded Q4 {split}/{name}/{path.name}", flush=True)
            if rows != manifest["split_counts"][split][name]:
                raise ValueError(f"Source row counts differ for {split}/{name}")
    result = copy.deepcopy(manifest)
    result["token_shape"] = [256, 4]
    result["representation"] = "final_q4"
    result["q4_reencoding"] = dict(source_manifest_sha256=args.source_manifest_sha256,
        tokenizer_bundle_sha256=campaign.BUNDLE_SHA, source="original saved preprocessed continuous_features",
        split_policy="unchanged", selection="unchanged", non_token_columns="unchanged", shards=inventory)
    # Never retain a source-token shard inventory as an inventory of Q4 bytes.
    for key in ("shards", "files", "outputs"):
        if key in result:
            result["q4_reencoding"][f"source_{key}"] = result.pop(key)
    (temporary / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (temporary / "manifest.sha256").write_text(sha256(temporary / "manifest.json") + "\n")
    temporary.rename(args.output_dir)
    print("PASS: final Q4 encoding; exact original events/splits/labels/positions retained", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "output-dir", "bundle-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--read-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.read_batch_size < 1:
        parser.error("Read batch size must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
