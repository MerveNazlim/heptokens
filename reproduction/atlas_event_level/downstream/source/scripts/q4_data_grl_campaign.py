#!/usr/bin/env python3
"""Final-Q4 conversion matched to the COMPLETE original Q1/Q8 campaign.

No new file selection, refitted preprocessing, or Q8 truncation is allowed.
Heavy imports are lazy: finalization/checking can run on a CPU submit worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

BUNDLE_SHA = "971a8768440414d076ec835e988fe2924f264d573d48c9f6a86c77c11c7f2081"
MEMBERSHIP_SHA = "d49371d0d81c57f19c57cf1f3751ff3037801c77e7116c1f3f19ab0a958b0d99"
VOCAB_SIZE = 262660
Q4_SIZES = dict(electrons=8192, muons=8192, taus=8192, photons=8192,
                jets=16384, tracks=16384)
Q4_FEATURE_CONFIG = "configs/datamodule/atlas_event_object_final_q4_export.yaml"
TOTALS = dict(total_rows=216095647, train_rows=194491678, val_rows=21603969,
              train_shards=3898, val_shards=440, input_file_count=2000,
              group_count=16)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gcloud(*arguments):
    subprocess.run(["gcloud", "storage", *map(str, arguments)], check=True)


def reference_group(reference_dir, group_id):
    reference_dir = Path(reference_dir)
    index = read(reference_dir / "index.json")
    if group_id not in index["groups"]:
        raise ValueError(f"Unexpected group: {group_id}")
    path = reference_dir / "conversion_status" / f"{group_id}.json"
    if sha256(path) != index["groups"][group_id]:
        raise ValueError("Original group-reference checksum mismatch")
    original = read(path)
    if original["group_id"] != group_id or original["invalid_input_file_count"] != 0:
        raise ValueError("Original group is invalid or incomplete")
    return original


def conversion_options(reference):
    from heptokens.data.data_grl_conversion import DataGrlConversionModule
    s = reference["conversion_settings"]
    # All non-tokenizer science settings come directly from the original status.
    policy = DataGrlConversionModule(
        datamodule_config=s["datamodule_config"], seed=s["seed"],
        train_frac=s["train_fraction_requested"], batch_size=s["batch_size"],
        max_seq_length=s["max_seq_length"], shard_rows=s["shard_rows"],
        row_group_rows=s["row_group_rows"], compression=s["compression"],
        event_bins=s["event_bins"], object_order=tuple(s["object_order"]),
        event_token_inputs=tuple(s["event_token_inputs"]),
        event_token_ranges=tuple(s["event_token_ranges"]),
        require_real_data=s["require_real_data"],
        **{f"{name}_token_id": value for name, value in s["special_token_ids"].items()},
    )
    if policy.manifest_settings() != s:
        raise ValueError("Original conversion settings cannot be reproduced exactly")
    args = SimpleNamespace(**policy.__dict__, num_objects=None, no_cls=False,
                           no_event_token=False, no_separators=False)
    return policy, args


def validate_q4_models(models):
    if set(models) != set(Q4_SIZES):
        raise ValueError("Exactly six final Q4 object tokenizers are required")
    for name, model in models.items():
        for field, expected in dict(codebook_size=Q4_SIZES[name],
                                    codebook_dim=16, num_quantizers=4).items():
            if int(getattr(model.hparams, field)) != expected:
                raise ValueError(f"{name}: {field} is not the final Q4 specification")


def feature_config_path():
    # Also works for the CPU finalizer invoked outside the extracted project.
    return Path(__file__).resolve().parents[1] / Q4_FEATURE_CONFIG


def feature_config_provenance():
    return dict(path=Q4_FEATURE_CONFIG, sha256=sha256(feature_config_path()))


def validate_q4_feature_schema(models, collections, preprocessors):
    """Check all six schemas even if an input contains no valid objects of a type."""
    import torch
    if set(collections) != set(Q4_SIZES) or set(preprocessors) != set(Q4_SIZES):
        raise ValueError("Exactly six Q4 feature collections and preprocessors are required")
    for name in Q4_SIZES:
        paths = collections[name]["inputs"]
        configured_names = [path.rsplit("/", 1)[-1] for path in paths]
        saved_names = [str(feature) for feature in models[name].feature_names]
        if not saved_names or configured_names != saved_names:
            raise ValueError(f"{name}: Q4 feature count/order mismatch: "
                             f"config={configured_names}, checkpoint={saved_names}")
        encoder_input = next((layer.in_features for layer in models[name].encoder.modules()
                              if isinstance(layer, torch.nn.Linear)), None)
        transformer = preprocessors[name]
        preprocessing_input = getattr(transformer, "n_features_in_", None)
        if preprocessing_input is None:
            preprocessing_input = getattr(getattr(transformer, "final_transformer", None),
                                          "n_features_in_", None)
        if encoder_input != len(paths) or preprocessing_input != len(paths):
            raise ValueError(f"{name}: Q4 feature dimensions disagree: config={len(paths)}, "
                             f"encoder={encoder_input}, preprocessing={preprocessing_input}")
        print(f"PASS {name}: {len(paths)} features; checkpoint order and preprocessing match",
              flush=True)


def validate_status(status, reference):
    for key in ("group_id", "source_counts", "membership_sha256", "total_rows",
                "train_rows", "val_rows", "conversion_settings"):
        if status.get(key) != reference.get(key):
            raise ValueError(f"Q4 differs from the original campaign: {key}")
    if (status.get("tokenizer_bundle_sha256") != BUNDLE_SHA or
            status.get("q4_vocab_size") != VOCAB_SIZE or
            status.get("q4_max_quantizers") != 4):
        raise ValueError("Wrong Q4 tokenizer bundle/vocabulary")
    if status.get("q4_feature_config") != feature_config_provenance():
        raise ValueError("Missing or mismatched Q4 export feature config")
    for split in ("train", "val"):
        original = reference["outputs"]["q8"][split]
        observed = status["outputs"]["q4"][split]
        # Same shard boundaries/order: each contains the same event identities
        # because the source traversal, split and per-shard RNG are unchanged.
        if [(Path(p["path"]).name, p["rows"]) for p in observed] != [
                (Path(p["path"]).name, p["rows"]) for p in original]:
            raise ValueError(f"Q4 {split} shard boundaries differ from Q8")
        if sum(p["rows"] for p in observed) != status[f"{split}_rows"]:
            raise ValueError("Q4 part row counts do not add up")
        for p in observed:
            if (Path(p["path"]).parts != ("q4", split, Path(p["path"]).name)
                    or p["bytes"] <= 0 or len(p["sha256"]) != 64):
                raise ValueError("Invalid Q4 output record")


def convert(reference, bundle_root, output_dir, input_dir):
    import h5py
    import numpy as np
    import pyarrow as pa
    import torch
    import tokenize_objects_to_grouped_parquet as grouped
    import tokenize_objects_to_parquet_with_atlasopenmagic_metadata as flat
    from tokenize_data_grl_multi_representation import AlignedShardWriter
    from package_final_q4_tokenizer_artifacts import PREPROCESS_SHA256

    policy, args = conversion_options(reference)
    output_dir, input_dir, bundle_root = map(Path, (output_dir, input_dir, bundle_root))
    group_id = reference["group_id"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to mix this conversion with previous output")
    # Preserve the original event/split policy, but use the schema the final
    # Q4 tokenizers actually learned, not the evolving generic training YAML.
    config = flat.load_data_config(str(feature_config_path()))
    collections = flat.collection_map(config["object_collections"])
    checkpoint_map = {name: str(bundle_root / "q4" / f"{name}.ckpt") for name in Q4_SIZES}
    preprocessing = {name: str(bundle_root / "preprocessing" / f"{name}.joblib") for name in Q4_SIZES}
    for name, path in preprocessing.items():
        if sha256(path) != PREPROCESS_SHA256[name]:
            raise ValueError(f"{name}: preprocessing differs from Q1/Q8")
    device = flat.choose_device("cuda")
    models = flat.load_models(checkpoint_map, device)
    validate_q4_models(models)
    preprocessors = flat.load_preprocessors(preprocessing)
    flat.validate_export_inputs(models, collections, list(policy.object_order), preprocessors)
    validate_q4_feature_schema(models, collections, preprocessors)
    vocabulary = grouped.build_grouped_vocabulary(
        models, list(policy.event_token_inputs),
        flat.parse_event_range_map(list(policy.event_token_ranges)), args)
    if vocabulary["vocab_size"] != VOCAB_SIZE or vocabulary["max_quantizers"] != 4:
        raise ValueError("Unexpected final Q4 vocabulary")
    writers = {split: AlignedShardWriter(
        output_dir, group_id=group_id, split=split, shard_rows=policy.shard_rows,
        row_group_rows=policy.row_group_rows, compression=policy.compression,
        seed=policy.seed + offset, representations=("q4",))
        for split, offset in (("train", 1), ("val", 2))}
    digest, source_counts = hashlib.sha256(), {}
    sources = list(reference["source_counts"])
    if sources != sorted(sources):
        raise ValueError("Original source order is not lexicographic")
    input_dir.mkdir(parents=True, exist_ok=True)
    # Stage/process one source at a time. No entire campaign/H5 payload in RAM.
    with torch.inference_mode():
        for i, source in enumerate(sources):
            path = input_dir / f"input-{i:05d}.h5"
            expected = reference["source_counts"][source]
            if expected["status"] != "valid" or expected["available"] != expected["processed"]:
                raise ValueError("Original input is invalid or was capped")
            print(f"Processing {i+1}/{len(sources)}: {source}", flush=True)
            gcloud("cp", source, path)
            if path.stat().st_size != expected["local_input_bytes"]:
                raise ValueError("Input H5 byte size changed since the original campaign")
            with h5py.File(path, "r") as handle:
                events = flat.infer_n_events(handle, config)
                if events != expected["available"]:
                    raise ValueError("Input H5 event count changed")
                meta = flat.sample_metadata_from_handle(handle, source,
                    metadata_source="h5", atlasopenmagic_release="")
                meta["source_file_events"] = events
                policy.validate_source_metadata(source, meta)
                encoded_source = source.encode()
                digest.update(len(encoded_source).to_bytes(8, "little"))
                digest.update(encoded_source)
                train_count = 0
                for start in range(0, events, policy.batch_size):
                    end = min(start + policy.batch_size, events)
                    selection = slice(start, end)
                    encoded = {}
                    for name in policy.object_order:
                        features, mask = flat.read_collection_batch(handle, collections[name],
                            selection, num_objects=None, global_mask_input=config.get("mask_input"))
                        features = flat.apply_object_preprocessing(features, mask,
                            preprocessors[name], name)
                        codes = flat.encode_object_batch(models[name], features, mask, device)
                        encoded[name] = (codes, mask)
                    context = flat.read_event_token_values(handle, vocabulary["event_tokens"], selection)
                    tokens, masks, types = grouped.assemble_grouped_rows(
                        encoded, context, end-start, vocabulary, args)
                    table = grouped.make_grouped_table(tokens, masks, types,
                        start_index=start, source_file=source, sample_metadata=meta,
                        write_legacy_columns=False, vocabulary=vocabulary, extra_columns=None)
                    indices = np.arange(start, end, dtype=np.uint64)
                    train = policy.train_mask(source, indices)
                    train_count += int(train.sum())
                    digest.update(indices.tobytes())
                    digest.update(train.astype(np.uint8).tobytes())
                    writers["train"].add({"q4": table.filter(pa.array(train))})
                    writers["val"].add({"q4": table.filter(pa.array(~train))})
                source_counts[source] = dict(available=events, processed=events,
                    train=train_count, val=events-train_count,
                    local_input_bytes=path.stat().st_size, status="valid")
                if source_counts[source] != expected:
                    raise ValueError(f"Original source split changed: {source}")
            path.unlink()  # Only this job's staged copy; never the GCS source.
    for writer in writers.values():
        writer.finish()
    status = dict(group_id=group_id, source_counts=source_counts,
        membership_sha256=digest.hexdigest(), conversion_settings=policy.manifest_settings(),
        train_rows=writers["train"].written_rows, val_rows=writers["val"].written_rows,
        total_rows=sum(w.written_rows for w in writers.values()),
        tokenizer_bundle_sha256=BUNDLE_SHA, q4_vocab_size=VOCAB_SIZE, q4_max_quantizers=4,
        q4_feature_config=feature_config_provenance(),
        outputs={"q4": {split: writer.files["q4"] for split, writer in writers.items()}})
    validate_status(status, reference)
    write(output_dir / "conversion_status" / f"{group_id}.json", status)
    print(f"PASS {group_id}: original event membership and shard boundaries preserved", flush=True)
    return status


def finalize(reference_dir, status_dir, output_dir):
    reference_dir, status_dir, output_dir = map(Path, (reference_dir, status_dir, output_dir))
    index = read(reference_dir / "index.json")
    expected_groups = set(index["groups"])
    observed_groups = {p.stem for p in status_dir.glob("group-*.json")}
    if observed_groups != expected_groups:
        raise ValueError("Missing or unexpected Q4 conversion groups")
    files, source_counts, digest = {"train": [], "val": []}, {}, hashlib.sha256()
    for group_id in sorted(expected_groups):
        reference = reference_group(reference_dir, group_id)
        status = read(status_dir / f"{group_id}.json")
        validate_status(status, reference)
        if source_counts.keys() & status["source_counts"].keys():
            raise ValueError("Duplicate H5 source between groups")
        source_counts.update(status["source_counts"])
        digest.update(group_id.encode())
        digest.update(status["membership_sha256"].encode())
        for split in files:
            files[split].extend(status["outputs"]["q4"][split])
    if digest.hexdigest() != MEMBERSHIP_SHA:
        raise ValueError("Campaign membership differs from the original Q1/Q8 data")
    manifest = dict(format_version=1, representation="q4", data_domain="real_data_only",
        seed=42, train_fraction_requested=0.9,
        split_method="blake2b(source_uri) xor event_index, then splitmix64",
        inventory_sha256=index["inventory_sha256"], membership_sha256=digest.hexdigest(),
        tokenizer_bundle_sha256=BUNDLE_SHA, input_file_count=len(source_counts),
        q4_feature_config=feature_config_provenance(),
        valid_input_file_count=len(source_counts), invalid_input_file_count=0,
        invalid_sources=[], group_count=len(expected_groups), source_counts=source_counts,
        train_rows=sum(p["rows"] for p in files["train"]),
        val_rows=sum(p["rows"] for p in files["val"]),
        train_shards=len(files["train"]), val_shards=len(files["val"]), files=files,
        train_bytes=sum(p["bytes"] for p in files["train"]),
        val_bytes=sum(p["bytes"] for p in files["val"]),
        settings=dict(max_seq_length=256, q4_vocab_size=VOCAB_SIZE, q4_max_quantizers=4))
    manifest["total_rows"] = manifest["train_rows"] + manifest["val_rows"]
    validate_manifest(manifest)
    path = output_dir / "manifest.json"
    write(path, manifest)
    (output_dir / "manifest.sha256").write_text(sha256(path) + "\n")
    print("PASS all 16 Q4 groups; 216,095,647 events and original split preserved")
    return manifest


def validate_manifest(manifest):
    for key, value in TOTALS.items():
        if manifest.get(key) != value:
            raise ValueError(f"Q4 manifest {key} differs: {manifest.get(key)} != {value}")
    if (manifest.get("membership_sha256") != MEMBERSHIP_SHA or
            manifest.get("tokenizer_bundle_sha256") != BUNDLE_SHA or
            manifest.get("representation") != "q4"):
        raise ValueError("Wrong Q4 dataset provenance")
    if manifest.get("q4_feature_config") != feature_config_provenance():
        raise ValueError("Wrong Q4 feature-config provenance")


def validate_prepared(prepared_dir):
    import pyarrow.parquet as pq
    prepared_dir = Path(prepared_dir)
    manifest = read(prepared_dir / "manifest.json")
    validate_manifest(manifest)
    vocabulary = None
    for split in ("train", "val"):
        records = manifest["files"][split]
        if {p.name for p in (prepared_dir / split).glob("*.parquet")} != {
                Path(r["path"]).name for r in records}:
            raise ValueError("Local shard files differ from the complete manifest")
        for record in records:
            path = prepared_dir / split / Path(record["path"]).name
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"Q4 shard checksum mismatch: {path.name}")
            parquet = pq.ParquetFile(path)
            observed = json.loads(parquet.schema_arrow.metadata[b"heptokens_token_vocabulary"])
            if vocabulary is None:
                vocabulary = observed
            if observed != vocabulary or observed["vocab_size"] != VOCAB_SIZE or observed["max_quantizers"] != 4:
                raise ValueError("Q4 shard vocabulary mismatch")
            if parquet.metadata.num_rows != record["rows"]:
                raise ValueError("Q4 shard row count mismatch")
    print("PASS every staged Q4 shard SHA, row count and vocabulary")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("convert", "finalize", "validate-prepared"))
    p.add_argument("--reference-dir", type=Path)
    p.add_argument("--group-id")
    p.add_argument("--bundle-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--status-dir", type=Path)
    p.add_argument("--prepared-dir", type=Path)
    args = p.parse_args()
    if args.mode == "convert":
        convert(reference_group(args.reference_dir, args.group_id), args.bundle_root,
                args.output_dir, args.input_dir)
    elif args.mode == "finalize":
        finalize(args.reference_dir, args.status_dir, args.output_dir)
    else:
        validate_prepared(args.prepared_dir)


if __name__ == "__main__":
    main()
