#!/usr/bin/env python3
"""Rebuild flat Paper0 features from original H5 for the sealed selected Q1 cohort."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import tokenize_objects_to_parquet_with_atlasopenmagic_metadata as legacy
from heptokens.data.continuous_schema import CONTINUOUS_SCHEMA_METADATA_KEY, ROLE_IDS, build_continuous_schema
from package_final_q4_tokenizer_artifacts import PREPROCESS_SHA256
from prepare_hzz_4l_multiclass import SPLITS, new_output, sha256, write_json
from prepare_paper0_atlas_hzz_selected_q1 import CLASSES, SELECTION_ID


Q1_MANIFEST_SHA = "eeb070d1150b12890f74384c63147df62e2c771b93866ede21e5c6b120be178f"
SEQ_LEN = 256
COMPACT = 96  # Verified against all actual positions; never silently truncate.
VOCAB_KEY = b"heptokens_token_vocabulary"
STRUCTURAL_KEYS = ("includes_cls", "object_order", "event_tokens", "sequence_layout", "special_tokens")
CONTINUOUS_COLUMNS = ("continuous_features", "continuous_feature_mask", "position_role_ids")
ARCHIVE_FLOAT32_ATOL = 1e-5


def reference_inputs(ref_path, q1_path, config_path, preprocessing_dir):
    old = pq.ParquetFile(ref_path).schema_arrow.metadata
    q1 = pq.ParquetFile(q1_path).schema_arrow.metadata
    vocabulary, vq1 = (json.loads(m[VOCAB_KEY]) for m in (old, q1))
    if (any(vocabulary[key] != vq1[key] for key in STRUCTURAL_KEYS)
            or vocabulary["includes_cls"] is not True or vocabulary["sequence_layout"] != "one_position_per_object"):
        raise ValueError("Archived Paper0 continuous layout does not match the original Q1 structure")
    legacy_schema = json.loads(old[CONTINUOUS_SCHEMA_METADATA_KEY])
    if legacy_schema.get("feature_columns", {}).get("raw") != "continuous_features":
        raise ValueError("Archived Paper0 continuous feature column differs")
    schema = deepcopy(legacy_schema)
    schema["version"] = 1
    schema["decoded_q8"] = None
    schema["feature_columns"].pop("decoded_q8", None)
    config = legacy.load_data_config(str(config_path))
    collections = legacy.collection_map(config["object_collections"])
    derived = build_continuous_schema(collections=collections, object_order=vocabulary["object_order"],
        event_token_specs=vocabulary["event_tokens"], type_ids=legacy.TYPE_IDS, include_decoded_q8=False)
    if any(derived[k] != schema[k] for k in ("max_feature_dim", "role_ids", "event_inputs", "feature_columns", "preprocessing")):
        raise ValueError("Continuous feature width, scaling, or roles differ from the original Paper0 archive")
    if set(derived["objects"]) != set(schema["objects"]) or any(
        any(derived["objects"][name][key] != schema["objects"][name][key]
            for key in ("type_id", "feature_count", "feature_names")) for name in derived["objects"]
    ):
        raise ValueError("Continuous object feature order differs from the original Paper0 archive")
    # Preserve the archived flat metadata for the model; only drop the unused
    # decoded-Q8 column declaration. Current hierarchical grouping is irrelevant.
    paths = {name: str(preprocessing_dir / f"{name}.joblib") for name in vocabulary["object_order"]}
    if set(paths) != set(PREPROCESS_SHA256) or any(sha256(p) != PREPROCESS_SHA256[n] for n, p in paths.items()):
        raise ValueError("Fitted preprocessing differs from the original export bundle")
    transformers = legacy.load_preprocessors(paths)
    for name, transformer in transformers.items():
        size = getattr(transformer, "n_features_in_", None)
        if size is None:
            size = getattr(getattr(transformer, "final_transformer", None), "n_features_in_", None)
        if size != schema["objects"][name]["feature_count"]:
            raise ValueError(f"Fitted feature dimension differs for {name}")
    return vocabulary, schema, config, collections, transformers


def assemble_slice(handle, slc, chosen, recipe, width=COMPACT):
    """Vectorized layout equivalent to the archived grouped exporter."""
    vocabulary, schema, config, collections, transformers = recipe
    rows = np.arange(len(chosen))
    features = np.zeros((len(chosen), width, schema["max_feature_dim"]), dtype=np.float32)
    roles = np.zeros((len(chosen), width), dtype=np.uint8)
    types = np.zeros_like(roles)
    special = vocabulary["special_tokens"]
    structure = np.full(roles.shape, special["pad"], dtype=np.uint16)
    cursor = np.ones(len(chosen), dtype=np.int64)
    roles[:, 0], structure[:, 0] = ROLE_IDS["cls"], special["cls"]
    values = legacy.read_normalized_event_values(handle, vocabulary["event_tokens"], slc)[chosen]
    context_tokens = legacy.read_event_token_values(handle, vocabulary["event_tokens"], slc)[chosen]
    for j, spec in enumerate(vocabulary["event_tokens"]):
        roles[:, j + 1], types[:, j + 1] = ROLE_IDS["event"], spec["type_id"]
        features[:, j + 1, 0], structure[:, j + 1] = values[:, j], context_tokens[:, j]
    cursor += len(vocabulary["event_tokens"])
    for name in vocabulary["object_order"]:
        obj, active = legacy.read_collection_batch(handle, collections[name], slc,
            num_objects=None, global_mask_input=config.get("mask_input"))
        obj, active = obj[chosen], active[chosen]
        obj = legacy.apply_object_preprocessing(obj, active, transformers[name], name)
        active = active.numpy()
        counts = active.sum(axis=1)
        if np.any(cursor + counts >= width):
            raise ValueError(f"Unexpected sequence length for {name}; refusing to truncate")
        r, c = np.nonzero(active)
        positions = (cursor[:, None] + np.cumsum(active, axis=1) - 1)[r, c]
        values = obj.numpy()[r, c]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite original fitted features: {name}")
        features[r, positions, :values.shape[-1]] = values
        roles[r, positions], types[r, positions] = ROLE_IDS["object"], schema["objects"][name]["type_id"]
        cursor += counts
        roles[rows, cursor], structure[rows, cursor] = ROLE_IDS["separator"], special["sep"]
        cursor += 1
    return features, roles, types, structure


def feature_mask(roles, types, schema):
    dims = np.zeros(roles.shape, dtype=np.uint8)
    dims[roles == ROLE_IDS["event"]] = 1
    for spec in schema["objects"].values():
        dims[(roles == ROLE_IDS["object"]) & (types == spec["type_id"])] = spec["feature_count"]
    return np.arange(schema["max_feature_dim"]) < dims[..., None]


def dense(batch, name, *sizes):
    a = batch[name]
    for width in sizes:
        if not pa.types.is_fixed_size_list(a.type) or a.type.list_size != width or a.null_count:
            raise ValueError(f"Bad sequence shape or nulls in {name}")
        a = a.flatten()
    if a.null_count:
        raise ValueError(f"Null scalar values in {name}")
    return a.to_numpy(zero_copy_only=False).reshape(len(batch), *sizes)


def arrow_sequence(values):
    result = pa.array(np.ascontiguousarray(values).reshape(-1))
    for width in reversed(values.shape[1:]):
        result = pa.FixedSizeListArray.from_arrays(result, width)
    return result


def expand(x, fill=0):
    result = np.full((len(x), SEQ_LEN, *x.shape[2:]), fill, dtype=x.dtype)
    result[:, :x.shape[1]] = x
    return result


def verify_layout(batch, roles, types, structure, quantizers, pad):
    masks = dense(batch, "mask", SEQ_LEN)
    original_types = dense(batch, "type_ids", SEQ_LEN)
    if not np.array_equal(masks, roles != ROLE_IDS["padding"]) or not np.array_equal(original_types, types):
        raise ValueError("H5 raw masks/type positions differ from the historical event sequence")
    non_objects = roles != ROLE_IDS["object"]
    tokens = dense(batch, "tokens", SEQ_LEN, quantizers)
    if not np.array_equal(tokens[:, :, 0][non_objects], structure[non_objects]):
        raise ValueError("H5 raw CLS/context/separator/padding differs from historical tokens")
    if quantizers > 1 and not np.all(tokens[:, :, 1:][non_objects] == pad):
        raise ValueError("Historical non-object position has unexpected residual tokens")


def load_membership(path, manifest):
    if sha256(path) != manifest["membership_sha256"]:
        raise ValueError("Selected Q1 membership hash differs")
    with np.load(path, allow_pickle=False) as archive:
        m = dict(archive)
    n = manifest["selected_identity_count"]
    names = m["sources"].tolist()
    if set(names) != set(manifest["h5_sources"]) or len(names) != len(set(names)) or any(Path(p).name != p for p in names):
        raise ValueError("Unexpected selected H5 source dictionary")
    if (any(a.shape != (n,) for key, a in m.items() if key != "sources")
            or not np.isin(m["source_id"], np.arange(len(names))).all()):
        raise ValueError("Selected membership arrays differ in shape/source domain")
    local_index = {}
    for source_id, name in enumerate(names):
        spec = manifest["h5_sources"][name]
        slots = np.flatnonzero(m["source_id"] == source_id)
        local = m["h5_row"][slots]
        if (np.any(np.diff(local) <= 0) or np.any(local < 0) or np.any(local >= spec["selected_events"])
                or np.any(local + spec["original_q1_event_index_offset"] != m["event_index"][slots])):
            raise ValueError(f"Selected membership's historical raw-H5 bridge differs: {name}")
        lookup = np.full(spec["selected_events"], -1, dtype=np.int64)
        lookup[local] = slots
        local_index[name] = lookup
    return m, local_index


def prepare_cache(cache, manifest, m, local_index, h5_dir, recipe):
    n = manifest["selected_identity_count"]
    width = recipe[1]["max_feature_dim"]
    features = np.memmap(cache / "features.f32", mode="w+", dtype=np.float32, shape=(n, COMPACT, width))
    roles = np.zeros((n, COMPACT), dtype=np.uint8)
    types = np.zeros_like(roles)
    structure = np.zeros_like(roles, dtype=np.uint16)
    produced = np.zeros(n, dtype=bool)
    for name, lookup in local_index.items():
        spec = manifest["h5_sources"][name]
        path = h5_dir / name
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"Original H5 input changed since selected membership freeze: {name}")
        with h5py.File(path, "r") as h5:
            for start in range(0, len(lookup), 4096):
                stop = min(len(lookup), start + 4096)
                chosen = np.flatnonzero(lookup[start:stop] >= 0)
                if not len(chosen):
                    continue
                idx = lookup[start + chosen]
                raw_event = h5["atlas/event/event_number"][start:stop][chosen]
                raw_run = h5["atlas/event/run_number"][start:stop][chosen]
                if (not np.array_equal(raw_event, m["event_number"][idx]) or not np.array_equal(raw_run, m["run_number"][idx])
                        or produced[idx].any()):
                    raise ValueError(f"H5 event audit key changed: {name}")
                features[idx], roles[idx], types[idx], structure[idx] = assemble_slice(h5, slice(start, stop), chosen, recipe)
                produced[idx] = True
        print(f"PASS rebuilt frozen H5 features for {name}: {np.count_nonzero(lookup >= 0):,} rows", flush=True)
    if not produced.all():
        raise ValueError("Frozen membership is missing rebuilt features")
    features.flush()
    return features, roles, types, structure


def verify_old_values(ref_path, manifest, schema, local_index, cache, vocabulary):
    """Compare every overlapping archived background event, including real feature values."""
    f = pq.ParquetFile(ref_path)
    cols = ("source_file", "event_index", "dsid", "tokens", "mask", "type_ids", *CONTINUOUS_COLUMNS)
    compared, max_delta = 0, 0.0
    seen = set()
    for batch in f.iter_batches(batch_size=1024, columns=cols):
        names = np.asarray(batch["source_file"].to_pylist())
        event = batch["event_index"].to_numpy()
        dsid = batch["dsid"].to_numpy()
        idx = np.full(len(batch), -1, dtype=np.int64)
        for source in np.unique(names):
            use = np.flatnonzero(names == source)
            spec = manifest["h5_sources"][source]
            # The original background DSID was exported as the first file
            # in both archives. Any unrelated control must provide its own bridge.
            if spec["original_q1_event_index_offset"] != 0 or spec["class_name"] != "background" or not np.all(dsid[use] == 700600):
                raise ValueError("Require the archived Paper0 paired background reference shard")
            if np.any(event[use] < 0) or np.any(event[use] >= len(local_index[source])):
                raise ValueError("Archive reference event IDs do not map to the original H5")
            idx[use] = local_index[source][event[use]]
        keep = idx >= 0
        if not keep.any():
            continue
        idx = idx[keep]
        if any(int(i) in seen for i in idx) or len(np.unique(idx)) != len(idx):
            raise ValueError("Archive reference repeats event identities")
        seen.update(map(int, idx))
        filtered = batch.filter(pa.array(keep))
        vals, role, typ, structural = [expand(a[idx]) for a in cache]
        verify_layout(filtered, role, typ, structural, vocabulary["max_quantizers"], vocabulary["special_tokens"]["pad"])
        archived = dense(filtered, "continuous_features", SEQ_LEN, schema["max_feature_dim"])
        if (not np.array_equal(role, dense(filtered, "position_role_ids", SEQ_LEN))
                or not np.array_equal(feature_mask(role, typ, schema), dense(filtered, "continuous_feature_mask", SEQ_LEN, schema["max_feature_dim"]))):
            raise ValueError("Original continuous archive feature validity/roles differ from rebuilt view")
        max_delta = max(max_delta, float(np.max(np.abs(vals - archived))))
        if not np.isfinite(archived).all() or max_delta > ARCHIVE_FLOAT32_ATOL:
            raise ValueError(f"Original continuous archive values differ from rebuilt view (max absolute difference {max_delta})")
        compared += len(idx)
    if not compared:
        raise ValueError("No overlapping events in archived flat Paper0 reference")
    return dict(reference_parquet_sha256=sha256(ref_path), verified_overlap_rows=compared,
                feature_values="float32 comparison at strict absolute tolerance; relative tolerance zero",
                absolute_tolerance=ARCHIVE_FLOAT32_ATOL, relative_tolerance=0.0, max_absolute_difference=max_delta,
                exact_match=["event IDs", "mask", "type_ids", "non-object token IDs", "feature mask", "roles"])


def add_columns(batch, idx, recipe, cache):
    vocabulary, schema, *_ = recipe
    values, roles, types, structure = [expand(a[idx]) for a in cache]
    verify_layout(batch, roles, types, structure, 1, vocabulary["special_tokens"]["pad"])
    table = pa.Table.from_batches([batch])
    if any(key in table.column_names for key in CONTINUOUS_COLUMNS):
        raise ValueError("Will not replace existing Q1 columns")
    for key, array in zip(CONTINUOUS_COLUMNS, (values, feature_mask(roles, types, schema), roles.astype(np.int64)), strict=True):
        table = table.append_column(key, arrow_sequence(array))
    meta = dict(table.schema.metadata or {})
    meta[CONTINUOUS_SCHEMA_METADATA_KEY] = json.dumps(schema, sort_keys=True).encode()
    return table.replace_schema_metadata(meta)


def materialize(q1, partial, manifest, m, local_index, recipe, cache):
    seen = np.zeros(manifest["selected_identity_count"], dtype=bool)
    shards, counts = {}, Counter()
    for rel, expected_sha in manifest["shards"].items():
        relative = Path(rel)
        if len(relative.parts) != 3 or relative.parts[0] not in SPLITS or relative.parts[1] not in CLASSES:
            raise ValueError(f"Unexpected selected parent shard path: {rel}")
        split, label = SPLITS.index(relative.parts[0]), CLASSES[relative.parts[1]][0]
        source, target = q1 / relative, partial / relative
        if sha256(source) != expected_sha:
            raise ValueError(f"Selected Q1 shard SHA differs: {rel}")
        parent = pq.ParquetFile(source)
        vocab = json.loads(parent.schema_arrow.metadata[VOCAB_KEY])
        if any(vocab[key] != recipe[0][key] for key in STRUCTURAL_KEYS):
            raise ValueError(f"Selected Q1 shard structure differs: {rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        try:
            for b in parent.iter_batches(batch_size=256):
                names = np.asarray(b["source_file"].to_pylist())
                rows = b["h5_row"].to_numpy()
                idx = np.full(len(b), -1, dtype=np.int64)
                for name in np.unique(names):
                    use = names == name
                    lookup = local_index[name]
                    if np.any(rows[use] < 0) or np.any(rows[use] >= len(lookup)):
                        raise ValueError(f"Selected Q1 raw H5 row is out of range: {rel}")
                    idx[use] = lookup[rows[use]]
                if (np.any(idx < 0) or seen[idx].any() or len(np.unique(idx)) != len(idx)
                        or np.any(m["split"][idx] != split) or np.any(m["labels"][idx] != label)
                        or np.any(b["label"].to_numpy() != label)
                        or np.any(b["dsid"].to_numpy() != CLASSES[relative.parts[1]][1])
                        or any(not np.array_equal(m[k][idx], b[k].to_numpy(zero_copy_only=False))
                               for k in ("event_index", "h5_row", "event_number", "run_number", "m4l", "channel"))):
                    raise ValueError(f"Selected event identity/metadata/split drift: {rel}")
                seen[idx] = True
                t = add_columns(b, idx, recipe, cache)
                if writer is None:
                    writer = pq.ParquetWriter(target, t.schema, compression="zstd")
                writer.write_table(t, row_group_size=128)
                counts[(relative.parts[0], relative.parts[1])] += len(b)
        finally:
            if writer is not None:
                writer.close()
        if pq.ParquetFile(target).metadata.num_rows != parent.metadata.num_rows:
            raise ValueError(f"Output row count differs: {rel}")
        shards[rel] = sha256(target)
        print(f"PASS Q1 identity/metadata and frozen continuous layout: {rel}", flush=True)
    if not seen.all() or any(counts[(split, name)] != manifest["split_counts"][split][name] for split in SPLITS for name in CLASSES):
        raise ValueError("Incomplete selected continuous view")
    return shards


def prepare(q1, h5_dir, output, preprocessing_dir, config_path, ref_path):
    q1, h5_dir, output, preprocessing_dir, config_path, ref_path = map(Path, (q1, h5_dir, output, preprocessing_dir, config_path, ref_path))
    parent_sha = sha256(q1 / "manifest.json")
    if parent_sha != Q1_MANIFEST_SHA:
        raise ValueError("Require exact frozen selected Paper0 Q1 manifest")
    manifest = json.loads((q1 / "manifest.json").read_text())
    if (manifest["selection_id"] != SELECTION_ID or manifest["identity_overlap_detected"] is not False
            or manifest["token_shape"] != [SEQ_LEN, 1] or manifest["n_classes"] != 2
            or set(manifest["shards"]) != {p.relative_to(q1).as_posix() for p in q1.rglob("*.parquet")}):
        raise ValueError("Selected Q1 parent does not match the sealed study")
    m, local_index = load_membership(q1 / "membership.npz", manifest)
    recipe = reference_inputs(ref_path, q1 / next(iter(manifest["shards"])), config_path, preprocessing_dir)
    target, partial = new_output(output, q1, h5_dir, preprocessing_dir)
    with TemporaryDirectory(prefix="paper0-continuous-cache-", dir=partial.parent) as temp:
        cache = prepare_cache(Path(temp), manifest, m, local_index, h5_dir, recipe)
        parity = verify_old_values(ref_path, manifest, recipe[1], local_index, cache, recipe[0])
        print(f"PASS legacy flat continuous feature parity: {parity}", flush=True)
        shards = materialize(q1, partial, manifest, m, local_index, recipe, cache)
        del cache
    shutil.copyfile(q1 / "membership.npz", partial / "membership.npz")
    if sha256(partial / "membership.npz") != manifest["membership_sha256"]:
        raise ValueError("Copied membership differs")
    result = deepcopy(manifest)
    result["shards"] = shards
    result["continuous_preparation"] = dict(source_selected_q1_manifest_sha256=parent_sha,
        source_selected_q1_shards=manifest["shards"], preprocessing_sha256=PREPROCESS_SHA256,
        feature_config_sha256=sha256(config_path), legacy_raw_continuous_parity=parity,
        schema=recipe[1], inputs=list(CONTINUOUS_COLUMNS),
        source="original H5, frozen object preprocessing, original context ranges and sequence positions", no_refit=True)
    write_json(partial / "manifest.json", result)
    partial.rename(target)
    print(f"PASS selected continuous dataset {target}; manifest {sha256(target / 'manifest.json')}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("selected-q1-dir", "h5-dir", "output-dir", "preprocessing-dir", "feature-config", "legacy-reference-parquet"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    a = parser.parse_args()
    torch.set_num_threads(1)
    prepare(a.selected_q1_dir, a.h5_dir, a.output_dir, a.preprocessing_dir, a.feature_config, a.legacy_reference_parquet)


if __name__ == "__main__":
    main()
