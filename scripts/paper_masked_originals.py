"""Attach aligned original inputs to saved Figure 6 predictions, on CPU only."""

from pathlib import Path
import json

import joblib
import numpy as np
import pyarrow.parquet as pq

import paper_masked_object_reconstruction as paper

IDENTITY_COLUMNS = ("source_file", "event_index", "mask", "type_ids")


def check_inputs(args):
    groups = [sorted((args.prepared_root / q / "val").glob("*.parquet")) for q in ("q8", "q1")]
    if not groups[0] or [p.name for p in groups[0]] != [p.name for p in groups[1]]:
        raise ValueError("Q8/Q1 downloaded validation shard names differ or are empty")
    records = [[paper.capacity.file_record(path) for path in paths] for paths in groups]
    count = 0
    for offset, batch in aligned_batches(
        *(batches(group, args.n_events, list(IDENTITY_COLUMNS)) for group in records)
    ):
        count = offset + batch.num_rows
    if count < args.n_events:
        raise ValueError(
            f"Only {count:,} aligned events available; requested {args.n_events:,}. No inference started"
        )
    paper.log.info("INPUTS OK: %s aligned Q8/Q1 validation events; no inference", count)


def batches(records, limit, columns):
    for record in records:
        paper.capacity.verify_file(record)
        if limit <= 0:
            break
        for batch in pq.ParquetFile(record["path"]).iter_batches(batch_size=128, columns=columns):
            count = min(limit, batch.num_rows)
            yield batch.slice(0, count)
            limit -= count
            if limit <= 0:
                break


def aligned_batches(left, right):
    """Compare the actual ordered event/object identities, across shard boundaries."""
    left, right = iter(left), iter(right)
    a, b = next(left, None), next(right, None)
    offset = 0
    while a is not None and b is not None:
        count = min(a.num_rows, b.num_rows)
        for column in IDENTITY_COLUMNS:
            if not a.column(column).slice(0, count).equals(b.column(column).slice(0, count)):
                raise ValueError(f"Input identity mismatch: {column} near event row {offset}")
        yield offset, b.slice(0, count)
        offset += count
        a = a.slice(count) if count < a.num_rows else next(left, None)
        b = b.slice(count) if count < b.num_rows else next(right, None)
    if a is not None or b is not None or offset == 0:
        raise ValueError("Input event prefixes differ in length or are empty")


def verify_original(directory, plan, obj):
    path = directory / "original_arrays" / f"{obj}.npz"
    receipt = paper.reco.read_sealed(path.with_suffix(".json"), "receipt_id")
    decoded, decoded_receipt = paper.verify_receipt(directory, plan, obj)
    if (
        receipt["plan_id"] != plan["plan_id"]
        or receipt["decoded_receipt_id"] != decoded_receipt["receipt_id"]
    ):
        raise ValueError(f"Original inputs belong to a different evaluation: {obj}")
    paper.capacity.verify_file(receipt["arrays"])
    with np.load(path, allow_pickle=False) as data, np.load(decoded, allow_pickle=False) as pred:
        if not np.array_equal(data["positions"], pred["positions"]):
            raise ValueError(f"Original input positions differ: {obj}")
        if data["feature_names"].tolist() != pred["feature_names"].tolist():
            raise ValueError(f"Original input feature order differs: {obj}")
        original = data["original"]
        if original.shape != pred["reference"].shape or not np.isfinite(original).all():
            raise ValueError(f"Missing/nonfinite original inputs: {obj}")
    return original, receipt


def attach(args):
    # Validate both predictions before using their shared masked-object selection.
    paper.audit_pair(args)
    directories = (args.q8_dir.resolve(), args.q1_dir.resolve())
    plans = [paper.reco.read_sealed(path / paper.PLAN, "plan_id") for path in directories]
    paths = sorted(args.continuous_dir.resolve().glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No aligned continuous validation shards: {args.continuous_dir}")
    records = [paper.capacity.file_record(path, content_hash=True) for path in paths]
    schema = None
    for path in paths:
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        current = json.loads(metadata.get(b"heptokens_continuous_schema", b"null"))
        if (
            not current
            or current.get("preprocessing", {}).get("objects")
            != "matching fitted VQ-VAE input transformer"
        ):
            raise ValueError(f"Missing/unrecognized original-feature preprocessing schema: {path}")
        if schema is not None and current != schema:
            raise ValueError(f"Continuous schemas differ: {path}")
        schema = current
    states = {}
    for obj in paper.OBJECTS:
        specs = [plan["tokenizers"][obj] for plan in plans]
        for spec in specs:
            paper.capacity.verify_file(spec["preprocessor"])
        if specs[0]["preprocessor"]["sha256"] != specs[1]["preprocessor"]["sha256"]:
            raise ValueError(f"Q8/Q1 original preprocessors differ: {obj}")
        names = specs[0]["feature_names"]
        obj_schema = schema["objects"][obj]
        if obj_schema["feature_names"] != names or obj_schema["feature_count"] != len(names):
            raise ValueError(f"Continuous feature order differs: {obj}")
        if obj_schema["type_id"] != plans[0]["vocabulary"]["objects"][obj]["type_id"]:
            raise ValueError(f"Continuous object type differs: {obj}")
        decoded, _ = paper.verify_receipt(directories[0], plans[0], obj)
        with np.load(decoded, allow_pickle=False) as data:
            positions = data["positions"]
        order = np.argsort(positions[:, 0], kind="stable")
        states[obj] = dict(
            positions=positions,
            order=order,
            rows=positions[order, 0],
            names=names,
            values=np.empty((len(positions), len(names)), dtype=np.float64),
            found=np.zeros(len(positions), dtype=bool),
        )
    columns = [*IDENTITY_COLUMNS, "continuous_features", "continuous_feature_mask"]
    limit = plans[0]["n_events"]
    for offset, batch in aligned_batches(
        batches(plans[0]["shards"], limit, list(IDENTITY_COLUMNS)),
        batches(records, limit, columns),
    ):
        for obj, state in states.items():
            order = state["order"]
            rows = state["rows"]
            start, stop = np.searchsorted(rows, [offset, offset + batch.num_rows])
            for index in order[start:stop]:
                event, position = state["positions"][index]
                row = int(event - offset)
                n = len(state["names"])
                if (
                    not batch.column("mask")[row].as_py()[position]
                    or batch.column("type_ids")[row].as_py()[position]
                    != schema["objects"][obj]["type_id"]
                ):
                    raise ValueError(f"Wrong original object at {(event, position)} for {obj}")
                values = batch.column("continuous_features")[row].values[int(position)].as_py()[:n]
                valid = (
                    batch.column("continuous_feature_mask")[row].values[int(position)].as_py()[:n]
                )
                if len(values) != n or len(valid) != n or not all(valid):
                    raise ValueError(f"Missing original features for {obj} at {(event, position)}")
                state["values"][index] = values
                state["found"][index] = True
    for record in records:
        paper.capacity.verify_file(record)
    for obj, state in states.items():
        if not state["found"].all():
            raise ValueError(f"Continuous shards do not cover all saved masked {obj}")
        transformer = joblib.load(plans[0]["tokenizers"][obj]["preprocessor"]["path"])
        original = np.asarray(transformer.inverse_transform(state["values"]))
        if original.shape != state["values"].shape or not np.isfinite(original).all():
            raise ValueError(f"Original inverse preprocessing produced invalid values: {obj}")
        for directory, plan in zip(directories, plans):
            _, decoded_receipt = paper.verify_receipt(directory, plan, obj)
            provenance = dict(
                plan_id=plan["plan_id"],
                object=obj,
                decoded_receipt_id=decoded_receipt["receipt_id"],
                continuous_shards=records,
                continuous_schema=schema,
                preprocessor=plan["tokenizers"][obj]["preprocessor"],
                meaning="Original preprocessed inputs, inverse-transformed with the export joblib; no VQ decode",
            )
            path = directory / "original_arrays" / f"{obj}.npz"
            if path.exists() or path.with_suffix(".json").exists():
                saved, receipt = verify_original(directory, plan, obj)
                if any(receipt.get(k) != v for k, v in provenance.items()) or not np.array_equal(
                    saved, original
                ):
                    raise ValueError(f"Original inputs changed; not overwriting {path}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                original=original,
                positions=state["positions"],
                feature_names=state["names"],
            )
            temporary.replace(path)
            paper.reco.write_json(
                path.with_suffix(".json"),
                paper.reco.seal(
                    {**provenance, "arrays": paper.capacity.file_record(path, content_hash=True)},
                    "receipt_id",
                ),
            )
        paper.log.info(
            "Original %s: %s identical Q8/Q1 masked inputs; no inference", obj, len(original)
        )
