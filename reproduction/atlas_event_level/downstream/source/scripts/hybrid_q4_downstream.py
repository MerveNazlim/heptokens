#!/usr/bin/env python3
"""Matched hybrid/Q4 downstream checks and test export.

Only the shared backbone is transferred. No hybrid pretraining head is used.
Class definitions and weights come from the pinned classification manifest.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from heptokens.models.foundation_grouped_cls_classifier import (
    LitGroupedCLSClassifier, _load_checkpoint_state,
)

SPLITS = ("train", "val", "test")
BACKBONES = {
    "hybrid": dict(max_quantizers=8, quantizer_embedding_dim=32, vocab_size=131588),
    "q4": dict(max_quantizers=4, quantizer_embedding_dim=64, vocab_size=262660),
}
COMMON = dict(hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1,
              max_seq_length=256, num_type_ids=12, use_type_embedding=True,
              use_position_embedding=True, object_projection_mode="identity")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_numpy(array, dtype):
    shape, values = [len(array)], array
    while pa.types.is_fixed_size_list(values.type):
        if values.null_count:
            raise ValueError("Null sequence values")
        shape.append(values.type.list_size)
        values = values.flatten()
    if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
        return np.asarray(array.to_pylist(), dtype=dtype)
    if values.null_count:
        raise ValueError("Null values")
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=dtype).reshape(shape)


def classes(manifest):
    if manifest.get("classes"):
        ordered = sorted(manifest["classes"], key=lambda item: int(item["label"]))
        if [int(item["label"]) for item in ordered] != list(range(len(ordered))):
            raise ValueError("Class labels are not contiguous")
        result = [(str(item["name"]), int(item["label"])) for item in ordered]
    else:
        result = [(name, int(manifest[name]["label"])) for name in ("background", "signal")]
    if len(result) < 2 or len({name for name, _ in result}) != len(result):
        raise ValueError("Invalid class definitions")
    if sorted(label for _, label in result) != list(range(len(result))):
        raise ValueError("Class labels must be unique 0..N-1")
    if int(manifest.get("n_classes", len(result))) != len(result):
        raise ValueError("Manifest class count differs from labels")
    return result


def validate_data(prepared, manifest_hash, representation, report_dir):
    prepared, report_dir = Path(prepared), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared / "manifest.json"
    if sha256(manifest_path) != manifest_hash:
        raise ValueError("Classification manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    definitions = classes(manifest)
    declared_dsids = {str(item['name']): set(map(int, item.get('dsids', [])))
                      for item in manifest.get('classes', [])}
    weights = manifest.get('recommended_cross_entropy_weights')
    if weights:
        train = manifest['split_counts']['train']
        expected_weights = [train['total'] / (len(definitions) * train[name]) for name, _ in definitions]
        if not np.allclose([weights[name] for name, _ in definitions], expected_weights, rtol=1e-10):
            raise ValueError('Class weights differ from inverse-frequency Zephyr training weights')
    if manifest.get("identity_overlap_detected") is not False:
        raise ValueError("Manifest does not assert disjoint splits")
    q = BACKBONES[representation]["max_quantizers"]
    if manifest.get("token_shape") != [256, q]:
        raise ValueError(f"Expected [256,{q}] inputs, not {manifest.get('token_shape')}")
    # Check identities, not just the manifest's assertion. Disk-backed to keep RAM bounded.
    database = report_dir / "classification_identity_check.sqlite"
    if database.exists():
        raise FileExistsError(database)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE identities (source TEXT, event INTEGER, PRIMARY KEY(source,event))")
    vocabulary, inventory, membership = None, {}, {}
    try:
        for split in SPLITS:
            split_rows = 0
            for name, label in definitions:
                paths = sorted((prepared / split / name).glob("*.parquet"))
                if not paths:
                    raise FileNotFoundError(f"Missing {split}/{name} shards")
                rows, digest = 0, hashlib.sha256()
                for path in paths:
                    parquet = pq.ParquetFile(path)
                    required = {"tokens", "mask", "type_ids", "label", "source_file", "event_index"}
                    if declared_dsids.get(name):
                        required.add('dsid')
                    if required - set(parquet.schema_arrow.names):
                        raise ValueError(f"Missing input columns in {path}")
                    encoded = (parquet.schema_arrow.metadata or {}).get(b"heptokens_token_vocabulary")
                    if encoded is None:
                        raise ValueError(f"Missing token vocabulary in {path}")
                    observed = json.loads(encoded)
                    if vocabulary is None:
                        vocabulary = observed
                    if observed != vocabulary:
                        raise ValueError("Different token vocabularies among shards")
                    for batch in parquet.iter_batches(batch_size=4096, columns=sorted(required)):
                        column = lambda key: batch.column(batch.schema.get_field_index(key))
                        labels = np.asarray(column("label").to_numpy(), dtype=np.int64)
                        sources = column("source_file").to_pylist()
                        indices = np.asarray(column("event_index").to_numpy(), dtype=np.int64)
                        if not np.all(labels == label) or any(s is None for s in sources) or np.any(indices < 0):
                            raise ValueError(f"Invalid labels/identities in {path}")
                        if declared_dsids.get(name) and set(map(int, column('dsid').to_pylist())) - declared_dsids[name]:
                            raise ValueError(f'Wrong DSID for {name} in {path}')
                        tokens = sequence_numpy(column("tokens"), np.int64)
                        mask = sequence_numpy(column("mask"), bool)
                        if tokens.shape[1:] != (256, q) or mask.shape[1:] != (256,):
                            raise ValueError("Input shape differs from manifest")
                        if np.any(tokens < 0) or np.any(tokens >= BACKBONES[representation]["vocab_size"]):
                            raise ValueError("Out-of-range token ID")
                        if not np.all(mask[:, 0]) or not np.all(tokens[:, 0, 0] == 1) or np.any(tokens[:, 0, 1:]):
                            raise ValueError("Missing/invalid CLS position")
                        records = [(str(s), int(i)) for s, i in zip(sources, indices, strict=True)]
                        try:
                            connection.executemany("INSERT INTO identities VALUES (?,?)", records)
                        except sqlite3.IntegrityError as error:
                            raise ValueError("Duplicate event or train/val/test identity overlap") from error
                        for source, event in records:
                            digest.update(json.dumps([source, event, label], separators=(",", ":")).encode() + b"\n")
                        rows += len(labels)
                    relative = str(path.relative_to(prepared))
                    inventory[relative] = dict(rows=parquet.metadata.num_rows, sha256=sha256(path))
                connection.commit()
                if rows != manifest["split_counts"][split][name]:
                    raise ValueError(f"Wrong row count in {split}/{name}")
                split_rows += rows
                membership[f"{split}/{name}"] = dict(rows=rows, sha256=digest.hexdigest())
            if split_rows != manifest["split_counts"][split]["total"]:
                raise ValueError(f"Wrong {split} total")
        if vocabulary["max_quantizers"] != q or vocabulary["vocab_size"] != BACKBONES[representation]["vocab_size"]:
            raise ValueError("Vocabulary does not match the pretrained backbone")
    finally:
        connection.close()
    report = dict(manifest_sha256=manifest_hash, representation=representation,
                  classes=definitions, split_counts=manifest["split_counts"],
                  membership=membership, shards=inventory, token_vocabulary=vocabulary,
                  identity_overlap_detected=False)
    (report_dir / "classification_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: pinned vocabulary, all shard rows/labels, CLS and disjoint event identities", flush=True)
    return manifest, definitions, vocabulary


def prepare_smoke(source, output, rows_per_class=128):
    """Only shorten validation for the smoke, ensuring all classes are present.

    Full training/test splits are read-only symlinks. No held-out test examples
    are used for training or validation; full production data is unchanged.
    """
    source, output = Path(source).resolve(), Path(output)
    if rows_per_class < 1:
        raise ValueError('rows_per_class must be positive')
    manifest = json.loads((source / 'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'test'):
        (output / split).symlink_to(source / split, target_is_directory=True)
    result = copy.deepcopy(manifest)
    counts = {'total': 0}
    for name, _ in classes(manifest):
        tables, remaining = [], rows_per_class
        for path in sorted((source / 'val' / name).glob('*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=rows_per_class):
                keep = min(remaining, batch.num_rows)
                tables.append(pa.Table.from_batches([batch.slice(0, keep)]))
                remaining -= keep
                if not remaining:
                    break
            if not remaining:
                break
        if not tables:
            raise ValueError(f'No smoke validation events for {name}')
        table = pa.concat_tables(tables)
        path = output / 'val' / name / 'part-00000.parquet'
        path.parent.mkdir(parents=True)
        pq.write_table(table, path)
        counts[name], counts['total'] = table.num_rows, counts['total'] + table.num_rows
    result['split_counts']['val'] = counts
    result['smoke_only'] = dict(source_manifest_sha256=sha256(source / 'manifest.json'),
        validation_policy=f'first up to {rows_per_class} saved validation events per class',
        train_and_test='unchanged read-only references', production_use=False)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'manifest.sha256').write_text(sha256(output / 'manifest.json') + '\n')
    print(f'PASS: smoke validation includes every class ({counts["total"]} events); full dataset unchanged', flush=True)


def vocabulary_semantics(vocabulary):
    # Continuous-feature annotations are auxiliary, not the token-ID mapping.
    keys = ("objects", "event", "event_tokens", "object_order", "special_tokens",
            "vocab_size", "max_quantizers", "includes_cls", "sequence_layout")
    return {key: vocabulary.get(key) for key in keys}


def validate_pretrain(path, representation, n_classes, class_weights=None, expected_epoch=9,
                      token_vocabulary=None, classifier_head="mlp"):
    checkpoint = _load_checkpoint_state(str(path))
    hp = checkpoint["hyper_parameters"]
    expected = {**COMMON, **BACKBONES[representation]}
    for name, value in expected.items():
        if hp.get(name) != value:
            raise ValueError(f"Pretraining mismatch: {name}={hp.get(name)!r}, expected {value!r}")
    if checkpoint.get("epoch") != expected_epoch:
        raise ValueError(f"Pretraining is not complete: epoch={checkpoint.get('epoch')}")
    if representation == "hybrid" and (hp.get("coarse_quantizers") != 4 or hp.get("latent_loss_weight") != 1.0):
        raise ValueError("This is not the completed coarse4+full-Q8-latent hybrid")
    if token_vocabulary is not None:
        saved = hp.get("token_vocabulary")
        if saved is None or vocabulary_semantics(saved) != vocabulary_semantics(token_vocabulary):
            raise ValueError("Classification token-ID mapping differs from pretrained vocabulary")
    if classifier_head not in {"linear", "mlp"}:
        raise ValueError("classifier_head must be linear or mlp")
    model = LitGroupedCLSClassifier(**expected, n_classes=n_classes, class_weights=class_weights,
                                  freeze_backbone=False,
                                  classifier_hidden_dim=256 if classifier_head == "mlp" else None)
    state = {key.removeprefix("model.backbone."): value for key, value in checkpoint["state_dict"].items()
             if key.startswith("model.backbone.")}
    if not state or any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError("Invalid backbone weights")
    model.backbone.load_state_dict(state, strict=True)
    if not isinstance(model.backbone.object_projection, torch.nn.Identity):
        raise ValueError("A learned projection was accidentally created")
    print(f"PASS: {representation} backbone transferred strictly; pretraining heads discarded", flush=True)
    return model


def validate_finetune(last_path, expected_epochs):
    last_path = Path(last_path)
    checkpoint = _load_checkpoint_state(str(last_path))
    if checkpoint.get('epoch') != expected_epochs - 1:
        raise ValueError('Fine-tuning ended before the requested epochs')
    if any(not torch.isfinite(v).all() for v in checkpoint['state_dict'].values() if v.is_floating_point()):
        raise ValueError('Fine-tuning produced non-finite model parameters')
    best = _load_checkpoint_state(str(last_path.with_name('best.ckpt')))
    scores = [s['best_model_score'] for s in best.get('callbacks', {}).values()
              if isinstance(s, dict) and s.get('monitor') == 'valid/auc' and s.get('best_model_score') is not None]
    if len(scores) != 1 or not torch.isfinite(torch.as_tensor(scores[0])) or not 0 <= float(scores[0]) <= 1:
        raise ValueError('Missing/non-finite validation-best AUC checkpoint')
    print('PASS: requested fine-tuning epochs completed; finite model and validation-best AUC checkpoint', flush=True)


def export_test(args):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
    manifest = json.loads((args.prepared_dir / "manifest.json").read_text())
    definitions = classes(manifest)
    if sha256(args.prepared_dir / "manifest.json") != args.manifest_sha256:
        raise ValueError("Test manifest changed")
    checkpoint = _load_checkpoint_state(str(args.checkpoint))
    hp = dict(checkpoint["hyper_parameters"])
    allowed = inspect.signature(LitGroupedCLSClassifier.__init__).parameters
    kwargs = {name: value for name, value in hp.items() if name in allowed}
    kwargs.update(backbone_ckpt_path=None, optimizer=None, scheduler=None)
    model = LitGroupedCLSClassifier(**kwargs)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(args.device)
    torch.set_float32_matmul_precision("medium")
    arrays = {key: [] for key in ("probabilities", "label", "source_file", "event_index")}
    seen = set()
    with torch.inference_mode():
        for name, label in definitions:
            count = 0
            for path in sorted((args.prepared_dir / "test" / name).glob("*.parquet")):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=args.batch_size,
                        columns=["tokens", "mask", "type_ids", "label", "source_file", "event_index"]):
                    column = lambda key: batch.column(batch.schema.get_field_index(key))
                    labels = np.asarray(column("label").to_numpy(), dtype=np.int64)
                    sources = column("source_file").to_pylist()
                    indices = np.asarray(column("event_index").to_numpy(), dtype=np.int64)
                    if not np.all(labels == label):
                        raise ValueError("Test class labels differ")
                    for source, index in zip(sources, indices, strict=True):
                        identity = (source, int(index))
                        if identity in seen:
                            raise ValueError("Repeated test event")
                        seen.add(identity)
                    inputs = {key: torch.from_numpy(sequence_numpy(column(key), dtype).copy()).to(args.device)
                              for key, dtype in (("tokens", np.int64), ("mask", bool), ("type_ids", np.int64))}
                    probabilities = model(inputs).softmax(-1).cpu().numpy()
                    if probabilities.shape != (len(labels), len(definitions)) or not np.isfinite(probabilities).all():
                        raise ValueError("Invalid classifier probabilities")
                    if not np.allclose(probabilities.sum(1), 1.0, atol=1e-5):
                        raise ValueError("Softmax probabilities do not sum to one")
                    for key, value in dict(probabilities=probabilities, label=labels,
                            source_file=np.asarray(sources, dtype=str), event_index=indices).items():
                        arrays[key].append(value)
                    count += len(labels)
            if count != manifest["split_counts"]["test"][name]:
                raise ValueError("Test split incomplete")
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    if len(seen) != manifest["split_counts"]["test"]["total"]:
        raise ValueError("Test identities do not cover the complete split")
    labels, probabilities = arrays["label"], arrays["probabilities"]
    score = probabilities[:, 1] if len(definitions) == 2 else probabilities
    auc_kwargs = {} if len(definitions) == 2 else dict(multi_class="ovr", average="macro")
    report = dict(checkpoint_selection="validation-best", checkpoint_sha256=sha256(args.checkpoint),
                  manifest_sha256=args.manifest_sha256, class_names=[name for name, _ in definitions],
                  test_events=len(labels), test_auc=float(roc_auc_score(labels, score, **auc_kwargs)),
                  accuracy=float(accuracy_score(labels, probabilities.argmax(1))),
                  balanced_accuracy=float(balanced_accuracy_score(labels, probabilities.argmax(1))),
                  confusion_matrix=confusion_matrix(labels, probabilities.argmax(1)).tolist())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    class_names = [name for name, _ in definitions]
    predictions = probabilities.argmax(1)
    arrays.update(labels=labels, predictions=predictions, class_names=np.asarray(class_names))
    from evaluate_grouped_hzz_multiclass import DISPLAY_NAMES, plot_confusion_matrix, plot_roc_curves
    display_names = [DISPLAY_NAMES.get(name, name) for name in class_names]
    normalized = confusion_matrix(labels, predictions, labels=range(len(definitions)), normalize='true')
    plot_confusion_matrix(normalized, display_names, args.output_dir)
    per_class = plot_roc_curves(labels, probabilities, display_names, args.output_dir)
    report['per_class_auc'] = {name: per_class[display_names[i]] for i, name in enumerate(class_names)}
    report['normalized_confusion_matrix'] = normalized.tolist()
    if len(definitions) > 2:
        report['macro_ovr_auc'] = report['test_auc']
        report['weighted_ovr_auc'] = float(roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted'))
    classification = classification_report(labels, predictions, labels=range(len(definitions)),
        target_names=class_names, output_dict=True, zero_division=0)
    with (args.output_dir / 'classification_report.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['class', 'precision', 'recall', 'f1-score', 'support', 'auc'])
        writer.writeheader()
        for name in class_names:
            writer.writerow(dict({'class': name, 'auc': report['per_class_auc'][name]}, **classification[name]))
    np.savez_compressed(args.output_dir / "test_predictions.npz", **arrays)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser('prepare-smoke')
    smoke.add_argument('--source-dir', type=Path, required=True)
    smoke.add_argument('--output-dir', type=Path, required=True)
    smoke.add_argument('--rows-per-class', type=int, default=128)
    check = sub.add_parser("preflight")
    check.add_argument("--prepared-dir", type=Path, required=True)
    check.add_argument("--manifest-sha256", required=True)
    check.add_argument("--representation", choices=BACKBONES, required=True)
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--report-dir", type=Path, required=True)
    check.add_argument("--classifier-head", choices=("linear", "mlp"), default="mlp")
    finish = sub.add_parser('check-finetune')
    finish.add_argument('--last-checkpoint', type=Path, required=True)
    finish.add_argument('--epochs', type=int, required=True)
    predict = sub.add_parser("export-test")
    for name in ("prepared-dir", "checkpoint", "output-dir"):
        predict.add_argument(f"--{name}", type=Path, required=True)
    predict.add_argument("--manifest-sha256", required=True)
    predict.add_argument("--batch-size", type=int, default=256)
    predict.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == 'prepare-smoke':
        prepare_smoke(args.source_dir, args.output_dir, args.rows_per_class)
        return
    if args.command == 'check-finetune':
        validate_finetune(args.last_checkpoint, args.epochs)
        return
    if args.command == "export-test":
        export_test(args)
        return
    manifest, definitions, vocabulary = validate_data(args.prepared_dir, args.manifest_sha256,
                                             args.representation, args.report_dir)
    weights = manifest.get("recommended_cross_entropy_weights")
    weights = [float(weights[name]) for name, _ in definitions] if weights else None
    validate_pretrain(args.checkpoint, args.representation, len(definitions), weights,
                      token_vocabulary=vocabulary, classifier_head=args.classifier_head)
    (args.report_dir / "n_classes.txt").write_text(str(len(definitions)) + "\n")


if __name__ == "__main__":
    main()
