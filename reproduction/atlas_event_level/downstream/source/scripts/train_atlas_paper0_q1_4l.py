#!/usr/bin/env python3
"""Matched, class-weighted projected-Q1 fits on the selected ATLAS binary cohort."""
from __future__ import annotations

import argparse
import csv
from functools import partial
import json
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader, Dataset

from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier, _load_checkpoint_state
from heptokens.models.foundation_continuous import LitContinuousCLSClassifier
from heptokens.models.foundation_continuous_multiclass import input_semantics
from heptokens.models.utils import warmup_cosine_scheduler
from hybrid_q4_downstream import sequence_numpy as array_numpy, sha256 as digest_file, vocabulary_semantics
from projected_q1_seed_multiclass import validate_pretrain_provenance, verified_vocabulary
from seed_multiclass_study import validate_pretrain_provenance as validate_continuous_source

SELECTION = "paper0_atlas_hzz_cms_core_q1_v1"
CLASSES = {"background": (0, 700600), "signal": (1, 345060)}
SPLITS = ("train", "val", "test")
SEED, EPOCHS, BATCH = 42, 10, 256
HP = dict(hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1, max_seq_length=256,
          num_type_ids=12, use_type_embedding=True, use_position_embedding=True,
          vocab_size=98820, max_quantizers=1, quantizer_embedding_dim=32, object_projection_mode="linear")
PRETRAIN_SHA = "71b3d436b37914e7f1214609e3142a914190a4cc6b4d0c0cbc03779dfe482bd7"
CONTINUOUS_SHA = "62e4d6cc4f66d0ea3ff99805169697e9e2ed67424751de118eafcf20e1414e13"
CONTINUOUS_ARM = "continuous_finetune"
CONTINUOUS_HP = {key: value for key, value in HP.items() if key not in {
    "vocab_size", "max_quantizers", "quantizer_embedding_dim", "object_projection_mode"
}} | {"architecture": "flat", "feature_grouping": "schema"}


class WeightedFlatClassifier(LitContinuousCLSClassifier):
    """Keep the original binary flat encoder/head; use the Q1 weighted loss."""
    def __init__(self, *, class_weights, **kwargs):
        super().__init__(**kwargs)
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.shape != (2,) or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Expected two positive, finite training class weights")
        self.register_buffer("class_weights", weights)

    _shared_step = LitGroupedCLSClassifier._shared_step


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


class EventRows(Dataset):
    def __init__(self, features, sources, indices):
        self.features = {key: torch.from_numpy(np.ascontiguousarray(val)) for key, val in features.items()}
        self.sources, self.indices = sources, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.features.items()}


def read_dataset(root, expected_digest, require_file_disjoint=False, *, continuous=False):
    path = root / "manifest.json"
    if digest_file(path) != expected_digest:
        raise ValueError("Incorrect selected data manifest hash")
    manifest = json.loads(path.read_text())
    if (manifest.get("selection_id") != SELECTION or manifest.get("token_shape") != [256, 1]
            or manifest.get("n_classes") != 2 or manifest.get("identity_overlap_detected") is not False):
        raise ValueError("Expected the selected, event-disjoint, independent-Q1 binary cohort")
    inventory = {p.relative_to(root).as_posix() for s in SPLITS for p in (root / s).rglob("*.parquet")}
    if inventory != set(manifest["shards"]):
        raise ValueError("Missing or extra selected Parquet shards")
    membership = root / "membership.npz"
    if digest_file(membership) != manifest["membership_sha256"]:
        raise ValueError("Selected membership checksum mismatch")
    with np.load(membership, allow_pickle=False) as data:
        identities = {key: data[key] for key in ("sources", "source_id", "event_index", "labels", "split")}
    sets, vocabulary, schema, seen, file_sets = {}, None, None, set(), []
    continuous_columns = ("continuous_features", "continuous_feature_mask", "position_role_ids")
    if continuous and "continuous_preparation" not in manifest:
        raise ValueError("Missing continuous feature provenance")
    for split_id, split in enumerate(SPLITS):
        cols = {key: [] for key in ("mask", "type_ids", "labels", "source", "index")}
        cols.update({key: [] for key in ("continuous_features", "position_role_ids") if continuous})
        if not continuous:
            cols["tokens"] = []
        counts = {}
        for name, (label, dsid) in CLASSES.items():
            counts[name] = 0
            for shard in sorted((root / split / name).glob("*.parquet")):
                rel = shard.relative_to(root).as_posix()
                if digest_file(shard) != manifest["shards"][rel]:
                    raise ValueError(f"Shard checksum mismatch: {rel}")
                parquet = pq.ParquetFile(shard)
                vocab = json.loads(parquet.schema_arrow.metadata[b"heptokens_token_vocabulary"])
                if (vocab["vocab_size"], vocab["max_quantizers"]) != (98820, 1):
                    raise ValueError("Wrong original Q1 vocabulary")
                if vocabulary is None:
                    vocabulary = vocab
                elif vocabulary_semantics(vocabulary) != vocabulary_semantics(vocab):
                    raise ValueError("Vocabulary differs across selected shards")
                if continuous:
                    current = json.loads(parquet.schema_arrow.metadata[b"heptokens_continuous_schema"])
                    if schema is None:
                        schema = current
                    elif input_semantics(current) != input_semantics(schema):
                        raise ValueError("Continuous schema differs across selected shards")
                for batch in parquet.iter_batches(batch_size=4096, columns=[
                        "tokens", "mask", "type_ids", "label", "dsid", "source_file", "event_index", "m4l", "channel",
                        *(continuous_columns if continuous else ())]):
                    get = lambda key: batch.column(batch.schema.get_field_index(key))
                    n = batch.num_rows
                    toks, mask, types = (array_numpy(get(k), dtype) for k, dtype in
                                         (("tokens", np.int32), ("mask", bool), ("type_ids", np.int32)))
                    labels, dsids, index, mass = (get(k).to_numpy(zero_copy_only=False)
                                                  for k in ("label", "dsid", "event_index", "m4l"))
                    source = get("source_file").to_pylist()
                    if (toks.shape != (n, 256, 1) or mask.shape != (n, 256) or types.shape != mask.shape
                            or not (mask[:, 0].all() and (toks[:, 0, 0] == 1).all())
                            or (toks < 0).any() or (toks >= 98820).any()
                            or (types < 0).any() or (types >= 12).any()
                            or not (labels == label).all() or not (dsids == dsid).all()
                            or not np.isfinite(mass).all() or not (mass > 70).all()
                            or not set(get("channel").to_pylist()) <= {"4e", "2e2mu", "4mu"}
                            or (index < 0).any() or any(not isinstance(s, str) or not s for s in source)):
                        raise ValueError(f"Invalid selected event payload: {rel}")
                    batch_ids = set(zip(source, index.tolist()))
                    if len(batch_ids) != n or seen.intersection(batch_ids):
                        raise ValueError(f"Repeated event identity in/across splits: {rel}")
                    seen.update(batch_ids)
                    counts[name] += n
                    payload = dict(mask=mask, type_ids=types, labels=labels, source=source, index=index)
                    if continuous:
                        features, valid, roles = (array_numpy(get(k), dtype) for k, dtype in
                            zip(continuous_columns, (np.float32, bool, np.int32)))
                        if (features.shape != (n, 256, schema["max_feature_dim"]) or valid.shape != features.shape
                                or roles.shape != mask.shape or not np.isfinite(features).all()
                                or (valid & ~mask[..., None]).any()
                                or not (roles[:, 0] == schema["role_ids"]["cls"]).all()):
                            raise ValueError(f"Invalid matched continuous inputs: {rel}")
                        payload.update(continuous_features=features, position_role_ids=roles)
                    else:
                        payload["tokens"] = toks
                    for key, values in payload.items():
                        cols[key].append(np.asarray(values))
        if min(counts.values()) <= 0 or manifest["split_counts"][split] != counts | {"total": sum(counts.values())}:
            raise ValueError(f"Incorrect selected class counts: {split}")
        values = {key: np.concatenate(chunks) for key, chunks in cols.items()}
        del cols
        file_sets.append(set(values["source"].tolist()))
        order = np.lexsort((values["index"], values["source"]))
        source, index = values.pop("source")[order], values.pop("index")[order]
        rows = identities["split"] == split_id
        expected_source = identities["sources"][identities["source_id"][rows]]
        expected_index, expected_label = identities["event_index"][rows], identities["labels"][rows]
        expected_order = np.lexsort((expected_index, expected_source))
        if not (np.array_equal(source, expected_source[expected_order])
                and np.array_equal(index, expected_index[expected_order])
                and np.array_equal(values["labels"][order], expected_label[expected_order])):
            raise ValueError(f"Dataset differs from the frozen selected membership: {split}")
        sets[split] = EventRows({key: val[order] for key, val in values.items()}, source, index)
        del values
    overlap = any(file_sets[i] & file_sets[j] for i in range(3) for j in range(i))
    if overlap != manifest.get("file_overlap_detected") or (overlap and require_file_disjoint):
        raise ValueError("Selected data violate the declared file-split policy")
    counts = manifest["split_counts"]["train"]
    weights = [counts["total"] / (2 * counts[name]) for name in CLASSES]
    if not np.allclose(weights, [manifest["recommended_cross_entropy_weights"][name] for name in CLASSES],
                       atol=0, rtol=1e-12):
        raise ValueError("Invalid inverse training-frequency class weights")
    return manifest, sets, schema if continuous else vocabulary, weights


def load_source(arm, plan_path, directory, input_spec):
    if arm == "scratch":
        if directory is not None:
            raise ValueError("Scratch must not read pretrained weights")
        return None
    if directory is None:
        raise ValueError("Missing frozen projected-Q1 pretraining artifacts")
    continuous = arm == CONTINUOUS_ARM
    specs = json.loads(plan_path.read_text())["sources"]
    spec = specs["continuous_flat_p42" if continuous else "q1_projected_p42"]
    expected = ("133.0", CONTINUOUS_SHA, CONTINUOUS_HP) if continuous else ("130.0", PRETRAIN_SHA, HP)
    if (spec["pretraining_job"], spec["checkpoint_sha256"], spec["expected_hparams"]) != expected:
        raise ValueError("Wrong Paper 0 pretrained source")
    ckpt, cfg, status = (directory / k for k in ("checkpoint.ckpt", "config.yaml", "status.txt"))
    if continuous:
        saved = validate_continuous_source(ckpt, cfg, status, spec)
        hp = saved["hyper_parameters"]
        if hp.get("use_role_embedding", True) is not True or input_semantics(hp["continuous_schema"]) != input_semantics(input_spec):
            raise ValueError("Continuous input schema or original role embedding changed")
        return ckpt
    saved = validate_pretrain_provenance(ckpt, cfg, status, spec)
    old_vocab = saved["hyper_parameters"].get("token_vocabulary")
    if old_vocab is None:
        old_vocab = verified_vocabulary(directory / "vocabulary.parquet", status, spec, input_spec)
    if vocabulary_semantics(old_vocab) != vocabulary_semantics(input_spec):
        raise ValueError("Q1 codebook mapping differs from the pretraining checkpoint")
    return ckpt


def make_model(input_spec, weights, ckpt, *, continuous=False):
    kwargs = dict(n_classes=2, class_weights=weights,
        freeze_backbone=False, classifier_hidden_dim=256, backbone_ckpt_path=str(ckpt) if ckpt else None,
        learning_rate=1e-4, optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.01),
        scheduler=partial(warmup_cosine_scheduler, warmup_epochs=1, min_lr=1e-6),
    )
    if continuous:
        return WeightedFlatClassifier(**CONTINUOUS_HP, continuous_schema=input_spec, **kwargs)
    return LitGroupedCLSClassifier(**HP, token_vocabulary=input_spec, **kwargs)


def score(model, data, device):
    model.eval().to(device)
    scores = []
    with torch.inference_mode():
        for batch in DataLoader(data, batch_size=BATCH, shuffle=False):
            logits = model({key: val.to(device) for key, val in batch.items()})
            scores.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
    score = np.concatenate(scores)
    if not np.isfinite(score).all() or len(score) != len(data):
        raise ValueError("Invalid or incomplete test scores")
    return score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--arm", required=True, choices=("scratch", "finetune", CONTINUOUS_ARM))
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--plan", type=Path, default=Path("multiclass_controls_plan.json"))
    p.add_argument("--require-file-disjoint", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    continuous = args.arm == CONTINUOUS_ARM
    manifest, sets, input_spec, weights = read_dataset(
        args.dataset_dir, args.manifest_sha256, args.require_file_disjoint, continuous=continuous)
    ckpt = load_source(args.arm, args.plan, args.source_dir, input_spec)
    pl.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("medium")
    model = make_model(input_spec, weights, ckpt, continuous=continuous)
    print(f"PASS selected counts={manifest['split_counts']}; train class weights={weights}", flush=True)
    if args.preflight_only:
        batch = next(iter(DataLoader(sets["val"], batch_size=8)))
        model.eval()
        with torch.inference_mode():
            assert torch.isfinite(model(batch)).all()
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expected exactly one allocated GPU")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(SEED)
    train = DataLoader(sets["train"], batch_size=BATCH, shuffle=True, generator=generator, pin_memory=True)
    val = DataLoader(sets["val"], batch_size=BATCH, shuffle=False, pin_memory=True)
    best = ModelCheckpoint(dirpath=out / "checkpoints", filename="best", monitor="valid/auc",
                           mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False)
    fit = pl.Trainer(accelerator="gpu", devices=1, precision="32-true", max_epochs=EPOCHS, callbacks=[best],
                     logger=CSVLogger(out, name="metrics"), log_every_n_steps=30, enable_progress_bar=False)
    fit.fit(model, train_dataloaders=train, val_dataloaders=val)
    if fit.global_step != EPOCHS * len(train) or not best.best_model_path:
        raise ValueError("Incomplete fit or missing best validation checkpoint")
    selected = _load_checkpoint_state(best.best_model_path)
    model.load_state_dict(selected["state_dict"], strict=True)
    test = sets["test"]
    labels = test.features["labels"].numpy()
    scores = score(model, test, "cuda")
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    np.savez_compressed(out / "test_predictions.npz", source_file=test.sources, event_index=test.indices,
                        label=labels, signal_score=scores)
    np.savez_compressed(out / "roc_arrays.npz", fpr=fpr, tpr=tpr, thresholds=thresholds)
    with (out / "roc.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("signal_efficiency", "background_efficiency", "background_rejection"))
        writer.writerows(zip(tpr, fpr, 1 / np.maximum(fpr, 1 / int((labels == 0).sum()))))
    report = dict(passed=True, arm=args.arm, selection_id=SELECTION, seed=SEED,
                  manifest_sha256=args.manifest_sha256, membership_sha256=manifest["membership_sha256"],
                  split_counts=manifest["split_counts"],
                  file_overlap_detected=manifest["file_overlap_detected"], train_class_weights=dict(zip(CLASSES, weights)),
                  model=("original continuous flat, 256-unit CLS MLP" if continuous
                         else "original projected Q1 (32-to-256), 256-unit CLS MLP"),
                  epochs=EPOCHS, batch_size=BATCH,
                  optimizer="AdamW", learning_rate=1e-4, weight_decay=0.01, loss="class-weighted cross-entropy",
                  best_epoch=int(selected["epoch"]), best_validation_auc=float(best.best_model_score),
                  pretraining_job=("133.0" if continuous else "130.0") if ckpt else None,
                  pretraining_sha256=(CONTINUOUS_SHA if continuous else PRETRAIN_SHA) if ckpt else None,
                  auc=float(roc_auc_score(labels, scores)), accuracy=float(accuracy_score(labels, scores > 0.5)),
                  evaluation_weighting="unweighted", test_events=len(test),
                  prediction_sha256=digest_file(out / "test_predictions.npz"),
                  best_checkpoint_sha256=digest_file(Path(best.best_model_path)))
    write_json(out / "evaluation_summary.json", report)
    write_json(out / "completion.json", report | {"evaluation_summary_sha256": digest_file(out / "evaluation_summary.json")})
    print(f"PASS {args.arm} selected test AUC={report['auc']:.6f}", flush=True)


if __name__ == "__main__":
    main()
