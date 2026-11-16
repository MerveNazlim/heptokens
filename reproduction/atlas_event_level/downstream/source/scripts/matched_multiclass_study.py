#!/usr/bin/env python3
"""Execution-side data, checkpoint, fine-tuning and held-out-test gates."""
from __future__ import annotations

import argparse
import inspect
import json
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from heptokens.data.continuous_schema import regroup_continuous_schema
from heptokens.models.foundation_continuous_matched_multiclass import LitMatchedContinuousMulticlass
from heptokens.models.foundation_continuous_multiclass import input_semantics
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier, _load_checkpoint_state
from hybrid_q4_downstream import classes, sequence_numpy, sha256, validate_data, validate_finetune, vocabulary_semantics
from prepare_q4_downstream_from_h5 import verify_inventory
from plot_matched_multiclass_comparison import NAMES, compare


def settings(path):
    return json.loads(Path(path).read_text())


def validate_alignment(source, prepared, report_dir, model_name, plan, inventory):
    verify_inventory(source, plan['source_manifest_sha256'], inventory, plan['input_inventory_sha256'])
    original, definitions, _ = validate_data(source, plan['source_manifest_sha256'], 'hybrid', report_dir)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if tuple(name for name, _ in definitions) != NAMES:
        raise ValueError('Wrong five-class order')
    for key in ('classes', 'split_counts', 'recommended_cross_entropy_weights', 'identity_overlap_detected'):
        if manifest.get(key) != original.get(key):
            raise ValueError(f'Prepared inputs changed original {key}')
    if original['split_counts'] != plan['split_counts']:
        raise ValueError('Source counts differ from the fixed matched experiment')
    q1 = model_name == 'q1_direct256'
    provenance = manifest.get('q1_reencoding' if q1 else 'continuous_preparation', {})
    if provenance.get('source_manifest_sha256') != plan['source_manifest_sha256']:
        raise ValueError('Prepared inputs belong to another source manifest')
    vocabulary, schema = None, None
    expected_paths = set()
    for split in ('train', 'val', 'test'):
        for name, _ in definitions:
            for path in sorted((source / split / name).glob('*.parquet')):
                relative = path.relative_to(source); expected_paths.add(relative)
                other = prepared / relative
                metadata = pq.ParquetFile(other).schema_arrow.metadata or {}
                observed = json.loads(metadata[b'heptokens_token_vocabulary'])
                if vocabulary is not None and observed != vocabulary:
                    raise ValueError('Mixed token vocabularies')
                vocabulary = observed
                if not q1:
                    observed_schema = json.loads(metadata[b'heptokens_continuous_schema'])
                    if schema is not None and observed_schema != schema:
                        raise ValueError('Mixed continuous schemas')
                    schema = observed_schema
                old_batches = pq.ParquetFile(path).iter_batches(batch_size=128)
                new_batches = pq.ParquetFile(other).iter_batches(batch_size=128)
                for old, new in zip_longest(old_batches, new_batches):
                    if old is None or new is None or old.num_rows != new.num_rows:
                        raise ValueError(f'Prepared event counts changed: {relative}')
                    keys = [key for key in old.schema.names if not q1 or key not in {'tokens', 'input_ids'}]
                    if not old.select(keys).equals(new.select(keys), check_metadata=False):
                        raise ValueError(f'Event identity/order, label or non-token columns changed: {relative}')
                    col = lambda key, dtype: sequence_numpy(new.column(new.schema.get_field_index(key)), dtype)
                    tokens = col('tokens', np.int64)
                    q, vocab_size = (1, 98820) if q1 else (8, 131588)
                    if tokens.shape[1:] != (256, q) or np.any(tokens < 0) or np.any(tokens >= vocab_size):
                        raise ValueError('Invalid token dimensions/range')
                    if q1 and 'input_ids' in new.schema.names and not np.array_equal(tokens, col('input_ids', np.int64)):
                        raise ValueError('Q1 input_ids alias differs')
                    if not q1:
                        features, valid, roles = col('continuous_features', np.float32), col('continuous_feature_mask', bool), col('position_role_ids', np.int64)
                        if features.shape[1:] != (256, schema['max_feature_dim']) or valid.shape != features.shape or roles.shape != features.shape[:2]:
                            raise ValueError('Invalid continuous feature/mask/role shape')
                        if not np.isfinite(features).all() or not np.all(roles[:, 0] == schema['role_ids']['cls']):
                            raise ValueError('Non-finite features or missing CLS')
    actual = {p.relative_to(prepared) for split in ('train', 'val', 'test') for p in (prepared / split).rglob('*.parquet')}
    if actual != expected_paths:
        raise ValueError('Prepared shard inventory changed')
    if vocabulary['max_quantizers'] != (1 if q1 else 8) or vocabulary['vocab_size'] != (98820 if q1 else 131588):
        raise ValueError('Wrong original token vocabulary')
    return manifest, vocabulary, schema


def classifier(model_name, schema, weights, checkpoint=None):
    common = dict(n_classes=5, class_weights=weights, backbone_ckpt_path=str(checkpoint) if checkpoint else None,
        freeze_backbone=False, classifier_hidden_dim=None, learning_rate=1e-4)
    if model_name == 'q1_direct256':
        return LitGroupedCLSClassifier(**common, max_quantizers=1, quantizer_embedding_dim=256,
            object_projection_mode='identity', vocab_size=98820)
    return LitMatchedContinuousMulticlass(**common, continuous_schema=schema,
        architecture='flat' if model_name == 'continuous_flat' else 'hierarchical',
        feature_grouping='schema' if model_name == 'continuous_flat' else 'kinematics_auxiliary_v1')


def preflight(args):
    plan = settings(args.plan)
    manifest, vocabulary, schema = validate_alignment(args.source_dir, args.prepared_dir, args.report_dir,
        args.model_name, plan, args.input_inventory)
    saved = _load_checkpoint_state(str(args.checkpoint)); hp = saved['hyper_parameters']
    if saved.get('epoch') != 9 or saved.get('global_step', 0) <= 0:
        raise ValueError('Pretraining did not complete ten epochs')
    for key, value in plan['models'][args.model_name]['expected_hparams'].items():
        if hp.get(key, 'schema' if key == 'feature_grouping' else None) != value:
            raise ValueError(f'Wrong pretrained {key}: {hp.get(key)!r}')
    if args.model_name == 'q1_direct256' and hp.get('token_vocabulary') is not None:
        if vocabulary_semantics(hp['token_vocabulary']) != vocabulary_semantics(vocabulary):
            raise ValueError('Original Q1 token-ID mapping differs from pretraining')
    weights = [manifest['recommended_cross_entropy_weights'][name] for name in NAMES]
    model = classifier(args.model_name, schema, weights, args.checkpoint)
    if not all(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError('Backbone was frozen')
    report = dict(passed=True, model=args.model_name, checkpoint_sha256=sha256(args.checkpoint),
        pretrain_epoch=saved['epoch'], pretrain_global_step=saved['global_step'],
        source_manifest_sha256=plan['source_manifest_sha256'], manifest_sha256=sha256(args.prepared_dir / 'manifest.json'),
        classes=NAMES, class_weights=weights, split_counts=manifest['split_counts'],
        classifier_head='linear_256_to_5', backbone_frozen=False, protocol=plan['protocol'])
    (args.report_dir / 'transfer_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS exact original events/splits/labels, strict pretrained transfer and matched fine-tuning protocol')


def check_fit(args):
    from omegaconf import OmegaConf
    validate_finetune(args.run_dir / 'checkpoints/last.ckpt', args.epochs)
    cfg = OmegaConf.load(args.run_dir / 'full_config.yaml')
    expected = {'seed': 42, 'datamodule.batch_size': 256, 'datamodule.num_workers': 0,
        'datamodule.n_classes': 5, 'datamodule.stream_batch_size': 4096,
        'datamodule.shuffle_buffer_size': 8192, 'model.freeze_backbone': False,
        'model.classifier_hidden_dim': None, 'model.learning_rate': 1e-4,
        'model.optimizer._target_': 'torch.optim.AdamW', 'model.optimizer.lr': 1e-4,
        'model.optimizer.weight_decay': 1e-2, 'model.scheduler.warmup_epochs': 1,
        'model.scheduler.min_lr': 1e-6, 'trainer.max_epochs': args.epochs,
        'trainer.devices': 1, 'trainer.val_check_interval': 1.0, 'trainer.limit_val_batches': 1.0,
        'callbacks.best_checkpoint.monitor': 'valid/auc', 'callbacks.best_checkpoint.mode': 'max'}
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f'Fine-tuning protocol drift: {key}')
    saved = _load_checkpoint_state(str(args.run_dir / 'checkpoints/last.ckpt'))
    # The reference classification loader uses drop_last=True for training.
    expected_steps = 200 if args.epochs == 1 else 10 * (1700673 // 256)
    if saved.get('global_step') != expected_steps:
        raise ValueError(f'Incomplete optimizer-step budget: expected {expected_steps}')
    print('PASS completed fit, finite validation-best checkpoint and exact fine-tuning protocol')


def export_test(args):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
    manifest = json.loads((args.prepared_dir / 'manifest.json').read_text())
    if tuple(name for name, _ in classes(manifest)) != NAMES:
        raise ValueError('Wrong held-out classes')
    saved = _load_checkpoint_state(str(args.checkpoint)); hp = dict(saved['hyper_parameters'])
    cls = LitGroupedCLSClassifier if args.model_name == 'q1_direct256' else LitMatchedContinuousMulticlass
    allowed = inspect.signature(cls.__init__).parameters
    kwargs = {key: value for key, value in hp.items() if key in allowed}
    kwargs.update(backbone_ckpt_path=None, optimizer=None, scheduler=None)
    model = cls(**kwargs); model.load_state_dict(saved['state_dict'], strict=True)
    if model.n_classes != 5 or model.model.classifier.out_features != 5:
        raise ValueError('Not a five-class linear classifier')
    model.eval().to(args.device); torch.set_float32_matmul_precision('medium')
    keys = [('tokens', np.int64), ('mask', bool), ('type_ids', np.int64)] if args.model_name == 'q1_direct256' else [
        ('continuous_features', np.float32), ('continuous_feature_mask', bool), ('position_role_ids', np.int64),
        ('mask', bool), ('type_ids', np.int64)]
    parts = {name: [] for name in ('probabilities', 'labels', 'source_file', 'event_index')}
    with torch.inference_mode():
        for label, name in enumerate(NAMES):
            count = 0
            for path in sorted((args.prepared_dir / 'test' / name).glob('*.parquet')):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=args.batch_size,
                        columns=[key for key, _ in keys] + ['label', 'source_file', 'event_index']):
                    col = lambda key: batch.column(batch.schema.get_field_index(key))
                    labels = np.asarray(col('label').to_numpy(), dtype=np.int64)
                    if not np.all(labels == label):
                        raise ValueError('Invalid held-out labels')
                    inputs = {key: torch.from_numpy(sequence_numpy(col(key), dtype).copy()).to(args.device) for key, dtype in keys}
                    probabilities = model(inputs).softmax(-1).cpu().numpy()
                    for key, value in dict(probabilities=probabilities, labels=labels,
                            source_file=np.asarray(col('source_file').to_pylist(), dtype=str),
                            event_index=np.asarray(col('event_index').to_numpy(), dtype=np.int64)).items():
                        parts[key].append(value)
                    count += len(labels)
            if count != manifest['split_counts']['test'][name]:
                raise ValueError(f'Incomplete held-out split: {name}')
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    arrays.update(class_names=np.asarray(NAMES), predictions=arrays['probabilities'].argmax(1))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / 'test_predictions.npz', **arrays)
    from plot_matched_multiclass_comparison import load_predictions
    checked = load_predictions(args.output_dir)
    if len(checked['labels']) != manifest['split_counts']['test']['total']:
        raise ValueError('Incomplete/duplicated held-out test events')
    labels, probabilities = checked['labels'], checked['probabilities']
    predictions = probabilities.argmax(1)
    display_name = {'q1_direct256': 'Q1 direct-256', 'continuous_flat': 'Continuous flat',
                    'hierarchical_kin_aux': 'Hierarchical'}[args.model_name]
    rows = compare([(display_name, args.output_dir)], args.output_dir / 'plots', expected_events=len(labels))
    macro = float(roc_auc_score(labels, probabilities, multi_class='ovr', average='macro'))
    report = dict(checkpoint_selection='validation-best', checkpoint_sha256=sha256(args.checkpoint),
        manifest_sha256=sha256(args.prepared_dir / 'manifest.json'), class_names=NAMES,
        test_events=len(labels), test_auc=macro, macro_ovr_auc=macro,
        weighted_ovr_auc=float(roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted')),
        accuracy=float(accuracy_score(labels, predictions)), balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        per_class_auc={row['class_name']: row['auc'] for row in rows},
        confusion_matrix=confusion_matrix(labels, predictions).tolist(),
        normalized_confusion_matrix=confusion_matrix(labels, predictions, normalize='true').tolist())
    classification = classification_report(labels, predictions, labels=range(5), target_names=NAMES, output_dict=True, zero_division=0)
    (args.output_dir / 'classification_report.json').write_text(json.dumps(classification, indent=2) + '\n')
    for name in ('summary.json', 'evaluation_summary.json'):
        (args.output_dir / name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pre = sub.add_parser('preflight')
    for name in ('source-dir', 'prepared-dir', 'checkpoint', 'report-dir', 'input-inventory', 'plan'):
        pre.add_argument(f'--{name}', type=Path, required=True)
    pre.add_argument('--model-name', required=True)
    fit = sub.add_parser('check-fit')
    fit.add_argument('--run-dir', type=Path, required=True); fit.add_argument('--epochs', type=int, required=True)
    export = sub.add_parser('export-test')
    for name in ('prepared-dir', 'checkpoint', 'output-dir'):
        export.add_argument(f'--{name}', type=Path, required=True)
    export.add_argument('--model-name', required=True)
    export.add_argument('--batch-size', type=int, default=256); export.add_argument('--device', default='cuda')
    args = parser.parse_args()
    {'preflight': preflight, 'check-fit': check_fit, 'export-test': export_test}[args.command](args)


if __name__ == '__main__':
    main()
