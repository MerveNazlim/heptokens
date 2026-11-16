#!/usr/bin/env python3
"""Future Q4/latest-hierarchical seed43/44 fine-tunes; all fine-tuning seed42.

Offline bind-plan requires downloaded SUCCESSFUL pretraining checkpoints,
configs and statuses. It never downloads, submits or starts training. Workers
use only the resulting checksum-bound plan. No global module patching is used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
from itertools import zip_longest
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from heptokens.models.foundation_continuous_matched_multiclass import LitMatchedContinuousMulticlass
from heptokens.models.foundation_grouped_cls_classifier import LitGroupedCLSClassifier, _load_checkpoint_state
from hybrid_q4_downstream import classes, sequence_numpy, sha256, validate_data, vocabulary_semantics
from matched_multiclass_study import check_fit, validate_alignment
from plot_matched_multiclass_comparison import NAMES, compare, load_predictions
from prepare_q4_downstream_from_h5 import verify_inventory
from seed_multiclass_study import MODEL_KEYS, TOKEN_KEYS, read_status, validate_pretrain_provenance, validate_export_provenance

CONFIGURATION_IDS = {'q4': 'q4_independent_tokenizers_v1',
    'hierarchical_kin_aux': 'hierarchical_kin_aux_287'}
RUN_KEYS = ('q4_p43', 'q4_p44', 'hierarchical_kin_aux_p43', 'hierarchical_kin_aux_p44')
Q4_MANIFEST_SHA = 'dd316af47f4150769a8908c48bea04199aa276712f579d2064128f79edc719d6'
Q4_BUNDLE_SHA = '971a8768440414d076ec835e988fe2924f264d573d48c9f6a86c77c11c7f2081'


def read_plan(path, key):
    plan = json.loads(Path(path).read_text())
    if plan.get('ready') is not True or set(plan.get('models', {})) != set(RUN_KEYS):
        raise ValueError('NOT READY: all four successful pretraining sources must be bound first')
    spec = plan['models'][key]
    family, seed = spec['model_family'], spec['pretrain_seed']
    if family not in CONFIGURATION_IDS or seed not in (43, 44) or key != f'{family}_p{seed}' or spec['finetune_seed'] != 42:
        raise ValueError('Require Q4/latest hierarchical pretraining seed43/44 and fine-tuning seed42')
    if spec['pretraining_configuration_id'] != CONFIGURATION_IDS[family]:
        raise ValueError('Wrong Q4/latest hierarchical configuration')
    for field in ('checkpoint_sha256', 'pretrain_config_sha256', 'pretrain_status_sha256', 'pretrain_completion_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', spec.get(field, '')):
            raise ValueError(f'Missing bound provenance: {field}')
    return plan, spec


def validate_source_receipt(checkpoint, config, status, completion, spec):
    saved = validate_pretrain_provenance(checkpoint, config, status, spec)
    if sha256(completion) != spec['pretrain_completion_sha256']:
        raise ValueError('Pinned pretraining completion receipt checksum mismatch')
    cfg, result = OmegaConf.load(config), read_status(status)
    for key, value in spec['expected_pretraining_config'].items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f'Pretraining input/masking/protocol mismatch: {key}')
    for key, value in spec['expected_pretraining_status'].items():
        if str(result.get(key)) != str(value):
            raise ValueError(f'Pretraining input/status provenance mismatch: {key}')
    hp = saved['hyper_parameters']
    for key, value in spec['expected_pretraining_hparams'].items():
        if hp.get(key) != value:
            raise ValueError(f'Pretraining objective/masking mismatch: {key}')
    receipt = json.loads(Path(completion).read_text())
    if receipt.get('checkpoint_sha256') != spec['checkpoint_sha256'] or receipt.get('full_config_sha256') != spec['pretrain_config_sha256']:
        raise ValueError('Completion receipt belongs to another checkpoint/configuration')
    expected = dict(passed=True, model_family=spec['model_family'], pretrain_seed=spec['pretrain_seed'],
        fine_tuning_seed=42, num_devices=6, batch_size_per_gpu=597, global_batch_size=3582, epochs=10, epoch=9)
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f'Incomplete/wrong pretraining completion receipt: {key}')
    updates, steps = receipt.get('updates_per_epoch'), receipt.get('expected_global_steps')
    if not isinstance(updates, int) or isinstance(updates, bool) or updates <= 0 or steps != updates * 10 \
            or receipt.get('rank_steps') != [updates] * 6 or receipt.get('global_step') != steps \
            or saved['global_step'] != steps:
        raise ValueError('Incomplete actual distributed optimizer-step budget in completion receipt')
    return saved, receipt


def bind_plan(recipe_path, source_path, output):
    """Bind future source receipts after pretraining succeeds; no submission."""
    recipe = json.loads(Path(recipe_path).read_text())
    sources = json.loads(Path(source_path).read_text())
    if recipe.get('ready') is not False or set(recipe.get('models', {})) != set(RUN_KEYS):
        raise ValueError('Expected the original NOT READY four-run recipe')
    if set(sources) != set(RUN_KEYS):
        raise ValueError('Supply all four downloaded pretraining sources, not an incomplete campaign')
    plan, receipts = deepcopy(recipe), {}
    jobs, hashes = set(), set()
    for key in RUN_KEYS:
        source, spec = sources[key], plan['models'][key]
        job = source.get('pretraining_job', '')
        if not re.fullmatch(r'\d+\.\d+', job) or job in jobs:
            raise ValueError('Require four distinct explicit pretraining job IDs')
        jobs.add(job)
        for remote, local, digest in (
            ('checkpoint', 'checkpoint_local', 'checkpoint_sha256'),
            ('pretrain_config', 'pretrain_config_local', 'pretrain_config_sha256'),
            ('pretrain_status', 'pretrain_status_local', 'pretrain_status_sha256'),
            ('pretrain_completion', 'pretrain_completion_local', 'pretrain_completion_sha256'),
        ):
            if not str(source.get(remote, '')).startswith('gs://'):
                raise ValueError(f'{key}: specify the original Google URI: {remote}')
            path = Path(source[local]).resolve()
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f'{key}: missing downloaded source {path}')
            spec[remote], spec[digest] = source[remote], sha256(path)
        if spec['checkpoint_sha256'] in hashes:
            raise ValueError('The same pretrained checkpoint cannot stand for two seeds')
        hashes.add(spec['checkpoint_sha256'])
        spec['pretraining_job'] = job
        cluster, proc = job.split('.')
        scope = f'/cluster_{cluster}_proc_{proc}/'
        if any(scope not in spec[name] for name in ('checkpoint', 'pretrain_config', 'pretrain_status', 'pretrain_completion')):
            raise ValueError('Source URIs belong to another declared pretraining job')
        if spec['pretrain_config'] != spec['checkpoint'].rsplit('/checkpoints/', 1)[0] + '/full_config.yaml':
            raise ValueError('Checkpoint/config URIs do not identify the same pretrained run')
        saved, receipt = validate_source_receipt(Path(source['checkpoint_local']), Path(source['pretrain_config_local']),
            Path(source['pretrain_status_local']), Path(source['pretrain_completion_local']), spec)
        if spec['model_family'] == 'hierarchical_kin_aux' and saved['hyper_parameters'].get('feature_grouping') != 'kinematics_auxiliary_v1':
            raise ValueError('Original hierarchical grouping cannot replace latest grouping')
        receipts[key] = dict(local_paths={name: str(Path(source[name]).resolve()) for name in
            ('checkpoint_local', 'pretrain_config_local', 'pretrain_status_local', 'pretrain_completion_local')},
            epoch=saved['epoch'], global_step=saved['global_step'], completion=receipt)
    plan.update(ready=True, binding_source='offline downloaded successful pretraining checkpoint/config/status',
                source_receipts=receipts)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite a bound plan; use a new explicit path')
    output.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'BOUND four verified sources: {output}; no training/submission performed')
    return plan


def validate_inputs(source, prepared, report_dir, model_name, plan, spec, inventory):
    if spec['model_family'] == 'hierarchical_kin_aux':
        return validate_alignment(source, prepared, report_dir, 'hierarchical_kin_aux', plan, inventory)
    verify_inventory(source, plan['source_manifest_sha256'], inventory, plan['input_inventory_sha256'])
    original, definitions, _ = validate_data(source, plan['source_manifest_sha256'], 'hybrid', report_dir / 'original')
    manifest, prepared_classes, vocabulary = validate_data(prepared, plan['q4_manifest_sha256'], 'q4', report_dir / 'q4')
    if tuple(name for name, _ in definitions) != NAMES or definitions != prepared_classes:
        raise ValueError('Wrong Q4 five-class order')
    for key in ('classes', 'split_counts', 'recommended_cross_entropy_weights', 'identity_overlap_detected'):
        if manifest.get(key) != original.get(key):
            raise ValueError(f'Q4 conversion changed original {key}')
    if original['split_counts'] != plan['split_counts']:
        raise ValueError('Canonical source counts changed')
    provenance = manifest.get('q4_reencoding', {})
    for key, value in (('source_manifest_sha256', plan['source_manifest_sha256']),
                       ('input_inventory_sha256', plan['input_inventory_sha256']),
                       ('tokenizer_bundle_sha256', Q4_BUNDLE_SHA)):
        if provenance.get(key) != value:
            raise ValueError(f'Q4 reencoding provenance differs: {key}')
    if not provenance.get('event_index_mapping'):
        raise ValueError('Missing successful Q4 H5-row/index correction provenance')
    expected_paths = set()
    for split in ('train', 'val', 'test'):
        for name in NAMES:
            for path in sorted((source / split / name).glob('*.parquet')):
                relative = path.relative_to(source); expected_paths.add(relative)
                other = prepared / relative
                recorded = provenance.get('shards', {}).get(str(relative), {})
                if recorded.get('sha256') != sha256(other) or recorded.get('source_sha256') != sha256(path):
                    raise ValueError('Q4 original/prepared shard checksum changed')
                for old, new in zip_longest(pq.ParquetFile(path).iter_batches(batch_size=128),
                                           pq.ParquetFile(other).iter_batches(batch_size=128)):
                    if old is None or new is None or old.num_rows != new.num_rows:
                        raise ValueError('Q4 event count changed')
                    keys = [key for key in old.schema.names if key not in {'tokens', 'input_ids'}]
                    if not old.select(keys).equals(new.select(keys), check_metadata=False):
                        raise ValueError('Q4 event identity/order, labels or non-token inputs changed')
    actual_paths = {p.relative_to(prepared) for split in ('train', 'val', 'test') for p in (prepared / split).rglob('*.parquet')}
    if actual_paths != expected_paths:
        raise ValueError('Q4 shard inventory differs')
    return manifest, vocabulary, None


def classifier(spec, hp, schema, vocabulary, weights, checkpoint=None):
    kwargs = {key: hp[key] for key in MODEL_KEYS if key in hp}
    common = dict(n_classes=5, class_weights=weights, freeze_backbone=False, classifier_hidden_dim=None,
        learning_rate=1e-4, backbone_ckpt_path=str(checkpoint) if checkpoint else None, **kwargs)
    if spec['model_family'] == 'hierarchical_kin_aux':
        if hp.get('architecture') != 'hierarchical' or hp.get('feature_grouping') != 'kinematics_auxiliary_v1':
            raise ValueError('Require the latest hierarchical grouping, not original groups')
        model = LitMatchedContinuousMulticlass(**common, continuous_schema=schema,
            architecture='hierarchical', feature_grouping='kinematics_auxiliary_v1',
            **{key: hp[key] for key in ('inner_num_heads', 'inner_num_layers') if key in hp})
    else:
        if hp.get('vocab_size') != 262660 or hp.get('max_quantizers') != 4 or hp.get('object_projection_mode') != 'identity':
            raise ValueError('Require independently trained Q4, identity projection and vocabulary262660')
        if hp.get('token_vocabulary') is None or vocabulary_semantics(hp['token_vocabulary']) != vocabulary_semantics(vocabulary):
            raise ValueError('Q4 pretrained/classification token-ID mapping differs')
        model = LitGroupedCLSClassifier(**common, token_vocabulary=vocabulary, **{key: hp[key] for key in TOKEN_KEYS})
    if checkpoint:
        state = _load_checkpoint_state(str(checkpoint))['state_dict']
        prefix = 'model.backbone.'
        transfer = {name.removeprefix(prefix): value for name, value in state.items() if name.startswith(prefix)}
        if not transfer:
            raise ValueError('Missing pretrained backbone')
        for name, value in transfer.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('Non-finite pretrained weights')
            if not torch.equal(model.backbone.state_dict()[name], value):
                raise ValueError('Strict backbone transfer changed a tensor')
    return model


def preflight(args):
    plan, spec = read_plan(args.plan, args.model_name)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest, vocabulary, schema = validate_inputs(args.source_dir, args.prepared_dir, args.report_dir,
        args.model_name, plan, spec, args.input_inventory)
    saved, receipt = validate_source_receipt(args.checkpoint, args.pretrain_config, args.pretrain_status,
                                            args.pretrain_completion, spec)
    weights = [manifest['recommended_cross_entropy_weights'][name] for name in NAMES]
    model = classifier(spec, saved['hyper_parameters'], schema, vocabulary, weights, args.checkpoint)
    if not all(p.requires_grad for p in model.backbone.parameters()) or not isinstance(model.model.classifier, torch.nn.Linear):
        raise ValueError('Require a fully unfrozen backbone and linear classifier')
    report = dict(passed=True, model=args.model_name, model_family=spec['model_family'], pretrain_seed=spec['pretrain_seed'],
        finetune_seed=42, plan_sha256=sha256(args.plan), checkpoint_sha256=spec['checkpoint_sha256'],
        checkpoint_uri=spec['checkpoint'], checkpoint_local_path=str(args.checkpoint.resolve()),
        pretrain_config_sha256=spec['pretrain_config_sha256'], pretrain_status_sha256=spec['pretrain_status_sha256'],
        pretrain_completion_sha256=spec['pretrain_completion_sha256'], pretraining_completion=receipt,
        pretraining_job=spec['pretraining_job'], pretraining_configuration_id=spec['pretraining_configuration_id'],
        pretraining_num_devices=spec['pretraining_num_devices'], pretraining_batch_size=spec['pretraining_batch_size'],
        pretraining_global_batch_size=spec['pretraining_global_batch_size'], pretraining_data_seed=spec['pretraining_data_seed'],
        pretraining_topology=dict(num_devices=spec['pretraining_num_devices'], batch_size_per_gpu=spec['pretraining_batch_size'],
            global_batch_size=spec['pretraining_global_batch_size'], data_seed=spec['pretraining_data_seed']),
        pretrain_epoch=saved['epoch'], pretrain_global_step=saved['global_step'],
        source_manifest_sha256=plan['source_manifest_sha256'], manifest_sha256=sha256(args.prepared_dir / 'manifest.json'),
        class_weights=weights, classes=NAMES, split_counts=manifest['split_counts'], classifier_head='linear_256_to_5',
        backbone_frozen=False, protocol=plan['protocol'],
        caveats=['Q4 pretraining seed42 global batch3584, new seeds43/44 global batch3582.'] if spec['model_family'] == 'q4' else [])
    (args.report_dir / 'transfer_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS exact original samples, successful pinned pretraining, strict transfer and fine-tuning seed42')
    return report


def export_test(args):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
    plan, spec = read_plan(args.plan, args.model_name)
    manifest = json.loads((args.prepared_dir / 'manifest.json').read_text())
    transfer = validate_export_provenance(args, plan, spec, manifest)
    saved = _load_checkpoint_state(str(args.checkpoint)); hp = dict(saved['hyper_parameters'])
    if str(Path(hp.get('backbone_ckpt_path', '')).resolve()) != transfer['checkpoint_local_path']:
        raise ValueError('Fine-tuned checkpoint used a different pretrained source')
    continuous = spec['model_family'] == 'hierarchical_kin_aux'
    cls = LitMatchedContinuousMulticlass if continuous else LitGroupedCLSClassifier
    allowed = inspect.signature(cls.__init__).parameters
    kwargs = {key: value for key, value in hp.items() if key in allowed}
    kwargs.update(backbone_ckpt_path=None, optimizer=None, scheduler=None)
    model = cls(**kwargs); model.load_state_dict(saved['state_dict'], strict=True)
    weights = [manifest['recommended_cross_entropy_weights'][name] for name in NAMES]
    if model.n_classes != 5 or not isinstance(model.model.classifier, torch.nn.Linear) or model.model.classifier.out_features != 5:
        raise ValueError('Not a five-class linear classifier')
    if not all(p.requires_grad for p in model.backbone.parameters()) or not np.allclose(model.class_weights.cpu(), weights):
        raise ValueError('Wrong class weights or frozen backbone')
    for key, value in spec['expected_hparams'].items():
        if key != 'seed' and hp.get(key) != value:
            raise ValueError(f'Fine-tuned representation mismatch: {key}')
    model.eval().to(args.device); torch.set_float32_matmul_precision('medium')
    keys = [('tokens', np.int64), ('mask', bool), ('type_ids', np.int64)]
    if continuous:
        keys = [('continuous_features', np.float32), ('continuous_feature_mask', bool),
                ('position_role_ids', np.int64), ('mask', bool), ('type_ids', np.int64)]
    parts = {name: [] for name in ('probabilities', 'labels', 'source_file', 'event_index')}
    with torch.inference_mode():
        for label, name in enumerate(NAMES):
            count = 0
            for path in sorted((args.prepared_dir / 'test' / name).glob('*.parquet')):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=args.batch_size,
                        columns=[key for key, _ in keys] + ['label', 'source_file', 'event_index']):
                    column = lambda key: batch.column(batch.schema.get_field_index(key))
                    labels = np.asarray(column('label').to_numpy(), dtype=np.int64)
                    if not np.all(labels == label):
                        raise ValueError('Invalid held-out labels')
                    inputs = {key: torch.from_numpy(sequence_numpy(column(key), dtype).copy()).to(args.device) for key, dtype in keys}
                    values = dict(probabilities=model(inputs).softmax(-1).cpu().numpy(), labels=labels,
                        source_file=np.asarray(column('source_file').to_pylist(), dtype=str),
                        event_index=np.asarray(column('event_index').to_numpy(), dtype=np.int64))
                    for key, value in values.items():
                        parts[key].append(value)
                    count += len(labels)
            if count != manifest['split_counts']['test'][name]:
                raise ValueError('Incomplete held-out class')
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    arrays.update(class_names=np.asarray(NAMES), predictions=arrays['probabilities'].argmax(1))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / 'test_predictions.npz', **arrays)
    checked = load_predictions(args.output_dir); labels, scores = checked['labels'], checked['probabilities']
    if len(labels) != manifest['split_counts']['test']['total']:
        raise ValueError('Incomplete/duplicated held-out events')
    rows = compare([(f"{spec['model_family']} pretrain seed {spec['pretrain_seed']}", args.output_dir)],
                   args.output_dir / 'plots', expected_events=len(labels))
    report = dict(transfer)
    report.update(pretraining_checkpoint_sha256=spec['checkpoint_sha256'], checkpoint_selection='validation-best',
        checkpoint_sha256=sha256(args.checkpoint), finetune_config_sha256=sha256(args.run_dir / 'full_config.yaml'),
        class_names=NAMES, test_events=len(labels), macro_ovr_auc=float(roc_auc_score(labels, scores, multi_class='ovr', average='macro')),
        weighted_ovr_auc=float(roc_auc_score(labels, scores, multi_class='ovr', average='weighted')),
        accuracy=float(accuracy_score(labels, scores.argmax(1))), balanced_accuracy=float(balanced_accuracy_score(labels, scores.argmax(1))),
        per_class_auc={row['class_name']: row['auc'] for row in rows},
        confusion_matrix=confusion_matrix(labels, scores.argmax(1), labels=range(5)).tolist(),
        normalized_confusion_matrix=confusion_matrix(labels, scores.argmax(1), labels=range(5), normalize='true').tolist())
    for name in ('summary.json', 'evaluation_summary.json'):
        (args.output_dir / name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2)); return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    bind = sub.add_parser('bind-plan')
    bind.add_argument('--recipe', type=Path, required=True); bind.add_argument('--sources', type=Path, required=True)
    bind.add_argument('--output', type=Path, required=True)
    pre = sub.add_parser('preflight')
    for name in ('source-dir', 'prepared-dir', 'checkpoint', 'pretrain-config', 'pretrain-status', 'pretrain-completion',
                 'report-dir', 'input-inventory', 'plan'):
        pre.add_argument(f'--{name}', type=Path, required=True)
    pre.add_argument('--model-name', required=True)
    fit = sub.add_parser('check-fit'); fit.add_argument('--run-dir', type=Path, required=True); fit.add_argument('--epochs', type=int, required=True)
    export = sub.add_parser('export-test')
    for name in ('prepared-dir', 'checkpoint', 'output-dir', 'plan', 'transfer-report', 'run-dir'):
        export.add_argument(f'--{name}', type=Path, required=True)
    export.add_argument('--model-name', required=True); export.add_argument('--batch-size', type=int, default=256)
    export.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.command == 'bind-plan':
        return bind_plan(args.recipe, args.sources, args.output)
    return {'preflight': preflight, 'check-fit': check_fit, 'export-test': export_test}[args.command](args)


if __name__ == '__main__':
    main()
