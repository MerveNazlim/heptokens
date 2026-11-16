#!/usr/bin/env python3
"""Plot 45 saved five-class evaluations, without training or model loading.

Fine-tuned and frozen comparisons vary the pretraining seed with downstream
training seed 42. Scratch comparisons vary the supervised-training seed and
have no pretraining seed/source. Raw probabilities are never ensembled.
All conditions are checked against the same held-out event identities.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from plot_matched_multiclass_comparison import load_predictions, load_q8_reference, SOURCE_SHA
from plot_multiclass_studies import (
    MAIN_REPRESENTATIONS, NAMES, Run, check_files, check_saved_metrics, file_hash,
    metrics, presentation_label, save_figure,
)
from plot_multiclass_seed_uncertainty import (
    fpr_grid, interpolate_roc, mean_sd, matrix_figure, roc_figure,
)

CONDITIONS = ('pretrained', 'scratch', 'frozen')
SEEDS = (42, 43, 44)
FAMILIES = {
    'q1': ('q1_projected', 'q1_projected_v1_linear_1x32'),
    'q4': ('q4', 'q4_independent_tokenizers_v1'),
    'q8': ('q8', 'q8_parallel_v1_linear_8x32'),
    'continuous': ('continuous_flat', 'continuous_flat_v1_schema'),
    'hierarchical': ('hierarchical_kin_aux', 'hierarchical_kin_aux_287'),
}
CONTROL_PROTOCOL = dict(epochs=10, batch_size=256, classifier_head='linear_256_to_5',
    learning_rate=1e-4, weight_decay=.01, optimizer='AdamW', full_optimizer_steps=66430,
    loss='inverse-training-frequency-weighted_cross_entropy', num_workers=0,
    stream_batch_size=4096, shuffle_buffer_size=8192, gpu_count_per_job=1,
    scheduler='epoch_warmup_cosine', warmup_epochs=1, min_lr=1e-6,
    validation='all_original_validation_events_every_epoch',
    checkpoint='maximum_validation_macro_one_vs_rest_AUC', training_drop_last=True,
    test='364441_original_events_once_after_training',
    plots='False Positive Rate x True Positive Rate; one panel per class; AUC legends')
CONDITION_TITLES = {
    'pretrained': 'Fine-tuned: pretraining seeds 42/43/44; fine-tuning seed 42',
    'scratch': 'From scratch: supervised-training seeds 42/43/44; no pretrained source',
    'frozen': 'Frozen backbone: pretraining seeds 42/43/44; classifier-training seed 42',
}


@dataclass(frozen=True)
class ConditionRun:
    run: Run
    pretrain_seed: int | None
    training_seed: int
    legacy_seed42_reference: bool = False
    pretraining_job: str | None = None
    evaluation_job: str | None = None
    canonical_plan: Path | None = None
    canonical_verification_report: Path | None = None
    canonical_transfer_report: Path | None = None

    @property
    def varying_seed(self):
        return self.training_seed if self.run.condition == 'scratch' else self.pretrain_seed


@dataclass(frozen=True)
class ConditionModel:
    id: str
    label: str
    representation: str
    model_family: str
    pretraining_configuration_id: str
    condition: str
    expected_seeds: tuple[int, ...]
    runs: tuple[ConditionRun, ...]
    caveats: tuple[str, ...] = ()


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _safe_id(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]+', value)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[a-fA-F0-9]{64}', value)


def check_replacement_configuration(config, root):
    """Bind the new comparison to the preserved baseline and resolved inputs."""
    pinned = config.get('replacement_provenance')
    if pinned is None:
        return
    for path_key, digest_key in (
            ('baseline_config', 'baseline_sha256'), ('canonical_plan', 'canonical_plan_sha256'),
            ('canonical_verification_report', 'canonical_verification_sha256')):
        if not isinstance(pinned.get(path_key), str) or not _digest(pinned.get(digest_key)):
            raise ValueError(f'Missing pinned replacement provenance: {path_key}/{digest_key}')
        if file_hash(root / pinned[path_key]) != pinned[digest_key]:
            raise ValueError(f'Pinned replacement provenance changed: {path_key}')
    original = json.loads((root / pinned['baseline_config']).read_text())
    retained = pinned.get('retained_runs', [])
    replacements = pinned.get('replacement_runs', [])
    if len(retained) != 27 or len(replacements) != 18:
        raise ValueError('Replacement comparison must preserve 27 inputs and replace only 18 Q4/Q8 runs')
    old_by_representation = {model['representation']: model for model in original['models']}
    current_retained = {run['id'] for model in config['models'] if model['representation'] not in ('q4', 'q8')
                        for members in model['conditions'].values() for run in members}
    if {row['id'] for row in retained} != current_retained:
        raise ValueError('Retained-input snapshot does not match the unchanged 27 runs')
    for model in config['models']:
        if model['representation'] not in ('q4', 'q8'):
            old = old_by_representation[model['representation']]
            if model != old:
                raise ValueError(f"Unchanged model configuration was modified: {model['id']}")
    for row in retained:
        directory = Path(row['input_dir'])
        for name, key in (('test_predictions.npz', 'predictions_sha256'),
                          ('evaluation_summary.json', 'evaluation_summary_sha256')):
            if file_hash(directory / name) != row[key]:
                raise ValueError(f"Retained unchanged input changed: {row['id']}/{name}")


def read_config(path, data_root=None):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get('version') != 1 or config.get('study_type') != 'multiclass_conditions' \
            or tuple(config.get('class_names', ())) != NAMES:
        raise ValueError('Require multiclass_conditions schema v1 and the canonical five class names/order')
    if not _integer(config.get('expected_test_events')) or config['expected_test_events'] < 5:
        raise ValueError('Specify expected_test_events explicitly')
    if tuple(config.get('expected_seeds', ())) != SEEDS or tuple(config.get('conditions', ())) != CONDITIONS:
        raise ValueError('Require three conditions and the complete expected seed set 42/43/44')
    root = Path(data_root).resolve() if data_root is not None else (path.parent / config.get('data_root', '.')).resolve()
    check_replacement_configuration(config, root)
    groups, model_ids, run_ids, paths, representations = [], set(), set(), set(), set()
    for model in config.get('models', []):
        model_id, representation = model.get('id'), model.get('representation')
        if not _safe_id(model_id) or model_id in model_ids:
            raise ValueError('Model IDs must be safe and unique')
        model_ids.add(model_id)
        if representation not in MAIN_REPRESENTATIONS or representation in representations:
            raise ValueError('Require five distinct main representations; hybrid is not in this comparison')
        representations.add(representation)
        if (model.get('model_family'), model.get('pretraining_configuration_id')) != FAMILIES[representation]:
            raise ValueError(f'{model_id}: incompatible model family/configuration; projected Q1 and latest hierarchy are required')
        if set(model.get('conditions', {})) != set(CONDITIONS):
            raise ValueError(f'{model_id}: require all three conditions')
        label = presentation_label(model.get('label', ''), representation)
        for condition in CONDITIONS:
            members = []
            for item in model['conditions'][condition]:
                identifier = item.get('id')
                if not _safe_id(identifier) or identifier in run_ids:
                    raise ValueError('Run IDs must be safe and unique across all 45 evaluations')
                run_ids.add(identifier)
                pretrain, training = item.get('pretrain_seed'), item.get('training_seed')
                if not _integer(training):
                    raise ValueError(f'{identifier}: specify an integer training_seed')
                if condition == 'scratch':
                    if pretrain is not None or training not in SEEDS:
                        raise ValueError(f'{identifier}: scratch varies training_seed and has no pretrain_seed')
                elif not _integer(pretrain) or pretrain not in SEEDS or training != 42:
                    raise ValueError(f'{identifier}: require pretrain_seed 42/43/44 and fixed training_seed=42')
                legacy = item.get('legacy_seed42_reference', False)
                if not isinstance(legacy, bool) or (legacy and not (condition == 'pretrained' and pretrain == 42)):
                    raise ValueError('Only an explicit fine-tuned pretraining-seed-42 reference may omit historical metadata')
                kind = item.get('kind', 'predictions')
                if kind not in ('predictions', 'q8_paired_reference', 'canonical_307_predictions') or (kind == 'q8_paired_reference' and
                        not (representation == 'q8' and legacy)):
                    raise ValueError('The legacy paired Q8 format is only a fine-tuned Q8 seed-42 reference')
                canonical_paths = [None, None, None]
                if kind == 'canonical_307_predictions':
                    if representation not in ('q4', 'q8') or legacy:
                        raise ValueError('Canonical 307 predictions require Q4/Q8 and cannot use legacy metadata flags')
                    keys = ('canonical_plan', 'canonical_verification_report', 'canonical_transfer_report')
                    if any(not isinstance(item.get(key), str) or not item[key] for key in keys):
                        raise ValueError(f'{identifier}: require all canonical plan/verification/transfer paths')
                    canonical_paths = [(root / item[key]).resolve() for key in keys]
                if not isinstance(item.get('path'), str) or not item['path']:
                    raise ValueError(f'{identifier}: specify a saved evaluation path')
                directory = (root / item['path']).resolve()
                if directory in paths:
                    raise ValueError('A saved evaluation directory cannot stand for multiple runs')
                paths.add(directory)
                run = Run(identifier, label, representation, condition, directory, kind)
                members.append(ConditionRun(run, pretrain, training, legacy,
                    item.get('pretraining_job'), item.get('evaluation_job'), *canonical_paths))
            found = [member.varying_seed for member in members]
            if len(found) != 3 or sorted(found) != list(SEEDS):
                raise ValueError(f'{model_id}/{condition}: require the complete expected seed set 42/43/44')
            groups.append(ConditionModel(f'{model_id}__{condition}', label, representation,
                model['model_family'], model['pretraining_configuration_id'], condition, SEEDS,
                tuple(sorted(members, key=lambda member: member.varying_seed)),
                tuple(model.get('condition_caveats', {}).get(condition, ()))))
    if representations != set(MAIN_REPRESENTATIONS) or len(groups) != 15:
        raise ValueError('Require exactly five models x three conditions x three seeds (45 runs)')
    return config, root, groups


def _expect(report, required, identifier, *, allow_missing=False):
    missing = []
    for key, expected in required.items():
        if key not in report:
            missing.append(key)
        elif report[key] != expected or (key.endswith('_seed') and expected is not None and not _integer(report[key])) \
                or (isinstance(expected, bool) and not isinstance(report[key], bool)):
            raise ValueError(f'{identifier}: conflicting {key}; expected {expected!r}')
    if missing and not allow_missing:
        raise ValueError(f'{identifier}: missing metadata {missing}')
    return missing


def check_run_metadata(group, member):
    run = member.run
    if run.kind == 'canonical_307_predictions':
        from multiclass_canonical_307_adapter import validate_canonical_run
        return validate_canonical_run(group, member)
    path = run.path / 'evaluation_summary.json'
    raw_report = json.loads(path.read_text()) if path.is_file() else {}
    report = raw_report
    # The original paired evaluator can save a two-branch list, not a single
    # run dictionary. Its scores/metrics are checked by check_saved_metrics;
    # unavailable seed fields remain explicit legacy assertions, never invented.
    if isinstance(report, list) and run.kind == 'q8_paired_reference' and member.legacy_seed42_reference:
        report = {}
    if not isinstance(report, dict):
        raise ValueError(f'{run.id}: invalid evaluation summary')
    caveats, missing = [], []
    if group.condition == 'pretrained':
        required = dict(model_family=group.model_family,
            pretraining_configuration_id=group.pretraining_configuration_id,
            pretrain_seed=member.pretrain_seed, finetune_seed=42)
        missing = _expect(report, required, run.id, allow_missing=member.legacy_seed42_reference)
        for key in ('pretraining_checkpoint_sha256', 'pretraining_job'):
            if key not in report:
                missing.append(key)
        if missing and not member.legacy_seed42_reference:
            raise ValueError(f'{run.id}: missing seed/configuration provenance {missing}')
        if 'backbone_frozen' in report and report['backbone_frozen'] is not False:
            raise ValueError(f'{run.id}: fine-tuned result reports a frozen backbone')
        source_digest, source_job = report.get('pretraining_checkpoint_sha256'), report.get('pretraining_job')
        if ('pretraining_checkpoint_sha256' in report or not member.legacy_seed42_reference) and not _digest(source_digest):
            raise ValueError(f'{run.id}: missing/invalid pretrained checkpoint checksum')
        if missing:
            caveats.append('Explicit legacy seed-42 configuration assertion; missing historical fields were not independently verified')
    else:
        frozen = group.condition == 'frozen'
        _expect(report, dict(run_key=run.id, model_family=group.model_family,
            configuration_id=group.pretraining_configuration_id, control_mode=group.condition,
            backbone_frozen=frozen, training_seed=member.training_seed,
            pretrain_seed=member.pretrain_seed), run.id)
        protocol = report.get('protocol')
        if not isinstance(protocol, dict):
            raise ValueError(f'{run.id}: missing common training protocol')
        _expect(protocol, CONTROL_PROTOCOL, f'{run.id}/protocol')
        # The common baseline copied these two fields unchanged into controls.
        # Top-level effective seeds/condition and the fit receipt are authoritative.
        caveats.append('Inherited protocol.seed and protocol.freeze_backbone are recorded, not used as effective run semantics')
        fit = report.get('control_fit_validation')
        if not isinstance(fit, dict):
            raise ValueError(f'{run.id}: missing control_fit_validation receipt')
        _expect(fit, dict(passed=True, run_key=run.id, control_mode=group.condition,
            training_seed=member.training_seed, pretrain_seed=member.pretrain_seed,
            epochs=10, expected_global_steps=66430), f'{run.id}/control_fit_validation')
        for key in ('initial_backbone_sha256', 'final_backbone_sha256',
                    'best_checkpoint_sha256', 'last_checkpoint_sha256'):
            if not _digest(fit.get(key)):
                raise ValueError(f'{run.id}: invalid saved fit receipt {key}')
        if fit['best_checkpoint_sha256'] != report.get('checkpoint_sha256'):
            raise ValueError(f'{run.id}: exported classifier differs from validation-best fit receipt')
        initial, final = fit['initial_backbone_sha256'], fit['final_backbone_sha256']
        if frozen:
            if not _digest(report.get('source_backbone_sha256')) or initial != final or initial != report['source_backbone_sha256']:
                raise ValueError(f'{run.id}: frozen source/initial/final backbone fingerprints differ')
            if report.get('source_key') != f'{group.model_family}_p{member.pretrain_seed}':
                raise ValueError(f'{run.id}: incompatible frozen source key')
            source = report.get('source_provenance')
            if not isinstance(source, dict):
                raise ValueError(f'{run.id}: missing frozen source provenance')
            _expect(source, dict(model_family=group.model_family, pretrain_seed=member.pretrain_seed,
                pretraining_configuration_id=group.pretraining_configuration_id), f'{run.id}/source')
            source_digest, source_job = source.get('checkpoint_sha256'), source.get('pretraining_job')
            if not _digest(source_digest):
                raise ValueError(f'{run.id}: missing/invalid frozen source checkpoint checksum')
        else:
            if any(report.get(key) is not None for key in
                   ('pretrain_seed', 'source_key', 'source_provenance', 'source_backbone_sha256')):
                raise ValueError(f'{run.id}: scratch must not contain a pretrained source')
            if initial == final:
                raise ValueError(f'{run.id}: scratch backbone did not change during training')
            source_digest = source_job = None
        caveats.append('Validated stored fit-integrity receipt; no checkpoint or tensor was loaded by this plotter')
    if source_digest is not None and not _digest(source_digest):
        raise ValueError(f'{run.id}: invalid pretrained checkpoint checksum')
    if source_digest is not None:
        source_digest = source_digest.lower()
    if source_job is not None and (not isinstance(source_job, str) or not re.fullmatch(r'\d+\.\d+', source_job)):
        raise ValueError(f'{run.id}: invalid pretraining job')
    if group.condition != 'scratch' and not member.legacy_seed42_reference and source_job is None:
        raise ValueError(f'{run.id}: missing pretraining job')
    if source_job is not None and member.pretraining_job is not None and source_job != member.pretraining_job:
        raise ValueError(f'{run.id}: unexpected pretraining job')
    if report.get('class_names') is not None and tuple(report['class_names']) != NAMES:
        raise ValueError(f'{run.id}: saved summary class order differs')
    if report.get('source_manifest_sha256') is not None and report['source_manifest_sha256'] != SOURCE_SHA:
        raise ValueError(f'{run.id}: conflicting canonical source manifest')
    return dict(condition=group.condition, varying_seed='training_seed' if group.condition == 'scratch' else 'pretrain_seed',
        pretrain_seed=member.pretrain_seed, training_seed=member.training_seed,
        expected_evaluation_job=member.evaluation_job,
        metadata_sha256=file_hash(path) if path.is_file() else None,
        source_checkpoint_sha256=source_digest, source_pretraining_job=source_job,
        legacy_metadata_not_available=missing, caveats=caveats,
        saved_metadata=raw_report)


def load(groups, expected_events):
    """Validate all 45 memberships while retaining only one raw evaluation."""
    runs = [member.run for group in groups for member in group.runs]
    check_files(runs)
    reports, curves, provenance, metadata = {}, {}, {}, {}
    for group in groups:
        digests = set()
        for member in group.runs:
            info = check_run_metadata(group, member)
            digest = info['source_checkpoint_sha256']
            if digest is not None and digest in digests:
                raise ValueError(f'{group.id}: repeated pretrained checkpoint across seeds')
            if digest is not None:
                digests.add(digest)
            metadata[member.run.id] = info
    # Where original fine-tuning metadata exists, frozen must use the same source.
    source_by_model_seed = {}
    for group in groups:
        for member in group.runs:
            if group.condition == 'scratch':
                continue
            key = group.model_family, member.pretrain_seed
            digest = metadata[member.run.id]['source_checkpoint_sha256']
            previous = source_by_model_seed.get(key)
            if previous is not None and digest is not None and previous != digest:
                raise ValueError(f'{member.run.id}: fine-tuned/frozen source checkpoint mismatch')
            if digest is not None:
                source_by_model_seed[key] = digest
    reference = None
    for run in runs:
        if run.kind == 'q8_paired_reference':
            continue
        current = load_predictions(run.path)
        if len(current['labels']) != expected_events:
            raise ValueError(f'{run.id}: expected {expected_events} held-out events')
        if reference is not None and any(not np.array_equal(current[key], reference[key])
                for key in ('labels', 'source_file', 'event_index')):
            raise ValueError(f'{run.id}: test event identities or labels differ across conditions/seeds')
        if reference is None:
            reference = {key: current[key] for key in ('labels', 'source_file', 'event_index')}
        reports[run.id], curves[run.id] = metrics(current)
        if run.kind == 'canonical_307_predictions':
            from multiclass_canonical_307_adapter import check_canonical_metrics
            group = next(group for group in groups if any(member.run.id == run.id for member in group.runs))
            member = next(member for member in group.runs if member.run.id == run.id)
            saved_report = check_canonical_metrics(group, member, reports[run.id], current)
        else:
            saved_report = check_saved_metrics(run, reports[run.id])
        provenance[run.id] = dict(event_ids_available=True,
            membership_check='exact sorted event IDs/labels against one reference across all 45 runs',
            predictions_sha256=file_hash(run.path / 'test_predictions.npz'),
            saved_report=saved_report, seed_provenance=metadata[run.id])
        del current
    if reference is None:
        raise ValueError('Require at least one evaluation with saved event identities')
    for run in runs:
        if run.kind != 'q8_paired_reference':
            continue
        current, info = load_q8_reference(run.path, reference)
        reports[run.id], curves[run.id] = metrics(current)
        info.update(predictions_sha256=file_hash(run.path / 'test_predictions.npz'),
            saved_report=check_saved_metrics(run, reports[run.id]), seed_provenance=metadata[run.id])
        provenance[run.id] = info
        del current
    return reports, curves, provenance


def aggregate(groups, reports, curves, grid):
    summaries, averaged = {}, {}
    for group in groups:
        n = len(group.runs)
        scratch = group.condition == 'scratch'
        result = dict(condition=group.condition, n_seeds=n, seeds=list(group.expected_seeds),
            varying_seed='training_seed' if scratch else 'pretrain_seed',
            pretrain_seeds=None if scratch else [member.pretrain_seed for member in group.runs],
            training_seeds=[member.training_seed for member in group.runs],
            fixed_training_seed=None if scratch else 42,
            finetune_seed=42 if group.condition == 'pretrained' else None,
            classifier_training_seed=42 if group.condition == 'frozen' else None,
            test_events_per_seed=reports[group.runs[0].run.id]['test_events'],
            class_counts=reports[group.runs[0].run.id]['class_counts'],
            uncertainty='sample standard deviation across independent runs (ddof=1), not a confidence interval',
            caveats=list(group.caveats), per_class_auc={})
        for key in ('accuracy', 'balanced_accuracy', 'macro_ovr_auc', 'weighted_ovr_auc'):
            mean, sd = mean_sd([reports[member.run.id][key] for member in group.runs])
            result[key] = dict(mean=mean, sd=sd)
        averaged[group.id] = {}
        for name in NAMES:
            mean, sd = mean_sd([reports[member.run.id]['per_class_auc'][name] for member in group.runs])
            result['per_class_auc'][name] = dict(mean=mean, sd=sd)
            values = np.asarray([interpolate_roc(curves[member.run.id][name], grid) for member in group.runs])
            averaged[group.id][name] = dict(fpr=grid, tpr_mean=values.mean(0), tpr_sd=values.std(0, ddof=1))
        matrices = np.asarray([reports[member.run.id]['normalized_confusion_matrix'] for member in group.runs])
        result['normalized_confusion_mean'] = matrices.mean(0).tolist()
        result['normalized_confusion_sd'] = matrices.std(0, ddof=1).tolist()
        summaries[group.id] = result
    return summaries, averaged


def check_expected_replacement_statistics(config, groups, reports):
    expected = config.get('expected_replacement_statistics')
    if expected is None:
        return
    observed = []
    for row in expected:
        condition = 'pretrained' if row['condition'] == 'finetune' else row['condition']
        selected = [group for group in groups
                    if group.model_family == row['model_family'] and group.condition == condition]
        if len(selected) != 1 or len(selected[0].runs) != 3 or row['n'] != 3:
            raise ValueError('Invalid expected cluster-307 group statistics')
        group = selected[0]
        result = dict(model_family=group.model_family, condition=condition)
        for metric, prefix in (('macro_ovr_auc', 'macro_auc'), ('accuracy', 'accuracy')):
            mean, sd = mean_sd([reports[member.run.id][metric] for member in group.runs])
            for name, value in ((f'{prefix}_mean', mean), (f'{prefix}_sample_std', sd)):
                if not np.isclose(value, row[name], atol=1e-10, rtol=0):
                    raise ValueError(f'{group.id}: recomputed {name} differs from independently verified cluster 307')
                result[name] = value
        observed.append(result)
    if len(observed) != 6:
        raise ValueError('Require all six Q4/Q8 condition sanity checks')
    print('PASS: all six Q4/Q8 macro-AUC and accuracy mean/sample-SD sanity checks')


def summary_figure(groups, summaries, *, show_note=True):
    """Compare the five representations across the three training conditions."""
    import matplotlib.pyplot as plt
    from plot_multiclass_studies import COLOURS
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    xs = np.arange(3, dtype=float)
    for axis, metric, title in zip(axes, ('macro_ovr_auc', 'accuracy'), ('Macro one-vs-rest AUC', 'Accuracy')):
        for index, representation in enumerate(MAIN_REPRESENTATIONS):
            selected = [next(group for group in groups if group.representation == representation
                             and group.condition == condition) for condition in CONDITIONS]
            means = [summaries[group.id][metric]['mean'] for group in selected]
            spreads = [summaries[group.id][metric]['sd'] for group in selected]
            axis.errorbar(xs + (index - 2) * .09, means, yerr=spreads, fmt='o',
                color=COLOURS[representation], capsize=3, markersize=4,
                label=presentation_label('', representation))
        axis.set(xticks=xs, xticklabels=('Fine-tuned', 'From scratch', 'Frozen backbone'),
                 ylabel=title, xlim=(-.4, 2.4))
        axis.grid(True, axis='y', color='0.85', alpha=.6)
        axis.tick_params(direction='in', top=True, right=True)
        axis.legend(fontsize=8, frameon=False, loc='best')
    if show_note:
        figure.suptitle('Three-run means with +/-1 sample SD; each run uses the same held-out events', fontsize=11)
    return figure


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_condition(groups, reports, curves, provenance, output, summaries, averaged,
                     *, xscale, fpr_min, auc_digits, formats):
    condition = groups[0].condition
    output.mkdir(parents=True, exist_ok=False)
    for scale in (('linear', 'log') if xscale == 'both' else (xscale,)):
        # The renderer is seed-agnostic for n=3. Override its historical default
        # subtitle rather than passing scratch seeds as pretraining seeds.
        figure = roc_figure(groups, summaries, averaged, curves, xscale=scale,
            fpr_min=fpr_min, auc_digits=auc_digits,
            subtitle=f'{CONDITION_TITLES[condition]}. Mean ROC +/-1 sample SD; not a confidence interval.')
        save_figure(figure, output, f'multiclass_roc_seed_uncertainty_{scale}', formats)
    figure = matrix_figure(groups, summaries)
    figure.suptitle(CONDITION_TITLES[condition], fontsize=11)
    save_figure(figure, output, 'multiclass_confusion_seed_means', formats)
    run_rows, summary_rows, auc_rows, matrix_rows, numeric = [], [], [], [], {}
    for group in groups:
        result = summaries[group.id]
        base = dict(model_id=group.id, model=group.label, representation=group.representation,
            condition=condition, varying_seed=result['varying_seed'],
            seeds=';'.join(map(str, group.expected_seeds)), n_seeds=len(group.runs),
            test_events_per_seed=result['test_events_per_seed'])
        row = dict(base, pretrain_seeds=None if condition == 'scratch' else base['seeds'],
            fixed_training_seed=result['fixed_training_seed'])
        for key in ('accuracy', 'balanced_accuracy', 'macro_ovr_auc', 'weighted_ovr_auc'):
            row[f'{key}_mean'], row[f'{key}_sd'] = result[key]['mean'], result[key]['sd']
        summary_rows.append(row)
        for member in group.runs:
            report = reports[member.run.id]
            run_rows.append(dict(model_id=group.id, model=group.label, condition=condition,
                run_id=member.run.id, pretrain_seed=member.pretrain_seed, training_seed=member.training_seed,
                varying_seed=result['varying_seed'], test_events=report['test_events'],
                **{key: report[key] for key in ('accuracy', 'balanced_accuracy', 'macro_ovr_auc', 'weighted_ovr_auc')},
                **{f'auc_{name}': report['per_class_auc'][name] for name in NAMES}))
            for name in NAMES:
                for key, values in curves[member.run.id][name].items():
                    numeric[f'{member.run.id}__{name}__{key}'] = values
        for name in NAMES:
            area = result['per_class_auc'][name]
            auc_rows.append(dict(base, class_name=name, auc_mean=area['mean'], auc_sd=area['sd']))
            for key, values in averaged[group.id][name].items():
                numeric[f'{group.id}__mean__{name}__{key}'] = values
        for key in ('mean', 'sd'):
            numeric[f'{group.id}__confusion_{key}'] = np.asarray(result[f'normalized_confusion_{key}'])
        for i, true_name in enumerate(NAMES):
            for j, predicted in enumerate(NAMES):
                matrix_rows.append(dict(base, true_class=true_name, predicted_class=predicted,
                    fraction_mean=result['normalized_confusion_mean'][i][j],
                    fraction_sd=result['normalized_confusion_sd'][i][j]))
    for name, rows in (('per_run_metrics.csv', run_rows), ('classification_seed_summary.csv', summary_rows),
            ('per_class_auc_seed_summary.csv', auc_rows), ('confusion_seed_mean_sd.csv', matrix_rows)):
        _csv(output / name, rows)
    with (output / 'roc_seed_mean_sd.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model_id', 'condition', 'class_name', 'n_seeds', 'false_positive_rate',
                         'true_positive_rate_mean', 'true_positive_rate_sd'])
        for group in groups:
            for name in NAMES:
                curve = averaged[group.id][name]
                writer.writerows((group.id, condition, name, len(group.runs), x, y, sd)
                    for x, y, sd in zip(curve['fpr'], curve['tpr_mean'], curve['tpr_sd'], strict=True))
    with (output / 'roc_individual_points.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run_id', 'condition', 'class_name', 'false_positive_rate', 'true_positive_rate', 'threshold'])
        for group in groups:
            for member in group.runs:
                for name in NAMES:
                    curve = curves[member.run.id][name]
                    writer.writerows((member.run.id, condition, name, x, y, threshold)
                        for x, y, threshold in zip(curve['fpr'], curve['tpr'], curve['thresholds'], strict=True))
    np.savez_compressed(output / 'seed_uncertainty_arrays.npz', **numeric)
    identifiers = [member.run.id for group in groups for member in group.runs]
    audit = dict(class_names=NAMES, condition=condition,
        task='one-vs-rest; the other four classes are negatives',
        seed_definition=CONDITION_TITLES[condition],
        uncertainty='pointwise +/-1 sample standard deviation (ddof=1), not a confidence interval',
        auc='Mean and sample SD of individual raw-ROC AUCs; NOT area under the mean interpolated ROC; no probability ensemble',
        roc_interpolation='raw polyline with duplicate-FPR vertical edges; unchanged shared linear/log grid',
        band_clipping='Displayed bands only are clipped to [0,1]; exported means and SDs are unchanged',
        confusion='Mean and sample SD of individually row-normalized matrices; N is per run, not summed over seeds',
        all_models_have_verified_event_ids=all(provenance[identifier]['event_ids_available'] for identifier in identifiers),
        models=[dict(id=group.id, label=group.label, representation=group.representation,
            model_family=group.model_family,
            representation_configuration_id=group.pretraining_configuration_id,
            pretraining_configuration_id=None if condition == 'scratch' else group.pretraining_configuration_id,
            condition=condition, caveats=list(group.caveats)) for group in groups],
        per_run_metrics={identifier: reports[identifier] for identifier in identifiers},
        summaries={group.id: summaries[group.id] for group in groups},
        provenance={identifier: provenance[identifier] for identifier in identifiers})
    (output / 'seed_uncertainty_summary.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')


def write_outputs(groups, reports, curves, provenance, output, *, grid=None, xscale='both',
                  fpr_min=1e-5, auc_digits=3, formats=('pdf', 'png')):
    output = Path(output)
    if len(groups) != 15 or len(reports) != 45:
        raise ValueError('Refuse incomplete comparisons; require all 45 saved evaluations')
    grid = fpr_grid(fpr_min=fpr_min) if grid is None else grid
    summaries, averaged = aggregate(groups, reports, curves, grid)
    output.mkdir(parents=True, exist_ok=False)
    for condition in CONDITIONS:
        selected = [group for group in groups if group.condition == condition]
        _write_condition(selected, reports, curves, provenance, output / condition, summaries, averaged,
            xscale=xscale, fpr_min=fpr_min, auc_digits=auc_digits, formats=formats)
    save_figure(summary_figure(groups, summaries), output, 'classification_summary_mean_sd', formats)
    summary_rows = []
    for group in groups:
        result = summaries[group.id]
        row = dict(model=group.label, model_id=group.id, condition=group.condition,
            varying_seed=result['varying_seed'], n_seeds=result['n_seeds'],
            test_events_per_seed=result['test_events_per_seed'])
        for metric in ('macro_ovr_auc', 'accuracy', 'balanced_accuracy', 'weighted_ovr_auc'):
            row[f'{metric}_mean'], row[f'{metric}_sd'] = result[metric]['mean'], result[metric]['sd']
        summary_rows.append(row)
    _csv(output / 'classification_summary.csv', summary_rows)
    summary = dict(task='five models x three conditions x three seeds; 45 saved evaluations',
        class_names=NAMES, conditions=CONDITIONS, no_training_or_inference=True,
        numeric_probabilities_ensembled=False, role_off_and_hybrid_excluded=True,
        input_files={member.run.id: dict(path=str(member.run.path),
            predictions_sha256=provenance[member.run.id]['predictions_sha256'])
            for group in groups for member in group.runs},
        condition_outputs={condition: str(output / condition) for condition in CONDITIONS})
    (output / 'conditions_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Wrote three separate five-model ROC/confusion figure sets and numeric audits: {output}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--xscale', choices=('linear', 'log', 'both'), default='both')
    parser.add_argument('--grid-points', type=int, default=2001)
    parser.add_argument('--fpr-min', type=float, default=1e-5)
    parser.add_argument('--auc-digits', type=int, choices=(3, 4), default=3)
    parser.add_argument('--formats', nargs='+', choices=('pdf', 'png', 'svg'), default=['pdf', 'png'])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    grid = fpr_grid(args.grid_points, args.fpr_min)
    config, root, groups = read_config(args.config, args.data_root)
    reports, curves, provenance = load(groups, config['expected_test_events'])
    check_expected_replacement_statistics(config, groups, reports)
    print(f"PASS: 45 saved runs, five models, three distinct conditions; {config['expected_test_events']:,} test events per run")
    for group in groups:
        for note in group.caveats:
            print(f'NOTE {group.label}/{group.condition}: {note}')
    if any(not item['event_ids_available'] for item in provenance.values()):
        print('NOTE: legacy fine-tuned Q8 seed 42 has manifest/input-digest provenance, not saved event IDs.')
    if args.check_only:
        print('Check only: no plots/files written, no model/checkpoint loading.'); return
    write_outputs(groups, reports, curves, provenance, args.output_dir or root / config['output_dir'],
        grid=grid, xscale=args.xscale, fpr_min=args.fpr_min, auc_digits=args.auc_digits, formats=args.formats)


if __name__ == '__main__':
    main()
