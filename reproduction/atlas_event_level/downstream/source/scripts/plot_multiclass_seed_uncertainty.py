#!/usr/bin/env python3
"""Plot saved five-class studies with pretraining-seed variability, never an ensemble.

JSON schema v1: class_names, expected_test_events, output_dir, data_root and
models[]. Each model has id/label/representation/model_family,
pretraining_configuration_id, expected_pretrain_seeds and runs[]. Each run has
id/path/pretrain_seed/finetune_seed, optionally kind and pretraining_job.
Only explicitly marked legacy_seed42_reference runs may omit new provenance.
All requested seeds must exist. Fine-tuning seed is fixed to 42 throughout.

Dependencies: NumPy, scikit-learn, matplotlib; no PyArrow, torch or inference.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_multiclass_studies import (
    COLOURS, DISPLAY, NAMES, Run, load_studies, main_figure_items, presentation_label, save_figure,
)


@dataclass(frozen=True)
class SeedRun:
    run: Run
    pretrain_seed: int
    finetune_seed: int
    legacy_seed42_reference: bool = False
    pretraining_job: str | None = None


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    representation: str
    model_family: str
    pretraining_configuration_id: str
    expected_pretrain_seeds: tuple[int, ...]
    runs: tuple[SeedRun, ...]
    caveats: tuple[str, ...] = ()


def _seed(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def read_config(path, data_root=None):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get('version') != 1 or tuple(config.get('class_names', ())) != NAMES:
        raise ValueError('Require schema version 1 and the original five class names/order')
    expected = config.get('expected_test_events')
    if not _seed(expected) or expected < 5:
        raise ValueError('Specify expected_test_events explicitly')
    root = Path(data_root).resolve() if data_root else (path.parent / config.get('data_root', '.')).resolve()
    models, run_ids, model_ids, paths = [], set(), set(), set()
    for group in config.get('models', []):
        group_id = group.get('id', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', group_id) or group_id in model_ids:
            raise ValueError('Model IDs must be safe and unique')
        model_ids.add(group_id)
        if group.get('representation') not in COLOURS or not group.get('label'):
            raise ValueError(f'{group_id}: require a known representation and label')
        for key in ('model_family', 'pretraining_configuration_id'):
            if not isinstance(group.get(key), str) or not group[key]:
                raise ValueError(f'{group_id}: specify {key}')
        expected_seeds = group.get('expected_pretrain_seeds', [])
        if not expected_seeds or not all(_seed(x) for x in expected_seeds) or len(set(expected_seeds)) != len(expected_seeds):
            raise ValueError(f'{group_id}: expected seeds must be distinct nonnegative integers')
        members = []
        for item in group.get('runs', []):
            run_id = item.get('id', '')
            if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id) or run_id in run_ids:
                raise ValueError('Run IDs must be safe and unique')
            run_ids.add(run_id)
            seed, fine = item.get('pretrain_seed'), item.get('finetune_seed')
            if not _seed(seed) or not _seed(fine) or fine != 42:
                raise ValueError(f'{run_id}: require pretrain_seed and fixed finetune_seed=42')
            legacy = item.get('legacy_seed42_reference', False)
            if not isinstance(legacy, bool) or (legacy and seed != 42):
                raise ValueError('Only an explicitly marked seed-42 reference may omit metadata')
            kind = item.get('kind', 'predictions')
            if kind not in {'predictions', 'q8_paired_reference'}:
                raise ValueError(f'{run_id}: unsupported prediction format')
            if kind == 'q8_paired_reference' and not (group['representation'] == 'q8' and legacy):
                raise ValueError('The older paired Q8 format is only a legacy seed-42 Q8 reference')
            if not isinstance(item.get('path'), str) or not item['path']:
                raise ValueError(f'{run_id}: specify an input evaluation directory')
            directory = (root / item['path']).resolve()
            if directory in paths:
                raise ValueError('The same prediction directory cannot stand for multiple runs')
            paths.add(directory)
            label = presentation_label(group['label'], group['representation'])
            run = Run(run_id, f"{label} pretrain seed {seed}", group['representation'],
                      'pretrained', directory, kind)
            members.append(SeedRun(run, seed, fine, legacy, item.get('pretraining_job')))
        seeds = [member.pretrain_seed for member in members]
        if len(seeds) != len(set(seeds)) or sorted(seeds) != sorted(expected_seeds):
            raise ValueError(f'{group_id}: require complete expected seed set {expected_seeds}; found {seeds}')
        models.append(Model(group_id, presentation_label(group['label'], group['representation']), group['representation'], group['model_family'],
            group['pretraining_configuration_id'], tuple(sorted(expected_seeds)),
            tuple(sorted(members, key=lambda member: member.pretrain_seed)), tuple(group.get('caveats', []))))
    if not models:
        raise ValueError('Specify at least one model')
    return config, root, models


def check_seed_metadata(models):
    """Do not confuse a new configuration or the same checkpoint with a new seed."""
    audit = {}
    for model in models:
        hashes = set()
        for member in model.runs:
            path = member.run.path / 'evaluation_summary.json'
            if not path.is_file():
                if not member.legacy_seed42_reference:
                    raise FileNotFoundError(f'{member.run.id}: missing {path}')
                report = {}
            else:
                report = json.loads(path.read_text())
                if not isinstance(report, dict):
                    raise ValueError(f'{member.run.id}: invalid evaluation summary')
            required = dict(pretrain_seed=member.pretrain_seed, finetune_seed=42,
                model_family=model.model_family,
                pretraining_configuration_id=model.pretraining_configuration_id)
            missing = []
            for key, value in required.items():
                if key not in report:
                    missing.append(key)
                elif report[key] != value or (key.endswith('_seed') and not _seed(report[key])):
                    raise ValueError(f'{member.run.id}: conflicting {key}; expected {value!r}')
            for key in ('pretraining_checkpoint_sha256', 'pretraining_job'):
                if key not in report:
                    missing.append(key)
            if missing and not member.legacy_seed42_reference:
                raise ValueError(f'{member.run.id}: missing seed/configuration provenance: {missing}')
            digest = report.get('pretraining_checkpoint_sha256')
            if digest is not None:
                if not isinstance(digest, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', digest):
                    raise ValueError(f'{member.run.id}: invalid pretraining checkpoint checksum')
                if digest.lower() in hashes:
                    raise ValueError(f'{model.id}: repeated pretrained checkpoint across seeds')
                hashes.add(digest.lower())
            job = report.get('pretraining_job')
            if job is not None and (not isinstance(job, str) or not job):
                raise ValueError(f'{member.run.id}: invalid pretraining_job')
            if job is not None and member.pretraining_job is not None and job != member.pretraining_job:
                raise ValueError(f'{member.run.id}: unexpected pretraining_job')
            observed = {key: report[key] for key in (*required, 'pretraining_checkpoint_sha256',
                'pretraining_job', 'pretraining_topology') if key in report}
            if 'pretraining_topology' not in observed:
                topology = {key: report[key] for key in
                    ('pretraining_num_devices', 'pretraining_batch_size_per_gpu', 'pretraining_global_batch')
                    if key in report}
                if topology:
                    observed['pretraining_topology'] = topology
            audit[member.run.id] = dict(expected=required, expected_pretraining_job=member.pretraining_job,
                observed=observed,
                legacy_metadata_not_available=missing,
                configuration_check='saved metadata' if not missing else 'explicit legacy configuration assertion; not independently verified')
    return audit


def load(models, expected_events):
    runs = [member.run for model in models for member in model.runs]
    # Existing strict reader checks files, scores, class order, event membership,
    # saved metric agreement and the canonical old-Q8 manifest/input digests.
    reports, curves, provenance = load_studies(runs, expected_events)
    metadata = check_seed_metadata(models)
    for run_id in provenance:
        provenance[run_id]['seed_provenance'] = metadata[run_id]
    return reports, curves, provenance


def fpr_grid(points=2001, fpr_min=1e-5):
    if points < 20 or not 0 < fpr_min < 1:
        raise ValueError('Require >=20 grid points and 0<fpr_min<1')
    positive = np.unique(np.concatenate((np.linspace(0, 1, points)[1:],
                                        np.geomspace(fpr_min, 1, points))))
    # Repeated FPR=0 records both the origin and the top of a vertical ROC edge.
    return np.concatenate(([0.0, 0.0], positive))


def interpolate_roc(curve, grid):
    """Interpolate the raw ROC polyline, retaining vertical-edge semantics.

    Collapsing duplicate FPRs to their maximum TPR before interpolation would
    wrongly raise the preceding sloped segment. Instead a point between FPRs
    interpolates from the previous vertical edge's top to the next edge's
    bottom. At an exact FPR use the upper/right-continuous value. Only the first
    of the two zero-FPR grid points is fixed to (0,0); the next stores the edge.
    Per-run AUCs are always from the original ROC, never this plotting grid.
    """
    x, y, grid = np.asarray(curve['fpr']), np.asarray(curve['tpr']), np.asarray(grid)
    if x.ndim != 1 or len(x) < 2 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all() \
            or np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0) \
            or np.any((x < 0) | (x > 1)) or np.any((y < 0) | (y > 1)) \
            or x[0] != 0 or x[-1] != 1 or y[-1] != 1:
        raise ValueError('Invalid monotone ROC endpoints')
    if grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all() or grid[0] != 0 or grid[1] != 0 or grid[-1] != 1 \
            or np.any(np.diff(grid) < 0):
        raise ValueError('Require an ordered common grid with two zero-FPR endpoints')
    left = np.searchsorted(x, grid, side='right') - 1
    right = np.minimum(left + 1, len(x) - 1)
    result = y[left].astype(float, copy=True)
    between = (x[left] != grid) & (right > left)
    weight = (grid[between] - x[left[between]]) / (x[right[between]] - x[left[between]])
    result[between] += weight * (y[right[between]] - y[left[between]])
    result[0], result[-1] = 0.0, 1.0
    return result


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def aggregate(models, reports, curves, grid):
    summaries, mean_curves = {}, {}
    for model in models:
        n = len(model.runs)
        result = dict(n_seeds=n, pretrain_seeds=list(model.expected_pretrain_seeds), finetune_seed=42,
            test_events_per_seed=reports[model.runs[0].run.id]['test_events'],
            class_counts=reports[model.runs[0].run.id]['class_counts'],
            uncertainty='sample standard deviation across pretraining runs (ddof=1)' if n > 1 else 'not estimated: one pretraining seed',
            per_class_auc={})
        for key in ('accuracy', 'balanced_accuracy', 'macro_ovr_auc', 'weighted_ovr_auc'):
            average, spread = mean_sd([reports[member.run.id][key] for member in model.runs])
            result[key] = dict(mean=average, sd=spread)
        mean_curves[model.id] = {}
        for name in NAMES:
            average, spread = mean_sd([reports[member.run.id]['per_class_auc'][name] for member in model.runs])
            result['per_class_auc'][name] = dict(mean=average, sd=spread)
            interpolated = np.asarray([interpolate_roc(curves[member.run.id][name], grid) for member in model.runs])
            mean_curves[model.id][name] = dict(fpr=grid, tpr_mean=interpolated.mean(0),
                tpr_sd=interpolated.std(0, ddof=1) if n > 1 else np.full(len(grid), np.nan))
        matrices = np.asarray([reports[member.run.id]['normalized_confusion_matrix'] for member in model.runs])
        result['normalized_confusion_mean'] = matrices.mean(0).tolist()
        result['normalized_confusion_sd'] = matrices.std(0, ddof=1).tolist() if n > 1 else None
        summaries[model.id] = result
    return summaries, mean_curves


def roc_figure(models, summaries, mean_curves, curves, *, xscale='linear', fpr_min=1e-5, auc_digits=3,
               include_hybrid=False, subtitle=None, clean_labels=False):
    models = list(models) if include_hybrid else main_figure_items(models)
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 10.4), constrained_layout=True)
    for axis, name, display in zip(axes.flat, NAMES, DISPLAY):
        for model in models:
            result = summaries[model.id]
            area = result['per_class_auc'][name]
            colour, label = COLOURS[model.representation], presentation_label(model.label, model.representation)
            if result['n_seeds'] > 1:
                curve = mean_curves[model.id][name]
                x, y, sd = curve['fpr'], curve['tpr_mean'], curve['tpr_sd']
                keep = x > 0 if xscale == 'log' else np.ones(len(x), dtype=bool)
                count = '' if clean_labels else f"; n={result['n_seeds']}"
                legend = f"{label} (AUC={area['mean']:.{auc_digits}f} ± {area['sd']:.{auc_digits}f}{count})"
                axis.fill_between(x[keep], np.clip(y[keep] - sd[keep], 0, 1),
                    np.clip(y[keep] + sd[keep], 0, 1), color=colour, alpha=0.16, linewidth=0)
            else:
                curve = curves[model.runs[0].run.id][name]
                x, y = curve['fpr'], curve['tpr']
                keep = x > 0 if xscale == 'log' else np.ones(len(x), dtype=bool)
                legend = f"{label} (AUC={area['mean']:.{auc_digits}f}; seed {model.expected_pretrain_seeds[0]} only)"
            axis.plot(x[keep], y[keep], color=colour, lw=2, label=legend)
        diagonal = np.geomspace(fpr_min, 1, 100) if xscale == 'log' else [0, 1]
        axis.plot(diagonal, diagonal, '--', color='0.55', lw=1, label='Random')
        axis.set(xscale=xscale, xlim=(fpr_min if xscale == 'log' else 0, 1), ylim=(0, 1.02),
            xlabel='False Positive Rate', ylabel='True Positive Rate', title=f'{display} vs rest')
        axis.grid(True, which='major', color='0.85', alpha=0.6)
        axis.tick_params(direction='in', top=True, right=True)
        legend_location = 'upper left' if xscale == 'log' and name != 'tth' else 'lower right'
        axis.legend(loc=legend_location, fontsize=8, frameon=xscale == 'log',
                    facecolor='white', edgecolor='none', framealpha=.95)
    axes.flat[-1].set_visible(False)
    if subtitle or not clean_labels:
        figure.suptitle(subtitle if subtitle is not None else
            'Pretraining-seed variability: mean ROC ±1 sample SD; fine-tuning seed 42. Single-seed curves have no band.', fontsize=11)
    return figure


def matrix_figure(models, summaries, *, include_hybrid=False, clean_labels=False):
    models = list(models) if include_hybrid else main_figure_items(models)
    columns = min(3, len(models)); rows = (len(models) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5.9 * rows),
                               squeeze=False, constrained_layout=True)
    active = []
    for axis, model in zip(axes.flat, models):
        report = summaries[model.id]
        matrix = np.asarray(report['normalized_confusion_mean'])
        image = axis.imshow(matrix, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        title = presentation_label(model.label, model.representation)
        if not clean_labels:
            description = f"mean of {report['n_seeds']} seeds" if report['n_seeds'] > 1 else f"seed {model.expected_pretrain_seeds[0]} only"
            title += f" | {description}\nN = {report['test_events_per_seed']:,} per seed"
        axis.set(xticks=range(5), yticks=range(5), xticklabels=DISPLAY, yticklabels=DISPLAY,
            xlabel='Predicted class', ylabel='True class',
            title=title)
        plt.setp(axis.get_xticklabels(), rotation=35, ha='right', fontsize=10)
        axis.tick_params(length=0)
        for i in range(5):
            for j in range(5):
                axis.text(j, i, f'{100 * matrix[i, j]:.1f}%', ha='center', va='center',
                          color='white' if matrix[i, j] >= 0.5 else 'black', fontsize=10)
        active.append(axis)
    for axis in list(axes.flat)[len(models):]:
        axis.set_visible(False)
    figure.colorbar(image, ax=active, label=('Fraction of true class' if clean_labels else
        'Fraction of true class (mean across seeds where available)'), shrink=0.85)
    return figure


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_outputs(models, reports, curves, provenance, output, *, grid=None, xscale='both',
                  fpr_min=1e-5, auc_digits=3, formats=('pdf', 'png')):
    grid = fpr_grid(fpr_min=fpr_min) if grid is None else grid
    summaries, averaged = aggregate(models, reports, curves, grid)
    main_models = main_figure_items(models)
    has_appendix = any(model.representation == 'hybrid' for model in models)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for scale in (('linear', 'log') if xscale == 'both' else (xscale,)):
        save_figure(roc_figure(models, summaries, averaged, curves, xscale=scale,
            fpr_min=fpr_min, auc_digits=auc_digits), output, f'multiclass_roc_seed_uncertainty_{scale}', formats)
        if has_appendix:
            save_figure(roc_figure(models, summaries, averaged, curves, xscale=scale,
                fpr_min=fpr_min, auc_digits=auc_digits, include_hybrid=True), output,
                f'appendix_multiclass_roc_seed_uncertainty_{scale}', formats)
    save_figure(matrix_figure(models, summaries), output, 'multiclass_confusion_seed_means', formats)
    if has_appendix:
        save_figure(matrix_figure(models, summaries, include_hybrid=True), output,
                    'appendix_multiclass_confusion_seed_means', formats)
    run_rows, summary_rows, auc_rows, matrix_rows, mean_rows, numeric = [], [], [], [], [], {}
    for model in models:
        summary = summaries[model.id]
        base = dict(model_id=model.id, model=presentation_label(model.label, model.representation), representation=model.representation,
            n_seeds=summary['n_seeds'], pretrain_seeds=';'.join(map(str, summary['pretrain_seeds'])),
            finetune_seed=42, test_events_per_seed=summary['test_events_per_seed'])
        summary_row = dict(base)
        for key in ('accuracy', 'balanced_accuracy', 'macro_ovr_auc', 'weighted_ovr_auc'):
            summary_row[f'{key}_mean'] = summary[key]['mean']; summary_row[f'{key}_sd'] = summary[key]['sd']
        summary_rows.append(summary_row)
        for member in model.runs:
            report = reports[member.run.id]
            run_rows.append(dict(model_id=model.id, run_id=member.run.id, pretrain_seed=member.pretrain_seed,
                finetune_seed=42, test_events=report['test_events'], accuracy=report['accuracy'],
                balanced_accuracy=report['balanced_accuracy'], macro_ovr_auc=report['macro_ovr_auc'],
                weighted_ovr_auc=report['weighted_ovr_auc'],
                **{f'auc_{name}': report['per_class_auc'][name] for name in NAMES}))
            for name in NAMES:
                for key, value in curves[member.run.id][name].items():
                    numeric[f'{member.run.id}__{name}__{key}'] = value
        for name in NAMES:
            area = summary['per_class_auc'][name]
            auc_rows.append(dict(base, class_name=name, auc_mean=area['mean'], auc_sd=area['sd']))
            curve = averaged[model.id][name]
            for key, value in curve.items():
                numeric[f'{model.id}__mean__{name}__{key}'] = value
            mean_rows.extend(dict(model_id=model.id, class_name=name, n_seeds=summary['n_seeds'],
                false_positive_rate=x, true_positive_rate_mean=y,
                true_positive_rate_sd=sd if summary['n_seeds'] > 1 else None) for x, y, sd in
                zip(curve['fpr'], curve['tpr_mean'], curve['tpr_sd'], strict=True))
        numeric[f'{model.id}__confusion_mean'] = np.asarray(summary['normalized_confusion_mean'])
        numeric[f'{model.id}__confusion_sd'] = (np.asarray(summary['normalized_confusion_sd'])
            if summary['n_seeds'] > 1 else np.full((5, 5), np.nan))
        for i, true_name in enumerate(NAMES):
            for j, predicted in enumerate(NAMES):
                matrix_rows.append(dict(base, true_class=true_name, predicted_class=predicted,
                    fraction_mean=summary['normalized_confusion_mean'][i][j],
                    fraction_sd=summary['normalized_confusion_sd'][i][j] if summary['n_seeds'] > 1 else None))
    for filename, rows in (('per_run_metrics.csv', run_rows), ('classification_seed_summary.csv', summary_rows),
        ('per_class_auc_seed_summary.csv', auc_rows), ('confusion_seed_mean_sd.csv', matrix_rows),
        ('roc_seed_mean_sd.csv', mean_rows)):
        _csv(output / filename, rows)
    # Full held-out curves can have millions of points: stream rather than
    # materializing a large list of per-point Python dictionaries in memory.
    with (output / 'roc_individual_points.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run_id', 'class_name', 'false_positive_rate', 'true_positive_rate', 'threshold'])
        for model in models:
            for member in model.runs:
                for name in NAMES:
                    curve = curves[member.run.id][name]
                    writer.writerows((member.run.id, name, x, y, threshold) for x, y, threshold in
                        zip(curve['fpr'], curve['tpr'], curve['thresholds'], strict=True))
    np.savez_compressed(output / 'seed_uncertainty_arrays.npz', **numeric)
    audit = dict(class_names=NAMES, task='one-vs-rest; all four other classes are negatives',
        estimand='pretraining-run variability conditional on fine-tuning seed 42; not a confidence interval or ensemble',
        uncertainty='pointwise sample standard deviation, ddof=1; not standard error',
        auc='mean and sample SD of individual raw-ROC AUCs, NOT area under the mean interpolated ROC',
        roc_interpolation='common linear+log grid; duplicate-FPR vertical edges handled on raw polyline; two FPR=0 grid entries encode origin and upper edge',
        band_clipping='Only displayed mean±SD bands clipped to [0,1]; numeric mean and SD are unchanged',
        single_seed='No uncertainty estimate/band; SD is null in JSON/CSV and NaN in NPZ',
        confusion='Arithmetic mean and sample SD of per-run row-normalized matrices; N is events per run, not summed over seeds',
        input_membership='Exact sorted event IDs/labels except explicitly recorded legacy Q8 source-manifest provenance',
        all_models_have_verified_event_ids=all(item.get('event_ids_available') for item in provenance.values()),
        models=[dict(id=model.id, label=presentation_label(model.label, model.representation), representation=model.representation,
            model_family=model.model_family, pretraining_configuration_id=model.pretraining_configuration_id,
            expected_pretrain_seeds=list(model.expected_pretrain_seeds), caveats=list(model.caveats),
            input_dirs={member.run.id: str(member.run.path) for member in model.runs}) for model in models],
        per_run_metrics=reports, summaries=summaries, provenance=provenance,
        figure_models=dict(main=[model.id for model in main_models],
            appendix=[model.id for model in models] if has_appendix else []))
    (output / 'seed_uncertainty_summary.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(f'Wrote seed-mean ROC/SD bands, consistent confusion matrices and numeric per-run/aggregate exports: {output}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--xscale', choices=('linear', 'log', 'both'), default='both')
    parser.add_argument('--fpr-min', type=float, default=1e-5)
    parser.add_argument('--grid-points', type=int, default=2001)
    parser.add_argument('--auc-digits', type=int, choices=(3, 4), default=3)
    parser.add_argument('--formats', nargs='+', choices=('pdf', 'png', 'svg'), default=['pdf', 'png'])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    grid = fpr_grid(args.grid_points, args.fpr_min)
    config, root, models = read_config(args.config, args.data_root)
    reports, curves, provenance = load(models, config['expected_test_events'])
    print(f"PASS: {len(models)} model groups, {sum(len(model.runs) for model in models)} saved runs; "
          f"{config['expected_test_events']:,} identical test events per run, fine-tuning seed 42")
    for model in models:
        for caveat in model.caveats:
            print(f'NOTE {model.label}: {caveat}')
    if not all(item.get('event_ids_available') for item in provenance.values()):
        print('NOTE: old Q8 scores have canonical manifest/input-digest provenance, not saved per-event IDs.')
    if args.check_only:
        print('Check only: no files written.'); return
    write_outputs(models, reports, curves, provenance, args.output_dir or root / config['output_dir'],
        grid=grid, xscale=args.xscale, fpr_min=args.fpr_min, auc_digits=args.auc_digits, formats=args.formats)


if __name__ == '__main__':
    main()
