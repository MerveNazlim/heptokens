#!/usr/bin/env python3
"""Five-class paper ROC panels, confusion matrices and numeric summaries.

CPU only: numpy, sklearn and matplotlib; no torch, model code, inference,
training, cloud access or fitting. Read saved test_predictions.npz exports.
Colour denotes representation; solid/dashed/dotted denotes pretrained/scratch/
frozen. An older paired Q8 export is supported with its original provenance.
Main figures show the five input representations; hybrid is retained in
separate appendix figures and in all numeric/provenance exports.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_curve, auc

from plot_matched_multiclass_comparison import DISPLAY, NAMES, load_predictions, load_q8_reference

COLOURS = {
    'q1': '#D55E00', 'q4': '#CC79A7', 'q8': '#0072B2',
    'hybrid': '#8064A2', 'continuous': '#009E73', 'hierarchical': '#8C564B',
}
LINESTYLES = {'pretrained': '-', 'scratch': '--', 'frozen': ':'}
MAIN_REPRESENTATIONS = ('q1', 'q4', 'q8', 'continuous', 'hierarchical')
PRESENTATION_LABELS = {
    'q1': 'RVQ-1', 'q4': 'RVQ-4', 'q8': 'RVQ-8',
    'continuous': 'direct continuous', 'hierarchical': 'hierarchical continuous',
    'hybrid': 'Q8 hybrid',
}


@dataclass(frozen=True)
class Run:
    id: str
    label: str
    representation: str
    condition: str
    path: Path
    kind: str = 'predictions'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path, data_root=None, condition='all'):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get('version') != 1 or tuple(config.get('class_names', ())) != NAMES:
        raise ValueError('Require configuration version 1 and the original five class names/order')
    root = Path(data_root).resolve() if data_root is not None else (path.parent / config.get('data_root', '.')).resolve()
    expected = config.get('expected_test_events')
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 5:
        raise ValueError('Specify the expected number of test events explicitly')
    result = []
    for row in config['runs']:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', row.get('id', '')) or not row.get('label'):
            raise ValueError('Each run requires a safe, unique id and a nonempty label')
        if row['representation'] not in COLOURS or row['condition'] not in LINESTYLES:
            raise ValueError(f"Unknown representation/condition in {row['id']}")
        kind = row.get('kind', 'predictions')
        if kind not in {'predictions', 'q8_paired_reference'}:
            raise ValueError(f'Unsupported saved prediction format: {kind}')
        if kind == 'q8_paired_reference' and row['representation'] != 'q8':
            raise ValueError('A paired Q8 export cannot be relabelled as another representation')
        if condition == 'all' or row['condition'] == condition:
            result.append(Run(row['id'], presentation_label(row['label'], row['representation']), row['representation'], row['condition'],
                              (root / row['path']).resolve(), kind))
    if not result or len({run.id for run in result}) != len(result):
        raise ValueError('Select at least one run; run ids must be unique')
    return config, root, result


def check_files(runs):
    missing = []
    for run in runs:
        prediction = run.path / 'test_predictions.npz'
        if not prediction.is_file():
            missing.append(f'{run.label}: {prediction}')
        if run.kind == 'q8_paired_reference' and not (run.path / 'evaluation_metadata.json').is_file():
            missing.append(f'{run.label}: {run.path / "evaluation_metadata.json"}')
    if missing:
        raise FileNotFoundError('Saved evaluations are not all available yet:\n  ' + '\n  '.join(missing)
            + '\nCopy the completed test evaluations first. This script does not train or run inference.')


def metrics(arrays):
    labels, scores = arrays['labels'], arrays['probabilities']
    curves, per_class = {}, {}
    for index, name in enumerate(NAMES):
        fpr, tpr, threshold = roc_curve(labels == index, scores[:, index])
        curves[name] = dict(fpr=fpr, tpr=tpr, thresholds=threshold)
        per_class[name] = float(auc(fpr, tpr))
    predictions = scores.argmax(1)
    counts = np.bincount(labels, minlength=5)
    raw = confusion_matrix(labels, predictions, labels=range(5))
    return dict(test_events=len(labels), class_counts={name: int(counts[i]) for i, name in enumerate(NAMES)},
        accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        macro_ovr_auc=float(np.mean(list(per_class.values()))),
        weighted_ovr_auc=float(np.average(list(per_class.values()), weights=counts)),
        per_class_auc=per_class, raw_confusion_matrix=raw.tolist(),
        normalized_confusion_matrix=(raw / counts[:, None]).tolist()), curves


def check_saved_metrics(run, observed):
    path = next((run.path / name for name in ('evaluation_summary.json', 'summary.json')
                 if (run.path / name).is_file()), None)
    if path is None:
        return None
    report = json.loads(path.read_text())
    if run.kind == 'q8_paired_reference':
        if not isinstance(report, list):
            raise ValueError('The original paired Q8 summary must contain its two named branches')
        selected = [row for row in report if row.get('run') == 'Q8 parallel']
        if len(selected) != 1:
            raise ValueError('Missing/ambiguous Q8 parallel summary')
        report = selected[0]
    if not isinstance(report, dict):
        raise ValueError(f'{run.label}: unsupported saved evaluation report')
    if report.get('test_events', observed['test_events']) != observed['test_events']:
        raise ValueError(f'{run.label}: saved event count disagrees with the arrays')
    for key in ('accuracy', 'balanced_accuracy', 'macro_ovr_auc', 'weighted_ovr_auc'):
        if key in report and not np.isclose(report[key], observed[key], atol=1e-8, rtol=0):
            raise ValueError(f'{run.label}: saved {key} disagrees with the arrays')
    if 'test_auc' in report and not np.isclose(report['test_auc'], observed['macro_ovr_auc'], atol=1e-8, rtol=0):
        raise ValueError(f'{run.label}: saved test_auc disagrees with the five-class macro AUC')
    for name, value in report.get('per_class_auc', {}).items():
        if name in NAMES and not np.isclose(value, observed['per_class_auc'][name], atol=1e-8, rtol=0):
            raise ValueError(f'{run.label}: saved {name} AUC disagrees with the arrays')
    return dict(path=str(path), sha256=file_hash(path))


def load_studies(runs, expected_events):
    check_files(runs)
    arrays, provenance = {}, {}
    reference = None
    for run in runs:
        if run.kind != 'predictions':
            continue
        try:
            current = load_predictions(run.path)
        except KeyError as error:
            raise ValueError(f'{run.label}: missing saved arrays {error}; ordinary exports require '
                'probabilities, labels, class_names, source_file and event_index. '
                'Use q8_paired_reference only for the original paired Q8 export.') from error
        if len(current['labels']) != expected_events:
            raise ValueError(f'{run.label}: expected {expected_events} held-out events')
        if reference is not None:
            for key in ('labels', 'source_file', 'event_index'):
                if not np.array_equal(current[key], reference[key]):
                    raise ValueError(f'{run.label}: test event identities or labels differ')
        reference = current if reference is None else reference
        arrays[run.id] = current
        provenance[run.id] = dict(event_ids_available=True, membership_check='exact sorted event IDs and labels')
    if reference is None:
        raise ValueError('At least one evaluation with saved event IDs is required')
    for run in runs:
        if run.kind == 'q8_paired_reference':
            arrays[run.id], provenance[run.id] = load_q8_reference(run.path, reference)
    reports, curves = {}, {}
    for run in runs:
        reports[run.id], curves[run.id] = metrics(arrays[run.id])
        saved = check_saved_metrics(run, reports[run.id])
        provenance[run.id].update(predictions_sha256=file_hash(run.path / 'test_predictions.npz'),
                                  saved_report=saved)
    return reports, curves, provenance


def presentation_label(label, representation=None):
    """Paper names only; never change model IDs, configuration or saved scores."""
    if representation is not None:
        return PRESENTATION_LABELS[representation]
    aliases = {
        'Q1': 'q1', 'Q1 direct-256': 'q1', 'Q1 direct': 'q1',
        'Q1 projected (32→256)': 'q1', 'Q1 projected (32->256)': 'q1',
        'Q4': 'q4', 'Q8': 'q8', 'Q8 parallel': 'q8',
        'Continuous': 'continuous', 'Continuous flat': 'continuous',
        'Hierarchical': 'hierarchical', 'Hybrid': 'hybrid', 'Q8 hybrid': 'hybrid',
    }
    return PRESENTATION_LABELS[aliases[label]] if label in aliases else label


def main_figure_items(items):
    """Select the paper representations without deleting appendix results."""
    selected = [item for representation in MAIN_REPRESENTATIONS
                for item in items if item.representation == representation]
    if not selected:
        raise ValueError('No main-paper representation was selected; use include_hybrid=True for appendix-only figures')
    return selected


def legend_label(run, multiple_conditions):
    label = presentation_label(run.label, run.representation)
    return f'{label} ({run.condition})' if multiple_conditions else label


def roc_figure(runs, reports, curves, *, xscale='linear', fpr_min=1e-5, auc_digits=3,
               include_hybrid=False):
    runs = list(runs) if include_hybrid else main_figure_items(runs)
    multiple = len({run.condition for run in runs}) > 1
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 10), constrained_layout=True)
    for axis, class_name, display in zip(axes.flat, NAMES, DISPLAY):
        for run in runs:
            values = curves[run.id][class_name]
            keep = values['fpr'] > 0 if xscale == 'log' else np.ones(len(values['fpr']), dtype=bool)
            label = legend_label(run, multiple)
            area = reports[run.id]['per_class_auc'][class_name]
            axis.plot(values['fpr'][keep], values['tpr'][keep], color=COLOURS[run.representation],
                linestyle=LINESTYLES[run.condition], lw=2,
                label=f'{label} (AUC={area:.{auc_digits}f})')
        diagonal = np.geomspace(fpr_min, 1, 200) if xscale == 'log' else np.array([0, 1])
        axis.plot(diagonal, diagonal, '--', color='0.55', lw=1, label='Random')
        axis.set(xscale=xscale, xlim=(fpr_min if xscale == 'log' else 0, 1), ylim=(0, 1.02),
            xlabel='False Positive Rate', ylabel='True Positive Rate', title=f'{display} vs rest')
        axis.grid(True, which='major', color='0.85', alpha=0.6)
        axis.tick_params(direction='in', top=True, right=True)
        axis.legend(loc='lower right', fontsize=8.5, frameon=False)
    axes.flat[-1].set_visible(False)
    return figure


def matrix_figure(runs, reports, *, include_hybrid=False):
    runs = list(runs) if include_hybrid else main_figure_items(runs)
    columns = min(3, len(runs)); rows = (len(runs) + columns - 1) // columns
    figure, grid = plt.subplots(rows, columns, figsize=(6 * columns, 5.9 * rows),
                                squeeze=False, constrained_layout=True)
    multiple = len({run.condition for run in runs}) > 1
    active = []
    for axis, run in zip(grid.flat, runs):
        report = reports[run.id]
        matrix = np.asarray(report['normalized_confusion_matrix'])
        image = axis.imshow(matrix, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        axis.set(xticks=range(5), yticks=range(5), xticklabels=DISPLAY, yticklabels=DISPLAY,
            xlabel='Predicted class', ylabel='True class',
            title=legend_label(run, multiple))
        plt.setp(axis.get_xticklabels(), rotation=35, ha='right', fontsize=10)
        axis.tick_params(length=0)
        for i in range(5):
            for j in range(5):
                axis.text(j, i, f'{100 * matrix[i, j]:.1f}%', ha='center', va='center',
                          color='white' if matrix[i, j] >= 0.5 else 'black', fontsize=10)
        active.append(axis)
    for axis in list(grid.flat)[len(runs):]:
        axis.set_visible(False)
    figure.colorbar(image, ax=active, label='Fraction of true class', shrink=0.85)
    return figure


def save_figure(figure, output, stem, formats):
    for extension in formats:
        figure.savefig(output / f'{stem}.{extension}', dpi=240)
    plt.close(figure)


def write_outputs(runs, reports, curves, provenance, output, *, xscale='linear',
                  fpr_min=1e-5, auc_digits=3, formats=('pdf', 'png'), individual=True):
    output = Path(output)
    main_runs = main_figure_items(runs)
    has_appendix = any(run.representation == 'hybrid' for run in runs)
    output.mkdir(parents=True, exist_ok=False)
    scales = ('linear', 'log') if xscale == 'both' else (xscale,)
    for scale in scales:
        figure = roc_figure(runs, reports, curves, xscale=scale, fpr_min=fpr_min, auc_digits=auc_digits)
        save_figure(figure, output, f'multiclass_roc_{scale}', formats)
        if has_appendix:
            figure = roc_figure(runs, reports, curves, xscale=scale, fpr_min=fpr_min,
                                auc_digits=auc_digits, include_hybrid=True)
            save_figure(figure, output, f'appendix_multiclass_roc_{scale}', formats)
    save_figure(matrix_figure(runs, reports), output, 'multiclass_confusion_matrices', formats)
    if has_appendix:
        save_figure(matrix_figure(runs, reports, include_hybrid=True), output,
                    'appendix_multiclass_confusion_matrices', formats)
    if individual:
        for run in runs:
            save_figure(matrix_figure([run], reports, include_hybrid=True), output, f'confusion_{run.id}', formats)
    summary_rows, auc_rows, numeric = [], [], {}
    for index, run in enumerate(runs):
        report = reports[run.id]
        base = dict(run_id=run.id, model=presentation_label(run.label, run.representation), representation=run.representation, condition=run.condition,
                    test_events=report['test_events'])
        summary_rows.append(dict(base, accuracy=report['accuracy'], balanced_accuracy=report['balanced_accuracy'],
            macro_ovr_auc=report['macro_ovr_auc'], weighted_ovr_auc=report['weighted_ovr_auc'],
            **{f'auc_{name}': value for name, value in report['per_class_auc'].items()}))
        for name in NAMES:
            auc_rows.append(dict(base, class_name=name, auc=report['per_class_auc'][name]))
            for key, values in curves[run.id][name].items():
                numeric[f'run{index}_{name}_{key}'] = values
    for filename, rows in (('classification_summary.csv', summary_rows), ('per_class_auc.csv', auc_rows)):
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / 'roc_points.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['run_id', 'class', 'false_positive_rate', 'true_positive_rate', 'threshold'])
        for run in runs:
            for name in NAMES:
                curve = curves[run.id][name]
                writer.writerows((run.id, name, x, y, threshold) for x, y, threshold in
                                 zip(curve['fpr'], curve['tpr'], curve['thresholds'], strict=True))
    np.savez_compressed(output / 'roc_arrays.npz', **numeric)
    audit = dict(class_names=NAMES, task='one-vs-rest; all four other classes are negatives',
        all_models_have_verified_event_ids=all(item.get('event_ids_available') for item in provenance.values()),
        runs=[dict(id=run.id, label=presentation_label(run.label, run.representation), representation=run.representation, condition=run.condition,
            input_dir=str(run.path), colour=COLOURS[run.representation], linestyle=LINESTYLES[run.condition]) for run in runs],
        metrics=reports, provenance=provenance,
        figure_models=dict(main=[run.id for run in main_runs],
            appendix=[run.id for run in runs] if has_appendix else []),
        test_metrics_weighting='unweighted test events; macro AUC equally averages the five class AUCs',
        log_fpr_note='FPR=0 points are omitted only from log-scale display; all points enter AUC and numeric exports')
    (output / 'comparison_summary.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'Wrote five-class ROC panels, confusion matrices, AUC/accuracy tables and numeric curves to {output}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, help='Override the directory containing results/')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--condition', choices=('all', *LINESTYLES), default='all')
    parser.add_argument('--xscale', choices=('linear', 'log', 'both'), default='linear')
    parser.add_argument('--fpr-min', type=float, default=1e-5)
    parser.add_argument('--auc-digits', type=int, choices=(3, 4), default=3)
    parser.add_argument('--formats', nargs='+', choices=('pdf', 'png', 'svg'), default=['pdf', 'png'])
    parser.add_argument('--no-individual-matrices', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.fpr_min < 1:
        parser.error('--fpr-min must lie between zero and one')
    config, root, runs = read_config(args.config, args.data_root, args.condition)
    reports, curves, provenance = load_studies(runs, config['expected_test_events'])
    print(f"PASS: {len(runs)} saved models; {config['expected_test_events']:,} test events and five classes")
    if any(not item.get('event_ids_available') for item in provenance.values()):
        print('NOTE: the older Q8 reference has manifest/input-digest provenance but no saved event IDs.')
    if args.check_only:
        print('Check only: no plots or files written.'); return
    output = args.output_dir or root / config['output_dir']
    write_outputs(runs, reports, curves, provenance, output, xscale=args.xscale, fpr_min=args.fpr_min,
        auc_digits=args.auc_digits, formats=args.formats, individual=not args.no_individual_matrices)


if __name__ == '__main__':
    main()
