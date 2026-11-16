#!/usr/bin/env python3
"""Plot the 45 new four-lepton fits, optionally comparing saved inclusive scores.

No model loading, event selection, inference, training or downloads. The
authoritative selection artifact, not the first run encountered, identifies
the held-out events. All inputs are verified before any plot is written.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from plot_matched_multiclass_comparison import SOURCE_SHA, load_predictions
from plot_multiclass_conditions import (
    CONDITIONS, FAMILIES, SEEDS, ConditionModel, ConditionRun,
    check_run_metadata, read_config as read_inclusive_config, write_outputs,
)
from plot_multiclass_seed_uncertainty import fpr_grid, mean_sd
from plot_multiclass_studies import (
    COLOURS, MAIN_REPRESENTATIONS, NAMES, Run, check_saved_metrics,
    file_hash, metrics, presentation_label, save_figure,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/plots/hzz_4l_multiclass.json'
NATIVE_CONDITION = {'pretrained': 'finetune', 'scratch': 'scratch', 'frozen': 'frozen'}
CONDITION_LABEL = {'pretrained': 'Fine-tuned', 'scratch': 'Scratch', 'frozen': 'Frozen'}
SELECTED_TEST_EVENTS = 74921
INCLUSIVE_TEST_EVENTS = 364441
IDENTITY = ('source_file', 'event_index', 'labels')
SCALARS = ('macro_ovr_auc', 'accuracy', 'balanced_accuracy', 'weighted_ovr_auc')


def read_config(path):
    config = json.loads(Path(path).read_text())
    expected = dict(version=1, study_type='hzz_4l_multiclass', class_names=list(NAMES),
        expected_test_events=SELECTED_TEST_EVENTS, expected_seeds=list(SEEDS),
        conditions=list(NATIVE_CONDITION.values()),
        model_families=[FAMILIES[name][0] for name in MAIN_REPRESENTATIONS])
    require_fields(config, expected, 'four-lepton plot config')
    if not isinstance(config.get('study_dir'), str) or not config['study_dir']:
        raise ValueError('Specify the four-lepton study directory')
    return config


def require_fields(actual, expected, label):
    if not isinstance(actual, dict):
        raise ValueError(f'{label}: require a JSON object')
    for key, value in expected.items():
        # bools and floats must not compare equal to integer seed/count fields.
        if key not in actual or type(actual[key]) is not type(value) or actual[key] != value:
            raise ValueError(f'{label}: conflicting or missing {key}; expected {value!r}')


def load_membership(directory, expected_events=SELECTED_TEST_EVENTS):
    """Validate the selector's receipt and decode its test source dictionary."""
    from prepare_hzz_4l_multiclass import load_selection

    directory = Path(directory).resolve()
    receipt, data = load_selection(directory)
    if receipt['source_manifest_sha256'] != SOURCE_SHA:
        raise ValueError('Four-lepton selection is not from the original inclusive dataset')
    test_rows = data['test']
    sources = np.asarray(receipt['sources'], dtype=str)[test_rows['source_id']]
    events, labels = test_rows['event_index'], test_rows['labels']
    order = np.lexsort((events, sources))
    test = dict(source_file=sources[order], event_index=events[order], labels=labels[order])
    if len(test['labels']) != expected_events or set(test['labels']) != set(range(len(NAMES))):
        raise ValueError(f'Expected {expected_events:,} selected test events and all five classes')
    return test, dict(path=str(directory / 'membership.npz'), membership_sha256=receipt['membership_sha256'],
        selection_receipt_sha256=file_hash(directory / 'selection_receipt.json'),
        selection_id=receipt['selection_id'], predicate=receipt['predicate'],
        source_manifest_sha256=receipt['source_manifest_sha256'], split_counts=receipt['split_counts'],
        test_identity_sha256=receipt['membership_identity_sha256']['test'])


def make_groups(study_dir):
    groups = []
    for representation in MAIN_REPRESENTATIONS:
        family, configuration = FAMILIES[representation]
        label = presentation_label('', representation)
        for condition in CONDITIONS:
            members = []
            scratch = condition == 'scratch'
            for seed in SEEDS:
                run_id = f"{family}_{NATIVE_CONDITION[condition]}_{'s' if scratch else 'p'}{seed}"
                path = Path(study_dir) / 'runs' / run_id / 'test_evaluation'
                members.append(ConditionRun(Run(run_id, label, representation, condition, path),
                    pretrain_seed=None if scratch else seed, training_seed=seed if scratch else 42))
            groups.append(ConditionModel(f'{family}__{condition}', label, representation,
                family, configuration, condition, SEEDS, tuple(members)))
    return groups


def members_by_key(groups):
    members = {(group.model_family, group.condition, member.varying_seed): (group, member)
               for group in groups for member in group.runs}
    expected = {(family, condition, seed) for family, _ in FAMILIES.values()
                for condition in CONDITIONS for seed in SEEDS}
    if len(groups) != 15 or sum(len(group.runs) for group in groups) != 45 or set(members) != expected:
        raise ValueError('Require all five families x three conditions x three seeds')
    return members


def compare_identity(actual, expected, label):
    if any(not np.array_equal(actual[key], expected[key]) for key in IDENTITY):
        raise ValueError(f'{label}: event identities or labels differ from the authoritative test membership')


def checked_predictions(run):
    if run.kind not in ('predictions', 'canonical_307_predictions'):
        raise ValueError(f'{run.id}: require predictions with explicit per-event identities')
    arrays = load_predictions(run.path)
    if arrays['event_index'].dtype.kind not in 'iu' or arrays['labels'].dtype.kind not in 'iu':
        raise ValueError(f'{run.id}: require integer event indices and class IDs')
    return arrays


def load_selected(groups, reference, membership):
    members = members_by_key(groups)
    missing = [str(member.run.path / name) for _, member in members.values()
               for name in ('test_predictions.npz', 'evaluation_summary.json', '../completion.json')
               if not (member.run.path / name).is_file()]
    if missing:
        raise FileNotFoundError('All 45 selected-sample fits are required; no inclusive fallback. '
            f'Missing {len(missing)} files; first missing: {missing[0]}')
    reports, curves, provenance, checkpoint_hashes = {}, {}, {}, {}
    for (family, condition, _), (group, member) in members.items():
        run = member.run
        path = run.path / 'evaluation_summary.json'
        report = json.loads(path.read_text())
        pred_hash = file_hash(run.path / 'test_predictions.npz')
        identity = dict(run_key=run.id, model_family=family, condition=NATIVE_CONDITION[condition],
            pretrain_seed=member.pretrain_seed, training_seed=member.training_seed)
        completion_path = run.path.parent / 'completion.json'
        require_fields(json.loads(completion_path.read_text()),
            dict(passed=True, **identity, evaluation_summary_sha256=file_hash(path)), f'{run.id}/completion')
        require_fields(report, dict(**identity, configuration_id=group.pretraining_configuration_id,
            checkpoint_selection='validation-best',
            membership_sha256=membership['membership_sha256'], test_events=len(reference['labels']),
            class_names=list(NAMES), prediction_sha256=pred_hash), run.id)
        if condition == 'scratch':
            require_fields(report, {'pretraining_checkpoint_sha256': None}, run.id)
        else:
            digest = report.get('pretraining_checkpoint_sha256')
            if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
                raise ValueError(f'{run.id}: invalid or missing pretraining checkpoint checksum')
            key = family, member.pretrain_seed
            if checkpoint_hashes.setdefault(key, digest) != digest:
                raise ValueError(f'{run.id}: fine-tuned and frozen checkpoint provenance differs')
        current = checked_predictions(run)
        compare_identity(current, reference, run.id)
        reports[run.id], curves[run.id] = metrics(current)
        saved_report = check_saved_metrics(run, reports[run.id])
        provenance[run.id] = dict(event_ids_available=True, predictions_sha256=pred_hash,
            membership_check='exact sorted source_file/event_index/labels against the selected test membership',
            membership_sha256=membership['membership_sha256'], saved_report=saved_report,
            completion_sha256=file_hash(completion_path),
            seed_provenance=dict(pretrain_seed=member.pretrain_seed, training_seed=member.training_seed,
                source_checkpoint_sha256=report['pretraining_checkpoint_sha256']))
    for family, _ in FAMILIES.values():
        if len({checkpoint_hashes[family, seed] for seed in SEEDS}) != len(SEEDS):
            raise ValueError(f'{family}: pretraining seeds must use distinct checkpoints')
    return reports, curves, provenance


def subset_indices(full, selected):
    """Exact vectorised join of two lexicographically sorted identity arrays."""
    result = np.empty(len(selected['labels']), dtype=np.intp)
    full_sources, full_events = full['source_file'], full['event_index']
    selected_sources, selected_events = selected['source_file'], selected['event_index']
    for name in np.unique(selected_sources):
        i, j = np.searchsorted(full_sources, name, side='left'), np.searchsorted(full_sources, name, side='right')
        lo, hi = np.searchsorted(selected_sources, name, side='left'), np.searchsorted(selected_sources, name, side='right')
        expected = selected_events[lo:hi]
        indices = i + np.searchsorted(full_events[i:j], expected)
        if np.any(indices >= j) or not np.array_equal(full_events[indices], expected):
            raise ValueError(f'Inclusive test set does not contain all selected event identities from {name}')
        result[lo:hi] = indices
    if not np.array_equal(full['labels'][result], selected['labels']):
        raise ValueError('Inclusive and selected truth labels disagree for the same events')
    return result


def metric_values(report):
    return {**{name: report[name] for name in SCALARS},
            **{f'auc_{name}': report['per_class_auc'][name] for name in NAMES}}


def compare_inclusive(groups, selected, selected_reports, selected_provenance, baseline_config, data_root=None):
    config, _, baseline_groups = read_inclusive_config(baseline_config, data_root)
    if config['expected_test_events'] != INCLUSIVE_TEST_EVENTS:
        raise ValueError(f'Use the original {INCLUSIVE_TEST_EVENTS:,}-event inclusive benchmark')
    inclusive = members_by_key(baseline_groups)
    pairs, audits, reference, indices = [], {}, None, None
    for key, (selected_group, selected_member) in members_by_key(groups).items():
        group, member = inclusive[key]
        run = member.run
        if group.pretraining_configuration_id != selected_group.pretraining_configuration_id:
            raise ValueError(f'{run.id}: inclusive/selected model configurations differ')
        metadata = check_run_metadata(group, member)
        selected_sha = selected_provenance[selected_member.run.id]['seed_provenance']['source_checkpoint_sha256']
        old_sha = metadata['source_checkpoint_sha256']
        if old_sha is not None and selected_sha != old_sha:
            raise ValueError(f'{run.id}: inclusive/selected pretrained checkpoints differ')
        current = checked_predictions(run)
        if len(current['labels']) != INCLUSIVE_TEST_EVENTS:
            raise ValueError(f'{run.id}: wrong inclusive test population size')
        if reference is None:
            reference = {key: current[key] for key in IDENTITY}
            indices = subset_indices(reference, selected)
        else:
            compare_identity(current, reference, run.id)
        full_report, _ = metrics(current)
        if run.kind == 'canonical_307_predictions':
            from multiclass_canonical_307_adapter import check_canonical_metrics
            check_canonical_metrics(group, member, full_report, current)
        else:
            check_saved_metrics(run, full_report)
        restricted_report, _ = metrics({name: values[indices] for name, values in current.items()})
        a, b, c = map(metric_values, (full_report, restricted_report, selected_reports[selected_member.run.id]))
        for name in a:
            pairs.append(dict(model_family=key[0], condition=NATIVE_CONDITION[key[1]], seed=key[2], metric=name,
                inclusive_run=run.id, selected_run=selected_member.run.id,
                inclusive_fit_inclusive_test=a[name], inclusive_fit_selected_test=b[name],
                selected_fit_selected_test=c[name], test_population_delta=b[name] - a[name],
                training_regime_delta=c[name] - b[name], overall_delta=c[name] - a[name]))
        audits[selected_member.run.id] = dict(inclusive_run=run.id, path=str(run.path),
            inclusive_predictions_sha256=file_hash(run.path / 'test_predictions.npz'),
            source_checkpoint_match='not applicable (scratch)' if key[1] == 'scratch' else
                ('verified' if old_sha is not None else 'legacy inclusive source checksum unavailable'),
            inclusive_test_events=INCLUSIVE_TEST_EVENTS, selected_test_events=len(indices),
            selected_subset_verified_by_ids_and_labels=True)
    return pairs, audits


def comparison_summary(pairs):
    by_key = {}
    for row in pairs:
        key = tuple(row[name] for name in ('model_family', 'condition', 'metric'))
        by_key.setdefault(key, []).append(row)
    results = []
    for (family, condition, metric), rows in by_key.items():
        if sorted(row['seed'] for row in rows) != list(SEEDS):
            raise ValueError(f'Incomplete inclusive/selected seed pairs for {family}/{condition}/{metric}')
        result = dict(model_family=family, condition=condition, metric=metric, n_seed_pairs=len(rows))
        for key in ('inclusive_fit_inclusive_test', 'inclusive_fit_selected_test',
                    'selected_fit_selected_test', 'test_population_delta', 'training_regime_delta', 'overall_delta'):
            result[f'{key}_mean'], result[f'{key}_sd'] = mean_sd([row[key] for row in rows])
        results.append(result)
    return results


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(output, pairs, audits, formats):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(exist_ok=False)
    summary = comparison_summary(pairs)
    write_csv(output / 'paired_run_comparison.csv', pairs)
    write_csv(output / 'paired_seed_summary.csv', summary)
    lookup = {(row['model_family'], row['condition'], row['metric']): row for row in summary}
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
    positions = [(rep, condition) for condition in CONDITIONS for rep in MAIN_REPRESENTATIONS]
    names = [f'{presentation_label("", rep)} / {CONDITION_LABEL[condition]}' for rep, condition in positions]
    for axis, metric, title in zip(axes, ('macro_ovr_auc', 'accuracy'), ('Macro one-vs-rest AUC', 'Accuracy')):
        for i, (rep, condition) in enumerate(positions):
            row = lookup[FAMILIES[rep][0], NATIVE_CONDITION[condition], metric]
            for prefix, offset, marker, label in (
                    ('test_population_delta', -.14, 's', 'Restrict test events; same fit'),
                    ('training_regime_delta', .14, 'o', 'Refit; same selected test events')):
                axis.errorbar(row[f'{prefix}_mean'], i + offset, xerr=row[f'{prefix}_sd'],
                    fmt=marker, color=COLOURS[rep], fillstyle='none' if marker == 's' else 'full',
                    capsize=2, label=label if i == 0 else None)
        axis.axvline(0, color='0.5', lw=1)
        axis.set(yticks=np.arange(len(names)), yticklabels=names, xlabel=f'Change in {title}')
        axis.grid(axis='x', alpha=.25)
        axis.legend(frameon=False, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Inclusive versus four-lepton benchmark: paired-run means ± 1 sample SD')
    save_figure(fig, output, 'inclusive_vs_selected_deltas', formats)
    (output / 'comparison_provenance.json').write_text(json.dumps(dict(
        task='saved-prediction comparison only', matched_selected_event_ids_and_labels=True,
        test_population_delta='inclusive-trained selected-test minus the same inclusive-trained full-test scores',
        training_regime_delta='selected-trained minus inclusive-trained, both on the exact selected test events',
        interpretation='Descriptive comparisons; the selection changes sample size, composition and update budget. No causal claim.',
        uncertainty='Sample SD over three paired seeds; not a confidence interval.',
        no_training_or_inference=True, pairing=audits), indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--study-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--inclusive-config', type=Path)
    parser.add_argument('--inclusive-data-root', type=Path)
    parser.add_argument('--xscale', choices=('linear', 'log', 'both'), default='both')
    parser.add_argument('--grid-points', type=int, default=2001)
    parser.add_argument('--fpr-min', type=float, default=1e-5)
    parser.add_argument('--formats', nargs='+', choices=('pdf', 'png', 'svg'), default=['pdf', 'png'])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if args.inclusive_data_root is not None and args.inclusive_config is None:
        parser.error('--inclusive-data-root requires --inclusive-config')
    config = read_config(args.config)
    study = (args.study_dir or ROOT / config['study_dir']).resolve()
    output = (args.output_dir or study / 'plots').resolve()
    if not args.check_only and output.exists():
        raise FileExistsError(f'Choose a new plot directory: {output}')
    grid = fpr_grid(args.grid_points, args.fpr_min)
    selected, membership = load_membership(study / 'selection', config['expected_test_events'])
    groups = make_groups(study)
    reports, curves, provenance = load_selected(groups, selected, membership)
    print(f'PASS: all 45 new fits use exactly the same {len(selected["labels"]):,} selected test events and labels')
    comparison = (compare_inclusive(groups, selected, reports, provenance, args.inclusive_config,
                    args.inclusive_data_root) if args.inclusive_config else None)
    if args.check_only:
        print('Check only: no plots, files, training or inference.')
        return
    write_outputs(groups, reports, curves, provenance, output, grid=grid, xscale=args.xscale,
                  fpr_min=args.fpr_min, formats=args.formats)
    if comparison is not None:
        write_comparison(output / 'inclusive_vs_selected', *comparison, args.formats)
    (output / 'selected_sample_provenance.json').write_text(json.dumps(dict(
        membership=membership, plot_config_sha256=file_hash(args.config), exact_membership_verified=True,
        selected_fits=45, inclusive_comparison_written=comparison is not None,
        no_training_or_inference=True, inclusive_fallback=False), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
