r"""Plot completed evaluation artifacts without inference or loading prediction rows.

Example (paths are relative to the current working directory)::

    .venv/bin/python scripts/plot_evaluation.py \
        --input results/evaluation/validation_v1 \
        --output results/evaluation/validation_figures_v1 --clean-confusion

Output must be new. A complete output report is published last; failure.json
invalidates partial artifacts. CSV is only hashed in bounded streaming blocks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


FAMILIES = ('svm', 'cnn_clean', 'resnet_clean', 'resnet_aug')
CONDITIONS = ('clean', '30', '20', '10')
METRICS = {'accuracy': 'Accuracy', 'macro_f1': 'Macro-F1'}
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
MARKERS = ('o', 's', '^', 'D')
LINE_NOTE = (
    'Networks: mean +/- sample SD across 3 training seeds (ddof=1). SVM: one model, no training error bars.\n'
    'At each SNR, average 5 noise repeats within each model first; noise-repeat SD is within-model variation.\n'
    'Training SD and noise-repeat SD describe this fixed split; neither is a population confidence interval.'
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json_snapshot(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8', newline='\n')
    temporary.replace(path)


def has_failure(folder, report):
    marker = folder / 'failure.json'
    return (marker.exists() or marker.is_symlink()
            or any(report.get(key) for key in ('failure', 'error', 'error_type')))


def verify_inputs(folder):
    report, report_sha = read_json_snapshot(folder / 'report.json')
    if report.get('status') != 'complete' or has_failure(folder, report):
        raise ValueError('Input requires a complete report without failure markers')
    split = report.get('split')
    if split not in ('validation', 'test'):
        raise ValueError('Input split must be validation or test')
    seeds = list(range(10, 15)) if split == 'validation' else list(range(100, 105))
    if (report.get('noise_seeds') != seeds or report.get('namespace') != split
            or report.get('heldout_test_evaluated') is not (split == 'test')
            or report.get('clean_repetitions') != 1 or report.get('model_count') != 10):
        raise ValueError('Report split/noise/model metadata differs from formal_v1')
    count = report.get('sample_count')
    if (type(count) is not int or count <= 0
            or report.get('prediction_count') != count * 160):
        raise ValueError('Invalid evaluation sample/prediction counts')
    expected = report.get('files_sha256', {})
    if set(expected) != {'summary.json', 'predictions.csv'}:
        raise ValueError('Report must bind summary.json and predictions.csv SHA256')
    summary, summary_sha = read_json_snapshot(folder / 'summary.json')
    csv_sha = file_sha256(folder / 'predictions.csv')
    if summary_sha != expected['summary.json'] or csv_sha != expected['predictions.csv']:
        raise ValueError('Evaluation summary/CSV SHA256 mismatch')

    # This pure module validates all 160 scalar records and the 10 model identities.
    # Do not import the evaluator: that would import training/inference dependencies.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.ect.evaluation_summary import summarize_metrics
    scalar_entries = [entry | {'metrics': {key: entry['metrics'][key] for key in METRICS}}
                      for entry in report['metrics']]
    if summary != summarize_metrics(scalar_entries, seeds):
        raise ValueError('Summary does not match report metrics and formal_v1 statistics')
    hashes = {'report.json': report_sha, 'summary.json': summary_sha,
              'predictions.csv': csv_sha}
    return report, summary, hashes


def verify_unchanged(folder, hashes):
    if has_failure(folder, {}):
        raise ValueError('Input acquired a failure marker during plotting')
    for name, digest in hashes.items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f'Input changed during plotting: {name}')


def clean_matrices(report, summary):
    """Average row percentages equally across the three networks, never pool seeds."""
    import numpy as np
    clean = {entry['run_id']: entry for entry in report['metrics']
             if entry['condition'] == 'clean'}
    matrices, shared_support = {}, None
    for family in FAMILIES:
        percentages = []
        for run_id in summary['families'][family]['run_ids']:
            metrics = clean[run_id]['metrics']
            matrix = np.asarray(metrics['confusion_matrix'])
            if (matrix.shape != (20, 20) or matrix.dtype.kind not in 'iu'
                    or (matrix < 0).any()):
                raise ValueError(f'Clean confusion for {run_id} must be 20 x 20 integer counts')
            matrix = matrix.astype(float)
            support = matrix.sum(axis=1)
            if matrix.sum() != report['sample_count'] or (support <= 0).any():
                raise ValueError(f'Clean confusion for {run_id} has invalid class support')
            if shared_support is None:
                shared_support = support
            elif not np.array_equal(support, shared_support):
                raise ValueError('Clean class supports differ across fixed-split models')
            accuracy = np.trace(matrix) / matrix.sum()
            macro_f1 = (2 * matrix.diagonal() / (support + matrix.sum(axis=0))).mean()
            if any(not math.isclose(value, metrics[key], rel_tol=1e-12, abs_tol=1e-12)
                   for key, value in (('accuracy', accuracy), ('macro_f1', macro_f1))):
                raise ValueError(f'Clean confusion disagrees with metrics for {run_id}')
            percentages.append(100 * matrix / support[:, None])
        matrices[family] = np.mean(percentages, axis=0)
    return matrices


def save_figure(fig, output, stem, plt):
    names = []
    try:
        for extension in ('png', 'svg'):
            name = f'{stem}.{extension}'
            path = output / name
            fig.savefig(path, dpi=200, facecolor='white')
            if extension == 'svg':
                # Matplotlib emits trailing spaces in SVG path data. Hash after cleanup.
                path.write_text('\n'.join(line.rstrip() for line in
                                          path.read_text(encoding='utf-8').splitlines()) + '\n',
                                encoding='utf-8', newline='\n')
            names.append(name)
    finally:
        plt.close(fig)
    return names


def draw_figures(report, summary, output, confusion):
    # Respect an explicit MPLCONFIGDIR; otherwise use a disposable writable directory.
    with tempfile.TemporaryDirectory(prefix='ect_plot_mpl_') as config:
        previous = os.environ.get('MPLCONFIGDIR')
        if previous is None:
            os.environ['MPLCONFIGDIR'] = config
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            import numpy as np
            with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                                        'svg.fonttype': 'none', 'svg.hashsalt': 'ect-evaluation',
                                        'axes.spines.top': False, 'axes.spines.right': False}):
                names, plotted = [], {}
                split = report['split']
                for key, label in METRICS.items():
                    fig, axis = plt.subplots(figsize=(10, 6.8))
                    plotted[key] = {}
                    for family, color, marker in zip(FAMILIES, COLORS, MARKERS):
                        records = [summary['families'][family]['conditions'][condition][key]
                                   for condition in CONDITIONS]
                        means = [record['mean'] for record in records]
                        sds = [record['training_sample_sd'] for record in records]
                        plotted[key][family] = {'mean': means, 'training_sample_sd': sds,
                            'within_training_seed_noise_sd': [record['within_training_seed_noise_sd']
                                                              for record in records]}
                        kwargs = dict(color=color, marker=marker, linewidth=1.8,
                                      markersize=6, label=family)
                        if family == 'svm':
                            axis.plot(range(4), means, **kwargs)
                        else:
                            axis.errorbar(range(4), means, yerr=sds, capsize=4, **kwargs)
                    axis.set(xticks=range(4), xticklabels=('Clean', '30 dB', '20 dB', '10 dB'),
                             xlabel='Evaluation condition (added noise SNR, AC reference)',
                             ylabel=label,
                             title=f'{label} | {split} split | formal_v1')
                    # Keep every SD whisker visible, including beyond the rate boundaries.
                    lower = min(r['mean'] - (r['training_sample_sd'] or 0)
                                for f in FAMILIES for c in CONDITIONS
                                for r in [summary['families'][f]['conditions'][c][key]])
                    upper = max(r['mean'] + (r['training_sample_sd'] or 0)
                                for f in FAMILIES for c in CONDITIONS
                                for r in [summary['families'][f]['conditions'][c][key]])
                    axis.set_ylim(min(0, lower - .025), max(1, upper + .025))
                    axis.grid(axis='y', alpha=.25)
                    axis.legend(loc='best', frameon=False)
                    fig.subplots_adjust(left=.10, right=.97, top=.90, bottom=.27)
                    fig.text(.10, .06, LINE_NOTE, fontsize=8.5, linespacing=1.6)
                    names.extend(save_figure(fig, output, key, plt))
                if confusion is not None:
                    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
                    for axis, family in zip(axes.flat, FAMILIES):
                        displayed = axis.imshow(confusion[family], cmap='Blues', vmin=0, vmax=100)
                        for row in range(20):
                            for col in range(20):
                                value = confusion[family][row, col]
                                if value >= .05:
                                    axis.text(col, row, f'{value:.1f}', ha='center', va='center',
                                              fontsize=6, color='white' if value > 55 else '#202020')
                        axis.set(xticks=np.arange(20), yticks=np.arange(20),
                                 xlabel='Predicted class_index', ylabel='True class_index',
                                 title=family + (' (single model)' if family == 'svm'
                                                 else ' (mean of 3 training seeds)'))
                        axis.tick_params(labelsize=8)
                    fig.suptitle(f'Clean confusion matrices | {split} split | formal_v1', y=.97)
                    fig.subplots_adjust(left=.07, right=.86, bottom=.13, top=.91,
                                        wspace=.27, hspace=.30)
                    color_axis = fig.add_axes([.89, .22, .018, .60])
                    fig.colorbar(displayed, cax=color_axis, label='Row-normalized (%)')
                    fig.text(.07, .035,
                             'Clean only. Networks: normalize each seed matrix by true-class row, '
                             'then average the 3 percentage matrices.\n'
                             'SVM: one row-normalized matrix. class_index 0..19; '
                             'class semantics are unverified. No confidence intervals.',
                             fontsize=10, linespacing=1.7)
                    names.extend(save_figure(fig, output, 'clean_confusion', plt))
                    plotted['clean_confusion_row_percent'] = {
                        family: matrix.tolist() for family, matrix in confusion.items()}
            return names, plotted, matplotlib.__version__
        finally:
            if previous is None:
                os.environ.pop('MPLCONFIGDIR', None)


def plot_evaluation(folder, output, include_confusion=False):
    folder, output = Path(folder).resolve(), Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation owns this directory; no failure marker is written to an existing one.
    output.mkdir(exist_ok=False)
    run_report = {'schema_version': 1, 'status': 'running',
                  'created_utc': datetime.now(timezone.utc).isoformat(),
                  'input_directory': str(folder), 'output_directory': str(output)}
    try:
        script_sha = file_sha256(__file__)
        report, summary, input_hashes = verify_inputs(folder)
        run_report['split'] = report['split']
        confusion = clean_matrices(report, summary) if include_confusion else None
        names, plotted, version = draw_figures(report, summary, output, confusion)
        image_hashes = {name: file_sha256(output / name) for name in names}
        provenance = {
            'schema_version': 1, 'created_utc': run_report['created_utc'],
            'split': report['split'], 'input_directory': str(folder),
            'script_path': str(Path(__file__).resolve()), 'script_sha256': script_sha,
            'report_sha256': input_hashes['report.json'],
            'summary_sha256': input_hashes['summary.json'],
            'csv_sha256': input_hashes['predictions.csv'],
            'input_files_sha256': input_hashes, 'image_sha256': image_hashes,
            'matplotlib': version, 'backend': 'Agg',
            'statistical_scope': summary['statistical_scope'],
            'statistical_interpretation': {
                'condition_order': list(CONDITIONS), 'metric_units': 'fraction in [0, 1]',
                'networks': 'Equal mean of 3 model means; error bars are training sample SD (ddof=1).',
                'noise': '5 repeats are averaged within each fixed model and SNR first. '
                         'Noise sample SD (ddof=1) is within-model variation, recorded separately; '
                         'it is not plotted as training variation or a confidence interval.',
                'svm': 'One deterministic model; no fabricated training seeds or training error bars.',
                'confidence_intervals': 'None. Fixed-split variation is not population uncertainty.',
                'clean_confusion': 'Clean only: row percentages per model, then equal mean over '
                                   '3 network training seeds; SVM single matrix. class_index 0..19.',
                'verification_scope': 'Report/summary consistency and file SHA only; '
                                      'CSV rows are not parsed and predictions are not independently verified.',
            },
            'plotted_values': plotted,
        }
        write_json(output / 'provenance.json', provenance)
        verify_unchanged(folder, input_hashes)
        if file_sha256(__file__) != script_sha:
            raise ValueError('Plot script changed during generation')
        artifact_hashes = image_hashes | {'provenance.json': file_sha256(output / 'provenance.json')}
        for name, digest in artifact_hashes.items():
            if file_sha256(output / name) != digest:
                raise ValueError(f'Output changed during generation: {name}')
        run_report.update(status='complete', script_sha256=script_sha,
                          input_files_sha256=input_hashes, files_sha256=artifact_hashes)
        # The success report is the last artifact, binding provenance and every figure.
        write_json(output / 'report.json', run_report)
        return run_report
    except BaseException as error:
        run_report.update(status='failed', error_type=type(error).__name__, error=str(error))
        try:
            write_json(output / 'failure.json', run_report)
        finally:
            (output / 'report.json').unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path, help='Complete evaluation directory')
    parser.add_argument('--output', required=True, type=Path, help='New figure directory (never overwrite)')
    parser.add_argument('--clean-confusion', action='store_true',
                        help='Also export Clean class_index 0..19 confusion matrices')
    args = parser.parse_args()
    try:
        report = plot_evaluation(args.input, args.output, args.clean_confusion)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'plot_evaluation: {type(error).__name__}: {error}\n')
    print(json.dumps({'status': report['status'], 'split': report['split'],
                      'output': report['output_directory'],
                      'figure_count': len(report['files_sha256']) - 1}, indent=2))


if __name__ == '__main__':
    main()
