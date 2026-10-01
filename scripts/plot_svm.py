"""Plot saved validation-only SVM results without evaluating any signals."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ect_matplotlib'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.ect.integrity import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run', type=Path, default=Path('results/baselines/svm_full_v1'))
    args = parser.parse_args()
    root = args.project_root.resolve()
    folder = args.run if args.run.is_absolute() else root / args.run
    report_path = folder / 'report.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'complete' or report['heldout_test_evaluated']:
        raise ValueError('Complete validation-only run required')
    for name in ('validation_predictions.csv', 'train_rows.json'):
        if file_sha256(folder / name) != report['files_sha256'][name]:
            raise ValueError('Result artifact checksum mismatch')
    output = folder / 'figures'; output.mkdir(exist_ok=True)
    candidates = report['candidates']
    labels = []
    for candidate in candidates:
        parameters = candidate['parameters']
        labels.append(f"{parameters['kernel']} C={parameters['C']:g}" +
                      (f" γ={parameters['gamma']}" if parameters['kernel'] == 'rbf' else ''))
    colors = ['#2563eb' if i == report['selected_candidate'] else '#94a3b8' for i in range(len(candidates))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].barh(labels, [c['metrics']['macro_f1'] for c in candidates], color=colors)
    axes[0].set(xlim=(0, 1), xlabel='Validation Macro-F1')
    axes[1].barh(labels, [c['fit_seconds'] for c in candidates], color=colors)
    axes[1].set(xlabel='CPU fit seconds (actual environment)')
    axes[0].invert_yaxis()
    fig.suptitle(f"SVM development search: {report['sample_counts']['train']} train / {report['sample_counts']['validation']} validation; test sealed")
    fig.tight_layout(); fig.savefig(output / 'candidate_search.png', dpi=150); plt.close(fig)
    metric = candidates[report['selected_candidate']]['metrics']
    matrix = np.asarray(metric['confusion_matrix'])
    denominator = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, denominator, out=np.zeros_like(matrix, dtype=float), where=denominator > 0)
    fig, axis = plt.subplots(figsize=(9, 8))
    displayed = axis.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    axis.set(xticks=np.arange(len(matrix)), yticks=np.arange(len(matrix)),
             xticklabels=metric['classes'], yticklabels=metric['classes'],
             xlabel='Predicted class index', ylabel='True class index',
             title=f"Selected SVM: validation recall by class\nAccuracy={metric['accuracy']:.3f}, Macro-F1={metric['macro_f1']:.3f}")
    fig.colorbar(displayed, ax=axis, label='Row proportion')
    fig.tight_layout(); fig.savefig(output / 'validation_confusion.png', dpi=150); plt.close(fig)
    (output / 'provenance.json').write_text(json.dumps({'report_sha256': file_sha256(report_path),
        'script_sha256': file_sha256(__file__), 'matplotlib': matplotlib.__version__,
        'image_sha256': {name: file_sha256(output / name) for name in ('candidate_search.png', 'validation_confusion.png')}}, indent=2) + '\n', newline='\n')
    print(str(output))


if __name__ == '__main__':
    main()
