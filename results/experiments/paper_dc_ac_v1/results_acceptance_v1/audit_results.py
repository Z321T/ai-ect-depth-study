"""Read-only E2 result audit; no model inference, fitting or test classification."""
import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'results/experiments/paper_dc_ac_v1'
SEEN = {}
ERRORS = []


def sha(path):
    path = Path(path)
    if path not in SEEN:
        with path.open('rb') as handle:
            h = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(block)
        SEEN[path] = h.hexdigest()
    return SEEN[path]


def hashes(mapping, base=ROOT):
    for name, expected in mapping.items():
        assert sha(base / name) == expected, name


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected):
    err = abs(float(actual) - float(expected))
    ERRORS.append(err)
    assert err <= 1e-12, (actual, expected)


def metrics(rows):
    cm = [[0] * 20 for _ in range(20)]
    for row in rows:
        cm[int(row['class_index'])][int(row['predicted_class_index'])] += 1
    per_class = {}
    for c in range(20):
        tp = cm[c][c]
        support = sum(cm[c])
        predicted = sum(row[c] for row in cm)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(c)] = {'precision': precision, 'recall': recall,
                             'f1-score': f1, 'support': float(support)}
    return {'accuracy': sum(cm[c][c] for c in range(20)) / len(rows),
            'macro_f1': statistics.mean(v['f1-score'] for v in per_class.values()),
            'confusion_matrix': cm, 'per_class': per_class}


def compare(actual, expected):
    assert actual['confusion_matrix'] == expected['confusion_matrix']
    for key in ['accuracy', 'macro_f1']:
        close(actual[key], expected[key])
    for c, values in actual['per_class'].items():
        for key, value in values.items():
            close(value, expected['per_class'][c][key])


def csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    status = read(BASE / 'execution_status.json')
    assert status['status'] == 'complete' and len(status['steps']) == 12
    assert all(s['status'] == 'complete' and s['exit_code'] == 0 for s in status['steps'])
    assert not list(BASE.rglob('*failure*.json'))
    registry = read(BASE / 'registry.json')
    assert sha(BASE / 'registry.json') == '29cad377ee788a99ec7f57791eadbfd0380fbd94200363f6edfe950b04fd842f'
    assert len(registry['code_sha256']) == 22
    hashes(registry['code_sha256'])
    hashes(registry['acceptance_files_sha256'])
    for path_key, sha_key in [('design_path', 'design_sha256'),
                              ('acceptance_path', 'acceptance_sha256'),
                              ('statistics_path', 'statistics_sha256'),
                              ('base_config_path', 'base_config_sha256'),
                              ('baseline_registry_path', 'baseline_registry_sha256')]:
        assert sha(ROOT / registry[path_key]) == registry[sha_key]
    for run in registry['runs']:
        assert sha(ROOT / run['config_path']) == run['config_file_sha256']
    summary = read(BASE / 'analysis_results_v1/summary.json')
    assert summary['status'] == 'complete' and not summary['heldout_test_evaluated']
    hashes(summary['inputs_sha256'])
    hashes(summary['outputs_sha256'], BASE / 'analysis_results_v1')
    scripts = [summary['script']]
    proof_records = 0
    for path in sorted((BASE / 'verification').glob('*.json')):
        proof = read(path)
        assert proof['status'] == 'verified' and proof['device'] == 'cuda'
        assert proof['same_device'] and not proof['heldout_test_evaluated']
        assert proof['prediction_counts'] == {'train': 24000, 'validation': 3200}
        assert proof['maximum_metric_error'] == 0
        assert all(v for k, v in proof['checks'].items() if k != 'test_classification_evaluated')
        assert proof['checks']['test_classification_evaluated'] is False
        hashes(proof['checked_files_sha256'])
        assert sha(ROOT / proof['run_dir'] / 'report.json') == proof['report_sha256']
        if 'calibration_buffer_max_abs_error' in proof:
            assert proof['calibration_buffer_max_abs_error'] == 0
            assert proof['same_device_calibration_exact']
        scripts.append(proof['script'])
        proof_records += sum(proof['prediction_counts'].values())
    assert len(scripts) == 7 and proof_records == 163200
    for script in scripts:
        assert hashlib.sha256(script['text'].encode()).hexdigest() == script['sha256']
        assert sha(ROOT / script['path']) == script['sha256']
    manifests = {split: [json.loads(line) for line in (ROOT / f'manifests/grouped_v1/{split}.jsonl').read_text().splitlines()]
                 for split in ['train', 'validation']}
    computed = {}
    group_metrics = {}
    compressed_files = 0
    rows_audited = 0
    reports = {}
    for cell in summary['cells']:
        key = (cell['seed'], cell['input'], cell['bn'])
        run_dir = ROOT / cell['run_dir']
        report = read(run_dir / 'report.json')
        reports[key] = report
        hashes(report['files_sha256'], run_dir)
        assert report['selected_epoch'] == cell['selected_epoch']
        for split in ['train', 'validation']:
            rows = csv_rows(run_dir / f'{split}_predictions.csv')
            ids = read(run_dir / f'{split}_rows.json')
            assert len(rows) == len(ids) == len(manifests[split])
            for row, index in zip(rows, ids):
                assert int(row['row_index']) == index
                assert all(str(row[k]) == str(manifests[split][index][k])
                           for k in ['wave_sha256', 'group_id', 'class_index'])
            result = metrics(rows)
            compare(result, report['metrics'][split])
            compare(result, cell['metrics'][split])
            computed[key + (split,)] = result
            grouped = defaultdict(list)
            for row in rows:
                grouped[row['group_id']].append(row)
            for group, group_rows in grouped.items():
                result_g = metrics(group_rows)
                compare(result_g, report['group_metrics'][split][group])
                group_metrics[key + (split, group)] = result_g
            rows_audited += len(rows)
        if cell['input'] == 'dc_ac':
            compression = read(run_dir / 'compression.json')
            assert compression['restore_refuses_overwrite']
            for name, info in compression['files'].items():
                zipped = run_dir / info['compressed_path']
                assert sha(zipped) == info['compressed_sha256']
                assert sha(run_dir / name) == info['original_sha256'] == report['files_sha256'][name]
                with gzip.open(zipped, 'rb') as f:
                    assert hashlib.sha256(f.read()).hexdigest() == info['original_sha256']
                compressed_files += 1
    assert rows_audited == 326400 and compressed_files == 12
    for aggregate in summary['aggregate']:
        values = [computed[(s, aggregate['input'], aggregate['bn'], 'validation')] for s in range(3)]
        acc = [v['accuracy'] for v in values]
        close(statistics.mean(acc), aggregate['accuracy_mean'])
        close(statistics.stdev(acc), aggregate['accuracy_sample_sd'])
        close(min(acc), aggregate['accuracy_worst'])
        close(statistics.mean(v['macro_f1'] for v in values), aggregate['macro_f1_mean'])
        close(statistics.mean(acc[s] - computed[(s, 'original', 'original_bn', 'validation')]['accuracy']
                              for s in range(3)), aggregate['paired_delta_mean'])
    table_counts = {}
    for filename in ['metrics.csv', 'groups.csv', 'per_class.csv']:
        rows = csv_rows(BASE / 'analysis_results_v1' / filename)
        table_counts[filename] = len(rows)
        for row in rows:
            key = (int(row['seed']), row['input'], row['bn'], row['split'])
            result = computed[key]
            if filename == 'groups.csv':
                result = group_metrics[key + (row['group_id'],)]
            if filename == 'per_class.csv':
                result = result['per_class'][row['class_index']]
                names = ['precision', 'recall', 'f1-score', 'support']
            else:
                names = ['accuracy', 'macro_f1']
            for name in names:
                close(row[name], result[name])
            if filename == 'metrics.csv':
                close(row['delta_accuracy_vs_original'], result['accuracy'] - computed[(key[0], 'original', 'original_bn', key[3])]['accuracy'])
    assert table_counts == {'metrics.csv': 24, 'groups.csv': 108, 'per_class.csv': 480}
    for history in summary['histories']:
        report = reports[(history['seed'], history['input'], history['bn'])]
        epochs = report['epochs']
        assert [e['epoch'] for e in epochs] == list(range(1, 1001))
        val = [e for e in epochs if e.get('validation') is not None]
        assert len(val) == 100
        for window, segment in [('first100', epochs[:100]), ('last100', epochs[-100:])]:
            close(statistics.mean(e['train_loss'] for e in segment), history[window + '_train_loss'])
            close(statistics.mean(e['online_train_accuracy'] for e in segment), history[window + '_online_accuracy'])
        for window, segment in [('first10', val[:10]), ('last10', val[-10:])]:
            close(statistics.mean(e['validation']['accuracy'] for e in segment), history[window + '_validation_accuracy'])
        close(epochs[-1]['train_loss'], history['epoch1000_train_loss'])
        close(epochs[-1]['online_train_accuracy'], history['epoch1000_online_accuracy'])
        close(epochs[-1]['validation']['accuracy'], history['epoch1000_validation_accuracy'])
    histories = {(h['seed'], h['input'], h['bn']): h for h in summary['histories']}
    for row in csv_rows(BASE / 'analysis_results_v1/histories.csv'):
        history = histories[(int(row['seed']), row['input'], row['bn'])]
        for name, value in row.items():
            if name not in ['input', 'bn', 'run_dir']:
                close(value, history[name])
    curves = csv_rows(BASE / 'analysis_results_v1/validation_curves.csv')
    assert len(curves) == 600
    for row in curves:
        report = reports[(int(row['seed']), row['input'], row['bn'])]
        epoch = report['epochs'][int(row['epoch']) - 1]
        for name in ['accuracy', 'macro_f1']:
            close(row[name], epoch['validation'][name])
    audit = {'schema_version': 1, 'status': 'verified', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'purpose': 'Independent hash, CSV metrics, group/class, aggregate, history and gzip audit; no new inference',
             'heldout_test_evaluated': False, 'frozen_source_count': 22, 'completed_steps': 12,
             'existing_cuda_proofs': 6, 'existing_cuda_target_prediction_count': proof_records,
             'csv_rows_independently_recounted': rows_audited, 'compressed_files_verified': compressed_files,
             'table_row_counts': table_counts, 'maximum_recomputed_metric_error': max(ERRORS),
             'checked_files_sha256': {str(p.relative_to(ROOT)): v for p, v in sorted(SEEN.items())},
             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = Path(__file__).with_name('audit.json')
    with output.open('x') as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'checked_files_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
