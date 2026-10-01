"""Train fixed feature SVMs on train, select on validation; heldout test sealed."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
import warnings
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_info, threadpool_limits
from .features import FEATURE_NAMES, extract_features
from .integrity import file_sha256
from .prepare import PreparedDataset


def stratified_rows(labels, per_class=None, seed=20261001):
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('Nonempty integer labels required')
    if per_class is not None and (type(per_class) is not int or per_class < 1):
        raise ValueError('Per-class sample cap must be positive or None')
    if per_class is None:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(y):
        rows = np.flatnonzero(y == cls)
        selected.extend(rng.choice(rows, size=min(per_class, len(rows)), replace=False).tolist())
    return np.array(sorted(selected), dtype=np.int64)


def _validate_config(config):
    if config.get('schema_version') != 1 or config.get('protocol') != 'svm_features_v1' or config.get('purpose') != 'development_validation_only':
        raise ValueError('Known development SVM protocol required')
    if type(config.get('threads')) is not int or config['threads'] < 1:
        raise ValueError('Positive integer thread count required')
    if type(config.get('seed')) is not int or config['seed'] < 0:
        raise ValueError('Nonnegative integer seed required')
    cap = config.get('train_per_class')
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError('Positive train_per_class or null required')
    if not isinstance(config.get('candidates'), list) or not config['candidates']:
        raise ValueError('At least one candidate required')
    for candidate in config['candidates']:
        if candidate.get('kernel') not in ('linear', 'rbf'):
            raise ValueError('Known SVM kernel required')
        value = candidate.get('C')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError('Positive finite C required')
        keys = {'kernel', 'C'}
        if candidate['kernel'] == 'rbf':
            keys.add('gamma')
            gamma = candidate.get('gamma')
            if gamma != 'scale' and (isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not np.isfinite(gamma) or gamma <= 0):
                raise ValueError('Positive finite gamma or scale required')
        if set(candidate) != keys:
            raise ValueError('Unexpected candidate parameters')


def _metrics(true, predicted, classes):
    detail = classification_report(true, predicted, labels=classes, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(true, predicted)),
            'macro_f1': float(f1_score(true, predicted, labels=classes, average='macro', zero_division=0)),
            'classes': classes.tolist(),
            'per_class': {str(cls): detail[str(cls)] for cls in classes},
            'confusion_matrix': confusion_matrix(true, predicted, labels=classes).tolist()}


def _features_from_store(store, split):
    batches, labels = [], []
    for signals, y in store.iter_batches(split, batch_size=512, normalize=False):
        batches.append(extract_features(signals))
        labels.append(y)
    return np.concatenate(batches), np.concatenate(labels)


def _source_sha256(path):
    """Normalize only CRLF line endings; retain a separate byte hash in reports."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def train_svm(project_root, config, output_dir, cache_dir='data/processed/grouped_v1', progress=None):
    """Publish a new validation-only run; a failure is recorded and re-raised."""
    _validate_config(config)
    config = json.loads(json.dumps(config, allow_nan=False))
    root = Path(project_root).resolve()
    output = Path(output_dir)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if not output.is_relative_to(root):
        raise ValueError('Run directory must be within project root')
    if output.exists():
        raise FileExistsError(f'Run directory exists: {output}; choose a new output')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    code_dir = Path(__file__).resolve().parent
    report = {'schema_version': 1, 'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'development_validation_only', 'heldout_test_evaluated': False, 'config': config,
              'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
              'feature_names': list(FEATURE_NAMES), 'feature_units': 'raw downsampled dataset units',
              'feature_source_sha256_lf': _source_sha256(code_dir / 'features.py'),
              'code_sha256': {f'src/ect/{name}': file_sha256(code_dir / name) for name in (
                  'baseline.py', 'features.py', 'prepare.py', 'dataset.py', 'preprocessing.py', 'integrity.py')},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
              'selection_rule': 'converged candidates: validation macro_f1, accuracy, then earlier candidate',
              'candidates': []}
    # mkdir is the atomic claim: two jobs cannot train/publish into one run.
    output.mkdir(exist_ok=False)
    try:
        with PreparedDataset(root, cache_dir) as store, threadpool_limits(limits=config['threads']):
            report['threadpools'] = threadpool_info()
            report['cache_path'] = str(store.folder.relative_to(root))
            report['cache_metadata_sha256'] = file_sha256(store.folder / 'metadata.json')
            report['manifest_sha256'] = store.metadata['manifest_sha256']
            report['raw_input_sha256'] = {name: info['sha256'] for name, info in store.metadata['inputs'].items()}
            extraction_started = time.perf_counter()
            train_features, train_labels = _features_from_store(store, 'train')
            validation_features, validation_labels = _features_from_store(store, 'validation')
            report['feature_extraction_seconds'] = time.perf_counter() - extraction_started
            manifest_parent = (root / store.metadata['summary_path']).parent
            validation_records = None
            for split, labels in (('train', train_labels), ('validation', validation_labels)):
                records = [json.loads(line) for line in (manifest_parent / f'{split}.jsonl').read_text().splitlines()]
                if not np.array_equal(labels, [r['class_index'] for r in records]):
                    raise ValueError('Cached labels differ from frozen manifest order')
                if split == 'validation':
                    validation_records = records
            rows = stratified_rows(train_labels, config['train_per_class'], config['seed'])
            features, labels = train_features[rows], train_labels[rows]
            classes = np.unique(labels)
            if len(classes) < 2 or not np.array_equal(classes, np.unique(validation_labels)):
                raise ValueError('Train and validation must cover the same >=2 classes')
            report['sample_counts'] = {'train': len(rows), 'validation': len(validation_labels)}
            report['available_train_count'] = len(train_labels)
            report['train_class_counts'] = {str(cls): int(np.sum(labels == cls)) for cls in classes}
            best_key, best_model, best_prediction = None, None, None
            for i, candidate in enumerate(config['candidates']):
                if candidate['kernel'] == 'linear':
                    classifier = LinearSVC(C=candidate['C'], dual=False, tol=1e-4, max_iter=10000,
                                           random_state=config['seed'])
                else:
                    classifier = SVC(C=candidate['C'], gamma=candidate['gamma'], kernel='rbf', tol=1e-3,
                                     max_iter=-1, cache_size=512, random_state=config['seed'])
                pipeline = Pipeline([('scale', StandardScaler()), ('svm', classifier)])
                fit_started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    pipeline.fit(features, labels)
                fit_seconds = time.perf_counter() - fit_started
                inference_started = time.perf_counter()
                prediction = pipeline.predict(validation_features)
                predict_seconds = time.perf_counter() - inference_started
                converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
                converged = converged and getattr(classifier, 'fit_status_', 0) == 0
                metric = _metrics(validation_labels, prediction, classes)
                record = {'index': i, 'parameters': candidate, 'resolved_parameters': classifier.get_params(),
                          'fit_seconds': fit_seconds, 'validation_predict_seconds': predict_seconds,
                          'converged': converged, 'warnings': [str(item.message) for item in caught],
                          'iterations': np.asarray(classifier.n_iter_).tolist(), 'metrics': metric}
                report['candidates'].append(record)
                key = (metric['macro_f1'], metric['accuracy'], -i)
                if converged and (best_key is None or key > best_key):
                    best_key, best_model, best_prediction = key, pipeline, prediction.copy()
                    report['selected_candidate'] = i
                if progress is not None:
                    progress(record)
            if best_model is None:
                raise ValueError('No converged candidate available for selection')
            scale = best_model.named_steps['scale']
            report['training_feature_scaler'] = {'mean': scale.mean_.tolist(), 'variance': scale.var_.tolist(),
                                                 'scale': scale.scale_.tolist(), 'n_samples_seen': int(scale.n_samples_seen_)}
            group_ids = np.array([r['group_id'] for r in validation_records])
            report['validation_group_metrics'] = {str(group): _metrics(validation_labels[group_ids == group],
                best_prediction[group_ids == group], classes) for group in np.unique(group_ids)}
            with tempfile.TemporaryDirectory(prefix='.svm-', dir=output) as temporary:
                stage = Path(temporary) / 'run'; stage.mkdir()
                bundle = {'pipeline': best_model, 'feature_names': list(FEATURE_NAMES), 'protocol': config['protocol']}
                joblib.dump(bundle, stage / 'model.joblib')
                loaded = joblib.load(stage / 'model.joblib')
                if not np.array_equal(loaded['pipeline'].predict(validation_features), best_prediction):
                    raise ValueError('Reloaded model predictions differ')
                report['model_roundtrip_verified'] = True
                (stage / 'train_rows.json').write_text(json.dumps(rows.tolist()) + '\n', newline='\n')
                with (stage / 'validation_predictions.csv').open('w', newline='') as handle:
                    writer = csv.writer(handle, lineterminator='\n')
                    writer.writerow(['row_index', 'wave_sha256', 'group_id', 'class_index', 'predicted_class_index'])
                    for i, (record, predicted) in enumerate(zip(validation_records, best_prediction)):
                        writer.writerow([i, record['wave_sha256'], record['group_id'], record['class_index'], int(predicted)])
                report['files_sha256'] = {name: file_sha256(stage / name) for name in (
                    'model.joblib', 'train_rows.json', 'validation_predictions.csv')}
                report.update(status='complete', elapsed_seconds=time.perf_counter() - started)
                (stage / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', newline='\n')
                for name in ('model.joblib', 'train_rows.json', 'validation_predictions.csv'):
                    (stage / name).rename(output / name)
                # report.json is the commit marker and is published last.
                (stage / 'report.json').rename(output / 'report.json')
        return report
    except BaseException as error:
        # Preserve failed runs without making partial output look successful.
        for name in ('model.joblib', 'train_rows.json', 'validation_predictions.csv', 'report.json'):
            (output / name).unlink(missing_ok=True)
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.perf_counter() - started)
        (output / 'failure.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', newline='\n')
        raise


def load_svm_model(run_dir):
    folder = Path(run_dir)
    report = json.loads((folder / 'report.json').read_text())
    required = {'model.joblib', 'train_rows.json', 'validation_predictions.csv'}
    if report['status'] != 'complete' or (folder / 'failure.json').exists() or set(report['files_sha256']) != required:
        raise ValueError('Complete model checksum table required')
    for name, digest in report['files_sha256'].items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f'Model run checksum mismatch: {name}')
    if _source_sha256(Path(__file__).with_name('features.py')) != report['feature_source_sha256_lf']:
        raise ValueError('Feature implementation differs from trained model')
    bundle = joblib.load(folder / 'model.joblib')
    if bundle['feature_names'] != list(FEATURE_NAMES) or bundle['protocol'] != 'svm_features_v1':
        raise ValueError('Incompatible feature protocol')
    return bundle
