"""Independently verify one paper run on train/validation; never classify test.

No producer loader, crop, prediction, standardization or metric routine is used.
All run files are read-only. A proof is published atomically at a NEW filename
only after every requested prediction, metric and identity check passes.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import types

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import torch

from src.ect.devices import resolve_device, seed_everything
from src.ect.paper_models import PaperResNeXt1D
from src.ect import devices as _devices, paper_models as _paper_models

# Bind the files of the modules actually imported for independent inference.
EXECUTED_MODULE_PATHS = {
    'src/ect/devices.py': Path(_devices.__file__).resolve(),
    'src/ect/paper_models.py': Path(_paper_models.__file__).resolve(),
}
IMPORTED_MODULE_BYTES = {name: path.read_bytes() for name, path in EXECUTED_MODULE_PATHS.items()}


SPLITS = ('train', 'validation')
ARTIFACTS = {'model.pt', 'train_rows.json', 'validation_rows.json',
             'train_predictions.csv', 'validation_predictions.csv'}
SOURCES = {f'src/ect/{name}.py' for name in (
    'paper_training', 'paper_registration', 'paper_models', 'paper_crops', 'devices', 'baseline',
    'prepare', 'dataset', 'preprocessing', 'integrity')}
SOURCES.add('scripts/train_paper_baseline.py')
REGISTRY_SOURCES = SOURCES | {'scripts/register_paper_baseline.py', 'scripts/verify_paper_run.py'}
BASE_PATH = 'config/paper_baseline_v1.json'
BASE_SHA256 = 'c14d18a8a451e249d2b3e31c21af9e5c3a49975576b9f3770f553c31a0dc3dc0'
DESIGN_PATH = 'results/paper_components/baseline_design_v1.json'
DESIGN_SHA256 = '341a764a31f1d9b1241244efd695b40bccfec86cd5cb6a3c24bd5548512d70c9'
CSV_FIELDS = ['row_index', 'wave_sha256', 'group_id', 'class_index', 'predicted_class_index']
METRIC_ATOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=invalid_constant, object_pairs_hook=unique_keys)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_loaded_inference_sources():
    """Compile, never execute, source and compare actual functions/methods."""
    for name, module in (('src/ect/devices.py', _devices), ('src/ect/paper_models.py', _paper_models)):
        path = EXECUTED_MODULE_PATHS[name]
        source = IMPORTED_MODULE_BYTES[name]
        require(path.read_bytes() == source, f'Imported source changed: {name}')
        compiled = compile(source, str(path), 'exec', dont_inherit=True)
        expected = {}
        def collect(code):
            expected[code.co_qualname] = code
            for child in code.co_consts:
                if isinstance(child, types.CodeType):
                    collect(child)
        collect(compiled)
        defaults = {}
        def definitions(node, prefix=''):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f'{prefix}.{child.name}' if prefix else child.name
                    try:
                        positional = tuple(ast.literal_eval(v) for v in child.args.defaults) or None
                        keyword = {arg.arg: ast.literal_eval(v) for arg, v in
                                   zip(child.args.kwonlyargs, child.args.kw_defaults) if v is not None} or None
                    except (ValueError, TypeError) as error:
                        raise ValueError(f'Literal source defaults required: {name}/{qualname}') from error
                    defaults[qualname] = positional, keyword
                    definitions(child, qualname + '.<locals>')
                elif isinstance(child, ast.ClassDef):
                    definitions(child, f'{prefix}.{child.name}' if prefix else child.name)
                else:
                    definitions(child, prefix)
        definitions(ast.parse(source))
        def check(function, qualname):
            require(function.__code__ == expected.get(qualname),
                    f'Loaded function/method differs from source: {name}/{qualname}')
            require(repr((function.__defaults__, function.__kwdefaults__)) == repr(defaults.get(qualname)),
                    f'Loaded defaults differ from source: {name}/{qualname}')
        for attr, value in vars(module).items():
            if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
                check(value, attr)
            if isinstance(value, type) and value.__module__ == module.__name__:
                for method_name, method in vars(value).items():
                    if isinstance(method, (staticmethod, classmethod)):
                        method = method.__func__
                    if isinstance(method, types.FunctionType):
                        qualname = f'{value.__qualname__}.{method_name}'
                        check(method, qualname)
    require(seed_everything is _devices.seed_everything and resolve_device is _devices.resolve_device and
            PaperResNeXt1D is _paper_models.PaperResNeXt1D, 'Executed inference aliases differ from modules')


def inside(root, path):
    target = Path(path)
    target = (target if target.is_absolute() else root / target).resolve()
    require(target.is_relative_to(root), f'Path outside project root: {target}')
    return target


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def evidence_not_failed(root, path):
    """A nested evidence file cannot hide its enclosing failed/running run."""
    parent = Path(path).parent
    while parent.is_relative_to(root):
        require(not (parent / 'failure.json').exists(), f'Failed registration evidence: {parent}')
        require(not (parent / 'running.marker').exists(), f'Running registration evidence: {parent}')
        if parent == root:
            break
        parent = parent.parent


def verify_execution_registration(root, run, report, cache, metadata, checked):
    """Independent JSON/hash identity check; no producer validators or inference.

    Acceptance semantics belong to the registrar. Here every acceptance file,
    source and immutable registry input is bound again to the completed run.
    """
    require('execution_registration' in report, 'Explicit execution_registration required')
    recorded = report['execution_registration']
    config = report['config']
    if config['purpose'] == 'paper_runner_smoke_validation_only':
        require(recorded is None, 'Smoke execution_registration must be null')
        return None
    fields = {'registry_path', 'registry_sha256', 'protocol', 'registered_utc', 'run_id',
              'seed', 'config_path', 'config_file_sha256', 'resolved_config_sha256',
              'output_dir', 'acceptance_sha256'}
    require(isinstance(recorded, dict) and set(recorded) == fields,
            'Complete full-purpose execution_registration required')

    def relative_path(label):
        require(isinstance(label, str), 'Root-relative registration path required')
        target = inside(root, label)
        require(target != root and not Path(label).is_absolute() and
                target.relative_to(root).as_posix() == label,
                f'Canonical root-relative registration path required: {label}')
        evidence_not_failed(root, target)
        return target

    def object_file(label, digest):
        path = relative_path(label)
        checked(path, digest)
        value = read_json(path)
        require(isinstance(value, dict), f'Registration JSON object required: {label}')
        return value

    registry_file = relative_path(recorded['registry_path'])
    checked(registry_file, recorded['registry_sha256'])
    registry = read_json(registry_file)
    directory = registry_file.parent.relative_to(root).as_posix()
    require(isinstance(registry, dict) and registry_file.name == 'registry.json' and
            type(registry.get('schema_version')) is int and registry['schema_version'] == 1 and
            registry.get('protocol') == config['protocol'] and
            registry.get('status') == 'frozen_execution_registration' and
            registry.get('test_classification_enabled') is False and
            registry.get('registration_dir') == directory and
            isinstance(registry.get('registered_utc'), str), 'Frozen execution registry required')
    require(registry.get('base_config_path') == BASE_PATH and
            registry.get('base_config_sha256') == BASE_SHA256 and
            registry.get('design_path') == DESIGN_PATH and
            registry.get('design_sha256') == DESIGN_SHA256, 'Immutable base/design registry anchors differ')
    base = object_file(BASE_PATH, BASE_SHA256)
    design = object_file(DESIGN_PATH, DESIGN_SHA256)
    require(type(design.get('schema_version')) is int and design['schema_version'] == 1 and
            design.get('protocol') == config['protocol'] and
            design.get('status') == 'design_registered_before_full_training' and
            design.get('config_path') == BASE_PATH and design.get('config_sha256') == BASE_SHA256 and
            all(design.get(key) is False for key in ('complete_training_started',
                'training_runner_implemented', 'execution_registry_bound')),
            'Original historical design identity differs')
    require(canonical(config) == canonical(base | {'seed': config['seed']}),
            'Full run must use immutable base plus registered seed')
    require(relative_path(base['cache_dir']) == cache and
            relative_path(report['cache_path']) == cache, 'Registered cache path differs')

    def checksum_table(value):
        require(isinstance(value, dict) and value, 'Complete nonempty registry SHA table required')
        require(all(isinstance(label, str) and isinstance(digest, str) and len(digest) == 64 and
                    all(c in '0123456789abcdef' for c in digest) for label, digest in value.items()),
                'Invalid registry SHA table entry')
        return value

    sources = checksum_table(registry.get('code_sha256'))
    require(set(sources) == REGISTRY_SOURCES, 'Complete 13-path registration source table required')
    for label, digest in sources.items():
        checked(EXECUTED_MODULE_PATHS.get(label, SOURCE_ROOT / label), digest)
        if label in report['code_sha256']:
            require(digest == report['code_sha256'][label], f'Registered/run source differs: {label}')
    components = checksum_table(design.get('component_code_sha256'))
    require(set(components) == {'src/ect/paper_models.py', 'src/ect/paper_crops.py'} and
            all(sources[label] == digest for label, digest in components.items()),
            'Historical component source identity differs')

    files = {BASE_PATH: BASE_SHA256, DESIGN_PATH: DESIGN_SHA256}
    def add_file(label, digest):
        path = relative_path(label)
        checked(path, digest)
        require(label not in files or files[label] == digest, f'Conflicting evidence checksum: {label}')
        files[label] = digest

    for label, key in (('results/paper_components/acceptance_v1/report.json',
                        'component_acceptance_report_sha256'),
                       ('results/paper_components/acceptance_v1/verification.json',
                        'component_acceptance_verification_sha256')):
        add_file(label, design[key])
    acceptance = object_file(registry['acceptance_path'], registry['acceptance_sha256'])
    checks = {'cpu_training', 'cuda_training', 'independent_crop_reload', 'csv_metrics',
              'full_regression', 'independent_code_review'}
    require(type(acceptance.get('schema_version')) is int and acceptance['schema_version'] == 1 and
            acceptance.get('status') == 'verified' and
            acceptance.get('protocol') == 'paper_baseline_runner_acceptance_v1' and
            acceptance.get('test_classification_enabled') is False and
            isinstance(acceptance.get('checks'), dict) and set(acceptance['checks']) == checks and
            all(acceptance['checks'][key] is True for key in checks), 'Complete runner acceptance required')
    require(checksum_table(acceptance.get('code_sha256')) == sources,
            'Acceptance/registered execution source table differs')
    for label, digest in checksum_table(acceptance.get('files_sha256')).items():
        add_file(label, digest)
    add_file(registry['acceptance_path'], registry['acceptance_sha256'])

    metadata_sha = checked(cache / 'metadata.json', report['cache_metadata_sha256'])
    identity = dict(cache_dir=cache.relative_to(root).as_posix(), cache_metadata_sha256=metadata_sha,
        cache_files_sha256=metadata['files_sha256'], manifest_sha256=metadata['manifest_sha256'],
        summary_path=metadata['summary_path'], summary_sha256=metadata['summary_sha256'],
        raw_inputs=metadata['inputs'], sample_counts=metadata['sample_counts'],
        normalization_sha256=metadata['files_sha256']['normalization.json'])
    require(canonical(registry.get('data_identity')) == canonical(identity), 'Registered cache identity differs')
    require(design.get('cache_metadata_sha256') == metadata_sha and
            design.get('manifest_sha256') == metadata['manifest_sha256'], 'Historical design data differs')
    counts = {s: metadata['sample_counts'][s] for s in SPLITS}
    require(canonical(report['sample_counts']) == canonical(counts) and
            canonical(report['available_sample_counts']) == canonical(counts),
            'Full-purpose counts must include all registered train/validation samples')
    add_file((cache / 'metadata.json').relative_to(root).as_posix(), metadata_sha)
    add_file(metadata['summary_path'], metadata['summary_sha256'])
    for label, digest in metadata['files_sha256'].items():
        require(Path(label).name == label, 'Cache file name must remain inside cache')
        add_file((cache / label).relative_to(root).as_posix(), digest)
    for label, digest in metadata['manifest_sha256'].items():
        add_file((Path(metadata['summary_path']).parent / label).as_posix(), digest)
    for info in metadata['inputs'].values():
        add_file(info['path'], info['sha256'])

    runs = registry.get('runs')
    require(isinstance(runs, list) and len(runs) == 3, 'Three registered configs required')
    expected_runs = []
    for seed in (0, 1, 2):
        run_id = f'resnext_s{seed}'
        config_path = f'{directory}/configs/{run_id}.json'
        file_digest = checked(relative_path(config_path))
        resolved = object_file(config_path, file_digest)
        require(canonical(resolved) == canonical(base | {'seed': seed}), f'Registered seed config differs: {seed}')
        expected = dict(run_id=run_id, seed=seed, config_path=config_path,
            config_file_sha256=file_digest,
            resolved_config_sha256=hashlib.sha256(canonical(resolved).encode()).hexdigest(),
            output_dir=f'{directory}/{run_id}', status_at_registration='registered')
        require(canonical(runs[seed]) == canonical(expected), f'Registered run identity differs: {seed}')
        add_file(config_path, file_digest)
        expected_runs.append(expected)
    require(checksum_table(registry.get('files_sha256')) == files,
            'Complete registry evidence file table differs')
    selected = expected_runs[config['seed']]
    require(relative_path(selected['output_dir']) == run, 'Full run output differs from registered directory')
    expected_registration = dict(registry_path=recorded['registry_path'],
        registry_sha256=recorded['registry_sha256'], protocol=config['protocol'],
        registered_utc=registry['registered_utc'], acceptance_sha256=registry['acceptance_sha256'],
        **{key: selected[key] for key in ('run_id', 'seed', 'config_path', 'config_file_sha256',
                                        'resolved_config_sha256', 'output_dir')})
    require(canonical(recorded) == canonical(expected_registration), 'Report execution_registration identity differs')
    return expected_registration


def numpy_metrics(labels, predictions, class_count):
    """Full-class confusion/precision/recall/F1, independently from sklearn."""
    true, pred = np.asarray(labels), np.asarray(predictions)
    require(true.ndim == pred.ndim == 1 and len(true) == len(pred) > 0,
            'Nonempty aligned metric arrays required')
    require(true.dtype.kind in 'iu' and pred.dtype.kind in 'iu' and
            np.all((true >= 0) & (true < class_count)) and
            np.all((pred >= 0) & (pred < class_count)), 'Metric class outside class list')
    matrix = np.bincount(true.astype(np.int64) * class_count + pred,
                         minlength=class_count ** 2).reshape(class_count, class_count)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = matrix.diagonal().astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros(class_count), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(class_count), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(class_count),
                   where=(support + predicted) != 0)
    return dict(accuracy=float(tp.sum() / len(true)), macro_f1=float(f1.mean()),
                confusion_matrix=matrix.tolist(),
                per_class={str(c): {'precision': float(precision[c]), 'recall': float(recall[c]),
                                  'f1-score': float(f1[c]), 'support': int(support[c])}
                           for c in range(class_count)})


def metric_error(actual, recorded, location):
    require(isinstance(recorded, dict) and set(actual) == set(recorded),
            f'Metric schema differs: {location}')
    require(actual['confusion_matrix'] == recorded['confusion_matrix'],
            f'Confusion matrix differs: {location}')
    require(set(actual['per_class']) == set(recorded['per_class']),
            f'Per-class keys differ: {location}')
    errors = []
    for name in ('accuracy', 'macro_f1'):
        require(type(recorded[name]) in (float, int) and np.isfinite(recorded[name]),
                f'Invalid metric: {location}/{name}')
        errors.append(abs(actual[name] - recorded[name]))
    for cls, detail in actual['per_class'].items():
        other = recorded['per_class'][cls]
        require(isinstance(other, dict) and set(other) == set(detail),
                f'Per-class schema differs: {location}/{cls}')
        for name, value in detail.items():
            require(type(other[name]) in (float, int) and np.isfinite(other[name]),
                    f'Invalid per-class metric: {location}/{cls}/{name}')
            if name == 'support':
                require(value == other[name], f'Support differs: {location}/{cls}')
            errors.append(abs(value - other[name]))
    maximum = max(errors)
    require(maximum <= METRIC_ATOL, f'Metric differs: {location}, maximum error={maximum}')
    return maximum


def validate_history(report):
    """Check the frozen method contract and independently select the epoch."""
    config = report['config']
    template = read_json(SOURCE_ROOT / 'config/paper_baseline_v1.json')
    require(isinstance(config, dict) and set(config) == set(template) | {'seed'},
            'Complete configuration required')
    adjustable = {'purpose', 'device', 'threads', 'batch_size', 'max_epochs',
                  'validation_every_epochs', 'train_per_class', 'validation_per_class',
                  'cache_dir', 'notes', 'seed'}
    for key in set(template) - adjustable:
        require(config[key] == template[key] and type(config[key]) is type(template[key]),
                f'Frozen configuration differs: {key}')
    require(config['purpose'] in ('exploratory_train_validation_method_reconstruction',
                                 'paper_runner_smoke_validation_only'), 'Unknown run purpose')
    require(config['device'] in ('auto', 'cpu', 'cuda'), 'Invalid recorded device')
    require(type(config['seed']) is int and config['seed'] in (0, 1, 2), 'Invalid seed')
    for key in ('threads', 'batch_size', 'max_epochs', 'validation_every_epochs'):
        require(type(config[key]) is int and config[key] > 0, f'Invalid {key}')
    for key in ('train_per_class', 'validation_per_class'):
        require(config[key] is None or (type(config[key]) is int and config[key] > 0),
                f'Invalid {key}')
    require(isinstance(config['cache_dir'], str) and config['cache_dir'] and
            isinstance(config['notes'], str), 'Cache path and notes required')
    if config['purpose'] == 'exploratory_train_validation_method_reconstruction':
        for key in ('threads', 'batch_size', 'max_epochs', 'validation_every_epochs',
                    'train_per_class', 'validation_per_class', 'device'):
            require(config[key] == template[key] and type(config[key]) is type(template[key]),
                    f'Full-data configuration differs: {key}')
    require(config['max_epochs'] % config['validation_every_epochs'] == 0,
            'Final epoch must include scheduled validation')
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    require(report.get('config_sha256') == digest, 'Configuration checksum differs')
    require(report.get('protocol') == config['protocol'] and report.get('purpose') == config['purpose'],
            'Report protocol/purpose differs')
    history = report.get('epochs')
    require(isinstance(history, list) and len(history) == config['max_epochs'],
            'Complete epoch history required')
    eligible = []
    for number, entry in enumerate(history, 1):
        require(type(entry.get('epoch')) is int and entry['epoch'] == number,
                'Epoch history order differs')
        lr = config['learning_rate']
        for milestone, value in zip(config['learning_rate_milestones_epochs'], config['learning_rate_values']):
            if number >= milestone:
                lr = value
        require(entry.get('learning_rate') == lr, 'Learning rate history differs')
        for key in ('train_loss', 'online_train_accuracy', 'elapsed_seconds'):
            require(type(entry.get(key)) in (int, float) and np.isfinite(entry[key]) and entry[key] >= 0,
                    f'Invalid history {key}')
        require(entry['online_train_accuracy'] <= 1, 'Invalid online train accuracy')
        due = number % config['validation_every_epochs'] == 0
        require((entry.get('validation') is not None) == due, 'Validation history interval differs')
        if due:
            require(set(entry['validation']) == {'accuracy', 'macro_f1'}, 'Validation history metrics required')
            for value in entry['validation'].values():
                require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1,
                        'Invalid history validation metric')
            eligible.append(entry)
    chosen = max(eligible, key=lambda entry: (entry['validation']['accuracy'], -entry['epoch']))
    require(type(report.get('selected_epoch')) is int and report['selected_epoch'] == chosen['epoch'],
            'Selected epoch differs from accuracy/earlier-epoch history')
    for key in ('accuracy', 'macro_f1'):
        require(report['metrics']['validation'][key] == chosen['validation'][key],
                'Selected validation metric differs from history')
    classes = report.get('classes')
    require(isinstance(classes, list) and len(classes) >= 2 and
            all(type(c) is int for c in classes) and classes == list(range(len(classes))),
            'Contiguous integer classes required')
    require(set(report['metrics']) == set(SPLITS) and set(report['group_metrics']) == set(SPLITS),
            'Only train/validation metrics allowed')
    require(set(report['sample_counts']) == set(SPLITS) and
            set(report['available_sample_counts']) == set(SPLITS), 'Sample count schema differs')
    return config


def independent_probabilities(model, raw, identities, mean, std, device, batch_size):
    """CPU PCG64 draws/slices, float64 normalization -> float32, mean softmax."""
    require(raw.shape == (len(identities), 250, 2) and np.isfinite(raw).all(),
            'Finite raw 250x2 signals required')
    result = []
    with torch.inference_mode():
        for begin in range(0, len(raw), batch_size):
            end = min(begin + batch_size, len(raw))
            crops = np.empty(((end - begin) * 10, 224, 2), dtype=np.float32)
            for index, identity in enumerate(identities[begin:end]):
                key = json.dumps(['validation', 10, identity, None], ensure_ascii=True,
                                 separators=(',', ':'), allow_nan=False).encode('ascii')
                entropy = int.from_bytes(hashlib.sha256(key).digest(), byteorder='big')
                generator = np.random.Generator(np.random.PCG64(entropy))
                starts = generator.integers(0, 27, size=10, dtype=np.int64)
                for crop, start in enumerate(starts):
                    crops[index * 10 + crop] = raw[begin + index, start:start + 224, :]
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                normalized = ((crops.astype(np.float64) - mean) / std).astype(np.float32)
            require(np.isfinite(normalized).all(), 'Nonfinite standardized crops')
            probabilities = []
            for offset in range(0, len(crops), batch_size):
                inputs = torch.from_numpy(np.ascontiguousarray(
                    normalized[offset:offset + batch_size].transpose(0, 2, 1))).to(device)
                logits = model(inputs)
                require(torch.isfinite(logits).all().item(), 'Nonfinite independent logits')
                probabilities.append(torch.softmax(logits, dim=1).float().cpu().numpy())
            averaged = np.concatenate(probabilities).reshape(end - begin, 10, -1).mean(axis=1)
            require(np.isfinite(averaged).all(), 'Nonfinite independent probabilities')
            result.append(averaged)
    return np.concatenate(result)


def verify_paper_run(project_root, run_dir, output_path, device='cpu', split='both', batch_size=128):
    """Publish and return independent evidence, or raise without a success file."""
    started = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    require(split in ('train', 'validation', 'both'), 'Only train/validation/both allowed; test sealed')
    require(device in ('auto', 'cpu', 'cuda'), 'Device must be auto/cpu/cuda')
    require(type(batch_size) is int and batch_size > 0, 'Positive integer batch size required')
    root = Path(project_root).resolve()
    run = inside(root, run_dir)
    output = Path(output_path)
    output = (output if output.is_absolute() else root / output).resolve()
    require(not output.is_relative_to(run), 'Verification output must be outside original run')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    require(not (run / 'failure.json').exists(), 'Failed run cannot be verified')
    require(not (run / 'running.marker').exists(), 'Incomplete running.marker run cannot be verified')
    evidence = {}

    def checked(path, expected=None):
        path = Path(path).resolve()
        actual = sha256(path)
        if expected is not None:
            require(isinstance(expected, str) and actual == expected, f'Checksum differs: {path}')
        if str(path) in evidence:
            require(actual == evidence[str(path)], f'File changed during verification: {path}')
        evidence[str(path)] = actual
        return actual

    report_sha = checked(run / 'report.json')
    report = read_json(run / 'report.json')
    require(report.get('status') == 'complete' and type(report.get('schema_version')) is int and
            report['schema_version'] == 1, 'Complete schema-1 report required')
    require(report.get('heldout_test_evaluated') is False and report.get('model_roundtrip_verified') is True,
            'Test must be sealed and CPU roundtrip completed')
    config = validate_history(report)
    require('execution_registration' in report and
            (report['execution_registration'] is None if config['purpose'] == 'paper_runner_smoke_validation_only'
             else isinstance(report['execution_registration'], dict)),
            'Smoke requires null; full purpose requires execution_registration')
    checked(SOURCE_ROOT / 'config/paper_baseline_v1.json')
    require(set(report.get('files_sha256', {})) == ARTIFACTS, 'Complete artifact SHA table required')
    artifacts = {name: checked(run / name, report['files_sha256'][name]) for name in sorted(ARTIFACTS)}
    require(set(report.get('code_sha256', {})) == SOURCES and
            set(report.get('code_sha256_lf', {})) == SOURCES, 'Complete executed source SHA tables required')
    sources = {}
    for name in sorted(SOURCES):
        path = EXECUTED_MODULE_PATHS.get(name, SOURCE_ROOT / name)
        sources[name] = checked(path, report['code_sha256'][name])
        if name in IMPORTED_MODULE_BYTES:
            require(hashlib.sha256(IMPORTED_MODULE_BYTES[name]).hexdigest() == sources[name],
                    f'Imported inference source changed since import: {name}')
        canonical = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        require(canonical == report['code_sha256_lf'][name], f'Canonical source differs: {name}')
    script_path = Path(__file__).resolve()
    script_sha = checked(script_path)
    script_text = script_path.read_bytes().decode('utf-8')
    verify_loaded_inference_sources()

    cache = inside(root, report['cache_path'])
    metadata_sha = checked(cache / 'metadata.json', report['cache_metadata_sha256'])
    norm_sha = checked(cache / 'normalization.json', report['normalization_sha256'])
    metadata = read_json(cache / 'metadata.json')
    stats = read_json(cache / 'normalization.json')
    require(metadata.get('schema_version') == 1 and metadata.get('signal_shape') == [250, 2] and
            metadata.get('x_dtype') == 'float32' and metadata.get('y_dtype') == 'int64' and
            metadata.get('cache_units') == 'raw input units, not standardized', 'Invalid cache protocol')
    summary_path = inside(root, metadata['summary_path'])
    summary_sha = checked(summary_path, metadata['summary_sha256'])
    summary = read_json(summary_path)
    require(metadata['manifest_sha256'] == summary['manifest_sha256'] == report['manifest_sha256'] and
            metadata['inputs'] == summary['inputs'], 'Manifest/raw provenance differs')
    require(metadata['sample_counts'] == summary['sample_counts'] and
            metadata['class_counts'] == summary['class_counts'], 'Cache counts differ from manifest summary')
    manifests = metadata['manifest_sha256']
    require(set(manifests) == {'train.jsonl', 'validation.jsonl', 'test.jsonl'}, 'Complete manifest SHA table required')
    for name, digest in manifests.items():
        checked(summary_path.parent / name, digest)
    require(report['raw_input_sha256'] == {k: v['sha256'] for k, v in metadata['inputs'].items()},
            'Raw input SHA table differs')
    for info in metadata['inputs'].values():
        checked(inside(root, info['path']), info['sha256'])
    cache_files = metadata['files_sha256']
    require(set(cache_files) == {f'{s}_{k}.npy' for s in (*SPLITS, 'test') for k in ('x', 'y')} |
            {'normalization.json'}, 'Complete cache SHA table required')
    for name, digest in cache_files.items():
        checked(cache / name, digest)
    require(stats == report['normalization'] and stats.get('fitted_split') == 'train' and
            stats.get('ddof') == 0 and stats.get('train_cache_sha256') == cache_files['train_x.npy'] and
            stats.get('train_manifest_sha256') == manifests['train.jsonl'] and
            stats.get('count_per_channel') == summary['sample_counts']['train'] * 250,
            'Frozen train normalization provenance differs')
    mean, std = np.asarray(stats['mean'], dtype=np.float64), np.asarray(stats['std'], dtype=np.float64)
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all() and np.isfinite(std).all() and
            (std > 0).all(), 'Invalid normalization mean/std')
    registration = verify_execution_registration(root, run, report, cache, metadata, checked)

    # Check both row/CSV identity chains even when only one split is inferred.
    selected, records, csv_predictions = {}, {}, {}
    class_count = len(report['classes'])
    for current in SPLITS:
        path = summary_path.parent / f'{current}.jsonl'
        all_records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        labels = np.asarray([record['class_index'] for record in all_records])
        require(len(all_records) == summary['sample_counts'][current] == report['available_sample_counts'][current],
                f'Available sample counts differ: {current}')
        require(labels.dtype.kind in 'iu' and np.array_equal(np.unique(labels), np.arange(class_count)),
                f'Manifest classes differ: {current}')
        rows = read_json(run / f'{current}_rows.json')
        require(isinstance(rows, list) and len(rows) > 0 and all(type(i) is int for i in rows) and
                rows == sorted(set(rows)) and rows[0] >= 0 and rows[-1] < len(labels),
                f'Unique ordered in-range row indices required: {current}')
        cap = config[f'{current}_per_class']
        if cap is None:
            expected = list(range(len(labels)))
        else:
            rng = np.random.default_rng(20261002)
            expected = []
            for cls in np.unique(labels):
                options = np.flatnonzero(labels == cls)
                expected.extend(rng.choice(options, size=min(cap, len(options)), replace=False).tolist())
            expected.sort()
        require(rows == expected and len(rows) == report['sample_counts'][current],
                f'Selected row protocol/count differs: {current}')
        selected[current] = rows
        records[current] = [all_records[i] for i in rows]
        require(np.array_equal(np.unique(labels[rows]), np.arange(class_count)),
                f'Selected classes differ: {current}')
        require(all(r.get('split') == current and isinstance(r.get('wave_sha256'), str) and
                    len(r['wave_sha256']) == 64 for r in records[current]), f'Manifest identities differ: {current}')
        with (run / f'{current}_predictions.csv').open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames == CSV_FIELDS, f'Prediction CSV schema differs: {current}')
            csv_rows = list(reader)
        require(len(csv_rows) == len(rows), f'Prediction CSV count differs: {current}')
        predictions = []
        for index, record, row in zip(rows, records[current], csv_rows):
            require(set(row) == set(CSV_FIELDS) and int(row['row_index']) == index and
                    row['wave_sha256'] == record['wave_sha256'] and row['group_id'] == str(record['group_id']) and
                    int(row['class_index']) == record['class_index'], f'Prediction CSV identity/order differs: {current}')
            prediction = int(row['predicted_class_index'])
            require(0 <= prediction < class_count, f'CSV prediction class outside class list: {current}')
            predictions.append(prediction)
        csv_predictions[current] = np.asarray(predictions, dtype=np.int64)

    # Configure old deterministic/TF32 settings BEFORE deserializing the state.
    seed_everything(config['seed'], threads=config['threads'])
    inference_device = resolve_device(device)
    def portable_storage(storage, location):
        require(location == 'cpu', 'Checkpoint must contain portable CPU storage')
        return storage
    saved = torch.load(run / 'model.pt', map_location=portable_storage, weights_only=True)
    require(type(saved.get('schema_version')) is int and saved['schema_version'] == 1 and
            saved.get('protocol') == report['protocol'] and saved.get('config') == config and
            saved.get('normalization') == stats and saved.get('selected_epoch') == report['selected_epoch'] and
            saved.get('classes') == report['classes'], 'Checkpoint metadata differs from report')
    require(isinstance(saved.get('state_dict'), dict) and saved['state_dict'] and
            all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and torch.isfinite(t).all().item()
                for t in saved['state_dict'].values()), 'Finite CPU state dictionary required')
    model = PaperResNeXt1D(num_classes=class_count, blocks_per_stage=config['blocks_per_stage'])
    model.load_state_dict(saved['state_dict'], strict=True)
    require(report.get('parameter_count') == sum(p.numel() for p in model.parameters()), 'Parameter count differs')
    model.to(inference_device).eval()

    requested = SPLITS if split == 'both' else (split,)
    metrics, group_metrics, prediction_counts, group_counts = {}, {}, {}, {}
    max_error = 0.0
    for current in requested:
        mappings = []
        try:
            raw = np.load(cache / f'{current}_x.npy', allow_pickle=False, mmap_mode='r')
            mappings.append(raw)
            labels = np.load(cache / f'{current}_y.npy', allow_pickle=False, mmap_mode='r')
            mappings.append(labels)
            n = summary['sample_counts'][current]
            require(raw.shape == (n, 250, 2) and raw.dtype == np.float32 and
                    labels.shape == (n,) and labels.dtype == np.int64, 'Cache shape/dtype differs')
            all_labels = [json.loads(line)['class_index'] for line in
                          (summary_path.parent / f'{current}.jsonl').read_text().splitlines()]
            require(np.array_equal(labels, all_labels), 'Cache labels differ from manifest order')
            rows = selected[current]
            true = np.asarray(labels[rows], dtype=np.int64)
            identities = [record['wave_sha256'] for record in records[current]]
            probabilities = independent_probabilities(model, raw[rows], identities, mean, std,
                                                     inference_device, batch_size)
            predicted = probabilities.argmax(axis=1)
            require(np.array_equal(predicted, csv_predictions[current]),
                    f'Independent predictions differ from CSV: {current}; mismatches='
                    f'{int(np.count_nonzero(predicted != csv_predictions[current]))}')
            metrics[current] = numpy_metrics(true, csv_predictions[current], class_count)
            max_error = max(max_error, metric_error(metrics[current], report['metrics'][current], current))
            groups = np.asarray([str(record['group_id']) for record in records[current]])
            require(set(report['group_metrics'][current]) == set(np.unique(groups)),
                    f'Group metric keys differ: {current}')
            group_metrics[current], group_counts[current] = {}, {}
            for group in np.unique(groups):
                mask = groups == group
                metric = numpy_metrics(true[mask], csv_predictions[current][mask], class_count)
                group_metrics[current][str(group)] = metric
                group_counts[current][str(group)] = int(mask.sum())
                max_error = max(max_error, metric_error(metric, report['group_metrics'][current][group],
                                                        f'{current}/group/{group}'))
            prediction_counts[current] = len(predicted)
        finally:
            for mapped in mappings:
                mapped._mmap.close()

    # Detect drift between initial validation and publishing the evidence.
    require(not (run / 'failure.json').exists(), 'Failure marker appeared during verification')
    require(not (run / 'running.marker').exists(), 'Incomplete running.marker appeared during verification')
    for path, digest in evidence.items():
        require(sha256(path) == digest, f'File changed during verification: {path}')
        evidence_not_failed(root, Path(path))
    verify_loaded_inference_sources()
    proof = dict(schema_version=1, status='verified', created_utc=created,
        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
        project_root=str(root), run_path=str(run), source_root=str(SOURCE_ROOT), split=split,
        device_request=device, device=str(inference_device), batch_size=batch_size,
        environment=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                         torch_cuda_build=torch.version.cuda),
        determinism=dict(seed=config['seed'], threads=torch.get_num_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            tf32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32)),
        selected_epoch=report['selected_epoch'], classes=report['classes'],
        script=dict(path=str(script_path), text=script_text, sha256=script_sha),
        report_sha256=report_sha, weights_sha256=artifacts['model.pt'], artifacts_sha256=artifacts,
        source_sha256=sources,
        executed_module_paths={name: str(path) for name, path in EXECUTED_MODULE_PATHS.items()},
        cache_metadata_sha256=metadata_sha, normalization_sha256=norm_sha,
        summary_sha256=summary_sha, manifest_sha256=manifests, cache_files_sha256=cache_files,
        raw_input_sha256=report['raw_input_sha256'], files_sha256=evidence,
        execution_registration=registration,
        prediction_counts=prediction_counts, group_counts=group_counts,
        maximum_metric_error=max_error, metric_tolerance=METRIC_ATOL,
        metrics=metrics, group_metrics=group_metrics,
        flags=dict(test_classification_evaluated=False, test_files_hashed_only=True,
            complete_artifacts_verified=True, source_verified=True, manifest_cache_verified=True,
            checkpoint_cpu_verified=True, history_selection_verified=True,
            all_run_row_csv_identities_verified=True, requested_predictions_match=True,
            requested_metrics_match=True, requested_group_metrics_match=True,
            independent_inference=True, original_run_read_only=True,
            execution_registration_verified=(registration is not None),
            both_splits_inferred=(split == 'both')))
    text = json.dumps(proof, indent=2, ensure_ascii=True, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    committed = False
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=output.parent, prefix='.paper-proof-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        require(not (run / 'failure.json').exists(), 'Failure marker appeared before publication')
        require(not (run / 'running.marker').exists(), 'Incomplete running.marker appeared before publication')
        # Atomic no-overwrite publication, including a concurrently created output.
        os.link(temporary, output)
        committed = True
    finally:
        failing = sys.exc_info()[0] is not None
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # link is the commit point. Cleanup cannot turn a committed
                # proof into a failed call; leave the scratch file for recovery.
                if not committed and not failing:
                    raise
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file outside the run')
    parser.add_argument('--split', choices=['train', 'validation', 'both'], default='both')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    try:
        proof = verify_paper_run(args.project_root, args.run, args.output,
                                 device=args.device, split=args.split, batch_size=args.batch_size)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'Verification failed: {type(error).__name__}: {error}\n')
    print(json.dumps({k: proof[k] for k in ('status', 'device', 'prediction_counts', 'maximum_metric_error')},
                     allow_nan=False))


if __name__ == '__main__':
    main()
