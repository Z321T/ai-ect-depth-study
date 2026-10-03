"""Independently replay train-only BN recalibration and verify both split predictions.

Never imports the BN producer, its loader, crops, normalizer or metrics.
Only a new proof outside both immutable runs is published, atomically.
"""
import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import torch
from scripts import verify_paper_run as old

PROTOCOL = 'paper_bn_recalibration_v1'
CONFIG = dict(calibration_batch_size=128, calibration_seed=0, calibration_epoch=0,
              order='manifest', passes=1, evaluation_batch_size=128, evaluation_seed=10,
              threads=1, deterministic=True, tf32=False, test_classification_enabled=False)
SOURCES = old.REGISTRY_SOURCES | {'src/ect/paper_bn.py', 'scripts/recalibrate_paper_bn.py',
                                'scripts/verify_paper_bn.py'}
ARTIFACTS = old.ARTIFACTS
SPLITS = old.SPLITS
BUFFER_RTOL = 2e-4
BUFFER_ATOL = 2e-5
require = old.require
sha256 = old.sha256
read_json = old.read_json
inside = old.inside
independent_probabilities = old.independent_probabilities
numpy_metrics = old.numpy_metrics
metric_error = old.metric_error
_SCRIPT_PATH = Path(__file__).resolve()
_SCRIPT_BYTES = _SCRIPT_PATH.read_bytes()
_OLD_PATH = Path(old.__file__).resolve()
_OLD_BYTES = _OLD_PATH.read_bytes()
_HELPER_ALIASES = {name: globals()[name] for name in (
    'require', 'sha256', 'read_json', 'inside', 'independent_probabilities', 'numpy_metrics', 'metric_error')}


def independent_calibration(original, raw, identities, mean, std, device):
    """Clone and manually replay the single seed0/epoch0 training-crop pass."""
    require(raw.shape == (len(identities), 250, 2) and len(raw) > 0 and np.isfinite(raw).all(),
            'Finite nonempty calibration inputs required')
    require(all(isinstance(s, str) and s.strip() for s in identities), 'Calibration identities required')
    mean, std = np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all() and
            np.isfinite(std).all() and (std > 0).all(), 'Invalid calibration normalization')
    model = copy.deepcopy(original).to(device)
    batchnorms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    require(batchnorms and all(m.track_running_stats for m in batchnorms), 'Tracked BN modules required')
    momenta = [m.momentum for m in batchnorms]
    for module in batchnorms:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    batches = 0
    try:
        with torch.no_grad():
            for begin in range(0, len(raw), 128):
                end = min(begin + 128, len(raw))
                crops = np.empty((end - begin, 224, 2), dtype=np.float32)
                for index, identity in enumerate(identities[begin:end]):
                    key = json.dumps(['training', 0, identity, 0], ensure_ascii=True,
                                     separators=(',', ':'), allow_nan=False).encode('ascii')
                    entropy = int.from_bytes(hashlib.sha256(key).digest(), 'big')
                    rng = np.random.Generator(np.random.PCG64(entropy))
                    start = int(rng.integers(0, 27, size=1, dtype=np.int64)[0])
                    crops[index] = raw[begin + index, start:start + 224]
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    normalized = ((crops.astype(np.float64) - mean) / std).astype(np.float32)
                require(np.isfinite(normalized).all(), 'Nonfinite calibration standardization')
                inputs = torch.from_numpy(np.ascontiguousarray(normalized.transpose(0, 2, 1))).to(device)
                require(torch.isfinite(model(inputs)).all().item(), 'Nonfinite calibration output')
                batches += 1
    finally:
        for module, momentum in zip(batchnorms, momenta):
            module.momentum = momentum
        model.eval()
    return model, dict(sample_count=len(raw), batch_count=batches,
                       last_batch_size=(len(raw) - 1) % 128 + 1)


def compare_bn_states(original, calibrated, replayed, same_device):
    """Check exact learned state and bounded independently recomputed BN buffers."""
    states = [m.state_dict() for m in (original, calibrated, replayed)]
    require(set(states[0]) == set(states[1]) == set(states[2]), 'Checkpoint state keys differ')
    allowed = {f'{name}.{field}' if name else field
               for name, module in original.named_modules() if isinstance(module, torch.nn.BatchNorm1d)
               for field in ('running_mean', 'running_var', 'num_batches_tracked')}
    changed, maximum, exact = [], 0., True
    parameters = set(dict(original.named_parameters()))
    for name, origin in states[0].items():
        saved, replay = (state[name].detach().cpu() for state in states[1:])
        origin = origin.detach().cpu()
        require(saved.shape == replay.shape == origin.shape and saved.dtype == replay.dtype == origin.dtype,
                f'State shape/dtype differs: {name}')
        require(torch.isfinite(saved).all().item() and torch.isfinite(replay).all().item() and
                torch.isfinite(origin).all().item(), f'Nonfinite state: {name}')
        if name not in allowed:
            require(torch.equal(saved, origin) and torch.equal(replay, origin),
                    f'Learned parameter or non-BN buffer changed: {name}')
            continue
        require(name not in parameters, f'BN whitelist overlaps learned parameter: {name}')
        if not torch.equal(saved, origin):
            changed.append(name)
        equal = torch.equal(saved, replay)
        exact = exact and equal
        error = float((saved.to(torch.float64) - replay.to(torch.float64)).abs().max().item())
        maximum = max(maximum, error)
        if name.endswith('num_batches_tracked'):
            require(equal, f'BN counter differs: {name}')
        elif same_device:
            require(equal, f'Same-device calibration must be exact: {name}; error={error}')
        else:
            require(torch.allclose(saved, replay, rtol=BUFFER_RTOL, atol=BUFFER_ATOL),
                    f'Cross-device calibration buffer differs: {name}; error={error}')
        if name.endswith('running_var'):
            require((saved >= 0).all().item(), f'Negative BN running variance: {name}')
    return dict(maximum_error=maximum, exact=exact, changed_buffers=sorted(changed))


def verify_loaded_sources():
    """Bind loaded verifier/helper functions to their immutable import-time bytes."""
    old.verify_loaded_inference_sources()
    for path, source, namespace in ((_SCRIPT_PATH, _SCRIPT_BYTES, globals()),
                                    (_OLD_PATH, _OLD_BYTES, vars(old))):
        require(path.read_bytes() == source, f'Verification source changed since import: {path}')
        code = compile(source, str(path), 'exec', dont_inherit=True)
        expected = {child.co_name: child for child in code.co_consts if isinstance(child, types.CodeType)}
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = namespace.get(node.name)
                require(isinstance(function, types.FunctionType) and function.__code__ == expected[node.name],
                        f'Loaded verification function differs from source: {path}/{node.name}')
                defaults = tuple(ast.literal_eval(value) for value in node.args.defaults) or None
                kwdefaults = {arg.arg: ast.literal_eval(value) for arg, value in
                              zip(node.args.kwonlyargs, node.args.kw_defaults) if value is not None} or None
                require(function.__defaults__ == defaults and function.__kwdefaults__ == kwdefaults,
                        f'Loaded verification defaults differ: {path}/{node.name}')
    require(all(globals()[name] is value and getattr(old, name) is value
                for name, value in _HELPER_ALIASES.items()), 'Independent helper aliases changed')


def relative_file(root, label):
    require(isinstance(label, str) and label and not Path(label).is_absolute(),
            'Root-relative evidence path required')
    path = inside(root, label)
    require(path != root and path.relative_to(root).as_posix() == label,
            f'Canonical root-relative evidence path required: {label}')
    old.evidence_not_failed(root, path)
    return path


def check_hash_table(table, expected_keys=None):
    require(isinstance(table, dict) and table and all(isinstance(k, str) and isinstance(v, str) and
            len(v) == 64 and all(c in '0123456789abcdef' for c in v) for k, v in table.items()),
            'Complete nonempty SHA256 table required')
    if expected_keys is not None:
        require(set(table) == set(expected_keys), 'Complete SHA256 table keys required')
    return table


def read_prediction_csv(path, rows, records, classes):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == old.CSV_FIELDS, f'Prediction CSV fields differ: {path}')
        csv_rows = list(reader)
    require(len(csv_rows) == len(rows), f'Prediction count differs: {path}')
    predictions = []
    for index, record, entry in zip(rows, records, csv_rows):
        require(set(entry) == set(old.CSV_FIELDS) and int(entry['row_index']) == index and
                entry['wave_sha256'] == record['wave_sha256'] and
                entry['group_id'] == str(record['group_id']) and
                int(entry['class_index']) == record['class_index'],
                f'Prediction identity/order differs: {path}')
        predicted = int(entry['predicted_class_index'])
        require(0 <= predicted < classes, f'Prediction class outside class list: {path}')
        predictions.append(predicted)
    return np.asarray(predictions, dtype=np.int64)


def checked_metrics(labels, predictions, records, classes, recorded, recorded_groups, location):
    actual = numpy_metrics(labels, predictions, classes)
    maximum = metric_error(actual, recorded, location)
    groups = np.asarray([str(record['group_id']) for record in records])
    require(isinstance(recorded_groups, dict) and set(recorded_groups) == set(groups),
            f'Group metric identities differ: {location}')
    details, counts = {}, {}
    for group in np.unique(groups):
        mask = groups == group
        details[group] = numpy_metrics(labels[mask], predictions[mask], classes)
        counts[group] = int(mask.sum())
        maximum = max(maximum, metric_error(details[group], recorded_groups[group], f'{location}/{group}'))
    return actual, details, counts, maximum


def check_registration(root, run, report, origin, origin_report, checked, required_anchors):
    """Independently bind registry anchors and acceptance's actual smoke proofs."""
    recorded = report['execution_registration']
    if report['purpose'] == 'smoke':
        require(recorded is None, 'Smoke execution_registration must be null')
        return None
    require(isinstance(recorded, dict) and set(recorded) == {'registry_path', 'registry_sha256', 'run_id'},
            'Complete full execution_registration required')
    path = relative_file(root, recorded['registry_path'])
    checked(path, recorded['registry_sha256'])
    registry = read_json(path)
    require(type(registry.get('schema_version')) is int and registry['schema_version'] == 1 and
            registry.get('protocol') == PROTOCOL and registry.get('status') == 'frozen_execution_registration' and
            registry.get('test_classification_enabled') is False and
            old.canonical(registry.get('config')) == old.canonical(report['config']) and
            isinstance(registry.get('created_utc'), str),
            'Frozen BN execution registry required')
    require(check_hash_table(registry.get('code_sha256'), SOURCES) == report['code_sha256'],
            'Registry/run source table differs')
    files = check_hash_table(registry.get('files_sha256'))
    for label, digest in files.items():
        checked(relative_file(root, label), digest)
    for path, digest in required_anchors.items():
        label = Path(path).relative_to(root).as_posix()
        require(files.get(label) == digest, f'Missing or different registration anchor: {label}')
    runs = registry.get('runs')
    fields = {'run_id', 'origin_run', 'output_dir', 'origin_report_sha256', 'origin_model_sha256',
              'seed', 'origin_verification_path', 'origin_verification_sha256'}
    require(isinstance(runs, list) and len(runs) == 3 and all(isinstance(r, dict) and set(r) == fields
            for r in runs), 'Three complete registered run entries required')
    require(sorted(r['seed'] for r in runs) == [0, 1, 2] and
            all(type(r['seed']) is int for r in runs) and
            len({r['run_id'] for r in runs}) == len({r['output_dir'] for r in runs}) ==
            len({r['origin_run'] for r in runs}) == 3, 'Registered seed/run identities differ')
    matches = [r for r in runs if r['run_id'] == recorded['run_id']]
    require(len(matches) == 1, 'Run missing from execution registry')
    entry = matches[0]
    require(relative_file(root, entry['output_dir']) == run and relative_file(root, entry['origin_run']) == origin
            and entry['seed'] == report['seed'] and entry['origin_report_sha256'] == report['origin_report_sha256']
            and entry['origin_model_sha256'] == report['origin_model_sha256'] and
            entry['origin_verification_path'] == report['origin_verification_path'],
            'Registered output/origin identity differs')
    for item in runs:
        original = relative_file(root, item['origin_run'])
        old_report_path, weights = original / 'report.json', original / 'model.pt'
        proof_path = relative_file(root, item['origin_verification_path'])
        for anchor, digest in ((old_report_path, item['origin_report_sha256']),
                               (weights, item['origin_model_sha256']),
                               (proof_path, item['origin_verification_sha256'])):
            require(files.get(anchor.relative_to(root).as_posix()) == digest,
                    'Registry missing origin/proof anchor')
            checked(anchor, digest)
        proof = read_json(proof_path)
        require(proof.get('schema_version') == 1 and proof.get('status') == 'verified' and
                proof.get('report_sha256') == item['origin_report_sha256'] and
                proof.get('weights_sha256') == item['origin_model_sha256'] and
                proof.get('prediction_counts') == {'train': 24000, 'validation': 3200} and
                proof.get('maximum_metric_error') == 0 and proof.get('flags', {}).get('both_splits_inferred') is True,
                'Verified full origin proof required')
        original_registration = read_json(old_report_path).get('execution_registration')
        require(isinstance(original_registration, dict), 'Full origin execution registration required')
        old_registry = relative_file(root, original_registration['registry_path'])
        require(files.get(old_registry.relative_to(root).as_posix()) == original_registration['registry_sha256'],
                'Old registry anchor missing')
        checked(old_registry, original_registration['registry_sha256'])
    acceptance_path = relative_file(root, registry['acceptance_path'])
    checked(acceptance_path, registry['acceptance_sha256'])
    require(files.get(registry['acceptance_path']) == registry['acceptance_sha256'],
            'Acceptance anchor missing from registry')
    acceptance = read_json(acceptance_path)
    require(type(acceptance.get('schema_version')) is int and acceptance['schema_version'] == 1 and
            acceptance.get('status') == 'verified' and acceptance.get('protocol') == 'paper_bn_acceptance_v1' and
            acceptance.get('code_sha256') == report['code_sha256'] and
            isinstance(acceptance.get('checks'), dict) and acceptance['checks'] and
            all(value is True for value in acceptance['checks'].values()), 'Complete verified BN acceptance required')
    acceptance_files = check_hash_table(acceptance.get('files_sha256'))
    for label, digest in acceptance_files.items():
        require(files.get(label) == digest, f'Registry missing acceptance evidence: {label}')
        checked(relative_file(root, label), digest)
    smoke_runs = acceptance.get('runs')
    require(isinstance(smoke_runs, list) and len(smoke_runs) == 2 and
            {r.get('device') for r in smoke_runs} == {'cpu', 'cuda'}, 'Actual CPU/CUDA smoke acceptance required')
    expected_checks = {'independent_calibration', 'bn_buffer_only_changes', 'parameters_unchanged',
                       'original_state_unchanged', 'both_splits_inferred', 'prediction_identities_verified',
                       'metrics_verified', 'test_classification_enabled'}
    # The registrar owns additional acceptance checks; verify actual evidence here.
    def smoke_proof(label, smoke_run, device):
        proof_path = relative_file(root, label)
        require(label in acceptance_files, 'Acceptance proof checksum required')
        proof = read_json(proof_path)
        report_path = smoke_run / 'report.json'
        require(report_path.relative_to(root).as_posix() in acceptance_files, 'Smoke report anchor required')
        smoke_report = read_json(report_path)
        require(proof.get('schema_version') == 1 and proof.get('status') == 'verified' and
                proof.get('protocol') == PROTOCOL and proof.get('device') == device and
                proof.get('report_sha256') == acceptance_files[report_path.relative_to(root).as_posix()] and
                proof.get('prediction_counts') == smoke_report.get('sample_counts') == {'train': 40, 'validation': 40}
                and smoke_report.get('status') == 'complete' and smoke_report.get('purpose') == 'smoke' and
                smoke_report.get('protocol') == PROTOCOL and smoke_report.get('code_sha256') == report['code_sha256'],
                'Actual completed 40/40 smoke report/proof required')
        checks = proof.get('checks', {})
        require(expected_checks <= set(checks) and all(checks[k] is True for k in expected_checks -
                {'test_classification_enabled'}) and checks['test_classification_enabled'] is False,
                'Independent smoke proof checks incomplete')
        require(proof.get('maximum_metric_error') == 0 and
                proof.get('verification_script', {}).get('sha256') == report['code_sha256']['scripts/verify_paper_bn.py'],
                'Smoke metric/source proof differs')
        require(hashlib.sha256(proof['verification_script']['text'].encode('utf-8')).hexdigest() ==
                proof['verification_script']['sha256'], 'Smoke verifier text SHA differs')
        for name, digest in check_hash_table(smoke_report.get('files_sha256'), ARTIFACTS).items():
            artifact = smoke_run / name
            require(acceptance_files.get(artifact.relative_to(root).as_posix()) == digest,
                    'Acceptance missing smoke artifact')
            checked(artifact, digest)
        for anchor, digest in check_hash_table(proof.get('checked_files_sha256')).items():
            # Proofs use absolute evidence paths; require the complete same-root chain.
            target = inside(root, anchor)
            checked(target, digest)
        return proof
    for item in smoke_runs:
        smoke_run = relative_file(root, item['run_dir'])
        proof = smoke_proof(item['verification_path'], smoke_run, item['device'])
        require(proof.get('same_device_calibration_exact') is True and
                proof.get('calibration_buffer_max_abs_error') == 0, 'Same-device smoke must be exact')
    cuda_run = relative_file(root, next(r for r in smoke_runs if r['device'] == 'cuda')['run_dir'])
    smoke_proof(acceptance['cpu_transfer_verification_path'], cuda_run, 'cpu')
    return recorded


def verify_paper_bn(project_root, run_dir, output_path, device='cpu'):
    """Publish independently recomputed evidence, or raise without publishing."""
    started = time.perf_counter()
    root = Path(project_root).resolve()
    run = inside(root, run_dir)
    output = inside(root, output_path)
    require(output != root and not output.is_relative_to(run), 'Proof must be outside recalibrated run')
    require(device in ('auto', 'cpu', 'cuda'), 'Device must be auto/cpu/cuda')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    old.evidence_not_failed(root, run / 'report.json')
    evidence = {}
    def checked(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None:
            require(isinstance(expected, str) and digest == expected, f'Checksum differs: {path}')
        require(str(path) not in evidence or evidence[str(path)] == digest,
                f'File changed during verification: {path}')
        evidence[str(path)] = digest
        return digest
    report_sha = checked(run / 'report.json')
    report = read_json(run / 'report.json')
    require(type(report.get('schema_version')) is int and report['schema_version'] == 1 and
            report.get('status') == 'complete' and report.get('protocol') == PROTOCOL and
            report.get('purpose') in ('smoke', 'full') and report.get('heldout_test_evaluated') is False,
            'Complete sealed-test schema1 BN report required')
    require(isinstance(report.get('config'), dict) and old.canonical(report['config']) == old.canonical(CONFIG),
            'Exact frozen calibration/evaluation configuration required')
    require('execution_registration' in report, 'Explicit execution_registration required')
    require(type(report.get('seed')) is int and report['seed'] in (0, 1, 2), 'Origin seed required')
    origin = relative_file(root, report['origin_run'])
    require(origin != run and not run.is_relative_to(origin) and not origin.is_relative_to(run),
            'Separate original/calibrated runs required')
    require(not output.is_relative_to(origin), 'Proof must be outside original run')
    origin_report_sha = checked(origin / 'report.json', report['origin_report_sha256'])
    origin_model_sha = checked(origin / 'model.pt', report['origin_model_sha256'])
    origin_report = read_json(origin / 'report.json')
    require(type(origin_report.get('schema_version')) is int and origin_report['schema_version'] == 1 and
            origin_report.get('status') == 'complete' and origin_report.get('heldout_test_evaluated') is False and
            origin_report.get('model_roundtrip_verified') is True, 'Complete sealed original run required')
    origin_config = old.validate_history(origin_report)
    require((report['purpose'] == 'smoke') ==
            (origin_config['purpose'] == 'paper_runner_smoke_validation_only'), 'Origin/run purpose differs')
    for key in ('selected_epoch', 'classes', 'normalization', 'cache_path', 'cache_metadata_sha256',
                'normalization_sha256', 'manifest_sha256', 'sample_counts', 'available_sample_counts'):
        require(report.get(key) == origin_report.get(key) and key in report and key in origin_report,
                f'Original report identity differs: {key}')
    require(report['seed'] == origin_config['seed'], 'Original seed differs')
    require(report.get('baseline_metrics') == origin_report['metrics'] and
            report.get('baseline_group_metrics') == origin_report['group_metrics'], 'Baseline references differ')
    require(set(report['metrics']) == set(report['group_metrics']) == set(report['sample_counts']) == set(SPLITS),
            'Only train/validation metric/count tables allowed')
    classes = report['classes']
    require(type(report['selected_epoch']) is int and classes == list(range(len(classes))) and
            all(type(c) is int for c in classes) and len(classes) >= 2, 'Integer epoch/classes required')
    if report['purpose'] == 'full':
        require(len(classes) == 20 and report['sample_counts'] == {'train': 24000, 'validation': 3200},
                'Full run requires all 24000/3200 samples and 20 classes')
    check_hash_table(report.get('files_sha256'), ARTIFACTS)
    check_hash_table(origin_report.get('files_sha256'), ARTIFACTS)
    for name in sorted(ARTIFACTS):
        checked(run / name, report['files_sha256'][name])
        checked(origin / name, origin_report['files_sha256'][name])
    verify_loaded_sources()
    check_hash_table(report.get('code_sha256'), SOURCES)
    check_hash_table(origin_report.get('code_sha256'), old.SOURCES)
    check_hash_table(origin_report.get('code_sha256_lf'), old.SOURCES)
    for label, digest in report['code_sha256'].items():
        path = old.EXECUTED_MODULE_PATHS.get(label, SOURCE_ROOT / label)
        checked(path, digest)
        if label in old.IMPORTED_MODULE_BYTES:
            require(hashlib.sha256(old.IMPORTED_MODULE_BYTES[label]).hexdigest() == digest,
                    f'Loaded inference source differs: {label}')
    for label, digest in origin_report['code_sha256'].items():
        require(report['code_sha256'][label] == digest, f'Frozen original source differs: {label}')
        require(hashlib.sha256((SOURCE_ROOT / label).read_bytes().replace(b'\r\n', b'\n')).hexdigest() ==
                origin_report['code_sha256_lf'][label], f'Canonical original source differs: {label}')
    checked(SOURCE_ROOT / 'config/paper_baseline_v1.json')
    script_sha = checked(_SCRIPT_PATH)
    require(hashlib.sha256(_SCRIPT_BYTES).hexdigest() == script_sha and
            hashlib.sha256(_OLD_BYTES).hexdigest() == report['code_sha256']['scripts/verify_paper_run.py'],
            'Loaded verifier source differs')
    cache = relative_file(root, report['cache_path'])
    checked(cache / 'metadata.json', report['cache_metadata_sha256'])
    checked(cache / 'normalization.json', report['normalization_sha256'])
    metadata, stats = read_json(cache / 'metadata.json'), read_json(cache / 'normalization.json')
    require(metadata.get('schema_version') == 1 and metadata.get('signal_shape') == [250, 2] and
            metadata.get('x_dtype') == 'float32' and metadata.get('y_dtype') == 'int64' and
            metadata.get('cache_units') == 'raw input units, not standardized', 'Invalid raw cache protocol')
    summary_path = relative_file(root, metadata['summary_path'])
    checked(summary_path, metadata['summary_sha256'])
    summary = read_json(summary_path)
    require(metadata['manifest_sha256'] == summary['manifest_sha256'] == report['manifest_sha256'] and
            metadata['inputs'] == summary['inputs'] and metadata['sample_counts'] == summary['sample_counts'] and
            metadata['class_counts'] == summary['class_counts'], 'Cache/manifest provenance differs')
    manifests = check_hash_table(metadata['manifest_sha256'], {'train.jsonl', 'validation.jsonl', 'test.jsonl'})
    for name, digest in manifests.items():
        checked(summary_path.parent / name, digest)
    require(origin_report['raw_input_sha256'] == {k: v['sha256'] for k, v in metadata['inputs'].items()},
            'Original raw input identities differ')
    for info in metadata['inputs'].values():
        checked(relative_file(root, info['path']), info['sha256'])
    cache_files = check_hash_table(metadata['files_sha256'],
        {f'{s}_{k}.npy' for s in (*SPLITS, 'test') for k in ('x', 'y')} | {'normalization.json'})
    for name, digest in cache_files.items():
        checked(cache / name, digest)
    require(stats == report['normalization'] and stats.get('fitted_split') == 'train' and stats.get('ddof') == 0
            and stats.get('train_cache_sha256') == cache_files['train_x.npy'] and
            stats.get('train_manifest_sha256') == manifests['train.jsonl'] and
            stats.get('count_per_channel') == summary['sample_counts']['train'] * 250,
            'Train-only normalization provenance differs')
    mean, std = np.asarray(stats['mean'], dtype=np.float64), np.asarray(stats['std'], dtype=np.float64)
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all() and np.isfinite(std).all() and
            (std > 0).all(), 'Finite positive normalization required')
    anchor_paths = [origin / 'report.json', *(origin / name for name in ARTIFACTS),
                    cache / 'metadata.json', *(cache / name for name in cache_files),
                    summary_path, *(summary_path.parent / name for name in manifests),
                    *(relative_file(root, info['path']) for info in metadata['inputs'].values())]
    anchors = {str(path): evidence[str(path)] for path in anchor_paths}
    old.verify_execution_registration(root, origin, origin_report, cache, metadata, checked)
    registration = check_registration(root, run, report, origin, origin_report, checked, anchors)
    selected, records, csv_predictions, baseline_predictions = {}, {}, {}, {}
    for split in SPLITS:
        all_records = [json.loads(line) for line in
                       (summary_path.parent / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(all_records) == summary['sample_counts'][split] == report['available_sample_counts'][split],
                f'Manifest available counts differ: {split}')
        require(all(type(r.get('class_index')) is int and r['class_index'] in classes and r.get('split') == split
                    and isinstance(r.get('wave_sha256'), str) and len(r['wave_sha256']) == 64 and
                    all(c in '0123456789abcdef' for c in r['wave_sha256']) and
                    type(r.get('group_id')) in (str, int) for r in all_records),
                f'Manifest labels/identities differ: {split}')
        labels = np.asarray([r['class_index'] for r in all_records], dtype=np.int64)
        require(np.array_equal(np.unique(labels), np.arange(len(classes))), f'Manifest classes differ: {split}')
        rows, old_rows = read_json(run / f'{split}_rows.json'), read_json(origin / f'{split}_rows.json')
        require(isinstance(rows, list) and rows and all(type(i) is int for i in rows) and
                rows == old_rows == sorted(set(rows)) and rows[0] >= 0 and rows[-1] < len(labels) and
                len(rows) == report['sample_counts'][split], f'Origin row order/selection differs: {split}')
        cap = origin_config[f'{split}_per_class']
        expected = list(range(len(labels)))
        if cap is not None:
            rng = np.random.default_rng(20261002)
            expected = sorted(i for cls in np.unique(labels) for i in
                rng.choice(np.flatnonzero(labels == cls), size=min(cap, int((labels == cls).sum())),
                           replace=False).tolist())
        require(rows == expected and (report['purpose'] != 'full' or rows == list(range(len(labels)))),
                f'Original selection protocol differs: {split}')
        selected[split], records[split] = rows, [all_records[i] for i in rows]
        require(np.array_equal(np.unique(labels[rows]), np.arange(len(classes))), 'Selected classes differ')
        csv_predictions[split] = read_prediction_csv(run / f'{split}_predictions.csv', rows, records[split], len(classes))
        baseline_predictions[split] = read_prediction_csv(origin / f'{split}_predictions.csv', rows, records[split], len(classes))
    # Configure deterministic math before either checkpoint is deserialized.
    old.seed_everything(report['seed'], threads=1)
    inference_device = old.resolve_device(device)
    def portable(storage, location):
        require(location == 'cpu', 'Portable CPU checkpoint storage required')
        return storage
    saved = torch.load(run / 'model.pt', weights_only=True, map_location=portable)
    origin_saved = torch.load(origin / 'model.pt', weights_only=True, map_location=portable)
    require(isinstance(saved, dict) and set(saved) == {'schema_version', 'protocol', 'normalization',
        'selected_epoch', 'classes', 'state_dict', 'origin_model_sha256', 'config'} and
        type(saved['schema_version']) is int and saved['schema_version'] == 1 and
        saved['protocol'] == PROTOCOL and old.canonical(saved['config']) == old.canonical(report['config']) and
        saved['origin_model_sha256'] == origin_model_sha, 'Calibrated checkpoint contract differs')
    require(type(origin_saved.get('schema_version')) is int and origin_saved['schema_version'] == 1 and
            origin_saved.get('protocol') == origin_report['protocol'] and
            old.canonical(origin_saved.get('config')) == old.canonical(origin_config),
            'Original checkpoint metadata differs')
    for checkpoint in (saved, origin_saved):
        require(all(old.canonical(checkpoint.get(k)) == old.canonical(report[k])
                    for k in ('normalization', 'selected_epoch', 'classes')),
                'Checkpoint epoch/classes/normalization differs')
        require(isinstance(checkpoint.get('state_dict'), dict) and checkpoint['state_dict'] and
                all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and torch.isfinite(t).all().item()
                    for t in checkpoint['state_dict'].values()), 'Finite portable state required')
    original = old.PaperResNeXt1D(num_classes=len(classes), blocks_per_stage=origin_config['blocks_per_stage'])
    original.load_state_dict(origin_saved['state_dict'], strict=True)
    original.to(inference_device).eval()
    calibrated = copy.deepcopy(original)
    calibrated.load_state_dict(saved['state_dict'], strict=True)
    calibrated.eval()
    require(sum(p.numel() for p in original.parameters()) == origin_report['parameter_count'],
            'Original parameter count differs')
    initial_state = {k: t.detach().cpu().clone() for k, t in original.state_dict().items()}
    environment = report.get('environment')
    require(isinstance(environment, dict) and environment.get('device') in ('cpu', 'cuda') and
            type(environment.get('threads')) is int and environment['threads'] == 1 and
            environment.get('deterministic_algorithms') is True and environment.get('tf32') is False,
            'Deterministic CPU/CUDA producer environment required')
    same_device = environment['device'] == inference_device.type
    metrics, group_metrics, group_counts, prediction_counts = {}, {}, {}, {}
    maximum_metric_error, comparison = 0., None
    for split in SPLITS:
        mapped = []
        try:
            raw = np.load(cache / f'{split}_x.npy', mmap_mode='r', allow_pickle=False)
            mapped.append(raw)
            labels = np.load(cache / f'{split}_y.npy', mmap_mode='r', allow_pickle=False)
            mapped.append(labels)
            n = summary['sample_counts'][split]
            require(raw.shape == (n, 250, 2) and raw.dtype == np.float32 and
                    labels.shape == (n,) and labels.dtype == np.int64, 'Cache array shape/dtype differs')
            manifest_labels = [json.loads(line)['class_index'] for line in
                (summary_path.parent / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
            require(np.array_equal(labels, manifest_labels), 'Cache labels differ from manifest order')
            rows = selected[split]
            values, true = raw[rows], labels[rows]
            identities = [record['wave_sha256'] for record in records[split]]
            if split == 'train':
                replayed, counts = independent_calibration(original, values, identities, mean, std, inference_device)
                comparison = compare_bn_states(original, calibrated, replayed, same_device)
                calibration = report.get('calibration')
                require(isinstance(calibration, dict) and set(calibration) == set(counts) |
                        {'changed_buffers', 'parameters_unchanged', 'original_state_unchanged'} and
                        all(type(calibration.get(k)) is int and calibration[k] == v for k, v in counts.items()) and
                        calibration['changed_buffers'] == comparison['changed_buffers'] and
                        calibration['parameters_unchanged'] is True and calibration['original_state_unchanged'] is True,
                        'Calibration counts/buffer change report differs')
            probabilities = independent_probabilities(calibrated, values, identities, mean, std, inference_device, 128)
            predicted = probabilities.argmax(axis=1)
            mismatches = int(np.count_nonzero(predicted != csv_predictions[split]))
            require(mismatches == 0, f'Independent prediction classes differ: {split}; mismatches={mismatches}')
            metrics[split], group_metrics[split], group_counts[split], error = checked_metrics(
                true, predicted, records[split], len(classes), report['metrics'][split], report['group_metrics'][split], split)
            maximum_metric_error = max(maximum_metric_error, error)
            _, _, _, error = checked_metrics(true, baseline_predictions[split], records[split], len(classes),
                origin_report['metrics'][split], origin_report['group_metrics'][split], f'baseline/{split}')
            maximum_metric_error = max(maximum_metric_error, error)
            prediction_counts[split] = len(predicted)
        finally:
            for array in mapped:
                array._mmap.close()
    require(all(torch.equal(t.detach().cpu(), initial_state[k]) for k, t in original.state_dict().items()),
            'Original in-memory model changed')
    verify_loaded_sources()
    for path, digest in evidence.items():
        require(sha256(path) == digest, f'File changed during verification: {path}')
        if Path(path).is_relative_to(root):
            old.evidence_not_failed(root, Path(path))
    proof = dict(schema_version=1, status='verified', protocol=PROTOCOL,
        created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
        device=str(inference_device), device_request=device, run_path=str(run), origin_run=report['origin_run'],
        report_sha256=report_sha, origin_report_sha256=origin_report_sha, origin_model_sha256=origin_model_sha,
        prediction_counts=prediction_counts, group_counts=group_counts, metrics=metrics, group_metrics=group_metrics,
        maximum_metric_error=maximum_metric_error, metric_tolerance=old.METRIC_ATOL,
        calibration_buffer_max_abs_error=comparison['maximum_error'],
        same_device_calibration_exact=(same_device and comparison['exact']),
        calibration_buffer_tolerance=dict(rtol=0. if same_device else BUFFER_RTOL,
                                         atol=0. if same_device else BUFFER_ATOL),
        calibration=report['calibration'], config=report['config'], execution_registration=registration,
        checked_files_sha256=evidence, verification_script=dict(path=str(_SCRIPT_PATH),
            text=_SCRIPT_BYTES.decode('utf-8'), sha256=script_sha),
        checks=dict(independent_calibration=True, bn_buffer_only_changes=True, parameters_unchanged=True,
            original_state_unchanged=True, calibration_train_only=True, calibration_crop_order_verified=True,
            checkpoint_inference_independent=True, both_splits_inferred=True, prediction_identities_verified=True,
            prediction_class_mismatches=0, metrics_verified=True, group_class_metrics_verified=True,
            source_snapshot_verified=True, registry_verified=(registration is not None),
            test_classification_enabled=False, test_files_hashed_only=True))
    payload = json.dumps(proof, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    committed = False
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=output.parent,
                prefix='.paper-bn-proof-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        old.evidence_not_failed(root, run / 'report.json')
        old.evidence_not_failed(root, origin / 'report.json')
        verify_loaded_sources()
        os.link(temporary, output)
        committed = True
    finally:
        failing = sys.exc_info()[0] is not None
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                if not committed and not failing:
                    raise
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    try:
        proof = verify_paper_bn(args.project_root, args.run, args.output, args.device)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, FloatingPointError) as error:
        parser.exit(1, f'BN verification failed: {type(error).__name__}: {error}\n')
    print(json.dumps({k: proof[k] for k in ('status', 'device', 'prediction_counts',
        'maximum_metric_error', 'calibration_buffer_max_abs_error')}, allow_nan=False))


if __name__ == '__main__':
    main()
