"""Independently replay E2 full250 DC/AC train/validation checkpoints.

Producer normalization, training, loaders, crop and metrics are never imported.
verify_run returns read-only evidence; the CLI atomically publishes a new proof.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
_OLD_PATH = SOURCE_ROOT / 'scripts/verify_paper_run.py'
_OLD_BYTES = _OLD_PATH.read_bytes()
_spec = importlib.util.spec_from_file_location('_dc_ac_old_verifier_helpers', _OLD_PATH)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)

from src.ect.devices import resolve_device, seed_everything
from src.ect.paper_models import PaperResNeXt1D

require, read_json, sha256 = old.require, old.read_json, old.sha256
inside, canonical = old.inside, old.canonical
SPLITS, CSV_FIELDS = old.SPLITS, old.CSV_FIELDS
PROTOCOL = 'paper_dc_ac_finite_v1'
NORMALIZATION_PROTOCOL = 'dc_ac_full250_v1'
FULL_PURPOSE = 'exploratory_train_validation_dc_ac'
SMOKE_PURPOSE = 'paper_dc_ac_smoke_validation_only'
ARTIFACTS = old.ARTIFACTS | {'normalization_dc_ac.json'}
SOURCES = old.REGISTRY_SOURCES | {'src/ect/dc_ac.py', 'src/ect/paper_dc_ac_training.py',
    'src/ect/paper_dc_ac_registration.py', 'src/ect/paper_dc_ac_bn.py',
    'scripts/train_paper_dc_ac.py', 'scripts/register_paper_dc_ac.py',
    'scripts/verify_paper_dc_ac.py', 'scripts/recalibrate_paper_dc_ac_bn.py',
    'scripts/run_paper_dc_ac_experiment.py'}
STATS_RTOL, STATS_ATOL = 1e-10, 1e-12
BUFFER_RTOL, BUFFER_ATOL = 2e-4, 2e-5
BN_PROTOCOL = 'paper_dc_ac_bn_v1'
_SCRIPT_PATH = Path(__file__).resolve()
_SCRIPT_BYTES = _SCRIPT_PATH.read_bytes()
STATS_FIELDS = {'schema_version', 'protocol', 'fitted_split', 'sample_count',
    'points_per_channel', 'epsilon', 'mu_d', 'sigma_d', 'sigma_a', 'scale_d',
    'scale_a', 'floor_d', 'floor_a', 'channels', 'signal_length'}
PROOF_CHECKS = {'independent_statistics', 'full_clean_train_fit_verified', 'full250_transform_before_crop',
    'initial_parameters_verified', 'independent_inference', 'both_splits_inferred', 'all_row_csv_identities_verified',
    'prediction_classes_exact', 'metrics_verified', 'group_class_metrics_verified', 'source_verified',
    'complete_artifacts_verified', 'cache_manifest_provenance_verified', 'checkpoint_schema_verified',
    'history_selection_verified', 'original_run_read_only'}
BN_CHECKS = {'independent_calibration', 'calibration_train_only', 'calibration_crop_order_verified',
    'bn_buffer_only_changes', 'parameters_unchanged', 'original_state_unchanged', 'calibration_buffers_verified',
    'origin_run_verified'}
ACCEPTANCE_CHECKS = {'cpu_training', 'cuda_training', 'independent_inference', 'cross_device_reload',
                     'full_regression', 'independent_code_review', 'fixed_bn_ablation'}


def validate_statistics(stats):
    require(isinstance(stats, dict) and set(stats) == STATS_FIELDS, 'DC/AC statistics schema differs')
    require(type(stats['schema_version']) is int and stats['schema_version'] == 1 and
            stats['protocol'] == NORMALIZATION_PROTOCOL and stats['fitted_split'] == 'train',
            'Train-only schema-1 DC/AC statistics required')
    for field, value in [('channels', 2), ('signal_length', 250)]:
        require(type(stats[field]) is int and stats[field] == value, f'Invalid statistics {field}')
    require(type(stats['sample_count']) is int and stats['sample_count'] > 0 and
            type(stats['points_per_channel']) is int and
            stats['points_per_channel'] == stats['sample_count'] * 250,
            'Complete train statistics count required')
    require(type(stats['epsilon']) in (int, float) and stats['epsilon'] == 1e-12,
            'Fixed 1e-12 scale floor required')
    for key in ('mu_d', 'sigma_d', 'sigma_a', 'scale_d', 'scale_a'):
        value = stats[key]
        require(isinstance(value, list) and len(value) == 2 and
                all(type(v) in (int, float) and np.isfinite(v) for v in value),
                f'Finite two-channel statistics required: {key}')
        if key != 'mu_d':
            require(all(v >= 0 for v in value), f'Negative scale: {key}')
    for suffix in ('d', 'a'):
        flags = stats[f'floor_{suffix}']
        require(isinstance(flags, list) and len(flags) == 2 and all(type(v) is bool for v in flags),
                'Explicit boolean scale floor flags required')
        sigma = stats[f'sigma_{suffix}']
        require(stats[f'scale_{suffix}'] == [max(v, 1e-12) for v in sigma] and
                flags == [v < 1e-12 for v in sigma], 'Scale/floor coefficients disagree')
    return stats


def fit_statistics(raw):
    """Two independent float64 passes over ALL clean train; bounded memory."""
    require(raw.ndim == 3 and raw.shape[1:] == (250, 2) and len(raw) > 0,
            'Nonempty entire train full250 two-channel input required')
    count = len(raw)
    dc_sum, ac_sum = np.zeros(2), np.zeros(2)
    for start in range(0, count, 256):
        values = np.asarray(raw[start:start+256], dtype=np.float64)
        require(np.isfinite(values).all(), 'Nonfinite clean train signal')
        dc = values.mean(axis=1)
        dc_sum += dc.sum(axis=0, dtype=np.float64)
        ac_sum += np.square(values - dc[:, None, :]).sum(axis=(0, 1), dtype=np.float64)
    mu = dc_sum / count
    dc_squared = np.zeros(2)
    for start in range(0, count, 256):
        dc = np.asarray(raw[start:start+256], dtype=np.float64).mean(axis=1)
        dc_squared += np.square(dc - mu).sum(axis=0, dtype=np.float64)
    sigma_d, sigma_a = np.sqrt(dc_squared / count), np.sqrt(ac_sum / (count * 250))
    stats = dict(schema_version=1, protocol=NORMALIZATION_PROTOCOL, fitted_split='train',
        sample_count=count, points_per_channel=count*250, epsilon=1e-12, mu_d=mu.tolist(),
        sigma_d=sigma_d.tolist(), sigma_a=sigma_a.tolist(),
        scale_d=np.maximum(sigma_d, 1e-12).tolist(), scale_a=np.maximum(sigma_a, 1e-12).tolist(),
        floor_d=(sigma_d < 1e-12).tolist(), floor_a=(sigma_a < 1e-12).tolist(),
        channels=2, signal_length=250)
    return validate_statistics(stats)


def transform_full250(raw, stats):
    validate_statistics(stats)
    values = np.asarray(raw, dtype=np.float64)
    require(values.ndim == 3 and values.shape[1:] == (250, 2) and len(values) > 0 and
            np.isfinite(values).all(), 'Finite full250 signals required before crop')
    dc = values.mean(axis=1, keepdims=True)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        transformed = (((dc - np.asarray(stats['mu_d'])) / np.asarray(stats['scale_d']) +
                        (values - dc) / np.asarray(stats['scale_a'])) / np.sqrt(2.)).astype(np.float32)
    require(np.isfinite(transformed).all(), 'Nonfinite DC/AC transform')
    return transformed


def independent_probabilities(model, raw, identities, stats, device, batch_size):
    """Manual SHA256/PCG64 crops of transformed full250; mean ten softmaxes."""
    require(type(batch_size) is int and batch_size > 0, 'Positive batch size required')
    require(raw.shape == (len(identities), 250, 2) and len(raw) > 0, 'Aligned raw signals required')
    result = []
    with torch.inference_mode():
        for begin in range(0, len(raw), batch_size):
            values = transform_full250(raw[begin:begin+batch_size], stats)
            crops = np.empty((len(values)*10, 224, 2), dtype=np.float32)
            for index, identity in enumerate(identities[begin:begin+batch_size]):
                require(isinstance(identity, str) and len(identity) == 64 and
                        all(c in '0123456789abcdef' for c in identity), 'Invalid waveform identity')
                key = json.dumps(['validation', 10, identity, None], ensure_ascii=True,
                                 separators=(',', ':'), allow_nan=False).encode('ascii')
                generator = np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest(), 'big')))
                for crop, start in enumerate(generator.integers(0, 27, size=10, dtype=np.int64)):
                    crops[index*10+crop] = values[index, start:start+224]
            probabilities = []
            for offset in range(0, len(crops), batch_size):
                inputs = torch.from_numpy(np.ascontiguousarray(crops[offset:offset+batch_size].transpose(0, 2, 1))).to(device)
                logits = model(inputs)
                require(logits.ndim == 2 and len(logits) == len(inputs) and torch.isfinite(logits).all().item(),
                        'Invalid independent logits')
                probabilities.append(torch.softmax(logits, dim=1).float().cpu().numpy())
            averaged = np.concatenate(probabilities).reshape(len(values), 10, -1).mean(axis=1)
            require(np.isfinite(averaged).all(), 'Nonfinite independent probabilities')
            result.append(averaged)
    return np.concatenate(result)


def validate_history(report):
    """Check the frozen method contract and independently select the epoch."""
    config = report['config']
    template = read_json(SOURCE_ROOT / 'config/paper_baseline_v1.json')
    template.update(protocol=PROTOCOL, normalization_protocol=NORMALIZATION_PROTOCOL)
    require(isinstance(config, dict) and set(config) == set(template) | {'seed'},
            'Complete configuration required')
    adjustable = {'purpose', 'device', 'threads', 'batch_size', 'max_epochs',
                  'validation_every_epochs', 'train_per_class', 'validation_per_class',
                  'cache_dir', 'notes', 'seed'}
    for key in set(template) - adjustable:
        require(config[key] == template[key] and type(config[key]) is type(template[key]),
                f'Frozen configuration differs: {key}')
    require(config['purpose'] in (FULL_PURPOSE, SMOKE_PURPOSE), 'Unknown run purpose')
    require(config['device'] in ('auto', 'cpu', 'cuda'), 'Invalid recorded device')
    require(type(config['seed']) is int and config['seed'] in (0, 1, 2), 'Invalid seed')
    for key in ('threads', 'batch_size', 'max_epochs', 'validation_every_epochs'):
        require(type(config[key]) is int and config[key] > 0, f'Invalid {key}')
    for key in ('train_per_class', 'validation_per_class'):
        require(config[key] is None or (type(config[key]) is int and config[key] > 0),
                f'Invalid {key}')
    require(isinstance(config['cache_dir'], str) and config['cache_dir'] and
            isinstance(config['notes'], str), 'Cache path and notes required')
    if config['purpose'] == FULL_PURPOSE:
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
    require(report.get('selection_rule') == 'validation accuracy then earlier one-based epoch' and
            report.get('crop_epoch_convention') == 'training RNG epoch = one-based recorded epoch minus one' and
            report.get('diagnostic_crop_protocol') ==
                'train and validation use the same seed-10 validation namespace, no optimizer update' and
            report.get('input_transform_order') == ['full250_dc_ac', 'float32', 'identity_crop'],
            'Input/crop/selection protocol metadata differs')
    require(canonical(report.get('optimizer')) == canonical(dict(name='Adam', betas=config['betas'],
            eps=config['optimizer_eps'], weight_decay=config['weight_decay'])), 'Optimizer metadata differs')
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


def initial_parameters_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(json.dumps(dict(name=name, shape=list(array.shape), dtype=str(array.dtype)),
                                 sort_keys=True, separators=(',', ':')).encode())
        digest.update(b'\n')
        digest.update(array.tobytes())
        digest.update(b'\n')
    return digest.hexdigest()


def hash_table(value, fields=None):
    require(isinstance(value, dict) and value and (fields is None or set(value) == set(fields)),
            'Complete checksum table required')
    require(all(isinstance(k, str) and isinstance(v, str) and len(v) == 64 and
                all(c in '0123456789abcdef' for c in v) for k, v in value.items()),
            'Invalid SHA256 checksum table')
    return value


def relative_file(root, label):
    require(isinstance(label, str) and bool(label), 'Root-relative path required')
    path = inside(root, label)
    require(not Path(label).is_absolute() and path != root and
            path.relative_to(root).as_posix() == label, 'Canonical project-relative path required')
    old.evidence_not_failed(root, path)
    return path


def verify_sources(report, checked):
    hash_table(report.get('code_sha256'), SOURCES)
    hash_table(report.get('code_sha256_lf'), SOURCES)
    for label in sorted(SOURCES):
        path = old.EXECUTED_MODULE_PATHS.get(label, SOURCE_ROOT / label)
        checked(path, report['code_sha256'][label])
        require(hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() ==
                report['code_sha256_lf'][label], f'Canonical source checksum differs: {label}')
        if label in old.IMPORTED_MODULE_BYTES:
            require(hashlib.sha256(old.IMPORTED_MODULE_BYTES[label]).hexdigest() == report['code_sha256'][label],
                    f'Imported model/device source differs: {label}')
    require(_OLD_BYTES == _OLD_PATH.read_bytes() and _SCRIPT_BYTES == _SCRIPT_PATH.read_bytes(),
            'Imported verifier source changed since import')
    require(seed_everything is old.seed_everything and resolve_device is old.resolve_device and
            PaperResNeXt1D is old.PaperResNeXt1D, 'Inference aliases differ from inspected runtime')
    old.verify_loaded_inference_sources()


def verify_registration(root, run, report, cache, metadata, checked):
    """Check the new frozen registry JSON and every bound file independently."""
    config, recorded = report['config'], report.get('execution_registration')
    require('execution_registration' in report, 'Explicit execution registration required')
    if config['purpose'] == SMOKE_PURPOSE:
        require(recorded is None, 'Smoke registration must be null')
        return None
    fields = {'registry_path', 'registry_sha256', 'protocol', 'registered_utc', 'run_id', 'seed',
              'config_path', 'config_file_sha256', 'resolved_config_sha256', 'output_dir',
              'acceptance_sha256', 'statistics_path', 'statistics_sha256', 'initial_parameters_sha256'}
    require(isinstance(recorded, dict) and set(recorded) == fields, 'Complete E2 execution registration required')

    def object_file(label, checksum):
        path = relative_file(root, label)
        checked(path, checksum)
        obj = read_json(path)
        require(isinstance(obj, dict), f'JSON object required: {label}')
        return obj

    registry_path = relative_file(root, recorded['registry_path'])
    registry = object_file(recorded['registry_path'], recorded['registry_sha256'])
    directory = registry_path.parent.relative_to(root).as_posix()
    require(registry_path.name == 'registry.json' and type(registry.get('schema_version')) is int and
            registry['schema_version'] == 1 and registry.get('protocol') == PROTOCOL and
            registry.get('status') == 'frozen_execution_registration' and
            registry.get('registration_dir') == directory and
            registry.get('test_classification_enabled') is False and
            isinstance(registry.get('registered_utc'), str) and registry['registered_utc'],
            'Frozen E2 registry required')
    require(registry.get('code_sha256') == report['code_sha256'] and
            registry.get('code_sha256_lf') == report['code_sha256_lf'], 'Registry/run sources differ')
    require(registry.get('base_config_path') == 'config/paper_dc_ac_v1.json' and
            registry.get('design_path') == 'docs/plans/2026-10-03-dc-ac-training.md',
            'E2 base/design paths differ')
    base = object_file(registry['base_config_path'], registry['base_config_sha256'])
    require(canonical(config) == canonical(base | {'seed': config['seed']}),
            'Full config differs from frozen E2 base plus seed')
    checked(relative_file(root, registry['design_path']), registry['design_sha256'])
    acceptance = object_file(registry['acceptance_path'], registry['acceptance_sha256'])
    require(type(acceptance.get('schema_version')) is int and acceptance['schema_version'] == 1 and
            acceptance.get('status') == 'accepted' and acceptance.get('protocol') == PROTOCOL and
            acceptance.get('test_classification_enabled') is False and
            acceptance.get('code_sha256') == report['code_sha256'], 'Bound verified E2 acceptance required')
    require(acceptance.get('checks') == {k:True for k in ACCEPTANCE_CHECKS}, 'All seven acceptance checks required')
    acceptance_files = hash_table(registry.get('acceptance_files_sha256'))
    require(acceptance_files == acceptance.get('files_sha256'), 'Acceptance file closure differs')
    for label, checksum in acceptance_files.items():
        checked(relative_file(root, label), checksum)
    check_acceptance_evidence(root, acceptance, report['code_sha256'], acceptance_files, checked)
    require(registry.get('cache_dir') == report['cache_path'] == config['cache_dir'] and
            relative_file(root, registry['cache_dir']) == cache and
            registry.get('cache_metadata_sha256') == report['cache_metadata_sha256'] and
            registry.get('manifest_sha256') == metadata['manifest_sha256'] and
            registry.get('raw_input_sha256') == report['raw_input_sha256'], 'Registered E2 data differs')
    registered_stats = object_file(registry['statistics_path'], registry['statistics_sha256'])
    require(canonical(registered_stats) == canonical(read_json(run/'normalization_dc_ac.json')),
            'Registered normalization artifact differs')
    require(registry['statistics_sha256'] == report['normalization_sha256'], 'Registered normalization hash differs')
    baseline = object_file(registry['baseline_registry_path'], registry['baseline_registry_sha256'])
    require(registry['baseline_registry_path'] == 'results/experiments/paper_baseline_v1/registry.json' and
            registry['baseline_registry_sha256'] == '791e85504ff55f91d829ed94464e81ae53789186a0879aa27c188c7b14cc77c4' and
            baseline.get('status') == 'frozen_execution_registration' and
            baseline.get('protocol') == 'paper_baseline_finite_v1' and
            baseline.get('test_classification_enabled') is False and
            set(baseline.get('code_sha256', {})) == old.REGISTRY_SOURCES,
            'Frozen original baseline registry required')
    for label, checksum in hash_table(baseline['code_sha256'], old.REGISTRY_SOURCES).items():
        require(report['code_sha256'][label] == checksum, f'Original frozen source differs: {label}')
        checked(SOURCE_ROOT/label, checksum)
    for label, checksum in hash_table(baseline['files_sha256']).items():
        checked(relative_file(root, label), checksum)
    expected_bn = dict(calibration_batch_size=128, calibration_seed=0, calibration_epoch=0,
        order='manifest', passes=1, evaluation_batch_size=128, evaluation_seed=10,
        threads=1, deterministic=True, tf32=False, test_classification_enabled=False)
    require(canonical(registry.get('bn_calibration')) == canonical(expected_bn), 'Frozen E1 BN rule differs')
    runs = registry.get('runs')
    require(isinstance(runs, list) and len(runs) == 3, 'All three seed registrations required')
    run_fields = {'run_id', 'seed', 'config_path', 'config_file_sha256', 'resolved_config_sha256',
                  'output_dir', 'initial_parameters_sha256'}
    for seed, item in enumerate(runs):
        require(isinstance(item, dict) and set(item) == run_fields and type(item['seed']) is int and
                item['seed'] == seed and item['run_id'] == f'resnext_s{seed}' and
                item['config_path'] == f'{directory}/configs/resnext_s{seed}.json' and
                item['output_dir'] == f'{directory}/resnext_s{seed}', 'Registered seed/output identity differs')
        resolved = object_file(item['config_path'], item['config_file_sha256'])
        require(canonical(resolved) == canonical(base | {'seed': seed}) and
                hashlib.sha256(canonical(resolved).encode()).hexdigest() == item['resolved_config_sha256'],
                'Frozen seed config checksum differs')
        seed_everything(seed, threads=config['threads'])
        initial = PaperResNeXt1D(num_classes=len(report['classes']), blocks_per_stage=config['blocks_per_stage'])
        require(initial_parameters_sha256(initial) == item['initial_parameters_sha256'],
                'Registered seed initialization differs')
    selected = runs[config['seed']]
    require(relative_file(root, selected['output_dir']) == run, 'Run is outside registered output')
    expected = dict(registry_path=recorded['registry_path'], registry_sha256=recorded['registry_sha256'],
        protocol=PROTOCOL, registered_utc=registry['registered_utc'], acceptance_sha256=registry['acceptance_sha256'],
        statistics_path=registry['statistics_path'], statistics_sha256=registry['statistics_sha256'], **selected)
    require(canonical(recorded) == canonical(expected) and
            report['initial_parameters_sha256'] == selected['initial_parameters_sha256'],
            'Report execution identity differs')
    return recorded


def check_acceptance_proof(root, proof, report, run, device, files, checked, bn=False):
    """Read-only acceptance semantics and complete proof-to-file identity chain."""
    flags = proof.get('checks')
    required = PROOF_CHECKS | (BN_CHECKS if bn else set())
    require(isinstance(flags, dict) and required <= set(flags) and
            all(flags[k] is True for k in required) and flags.get('test_classification_evaluated') is False,
            'Complete independent acceptance proof checks required')
    require(type(proof.get('schema_version')) is int and proof['schema_version'] == 1 and
            proof.get('status') == 'verified' and proof.get('protocol') == report['protocol'] and
            proof.get('run_dir') == run.relative_to(root).as_posix() and proof.get('device') == device and
            proof.get('heldout_test_evaluated') is False, 'Acceptance proof identity differs')
    pairs = {'report_sha256':sha256(run/'report.json'), 'weights_sha256':report['files_sha256']['model.pt'],
        'artifacts_sha256':report['files_sha256'], 'source_sha256':report['code_sha256'],
        'cache_metadata_sha256':report['cache_metadata_sha256'], 'normalization_sha256':report['normalization_sha256'],
        'prediction_counts':report['sample_counts'], 'config':report['config'], 'selected_epoch':report['selected_epoch'],
        'initial_parameters_sha256':report['initial_parameters_sha256'], 'execution_registration':report['execution_registration']}
    for name, expected in pairs.items():
        require(canonical(proof.get(name)) == canonical(expected), f'Acceptance proof differs: {name}')
    for split in SPLITS:
        old.metric_error(report['metrics'][split], proof.get('metrics', {}).get(split), f'acceptance/{split}')
        grouped = proof.get('group_metrics', {}).get(split)
        require(isinstance(grouped, dict) and set(grouped) == set(report['group_metrics'][split]),
                'Acceptance group metric keys differ')
        for group, metric in report['group_metrics'][split].items():
            old.metric_error(metric, grouped[group], f'acceptance/{split}/{group}')
    require(proof.get('metric_tolerance') == 1e-12 and
            proof.get('statistics_tolerance') == {'rtol':STATS_RTOL, 'atol':STATS_ATOL} and
            type(proof.get('maximum_metric_error')) in (int, float) and
            np.isfinite(proof['maximum_metric_error']) and 0 <= proof['maximum_metric_error'] <= 1e-12 and
            type(proof.get('maximum_statistics_error')) in (int, float) and
            np.isfinite(proof['maximum_statistics_error']) and proof['maximum_statistics_error'] >= 0,
            'Acceptance proof tolerances/error invalid')
    script = proof.get('script', {})
    require(script.get('sha256') == hashlib.sha256(_SCRIPT_BYTES).hexdigest() and
            script.get('text') == _SCRIPT_BYTES.decode('utf-8'), 'Acceptance verifier source differs')
    same = report['environment']['device'] == device
    require(proof.get('same_device') is same and proof.get('allow_device_change') is (not same),
            'Acceptance inference device differs')
    if bn:
        for name in ('calibration', 'origin_run', 'origin_report_sha256', 'origin_model_sha256'):
            require(canonical(proof.get(name)) == canonical(report[name]), f'Acceptance BN identity differs: {name}')
        require(proof.get('calibration_buffer_tolerance') ==
                dict(rtol=0. if same else BUFFER_RTOL, atol=0. if same else BUFFER_ATOL) and
                type(proof.get('calibration_buffer_max_abs_error')) in (int, float) and
                np.isfinite(proof['calibration_buffer_max_abs_error']) and proof['calibration_buffer_max_abs_error'] >= 0 and
                (not same or (proof.get('same_device_calibration_exact') is True and
                              proof['calibration_buffer_max_abs_error'] == 0)), 'Acceptance BN replay differs')
    chain = hash_table(proof.get('checked_files_sha256'))
    cache = relative_file(root, report['cache_path'])
    metadata = read_json(cache/'metadata.json')
    summary = relative_file(root, metadata['summary_path'])
    required_paths = {run/'report.json', cache/'metadata.json', summary, SOURCE_ROOT/'config/paper_baseline_v1.json'}
    required_paths.update(run/n for n in ARTIFACTS)
    required_paths.update(SOURCE_ROOT/n for n in SOURCES)
    required_paths.update(cache/n for n in metadata['files_sha256'])
    required_paths.update(summary.parent/n for n in metadata['manifest_sha256'])
    required_paths.update(relative_file(root, info['path']) for info in metadata['inputs'].values())
    if bn:
        origin = relative_file(root, report['origin_run'])
        required_paths.update({origin/'report.json', *(origin/n for n in ARTIFACTS)})
    require({str(p) for p in required_paths} <= set(chain), 'Acceptance proof identity chain is pruned')
    permitted_sources = {SOURCE_ROOT/n for n in SOURCES} | {SOURCE_ROOT/'config/paper_baseline_v1.json'}
    for label, checksum in chain.items():
        path = Path(label).resolve()
        require(path.is_relative_to(root) or path in permitted_sources, 'Acceptance evidence outside authorized roots')
        if path.is_relative_to(root):
            old.evidence_not_failed(root, path)
        base = root if path.is_relative_to(root) else SOURCE_ROOT
        require(files.get(path.relative_to(base).as_posix()) == checksum, 'Acceptance files omit proof evidence')
        checked(path, checksum)


def check_acceptance_evidence(root, acceptance, sources, files, checked):
    reports = {}
    for key, bn in (('runs', False), ('bn_runs', True)):
        runs = acceptance.get(key)
        require(isinstance(runs, list) and len(runs) == 2 and {r.get('device') for r in runs} == {'cpu', 'cuda'},
                'Actual CPU/CUDA training and BN acceptance required')
        for item in runs:
            run = relative_file(root, item['run_dir'])
            path = relative_file(root, item['verification_path'])
            report, proof = read_json(run/'report.json'), read_json(path)
            require(report.get('status') == 'complete' and report.get('heldout_test_evaluated') is False and
                    report.get('code_sha256') == sources and report.get('environment', {}).get('device') == item['device'] and
                    report.get('classes') == list(range(20)) and report.get('sample_counts') == {'train':40, 'validation':40},
                    'Actual complete matched CPU/CUDA smoke reports required')
            cfg = report['config']
            require(cfg.get('purpose') == SMOKE_PURPOSE and cfg.get('seed') == 0 and cfg.get('max_epochs') == 2 and
                    cfg.get('train_per_class') == cfg.get('validation_per_class') == 2 and
                    type(report.get('parameter_update_l2')) in (int, float) and report['parameter_update_l2'] > 0,
                    'Acceptance smoke budget/learning differs')
            for evidence in (run/'report.json', path, *(run/n for n in ARTIFACTS)):
                require(evidence.relative_to(root).as_posix() in files, 'Acceptance smoke file chain omitted')
                checked(evidence, files[evidence.relative_to(root).as_posix()])
            if bn:
                require(report.get('origin_run') == reports[('runs', item['device'])][0].relative_to(root).as_posix(),
                        'BN acceptance origin differs')
            check_acceptance_proof(root, proof, report, run, item['device'], files, checked, bn)
            reports[(key, item['device'])] = run, report
        cross_key = 'bn_cross_device_verification_path' if bn else 'cross_device_verification_path'
        path = relative_file(root, acceptance[cross_key])
        require(path.relative_to(root).as_posix() in files, 'Cross-device acceptance proof omitted')
        checked(path, files[path.relative_to(root).as_posix()])
        run, report = reports[(key, 'cuda')]
        check_acceptance_proof(root, read_json(path), report, run, 'cpu', files, checked, bn)
    for key in ('regression_path', 'review_path'):
        path = relative_file(root, acceptance[key])
        require(path.relative_to(root).as_posix() in files, 'Regression/review chain omitted')
        checked(path, files[path.relative_to(root).as_posix()])
        record = read_json(path)
        require(record.get('status') in ('verified', 'passed', 'accepted') and
                record.get('source_sha256', record.get('code_sha256')) == sources, 'Bound successful review/regression required')
        if key == 'regression_path':
            require(type(record.get('exit_code')) is int and record['exit_code'] == 0 and
                    type(record.get('skipped')) is int and record['skipped'] == 0 and
                    record.get('cuda_available') is True, 'Real CUDA regression without skips required')
            for label, checksum in hash_table(record.get('files_sha256')).items():
                require(files.get(label) == checksum, 'Acceptance regression log chain differs')
                checked(relative_file(root, label), checksum)
        else:
            require(record.get('unresolved_findings') == [], 'Unresolved independent review findings')


def read_prediction_csv(path, rows, records, class_count):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == CSV_FIELDS, 'Prediction CSV schema differs')
        values = list(reader)
    require(len(values) == len(rows), 'Prediction CSV count differs')
    predictions = []
    for index, record, row in zip(rows, records, values):
        require(set(row) == set(CSV_FIELDS) and None not in row.values() and
                int(row['row_index']) == index and row['wave_sha256'] == record['wave_sha256'] and
                row['group_id'] == str(record['group_id']) and int(row['class_index']) == record['class_index'],
                'Prediction CSV identity/order differs')
        prediction = int(row['predicted_class_index'])
        require(0 <= prediction < class_count, 'Prediction class outside class list')
        predictions.append(prediction)
    return np.asarray(predictions, dtype=np.int64)


def checked_metrics(true, predictions, records, classes, report_metrics, report_groups, split):
    metrics = old.numpy_metrics(true, predictions, classes)
    error = old.metric_error(metrics, report_metrics, split)
    groups = np.asarray([str(r['group_id']) for r in records])
    require(isinstance(report_groups, dict) and set(report_groups) == set(np.unique(groups)),
            f'Group metric keys differ: {split}')
    grouped, counts = {}, {}
    for group in np.unique(groups):
        mask = groups == group
        grouped[group] = old.numpy_metrics(true[mask], predictions[mask], classes)
        counts[group] = int(mask.sum())
        error = max(error, old.metric_error(grouped[group], report_groups[group], f'{split}/group/{group}'))
    return metrics, grouped, counts, error


def load_checkpoint(run, report, stats):
    def portable(storage, location):
        require(location == 'cpu', 'Portable CPU checkpoint storage required')
        return storage
    checkpoint = torch.load(run/'model.pt', map_location=portable, weights_only=True)
    fields = {'schema_version', 'protocol', 'config', 'normalization', 'selected_epoch', 'classes', 'state_dict'}
    require(isinstance(checkpoint, dict) and set(checkpoint) == fields and
            type(checkpoint['schema_version']) is int and checkpoint['schema_version'] == 1 and
            checkpoint['protocol'] == report['protocol'] and
            canonical(checkpoint['config']) == canonical(report['config']) and
            canonical(checkpoint['normalization']) == canonical(stats) and
            type(checkpoint['selected_epoch']) is int and checkpoint['selected_epoch'] == report['selected_epoch'] and
            canonical(checkpoint['classes']) == canonical(report['classes']), 'Checkpoint schema/metadata differs')
    state = checkpoint['state_dict']
    require(isinstance(state, dict) and state and all(isinstance(k, str) and isinstance(t, torch.Tensor) and
            t.device.type == 'cpu' and torch.isfinite(t).all().item() for k, t in state.items()),
            'Finite CPU checkpoint state dictionary required')
    return state


def verify_run(root, run_dir, device='cpu', allow_device_change=False):
    """Return an independent proof for every train/validation row, read-only."""
    started = time.perf_counter()
    root = Path(root).resolve()
    run = inside(root, run_dir)
    old.evidence_not_failed(root, run/'report.json')
    require(device in ('auto', 'cpu', 'cuda') and type(allow_device_change) is bool, 'Invalid device options')
    evidence = {}

    def checked(path, expected=None):
        path = Path(path).resolve()
        if path.is_relative_to(root):
            old.evidence_not_failed(root, path)
        value = sha256(path)
        if expected is not None:
            require(isinstance(expected, str) and value == expected, f'Checksum differs: {path}')
        require(str(path) not in evidence or evidence[str(path)] == value, f'File changed during verification: {path}')
        evidence[str(path)] = value
        return value

    report_sha = checked(run/'report.json')
    report = read_json(run/'report.json')
    if isinstance(report, dict) and report.get('protocol') == 'paper_dc_ac_bn_v1':
        return verify_bn_run(root, run, device, allow_device_change)
    require(isinstance(report, dict) and report.get('status') == 'complete' and
            type(report.get('schema_version')) is int and report['schema_version'] == 1 and
            report.get('heldout_test_evaluated') is False and report.get('model_roundtrip_verified') is True,
            'Complete sealed schema-1 roundtrip report required')
    config = validate_history(report)
    checked(SOURCE_ROOT/'config/paper_baseline_v1.json', old.BASE_SHA256)
    verify_sources(report, checked)
    hash_table(report.get('files_sha256'), ARTIFACTS)
    artifacts = {name: checked(run/name, report['files_sha256'][name]) for name in sorted(ARTIFACTS)}
    require(report.get('normalization_sha256') == artifacts['normalization_dc_ac.json'],
            'New normalization artifact SHA differs')
    cache = relative_file(root, report['cache_path'])
    require(relative_file(root, config['cache_dir']) == cache, 'Config/report cache path differs')
    metadata_sha = checked(cache/'metadata.json', report['cache_metadata_sha256'])
    checked(cache/'normalization.json', report['old_cache_normalization_sha256'])
    metadata = read_json(cache/'metadata.json')
    require(type(metadata.get('schema_version')) is int and metadata['schema_version'] == 1 and
            metadata.get('signal_shape') == [250, 2] and metadata.get('x_dtype') == 'float32' and
            metadata.get('y_dtype') == 'int64' and metadata.get('cache_units') == 'raw input units, not standardized',
            'Invalid raw cache schema')
    summary_path = relative_file(root, metadata['summary_path'])
    checked(summary_path, metadata['summary_sha256'])
    summary = read_json(summary_path)
    require(metadata['manifest_sha256'] == summary['manifest_sha256'] == report['manifest_sha256'] and
            metadata['inputs'] == summary['inputs'] and metadata['sample_counts'] == summary['sample_counts'] and
            metadata['class_counts'] == summary['class_counts'], 'Cache/manifest provenance differs')
    manifests = hash_table(metadata['manifest_sha256'], {'train.jsonl', 'validation.jsonl', 'test.jsonl'})
    for name, checksum in manifests.items():
        checked(summary_path.parent/name, checksum)
    require(report['raw_input_sha256'] == {k:v['sha256'] for k, v in metadata['inputs'].items()},
            'Raw input provenance differs')
    for info in metadata['inputs'].values():
        checked(relative_file(root, info['path']), info['sha256'])
    cache_files = hash_table(metadata['files_sha256'],
        {f'{s}_{k}.npy' for s in (*SPLITS, 'test') for k in ('x', 'y')} | {'normalization.json'})
    for name, checksum in cache_files.items():
        checked(cache/name, checksum)
    artifact = read_json(run/'normalization_dc_ac.json')
    require(isinstance(artifact, dict) and set(artifact) == {'statistics', 'provenance'},
            'Complete normalization artifact required')
    stats = validate_statistics(artifact['statistics'])
    require(canonical(stats) == canonical(report.get('normalization')), 'Report normalization differs')
    provenance = dict(cache_metadata_sha256=metadata_sha, train_signal_sha256=cache_files['train_x.npy'],
        train_manifest_sha256=manifests['train.jsonl'], fit_source_sha256=report['code_sha256']['src/ect/dc_ac.py'])
    require(canonical(artifact['provenance']) == canonical(provenance), 'Train-only fit provenance differs')
    require(canonical(report.get('normalization_provenance')) == canonical(provenance),
            'Report normalization provenance differs')
    registration = verify_registration(root, run, report, cache, metadata, checked)
    classes = len(report['classes'])
    if config['purpose'] == FULL_PURPOSE:
        require(classes == 20 and report['sample_counts'] == report['available_sample_counts'] ==
                {'train':24000, 'validation':3200}, 'Full run requires 20 classes and all 24000/3200 rows')
    selected, records, all_records, csv_predictions = {}, {}, {}, {}
    for split in SPLITS:
        entries = [json.loads(line) for line in (summary_path.parent/f'{split}.jsonl').read_text().splitlines()]
        require(len(entries) == summary['sample_counts'][split] == report['available_sample_counts'][split] and
                all(type(r.get('class_index')) is int and 0 <= r['class_index'] < classes and
                    r.get('split') == split and isinstance(r.get('wave_sha256'), str) and len(r['wave_sha256']) == 64 and
                    all(c in '0123456789abcdef' for c in r['wave_sha256']) and type(r.get('group_id')) in (str, int)
                    for r in entries), f'Manifest identity/classes/counts differ: {split}')
        labels = np.asarray([r['class_index'] for r in entries], dtype=np.int64)
        require(np.array_equal(np.unique(labels), np.arange(classes)), 'Manifest classes differ')
        rows = read_json(run/f'{split}_rows.json')
        require(isinstance(rows, list) and rows and all(type(i) is int for i in rows) and
                rows == sorted(set(rows)) and rows[0] >= 0 and rows[-1] < len(labels) and
                len(rows) == report['sample_counts'][split], 'Unique ordered in-range selected rows required')
        expected = list(range(len(labels)))
        cap = config[f'{split}_per_class']
        if cap is not None:
            rng = np.random.default_rng(20261002)
            expected = sorted(i for cls in np.unique(labels) for i in rng.choice(np.flatnonzero(labels == cls),
                size=min(cap, int((labels == cls).sum())), replace=False).tolist())
        require(rows == expected and np.array_equal(np.unique(labels[rows]), np.arange(classes)),
                'Selected row/class protocol differs')
        selected[split], all_records[split] = rows, entries
        records[split] = [entries[i] for i in rows]
        csv_predictions[split] = read_prediction_csv(run/f'{split}_predictions.csv', rows, records[split], classes)
    # Configure deterministic/TF32 behavior before loading any checkpoint.
    seed_everything(config['seed'], threads=config['threads'])
    inference_device = resolve_device(device)
    environment = report.get('environment')
    require(isinstance(environment, dict) and environment.get('device') in ('cpu', 'cuda') and
            type(environment.get('threads')) is int and environment['threads'] == config['threads'] and
            environment.get('deterministic_algorithms') is True and environment.get('tf32') is False,
            'Deterministic producer environment required')
    same_device = environment['device'] == inference_device.type
    require(same_device or allow_device_change, 'Cross-device inference requires --allow-device-change')
    state = load_checkpoint(run, report, stats)
    model = PaperResNeXt1D(num_classes=classes, blocks_per_stage=config['blocks_per_stage'])
    initial_hash = initial_parameters_sha256(model)
    require(initial_hash == report.get('initial_parameters_sha256'), 'Seed initialization parameter identity differs')
    require(report.get('parameter_count') == sum(p.numel() for p in model.parameters()), 'Parameter count differs')
    expected_state = model.state_dict()
    require(set(state) == set(expected_state) and all(state[k].shape == t.shape and state[k].dtype == t.dtype
            for k, t in expected_state.items()), 'Checkpoint state shape/dtype differs')
    update_l2 = float(sum(((state[name].double()-parameter.detach().cpu().double())**2).sum().item()
                          for name, parameter in model.named_parameters())**.5)
    require(type(report.get('parameter_update_l2')) in (int, float) and
            np.isfinite(report['parameter_update_l2']) and report['parameter_update_l2'] > 0 and
            abs(update_l2-report['parameter_update_l2']) <= 1e-12, 'Initialized parameter update norm differs')
    model.load_state_dict(state, strict=True)
    require(canonical(bn_summary(model)) == canonical(report['epochs'][report['selected_epoch']-1].get('bn')),
            'Selected checkpoint BN summary differs from history')
    bn_names = set(bn_summary(model))
    batches = (report['sample_counts']['train'] + config['batch_size']-1)//config['batch_size']
    for entry in report['epochs']:
        summary_bn = entry.get('bn')
        require(isinstance(summary_bn, dict) and set(summary_bn) == bn_names, 'Complete history BN summary required')
        for summary_bn in summary_bn.values():
            require(isinstance(summary_bn, dict) and set(summary_bn) ==
                    {'variance_min', 'variance_median', 'variance_max', 'mean_abs_max', 'batches'} and
                    type(summary_bn['batches']) is int and summary_bn['batches'] == entry['epoch']*batches and
                    all(type(summary_bn[k]) in (int, float) and np.isfinite(summary_bn[k]) and summary_bn[k] >= 0
                        for k in ('variance_min', 'variance_median', 'variance_max', 'mean_abs_max')) and
                    summary_bn['variance_min'] <= summary_bn['variance_median'] <= summary_bn['variance_max'],
                    'Invalid epoch BN statistics/counts')
    model.to(inference_device).eval()
    metrics, group_metrics, counts, group_counts = {}, {}, {}, {}
    maximum_error, statistics_error = 0., 0.
    for split in SPLITS:
        mapped = []
        try:
            raw = np.load(cache/f'{split}_x.npy', mmap_mode='r', allow_pickle=False)
            mapped.append(raw)
            labels = np.load(cache/f'{split}_y.npy', mmap_mode='r', allow_pickle=False)
            mapped.append(labels)
            n = summary['sample_counts'][split]
            require(raw.shape == (n, 250, 2) and raw.dtype == np.float32 and
                    labels.shape == (n,) and labels.dtype == np.int64 and
                    np.array_equal(labels, [r['class_index'] for r in all_records[split]]),
                    'Cache shape/dtype/manifest labels differ')
            if split == 'train':
                fitted = fit_statistics(raw)
                for key in STATS_FIELDS:
                    if key in ('mu_d', 'sigma_d', 'sigma_a', 'scale_d', 'scale_a'):
                        actual, expected = np.asarray(fitted[key]), np.asarray(stats[key])
                        statistics_error = max(statistics_error, float(np.max(np.abs(actual-expected))))
                        require(np.allclose(actual, expected, rtol=STATS_RTOL, atol=STATS_ATOL),
                                f'Independent full-train statistics differ: {key}')
                    else:
                        require(fitted[key] == stats[key], f'Independent fit schema/count/floor differs: {key}')
            rows = selected[split]
            predictions = independent_probabilities(model, raw[rows], [r['wave_sha256'] for r in records[split]],
                stats, inference_device, config['batch_size']).argmax(axis=1)
            mismatches = int(np.count_nonzero(predictions != csv_predictions[split]))
            require(mismatches == 0, f'Independent prediction classes differ: {split}; mismatches={mismatches}')
            metric, groups, grouped_counts, error = checked_metrics(labels[rows], predictions, records[split], classes,
                report['metrics'][split], report['group_metrics'][split], split)
            metrics[split], group_metrics[split], group_counts[split] = metric, groups, grouped_counts
            counts[split], maximum_error = len(predictions), max(maximum_error, error)
        finally:
            for array in mapped:
                array._mmap.close()
    verify_sources(report, checked)
    for label, checksum in evidence.items():
        require(sha256(label) == checksum, f'File changed during verification: {label}')
        if Path(label).is_relative_to(root):
            old.evidence_not_failed(root, Path(label))
    script_sha = checked(_SCRIPT_PATH)
    return dict(schema_version=1, status='verified', protocol=PROTOCOL,
        created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        run_dir=run.relative_to(root).as_posix(), run_path=str(run), project_root=str(root),
        device=str(inference_device), device_request=device, same_device=same_device,
        allow_device_change=allow_device_change, heldout_test_evaluated=False,
        config=config, execution_registration=registration, selected_epoch=report['selected_epoch'],
        initial_parameters_sha256=initial_hash, report_sha256=report_sha, weights_sha256=artifacts['model.pt'],
        artifacts_sha256=artifacts, source_sha256=report['code_sha256'], cache_metadata_sha256=metadata_sha,
        normalization_sha256=artifacts['normalization_dc_ac.json'], prediction_counts=counts, group_counts=group_counts,
        metrics=metrics, group_metrics=group_metrics, maximum_metric_error=maximum_error,
        maximum_statistics_error=statistics_error, metric_tolerance=old.METRIC_ATOL,
        statistics_tolerance=dict(rtol=STATS_RTOL, atol=STATS_ATOL), checked_files_sha256=evidence,
        script=dict(path=str(_SCRIPT_PATH), sha256=script_sha, text=_SCRIPT_BYTES.decode('utf-8')),
        checks=dict(independent_statistics=True, full_clean_train_fit_verified=True,
            full250_transform_before_crop=True, initial_parameters_verified=True,
            independent_inference=True, both_splits_inferred=True, all_row_csv_identities_verified=True,
            prediction_classes_exact=True, metrics_verified=True, group_class_metrics_verified=True,
            source_verified=True, complete_artifacts_verified=True, cache_manifest_provenance_verified=True,
            checkpoint_schema_verified=True, history_selection_verified=True, original_run_read_only=True,
            test_classification_evaluated=False))


def bn_rule():
    return dict(calibration_batch_size=128, calibration_seed=0, calibration_epoch=0,
        order='manifest', passes=1, evaluation_batch_size=128, evaluation_seed=10,
        threads=1, deterministic=True, tf32=False, test_classification_enabled=False)


def bn_summary(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            variance = module.running_var.detach().cpu().numpy()
            result[name] = dict(variance_min=float(variance.min()), variance_median=float(np.median(variance)),
                variance_max=float(variance.max()), mean_abs_max=float(module.running_mean.detach().abs().max().cpu()),
                batches=int(module.num_batches_tracked.detach().cpu()))
    return result


def independent_calibration(model, raw, identities, stats, device):
    """One manifest-order 128-batch CMA pass with manually seeded train crops."""
    require(raw.shape == (len(identities), 250, 2) and len(raw) > 0, 'Aligned calibration train required')
    updated = copy.deepcopy(model).train()
    modules = [m for m in updated.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    require(modules and all(m.track_running_stats and m.momentum == .01 for m in modules),
            'Original tracked BN with momentum .01 required')
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
    with torch.no_grad():
        for begin in range(0, len(raw), 128):
            full = transform_full250(raw[begin:begin+128], stats)
            cropped = np.empty((len(full), 224, 2), dtype=np.float32)
            for index, identity in enumerate(identities[begin:begin+128]):
                key = json.dumps(['training', 0, identity, 0], ensure_ascii=True,
                                 separators=(',', ':'), allow_nan=False).encode('ascii')
                generator = np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest(), 'big')))
                start = int(generator.integers(0, 27))
                cropped[index] = full[index, start:start+224]
            logits = updated(torch.from_numpy(np.ascontiguousarray(cropped.transpose(0, 2, 1))).to(device))
            require(torch.isfinite(logits).all().item(), 'Nonfinite independent calibration logits')
    batches = (len(raw)+127)//128
    require(all(int(m.num_batches_tracked) == batches for m in modules), 'Independent CMA counter differs')
    for module in modules:
        module.momentum = .01
    updated.eval()
    return updated


def verify_bn_run(root, run_dir, device='cpu', allow_device_change=False):
    """Independently verify origin, buffer-only intervention and both split predictions."""
    started = time.perf_counter()
    root = Path(root).resolve()
    run = inside(root, run_dir)
    old.evidence_not_failed(root, run/'report.json')
    require(device in ('auto', 'cpu', 'cuda') and type(allow_device_change) is bool, 'Invalid device options')
    report_sha = sha256(run/'report.json')
    report = read_json(run/'report.json')
    require(isinstance(report, dict) and report.get('protocol') == BN_PROTOCOL and
            type(report.get('schema_version')) is int and report['schema_version'] == 1 and
            report.get('status') == 'complete' and report.get('heldout_test_evaluated') is False,
            'Complete sealed BN report required')
    origin = relative_file(root, report['origin_run'])
    require(origin != run and not run.is_relative_to(origin) and not origin.is_relative_to(run),
            'Separate original and BN runs required')
    origin_report = read_json(origin/'report.json')
    require(origin_report.get('protocol') == PROTOCOL, 'E2 training origin required')
    # This checks EVERY original artifact, prediction, metric, fit and registry.
    origin_proof = verify_run(root, origin, device, allow_device_change)
    evidence = dict(origin_proof['checked_files_sha256'])

    def checked(path, expected=None):
        path = Path(path).resolve()
        if path.is_relative_to(root):
            old.evidence_not_failed(root, path)
        checksum = sha256(path)
        require(expected is None or checksum == expected, f'Checksum differs: {path}')
        require(str(path) not in evidence or evidence[str(path)] == checksum, f'Evidence drift: {path}')
        evidence[str(path)] = checksum
        return checksum

    checked(run/'report.json', report_sha)
    require(report['origin_report_sha256'] == origin_proof['report_sha256'] and
            report['origin_model_sha256'] == origin_proof['weights_sha256'], 'Original report/weights differ')
    variable = {'protocol', 'status', 'created_utc', 'environment', 'code_sha256', 'code_sha256_lf',
                'metrics', 'group_metrics', 'files_sha256', 'elapsed_seconds'}
    additions = {'origin_run', 'origin_report_sha256', 'origin_model_sha256', 'baseline_metrics',
                 'baseline_group_metrics', 'bn_execution_registration', 'calibration', 'bn_before', 'bn_after'}
    require(set(report) == set(origin_report) | additions, 'Complete origin-compatible BN report required')
    for field in set(origin_report) - variable:
        require(canonical(report[field]) == canonical(origin_report[field]), f'Origin metadata differs: {field}')
    require(canonical(report['baseline_metrics']) == canonical(origin_report['metrics']) and
            canonical(report['baseline_group_metrics']) == canonical(origin_report['group_metrics']) and
            canonical(report['bn_execution_registration']) == canonical(origin_proof['execution_registration']),
            'Origin baseline/registration differs')
    require(set(report['metrics']) == set(report['group_metrics']) == set(SPLITS), 'Only train/validation BN metrics allowed')
    verify_sources(report, checked)
    require(report['code_sha256'] == origin_proof['source_sha256'], 'Origin/BN source closure differs')
    hash_table(report['files_sha256'], ARTIFACTS)
    artifacts = {n:checked(run/n, report['files_sha256'][n]) for n in sorted(ARTIFACTS)}
    for name in ('normalization_dc_ac.json', 'train_rows.json', 'validation_rows.json'):
        require(artifacts[name] == origin_proof['artifacts_sha256'][name], 'Origin normalization/rows changed')
    config, stats = report['config'], report['normalization']
    seed_everything(config['seed'], threads=1)
    inference_device = resolve_device(device)
    environment = report['environment']
    require(isinstance(environment, dict) and environment.get('device') in ('cpu', 'cuda') and
            type(environment.get('threads')) is int and environment['threads'] == 1 and
            environment.get('deterministic_algorithms') is True and environment.get('tf32') is False,
            'Deterministic BN producer environment required')
    same_device = environment['device'] == inference_device.type
    require(same_device or allow_device_change, 'Cross-device BN replay requires --allow-device-change')
    origin_state = load_checkpoint(origin, origin_report, stats)
    def portable(storage, location):
        require(location == 'cpu', 'Portable CPU BN checkpoint required')
        return storage
    saved = torch.load(run/'model.pt', map_location=portable, weights_only=True)
    expected_fields = {'schema_version', 'protocol', 'config', 'normalization', 'selected_epoch', 'classes',
                       'state_dict', 'origin_model_sha256'}
    require(isinstance(saved, dict) and set(saved) == expected_fields and
            type(saved['schema_version']) is int and saved['schema_version'] == 1 and
            saved['protocol'] == BN_PROTOCOL and saved['origin_model_sha256'] == report['origin_model_sha256'] and
            all(canonical(saved[k]) == canonical(report[k]) for k in ('config', 'normalization', 'classes', 'selected_epoch')),
            'BN checkpoint schema/metadata differs')
    state = saved['state_dict']
    require(isinstance(state, dict) and set(state) == set(origin_state) and
            all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and torch.isfinite(t).all().item() and
                t.shape == origin_state[n].shape and t.dtype == origin_state[n].dtype for n, t in state.items()),
            'Finite portable BN state shape/dtype required')
    original = PaperResNeXt1D(num_classes=len(report['classes']), blocks_per_stage=config['blocks_per_stage'])
    original.load_state_dict(origin_state, strict=True)
    allowed = {f'{name}.{suffix}' for name, m in original.named_modules() if isinstance(m, torch.nn.BatchNorm1d)
               for suffix in ('running_mean', 'running_var', 'num_batches_tracked')}
    changed = sorted(n for n in state if not torch.equal(state[n], origin_state[n]))
    require(not set(changed)-allowed, 'Non-BN state changed')
    calibrated = copy.deepcopy(original)
    calibrated.load_state_dict(state, strict=True)
    require(canonical(report['bn_before']) == canonical(bn_summary(original)) and
            canonical(report['bn_after']) == canonical(bn_summary(calibrated)), 'BN summary differs')
    original.to(inference_device).eval()
    calibrated.to(inference_device).eval()
    ntrain = report['sample_counts']['train']
    calibration = dict(config=bn_rule(), sample_count=ntrain, batch_count=(ntrain+127)//128,
        last_batch_size=(ntrain-1)%128+1, changed_buffers=changed,
        parameters_unchanged=True, original_state_unchanged=True)
    require(canonical(calibration) == canonical(report['calibration']), 'Fixed calibration protocol/counts differ')
    cache = relative_file(root, report['cache_path'])
    metadata = read_json(cache/'metadata.json')
    summary_path = relative_file(root, metadata['summary_path'])
    metrics, group_metrics, counts, group_counts = {}, {}, {}, {}
    maximum_error, buffer_error, buffers_exact = 0., 0., True
    for split in SPLITS:
        mapped = []
        try:
            raw = np.load(cache/f'{split}_x.npy', mmap_mode='r', allow_pickle=False)
            mapped.append(raw)
            labels = np.load(cache/f'{split}_y.npy', mmap_mode='r', allow_pickle=False)
            mapped.append(labels)
            rows = read_json(run/f'{split}_rows.json')
            entries = [json.loads(line) for line in (summary_path.parent/f'{split}.jsonl').read_text().splitlines()]
            records = [entries[i] for i in rows]
            ids = [r['wave_sha256'] for r in records]
            csv_predictions = read_prediction_csv(run/f'{split}_predictions.csv', rows, records, len(report['classes']))
            if split == 'train':
                replayed = independent_calibration(original, raw[rows], ids, stats, inference_device)
                replay_state = replayed.state_dict()
                for name in allowed:
                    expected, actual = replay_state[name].detach().cpu(), state[name]
                    exact = torch.equal(actual, expected)
                    buffers_exact = buffers_exact and exact
                    if actual.dtype.is_floating_point:
                        error = float((actual.double()-expected.double()).abs().max())
                        buffer_error = max(buffer_error, error)
                        require(exact if same_device else torch.allclose(actual, expected, rtol=BUFFER_RTOL, atol=BUFFER_ATOL),
                                f'Independent BN buffer differs: {name}')
                    else:
                        require(exact, f'Independent BN counter differs: {name}')
                for module in calibrated.modules():
                    if isinstance(module, torch.nn.BatchNorm1d):
                        require(int(module.num_batches_tracked) == calibration['batch_count'] and
                                bool((module.running_var >= 0).all()), 'Invalid calibrated BN counter/variance')
            predictions = independent_probabilities(calibrated, raw[rows], ids, stats, inference_device, 128).argmax(axis=1)
            mismatches = int(np.count_nonzero(predictions != csv_predictions))
            require(mismatches == 0, f'Independent BN classes differ: {split}; mismatches={mismatches}')
            metric, grouped, grouped_counts, error = checked_metrics(labels[rows], predictions, records, len(report['classes']),
                report['metrics'][split], report['group_metrics'][split], split)
            metrics[split], group_metrics[split], group_counts[split] = metric, grouped, grouped_counts
            counts[split], maximum_error = len(predictions), max(maximum_error, error)
        finally:
            for array in mapped:
                array._mmap.close()
    require(all(torch.equal(t.detach().cpu(), origin_state[n]) for n, t in original.state_dict().items()),
            'Original in-memory state changed')
    verify_sources(report, checked)
    for label, checksum in evidence.items():
        require(sha256(label) == checksum, f'Evidence drift: {label}')
        if Path(label).is_relative_to(root):
            old.evidence_not_failed(root, Path(label))
    proof = origin_proof | dict(protocol=BN_PROTOCOL, run_dir=run.relative_to(root).as_posix(), run_path=str(run),
        created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        device=str(inference_device), device_request=device, same_device=same_device,
        allow_device_change=allow_device_change, origin_run=report['origin_run'],
        report_sha256=report_sha, weights_sha256=artifacts['model.pt'], artifacts_sha256=artifacts,
        origin_report_sha256=report['origin_report_sha256'], origin_model_sha256=report['origin_model_sha256'],
        metrics=metrics, group_metrics=group_metrics, prediction_counts=counts, group_counts=group_counts,
        maximum_metric_error=maximum_error, calibration=calibration,
        calibration_buffer_max_abs_error=buffer_error, same_device_calibration_exact=(same_device and buffers_exact),
        calibration_buffer_tolerance=dict(rtol=0. if same_device else BUFFER_RTOL, atol=0. if same_device else BUFFER_ATOL),
        checked_files_sha256=evidence)
    proof['checks'] = origin_proof['checks'] | dict(independent_calibration=True, calibration_train_only=True,
        calibration_crop_order_verified=True, bn_buffer_only_changes=True, parameters_unchanged=True,
        original_state_unchanged=True, calibration_buffers_verified=True, origin_run_verified=True)
    return proof


def publish_proof(root, run, output, proof):
    """Atomic no-overwrite publication after a final evidence drift check."""
    root = Path(root).resolve()
    run = inside(root, run)
    output = Path(output)
    output = (output if output.is_absolute() else root/output).resolve()
    require(not output.is_relative_to(run), 'Proof must be outside verified run')
    if proof.get('origin_run'):
        require(not output.is_relative_to(inside(root, proof['origin_run'])), 'Proof must be outside origin run')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    payload = json.dumps(proof, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary, committed = None, False
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=output.parent,
                prefix='.paper-dc-ac-proof-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        for label, checksum in proof['checked_files_sha256'].items():
            require(sha256(label) == checksum, f'Evidence changed before publication: {label}')
            if Path(label).is_relative_to(root):
                old.evidence_not_failed(root, Path(label))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-device-change', action='store_true')
    args = parser.parse_args()
    try:
        root = args.project_root.resolve()
        output = args.output if args.output.is_absolute() else root/args.output
        if output.exists():
            raise FileExistsError(f'Output already exists: {output}')
        require(not output.resolve().is_relative_to(inside(root, args.run)), 'Proof must be outside verified run')
        proof = verify_run(root, args.run, args.device, args.allow_device_change)
        publish_proof(root, args.run, args.output, proof)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, FloatingPointError) as error:
        parser.exit(1, f'DC/AC verification failed: {type(error).__name__}: {error}\n')
    print(json.dumps({k:proof[k] for k in ('status', 'protocol', 'device', 'prediction_counts',
        'maximum_metric_error', 'maximum_statistics_error')}, allow_nan=False))


if __name__ == '__main__':
    main()
