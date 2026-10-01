"""Validation-only 1D training; portable checkpoints and identity-keyed noise."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
import numpy as np
import scipy
import sklearn
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits
from .baseline import stratified_rows
from .devices import resolve_device, seed_everything
from .integrity import file_sha256
from .models import build_model
from .noise import add_ac_noise, augment_training
from .prepare import PreparedDataset
from .preprocessing import standardize

CONDITIONS = {'clean': None, '30': 30, '20': 20, '10': 10}
CODE_FILES = ('deep.py', 'models.py', 'devices.py', 'noise.py', 'baseline.py',
              'prepare.py', 'dataset.py', 'preprocessing.py', 'integrity.py')
ARTIFACTS = {'model.pt', 'train_rows.json', 'validation_rows.json', 'validation_predictions.csv'}


def _canonical_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', newline='\n')


def _validate_config(c):
    keys = {'schema_version', 'protocol', 'purpose', 'model', 'augmentation', 'seed',
            'sampling_seed', 'device', 'threads', 'batch_size', 'max_epochs', 'patience',
            'learning_rate', 'weight_decay', 'train_per_class', 'validation_per_class'}
    if set(c) != keys or c['schema_version'] != 1 or c['protocol'] != 'deep_ac_v1':
        raise ValueError('Known complete deep_ac_v1 configuration required')
    if c['purpose'] not in ('smoke_validation_only', 'development_validation_only'):
        raise ValueError('Validation-only purpose required')
    if c['model'] not in ('cnn', 'resnet') or type(c['augmentation']) is not bool:
        raise ValueError('Known model and boolean augmentation required')
    if c['augmentation'] and c['model'] != 'resnet':
        raise ValueError('Current augmentation comparison is ResNet only')
    if c['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Device must be auto/cpu/cuda')
    for key in ('threads', 'batch_size', 'max_epochs', 'patience'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'Positive integer {key} required')
    for key in ('seed', 'sampling_seed'):
        if type(c[key]) is not int or not 0 <= c[key] < 2 ** 32:
            raise ValueError(f'Nonnegative uint32 {key} required')
    for key in ('learning_rate', 'weight_decay'):
        if isinstance(c[key], bool) or not isinstance(c[key], (int, float)) or not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f'Finite nonnegative {key} required')
    if c['learning_rate'] == 0:
        raise ValueError('Positive learning_rate required')
    for key in ('train_per_class', 'validation_per_class'):
        if c[key] is not None and (type(c[key]) is not int or c[key] < 1):
            raise ValueError(f'Positive {key} or null required')
        if c['purpose'] == 'development_validation_only' and c[key] is not None:
            raise ValueError('Full development runs must use complete train/validation')


def selection_key(metrics, augmentation):
    names = CONDITIONS if augmentation else ('clean',)
    return tuple(float(np.mean([metrics[k][metric] for k in names]))
                 for metric in ('macro_f1', 'accuracy'))


def _metrics(labels, predictions, classes):
    detail = classification_report(labels, predictions, labels=classes, output_dict=True, zero_division=0)
    return dict(accuracy=float(accuracy_score(labels, predictions)),
                macro_f1=float(f1_score(labels, predictions, labels=classes, average='macro', zero_division=0)),
                per_class={str(k): detail[str(k)] for k in classes},
                confusion_matrix=confusion_matrix(labels, predictions, labels=classes).tolist())


def _input_tensor(raw, stats, device):
    values = np.ascontiguousarray(standardize(raw, stats).transpose(0, 2, 1))
    return torch.from_numpy(values).to(device)


def predict_logits(bundle, signals, sample_ids, snr_db=None, seed=10,
                   namespace='validation', batch_size=128):
    """Return CPU NumPy logits for raw-unit B,250,2 signals."""
    x = np.asarray(signals)
    if x.ndim != 3 or x.shape[1:] != (250, 2) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty B,250,2 input required')
    if len(sample_ids) != len(x) or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Matching identities and positive batch size required')
    model = bundle['model']
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            raw = x[start:start + batch_size]
            if snr_db is not None:
                raw, _ = add_ac_noise(raw, sample_ids[start:start + batch_size], snr_db, seed, namespace)
            logits = model(_input_tensor(raw, bundle['normalization'], device))
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite model output')
            predictions.append(logits.cpu().numpy())
    return np.concatenate(predictions)


def predict_raw(bundle, signals, sample_ids, snr_db=None, seed=10,
                namespace='validation', batch_size=128):
    """Predict classes with frozen normalization and optional identity noise."""
    return predict_logits(bundle, signals, sample_ids, snr_db, seed, namespace, batch_size).argmax(axis=1)


def compare_logits(reference, transferred):
    """Record cross-device drift; near-tie argmax changes are not corruption."""
    a, b = np.asarray(reference), np.asarray(transferred)
    if a.ndim != 2 or a.shape != b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Matching finite nonempty logit matrices required')
    return dict(max_abs_error=float(np.max(np.abs(a - b))),
                within_tolerance=bool(np.allclose(a, b, rtol=1e-4, atol=1e-5)),
                rtol=1e-4, atol=1e-5,
                argmax_difference_count=int(np.sum(a.argmax(axis=1) != b.argmax(axis=1))))


def _evaluate(bundle, raw, labels, ids, classes, batch_size):
    predictions = {name: predict_raw(bundle, raw, ids, snr_db=snr, batch_size=batch_size)
                   for name, snr in CONDITIONS.items()}
    return {name: _metrics(labels, prediction, classes) for name, prediction in predictions.items()}, predictions


def train_deep(project_root, config, output_dir, cache_dir='data/processed/grouped_v1', progress=None):
    """Claim a new directory, train on train only, and select on validation."""
    _validate_config(config)
    config = json.loads(json.dumps(config, allow_nan=False))
    root = Path(project_root).resolve()
    output = Path(output_dir)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if not output.is_relative_to(root):
        raise ValueError('Run directory must stay within project root')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)  # atomic ownership; no existing-run overwrite
    started = time.perf_counter()
    report = dict(schema_version=1, status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                  config=config, purpose=config['purpose'], heldout_test_evaluated=False, epochs=[])
    try:
        code = Path(__file__).parent
        report['config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        report['code_sha256'] = {f'src/ect/{n}': file_sha256(code / n) for n in CODE_FILES}
        report['code_sha256_lf'] = {f'src/ect/{n}': _canonical_sha(code / n) for n in CODE_FILES}
        seed_everything(config['seed'], threads=config['threads'])
        device = resolve_device(config['device'])
        report['environment'] = dict(python=platform.python_version(), numpy=np.__version__,
            scipy=scipy.__version__, sklearn=sklearn.__version__,
            torch=str(torch.__version__), torch_cuda_build=torch.version.cuda, device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
            threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark, tf32=False)
        report['noise_protocol'] = dict(reference='joint AC power in raw downsampled units',
            calibration='finite energy, per-channel centered Gaussian draw', validation_seed=10,
            validation_namespace='validation', conditions=CONDITIONS,
            training='identity + training seed + epoch; 50% clean, 50% uniform 10..30 dB',
            zero_ac='unchanged, SNR not applicable')
        report['selection_rule'] = ('equal mean clean/30/20/10 dB validation macro_f1, accuracy, earlier epoch'
                                    if config['augmentation'] else 'clean validation macro_f1, accuracy, earlier epoch')
        with PreparedDataset(root, cache_dir) as store, threadpool_limits(limits=config['threads']):
            report.update(cache_path=str(store.folder.relative_to(root)),
                cache_metadata_sha256=file_sha256(store.folder / 'metadata.json'),
                normalization_sha256=file_sha256(store.folder / 'normalization.json'),
                manifest_sha256=store.metadata['manifest_sha256'],
                raw_input_sha256={n: v['sha256'] for n, v in store.metadata['inputs'].items()},
                class_mapping_status='unverified')
            parent = (root / store.metadata['summary_path']).parent
            records, selected, xs, ys = {}, {}, {}, {}
            for split in ('train', 'validation'):
                records[split] = [json.loads(line) for line in (parent / f'{split}.jsonl').read_text().splitlines()]
                raw, labels = store.arrays[split]
                if not np.array_equal(labels, [r['class_index'] for r in records[split]]):
                    raise ValueError('Cached labels differ from frozen manifest order')
                selected[split] = stratified_rows(labels, config[f'{split}_per_class'], config['sampling_seed'])
                xs[split], ys[split] = raw[selected[split]], labels[selected[split]]
                records[split] = [records[split][i] for i in selected[split]]
            classes = np.unique(ys['train'])
            if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))) or not np.array_equal(classes, np.unique(ys['validation'])):
                raise ValueError('Train/validation must cover the same contiguous zero-based classes')
            report['sample_counts'] = {s: len(ys[s]) for s in ys}
            report['available_sample_counts'] = {s: len(store.arrays[s][1]) for s in ys}
            report['classes'] = classes.tolist()
            report['normalization'] = store.stats
            model = build_model(config['model'], num_classes=len(classes)).to(device)
            initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            report['parameter_count'] = sum(p.numel() for p in model.parameters())
            bundle = dict(model=model, normalization=store.stats)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            loss_fn = torch.nn.CrossEntropyLoss()
            ids = {s: [r['wave_sha256'] for r in records[s]] for s in records}
            noise_masks = {}
            report['validation_noise_summary'] = {}
            for condition, snr in CONDITIONS.items():
                if snr is None:
                    continue
                _, info = add_ac_noise(xs['validation'], ids['validation'], snr, 10, 'validation')
                mask = np.asarray(info['snr_applicable'], dtype=bool)
                noise_masks[condition] = mask
                actual = 10 * np.log10(np.asarray(info['ac_power'])[mask] / np.asarray(info['noise_power'])[mask])
                report['validation_noise_summary'][condition] = dict(applicable_count=int(mask.sum()),
                    not_applicable_count=int((~mask).sum()),
                    max_snr_error_db=float(np.max(np.abs(actual - snr))) if len(actual) else None)
            best_key, best_state, best_predictions, stale = None, None, None, 0
            for epoch in range(config['max_epochs']):
                epoch_started = time.perf_counter()
                model.train()
                order = np.random.default_rng(np.random.SeedSequence([config['seed'], epoch, 2026])).permutation(len(ys['train']))
                loss_sum, augmented_count, zero_ac_count = 0., 0, 0
                for start in range(0, len(order), config['batch_size']):
                    rows = order[start:start + config['batch_size']]
                    raw = xs['train'][rows]
                    if config['augmentation']:
                        raw, info = augment_training(raw, [ids['train'][i] for i in rows], config['seed'], epoch)
                        augmented_count += int(np.sum(info['augmented']))
                        zero_ac_count += int(np.sum(np.logical_and(info['augmented'], np.logical_not(info['snr_applicable']))))
                    inputs = _input_tensor(raw, store.stats, device)
                    labels = torch.from_numpy(np.array(ys['train'][rows], copy=True)).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(inputs), labels)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError('Nonfinite gradient')
                    optimizer.step()
                    loss_sum += float(loss.detach().cpu()) * len(rows)
                metrics, predictions = _evaluate(bundle, xs['validation'], ys['validation'], ids['validation'], classes, config['batch_size'])
                key = (*selection_key(metrics, config['augmentation']), -epoch)
                record = dict(epoch=epoch, train_loss=loss_sum / len(order), augmented_count=augmented_count,
                              augmented_zero_ac_count=zero_ac_count, validation_metrics=metrics,
                              selection_score=list(key[:2]), elapsed_seconds=time.perf_counter() - epoch_started)
                report['epochs'].append(record)
                if best_key is None or key > best_key:
                    best_key, stale = key, 0
                    best_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
                    best_predictions = predictions
                    report.update(selected_epoch=epoch, validation_metrics=metrics)
                else:
                    stale += 1
                if progress is not None:
                    progress(record)
                if stale >= config['patience']:
                    break
            model.load_state_dict(best_state)
            report['parameter_update_l2'] = float(np.sqrt(sum(
                torch.sum((p.detach().cpu() - initial[n]) ** 2).item() for n, p in model.named_parameters())))
            report['validation_group_metrics'] = {}
            groups = np.array([r['group_id'] for r in records['validation']])
            for group in np.unique(groups):
                mask = groups == group
                report['validation_group_metrics'][str(group)] = {c: _metrics(ys['validation'][mask], p[mask], classes)
                                                                  for c, p in best_predictions.items()}
            with tempfile.TemporaryDirectory(prefix='.deep-', dir=output) as temporary:
                stage = Path(temporary)
                checkpoint = dict(schema_version=1, protocol='deep_ac_v1', model_name=config['model'],
                                  num_classes=len(classes), state_dict=best_state, normalization=store.stats,
                                  config=config, selected_epoch=report['selected_epoch'])
                torch.save(checkpoint, stage / 'model.pt')
                loaded = torch.load(stage / 'model.pt', map_location='cpu', weights_only=True)
                restored = build_model(config['model'], num_classes=len(classes))
                restored.load_state_dict(loaded['state_dict'])
                report['checkpoint_tensors_saved_on_cpu'] = all(t.device.type == 'cpu' for t in best_state.values())
                restored.to(device)
                restored_bundle = dict(model=restored, normalization=loaded['normalization'])
                for condition, snr in CONDITIONS.items():
                    predictions = predict_raw(restored_bundle, xs['validation'], ids['validation'], snr_db=snr,
                                              batch_size=config['batch_size'])
                    if not np.array_equal(predictions, best_predictions[condition]):
                        raise ValueError('Same-device reloaded predictions differ from selected model')
                report['model_roundtrip_verified'] = True
                report['roundtrip_device'] = str(device)
                restored.cpu()
                report['cpu_transfer_comparison'] = {}
                for condition, snr in CONDITIONS.items():
                    reference = predict_logits(bundle, xs['validation'], ids['validation'], snr_db=snr, batch_size=config['batch_size'])
                    cpu_logits = predict_logits(restored_bundle, xs['validation'], ids['validation'], snr_db=snr, batch_size=config['batch_size'])
                    comparison = compare_logits(reference, cpu_logits)
                    comparison['cpu_metrics'] = _metrics(ys['validation'], cpu_logits.argmax(axis=1), classes)
                    report['cpu_transfer_comparison'][condition] = comparison
                for split in selected:
                    _json(stage / f'{split}_rows.json', selected[split].tolist())
                with (stage / 'validation_predictions.csv').open('w', newline='') as handle:
                    writer = csv.writer(handle, lineterminator='\n')
                    writer.writerow(['condition', 'noise_seed', 'snr_applicable', 'row_index', 'wave_sha256', 'group_id', 'class_index', 'predicted_class_index'])
                    for condition, prediction in best_predictions.items():
                        for position, (row_index, row, predicted) in enumerate(zip(selected['validation'], records['validation'], prediction)):
                            writer.writerow([condition, '' if condition == 'clean' else 10,
                                             '' if condition == 'clean' else bool(noise_masks[condition][position]), int(row_index),
                                             row['wave_sha256'], row['group_id'], row['class_index'], int(predicted)])
                report['files_sha256'] = {n: file_sha256(stage / n) for n in sorted(ARTIFACTS)}
                report.update(status='complete', elapsed_seconds=time.perf_counter() - started)
                _json(stage / 'report.json', report)
                for name in sorted(ARTIFACTS):
                    (stage / name).rename(output / name)
                (stage / 'report.json').rename(output / 'report.json')
        return report
    except BaseException as error:
        for name in ARTIFACTS | {'report.json'}:
            (output / name).unlink(missing_ok=True)
        report.update(status='failed', error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started)
        _json(output / 'failure.json', report)
        raise


def load_deep_model(run_dir, device='cpu'):
    """Verify completed local artifacts, load CPU tensors, then move model."""
    folder = Path(run_dir)
    if (folder / 'failure.json').exists():
        raise ValueError('Failed run cannot be loaded')
    report = json.loads((folder / 'report.json').read_text())
    if report.get('status') != 'complete' or set(report.get('files_sha256', {})) != ARTIFACTS:
        raise ValueError('Complete checkpoint checksum table required')
    if type(report.get('schema_version')) is not int or report['schema_version'] != 1:
        raise ValueError('Unsupported run schema')
    _validate_config(report['config'])
    config_digest = hashlib.sha256(json.dumps(report['config'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if report.get('config_sha256') != config_digest:
        raise ValueError('Configuration digest mismatch')
    epochs = report.get('epochs', [])
    if not epochs or [e.get('epoch') for e in epochs] != list(range(len(epochs))):
        raise ValueError('Contiguous epoch selection record required')
    selected = max(epochs, key=lambda e: (*selection_key(e['validation_metrics'], report['config']['augmentation']), -e['epoch']))
    if report.get('selected_epoch') != selected['epoch'] or report.get('validation_metrics') != selected['validation_metrics']:
        raise ValueError('Contradictory epoch/validation selection record')
    for name, digest in report['files_sha256'].items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f'Model run checksum mismatch: {name}')
    expected_code = {f'src/ect/{n}': _canonical_sha(Path(__file__).with_name(n)) for n in CODE_FILES}
    if report.get('code_sha256_lf') != expected_code:
        raise ValueError('Training/inference implementation differs from checkpoint')
    checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
    if (checkpoint.get('schema_version') != 1 or checkpoint.get('protocol') != 'deep_ac_v1'
            or checkpoint.get('config') != report['config'] or checkpoint.get('normalization') != report['normalization']
            or checkpoint.get('selected_epoch') != report['selected_epoch']
            or checkpoint.get('model_name') != report['config']['model']
            or type(checkpoint.get('num_classes')) is not int
            or report.get('classes') != list(range(checkpoint['num_classes']))):
        raise ValueError('Checkpoint metadata differs from selection report')
    model = build_model(checkpoint['model_name'], num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(resolve_device(device)).eval()
    return dict(model=model, normalization=checkpoint['normalization'], report=report)
