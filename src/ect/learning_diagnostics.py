"""Frozen-weight train/validation diagnostics; batch BN is a counterfactual only."""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from .devices import resolve_device, seed_everything
from .evaluation import (inside, write_json, record_failure, count_metrics,
                         load_registered, evaluation_code)
from .features import extract_features
from .integrity import file_sha256
from .prepare import PreparedDataset
from .preprocessing import standardize

CODE = ('src/ect/learning_diagnostics.py', 'scripts/diagnose_learning.py')
MODES = {'frozen': None, 'batch_stats_order_20261002': 20261002,
         'batch_stats_order_20261003': 20261003}


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def diagnostic_logits(bundle, raw, batch_size=128, batch_stats=False, order_seed=None):
    """Clone model, infer without gradients, return original row order.

    Batch-statistics mode uses the current mixed batch, including other samples.
    It is neither an independent per-sample prediction nor a deployment score.
    Running buffers, parameters and mode of the original model remain unchanged.
    """
    values = np.asarray(raw)
    if (values.ndim != 3 or values.shape[1:] != (250, 2) or not len(values)
            or not np.isfinite(values).all()):
        raise ValueError('Finite nonempty B,250,2 input required')
    if type(batch_size) is not int or batch_size < 1 or type(batch_stats) is not bool:
        raise ValueError('Positive batch size and boolean batch_stats required')
    if order_seed is not None and (type(order_seed) is not int or not 0 <= order_seed < 2**32):
        raise ValueError('Order seed must be uint32 or None')
    model = copy.deepcopy(bundle['model']).eval()
    if batch_stats:
        found = False
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.train()
                module.track_running_stats = False
                found = True
        if not found:
            raise ValueError('Batch-statistics counterfactual requires BatchNorm')
    device = next(model.parameters()).device
    order = (np.arange(len(values)) if order_seed is None
             else np.random.default_rng(order_seed).permutation(len(values)))
    result = None
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            rows = order[start:start+batch_size]
            normalized = standardize(values[rows], bundle['normalization'])
            inputs = torch.from_numpy(np.ascontiguousarray(normalized.transpose(0, 2, 1))).to(device)
            outputs = model(inputs)
            if not torch.isfinite(outputs).all():
                raise ValueError('Nonfinite diagnostic output')
            logits = outputs.cpu().numpy()
            if logits.ndim != 2 or logits.shape[0] != len(rows) or logits.shape[1] < 2:
                raise ValueError('Classification logits required')
            if result is None:
                result = np.empty((len(values), logits.shape[1]), dtype=logits.dtype)
            if logits.shape[1] != result.shape[1]:
                raise ValueError('Class count changed across batches')
            result[rows] = logits
    return result


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError('Finite nonempty quantile input required')
    return {str(q): float(np.quantile(array, q)) for q in (0., .1, .5, .9, 1.)}


def signal_summary(raw, stats):
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (250, 2) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty B,250,2 signals required')
    normalized = standardize(values, stats).astype(np.float64)
    ac = normalized - normalized.mean(axis=1, keepdims=True)
    dc = normalized.mean(axis=1)
    rms_channel = np.sqrt(np.mean(ac**2, axis=1))
    return dict(sample_count=len(values),
        raw_channel_mean=values.mean(axis=(0, 1)).tolist(),
        raw_channel_std=values.std(axis=(0, 1)).tolist(),
        normalized_channel_mean=normalized.mean(axis=(0, 1)).tolist(),
        normalized_channel_std=normalized.std(axis=(0, 1)).tolist(),
        normalized_between_sample_dc_std=dc.std(axis=0).tolist(),
        normalized_joint_ac_rms_quantiles=quantiles(np.sqrt(np.mean(ac**2, axis=(1, 2)))),
        normalized_channel_ac_rms_quantiles=[quantiles(rms_channel[:, c]) for c in range(2)],
        normalized_sample_dc_quantiles=[quantiles(dc[:, c]) for c in range(2)])


def bn_summary(model):
    return {name: dict(eps=float(module.eps), momentum=module.momentum,
        running_mean_quantiles=quantiles(module.running_mean.detach().cpu().numpy()),
        running_var_quantiles=quantiles(module.running_var.detach().cpu().numpy()),
        num_batches_tracked=int(module.num_batches_tracked.detach().cpu()))
        for name, module in model.named_modules() if isinstance(module, nn.BatchNorm1d)}


def diagnose_learning(project_root, output_dir, device='auto',
        registry='results/experiments/formal_v1/registry.json',
        cache_dir='data/processed/grouped_v1', progress=None):
    root = Path(project_root).resolve()
    output = inside(root, output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = dict(schema_version=1, protocol='frozen_learning_diagnostic_v1', status='running',
        created_utc=datetime.now(timezone.utc).isoformat(), test_classification_evaluated=False,
        splits=['train', 'validation'], batch_size=128, modes=MODES,
        interpretation='Exploratory diagnostics, not new model selection or performance improvement',
        batch_statistics_scope='Current-batch-dependent, transductive counterfactual; not deployment performance',
        metrics=[], models={})
    try:
        executed_root = Path(__file__).resolve().parents[2]
        code = {p: file_sha256(executed_root/p) for p in CODE}
        if any(file_sha256(inside(root, p)) != h for p, h in code.items()):
            raise ValueError('Project source differs from executed diagnostic implementation')
        report['code_sha256'] = code | evaluation_code(root)
        seed_everything(0, 1)
        chosen = resolve_device(device)
        report['environment'] = dict(torch=str(torch.__version__), numpy=np.__version__,
            device=str(chosen), device_name=torch.cuda.get_device_name(chosen) if chosen.type == 'cuda' else 'cpu',
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), tf32=False)
        registry_path = inside(root, registry)
        report.update(registry_path=str(registry_path.relative_to(root)), registry_sha256=file_sha256(registry_path))
        with PreparedDataset(root, cache_dir) as store, threadpool_limits(limits=1):
            registration, models, bindings = load_registered(root, registry_path, store, chosen)
            report.update(model_bindings=bindings, cache_metadata_sha256=file_sha256(store.folder/'metadata.json'),
                normalization_sha256=file_sha256(store.folder/'normalization.json'),
                manifest_sha256=store.metadata['manifest_sha256'], raw_input_sha256=registration['raw_input_sha256'],
                normalization=store.stats, input_summary={}, sample_counts={}, class_mapping_status='unverified')
            manifests = (root/store.metadata['summary_path']).parent
            records = {}
            for split in ('train', 'validation'):
                records[split] = [json.loads(line) for line in (manifests/f'{split}.jsonl').read_text().splitlines()]
                raw, labels = store.arrays[split]
                if not np.array_equal(labels, [r['class_index'] for r in records[split]]):
                    raise ValueError('Labels differ from frozen manifest')
                report['sample_counts'][split] = len(labels)
                report['input_summary'][split] = signal_summary(raw, store.stats)
            before = {m['run_id']: state_digest(m['bundle']['model']) for m in models if m['family'] != 'svm'}
            prediction_count = 0
            with (output/'predictions.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['split', 'run_id', 'family', 'training_seed',
                    'mode', 'order_seed', 'row_index', 'wave_sha256', 'group_id', 'true_class', 'predicted_class'],
                    lineterminator='\n')
                writer.writeheader()
                for model in models:
                    run_id = model['run_id']
                    if model['family'] != 'svm':
                        bundle = model['bundle']
                        training = bundle['report']
                        report['models'][run_id] = dict(family=model['family'], training_seed=model['training_seed'],
                            parameter_count=training['parameter_count'], selected_epoch=training['selected_epoch'],
                            epochs_completed=len(training['epochs']),
                            selected_train_loss=training['epochs'][training['selected_epoch']]['train_loss'],
                            final_train_loss=training['epochs'][-1]['train_loss'], bn=bn_summary(bundle['model']))
                    else:
                        report['models'][run_id] = dict(family='svm', training_seed=None)
                    for split in ('train', 'validation'):
                        raw, labels = store.arrays[split]
                        classes = np.asarray(models[0]['bundle']['report']['classes'])
                        active_modes = {'frozen': None} if model['family'] == 'svm' else MODES
                        for mode, order_seed in active_modes.items():
                            if model['family'] == 'svm':
                                prediction = model['bundle']['pipeline'].predict(extract_features(raw))
                            else:
                                prediction = diagnostic_logits(bundle, raw, batch_stats=mode != 'frozen',
                                    order_seed=order_seed).argmax(axis=1)
                            metrics = count_metrics(labels, prediction, classes)
                            groups = np.array([r['group_id'] for r in records[split]])
                            report['metrics'].append(dict(run_id=run_id, family=model['family'],
                                training_seed=model['training_seed'], split=split, mode=mode, order_seed=order_seed,
                                metrics=metrics, group_metrics={str(g):count_metrics(labels[groups == g],
                                    prediction[groups == g], classes) for g in np.unique(groups)}))
                            for index, (record, actual, predicted) in enumerate(zip(records[split], labels, prediction)):
                                writer.writerow(dict(split=split, run_id=run_id, family=model['family'],
                                    training_seed=model['training_seed'], mode=mode, order_seed=order_seed,
                                    row_index=index, wave_sha256=record['wave_sha256'], group_id=record['group_id'],
                                    true_class=int(actual), predicted_class=int(predicted)))
                            prediction_count += len(prediction)
                            if progress:
                                progress(f'{run_id} {split} {mode}: Accuracy={metrics["accuracy"]:.6f}')
            after = {m['run_id']:state_digest(m['bundle']['model']) for m in models if m['family'] != 'svm'}
            if before != after:
                raise ValueError('Diagnostic modified original model state')
            # Verify original reports and checkpoint files remain exactly the same after inference.
            for run in registration['runs']:
                folder = inside(root, run['output_dir'])
                binding = bindings[run['run_id']]
                if file_sha256(folder/'report.json') != binding['report_sha256'] or any(
                        file_sha256(folder/name) != h for name, h in binding['files_sha256'].items()):
                    raise ValueError('Original checkpoint artifacts changed during diagnostics')
            baseline = inside(root, registration['plan']['baseline_report'])
            if file_sha256(baseline) != bindings['svm']['report_sha256'] or any(
                    file_sha256(baseline.parent/name) != h for name,h in bindings['svm']['files_sha256'].items()):
                raise ValueError('Original SVM artifacts changed during diagnostics')
            report.update(prediction_count=prediction_count, original_model_state_sha256=before,
                checkpoint_files_unchanged=True, original_model_states_unchanged=True,
                files_sha256={'predictions.csv':file_sha256(output/'predictions.csv')},
                elapsed_seconds=time.perf_counter()-started, status='complete')
            write_json(output/'report.json', report)
        return report
    except BaseException as error:
        record_failure(output, report, error, started)
        raise
