"""Fixed train-only BN sidecar for completed, selected E2 DC/AC models.

The E2 registry binds this module and its CLI before full training. This
intervention has no separate registry, optimizer or checkpoint selection.
The training configuration and complete epoch history remain those of the
origin; only this report/checkpoint's top-level protocol identifies the ablation.
"""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .devices import resolve_device, seed_everything
from .integrity import file_sha256
from .paper_crops import training_crops
from .prepare import PreparedDataset

PROTOCOL = 'paper_dc_ac_bn_v1'
ARTIFACTS = {'model.pt', 'train_rows.json', 'validation_rows.json',
             'train_predictions.csv', 'validation_predictions.csv',
             'normalization_dc_ac.json'}
SOURCE_ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = ('src/ect/paper_dc_ac_bn.py', 'scripts/recalibrate_paper_dc_ac_bn.py')
_SOURCE_BYTES = {name: (SOURCE_ROOT / name).read_bytes() for name in CODE_FILES
                 if (SOURCE_ROOT / name).exists()}


def fixed_config():
    return dict(calibration_batch_size=128, calibration_seed=0, calibration_epoch=0,
        order='manifest', passes=1, evaluation_batch_size=128, evaluation_seed=10,
        threads=1, deterministic=True, tf32=False, test_classification_enabled=False)


def config_matches(config):
    return (isinstance(config, dict) and
        json.dumps(config, sort_keys=True, allow_nan=False) ==
        json.dumps(fixed_config(), sort_keys=True, allow_nan=False))


def allowed_bn_buffers(model):
    return {f'{name}.{suffix}' if name else suffix
            for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm1d)
            for suffix in ('running_mean', 'running_var', 'num_batches_tracked')}


def state_changes(original, updated):
    """Return exact changed BN-buffer names; reject every other state change."""
    before, after = original.state_dict(), updated.state_dict()
    if set(before) != set(after):
        raise ValueError('State keys changed')
    changed = []
    for name in before:
        a, b = before[name].detach().cpu(), after[name].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('State shape/dtype changed')
        if not torch.equal(a, b):
            changed.append(name)
    if set(changed) - allowed_bn_buffers(original):
        raise ValueError('Non-BN state changed')
    old_parameters, new_parameters = dict(original.named_parameters()), dict(updated.named_parameters())
    if set(old_parameters) != set(new_parameters):
        raise ValueError('Learned parameter keys changed')
    for name, parameter in old_parameters.items():
        if not torch.equal(parameter.detach().cpu(), new_parameters[name].detach().cpu()):
            raise ValueError('Learned parameter changed')
    if any(not torch.isfinite(value).all() for value in after.values()):
        raise ValueError('Nonfinite recalibrated state')
    return sorted(changed)


def calibrate_dc_ac_bn(model, raw, ids, stats, split='train'):
    """Copy; full250 transform, seed0/epoch0 crops, one 128-batch CMA pass.

    The final partial batch counts equally as a BN update, as in E1. The
    original model's state, modes and gradients are never modified.
    """
    if split != 'train':
        raise ValueError('Calibration only accepts train')
    values = np.asarray(raw)
    if isinstance(ids, (str, bytes)):
        raise ValueError('Matching nonblank identities required')
    try:
        identities = list(ids)
    except TypeError as error:
        raise ValueError('Matching nonblank identities required') from error
    if (values.dtype.kind not in 'fiu' or values.ndim != 3 or
        values.shape[1:] != (250, 2) or not len(values) or len(values) != len(identities) or
        any(not isinstance(identity, str) or not identity.strip() for identity in identities) or
        not np.isfinite(values).all()):
        raise ValueError('Finite nonempty train waveforms and matching identities required')
    if not isinstance(model, torch.nn.Module):
        raise ValueError('Torch model required')
    bns = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    if not bns or any(not module.track_running_stats for module in bns):
        raise ValueError('Tracked BatchNorm1d required')
    if any(isinstance(module, torch.nn.modules.dropout._DropoutNd) for module in model.modules()):
        raise ValueError('Dropout is outside this fixed intervention')
    if any(module.momentum != .01 for module in bns):
        raise ValueError('E2 BatchNorm momentum .01 required')
    from .dc_ac import transform_dc_ac
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    updated = copy.deepcopy(model).train()
    bns = [module for module in updated.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    for module in bns:
        module.reset_running_stats()
        module.momentum = None
    tensors = list(updated.parameters()) + list(updated.buffers())
    device = tensors[0].device
    with torch.no_grad():
        for start in range(0, len(values), 128):
            # Never crop raw data before computing the per-sample DC component.
            full = transform_dc_ac(values[start:start + 128], stats)
            cropped = training_crops(full, identities[start:start + 128], 0, 0)
            inputs = torch.from_numpy(np.ascontiguousarray(cropped.transpose(0, 2, 1))).to(device)
            logits = updated(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite calibration logits')
    for module in bns:
        module.momentum = .01
    updated.eval()
    changed = state_changes(model, updated)
    if any(not torch.equal(value, model.state_dict()[name].detach().cpu()) for name, value in before.items()):
        raise ValueError('Original state changed')
    batches = math.ceil(len(values) / 128)
    if any(int(module.num_batches_tracked) != batches for module in bns):
        raise ValueError('Calibration batch count differs')
    return updated, dict(sample_count=len(values), batch_count=batches,
        last_batch_size=(len(values) - 1) % 128 + 1, changed_buffers=changed,
        parameters_unchanged=True, original_state_unchanged=True)


def _parent():
    # Lazy import allows calibration component tests while parent development
    # is in progress, and avoids a registry/trainer/sidecar import cycle.
    from . import paper_dc_ac_training
    return paper_dc_ac_training


def write_json(path, value):
    _parent().write_json(path, value)


def _same(a, b):
    """JSON identity, including bool-versus-int and float-versus-int types."""
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def inside(root, path):
    target = _parent()._inside(root, path)
    if target == root:
        raise ValueError('Path must be within project root, not the root itself')
    return target


def source_hashes(canonical=False):
    """Bind the parent's complete executable closure and this sidecar/CLI."""
    training = _parent()
    hashes = training.source_hashes(canonical=canonical)
    for name, data in _SOURCE_BYTES.items():
        if (SOURCE_ROOT / name).read_bytes() != data:
            raise ValueError(f'Source changed after import: {name}')
    training._verify_loaded_module(sys.modules[__name__], Path(__file__).resolve(),
                                  _SOURCE_BYTES['src/ect/paper_dc_ac_bn.py'])
    for name in CODE_FILES:
        data = (SOURCE_ROOT / name).read_bytes()
        digest = hashlib.sha256(data.replace(b'\r\n', b'\n') if canonical else data).hexdigest()
        if name in hashes and hashes[name] != digest:
            raise ValueError(f'Parent/sidecar source identity differs: {name}')
        hashes[name] = digest
    return hashes


def _origin_registration(root, origin, old):
    """Full origins use their main E2 registry, including its fixed BN config."""
    training = _parent()
    if old['purpose'] == training.SMOKE_PURPOSE:
        if old.get('execution_registration') is not None:
            raise ValueError('Smoke origin must not claim full registration')
        return None
    if old['purpose'] != training.FULL_PURPOSE:
        raise ValueError('Completed E2 full or smoke origin required')
    if old['sample_counts'] != {'train': 24000, 'validation': 3200}:
        raise ValueError('Full train/validation origin required')
    registered = old.get('execution_registration')
    if not isinstance(registered, dict):
        raise ValueError('Full origin E2 execution registration required')
    path = inside(root, registered['registry_path'])
    registry = training.json_read(path)
    if (file_sha256(path) != registered['registry_sha256'] or
        not config_matches(registry.get('bn_calibration')) or
        not _same(registry.get('code_sha256'), source_hashes())):
        raise ValueError('Origin registration must bind fixed BN config and sidecar sources')
    from .paper_dc_ac_registration import validate_registered_run
    checked = validate_registered_run(root, path, old['config'], inside(root, old['cache_path']), origin)
    if not _same(checked, registered):
        raise ValueError('Origin execution registration changed')
    return checked


def _selected_data(root, origin, old, store):
    """Only train/validation origin rows; no test classification or fitting."""
    training = _parent()
    if (file_sha256(store.folder / 'metadata.json') != old['cache_metadata_sha256'] or
        not _same(store.metadata['manifest_sha256'], old['manifest_sha256'])):
        raise ValueError('Origin/cache identity differs')
    artifact = training.validate_normalization_artifact(training.json_read(origin / 'normalization_dc_ac.json'))
    provenance = artifact['provenance']
    if (not _same(artifact['statistics'], old['normalization']) or
        provenance['train_signal_sha256'] != store.metadata['files_sha256']['train_x.npy'] or
        provenance['train_manifest_sha256'] != store.metadata['manifest_sha256']['train.jsonl'] or
        provenance['cache_metadata_sha256'] != file_sha256(store.folder / 'metadata.json')):
        raise ValueError('Frozen DC/AC train provenance differs')
    manifests = inside(root, store.metadata['summary_path']).parent
    data = {}
    for split in ('train', 'validation'):
        rows = [json.loads(line) for line in (manifests / f'{split}.jsonl').read_text().splitlines()]
        raw, labels = store.arrays[split]
        if not np.array_equal(labels, [row['class_index'] for row in rows]):
            raise ValueError('Manifest/cache labels differ')
        selected = training.json_read(origin / f'{split}_rows.json')
        if (not isinstance(selected, list) or not selected or
            any(type(i) is not int or not 0 <= i < len(rows) for i in selected) or
            len(set(selected)) != len(selected) or len(selected) != old['sample_counts'][split]):
            raise ValueError('Origin row selection invalid')
        if old['purpose'] == training.FULL_PURPOSE and selected != list(range(len(rows))):
            raise ValueError('Full calibration/evaluation requires complete manifest order')
        if selected != sorted(selected):
            raise ValueError('Calibration/evaluation requires manifest order')
        records = [rows[i] for i in selected]
        data[split] = dict(raw=raw[selected], labels=labels[selected], selected=selected,
            records=records, ids=[row['wave_sha256'] for row in records])
    return data


def run_bn(root, origin_run, output_dir, device='auto'):
    """Apply the fixed BN intervention to one completed E2 selected model."""
    root = Path(root).resolve()
    origin, output = inside(root, origin_run), inside(root, output_dir)
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(origin) or origin.is_relative_to(output):
        raise ValueError('Output must not contain or lie within origin')
    training = _parent()
    sources, sources_lf = source_hashes(), source_hashes(canonical=True)
    # Set deterministic algorithms and TF32 before any independent loading.
    seed_everything(0, threads=1)
    selected_device = resolve_device(device)
    original = training.load_paper_dc_ac_model(origin, device=str(selected_device))
    old = original['report']
    registration = _origin_registration(root, origin, old)
    if old['protocol'] != 'paper_dc_ac_finite_v1' or old['config']['protocol'] != 'paper_dc_ac_finite_v1':
        raise ValueError('Completed E2 finite selected model required')
    origin_hashes = {name: file_sha256(origin / name) for name in (*ARTIFACTS, 'report.json')}
    report = copy.deepcopy(old)
    report.update(protocol=PROTOCOL, status='running', created_utc=datetime.now(timezone.utc).isoformat(),
        origin_run=origin.relative_to(root).as_posix(), origin_report_sha256=origin_hashes['report.json'],
        origin_model_sha256=origin_hashes['model.pt'], code_sha256=sources, code_sha256_lf=sources_lf,
        baseline_metrics=copy.deepcopy(old['metrics']), baseline_group_metrics=copy.deepcopy(old['group_metrics']),
        bn_execution_registration=registration)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        with (output / 'running.marker').open('x', newline='\n') as handle:
            handle.write('Incomplete BN intervention; refuse loading.\n')
        seed_everything(old['config']['seed'], threads=1)
        report['environment'] = dict(device=str(selected_device),
            device_name=torch.cuda.get_device_name(selected_device) if selected_device.type == 'cuda' else 'cpu',
            python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
            torch_cuda_build=torch.version.cuda, threads=torch.get_num_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            tf32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32))
        with PreparedDataset(root, inside(root, old['cache_path'])) as store, threadpool_limits(limits=1):
            data = _selected_data(root, origin, old, store)
            model, summary = calibrate_dc_ac_bn(original['model'], data['train']['raw'],
                data['train']['ids'], original['normalization'])
            report['calibration'] = summary | dict(config=fixed_config())
            report['bn_before'] = training.bn_summary(original['model'])
            report['bn_after'] = training.bn_summary(model)
            # Copy metadata exactly; this preserves all parent checkpoint fields
            # while replacing only the state and top-level intervention protocol.
            saved = torch.load(origin / 'model.pt', map_location='cpu', weights_only=True)
            saved.update(protocol=PROTOCOL, origin_model_sha256=origin_hashes['model.pt'],
                state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
            torch.save(saved, output / 'model.pt')
            disk = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
            restored = copy.deepcopy(model).cpu()
            restored.load_state_dict(disk['state_dict'], strict=True)
            if any(not torch.equal(value, disk['state_dict'][name]) for name, value in restored.state_dict().items()):
                raise ValueError('Portable state roundtrip differs')
            restored.to(selected_device).eval()
            for name in ('train_rows.json', 'validation_rows.json', 'normalization_dc_ac.json'):
                (output / name).write_bytes((origin / name).read_bytes())
            report['metrics'], report['group_metrics'] = {}, {}
            bundle = dict(model=model, normalization=original['normalization'], report=report)
            restored_bundle = bundle | dict(model=restored)
            for split, values in data.items():
                pred = training.predict_paper_dc_ac(bundle, values['raw'], values['ids'], batch_size=128).argmax(1)
                reloaded = training.predict_paper_dc_ac(restored_bundle, values['raw'], values['ids'], batch_size=128).argmax(1)
                if not np.array_equal(pred, reloaded):
                    raise ValueError('Same-device checkpoint prediction differs')
                report['metrics'][split] = training.metrics(values['labels'], pred, old['classes'])
                groups = np.asarray([row['group_id'] for row in values['records']])
                report['group_metrics'][split] = {str(group): training.metrics(values['labels'][groups == group],
                    pred[groups == group], old['classes']) for group in np.unique(groups)}
                with (output / f'{split}_predictions.csv').open('w', newline='') as handle:
                    writer = csv.writer(handle, lineterminator='\n')
                    writer.writerow(['row_index', 'wave_sha256', 'group_id', 'class_index', 'predicted_class_index'])
                    for i, row, prediction in zip(values['selected'], values['records'], pred):
                        writer.writerow([i, row['wave_sha256'], row['group_id'], row['class_index'], int(prediction)])
            report['model_roundtrip_verified'] = True
            report['files_sha256'] = {name: file_sha256(output / name) for name in sorted(ARTIFACTS)}
        if (sources != source_hashes() or sources_lf != source_hashes(canonical=True) or
            any(file_sha256(origin / name) != digest for name, digest in origin_hashes.items())):
            raise ValueError('Source/origin changed during BN intervention')
        if not _same(_origin_registration(root, origin, old), registration):
            raise ValueError('Origin registration changed during BN intervention')
        report.update(status='complete', elapsed_seconds=time.perf_counter() - started)
        write_json(output / '.report.pending.json', report)
        (output / '.report.pending.json').replace(output / 'report.json')
        (output / 'running.marker').unlink()
        return report
    except BaseException as error:
        # Invalidate a published success before any fallible diagnostic write.
        # If marker removal failed, its presence also prevents loading.
        (output / 'report.json').unlink(missing_ok=True)
        failed = dict(status='failed', protocol=PROTOCOL, error_type=type(error).__name__, error=str(error))
        write_json(output / '.failure.pending.json', failed)
        (output / '.failure.pending.json').replace(output / 'failure.json')
        raise


def load_bn_model(root, run_dir, device='cpu'):
    """Verify complete sidecar, origin selection, registration and BN-only state."""
    root = Path(root).resolve()
    folder = inside(root, run_dir)
    if (folder / 'failure.json').exists() or (folder / 'running.marker').exists():
        raise ValueError('Failed or incomplete BN intervention cannot be loaded')
    training = _parent()
    report = training.json_read(folder / 'report.json')
    if (report.get('status') != 'complete' or type(report.get('schema_version')) is not int or
        report['schema_version'] != 1 or report.get('protocol') != PROTOCOL or
        report.get('heldout_test_evaluated') is not False or
        not _same(report.get('code_sha256'), source_hashes()) or
        not _same(report.get('code_sha256_lf'), source_hashes(canonical=True)) or
        set(report.get('files_sha256', {})) != ARTIFACTS):
        raise ValueError('Complete bound E2 BN report required')
    for name, digest in report['files_sha256'].items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f'BN artifact checksum differs: {name}')
    origin = inside(root, report['origin_run'])
    if report['origin_run'] != origin.relative_to(root).as_posix() or origin.is_relative_to(folder) or folder.is_relative_to(origin):
        raise ValueError('Canonical independent origin directory required')
    if (file_sha256(origin / 'report.json') != report['origin_report_sha256'] or
        file_sha256(origin / 'model.pt') != report['origin_model_sha256']):
        raise ValueError('Origin checksum differs')
    seed_everything(report['config']['seed'], threads=1)
    original = training.load_paper_dc_ac_model(origin, device='cpu')
    old = original['report']
    variable = {'protocol', 'status', 'created_utc', 'environment', 'code_sha256', 'code_sha256_lf',
                'metrics', 'group_metrics', 'files_sha256', 'elapsed_seconds'}
    additions = {'origin_run', 'origin_report_sha256', 'origin_model_sha256', 'baseline_metrics',
                 'baseline_group_metrics', 'bn_execution_registration', 'calibration', 'bn_before', 'bn_after'}
    if set(report) != set(old) | additions:
        raise ValueError('Complete parent-compatible report metadata required')
    for name in set(old) - variable:
        if not _same(report[name], old[name]):
            raise ValueError(f'Origin metadata differs: {name}')
    if (not _same(report['baseline_metrics'], old['metrics']) or
        not _same(report['baseline_group_metrics'], old['group_metrics']) or
        set(report['metrics']) != {'train', 'validation'} or
        set(report['group_metrics']) != {'train', 'validation'}):
        raise ValueError('Origin baseline/split metrics differ')
    registered = _origin_registration(root, origin, old)
    if not _same(report['bn_execution_registration'], registered):
        raise ValueError('Origin BN execution registration differs')
    for name in ('normalization_dc_ac.json', 'train_rows.json', 'validation_rows.json'):
        if file_sha256(folder / name) != old['files_sha256'][name]:
            raise ValueError(f'Origin normalization/row selection changed: {name}')
    saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
    origin_saved = torch.load(origin / 'model.pt', map_location='cpu', weights_only=True)
    if set(saved) != set(origin_saved) | {'origin_model_sha256'}:
        raise ValueError('Complete parent-compatible checkpoint metadata required')
    for name in set(origin_saved) - {'protocol', 'state_dict'}:
        if not _same(saved[name], origin_saved[name]):
            raise ValueError(f'Checkpoint origin metadata differs: {name}')
    if saved['protocol'] != PROTOCOL or saved['origin_model_sha256'] != report['origin_model_sha256']:
        raise ValueError('Checkpoint protocol/origin differs')
    if any(value.device.type != 'cpu' for value in saved['state_dict'].values()):
        raise ValueError('CPU-portable checkpoint required')
    model = copy.deepcopy(original['model']).cpu()
    expected_state = model.state_dict()
    if set(saved['state_dict']) != set(expected_state) or any(
        value.shape != expected_state[name].shape or value.dtype != expected_state[name].dtype
        for name, value in saved['state_dict'].items()):
        raise ValueError('Checkpoint state keys/shape/dtype differ')
    model.load_state_dict(saved['state_dict'], strict=True)
    changed = state_changes(original['model'], model)
    count = report['sample_counts']['train']
    expected = dict(config=fixed_config(), sample_count=count, batch_count=math.ceil(count / 128),
        last_batch_size=(count - 1) % 128 + 1, changed_buffers=changed,
        parameters_unchanged=True, original_state_unchanged=True)
    if not _same(report['calibration'], expected):
        raise ValueError('Calibration config/summary or changed BN buffers differ')
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            if int(module.num_batches_tracked) != expected['batch_count'] or (module.running_var < 0).any():
                raise ValueError('BN counter/variance differs')
            module.momentum = .01
    if (not _same(report['bn_before'], training.bn_summary(original['model'])) or
        not _same(report['bn_after'], training.bn_summary(model))):
        raise ValueError('BN buffer summaries differ')
    seed_everything(report['config']['seed'], threads=1)
    model.to(resolve_device(device)).eval()
    return dict(model=model, normalization=report['normalization'], report=report)
