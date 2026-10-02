"""Freeze exploratory paper baselines; verify execution identity without inference.

The historical design and base JSON are immutable anchors. Runner acceptance is
supplied separately; a verified JSON flag alone is insufficient without the full
source/artifact checksum tables. No test classification is performed here.
"""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path

from .integrity import file_sha256
from .prepare import PreparedDataset

PROTOCOL = 'paper_baseline_finite_v1'
BASE_SHA256 = 'c14d18a8a451e249d2b3e31c21af9e5c3a49975576b9f3770f553c31a0dc3dc0'
DESIGN_PATH = 'results/paper_components/baseline_design_v1.json'
DESIGN_SHA256 = '341a764a31f1d9b1241244efd695b40bccfec86cd5cb6a3c24bd5548512d70c9'
COMPONENT_REPORT = 'results/paper_components/acceptance_v1/report.json'
COMPONENT_VERIFICATION = 'results/paper_components/acceptance_v1/verification.json'
CHECKS = ('cpu_training', 'cuda_training', 'independent_crop_reload', 'csv_metrics',
          'full_regression', 'independent_code_review')


def _inside(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Registration paths must stay inside project root')
    return path


def _relative(root, path):
    return _inside(root, path).relative_to(root).as_posix()


def _read(path):
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'Missing or invalid registration evidence: {path}') from error
    if not isinstance(result, dict):
        raise ValueError(f'JSON object required: {path}')
    return result


def _sha(path):
    try:
        return file_sha256(path)
    except OSError as error:
        raise ValueError(f'Missing or unreadable checksum input: {path}') from error


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _config_digest(config):
    return hashlib.sha256(_canonical(config).encode()).hexdigest()


def write_json(path, value):
    """Publish one JSON atomically; caller owns the registration directory."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.pending')
    with temporary.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _not_failed(root, path, *, allow_running_at=None):
    # Evidence may be nested under a run directory. Reject its failure marker
    # even if the specified artifact lives one or more levels below it.
    parent = _inside(root, path).parent
    while parent.is_relative_to(root):
        if (parent / 'failure.json').exists():
            raise ValueError(f'Failed run/evidence cannot authorize registration: {parent}')
        if parent != allow_running_at and (parent / 'running.marker').exists():
            raise ValueError(f'Running run/evidence cannot authorize registration: {parent}')
        if parent == root:
            break
        parent = parent.parent


def _table(value, *, nonempty=True):
    if not isinstance(value, dict) or (nonempty and not value):
        raise ValueError('Complete nonempty checksum table required')
    for label, digest in value.items():
        if (not isinstance(label, str) or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Invalid SHA256 table entry')
    return value


def _sources():
    # Local import allows paper_training to call this module without a cycle.
    runner = importlib.import_module(f'{__package__}.paper_training')
    runner.verify_imported_sources()
    names = runner.CODE_FILES
    if (not isinstance(names, tuple) or 'paper_training.py' not in names
            or len(set(names)) != len(names)):
        raise ValueError('Complete runner CODE_FILES tuple required')
    files = {}
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith('.py'):
            raise ValueError('CODE_FILES must name package Python source files')
        module = importlib.import_module(f'{__package__}.{Path(name).stem}')
        files[f'src/ect/{name}'] = Path(module.__file__).resolve()
    files['src/ect/paper_registration.py'] = Path(__file__).resolve()
    executing_root = Path(__file__).resolve().parents[2]
    for name in ('train_paper_baseline.py', 'register_paper_baseline.py','verify_paper_run.py'):
        files[f'scripts/{name}'] = executing_root / 'scripts' / name
    return runner, {label: _sha(path) for label, path in files.items()}


def _artifact_table(root, table):
    table = _table(table)
    normalized = {}
    for label, digest in table.items():
        path = _inside(root, label)
        if Path(label).is_absolute() or _relative(root, path) != label:
            raise ValueError('Evidence checksum keys must be canonical root-relative paths')
        _not_failed(root, path)
        if _sha(path) != digest:
            raise ValueError(f'Evidence checksum mismatch: {label}')
        normalized[label] = digest
    return normalized


def _metrics_agree(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(_metrics_agree(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(_metrics_agree(x,y) for x,y in zip(a,b))
    if type(a) in (int,float) and type(b) in (int,float):
        return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12
    return type(a) is type(b) and a==b


def _runner_evidence(root, acceptance, base, identity, files):
    """Bind actual CPU/CUDA smoke reports to independently reproduced results."""
    runner,_ = _sources()
    runs=acceptance.get('runs')
    if not isinstance(runs,list) or len(runs)!=2 or {r.get('device') for r in runs if isinstance(r,dict)}!={'cpu','cuda'}:
        raise ValueError('Actual CPU/CUDA smoke runner evidence required')
    expected_config=base | dict(purpose=runner.SMOKE_PURPOSE,seed=0,max_epochs=2,
        validation_every_epochs=1,train_per_class=2,validation_per_class=2)
    expected_sources=runner.source_hashes()
    verifier_file=Path(__file__).resolve().parents[2]/'scripts/verify_paper_run.py'
    required_flags=('complete_artifacts_verified','source_verified','manifest_cache_verified',
        'checkpoint_cpu_verified','history_selection_verified','all_run_row_csv_identities_verified',
        'requested_predictions_match','requested_metrics_match','requested_group_metrics_match',
        'independent_inference','original_run_read_only','both_splits_inferred')
    for item in runs:
        if set(item)!={'device','run_dir','verification_path'}:
            raise ValueError('Complete CPU/CUDA smoke evidence paths required')
        run=_inside(root,item['run_dir'])
        proof_path=_inside(root,item['verification_path'])
        _not_failed(root,run/'report.json')
        _not_failed(root,proof_path)
        required=[run/'report.json',*[run/n for n in runner.ARTIFACTS],proof_path]
        if any(files.get(_relative(root,p))!=_sha(p) for p in required):
            raise ValueError('Complete smoke report/checkpoint/prediction/proof artifact table required')
        bundle=runner.load_paper_model(run,'cpu')
        report=bundle['report']
        if (_canonical(report['config'])!=_canonical(expected_config) or report['environment']['device']!=item['device']
            or report.get('heldout_test_evaluated') is not False or report.get('model_roundtrip_verified') is not True
            or report.get('parameter_update_l2',0)<=0 or report.get('code_sha256')!=expected_sources
            or report.get('cache_metadata_sha256')!=identity['cache_metadata_sha256']
            or report.get('normalization_sha256')!=identity['normalization_sha256']
            or report.get('manifest_sha256')!=identity['manifest_sha256']
            or report.get('raw_input_sha256')!={k:v['sha256'] for k,v in identity['raw_inputs'].items()}
            or report.get('available_sample_counts')!={s:identity['sample_counts'][s] for s in ('train','validation')}):
            raise ValueError('CPU/CUDA smoke method/source/data identity differs')
        meta=_read(_inside(root,identity['cache_dir'])/'metadata.json')
        counts={s:sum(min(2,n) for n in meta['class_counts'][s].values()) for s in ('train','validation')}
        if report.get('sample_counts')!=counts:
            raise ValueError('Smoke sample coverage differs from frozen data')
        proof=_read(proof_path)
        flags=proof.get('flags',{})
        script=proof.get('script',{})
        if (type(proof.get('schema_version')) is not int or proof['schema_version']!=1 or proof.get('status')!='verified'
            or proof.get('device')!=item['device'] or proof.get('split')!='both'
            or proof.get('report_sha256')!=_sha(run/'report.json') or proof.get('weights_sha256')!=report['files_sha256']['model.pt']
            or proof.get('artifacts_sha256')!=report['files_sha256'] or proof.get('source_sha256')!=expected_sources
            or proof.get('cache_metadata_sha256')!=identity['cache_metadata_sha256']
            or proof.get('normalization_sha256')!=identity['normalization_sha256']
            or proof.get('manifest_sha256')!=identity['manifest_sha256'] or proof.get('raw_input_sha256')!=report['raw_input_sha256']
            or proof.get('prediction_counts')!=counts or proof.get('selected_epoch')!=report['selected_epoch']
            or proof.get('classes')!=report['classes'] or not _metrics_agree(proof.get('metrics'),report['metrics'])
            or not _metrics_agree(proof.get('group_metrics'),report['group_metrics'])
            or not isinstance(proof.get('maximum_metric_error'),(int,float)) or not 0<=proof['maximum_metric_error']<=1e-12
            or flags.get('test_classification_evaluated') is not False or any(flags.get(k) is not True for k in required_flags)
            or script.get('sha256')!=_sha(verifier_file) or script.get('text')!=verifier_file.read_text()):
            raise ValueError('Independent CPU/CUDA smoke proof is incomplete or unrelated')
        deterministic=proof.get('determinism',{})
        if (deterministic.get('seed')!=0 or deterministic.get('threads')!=1
            or deterministic.get('deterministic_algorithms') is not True
            or deterministic.get('cudnn_deterministic') is not True
            or deterministic.get('cudnn_benchmark') is not False or deterministic.get('tf32') is not False):
            raise ValueError('Independent smoke determinism/TF32 settings differ')
        expected_group_counts={}
        for split in ('train','validation'):
            rows=_read_rows(run/f'{split}_rows.json')
            all_records=[json.loads(line) for line in (_inside(root,identity['summary_path']).parent/f'{split}.jsonl').read_text().splitlines()]
            counts_by_group={}
            for row in rows:
                group=str(all_records[row]['group_id'])
                counts_by_group[group]=counts_by_group.get(group,0)+1
            expected_group_counts[split]=counts_by_group
        if proof.get('group_counts')!=expected_group_counts:
            raise ValueError('Smoke independent group coverage differs')


def _read_rows(path):
    rows=json.loads(path.read_text())
    if not isinstance(rows,list) or any(type(n) is not int or n<0 for n in rows):
        raise ValueError('Smoke row indices required')
    return rows


def _collect_inputs(root, config_path, cache_dir, acceptance_path):
    config_file = _inside(root, config_path)
    if _sha(config_file) != BASE_SHA256:
        raise ValueError('Original finite-budget base config SHA256 required; no tuning allowed')
    base = _read(config_file)
    cache = _inside(root, cache_dir)
    if cache != _inside(root, base['cache_dir']):
        raise ValueError('Cache path differs from original base config')
    runner, sources = _sources()
    for seed in (0, 1, 2):
        runner.validate_config(base | {'seed': seed})

    design_file = _inside(root, DESIGN_PATH)
    if _sha(design_file) != DESIGN_SHA256:
        raise ValueError('Original historical design registration checksum required')
    _not_failed(root, design_file)
    design = _read(design_file)
    if (type(design.get('schema_version')) is not int or design['schema_version'] != 1
            or design.get('protocol') != PROTOCOL
            or design.get('status') != 'design_registered_before_full_training'
            or design.get('config_path') != _relative(root, config_file)
            or design.get('config_sha256') != BASE_SHA256
            or any(design.get(key) is not False for key in
                   ('complete_training_started', 'training_runner_implemented', 'execution_registry_bound'))):
        raise ValueError('Historical design identity/config differs')
    components = _table(design.get('component_code_sha256'))
    if set(components) != {'src/ect/paper_models.py', 'src/ect/paper_crops.py'}:
        raise ValueError('Complete historical component source table required')
    if any(sources.get(key) != digest for key, digest in components.items()):
        raise ValueError('Historical component source checksum differs from executed source')

    files = {_relative(root, config_file): BASE_SHA256, DESIGN_PATH: DESIGN_SHA256}
    for relative, key in ((COMPONENT_REPORT, 'component_acceptance_report_sha256'),
                          (COMPONENT_VERIFICATION, 'component_acceptance_verification_sha256')):
        path = _inside(root, relative)
        _not_failed(root, path)
        if _sha(path) != design.get(key):
            raise ValueError('Historical component acceptance checksum mismatch')
        files[relative] = design[key]

    acceptance_file = _inside(root, acceptance_path)
    _not_failed(root, acceptance_file)
    acceptance = _read(acceptance_file)
    if (type(acceptance.get('schema_version')) is not int or acceptance['schema_version'] != 1
            or acceptance.get('status') != 'verified'
            or acceptance.get('protocol') != 'paper_baseline_runner_acceptance_v1'
            or acceptance.get('test_classification_enabled') is not False
            or not isinstance(acceptance.get('checks'), dict)
            or set(acceptance['checks']) != set(CHECKS)
            or any(acceptance['checks'][key] is not True for key in CHECKS)):
        raise ValueError('Complete verified CPU/CUDA runner acceptance required')
    # Runner and registration sources must all match the acceptance evidence.
    accepted_sources = _table(acceptance.get('code_sha256'))
    if set(accepted_sources) != set(sources):
        raise ValueError('Complete acceptance execution source table required')
    if any(sources[label] != digest for label, digest in accepted_sources.items()):
        raise ValueError('Acceptance source checksum differs from actual executed code')
    files.update(_artifact_table(root, acceptance.get('files_sha256')))
    acceptance_digest = _sha(acceptance_file)
    files[_relative(root, acceptance_file)] = acceptance_digest

    # PreparedDataset hashes all cache/manifest/raw inputs and only maps arrays
    # for integrity checking. It does not classify or iterate over test signals.
    metadata_file = cache / 'metadata.json'
    metadata = _read(metadata_file)
    _inside(root, metadata['summary_path'])
    for info in metadata['inputs'].values():
        _inside(root, info['path'])
    for name in metadata['files_sha256']:
        if Path(name).name != name:
            raise ValueError('Cache file names must stay inside cache directory')
    if set(metadata['manifest_sha256']) != {'train.jsonl', 'validation.jsonl', 'test.jsonl'}:
        raise ValueError('Complete manifest identity table required')
    try:
        with PreparedDataset(root, cache) as store:
            meta = store.metadata
            metadata_digest = _sha(metadata_file)
            if (metadata_digest != design.get('cache_metadata_sha256')
                    or meta['manifest_sha256'] != design.get('manifest_sha256')):
                raise ValueError('Cache/manifest identity differs from original design')
            summary = _inside(root, meta['summary_path'])
            files[_relative(root, metadata_file)] = metadata_digest
            files[_relative(root, summary)] = meta['summary_sha256']
            for name, digest in meta['files_sha256'].items():
                files[_relative(root, cache / name)] = digest
            for name, digest in meta['manifest_sha256'].items():
                files[_relative(root, summary.parent / name)] = digest
            for info in meta['inputs'].values():
                files[_relative(root, info['path'])] = info['sha256']
            identity = dict(cache_dir=_relative(root, cache), cache_metadata_sha256=metadata_digest,
                            cache_files_sha256=meta['files_sha256'], manifest_sha256=meta['manifest_sha256'],
                            summary_path=meta['summary_path'], summary_sha256=meta['summary_sha256'],
                            raw_inputs=meta['inputs'], sample_counts=meta['sample_counts'],
                            normalization_sha256=meta['files_sha256']['normalization.json'])
    except (OSError, KeyError, TypeError) as error:
        raise ValueError('Incomplete or unreadable prepared data identity') from error
    _runner_evidence(root,acceptance,base,identity,files)
    return base, dict(base_config_path=_relative(root, config_file), base_config_sha256=BASE_SHA256,
                      design_path=DESIGN_PATH, design_sha256=DESIGN_SHA256,
                      acceptance_path=_relative(root, acceptance_file), acceptance_sha256=acceptance_digest,
                      code_sha256=sources, data_identity=identity), files


def _run(root, output, base, seed, digest):
    config = base | {'seed': seed}
    run_id = f'resnext_s{seed}'
    return dict(run_id=run_id, seed=seed,
                config_path=_relative(root, output / 'configs' / f'{run_id}.json'),
                config_file_sha256=digest, resolved_config_sha256=_config_digest(config),
                output_dir=_relative(root, output / run_id), status_at_registration='registered')


def register_paper_baseline(project_root, output_dir, acceptance_path,
                            config_path='config/paper_baseline_v1.json',
                            cache_dir='data/processed/grouped_v1'):
    """Create immutable registry.json plus three base|seed configs, never train."""
    root = Path(project_root).resolve()
    output = _inside(root, output_dir)
    if output.exists():
        raise FileExistsError('Registration directory already exists; use a new output')
    _not_failed(root, output / 'registry.json')
    base, evidence, files = _collect_inputs(root, config_path, cache_dir, acceptance_path)
    # Preflight validation never owns an output directory. mkdir is the atomic
    # claim; losing a concurrent race must not mark somebody else's run failed.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        with (output / 'running.marker').open('x', encoding='utf-8') as marker:
            marker.write('Registration in progress; registry.json is the final commit.\n')
        (output / 'configs').mkdir()
        runs = []
        for seed in (0, 1, 2):
            path = output / 'configs' / f'resnext_s{seed}.json'
            write_json(path, base | {'seed': seed})
            run = _run(root, output, base, seed, _sha(path))
            runs.append(run)
            files[run['config_path']] = run['config_file_sha256']
        registry = dict(schema_version=1, protocol=PROTOCOL,
                        status='frozen_execution_registration',
                        registered_utc=datetime.now(timezone.utc).isoformat(),
                        registration_dir=_relative(root, output),
                        test_classification_enabled=False, **evidence,
                        files_sha256=files, runs=runs)
        # Success is the final atomic commit; no fallible writes follow it.
        (output / 'running.marker').unlink()
        write_json(output / 'registry.json', registry)
    except BaseException as error:
        # Failure is authoritative even when removing a partial success fails.
        write_json(output / 'failure.json', {'status': 'failed', 'error': str(error)})
        try:
            (output / 'registry.json').unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return registry


def validate_registered_run(project_root, registry_path, config, cache_dir, output_dir):
    """Read-only full identity check; return metadata for a training report.

    config must remain the canonical base|seed dict, including device='auto'.
    The training CLI passes device_override separately to the trainer.
    """
    root = Path(project_root).resolve()
    registry_file = _inside(root, registry_path)
    output = _inside(root, output_dir)
    _not_failed(root, registry_file)
    # The trainer revalidates before publishing success while its own marker
    # still protects unfinished artifacts. Evidence and registry markers remain
    # disallowed, and an output failure marker is always authoritative.
    _not_failed(root, output / 'report.json', allow_running_at=output)
    registry_digest = _sha(registry_file)
    registry = _read(registry_file)
    directory = registry_file.parent
    if (registry_file.name != 'registry.json'
            or type(registry.get('schema_version')) is not int or registry['schema_version'] != 1
            or registry.get('protocol') != PROTOCOL
            or registry.get('status') != 'frozen_execution_registration'
            or registry.get('test_classification_enabled') is not False
            or registry.get('registration_dir') != _relative(root, directory)
            or not isinstance(registry.get('registered_utc'), str)):
        raise ValueError('Frozen execution registration required')
    try:
        base, evidence, files = _collect_inputs(root, registry['base_config_path'], cache_dir,
                                              registry['acceptance_path'])
        if any(_canonical(registry.get(key)) != _canonical(value) for key, value in evidence.items()):
            raise ValueError('Registered execution source/evidence/data identity changed')
        runs = registry.get('runs')
        if not isinstance(runs, list) or len(runs) != 3:
            raise ValueError('Three complete immutable run configs required')
        for seed, run in enumerate(runs):
            path = directory / 'configs' / f'resnext_s{seed}.json'
            expected = _run(root, directory, base, seed, _sha(path))
            if (_canonical(run) != _canonical(expected)
                    or _canonical(_read(path)) != _canonical(base | {'seed': seed})):
                raise ValueError('Registered run config/path/output/seed identity changed')
            files[expected['config_path']] = expected['config_file_sha256']
        if _table(registry.get('files_sha256')) != files:
            raise ValueError('Complete registered file checksum table differs')
        seed = config.get('seed') if isinstance(config, dict) else None
        if type(seed) is not int or seed not in (0, 1, 2):
            raise ValueError('Registered seed 0/1/2 required')
        selected = runs[seed]
        if (_canonical(config) != _canonical(base | {'seed': seed})
                or output != _inside(root, selected['output_dir'])):
            raise ValueError('Canonical resolved config and registered run output required')
        if _sha(registry_file) != registry_digest:
            raise ValueError('Registry changed during read-only validation')
    except (KeyError, TypeError, OSError) as error:
        raise ValueError('Incomplete frozen registration identity') from error
    return dict(registry_path=_relative(root, registry_file), registry_sha256=registry_digest,
                protocol=PROTOCOL, registered_utc=registry['registered_utc'], run_id=selected['run_id'],
                seed=seed, config_path=selected['config_path'],
                config_file_sha256=selected['config_file_sha256'],
                resolved_config_sha256=selected['resolved_config_sha256'],
                output_dir=selected['output_dir'], acceptance_sha256=registry['acceptance_sha256'])
