"""Freeze full-data training configs only after the fixed test-side data audits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.ect.deep import CODE_FILES, _validate_config
from src.ect.integrity import file_sha256
from src.ect.prepare import PreparedDataset

FAMILIES = [{'id':'cnn_clean','model':'cnn','augmentation':False},
            {'id':'resnet_clean','model':'resnet','augmentation':False},
            {'id':'resnet_aug','model':'resnet','augmentation':True}]


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Experiment path must stay within project root')
    return path


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', newline='\n')


def register(project_root, plan_path='config/experiment_v1.json',
             output_dir='results/experiments/formal_v1', cache_dir='data/processed/grouped_v1'):
    root = Path(project_root).resolve()
    output = _inside(root, output_dir)
    if output.exists():
        raise FileExistsError('Registration directory exists; never overwrite a frozen protocol')
    plan_file = _inside(root, plan_path)
    plan = json.loads(plan_file.read_text())
    if (plan.get('schema_version') != 1 or plan.get('protocol') != 'formal_v1'
            or plan.get('training_seeds') != [0,1,2] or plan.get('families') != FAMILIES
            or plan.get('validation_noise_seeds') != list(range(10,15))
            or plan.get('test_noise_seeds') != list(range(100,105)) or plan.get('noise_snr_db') != [30,20,10]):
        raise ValueError('Fixed formal_v1 protocol required')
    template_file = _inside(root, plan['template_config'])
    base = json.loads(template_file.read_text())
    _validate_config(base)
    frozen = dict(purpose='development_validation_only', train_per_class=None, validation_per_class=None,
                  max_epochs=30, patience=8, batch_size=128, learning_rate=.001, weight_decay=.0001,
                  sampling_seed=20261001, device='auto', threads=1)
    if any(base[k] != value for k, value in frozen.items()):
        raise ValueError('D011 full-data training parameters must remain frozen')
    code_paths = [f'src/ect/{name}' for name in CODE_FILES] + ['src/ect/similarity.py', 'src/ect/features.py',
        'scripts/register_experiments.py', 'scripts/diagnose_similarity.py', 'scripts/train_deep.py']
    code_paths = list(dict.fromkeys(code_paths))
    code_hashes = {p:file_sha256(_inside(root,p)) for p in code_paths}
    audits = {}
    with PreparedDataset(root, cache_dir) as store:
        meta = store.metadata
        if meta['protocol'] != plan['data_protocol']:
            raise ValueError('Dataset protocol differs from registration')
        metadata_digest = file_sha256(store.folder / 'metadata.json')
        seen = set()
        for relative in plan['audit_reports']:
            path = _inside(root, relative)
            audit = json.loads(path.read_text())
            reference = audit.get('reference_split')
            if audit.get('query_split') != 'test' or reference not in ('train','validation') or reference in seen:
                raise ValueError('Exactly test-to-train and test-to-validation audits required')
            seen.add(reference)
            expected_manifests = {f'{s}.jsonl':meta['manifest_sha256'][f'{s}.jsonl'] for s in ('test',reference)}
            expected_cache = {f'{s}_{k}.npy':meta['files_sha256'][f'{s}_{k}.npy'] for s in ('test',reference) for k in ('x','y')}
            if (audit.get('cache_metadata_sha256') != metadata_digest
                    or audit.get('schema_version') != 1 or audit.get('summary_sha256') != meta['summary_sha256']
                    or audit.get('raw_inputs_declared_by_cache') != meta['inputs']
                    or audit.get('manifest_sha256') != expected_manifests
                    or audit.get('cache_files_sha256') != expected_cache
                    or audit.get('thresholds') != {'raw':.001,'shape':.01}
                    or audit.get('query_count') != meta['sample_counts']['test']
                    or audit.get('reference_count') != meta['sample_counts'][reference]
                    or audit.get('exhaustive_pair_count') != meta['sample_counts']['test'] * meta['sample_counts'][reference]
                    or any(audit.get('code_sha256',{}).get(p) != code_hashes[p]
                           for p in ('src/ect/similarity.py','scripts/diagnose_similarity.py'))):
                raise ValueError('Audit data identity, algorithm or frozen thresholds differ')
            for kind in ('raw','shape'):
                if audit[kind]['candidate_pair_count'] != 0 or audit[kind]['candidate_query_count'] != 0:
                    raise ValueError('Near-duplicate candidates require documented review before registration')
            if (path.parent / 'failure.json').exists():
                raise ValueError('Failed data audit cannot authorize registration')
            if audit.get('figure_created') is not False or set(audit.get('artifacts_sha256',{})) != {'nearest.csv'}:
                raise ValueError('Complete candidate-free audit artifact table required')
            for filename, digest in audit['artifacts_sha256'].items():
                if Path(filename).name != filename or file_sha256(path.parent / filename) != digest:
                    raise ValueError('Audit artifact checksum mismatch')
            audits[relative] = file_sha256(path)
        if seen != {'train','validation'}:
            raise ValueError('Both test-side audits required')
        baseline_file = _inside(root, plan['baseline_report'])
        baseline = json.loads(baseline_file.read_text())
        if (baseline.get('status') != 'complete' or baseline.get('heldout_test_evaluated') is not False
                or baseline.get('cache_metadata_sha256') != metadata_digest
                or baseline.get('manifest_sha256') != meta['manifest_sha256']
                or baseline.get('sample_counts') != {s:meta['sample_counts'][s] for s in ('train','validation')}):
            raise ValueError('Complete full-data validation-only SVM baseline required')
        baseline_artifacts = {'model.joblib','train_rows.json','validation_predictions.csv'}
        if (baseline_file.parent/'failure.json').exists() or set(baseline.get('files_sha256',{})) != baseline_artifacts:
            raise ValueError('Complete nonfailed SVM artifact table required')
        for name, digest in baseline['files_sha256'].items():
            try:
                actual = file_sha256(baseline_file.parent/name)
            except OSError as error:
                raise ValueError(f'SVM artifact missing/unreadable: {name}') from error
            if actual != digest:
                raise ValueError(f'SVM artifact checksum mismatch: {name}')
        registry = dict(schema_version=1, protocol='formal_v1',
            registered_utc=datetime.now(timezone.utc).isoformat(), status='frozen_training_registration',
            test_classification_evaluated=False, test_signals_used_for_data_audit=True,
            plan=plan, plan_file_sha256=file_sha256(plan_file), template_file_sha256=file_sha256(template_file),
            data_protocol=meta['protocol'], sample_counts=meta['sample_counts'],
            manifest_sha256=meta['manifest_sha256'], cache_metadata_sha256=metadata_digest,
            raw_input_sha256={k:v['sha256'] for k,v in meta['inputs'].items()},
            audit_reports_sha256=audits, baseline_report_sha256=file_sha256(baseline_file),
            class_mapping_file_sha256=file_sha256(root / 'config/class_mapping.json'), class_mapping_status='unverified',
            code_sha256=code_hashes, runs=[],
            epoch_selection=dict(clean='clean validation macro_f1 then accuracy then earlier epoch',
                augmented='equal clean/30/20/10 dB validation macro_f1 then accuracy then earlier epoch',
                validation_noise_seed=10, validation_namespace='validation'),
            repeated_evaluation=dict(validation_noise_seeds=list(range(10,15)), test_noise_seeds=list(range(100,105)),
                namespace_by_split={'validation':'validation','test':'test'}, conditions=['clean',30,20,10]),
            interpretation=['No evidence of source independence beyond the stated hashes/groups/distance tests.',
                'Augmented versus clean ResNet differs in training augmentation AND epoch selection objective.',
                'Training seeds are replicates on a fixed split, not a population confidence interval.',
                'SVM reused from fixed nine-candidate selection; deterministic repeats are not independent training.'],
            test_release_gate='All nine full-data runs verified; protocol frozen; no tuning based on test classifications.')
    # Atomic directory ownership; registry is published only after its configs.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix='.registration-', dir=output) as tmp:
            stage = Path(tmp)
            configs = stage / 'configs'
            configs.mkdir()
            for family in FAMILIES:
                for seed in plan['training_seeds']:
                    run_id = f"{family['id']}_s{seed}"
                    config = base | {'model':family['model'],'augmentation':family['augmentation'],'seed':seed}
                    _validate_config(config)
                    path = configs / f'{run_id}.json'
                    _json(path, config)
                    registry['runs'].append(dict(run_id=run_id, family=family['id'], seed=seed,
                        config_path=str((output/'configs'/path.name).relative_to(root)),
                        config_file_sha256=file_sha256(path),
                        resolved_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                        output_dir=str((output/run_id).relative_to(root)),status_at_registration='registered'))
            _json(stage / 'registry.json', registry)
            configs.rename(output/'configs')
            (stage/'registry.json').rename(output/'registry.json')
    except BaseException as error:
        (output/'registry.json').unlink(missing_ok=True)
        _json(output/'failure.json', {'status':'failed','error':str(error)})
        raise
    return registry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=PROJECT_ROOT)
    parser.add_argument('--plan',default='config/experiment_v1.json')
    parser.add_argument('--cache',default='data/processed/grouped_v1')
    parser.add_argument('--output',default='results/experiments/formal_v1')
    args=parser.parse_args()
    registry=register(args.project_root,args.plan,args.output,args.cache)
    print(json.dumps({'status':registry['status'],'runs':len(registry['runs']),
                      'test_classification_evaluated':False},indent=2))


if __name__=='__main__':
    main()
