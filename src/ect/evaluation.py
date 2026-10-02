"""Evaluate frozen full-data models without fitting or selecting anything."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .baseline import load_svm_model
from .deep import load_deep_model, predict_raw
from .devices import resolve_device, seed_everything
from .features import extract_features
from .integrity import file_sha256
from .noise import add_ac_noise
from .prepare import PreparedDataset

EVALUATION_CODE = ('src/ect/evaluation.py', 'src/ect/evaluation_summary.py',
                   'scripts/evaluate_registered.py')
FAMILIES = {'cnn_clean': ('cnn', False), 'resnet_clean': ('resnet', False),
            'resnet_aug': ('resnet', True)}


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Evaluation paths must stay within project root')
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', newline='\n')


def evaluation_code(root):
    executed_root = Path(__file__).resolve().parents[2]
    code = {p:file_sha256(executed_root/p) for p in EVALUATION_CODE}
    if any(file_sha256(inside(root,p)) != digest for p,digest in code.items()):
        raise ValueError('Project evaluation source differs from the executed implementation')
    return code


def record_failure(output, report, error, started):
    report.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started)
    # Mark failure first: if a complete report cannot be deleted, readers still reject it.
    try:
        write_json(output/'failure.json',report)
    finally:
        try:
            (output/'report.json').unlink(missing_ok=True)
        except OSError:
            pass


def count_metrics(labels, predictions, classes):
    y, p, c = np.asarray(labels), np.asarray(predictions), np.asarray(classes)
    if (y.ndim != 1 or not len(y) or p.shape != y.shape
            or y.dtype.kind not in 'iu' or p.dtype.kind not in 'iu'
            or not np.array_equal(c, np.arange(len(c)))
            or not np.isin(y, c).all() or not np.isin(p, c).all()):
        raise ValueError('Matching integer classes and contiguous class order required')
    cm = np.zeros((len(c), len(c)), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    tp, support, predicted = np.diag(cm).astype(float), cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(len(c)), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(len(c)), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(c)), where=support + predicted != 0)
    return dict(accuracy=float(tp.sum() / len(y)), macro_f1=float(f1.mean()),
        confusion_matrix=cm.tolist(), per_class={str(k): dict(precision=float(precision[k]),
            recall=float(recall[k]), **{'f1-score': float(f1[k])}, support=int(support[k])) for k in c})


def validate_acceptance(root, path, registry_sha256, code_sha256, expected_samples):
    """Require the independently verified validation output from this implementation."""
    root = Path(root).resolve()
    acceptance = json.loads(inside(root, path).read_text())
    report_path = inside(root, acceptance['evaluation_report_path'])
    if file_sha256(report_path) != acceptance.get('evaluation_report_sha256'):
        raise ValueError('Validation acceptance report checksum differs')
    report = json.loads(report_path.read_text())
    if (acceptance.get('status') != 'passed' or report.get('status') != 'complete'
            or (report_path.parent / 'failure.json').exists()
            or report.get('split') != 'validation' or report.get('heldout_test_evaluated') is not False
            or acceptance.get('test_classification_evaluated') is not False
            or acceptance.get('registry_sha256') != registry_sha256
            or report.get('registry_sha256') != registry_sha256
            or acceptance.get('evaluation_code_sha256') != code_sha256
            or report.get('code_sha256') != code_sha256
            or report.get('model_count') != 10
            or report.get('sample_count') != expected_samples
            or report.get('prediction_count') != report.get('sample_count', 0) * 10 * 16
            or acceptance.get('predictions_verified') != report.get('prediction_count')
            or set(report.get('files_sha256', {})) != {'predictions.csv', 'summary.json'}):
        raise ValueError('Complete current validation acceptance required before test classification')
    for name, digest in report['files_sha256'].items():
        if file_sha256(report_path.parent / name) != digest:
            raise ValueError('Validation acceptance artifact checksum differs')
    from .evaluation_summary import summarize_metrics
    recomputed = summarize_metrics([entry | {'metrics': {k:entry['metrics'][k]
        for k in ('accuracy','macro_f1')}} for entry in report['metrics']],range(10,15))
    if json.loads((report_path.parent/'summary.json').read_text()) != recomputed:
        raise ValueError('Validation acceptance summary is incomplete or inconsistent')
    return acceptance


def load_registered(root, registry_path, store, device):
    """Verify registry, training reports and checkpoints before any inference."""
    r = json.loads(registry_path.read_text())
    if r.get('protocol') != 'formal_v1' or r.get('status') != 'frozen_training_registration':
        raise ValueError('Frozen formal_v1 registration required')
    plan_path = root / 'config/experiment_v1.json'
    if file_sha256(plan_path) != r['plan_file_sha256'] or json.loads(plan_path.read_text()) != r['plan']:
        raise ValueError('Registered experiment plan differs')
    if (r['repeated_evaluation'] != dict(validation_noise_seeds=list(range(10,15)),
            test_noise_seeds=list(range(100,105)), namespace_by_split={'validation':'validation', 'test':'test'},
            conditions=['clean',30,20,10]) or r['plan']['training_seeds'] != [0,1,2]):
        raise ValueError('Fixed repeated evaluation protocol required')
    for name, digest in r['code_sha256'].items():
        if file_sha256(inside(root, name)) != digest:
            raise ValueError(f'Registered source differs: {name}')
    if file_sha256(inside(root, r['plan']['template_config'])) != r['template_file_sha256']:
        raise ValueError('Registered template differs')
    if file_sha256(root/'config/class_mapping.json') != r['class_mapping_file_sha256']:
        raise ValueError('Class mapping identity differs; use a new protocol for semantic evaluation')
    if (r['cache_metadata_sha256'] != file_sha256(store.folder/'metadata.json')
            or r['manifest_sha256'] != store.metadata['manifest_sha256']
            or r['sample_counts'] != store.metadata['sample_counts']
            or r['raw_input_sha256'] != {k:v['sha256'] for k,v in store.metadata['inputs'].items()}):
        raise ValueError('Registered data identity differs')
    for name, digest in r['audit_reports_sha256'].items():
        if file_sha256(inside(root,name)) != digest:
            raise ValueError('Registered test-side audit differs')
    proof_path = registry_path.parent/'training_verification.json'
    proof = json.loads(proof_path.read_text())
    if proof.get('status') != 'passed' or proof.get('registry_sha256') != file_sha256(registry_path):
        raise ValueError('Verified complete training evidence required')
    proofs = {x['run_id']:x for x in proof['runs']}
    expected = {f'{family}_s{seed}' for family in FAMILIES for seed in range(3)}
    if len(r['runs']) != 9 or {x['run_id'] for x in r['runs']} != expected or set(proofs) != expected:
        raise ValueError('Exactly nine registered and verified network runs required')
    models, bindings = [], {'training_verification.json':file_sha256(proof_path)}
    for run in r['runs']:
        config_path = inside(root,run['config_path'])
        config = json.loads(config_path.read_text())
        folder = inside(root,run['output_dir'])
        report_path = folder/'report.json'
        if file_sha256(config_path) != run['config_file_sha256'] or file_sha256(report_path) != proofs[run['run_id']]['report_sha256']:
            raise ValueError('Registered configuration or verified training report differs')
        bundle = load_deep_model(folder, device=str(device))
        report = bundle['report']
        family, seed = run['family'], run['seed']
        if (run['run_id'] != f'{family}_s{seed}' or seed not in (0,1,2) or family not in FAMILIES
                or (config['model'], config['augmentation']) != FAMILIES[family]
                or config['seed'] != seed or config != report['config']
                or config['purpose'] != 'development_validation_only'
                or config['train_per_class'] is not None or config['validation_per_class'] is not None
                or report['config_sha256'] != run['resolved_config_sha256']
                or report['normalization'] != store.stats
                or report['cache_metadata_sha256'] != r['cache_metadata_sha256']
                or report['manifest_sha256'] != r['manifest_sha256']
                or report['raw_input_sha256'] != r['raw_input_sha256']
                or report['heldout_test_evaluated'] is not False
                or report['sample_counts'] != {s:r['sample_counts'][s] for s in ('train','validation')}):
            raise ValueError('Full-data frozen checkpoint/data metadata differ')
        for split in ('train','validation'):
            if json.loads((folder/f'{split}_rows.json').read_text()) != list(range(r['sample_counts'][split])):
                raise ValueError('Full registered rows required')
        models.append(dict(run_id=run['run_id'],family=family,training_seed=seed,bundle=bundle))
        bindings[run['run_id']] = dict(report_sha256=file_sha256(report_path), files_sha256=report['files_sha256'])
    baseline_path = inside(root,r['plan']['baseline_report'])
    baseline = json.loads(baseline_path.read_text())
    if (file_sha256(baseline_path) != r['baseline_report_sha256']
            or baseline['cache_metadata_sha256'] != r['cache_metadata_sha256']
            or baseline['manifest_sha256'] != r['manifest_sha256']
            or baseline['heldout_test_evaluated'] is not False):
        raise ValueError('Registered complete SVM baseline required')
    svm = load_svm_model(baseline_path.parent)
    if not np.array_equal(svm['pipeline'].classes_, models[0]['bundle']['report']['classes']):
        raise ValueError('SVM and network class order differ')
    models.append(dict(run_id='svm',family='svm',training_seed=None,bundle=svm))
    bindings['svm'] = dict(report_sha256=file_sha256(baseline_path), files_sha256=baseline['files_sha256'])
    return r, models, bindings


def evaluate_registered(project_root, registry='results/experiments/formal_v1/registry.json',
                        split='validation', output_dir='results/evaluation/validation_v1',
                        cache_dir='data/processed/grouped_v1', device='auto', acceptance=None, progress=None):
    if split not in ('validation','test'):
        raise ValueError('Evaluation split must be validation/test')
    if split == 'test' and acceptance is None:
        raise ValueError('Validation acceptance required before test classification')
    root = Path(project_root).resolve()
    output = inside(root,output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = dict(schema_version=1,status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                  split=split,heldout_test_evaluated=split=='test')
    try:
        registry_path = inside(root,registry)
        digest = file_sha256(registry_path)
        code = evaluation_code(root)
        if split == 'test':
            frozen = json.loads(registry_path.read_text())
            validate_acceptance(root,acceptance,digest,code,frozen['sample_counts']['validation'])
            report['validation_acceptance_sha256'] = file_sha256(inside(root,acceptance))
        seed_everything(0,threads=1)
        actual_device = resolve_device(device)
        report.update(registry_path=str(registry_path.relative_to(root)),registry_sha256=digest,
            code_sha256=code,environment=dict(torch=str(torch.__version__),torch_cuda_build=torch.version.cuda,
                device=str(actual_device),device_name=torch.cuda.get_device_name(actual_device) if actual_device.type=='cuda' else 'cpu',
                deterministic_algorithms=True,tf32=False,threads=1),class_mapping_status='unverified')
        from .evaluation_summary import summarize_metrics
        with PreparedDataset(root,cache_dir) as store, threadpool_limits(limits=1):
            r, models, bindings = load_registered(root,registry_path,store,actual_device)
            if split == 'test':
                accepted = json.loads(inside(root,acceptance).read_text())
                validation_report = json.loads(inside(root,accepted['evaluation_report_path']).read_text())
                if validation_report['model_bindings'] != bindings:
                    raise ValueError('Test model weights differ from validated model weights')
            raw, labels = store.arrays[split]
            manifest = (root/store.metadata['summary_path']).parent/f'{split}.jsonl'
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            if not np.array_equal(labels,[x['class_index'] for x in records]):
                raise ValueError('Evaluation labels differ from manifest order')
            ids = [x['wave_sha256'] for x in records]
            groups = np.array([x['group_id'] for x in records])
            classes = models[0]['bundle']['report']['classes']
            if any(x['bundle']['report']['classes'] != classes for x in models if x['family'] != 'svm'):
                raise ValueError('Checkpoint class order differs')
            seeds = r['repeated_evaluation'][f'{split}_noise_seeds']
            conditions = [('clean',None,None)] + [(str(snr),snr,seed) for snr in (30,20,10) for seed in seeds]
            entries, noise_records, prediction_count = [], [], 0
            csv_path = output/'predictions.csv'
            with csv_path.open('w',newline='') as handle:
                writer = csv.writer(handle,lineterminator='\n')
                writer.writerow(['split','run_id','family','training_seed','condition','noise_seed','snr_applicable',
                                 'row_index','wave_sha256','group_id','class_index','predicted_class_index'])
                for condition,snr,seed in conditions:
                    if snr is None:
                        signals, applicable, noise_info = raw, [None]*len(raw), None
                    else:
                        signals, info = add_ac_noise(raw,ids,snr,seed,split)
                        applicable = info['snr_applicable']
                        active = np.asarray(applicable)
                        signal_power, noise_power = np.array(info['ac_power']), np.array(info['noise_power'])
                        actual_snr = 10*np.log10(signal_power[active]/noise_power[active])
                        noise_info = dict(condition=condition,noise_seed=seed,namespace=split,
                            applicable_count=int(active.sum()),not_applicable_count=int((~active).sum()),
                            max_snr_error_db=float(np.max(np.abs(actual_snr-snr))) if active.any() else None,
                            shared_raw_float64_sha256=hashlib.sha256(np.ascontiguousarray(signals,dtype='<f8').tobytes()).hexdigest())
                        noise_records.append(noise_info)
                    features = extract_features(signals)
                    for model in models:
                        if model['family']=='svm':
                            predictions = model['bundle']['pipeline'].predict(features)
                        else:
                            predictions = predict_raw(model['bundle'],signals,ids,batch_size=128)
                        entry = {k:model[k] for k in ('run_id','family','training_seed')}
                        entry.update(condition=condition,noise_seed=seed,metrics=count_metrics(labels,predictions,classes),
                            group_metrics={str(g):count_metrics(labels[groups==g],predictions[groups==g],classes) for g in np.unique(groups)})
                        entries.append(entry)
                        for i,(record,predicted) in enumerate(zip(records,predictions)):
                            writer.writerow([split,model['run_id'],model['family'],'' if model['training_seed'] is None else model['training_seed'],
                                condition,'' if seed is None else seed,'' if snr is None else applicable[i],i,
                                record['wave_sha256'],record['group_id'],record['class_index'],int(predicted)])
                        prediction_count += len(predictions)
                    if progress:
                        progress(dict(condition=condition,noise_seed=seed,predictions_written=prediction_count))
            summary = summarize_metrics([entry | {'metrics': {k:entry['metrics'][k]
                for k in ('accuracy','macro_f1')}} for entry in entries],seeds)
            write_json(output/'summary.json',summary)
            report.update(status='complete',sample_count=len(labels),model_count=len(models),prediction_count=prediction_count,
                noise_seeds=seeds,namespace=split,clean_repetitions=1,metrics=entries,noise_diagnostics=noise_records,
                model_bindings=bindings,cache_metadata_sha256=file_sha256(store.folder/'metadata.json'),
                manifest_sha256=store.metadata['manifest_sha256'],normalization_sha256=file_sha256(store.folder/'normalization.json'),
                files_sha256={n:file_sha256(output/n) for n in ('predictions.csv','summary.json')},
                elapsed_seconds=time.perf_counter()-started)
            write_json(output/'report.json',report)
        return report
    except BaseException as error:
        record_failure(output,report,error,started)
        raise
