"""One fixed train-only BN-buffer intervention on immutable paper weights.

No optimizer, weight averaging, validation fitting or checkpoint reselection.
Full experiments require a separate execution registry backed by real smoke
proofs. Old source files and their loaders remain unchanged.
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

from . import paper_training as _training
from .devices import resolve_device, seed_everything
from .integrity import file_sha256
from .paper_crops import training_crops, ten_crop_probabilities
from .prepare import PreparedDataset
from .preprocessing import standardize

PROTOCOL = 'paper_bn_recalibration_v1'
ARTIFACTS = _training.ARTIFACTS
SOURCE_ROOT = Path(__file__).resolve().parents[2]
OLD_REGISTRY = 'results/experiments/paper_baseline_v1/registry.json'
OLD_REGISTRY_SHA = '791e85504ff55f91d829ed94464e81ae53789186a0879aa27c188c7b14cc77c4'
OLD_SOURCES = tuple(json.loads((SOURCE_ROOT/OLD_REGISTRY).read_text())['code_sha256'])
CODE_FILES = OLD_SOURCES + ('src/ect/paper_bn.py', 'scripts/recalibrate_paper_bn.py',
                           'scripts/verify_paper_bn.py')
_SOURCE_BYTES = {p:(SOURCE_ROOT/p).read_bytes() for p in CODE_FILES if p!='scripts/verify_paper_bn.py'}


def fixed_config():
    return dict(calibration_batch_size=128, calibration_seed=0, calibration_epoch=0,
        order='manifest', passes=1, evaluation_batch_size=128, evaluation_seed=10,
        threads=1, deterministic=True, tf32=False, test_classification_enabled=False)


def config_matches(config):
    return (isinstance(config,dict) and
        json.dumps(config,sort_keys=True,allow_nan=False)==json.dumps(fixed_config(),sort_keys=True,allow_nan=False))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', newline='\n')


def inside(root, path):
    p=Path(path)
    p=(p if p.is_absolute() else root/p).resolve()
    if not p.is_relative_to(root) or p==root:
        raise ValueError('Path must be within project root')
    return p


def source_hashes():
    _training.verify_imported_sources()
    for name, data in _SOURCE_BYTES.items():
        if (SOURCE_ROOT/name).read_bytes()!=data:
            raise ValueError(f'Source changed after import: {name}')
    _training._verify_loaded_module(sys.modules[__name__], Path(__file__).resolve(),
                                    _SOURCE_BYTES['src/ect/paper_bn.py'])
    return {name:file_sha256(SOURCE_ROOT/name) for name in CODE_FILES}


def allowed_bn_buffers(model):
    return {f'{name}.{suffix}' if name else suffix
            for name,module in model.named_modules() if isinstance(module,torch.nn.BatchNorm1d)
            for suffix in ('running_mean','running_var','num_batches_tracked')}


def state_changes(original, updated):
    before,after=original.state_dict(),updated.state_dict()
    if set(before)!=set(after):
        raise ValueError('State keys changed')
    allowed=allowed_bn_buffers(original)
    changed=[n for n in before if not torch.equal(before[n].detach().cpu(),after[n].detach().cpu())]
    if set(changed)-allowed:
        raise ValueError('Non-BN state changed')
    for n,p in original.named_parameters():
        if not torch.equal(p.detach().cpu(),dict(updated.named_parameters())[n].detach().cpu()):
            raise ValueError('Learned parameter changed')
    if any(not torch.isfinite(v).all() for v in after.values()):
        raise ValueError('Nonfinite recalibrated state')
    return sorted(changed)


def recalibrate_bn(model, raw, identities, normalization, split='train'):
    """Copy model; one manifest-order 128-batch CMA pass, seed0/epoch0 crops."""
    if split!='train':
        raise ValueError('Calibration only accepts train')
    values=np.asarray(raw)
    ids=list(identities)
    if (values.ndim!=3 or values.shape[1:]!=(250,2) or len(values)==0 or
        len(values)!=len(ids) or any(not isinstance(v,str) or not v for v in ids) or
        not np.isfinite(values).all()):
        raise ValueError('Finite nonempty train waveforms and matching identities required')
    if not isinstance(model,torch.nn.Module):
        raise ValueError('Torch model required')
    modules=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm1d)]
    if not modules or any(not m.track_running_stats for m in modules):
        raise ValueError('Tracked BatchNorm1d required')
    if any(isinstance(m,torch.nn.modules.dropout._DropoutNd) for m in model.modules()):
        raise ValueError('Dropout is outside this frozen intervention')
    original_state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    updated=copy.deepcopy(model).train()
    bns=[m for m in updated.modules() if isinstance(m,torch.nn.BatchNorm1d)]
    momenta=[m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum=None
    device=next(updated.parameters()).device
    with torch.no_grad():
        for start in range(0,len(values),128):
            cropped=training_crops(values[start:start+128],ids[start:start+128],0,0)
            normalized=standardize(cropped,normalization)
            if not np.isfinite(normalized).all():
                raise ValueError('Nonfinite standardized calibration input')
            inputs=torch.from_numpy(np.ascontiguousarray(normalized.transpose(0,2,1))).to(device)
            logits=updated(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite calibration logits')
    for m,momentum in zip(bns,momenta):
        m.momentum=momentum
    updated.eval()
    changed=state_changes(model,updated)
    if any(not torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in original_state.items()):
        raise ValueError('Original state changed')
    batches=math.ceil(len(values)/128)
    if any(int(m.num_batches_tracked)!=batches for m in bns):
        raise ValueError('Calibration batch count differs')
    summary=dict(sample_count=len(values),batch_count=batches,last_batch_size=(len(values)-1)%128+1,
        changed_buffers=changed,parameters_unchanged=True,original_state_unchanged=True)
    return updated,summary


def evidence_not_failed(root, path):
    parent=Path(path).resolve().parent
    while parent.is_relative_to(root):
        if (parent/'failure.json').exists():
            raise ValueError(f'Failed evidence: {parent}')
        if (parent/'running.marker').exists():
            raise ValueError(f'Running evidence: {parent}')
        if parent==root:
            break
        parent=parent.parent


def _check_files(root, table):
    if not isinstance(table,dict) or not table:
        raise ValueError('Nonempty complete identity table required')
    for name,digest in table.items():
        path=inside(root,name)
        evidence_not_failed(root,path)
        if file_sha256(path)!=digest:
            raise ValueError(f'Identity checksum differs: {name}')


def _actual_proof(root, proof, run, device, report, same_device):
    required=('independent_calibration','bn_buffer_only_changes','parameters_unchanged',
        'original_state_unchanged','calibration_train_only','calibration_crop_order_verified',
        'checkpoint_inference_independent','both_splits_inferred','prediction_identities_verified',
        'metrics_verified','group_class_metrics_verified','source_snapshot_verified')
    checks=proof.get('checks',{})
    if (type(proof.get('schema_version'))!=int or proof['schema_version']!=1 or
        proof.get('protocol')!=PROTOCOL or proof.get('status')!='verified' or proof.get('device')!=device or
        proof.get('run_path')!=str(run) or proof.get('report_sha256')!=file_sha256(run/'report.json') or
        proof.get('prediction_counts')!=report['sample_counts'] or proof.get('maximum_metric_error')!=0 or
        proof.get('metrics')!=report['metrics'] or proof.get('group_metrics')!=report['group_metrics'] or
        proof.get('config')!=report['config'] or proof.get('calibration')!=report['calibration'] or
        any(checks.get(k) is not True for k in required) or checks.get('test_classification_enabled') is not False or
        type(checks.get('prediction_class_mismatches'))!=int or checks['prediction_class_mismatches']!=0):
        raise ValueError('Actual independent verifier proof required')
    script=proof.get('verification_script',{})
    expected=source_hashes()['scripts/verify_paper_bn.py']
    if (script.get('sha256')!=expected or not isinstance(script.get('text'),str) or
        hashlib.sha256(script['text'].encode('utf-8')).hexdigest()!=expected):
        raise ValueError('Independent verifier source text/hash required')
    if same_device:
        if proof.get('same_device_calibration_exact') is not True or proof.get('calibration_buffer_max_abs_error')!=0:
            raise ValueError('Exact same-device calibration proof required')
    elif (proof.get('calibration_buffer_tolerance')!={'rtol':2e-4,'atol':2e-5} or
          not isinstance(proof.get('calibration_buffer_max_abs_error'),(int,float)) or
          not np.isfinite(proof['calibration_buffer_max_abs_error'])):
        raise ValueError('Bounded cross-device calibration proof required')
    files=proof.get('checked_files_sha256')
    if not isinstance(files,dict) or not files:
        raise ValueError('Independent proof identity chain required')
    origin=inside(root,report['origin_run'])
    origin_report=json.loads((origin/'report.json').read_text())
    cache=inside(root,origin_report['cache_path'])
    metadata=json.loads((cache/'metadata.json').read_text())
    summary=inside(root,metadata['summary_path'])
    required_paths={run/'report.json',origin/'report.json',cache/'metadata.json',summary,
        SOURCE_ROOT/'config/paper_baseline_v1.json'}
    required_paths.update(run/n for n in ARTIFACTS)
    required_paths.update(origin/n for n in ARTIFACTS)
    required_paths.update(SOURCE_ROOT/n for n in CODE_FILES)
    required_paths.update(cache/n for n in metadata['files_sha256'])
    required_paths.update(summary.parent/n for n in metadata['manifest_sha256'])
    required_paths.update(inside(root,v['path']) for v in metadata['inputs'].values())
    if not {str(p) for p in required_paths}<=set(files):
        raise ValueError('Complete original/source/cache/manifest/raw proof identity coverage required')
    for p in (run/'report.json',*(run/n for n in ARTIFACTS)):
        if files.get(str(p))!=file_sha256(p):
            raise ValueError('Proof must bind actual smoke artifacts')
    permitted_sources={SOURCE_ROOT/n for n in CODE_FILES}|{SOURCE_ROOT/'config/paper_baseline_v1.json'}
    for label,digest in files.items():
        p=Path(label).resolve()
        if p.is_relative_to(root):
            evidence_not_failed(root,p)
        elif p not in permitted_sources:
            raise ValueError('Proof identity path outside project/source roots')
        if file_sha256(p)!=digest:
            raise ValueError('Independent proof identity checksum differs')


def _acceptance(root, path):
    p=inside(root,path)
    evidence_not_failed(root,p)
    a=json.loads(p.read_text())
    if (type(a.get('schema_version'))!=int or a['schema_version']!=1 or a.get('protocol')!='paper_bn_acceptance_v1' or
        a.get('status')!='verified' or a.get('code_sha256')!=source_hashes()):
        raise ValueError('Actual bound BN acceptance required')
    _check_files(root,a['files_sha256'])
    runs=a.get('runs',[])
    if len(runs)!=2 or {r['device'] for r in runs}!={'cpu','cuda'}:
        raise ValueError('CPU and actual CUDA smoke evidence required')
    for item in runs:
        run=inside(root,item['run_dir'])
        bundle=load_recalibrated_model(root,run)
        report=bundle['report']
        proof_path=inside(root,item['verification_path'])
        evidence_not_failed(root,proof_path)
        proof=json.loads(proof_path.read_text())
        if (report['purpose']!='smoke' or report['sample_counts']!={'train':40,'validation':40} or
            report['environment']['device']!=item['device'] or
            proof.get('status')!='verified' or proof.get('device')!=item['device'] or
            proof.get('prediction_counts')!=report['sample_counts'] or
            proof.get('report_sha256')!=file_sha256(run/'report.json') or
            proof.get('maximum_metric_error',1)>1e-12 or
            not proof.get('same_device_calibration_exact')):
            raise ValueError('Smoke proof/report semantics differ')
        _actual_proof(root,proof,run,item['device'],report,same_device=True)
        for evidence in (run/'report.json',proof_path,*(run/n for n in ARTIFACTS)):
            if a['files_sha256'].get(evidence.relative_to(root).as_posix())!=file_sha256(evidence):
                raise ValueError('Acceptance does not bind actual smoke artifacts')
    transferred=inside(root,a['cpu_transfer_verification_path'])
    proof=json.loads(transferred.read_text())
    cuda_run=inside(root,next(r['run_dir'] for r in runs if r['device']=='cuda'))
    if (proof.get('status')!='verified' or proof.get('device')!='cpu' or
        proof.get('report_sha256')!=file_sha256(cuda_run/'report.json') or
        proof.get('prediction_counts')!={'train':40,'validation':40} or
        proof.get('maximum_metric_error',1)>1e-12 or
        a['files_sha256'].get(transferred.relative_to(root).as_posix())!=file_sha256(transferred)):
        raise ValueError('Bound CUDA-to-CPU proof required')
    _actual_proof(root,proof,cuda_run,'cpu',json.loads((cuda_run/'report.json').read_text()),same_device=False)
    return a


def register_experiment(project_root, output_dir, acceptance_path):
    """Freeze all three original model identities before full intervention."""
    root=Path(project_root).resolve()
    folder=inside(root,output_dir)
    if folder.exists():
        raise FileExistsError(folder)
    sources=source_hashes()
    acceptance=_acceptance(root,acceptance_path)
    old=inside(root,OLD_REGISTRY)
    if file_sha256(old)!=OLD_REGISTRY_SHA:
        raise ValueError('Old immutable registry differs')
    registry=json.loads(old.read_text())
    anchors=dict(acceptance['files_sha256'])
    anchors[inside(root,acceptance_path).relative_to(root).as_posix()]=file_sha256(inside(root,acceptance_path))
    anchors[OLD_REGISTRY]=OLD_REGISTRY_SHA
    runs=[]
    for seed in (0,1,2):
        run_id=f'resnext_s{seed}'
        origin=old.parent/run_id
        loaded=_training.load_paper_model(origin)
        report=loaded['report']
        proof_path=old.parent/'verification'/f'{run_id}_cuda.json'
        proof=json.loads(proof_path.read_text())
        if (report['config']['purpose']!=_training.FULL_PURPOSE or report['config']['seed']!=seed or
            report['classes']!=list(range(20)) or
            report['sample_counts']!={'train':24000,'validation':3200} or
            proof.get('status')!='verified' or proof.get('report_sha256')!=file_sha256(origin/'report.json') or
            proof.get('prediction_counts')!=report['sample_counts'] or proof.get('maximum_metric_error',1)>1e-12):
            raise ValueError('Verified full original run required')
        for p in (origin/'report.json',proof_path,*(origin/n for n in ARTIFACTS)):
            anchors[p.relative_to(root).as_posix()]=file_sha256(p)
        with PreparedDataset(root,report['cache_path']) as store:
            cache=store.folder
            meta=store.metadata
            for p in (cache/'metadata.json',*(cache/n for n in meta['files_sha256'])):
                anchors[p.relative_to(root).as_posix()]=file_sha256(p)
            summary=root/meta['summary_path']
            for p in (summary,*(summary.parent/n for n in meta['manifest_sha256'])):
                anchors[p.relative_to(root).as_posix()]=file_sha256(p)
            for item in meta['inputs'].values():
                anchors[item['path']]=item['sha256']
        runs.append(dict(run_id=run_id,seed=seed,origin_run=origin.relative_to(root).as_posix(),
            output_dir=(folder/run_id).relative_to(root).as_posix(),
            origin_report_sha256=file_sha256(origin/'report.json'),origin_model_sha256=file_sha256(origin/'model.pt'),
            origin_verification_path=proof_path.relative_to(root).as_posix(),origin_verification_sha256=file_sha256(proof_path)))
    if sources!=source_hashes():
        raise ValueError('Source changed during registration')
    result=dict(schema_version=1,protocol=PROTOCOL,status='frozen_execution_registration',
        created_utc=datetime.now(timezone.utc).isoformat(),config=fixed_config(),
        test_classification_enabled=False,code_sha256=sources,files_sha256=anchors,
        acceptance_path=inside(root,acceptance_path).relative_to(root).as_posix(),
        acceptance_sha256=file_sha256(inside(root,acceptance_path)),runs=runs)
    folder.parent.mkdir(parents=True,exist_ok=True)
    folder.mkdir(exist_ok=False)
    write_json(folder/'registry.json',result)
    return result


def validate_registration(root, path, origin, output):
    p=inside(root,path)
    evidence_not_failed(root,p)
    registry=json.loads(p.read_text())
    if (p.name!='registry.json' or type(registry.get('schema_version'))!=int or registry['schema_version']!=1 or registry.get('protocol')!=PROTOCOL or
        registry.get('status')!='frozen_execution_registration' or not config_matches(registry.get('config')) or
        registry.get('test_classification_enabled') is not False or registry.get('code_sha256')!=source_hashes()):
        raise ValueError('Frozen BN execution registration required')
    _check_files(root,registry['files_sha256'])
    if registry['files_sha256'].get(OLD_REGISTRY)!=OLD_REGISTRY_SHA:
        raise ValueError('Original registry anchor required')
    if file_sha256(inside(root,registry['acceptance_path']))!=registry['acceptance_sha256']:
        raise ValueError('Acceptance checksum differs')
    _acceptance(root,registry['acceptance_path'])
    runs=registry.get('runs',[])
    if len(runs)!=3 or [r['seed'] for r in runs]!=[0,1,2] or any(type(r['seed'])!=int for r in runs):
        raise ValueError('All three ordered seed registrations required')
    found=[]
    for item in runs:
        expected_id=f'resnext_s{item["seed"]}'
        if (item['run_id']!=expected_id or inside(root,item['output_dir'])!=p.parent/expected_id or
            inside(root,item['origin_run'])!=inside(root,OLD_REGISTRY).parent/expected_id or
            registry['files_sha256'].get(item['origin_run']+'/report.json')!=item['origin_report_sha256'] or
            registry['files_sha256'].get(item['origin_run']+'/model.pt')!=item['origin_model_sha256'] or
            registry['files_sha256'].get(item['origin_verification_path'])!=item['origin_verification_sha256']):
            raise ValueError('Registered run identity differs')
        if inside(root,item['origin_run'])==origin and inside(root,item['output_dir'])==output:
            found.append(item)
    if len(found)!=1:
        raise ValueError('Origin/output is not a registered pair')
    return dict(registry_path=p.relative_to(root).as_posix(),registry_sha256=file_sha256(p),run_id=found[0]['run_id'])


def run_recalibration(project_root, origin_run, output_dir, device='auto', smoke=False, registry_path=None):
    root=Path(project_root).resolve()
    origin=inside(root,origin_run)
    output=inside(root,output_dir)
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(origin) or origin.is_relative_to(output):
        raise ValueError('Output must not contain or lie within origin')
    if not smoke and registry_path is None:
        raise ValueError('Full execution registration required')
    sources=source_hashes()
    original=_training.load_paper_model(origin,device=device)
    old_report=original['report']
    if smoke and old_report['purpose']!=_training.SMOKE_PURPOSE:
        raise ValueError('Smoke requires smoke origin')
    if not smoke and old_report['sample_counts']!={'train':24000,'validation':3200}:
        raise ValueError('Full train/validation counts required')
    registration=None if smoke else validate_registration(root,registry_path,origin,output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.mkdir(exist_ok=False)
    started=time.perf_counter()
    report=dict(schema_version=1,protocol=PROTOCOL,status='running',purpose='smoke' if smoke else 'full',
        created_utc=datetime.now(timezone.utc).isoformat(),config=fixed_config(),execution_registration=registration,
        origin_run=origin.relative_to(root).as_posix(),origin_report_sha256=file_sha256(origin/'report.json'),
        origin_model_sha256=file_sha256(origin/'model.pt'),code_sha256=sources,
        origin_verification_path=None if smoke else
            (origin.parent/'verification'/f'{origin.name}_cuda.json').relative_to(root).as_posix(),
        seed=old_report['config']['seed'],selected_epoch=old_report['selected_epoch'],classes=old_report['classes'],
        heldout_test_evaluated=False,class_mapping_status='unverified',
        normalization=original['normalization'],baseline_metrics=old_report['metrics'],
        baseline_group_metrics=old_report['group_metrics'],cache_path=old_report['cache_path'],
        cache_metadata_sha256=old_report['cache_metadata_sha256'],normalization_sha256=old_report['normalization_sha256'],
        manifest_sha256=old_report['manifest_sha256'],sample_counts=old_report['sample_counts'],
        available_sample_counts=old_report['available_sample_counts'])
    try:
        (output/'running.marker').write_text('Incomplete intervention; refuse loading.\n',newline='\n')
        seed_everything(report['seed'],threads=1)
        selected_device=resolve_device(device)
        report['environment']=dict(device=str(selected_device),device_name=torch.cuda.get_device_name(selected_device)
            if selected_device.type=='cuda' else 'cpu',python=platform.python_version(),numpy=np.__version__,
            torch=str(torch.__version__),torch_cuda_build=torch.version.cuda,threads=torch.get_num_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            tf32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32))
        with PreparedDataset(root,report['cache_path']) as store, threadpool_limits(limits=1):
            if (file_sha256(store.folder/'metadata.json')!=report['cache_metadata_sha256'] or
                store.stats!=report['normalization']):
                raise ValueError('Origin/cache identity differs')
            manifests=(root/store.metadata['summary_path']).parent
            records,raw,labels,ids,indices={},{},{},{},{}
            for split in ('train','validation'):
                rows=[json.loads(line) for line in (manifests/f'{split}.jsonl').read_text().splitlines()]
                x,y=store.arrays[split]
                if not np.array_equal(y,[r['class_index'] for r in rows]):
                    raise ValueError('Manifest/cache labels differ')
                selected=json.loads((origin/f'{split}_rows.json').read_text())
                if (not isinstance(selected,list) or not selected or any(type(i)!=int or not 0<=i<len(rows) for i in selected)
                    or len(set(selected))!=len(selected) or len(selected)!=report['sample_counts'][split]):
                    raise ValueError('Origin rows invalid')
                if not smoke and selected!=list(range(len(rows))):
                    raise ValueError('Full calibration/evaluation requires complete manifest order')
                indices[split]=selected
                records[split]=[rows[i] for i in selected]
                raw[split],labels[split]=x[selected],y[selected]
                ids[split]=[r['wave_sha256'] for r in records[split]]
            model,calibration=recalibrate_bn(original['model'],raw['train'],ids['train'],store.stats)
            report['calibration']=calibration
            report['metrics'],report['group_metrics']={},{}
            saved=dict(schema_version=1,protocol=PROTOCOL,config=report['config'],normalization=store.stats,
                selected_epoch=report['selected_epoch'],classes=report['classes'],origin_model_sha256=report['origin_model_sha256'],
                state_dict={n:v.detach().cpu().clone() for n,v in model.state_dict().items()})
            torch.save(saved,output/'model.pt')
            restored=copy.deepcopy(model).cpu()
            disk=torch.load(output/'model.pt',map_location='cpu',weights_only=True)
            restored.load_state_dict(disk['state_dict'],strict=True)
            if any(not torch.equal(v,disk['state_dict'][n]) for n,v in restored.state_dict().items()):
                raise ValueError('Portable state roundtrip differs')
            for split in ('train','validation'):
                pred=ten_crop_probabilities(model,raw[split],ids[split],store.stats,seed=10,batch_size=128).argmax(1)
                report['metrics'][split]=_training.metrics(labels[split],pred,report['classes'])
                groups=np.asarray([r['group_id'] for r in records[split]])
                report['group_metrics'][split]={str(g):_training.metrics(labels[split][groups==g],pred[groups==g],
                    report['classes']) for g in np.unique(groups)}
                write_json(output/f'{split}_rows.json',indices[split])
                with (output/f'{split}_predictions.csv').open('w',newline='') as handle:
                    writer=csv.writer(handle,lineterminator='\n')
                    writer.writerow(['row_index','wave_sha256','group_id','class_index','predicted_class_index'])
                    for i,r,prediction in zip(indices[split],records[split],pred):
                        writer.writerow([i,r['wave_sha256'],r['group_id'],r['class_index'],int(prediction)])
            report['files_sha256']={n:file_sha256(output/n) for n in sorted(ARTIFACTS)}
        if (sources!=source_hashes() or file_sha256(origin/'model.pt')!=report['origin_model_sha256'] or
            file_sha256(origin/'report.json')!=report['origin_report_sha256']):
            raise ValueError('Source/origin changed during experiment')
        if not smoke and validate_registration(root,registry_path,origin,output)!=registration:
            raise ValueError('Registration changed during experiment')
        report.update(status='complete',elapsed_seconds=time.perf_counter()-started)
        write_json(output/'.report.pending.json',report)
        (output/'.report.pending.json').replace(output/'report.json')
        (output/'running.marker').unlink()
        return report
    except BaseException as error:
        write_json(output/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error)))
        (output/'report.json').unlink(missing_ok=True)
        raise


def load_recalibrated_model(project_root, run_dir, device='cpu'):
    root=Path(project_root).resolve()
    folder=inside(root,run_dir)
    if (folder/'failure.json').exists() or (folder/'running.marker').exists():
        raise ValueError('Failed or incomplete recalibration cannot be loaded')
    report=json.loads((folder/'report.json').read_text())
    if (report.get('status')!='complete' or type(report.get('schema_version'))!=int or report['schema_version']!=1 or report.get('protocol')!=PROTOCOL or
        report.get('purpose') not in ('smoke','full') or report.get('heldout_test_evaluated') is not False or
        not config_matches(report.get('config')) or report.get('code_sha256')!=source_hashes() or
        type(report.get('seed'))!=int or type(report.get('selected_epoch'))!=int or
        not isinstance(report.get('classes'),list) or any(type(c)!=int for c in report['classes']) or
        set(report.get('files_sha256',{}))!=ARTIFACTS):
        raise ValueError('Complete bound recalibration report required')
    for name,digest in report['files_sha256'].items():
        if file_sha256(folder/name)!=digest:
            raise ValueError('Recalibration artifact checksum differs')
    origin=inside(root,report['origin_run'])
    if (file_sha256(origin/'report.json')!=report['origin_report_sha256'] or
        file_sha256(origin/'model.pt')!=report['origin_model_sha256']):
        raise ValueError('Origin checksum differs')
    original=_training.load_paper_model(origin,device=device)
    old=original['report']
    if (report['selected_epoch']!=old['selected_epoch'] or report['seed']!=old['config']['seed'] or
        report['classes']!=old['classes'] or report['normalization']!=original['normalization'] or
        report['sample_counts']!=old['sample_counts'] or report['baseline_metrics']!=old['metrics'] or
        report['baseline_group_metrics']!=old['group_metrics'] or
        set(report['metrics'])!={'train','validation'} or set(report['group_metrics'])!={'train','validation'}):
        raise ValueError('Origin selection/data/metrics metadata differs')
    if report['purpose']=='smoke':
        if report['execution_registration'] is not None or old['purpose']!=_training.SMOKE_PURPOSE:
            raise ValueError('Smoke origin/registration differs')
    else:
        registered=report['execution_registration']
        if not isinstance(registered,dict) or validate_registration(root,registered['registry_path'],origin,folder)!=registered:
            raise ValueError('Full registration differs')
    saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    if (type(saved.get('schema_version'))!=int or type(saved.get('selected_epoch'))!=int or
        not config_matches(saved.get('config')) or any(type(c)!=int for c in saved.get('classes',[]))):
        raise ValueError('Checkpoint metadata types differ')
    for name in ('schema_version','protocol','normalization','selected_epoch','classes','origin_model_sha256','config'):
        if saved.get(name)!=report[name]:
            raise ValueError(f'Checkpoint metadata differs: {name}')
    model=copy.deepcopy(original['model']).cpu()
    if any(v.device.type!='cpu' for v in saved['state_dict'].values()):
        raise ValueError('CPU-portable checkpoint required')
    model.load_state_dict(saved['state_dict'],strict=True)
    if state_changes(original['model'],model)!=report['calibration']['changed_buffers']:
        raise ValueError('Changed BN buffers differ')
    count=report['sample_counts']['train']
    if (report['calibration']!=dict(sample_count=count,batch_count=math.ceil(count/128),
        last_batch_size=(count-1)%128+1,changed_buffers=report['calibration']['changed_buffers'],
        parameters_unchanged=True,original_state_unchanged=True)):
        raise ValueError('Calibration metadata differs')
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm1d):
            if int(module.num_batches_tracked)!=math.ceil(count/128) or (module.running_var<0).any():
                raise ValueError('BN counters/variance differ')
    for split in ('train','validation'):
        if file_sha256(folder/f'{split}_rows.json')!=old['files_sha256'][f'{split}_rows.json']:
            raise ValueError('Evaluation/calibration row selections changed')
    seed_everything(report['seed'],threads=1)
    model.to(resolve_device(device)).eval()
    return dict(model=model,normalization=report['normalization'],report=report)
