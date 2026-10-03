"""Freeze E2 source, train-fitted scales and matched three-seed configurations."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .integrity import file_sha256
from .prepare import PreparedDataset
from .paper_registration import _metrics_agree

BASE = 'config/paper_dc_ac_v1.json'
DESIGN = 'docs/plans/2026-10-03-dc-ac-training.md'
OLD_REGISTRY = 'results/experiments/paper_baseline_v1/registry.json'
OLD_SHA = '791e85504ff55f91d829ed94464e81ae53789186a0879aa27c188c7b14cc77c4'
CHECKS = {'cpu_training','cuda_training','independent_inference','cross_device_reload',
          'full_regression','independent_code_review','fixed_bn_ablation'}


def inside(root, path):
    root=Path(root).resolve()
    target=Path(path)
    target=(target if target.is_absolute() else root/target).resolve()
    if target==root or not target.is_relative_to(root):
        raise ValueError('Evidence paths must remain inside project root')
    return target


def read(path):
    from scripts.verify_paper_run import read_json
    try:
        result=read_json(path)
    except OSError as error:
        raise ValueError(f'Missing evidence: {path}') from error
    if not isinstance(result,dict): raise ValueError('Evidence must be JSON object')
    return result


def not_failed(root, path):
    parent=inside(root,path).parent
    while parent.is_relative_to(root):
        if (parent/'failure.json').exists(): raise ValueError(f'Failed evidence: {parent}')
        if (parent/'running.marker').exists(): raise ValueError(f'Running evidence: {parent}')
        if parent==root: break
        parent=parent.parent


def validate_files(root, table):
    if not isinstance(table,dict) or not table: raise ValueError('Full evidence checksum table required')
    for label,digest in table.items():
        path=inside(root,label)
        if Path(label).is_absolute() or str(path.relative_to(root))!=label:
            raise ValueError('Canonical root-relative checksum path required')
        not_failed(root,path)
        if not isinstance(digest,str) or len(digest)!=64 or file_sha256(path)!=digest:
            raise ValueError(f'Evidence checksum mismatch: {label}')
    return table


def write_json(path,value):
    path=Path(path)
    pending=path.with_name('.'+path.name+'.pending')
    with pending.open('x',newline='\n') as handle: handle.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    pending.replace(path)


def calibration_config():
    return dict(calibration_batch_size=128,calibration_seed=0,calibration_epoch=0,
        order='manifest',passes=1,evaluation_batch_size=128,evaluation_seed=10,
        threads=1,deterministic=True,tf32=False,test_classification_enabled=False)


PROOF_CHECKS = {'independent_statistics','full_clean_train_fit_verified','full250_transform_before_crop',
    'initial_parameters_verified','independent_inference','both_splits_inferred','all_row_csv_identities_verified',
    'prediction_classes_exact','metrics_verified','group_class_metrics_verified','source_verified',
    'complete_artifacts_verified','cache_manifest_provenance_verified','checkpoint_schema_verified',
    'history_selection_verified','original_run_read_only'}
BN_CHECKS = {'independent_calibration','calibration_train_only','calibration_crop_order_verified',
    'bn_buffer_only_changes','parameters_unchanged','original_state_unchanged','calibration_buffers_verified',
    'origin_run_verified'}


def validate_proof_contract(root,proof,report,run,device,files,bn=False):
    required=PROOF_CHECKS | (BN_CHECKS if bn else set())
    flags=proof.get('checks')
    if (not isinstance(flags,dict) or not required.issubset(flags) or
        any(flags.get(name) is not True for name in required) or
        flags.get('test_classification_evaluated') is not False):
        raise ValueError('Complete independent proof checks required')
    if (proof.get('schema_version')!=1 or proof.get('status')!='verified' or
        proof.get('protocol')!=report.get('protocol') or proof.get('heldout_test_evaluated') is not False or
        proof.get('run_dir')!=run.relative_to(root).as_posix() or proof.get('device')!=device or
        proof.get('report_sha256')!=file_sha256(run/'report.json') or
        proof.get('weights_sha256')!=report['files_sha256']['model.pt'] or
        proof.get('artifacts_sha256')!=report['files_sha256'] or
        proof.get('source_sha256')!=report['code_sha256'] or
        proof.get('cache_metadata_sha256')!=report['cache_metadata_sha256'] or
        proof.get('normalization_sha256')!=report['normalization_sha256'] or
        proof.get('prediction_counts')!=report['sample_counts'] or
        proof.get('config')!=report['config'] or proof.get('selected_epoch')!=report['selected_epoch'] or
        proof.get('initial_parameters_sha256')!=report['initial_parameters_sha256'] or
        proof.get('execution_registration')!=report['execution_registration'] or
        not _metrics_agree(proof.get('metrics'),report['metrics']) or
        not _metrics_agree(proof.get('group_metrics'),report['group_metrics'])):
        raise ValueError('Independent proof identity or metrics differs')
    if (type(proof.get('maximum_metric_error')) not in (int,float) or
        not math.isfinite(proof['maximum_metric_error']) or not 0<=proof['maximum_metric_error']<=1e-12 or
        proof.get('metric_tolerance')!=1e-12 or proof.get('statistics_tolerance')!={'rtol':1e-10,'atol':1e-12} or
        type(proof.get('maximum_statistics_error')) not in (int,float) or
        not math.isfinite(proof['maximum_statistics_error']) or proof['maximum_statistics_error']<0):
        raise ValueError('Independent proof tolerances or numeric errors invalid')
    script=proof.get('script',{})
    if script.get('sha256')!=file_sha256(root/'scripts/verify_paper_dc_ac.py') or script.get('text')!=(root/'scripts/verify_paper_dc_ac.py').read_text():
        raise ValueError('Independent verifier source differs')
    checked=proof.get('checked_files_sha256')
    if not isinstance(checked,dict) or not checked:raise ValueError('Full independently checked file chain required')
    for label,digest in checked.items():
        path=inside(root,label)
        if files.get(path.relative_to(root).as_posix())!=digest or file_sha256(path)!=digest:
            raise ValueError('Pruned or changed independent proof file chain')
    same=report['environment']['device']==device
    if proof.get('same_device') is not same or proof.get('allow_device_change') is not (not same):
        raise ValueError('Actual independent inference device identity differs')
    if bn and (proof.get('calibration')!=report['calibration'] or
        proof.get('origin_run')!=report['origin_run'] or
        proof.get('origin_report_sha256')!=report['origin_report_sha256'] or
        proof.get('origin_model_sha256')!=report['origin_model_sha256'] or
        (same and proof.get('same_device_calibration_exact') is not True)):
        raise ValueError('Fixed BN independent intervention identity differs')


def check_acceptance(root, path):
    from . import paper_dc_ac_training as training
    acceptance=read(inside(root,path))
    if (acceptance.get('schema_version')!=1 or acceptance.get('status')!='accepted' or
        acceptance.get('protocol')!='paper_dc_ac_finite_v1' or
        acceptance.get('checks')!={name:True for name in CHECKS} or
        acceptance.get('test_classification_enabled') is not False or
        acceptance.get('code_sha256')!=training.source_hashes()):
        raise ValueError('Complete actual E2 acceptance required')
    files=validate_files(root,acceptance.get('files_sha256'))
    runs=acceptance.get('runs')
    if not isinstance(runs,list) or len(runs)!=2 or {r.get('device') for r in runs}!={'cpu','cuda'}:
        raise ValueError('Actual CPU/CUDA smoke runs required')
    for item in runs:
        run=inside(root,item['run_dir'])
        proof_path=inside(root,item['verification_path'])
        report=training.load_paper_dc_ac_model(run,'cpu')['report']
        proof=read(proof_path)
        for p in (run/'report.json',proof_path,*[run/n for n in training.ARTIFACTS]):
            if files.get(p.relative_to(root).as_posix())!=file_sha256(p):
                raise ValueError('Smoke artifact/proof chain incomplete')
        cfg=report['config']
        if (cfg['purpose']!=training.SMOKE_PURPOSE or cfg['seed']!=0 or cfg['max_epochs']!=2 or
            cfg['train_per_class']!=2 or cfg['validation_per_class']!=2 or
            report['environment']['device']!=item['device'] or report['parameter_update_l2']<=0 or
            report.get('heldout_test_evaluated') is not False or report['sample_counts']!={'train':40,'validation':40}):
            raise ValueError('Actual 20-class CPU/CUDA minimal learning required')
        validate_proof_contract(root,proof,report,run,item['device'],files)
    for key in ('cross_device_verification_path','bn_cross_device_verification_path','regression_path','review_path'):
        evidence=inside(root,acceptance.get(key,''))
        if files.get(evidence.relative_to(root).as_posix())!=file_sha256(evidence):
            raise ValueError('Actual cross-device/regression/review chain required')
        record=read(evidence)
        if record.get('status') not in ('verified','passed','accepted'):
            raise ValueError(f'Unsuccessful acceptance step: {key}')
        if record.get('source_sha256',record.get('code_sha256'))!=training.source_hashes():
            raise ValueError('Acceptance source identity differs')
    cross=read(inside(root,acceptance['cross_device_verification_path']))
    if (cross.get('device')!='cpu' or cross.get('report_sha256')!=file_sha256(inside(root,next(r['run_dir'] for r in runs if r['device']=='cuda'))/'report.json') or
        cross.get('prediction_counts')!={'train':40,'validation':40}):
        raise ValueError('Actual CUDA-to-CPU independent reload required')
    cuda_run=inside(root,next(r['run_dir'] for r in runs if r['device']=='cuda'))
    validate_proof_contract(root,cross,training.load_paper_dc_ac_model(cuda_run,'cpu')['report'],cuda_run,'cpu',files)
    bn_runs=acceptance.get('bn_runs')
    if not isinstance(bn_runs,list) or len(bn_runs)!=2 or {r.get('device') for r in bn_runs}!={'cpu','cuda'}:
        raise ValueError('Actual CPU/CUDA fixed BN ablation proofs required')
    from .paper_dc_ac_bn import load_bn_model
    for item in bn_runs:
        run=inside(root,item['run_dir']);proof_path=inside(root,item['verification_path'])
        report=load_bn_model(root,run,'cpu')['report']
        for p in (run/'report.json',proof_path,*[run/n for n in training.ARTIFACTS]):
            if files.get(p.relative_to(root).as_posix())!=file_sha256(p):
                raise ValueError('BN smoke artifact/proof chain incomplete')
        if report['origin_run']!=next(r['run_dir'] for r in runs if r['device']==item['device']):
            raise ValueError('BN smoke must use matched accepted origin')
        validate_proof_contract(root,read(proof_path),report,run,item['device'],files,bn=True)
    bn_cuda=inside(root,next(r['run_dir'] for r in bn_runs if r['device']=='cuda'))
    validate_proof_contract(root,read(inside(root,acceptance['bn_cross_device_verification_path'])),
        load_bn_model(root,bn_cuda,'cpu')['report'],bn_cuda,'cpu',files,bn=True)
    regression=read(inside(root,acceptance['regression_path']))
    if regression.get('exit_code')!=0 or regression.get('skipped')!=0 or regression.get('cuda_available') is not True:
        raise ValueError('Real GPU regression without skips required')
    for label,digest in regression.get('files_sha256',{}).items():
        if files.get(label)!=digest: raise ValueError('Regression log chain incomplete')
    review=read(inside(root,acceptance['review_path']))
    if review.get('unresolved_findings')!=[]: raise ValueError('Code review findings remain unresolved')
    return acceptance


def register_experiment(project_root, output_dir, acceptance_path):
    from . import paper_dc_ac_training as training
    root=Path(project_root).resolve()
    acceptance_path=inside(root,acceptance_path)
    acceptance=check_acceptance(root,acceptance_path)
    if file_sha256(root/OLD_REGISTRY)!=OLD_SHA: raise ValueError('Baseline registry identity differs')
    old=read(root/OLD_REGISTRY)
    validate_files(root,old['code_sha256'])
    base=read(root/BASE)
    training.validate_config(base|dict(seed=0))
    output=inside(root,output_dir)
    if output.exists(): raise FileExistsError(output)
    cache=inside(root,base['cache_dir'])
    with PreparedDataset(root,cache) as store:
        if store.metadata['sample_counts']['train']!=24000 or store.metadata['sample_counts']['validation']!=3200:
            raise ValueError('Full grouped_v1 counts required')
        fitted=training.fitted_normalization(root,cache,store)
        if fitted['statistics']['sample_count']!=24000: raise ValueError('Full train normalization fit required')
        identity=dict(cache_dir=base['cache_dir'],cache_metadata_sha256=file_sha256(cache/'metadata.json'),
            manifest_sha256=store.metadata['manifest_sha256'],raw_input_sha256={k:v['sha256'] for k,v in store.metadata['inputs'].items()})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.mkdir()
    try:
        write_json(output/'normalization_dc_ac.json',fitted)
        registry=dict(schema_version=1,protocol=base['protocol'],status='frozen_execution_registration',
            registered_utc=datetime.now(timezone.utc).isoformat(),registration_dir=output.relative_to(root).as_posix(),
            test_classification_enabled=False,base_config_path=BASE,base_config_sha256=file_sha256(root/BASE),
            design_path=DESIGN,design_sha256=file_sha256(root/DESIGN),
            baseline_registry_path=OLD_REGISTRY,baseline_registry_sha256=OLD_SHA,
            acceptance_path=acceptance_path.relative_to(root).as_posix(),acceptance_sha256=file_sha256(acceptance_path),
            acceptance_files_sha256=acceptance['files_sha256'],code_sha256=training.source_hashes(),
            code_sha256_lf=training.source_hashes(canonical=True),statistics_path=(output/'normalization_dc_ac.json').relative_to(root).as_posix(),
            statistics_sha256=file_sha256(output/'normalization_dc_ac.json'),bn_calibration=calibration_config(),runs=[],**identity)
        (output/'configs').mkdir()
        for seed in (0,1,2):
            config=base|dict(seed=seed)
            config_path=output/'configs'/f'resnext_s{seed}.json'
            write_json(config_path,config)
            registry['runs'].append(dict(run_id=f'resnext_s{seed}',seed=seed,
                config_path=config_path.relative_to(root).as_posix(),config_file_sha256=file_sha256(config_path),
                resolved_config_sha256=training.config_digest(config),
                output_dir=(output/f'resnext_s{seed}').relative_to(root).as_posix(),
                initial_parameters_sha256=training.initialization_sha256(seed)))
        write_json(output/'registry.json',registry)
    except BaseException as error:
        write_json(output/'failure.json',dict(status='failed',error=str(error)))
        raise
    return registry


def validate_registered_run(project_root, registry_path, config, cache, output):
    from . import paper_dc_ac_training as training
    root=Path(project_root).resolve()
    path=inside(root,registry_path)
    not_failed(root,path)
    registry=read(path)
    if (registry.get('schema_version')!=1 or registry.get('protocol')!=config.get('protocol') or
        registry.get('status')!='frozen_execution_registration' or
        registry.get('test_classification_enabled') is not False or
        registry.get('code_sha256')!=training.source_hashes() or
        registry.get('code_sha256_lf')!=training.source_hashes(canonical=True) or
        registry.get('bn_calibration')!=calibration_config() or
        registry.get('registration_dir')!=path.parent.relative_to(root).as_posix() or
        registry.get('baseline_registry_path')!=OLD_REGISTRY or registry.get('baseline_registry_sha256')!=OLD_SHA):
        raise ValueError('Frozen E2 registry required; source or protocol differs')
    validate_files(root,{registry[k]:registry[k.replace('_path','_sha256')] for k in
        ('base_config_path','design_path','acceptance_path','statistics_path','baseline_registry_path')})
    validate_files(root,registry['acceptance_files_sha256'])
    if registry['base_config_path']!=BASE or registry['design_path']!=DESIGN:
        raise ValueError('E2 base/design anchors differ')
    check_acceptance(root,registry['acceptance_path'])
    base=read(root/BASE)
    matches=[r for r in registry.get('runs',[]) if r.get('seed')==config.get('seed')]
    if len(registry.get('runs',[]))!=3 or {r.get('seed') for r in registry['runs']}!={0,1,2} or len(matches)!=1:
        raise ValueError('Three seed execution registration required')
    for row in registry['runs']:
        cfg=read(inside(root,row['config_path']))
        training.validate_config(cfg)
        if (cfg!=base|dict(seed=row['seed']) or file_sha256(root/row['config_path'])!=row['config_file_sha256'] or
            training.config_digest(cfg)!=row['resolved_config_sha256'] or row['initial_parameters_sha256']!=training.initialization_sha256(row['seed'])):
            raise ValueError('Per-seed frozen configuration differs')
    row=matches[0]
    if config!=read(root/row['config_path']) or inside(root,row['output_dir'])!=Path(output).resolve():
        raise ValueError('Registered config/output identity differs')
    cache=Path(cache).resolve()
    if inside(root,registry['cache_dir'])!=cache or file_sha256(cache/'metadata.json')!=registry['cache_metadata_sha256']:
        raise ValueError('Frozen cache differs')
    artifact=training.validate_normalization_artifact(read(root/registry['statistics_path']))
    if artifact['provenance']['cache_metadata_sha256']!=registry['cache_metadata_sha256']:
        raise ValueError('Frozen statistics provenance differs')
    return dict(registry_path=path.relative_to(root).as_posix(),registry_sha256=file_sha256(path),
        protocol=registry['protocol'],registered_utc=registry['registered_utc'],acceptance_sha256=registry['acceptance_sha256'],
        statistics_path=registry['statistics_path'],statistics_sha256=registry['statistics_sha256'],**row)
