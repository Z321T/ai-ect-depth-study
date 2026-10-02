"""Small real models and a registration-shaped identity fixture, not formal evidence."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
from fixture_data import make_dataset
from src.ect.baseline import train_svm
from src.ect.deep import train_deep, CODE_FILES
from src.ect.integrity import file_sha256
from src.ect.prepare import prepare_dataset

PROJECT = Path(__file__).resolve().parents[1]


def make_evaluation(root):
    root = Path(root)
    _, manifest = make_dataset(root)
    prepare_dataset(root,manifest,'cache')
    config_dir=root/'config'; config_dir.mkdir()
    plan=json.loads((PROJECT/'config/experiment_v1.json').read_text())
    for name in ('experiment_v1.json','deep_v1.json','class_mapping.json'):
        (config_dir/name).write_bytes((PROJECT/'config'/name).read_bytes())
    source_paths=[f'src/ect/{n}' for n in CODE_FILES]+['src/ect/features.py']
    for relative in source_paths+['src/ect/evaluation.py','src/ect/evaluation_summary.py','scripts/evaluate_registered.py']:
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((PROJECT/relative).read_bytes())
    svm_config=json.loads((PROJECT/'config/svm_v1.json').read_text())
    svm_config['candidates']=[{'kernel':'rbf','C':1.,'gamma':.1}]
    baseline=root/plan['baseline_report']
    train_svm(root,svm_config,baseline.parent,'cache')
    meta=json.loads((root/'cache/metadata.json').read_text())
    folder=root/'results/experiments/formal_v1';folder.mkdir(parents=True)
    configs=folder/'configs';configs.mkdir()
    r=dict(schema_version=1,protocol='formal_v1',status='frozen_training_registration',
        registered_utc=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),plan=plan,
        plan_file_sha256=file_sha256(config_dir/'experiment_v1.json'),
        template_file_sha256=file_sha256(config_dir/'deep_v1.json'),
        class_mapping_file_sha256=file_sha256(config_dir/'class_mapping.json'),
        code_sha256={p:file_sha256(root/p) for p in source_paths},
        cache_metadata_sha256=file_sha256(root/'cache/metadata.json'),
        manifest_sha256=meta['manifest_sha256'],sample_counts=meta['sample_counts'],
        raw_input_sha256={k:v['sha256'] for k,v in meta['inputs'].items()},
        audit_reports_sha256={},baseline_report_sha256=file_sha256(baseline),runs=[],
        repeated_evaluation=dict(validation_noise_seeds=list(range(10,15)),test_noise_seeds=list(range(100,105)),
            namespace_by_split={'validation':'validation','test':'test'},conditions=['clean',30,20,10]))
    proofs=[]
    for family,model,aug in [('cnn_clean','cnn',False),('resnet_clean','resnet',False),('resnet_aug','resnet',True)]:
        for seed in range(3):
            run_id=f'{family}_s{seed}'
            config=json.loads((config_dir/'deep_v1.json').read_text())
            config.update(model=model,augmentation=aug,seed=seed,device='cpu',max_epochs=1,batch_size=2)
            config_path=configs/f'{run_id}.json';config_path.write_text(json.dumps(config))
            output=folder/run_id;train_deep(root,config,output,'cache')
            r['runs'].append(dict(run_id=run_id,family=family,seed=seed,
                config_path=str(config_path.relative_to(root)),config_file_sha256=file_sha256(config_path),
                resolved_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                output_dir=str(output.relative_to(root))))
            proofs.append(dict(run_id=run_id,report_sha256=file_sha256(output/'report.json')))
    registry=folder/'registry.json';registry.write_text(json.dumps(r))
    (folder/'training_verification.json').write_text(json.dumps(dict(status='passed',registry_sha256=file_sha256(registry),runs=proofs)))
    return registry
