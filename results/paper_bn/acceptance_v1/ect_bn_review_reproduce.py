import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path('/home/lucky/devdir/rgzndl')
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.ect.audit import waveform_hash
from src.ect.prepare import prepare_dataset
from src.ect.paper_training import train_paper_baseline
from src.ect import paper_bn as bn

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixture(root):
    raw = root/'data/raw'
    raw.mkdir(parents=True)
    data = np.empty((3,1,1,20,2,1250,2), dtype=np.float32)
    t = np.arange(1250)/2500
    for person, offset in enumerate((10,1000,2000)):
        for repeat in range(20):
            for cls in range(2):
                data[person,0,0,repeat,cls,:,0] = offset+cls+repeat/32+np.sin(2*np.pi*50*t)
                data[person,0,0,repeat,cls,:,1] = 2*offset+cls+repeat/64+np.cos(2*np.pi*50*t)
    p=raw/'fixture.npy'
    np.save(p,data)
    folder=root/'manifests/fixture'
    folder.mkdir(parents=True)
    summary=dict(schema_version=1,protocol='fixture',class_mapping_status='unverified',
        inputs={'train':dict(path='data/raw/fixture.npy',sha256=sha(p))},
        manifest_sha256={},sample_counts={},class_counts={},components=[[f'train:{p}'] for p in range(3)])
    for person, split in enumerate(('train','validation','test')):
        rows=[]
        for repeat in range(20):
            for cls in range(2):
                origin=dict(source='train',person=person,angle=0,direction=0,repeat=repeat,class_index=cls)
                rows.append(dict(split=split,group_id=person,class_index=cls,
                    wave_sha256=waveform_hash(data[person,0,0,repeat,cls]),representative=origin,origins=[origin]))
        p=folder/f'{split}.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        summary['manifest_sha256'][p.name]=sha(p)
        summary['sample_counts'][split]=40
        summary['class_counts'][split]={'0':20,'1':20}
    (folder/'summary.json').write_text(json.dumps(summary))
    prepare_dataset(root,folder,'cache')
    cfg=json.loads((ROOT/'config/paper_baseline_v1.json').read_text())
    cfg.update(purpose='paper_runner_smoke_validation_only',seed=0,device='cpu',max_epochs=2,
        batch_size=40,validation_every_epochs=1,cache_dir='cache')
    train_paper_baseline(root,cfg,'origin')

torch.set_num_threads(1)
root=Path(tempfile.mkdtemp(prefix='ect_bn_review_fixture_'))
fixture(root)
(root/'evidence').mkdir()
origin_before={p.name:sha(p) for p in (root/'origin').iterdir() if p.is_file()}
for device in ('cpu','cuda'):
    # Both calibration runs execute on CPU. Only the report device is relabelled.
    report=bn.run_recalibration(root,'origin',device,device='cpu',smoke=True)
    if device=='cuda':
        report['environment']['device']='cuda'
        report['environment']['device_name']='FAKE CUDA: CPU fixture was actually used'
        bn.write_json(root/device/'report.json',report)
    proof=dict(status='verified',device=device,report_sha256=sha(root/device/'report.json'),
        prediction_counts={'train':40,'validation':40},maximum_metric_error=0,
        same_device_calibration_exact=True)
    bn.write_json(root/'evidence'/f'{device}_proof.json',proof)
transfer=dict(status='verified',device='cpu',report_sha256=sha(root/'cuda/report.json'),
    prediction_counts={'train':40,'validation':40},maximum_metric_error=0)
bn.write_json(root/'evidence/transfer.json',transfer)
files={}
for device in ('cpu','cuda'):
    for p in (root/device/'report.json',root/'evidence'/f'{device}_proof.json',*(root/device/n for n in bn.ARTIFACTS)):
        files[p.relative_to(root).as_posix()]=sha(p)
files['evidence/transfer.json']=sha(root/'evidence/transfer.json')
a=dict(schema_version=1,protocol='paper_bn_acceptance_v1',status='verified',
    code_sha256=bn.source_hashes(),files_sha256=files,
    runs=[dict(device=d,run_dir=d,verification_path=f'evidence/{d}_proof.json') for d in ('cpu','cuda')],
    cpu_transfer_verification_path='evidence/transfer.json')
bn.write_json(root/'evidence/acceptance.json',a)
accepted=bn._acceptance(root,'evidence/acceptance.json')
assert accepted==a
out=dict(fixture_root=str(root),acceptance_without_independent_verifier_accepted=True,
    actual_calibration_device_for_both_runs='cpu',proof_fields=sorted(transfer),
    original_files_unchanged=origin_before=={p.name:sha(p) for p in (root/'origin').iterdir() if p.is_file()},
    producer_sha256=sha(ROOT/'src/ect/paper_bn.py'))
# A failure marker beside the acceptance is not consulted either.
bn.write_json(root/'evidence/failure.json',dict(status='failed',error='fixture acceptance failed'))
out['failed_acceptance_parent_accepted']=bn._acceptance(root,'evidence/acceptance.json')==a
out['failure_marker_path']='evidence/failure.json'
Path('/tmp/ect_bn_review_reproduction.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
