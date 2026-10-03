"""Adversarial acceptance tests using genuine hashed 40/40 CPU artifacts."""
import json
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from src.ect.audit import waveform_hash
from src.ect.prepare import prepare_dataset
from src.ect.paper_training import train_paper_baseline
from src.ect import paper_bn as bn


class BNAcceptanceTests(unittest.TestCase):
    def fixture(self, root):
        raw=root/'data/raw';raw.mkdir(parents=True)
        values=np.empty((3,1,1,20,2,1250,2),dtype='f4')
        t=np.arange(1250)/2500
        for person,dc in enumerate((10,1000,2000)):
            for repeat in range(20):
                for cls in range(2):
                    values[person,0,0,repeat,cls,:,0]=dc+cls+repeat/32+np.sin(2*np.pi*50*t)
                    values[person,0,0,repeat,cls,:,1]=dc*2+cls+repeat/64+np.cos(2*np.pi*50*t)
        np.save(raw/'fixture.npy',values)
        folder=root/'manifests/fixture';folder.mkdir(parents=True)
        summary=dict(schema_version=1,protocol='fixture',class_mapping_status='unverified',
            inputs={'train':dict(path='data/raw/fixture.npy',sha256=bn.file_sha256(raw/'fixture.npy'))},
            manifest_sha256={},sample_counts={},class_counts={},components=[[f'train:{i}'] for i in range(3)])
        for person,split in enumerate(('train','validation','test')):
            records=[]
            for repeat in range(20):
                for cls in range(2):
                    origin=dict(source='train',person=person,angle=0,direction=0,repeat=repeat,class_index=cls)
                    records.append(dict(split=split,group_id=person,class_index=cls,
                        wave_sha256=waveform_hash(values[person,0,0,repeat,cls]),representative=origin,origins=[origin]))
            p=folder/f'{split}.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in records))
            summary['manifest_sha256'][p.name]=bn.file_sha256(p)
            summary['sample_counts'][split]=40;summary['class_counts'][split]={'0':20,'1':20}
        bn.write_json(folder/'summary.json',summary)
        prepare_dataset(root,folder,'cache')
        cfg=json.loads((bn.SOURCE_ROOT/'config/paper_baseline_v1.json').read_text())
        cfg.update(purpose='paper_runner_smoke_validation_only',seed=0,device='cpu',max_epochs=2,
            batch_size=40,validation_every_epochs=1,cache_dir='cache')
        train_paper_baseline(root,cfg,'origin')
        evidence=root/'evidence';evidence.mkdir()
        files={}
        for device in ('cpu','cuda'):
            report=bn.run_recalibration(root,'origin',device,device='cpu',smoke=True)
            if device=='cuda':
                report['environment']['device']='cuda';bn.write_json(root/device/'report.json',report)
            proof=dict(status='verified',device=device,report_sha256=bn.file_sha256(root/device/'report.json'),
                prediction_counts={'train':40,'validation':40},maximum_metric_error=0,same_device_calibration_exact=True)
            bn.write_json(evidence/f'{device}_proof.json',proof)
            for p in (root/device/'report.json',evidence/f'{device}_proof.json',*(root/device/n for n in bn.ARTIFACTS)):
                files[p.relative_to(root).as_posix()]=bn.file_sha256(p)
        bn.write_json(evidence/'transfer.json',dict(status='verified',device='cpu',
            report_sha256=bn.file_sha256(root/'cuda/report.json'),prediction_counts={'train':40,'validation':40},maximum_metric_error=0))
        files['evidence/transfer.json']=bn.file_sha256(evidence/'transfer.json')
        bn.write_json(evidence/'acceptance.json',dict(schema_version=1,status='verified',protocol='paper_bn_acceptance_v1',
            code_sha256=bn.source_hashes(),files_sha256=files,runs=[dict(device=d,run_dir=d,
                verification_path=f'evidence/{d}_proof.json') for d in ('cpu','cuda')],
            cpu_transfer_verification_path='evidence/transfer.json'))

    def test_claim_only_cpu_cuda_and_migration_proofs_cannot_authorize_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with self.assertRaises(ValueError): bn._acceptance(root,'evidence/acceptance.json')

    def test_failed_and_running_acceptance_evidence_is_refused_before_proof_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            for marker in ('failure.json','running.marker'):
                p=root/'evidence'/marker;p.write_text('{}')
                with self.subTest(marker=marker),self.assertRaisesRegex(ValueError,'Failed|Running'):
                    bn._acceptance(root,'evidence/acceptance.json')
                p.unlink()

    def test_actual_proof_cannot_prune_origin_sources_or_input_identity_chain(self):
        from scripts.verify_paper_bn import verify_paper_bn
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            run=root/'cpu';report=json.loads((run/'report.json').read_text())
            proof=verify_paper_bn(root,'cpu','evidence/real_cpu_proof.json',device='cpu')
            bn._actual_proof(root,proof,run,'cpu',report,True)
            minimal={str(p):bn.file_sha256(p) for p in (run/'report.json',*(run/n for n in bn.ARTIFACTS))}
            pruned=copy.deepcopy(proof);pruned['checked_files_sha256']=minimal
            with self.assertRaises(ValueError): bn._actual_proof(root,pruned,run,'cpu',report,True)
