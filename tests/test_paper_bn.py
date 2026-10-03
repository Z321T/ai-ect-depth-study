import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fixture_data import make_dataset
from src.ect.prepare import prepare_dataset, PreparedDataset
from src.ect.paper_training import train_paper_baseline
from src.ect.paper_crops import training_crops
from src.ect.preprocessing import standardize


class PaperBNTests(unittest.TestCase):
    def module(self):
        from src.ect import paper_bn
        return paper_bn

    def model(self):
        torch.manual_seed(3)
        return torch.nn.Sequential(torch.nn.Conv1d(2, 3, 1),
            torch.nn.BatchNorm1d(3, momentum=.01), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2)).eval()

    def test_only_bn_buffers_change_original_modes_gradients_and_parameters_preserved(self):
        m = self.module()
        model = self.model()
        model[2].train()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        state = copy.deepcopy(model.state_dict())
        modes = [x.training for x in model.modules()]
        raw = np.random.default_rng(12).normal(size=(129, 250, 2)).astype('f4')
        ids = [str(i) for i in range(len(raw))]
        norm = dict(mean=[0.,0.], std=[1.,1.])
        updated, summary = m.recalibrate_bn(model, raw, ids, norm)
        self.assertEqual(summary['batch_count'], 2)
        self.assertEqual(summary['last_batch_size'], 1)
        self.assertTrue(summary['parameters_unchanged'])
        self.assertTrue(summary['original_state_unchanged'])
        self.assertEqual([x.training for x in model.modules()], modes)
        for name, tensor in state.items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[name]))
        for name,p in model.named_parameters():
            self.assertTrue(torch.equal(p,dict(updated.named_parameters())[name]))
            self.assertTrue(torch.equal(p.grad,torch.ones_like(p)))
        self.assertFalse(updated.training)
        self.assertEqual(updated[1].momentum,.01)
        self.assertEqual(int(updated[1].num_batches_tracked),2)
        self.assertTrue(summary['changed_buffers'])

    def test_one_pass_equals_direct_manual_bn_updates_with_fixed_order_and_crops(self):
        m = self.module()
        original = self.model()
        raw = np.random.default_rng(1).normal(size=(257,250,2)).astype('f4')
        ids = [str(i) for i in range(len(raw))]
        norm = dict(mean=[.1,-.2],std=[2.,3.])
        expected = copy.deepcopy(original).train()
        expected[1].reset_running_stats()
        expected[1].momentum=None
        with torch.no_grad():
            for start in range(0,len(raw),128):
                cropped = training_crops(raw[start:start+128],ids[start:start+128],0,0)
                inputs = torch.from_numpy(np.ascontiguousarray(standardize(cropped,norm).transpose(0,2,1)))
                expected(inputs)
        actual, summary = m.recalibrate_bn(original,raw,ids,norm)
        for name,t in expected.state_dict().items():
            self.assertTrue(torch.equal(t,actual.state_dict()[name]),name)
        self.assertEqual(summary['batch_count'],3)

    def test_invalid_inputs_and_nontrain_or_dropout_are_rejected_without_touching_original(self):
        m = self.module()
        model = self.model()
        raw = np.ones((2,250,2),dtype='f4')
        norm = dict(mean=[0.,0.],std=[1.,1.])
        for split in ['validation','test']:
            with self.assertRaises(ValueError):
                m.recalibrate_bn(model,raw,['a','b'],norm,split=split)
        for bad, ids in [(raw, ['a']), (raw[:0], []), (raw*np.nan,['a','b'])]:
            with self.assertRaises(ValueError):
                m.recalibrate_bn(model,bad,ids,norm)
        with self.assertRaises(ValueError):
            m.recalibrate_bn(torch.nn.Sequential(torch.nn.BatchNorm1d(2),torch.nn.Dropout()),raw,['a','b'],norm)

    def fixture(self, root):
        _, manifests = make_dataset(root)
        prepare_dataset(root,manifests,'cache')
        base=json.loads((Path(__file__).resolve().parents[1]/'config/paper_baseline_v1.json').read_text())
        cfg=base|dict(purpose='paper_runner_smoke_validation_only',seed=0,device='cpu',max_epochs=2,
            batch_size=2,validation_every_epochs=1,cache_dir='cache')
        train_paper_baseline(root,cfg,'origin')

    def test_real_runner_smoke_never_classifies_test_and_reload_checks_artifacts(self):
        m=self.module()
        class Guard(dict):
            def __getitem__(self,key):
                if key=='test': raise AssertionError('test classification accessed')
                return super().__getitem__(key)
        class Sealed(PreparedDataset):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                self.arrays=Guard(self.arrays)
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            before=m.file_sha256(Path(tmp,'origin/model.pt'))
            with patch.object(m,'PreparedDataset',Sealed):
                report=m.run_recalibration(tmp,'origin','new',device='cpu',smoke=True)
            self.assertEqual(report['sample_counts'],dict(train=2,validation=2))
            self.assertFalse(report['heldout_test_evaluated'])
            self.assertEqual(report['origin_model_sha256'],before)
            self.assertEqual(m.file_sha256(Path(tmp,'origin/model.pt')),before)
            self.assertEqual(report['calibration']['sample_count'],2)
            bundle=m.load_recalibrated_model(tmp,'new')
            self.assertFalse(bundle['model'].training)
            with self.assertRaises(FileExistsError):
                m.run_recalibration(tmp,'origin','new',smoke=True)
            with Path(tmp,'new/model.pt').open('ab') as handle: handle.write(b'bad')
            with self.assertRaises(ValueError): m.load_recalibrated_model(tmp,'new')

    def test_full_without_registration_does_not_create_output_and_smoke_requires_smoke_origin(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            with self.assertRaisesRegex(ValueError,'registration'):
                m.run_recalibration(tmp,'origin','new')
            self.assertFalse(Path(tmp,'new').exists())

    def test_failed_calibration_preserves_origin_and_rejects_mixed_states(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            before=m.file_sha256(Path(tmp,'origin/model.pt'))
            with patch.object(m,'recalibrate_bn',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    m.run_recalibration(tmp,'origin','new',smoke=True)
            self.assertTrue(Path(tmp,'new/failure.json').exists())
            self.assertFalse(Path(tmp,'new/report.json').exists())
            self.assertEqual(m.file_sha256(Path(tmp,'origin/model.pt')),before)
            with self.assertRaises(ValueError): m.load_recalibrated_model(tmp,'new')

    def test_rejects_checkpoint_metadata_and_learning_parameter_changes_even_with_new_file_hash(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            report=m.run_recalibration(tmp,'origin','new',smoke=True)
            run=Path(tmp,'new')
            saved=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
            for mutation in ('selected_epoch','parameter'):
                changed=copy.deepcopy(saved)
                if mutation=='selected_epoch': changed['selected_epoch']=999
                else:
                    key=next(n for n in changed['state_dict'] if n.endswith('weight'))
                    changed['state_dict'][key].add_(1.)
                torch.save(changed,run/'model.pt')
                altered=copy.deepcopy(report)
                altered['files_sha256']['model.pt']=m.file_sha256(run/'model.pt')
                m.write_json(run/'report.json',altered)
                with self.assertRaises(ValueError): m.load_recalibrated_model(tmp,'new')

    def test_success_marker_removal_failure_cannot_leave_loadable_success(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            unlink=Path.unlink
            write=m.write_json
            def fail_unlink(path,*args,**kw):
                if path.name=='running.marker': raise OSError('injected unlink')
                return unlink(path,*args,**kw)
            def fail_failure(path,value):
                if Path(path).name=='failure.json': raise OSError('injected failure write')
                return write(path,value)
            with patch.object(Path,'unlink',fail_unlink),patch.object(m,'write_json',fail_failure):
                with self.assertRaises(OSError): m.run_recalibration(tmp,'origin','new',smoke=True)
            self.assertTrue(Path(tmp,'new/running.marker').exists())
            with self.assertRaises(ValueError): m.load_recalibrated_model(tmp,'new')

    def test_registration_rejects_arbitrary_claimed_acceptance_without_real_artifacts(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fake=dict(schema_version=1,status='verified',protocol='paper_bn_acceptance_v1',
                code_sha256=m.source_hashes(),checks={'cpu':True,'cuda':True},runs=[],files_sha256={})
            m.write_json(root/'acceptance.json',fake)
            with self.assertRaises(ValueError): m.register_experiment(tmp,'experiment','acceptance.json')
            self.assertFalse((root/'experiment').exists())

    def test_bool_is_not_a_numeric_protocol_setting(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            report=m.run_recalibration(tmp,'origin','new',smoke=True)
            for field in ('config','schema_version','seed','selected_epoch'):
                altered=copy.deepcopy(report)
                if field=='config': altered['config']['passes']=True
                elif field=='seed': altered[field]=False
                elif field=='selected_epoch': altered[field]=bool(altered[field])
                else: altered[field]=True
                m.write_json(Path(tmp,'new/report.json'),altered)
                with self.subTest(field=field),self.assertRaises(ValueError):
                    m.load_recalibrated_model(tmp,'new')

    @unittest.skipUnless(torch.cuda.is_available(),'actual CUDA required')
    def test_actual_cuda_calibration_preserves_parameters_and_has_portable_state(self):
        m=self.module()
        model=self.model().cuda()
        raw=np.random.default_rng(3).normal(size=(130,250,2)).astype('f4')
        updated, summary=m.recalibrate_bn(model,raw,[str(i) for i in range(len(raw))],dict(mean=[0.,0.],std=[1.,1.]))
        self.assertTrue(summary['parameters_unchanged'])
        self.assertTrue(summary['original_state_unchanged'])
        self.assertTrue(all(torch.isfinite(v).all() for v in updated.cpu().state_dict().values()))
