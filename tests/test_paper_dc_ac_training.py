import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fixture_data import make_dataset
from src.ect.prepare import prepare_dataset, PreparedDataset

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT/'config/paper_baseline_v1.json').read_text())
BASE.update(protocol='paper_dc_ac_finite_v1', normalization_protocol='dc_ac_full250_v1',
            purpose='exploratory_train_validation_dc_ac')
SMOKE = BASE | dict(purpose='paper_dc_ac_smoke_validation_only', seed=0, device='cpu',
    max_epochs=2, batch_size=2, validation_every_epochs=1, cache_dir='cache')

class DcAcTrainingTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('src.ect.paper_dc_ac_training'),
                             'DC/AC trainer must exist')
        from src.ect import paper_dc_ac_training
        return paper_dc_ac_training

    def test_train_full_transform_independent_stats_and_reload(self):
        m = self.module()
        class Sealed(PreparedDataset):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                class Guard(dict):
                    def __getitem__(self,key):
                        if key=='test': raise AssertionError('test classification accessed')
                        return super().__getitem__(key)
                self.arrays=Guard(self.arrays)
        with tempfile.TemporaryDirectory() as tmp:
            _, manifests=make_dataset(tmp)
            prepare_dataset(tmp,manifests,'cache')
            with patch.object(m,'PreparedDataset',Sealed):
                report=m.train_paper_dc_ac(tmp,SMOKE,'run')
            self.assertFalse(report['heldout_test_evaluated'])
            self.assertGreater(report['parameter_update_l2'],0)
            self.assertEqual(report['normalization']['protocol'],'dc_ac_full250_v1')
            self.assertEqual(report['normalization']['sample_count'],2)
            self.assertEqual(report['input_transform_order'],['full250_dc_ac','float32','identity_crop'])
            self.assertEqual(report['initial_parameters_sha256'],m.initialization_sha256(0,2,3))
            self.assertEqual(report['selected_epoch'],m.select_epoch(report['epochs'])['epoch'])
            bundle=m.load_paper_dc_ac_model(Path(tmp,'run'),'cpu')
            with PreparedDataset(tmp,'cache') as store:
                for split in ('train','validation'):
                    ids=[json.loads(s)['wave_sha256'] for s in (manifests/f'{split}.jsonl').read_text().splitlines()]
                    p=m.predict_paper_dc_ac(bundle,store.arrays[split][0],ids,batch_size=2)
                    self.assertEqual(len(p),2)
                    self.assertTrue(np.isfinite(p).all())
            original=m.json_read(Path(tmp,'run/report.json'))
            for field,value in [('initial_parameters_sha256','0'*64),('normalization',{}),('code_sha256_lf',{}),('code_sha256',{}),('heldout_test_evaluated',True),('protocol','wrong'),('input_transform_order',[])]:
                changed=copy.deepcopy(original);changed[field]=value
                m.write_json(Path(tmp,'run/report.json'),changed)
                with self.subTest(field=field),self.assertRaises(ValueError):
                    m.load_paper_dc_ac_model(Path(tmp,'run'))
            m.write_json(Path(tmp,'run/report.json'),original)
            Path(tmp,'run/failure.json').write_text('{}')
            with self.assertRaises(ValueError): m.load_paper_dc_ac_model(Path(tmp,'run'))

    def test_config_factors_fixed_and_full_requires_registration(self):
        m=self.module()
        for field,value in [('normalization_protocol','crop_dc_ac'),('learning_rate',.001),('max_epochs',True),
                            ('test_classification_enabled',True),('noise_augmentation',True)]:
            with self.subTest(field=field), self.assertRaises(ValueError): m.validate_config(SMOKE|{field:value})
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError,'registration'):
            m.train_paper_dc_ac(tmp,BASE|dict(seed=0),'full')

    def test_training_initialization_matches_original_and_schedule_is_literal(self):
        m=self.module()
        from src.ect.devices import seed_everything
        from src.ect.paper_models import PaperResNeXt1D
        for seed in (0,1,2):
            seed_everything(seed,threads=1)
            old=PaperResNeXt1D(num_classes=20,blocks_per_stage=3)
            self.assertEqual(m.parameters_sha256(old),m.initialization_sha256(seed,20,3))
        self.assertEqual([m.learning_rate_for_epoch(SMOKE,e) for e in (1000,5000,7500)], [4e-5,4e-6,4e-7])

    def test_actual_training_crops_receive_full250_transformation(self):
        m=self.module()
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            _,manifests=make_dataset(tmp);prepare_dataset(tmp,manifests,'cache')
            with PreparedDataset(tmp,'cache') as store:
                raw=store.arrays['train'][0].copy().astype('float64')
            identities=[json.loads(s)['wave_sha256'] for s in (manifests/'train.jsonl').read_text().splitlines()]
            dc=raw.mean(axis=1);ac=raw-dc[:,None,:]
            expected=((dc-dc.mean(axis=0))[:,None,:]/dc.std(axis=0)+ac/np.sqrt((ac*ac).mean(axis=(0,1))))/np.sqrt(2)
            mapping=dict(zip(identities,expected.astype('float32')))
            original=m.training_crops
            def observe(full,ids,seed,epoch):
                np.testing.assert_allclose(full,np.stack([mapping[identity] for identity in ids]),rtol=1e-6,atol=1e-6)
                self.assertEqual(full.shape[1:],(250,2))
                calls.append(epoch)
                return original(full,ids,seed,epoch)
            with patch.object(m,'training_crops',observe):m.train_paper_dc_ac(tmp,SMOKE,'run')
            self.assertEqual(calls,[0,1])
