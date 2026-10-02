import importlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch
import numpy as np


class EvaluationTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('src.ect.evaluation')

    def test_metrics_count_all_classes_and_nominal_groups(self):
        m = self.module()
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.array([0, 0, 0, 0, 1, 2])
        x = m.count_metrics(y, p, [0, 1, 2])
        self.assertAlmostEqual(x['accuracy'], 4/6)
        self.assertAlmostEqual(x['macro_f1'], (2/3 + 2/3 + 2/3)/3)
        self.assertEqual(x['per_class']['2']['support'], 2)
        with self.assertRaises(ValueError):
            m.count_metrics(y, np.array([0, 0, 5, 0, 1, 2]), [0, 1, 2])

    def test_validation_default_and_test_requires_bound_acceptance_before_data_access(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(m, 'PreparedDataset', side_effect=AssertionError('premature data access')):
                with self.assertRaisesRegex(ValueError, 'acceptance'):
                    m.evaluate_registered(tmp, split='test', output_dir='out')
            self.assertFalse(Path(tmp, 'out').exists())
            with self.assertRaises(ValueError):
                m.evaluate_registered(tmp, split='train', output_dir='out')

    def test_acceptance_rejects_stale_code_predictions_and_registry(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / 'validation/report.json'
            report_path.parent.mkdir()
            (report_path.parent / 'predictions.csv').write_text('fixed\n')
            (report_path.parent / 'summary.json').write_text('{}\n')
            report = dict(status='complete', split='validation', heldout_test_evaluated=False,
                          registry_sha256='registry', code_sha256={'evaluation': 'code'},
                          files_sha256={n: m.file_sha256(report_path.parent/n) for n in ('predictions.csv','summary.json')},
                          prediction_count=160, sample_count=1, model_count=10)
            from src.ect.evaluation_summary import summarize_metrics
            entries=[]
            for family,seed in [(f,s) for f in ('cnn_clean','resnet_clean','resnet_aug') for s in range(3)]+[('svm',None)]:
                for condition,noise in [('clean',None)]+[(c,s) for c in ('30','20','10') for s in range(10,15)]:
                    entries.append(dict(run_id=f'{family}_s{seed}',family=family,training_seed=seed,
                        condition=condition,noise_seed=noise,metrics={'accuracy':.5,'macro_f1':.5}))
            report['metrics']=entries
            (report_path.parent/'summary.json').write_text(json.dumps(summarize_metrics(entries,range(10,15))))
            report['files_sha256']['summary.json']=m.file_sha256(report_path.parent/'summary.json')
            report_path.write_text(json.dumps(report))
            acceptance = dict(status='passed', evaluation_report_path='validation/report.json',
                evaluation_report_sha256=m.file_sha256(report_path), registry_sha256='registry',
                evaluation_code_sha256={'evaluation':'code'}, predictions_verified=160,
                test_classification_evaluated=False)
            path = root/'acceptance.json'
            path.write_text(json.dumps(acceptance))
            m.validate_acceptance(root, path, 'registry', {'evaluation':'code'}, expected_samples=1)
            for kind in ('code', 'csv', 'registry', 'coverage', 'failure'):
                with self.subTest(kind=kind):
                    if kind=='csv':
                        (report_path.parent/'predictions.csv').write_text('corrupt\n')
                    elif kind=='failure':
                        (report_path.parent/'failure.json').write_text('{}')
                    else:
                        a=acceptance.copy()
                        if kind=='registry':a['registry_sha256']='wrong'
                        elif kind=='coverage':a['predictions_verified']=1
                        else:a['evaluation_code_sha256']={}
                        path.write_text(json.dumps(a))
                    with self.assertRaises(ValueError):
                        m.validate_acceptance(root,path,'registry',{'evaluation':'code'},expected_samples=1)
                    path.write_text(json.dumps(acceptance))
                    (report_path.parent/'predictions.csv').write_text('fixed\n')
                    (report_path.parent/'failure.json').unlink(missing_ok=True)
            with self.assertRaises(ValueError):
                m.validate_acceptance(root,path,'registry',{'evaluation':'code'},expected_samples=2)
            (report_path.parent/'summary.json').write_text('{}')
            report['files_sha256']['summary.json']=m.file_sha256(report_path.parent/'summary.json')
            report_path.write_text(json.dumps(report))
            acceptance['evaluation_report_sha256']=m.file_sha256(report_path)
            path.write_text(json.dumps(acceptance))
            with self.assertRaises(ValueError):
                m.validate_acceptance(root,path,'registry',{'evaluation':'code'},expected_samples=1)

    def test_executed_source_is_bound_and_failure_marker_survives_unlink_failure(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for relative in m.EVALUATION_CODE:
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes((Path(__file__).resolve().parents[1]/relative).read_bytes())
            self.assertEqual(m.evaluation_code(root),{n:m.file_sha256(root/n) for n in m.EVALUATION_CODE})
            with (root/'src/ect/evaluation.py').open('a') as h:h.write('\n# unexecuted copy\n')
            with self.assertRaisesRegex(ValueError,'executed'):
                m.evaluation_code(root)
            output=root/'failure';output.mkdir();report_path=output/'report.json';report_path.write_text('{"status":"complete"}')
            original=Path.unlink
            def unlink(path,*args,**kwargs):
                if path==report_path:raise PermissionError('cannot delete report')
                return original(path,*args,**kwargs)
            with patch.object(Path,'unlink',unlink):
                m.record_failure(output,dict(status='complete'),OSError('close failed'),0.)
            self.assertEqual(json.loads((output/'failure.json').read_text())['status'],'failed')

    def test_no_overwrite_failure_marker_and_cli_help(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'out').mkdir()
            with self.assertRaises(FileExistsError):
                m.evaluate_registered(tmp, output_dir='out')
            with self.assertRaises(FileNotFoundError):
                m.evaluate_registered(tmp, output_dir='new')
            self.assertTrue((root/'new/failure.json').exists())
            self.assertFalse((root/'new/report.json').exists())
        result=subprocess.run([str(Path(__file__).resolve().parents[1]/'.venv/bin/python'),
            'scripts/evaluate_registered.py','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_real_models_share_noise_and_validation_never_indexes_test(self):
        from fixture_evaluation import make_evaluation
        from src.ect.prepare import PreparedDataset
        from src.ect.noise import add_ac_noise
        import csv
        m=self.module()
        class Guard(dict):
            def __getitem__(self,k):
                if k=='test':raise AssertionError('Test classification accessed')
                return super().__getitem__(k)
        class Sealed(PreparedDataset):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw);self.arrays=Guard(self.arrays)
        with tempfile.TemporaryDirectory() as tmp:
            make_evaluation(tmp)
            with patch.object(m,'PreparedDataset',Sealed):
                report=m.evaluate_registered(tmp,output_dir='evaluation',cache_dir='cache',device='cpu')
            self.assertFalse(report['heldout_test_evaluated'])
            self.assertEqual(report['prediction_count'],320)
            self.assertEqual(len(report['metrics']),160)
            self.assertEqual(len(report['noise_diagnostics']),15)
            with Path(tmp,'evaluation/predictions.csv').open() as handle:rows=list(csv.DictReader(handle))
            self.assertEqual(sum(r['condition']=='clean' for r in rows),20)
            self.assertEqual({r['noise_seed'] for r in rows if r['condition']!='clean'},set(map(str,range(10,15))))
            acceptance=dict(status='passed',evaluation_report_path='evaluation/report.json',
                evaluation_report_sha256=m.file_sha256(Path(tmp,'evaluation/report.json')),
                registry_sha256=report['registry_sha256'],evaluation_code_sha256=report['code_sha256'],
                predictions_verified=320,test_classification_evaluated=False)
            Path(tmp,'acceptance.json').write_text(json.dumps(acceptance))
            test_report=m.evaluate_registered(tmp,split='test',output_dir='test_evaluation',cache_dir='cache',
                device='cpu',acceptance='acceptance.json')
            self.assertEqual(test_report['noise_seeds'],list(range(100,105)))
            self.assertTrue(test_report['heldout_test_evaluated'])
            self.assertEqual(test_report['prediction_count'],320)
            proof=Path(tmp,'results/experiments/formal_v1/training_verification.json')
            proof_text=proof.read_text();proof.write_text(proof_text+'\n')
            with self.assertRaisesRegex(ValueError,'weights differ'):
                m.evaluate_registered(tmp,split='test',output_dir='changed_bindings',cache_dir='cache',
                    device='cpu',acceptance='acceptance.json')
            proof.write_text(proof_text)
            with PreparedDataset(tmp,'cache') as store:
                records=[json.loads(s) for s in Path(tmp,'manifests/fixture/validation.jsonl').read_text().splitlines()]
                for item in report['noise_diagnostics']:
                    noisy,_=add_ac_noise(store.arrays['validation'][0],[r['wave_sha256'] for r in records],
                        float(item['condition']),item['noise_seed'],'validation')
                    import hashlib
                    self.assertEqual(hashlib.sha256(np.asarray(noisy,dtype='<f8').tobytes()).hexdigest(),item['shared_raw_float64_sha256'])
            # Registration source drift must fail, retaining no complete report.
            with Path(tmp,'src/ect/noise.py').open('a') as h:h.write('\n# changed\n')
            with self.assertRaisesRegex(ValueError,'source differs'):
                m.evaluate_registered(tmp,output_dir='drift',cache_dir='cache',device='cpu')
            self.assertFalse(Path(tmp,'drift/report.json').exists())


if __name__ == '__main__':
    unittest.main()
