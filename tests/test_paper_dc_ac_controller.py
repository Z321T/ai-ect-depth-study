import importlib.util
import unittest

class DcAcControllerTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.run_paper_dc_ac_experiment'),'Automatic experiment controller must exist')
        from scripts import run_paper_dc_ac_experiment
        return run_paper_dc_ac_experiment

    def test_three_full_seeds_then_fixed_bn_with_each_independent_proof(self):
        m=self.module()
        registry={'runs':[{'run_id':f'resnext_s{i}','seed':i,'config_path':f'exp/configs/s{i}.json',
                          'output_dir':f'exp/resnext_s{i}'} for i in (0,1,2)]}
        steps=m.execution_steps(registry,'exp/registry.json')
        self.assertEqual([s['kind'] for s in steps],['train','verify']*3+['bn','verify_bn']*3)
        self.assertEqual([s['seed'] for s in steps if s['kind']=='train'],[0,1,2])
        for step in steps:
            self.assertNotIn('test',step['command'])
        self.assertEqual([s['output'] for s in steps if s['kind']=='bn'],['exp/bn/resnext_s0','exp/bn/resnext_s1','exp/bn/resnext_s2'])

    def test_relaunch_or_existing_outputs_refused_before_lock(self):
        m=self.module()
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'exp';folder.mkdir()
            (folder/'execution_status.json').write_text('{}')
            with self.assertRaises(FileExistsError):m.ensure_fresh_execution(root,folder,[])
            self.assertFalse((folder/'controller.running').exists())
            (folder/'execution_status.json').unlink()
            (folder/'resnext_s0').mkdir()
            with self.assertRaises(FileExistsError):m.ensure_fresh_execution(root,folder,[{'output':'exp/resnext_s0'}])

    def test_four_cell_summary_preserves_negative_results_and_predictions(self):
        m=self.module()
        import csv
        import hashlib
        import json
        from pathlib import Path
        import shutil
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'results/experiments/paper_dc_ac_v1';folder.mkdir(parents=True)
            (folder/'verification').mkdir()
            script=root/'scripts/run_paper_dc_ac_experiment.py';script.parent.mkdir();shutil.copyfile(m.__file__,script)
            for seed in (0,1,2):
                metric={'accuracy':.7,'macro_f1':.68,'per_class':{'0':{'precision':.5,'recall':.5,'f1-score':.5,'support':2}}}
                epochs=[{'epoch':i,'train_loss':1/i,'online_train_accuracy':.6,'validation':{'accuracy':.7,'macro_f1':.68} if i%10==0 else None} for i in range(1,1001)]
                r={'status':'complete','heldout_test_evaluated':False,'selected_epoch':10,'metrics':{'train':metric,'validation':metric},
                   'group_metrics':{'train':{'1':metric},'validation':{'3':metric}},'epochs':epochs}
                paths=[root/'results/experiments/paper_baseline_v1'/f'resnext_s{seed}',
                       root/'results/experiments/paper_bn_recalibration_v1'/f'resnext_s{seed}',
                       folder/f'resnext_s{seed}',folder/'bn'/f'resnext_s{seed}']
                for index,path in enumerate(paths):
                    path.mkdir(parents=True)
                    rr=json.loads(json.dumps(r))
                    for split in ('train','validation'):rr['metrics'][split]['accuracy']=[.7,.6,.8,.65][index]
                    (path/'report.json').write_text(json.dumps(rr))
                    if index>=2:
                        digest=hashlib.sha256((path/'report.json').read_bytes()).hexdigest()
                        proof=folder/'verification'/f'resnext_s{seed}{"_bn" if index==3 else ""}_cuda.json'
                        proof.write_text(json.dumps({'status':'verified','report_sha256':digest}))
                        for split in ('train','validation'):(path/f'{split}_predictions.csv').write_text('fixture_prediction\n')
            with patch.object(m,'__file__',str(script)):
                summary=m.summarize(root,folder,{'protocol':'paper_dc_ac_finite_v1'})
            self.assertEqual(len(summary['cells']),12)
            for row,accuracy in zip(summary['aggregate'],[.7,.6,.8,.65]):self.assertAlmostEqual(row['accuracy_mean'],accuracy)
            self.assertLess(summary['aggregate'][3]['paired_delta_mean'],0)
            with (folder/'analysis_results_v1/metrics.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),24)
            self.assertTrue((folder/'resnext_s0/train_predictions.csv').exists())
            self.assertTrue((folder/'resnext_s0/train_predictions.csv.gz').exists())

    def test_source_guard_failure_cleans_lock_and_records_failure(self):
        m=self.module()
        from pathlib import Path
        import json,tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'experiment';folder.mkdir()
            with patch('src.ect.paper_dc_ac_training.source_hashes',side_effect=ValueError('source drift')):
                with self.assertRaises(ValueError):m.run_controller(root,folder/'registry.json',{},'cpu',[])
            self.assertFalse((folder/'controller.running').exists())
            self.assertEqual(json.loads((folder/'execution_failure.json').read_text())['status'],'failed')
