import importlib.util
from pathlib import Path
import tempfile
import unittest

class DcAcRegistrationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('src.ect.paper_dc_ac_registration'),'Execution registrar must exist')
        from src.ect import paper_dc_ac_registration
        return paper_dc_ac_registration

    def test_no_flags_only_acceptance_or_overwrite(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'acceptance.json').write_text('{"status":"accepted","checks":{"cpu_training":true,"cuda_training":true}}')
            with self.assertRaises(ValueError): m.register_experiment(root,'experiment','acceptance.json')
            self.assertFalse((root/'experiment').exists())

    def test_training_must_match_frozen_registry(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): m.validate_registered_run(Path(tmp),'missing',{},Path(tmp),Path(tmp))

    def test_evidence_rejects_nested_failure(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/'smoke';folder.mkdir();(folder/'failure.json').write_text('{}')
            (folder/'proof.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Failed'):m.validate_files(root,{'smoke/proof.json':m.file_sha256(folder/'proof.json')})

    def test_stripped_independent_flags_or_file_chain_are_not_accepted(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            proof={'status':'verified','checks':{'test_classification_evaluated':False}}
            with self.assertRaises(ValueError):m.validate_proof_contract(root,proof,{},root,'cpu',{})

    def test_proof_chain_cannot_drop_inputs(self):
        m=self.module()
        from fixture_data import make_dataset
        from src.ect.prepare import prepare_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,manifests=make_dataset(root);prepare_dataset(root,manifests,'cache')
            run=root/'run';run.mkdir();(run/'report.json').write_text('{}')
            report={'cache_path':'cache','files_sha256':{},'code_sha256':{}}
            with self.assertRaises(ValueError):
                m.check_proof_coverage(root,{str(run/'report.json'):m.file_sha256(run/'report.json')},run,report)

    def test_full_regression_needs_completed_test_count_and_actual_log(self):
        m=self.module()
        root=Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            m.validate_regression(root,{'exit_code':0,'skipped':0,'cuda_available':True},{})

    def test_baseline_dataset_identity_cannot_change_with_same_counts(self):
        m=self.module()
        import copy,json
        from fixture_data import make_dataset
        from src.ect.prepare import prepare_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,manifests=make_dataset(root);prepare_dataset(root,manifests,'cache')
            cache=root/'cache';metadata=json.loads((cache/'metadata.json').read_text())
            baseline=dict(cache_dir='cache',cache_metadata_sha256=m.file_sha256(cache/'metadata.json'),
                cache_files_sha256=metadata['files_sha256'],manifest_sha256=metadata['manifest_sha256'],
                summary_path=metadata['summary_path'],summary_sha256=metadata['summary_sha256'],
                raw_inputs=metadata['inputs'],sample_counts=metadata['sample_counts'],
                normalization_sha256=metadata['files_sha256']['normalization.json'])
            m.validate_baseline_data(root,cache,metadata,baseline)
            changed=copy.deepcopy(metadata);changed['inputs']['train']['sha256']='0'*64
            with self.assertRaises(ValueError):m.validate_baseline_data(root,cache,changed,baseline)
