"""Formal evaluation summaries: training repeats and noise repeats stay separate."""
import copy
import importlib
import importlib.util
import json
import math
import unittest


VALIDATION_SEEDS = (10, 11, 12, 13, 14)
TEST_SEEDS = (100, 101, 102, 103, 104)
DEEP_FAMILIES = ('cnn_clean', 'resnet_clean', 'resnet_aug')


def complete_entries(noise_seeds=VALIDATION_SEEDS):
    """160 scalar records; hand-chosen noise deviations sum to zero."""
    entries = []
    identities = [(f'{family}_s{seed}', family, seed)
                  for family in DEEP_FAMILIES for seed in (0, 1, 2)]
    identities.append(('svm', 'svm', None))
    for run_id, family, seed in identities:
        base = .7 if seed is None else .2 * (seed + 1)
        clean = .8 if seed is None else .25 + .2 * seed
        observations = [('clean', None, clean)]
        for condition, scale in (('30', 1), ('20', 2), ('10', .5)):
            observations.extend((condition, noise_seed, base + scale * deviation)
                                for noise_seed, deviation in zip(
                                    noise_seeds, (-.1, -.05, 0, .05, .1)))
        for condition, noise_seed, value in observations:
            entries.append(dict(run_id=run_id, family=family, training_seed=seed,
                                condition=condition, noise_seed=noise_seed,
                                metrics={'accuracy': value, 'macro_f1': value / 2,
                                         'balanced_accuracy': 1 - value}))
    return entries


class EvaluationSummaryTests(unittest.TestCase):
    def summarize(self, entries, noise_seeds=VALIDATION_SEEDS):
        # A missing implementation is an explicit RED assertion, not an import typo.
        name = 'src.ect.evaluation_summary'
        self.assertIsNotNone(importlib.util.find_spec(name),
                             'evaluation_summary module is not implemented')
        return importlib.import_module(name).summarize_metrics(entries, noise_seeds)

    def test_clean_has_one_observation_and_no_noise_sd(self):
        result = self.summarize(complete_entries())
        run = result['per_run']['cnn_clean_s0']
        self.assertEqual(run['family'], 'cnn_clean')
        self.assertEqual(run['training_seed'], 0)
        self.assertEqual(run['conditions']['clean']['accuracy'],
                         {'mean': .25, 'noise_sample_sd': None})
        clean = result['families']['cnn_clean']['conditions']['clean']['accuracy']
        self.assertAlmostEqual(clean['mean'], .45)
        self.assertAlmostEqual(clean['training_sample_sd'], .2)
        self.assertEqual(clean['n_training_seeds'], 3)
        self.assertEqual(clean['within_training_seed_noise_sd'], [
            {'training_seed': seed, 'noise_sample_sd': None} for seed in (0, 1, 2)])

    def test_noise_sd_uses_ddof_one_within_each_training_seed(self):
        result = self.summarize(complete_entries())
        run = result['per_run']['cnn_clean_s0']['conditions']
        # Five deviations: squared sum .025, sample variance .025 / 4.
        for condition, expected_sd in (('30', math.sqrt(.00625)),
                                       ('20', math.sqrt(.025)),
                                       ('10', math.sqrt(.0015625))):
            with self.subTest(condition=condition):
                self.assertAlmostEqual(run[condition]['accuracy']['mean'], .2)
                self.assertAlmostEqual(run[condition]['accuracy']['noise_sample_sd'], expected_sd)
                self.assertAlmostEqual(run[condition]['macro_f1']['noise_sample_sd'], expected_sd / 2)

    def test_family_sd_uses_three_run_means_not_fifteen_observations(self):
        result = self.summarize(complete_entries())
        for family in DEEP_FAMILIES:
            aggregate = result['families'][family]
            self.assertEqual(aggregate['training_seeds'], [0, 1, 2])
            for condition in ('30', '20', '10'):
                metrics = aggregate['conditions'][condition]
                self.assertAlmostEqual(metrics['accuracy']['mean'], .4)
                self.assertAlmostEqual(metrics['accuracy']['training_sample_sd'], .2)
                self.assertAlmostEqual(metrics['macro_f1']['mean'], .2)
                self.assertAlmostEqual(metrics['macro_f1']['training_sample_sd'], .1)
                self.assertAlmostEqual(metrics['balanced_accuracy']['mean'], .6)
                self.assertEqual(metrics['accuracy']['n_training_seeds'], 3)
            within = aggregate['conditions']['30']['accuracy']['within_training_seed_noise_sd']
            self.assertEqual([item['training_seed'] for item in within], [0, 1, 2])
            for item in within:
                self.assertAlmostEqual(item['noise_sample_sd'], math.sqrt(.00625))

    def test_svm_has_one_model_without_fabricated_training_sd(self):
        result = self.summarize(complete_entries())
        svm = result['families']['svm']
        self.assertEqual(svm['training_seeds'], [None])
        for condition, mean in (('clean', .8), ('30', .7), ('20', .7), ('10', .7)):
            metric = svm['conditions'][condition]['accuracy']
            self.assertAlmostEqual(metric['mean'], mean)
            self.assertIsNone(metric['training_sample_sd'])
            self.assertEqual(metric['n_training_seeds'], 1)
            self.assertIsNone(metric['within_training_seed_noise_sd'][0]['training_seed'])
        self.assertAlmostEqual(svm['conditions']['30']['accuracy']
                               ['within_training_seed_noise_sd'][0]['noise_sample_sd'],
                               math.sqrt(.00625))

    def test_different_within_seed_noise_spreads_remain_separately_identified(self):
        entries = complete_entries()
        for entry in entries:
            if entry['family'] == 'cnn_clean' and entry['condition'] == '30':
                seed = entry['training_seed']
                center = (.2, .4, .6)[seed]
                factor = (.5, 1, 2)[seed]
                value = center + factor * (entry['metrics']['accuracy'] - center)
                entry['metrics'] = {'accuracy': value, 'macro_f1': value / 2,
                                    'balanced_accuracy': 1 - value}
        aggregate = self.summarize(entries)['families']['cnn_clean']['conditions']['30']['accuracy']
        self.assertAlmostEqual(aggregate['mean'], .4)
        self.assertAlmostEqual(aggregate['training_sample_sd'], .2)
        within = aggregate['within_training_seed_noise_sd']
        for item, seed, variance in zip(within, (0, 1, 2), (.0015625, .00625, .025)):
            self.assertEqual(item['training_seed'], seed)
            self.assertAlmostEqual(item['noise_sample_sd'], math.sqrt(variance))

    def test_replacing_one_observation_with_a_duplicate_cannot_hide_a_missing_seed(self):
        entries = complete_entries()
        entries[2] = copy.deepcopy(entries[1])
        self.assertEqual(len(entries), 160)
        with self.assertRaises(ValueError):
            self.summarize(entries)

    def test_test_seeds_and_unordered_iterables_are_supported_without_mutation(self):
        entries = complete_entries(TEST_SEEDS)
        before = copy.deepcopy(entries)
        expected = self.summarize(entries, TEST_SEEDS)
        actual = self.summarize(iter(reversed(entries)), iter(reversed(TEST_SEEDS)))
        self.assertEqual(actual, expected)
        self.assertEqual(entries, before)
        self.assertEqual(len(actual['per_run']), 10)
        self.assertEqual(set(actual['families']), {*DEEP_FAMILIES, 'svm'})
        self.assertEqual(json.loads(json.dumps(actual, allow_nan=False)), actual)
        scope = actual['statistical_scope']
        self.assertEqual(scope['protocol'], 'formal_v1')
        self.assertEqual(scope['noise_seeds'], list(TEST_SEEDS))
        self.assertEqual(scope['sample_sd_ddof'], 1)
        self.assertFalse(scope['noise_repeats_are_independent_training_runs'])
        self.assertFalse(scope['population_confidence_interval'])

    def test_arbitrary_run_labels_preserve_bijective_model_identity(self):
        entries = complete_entries()
        for entry in entries:
            entry['run_id'] = 'registered:' + entry['run_id']
        result = self.summarize(entries)
        self.assertIn('registered:cnn_clean_s0', result['per_run'])

    def test_zero_noise_variation_and_boundary_metrics_are_valid(self):
        entries = complete_entries()
        for entry in entries:
            entry['metrics'] = {'accuracy': 0, 'macro_f1': 1}
        result = self.summarize(entries)
        noise = result['families']['cnn_clean']['conditions']['30']
        self.assertEqual(noise['accuracy']['mean'], 0)
        self.assertEqual(noise['accuracy']['training_sample_sd'], 0)
        self.assertEqual(result['per_run']['svm']['conditions']['30']
                         ['macro_f1']['noise_sample_sd'], 0)

    def test_missing_observations_conditions_and_runs_are_rejected(self):
        full = complete_entries()
        cases = [[], full[1:], full[:1] + full[2:],
                 [e for e in full if e['condition'] != '10'],
                 [e for e in full if e['run_id'] != 'cnn_clean_s2'],
                 [e for e in full if e['family'] != 'svm']]
        for entries in cases:
            with self.subTest(count=len(entries)), self.assertRaises(ValueError):
                self.summarize(entries)

    def test_duplicate_clean_or_noise_observations_are_rejected(self):
        for index in (0, 1):
            with self.subTest(index=index), self.assertRaises(ValueError):
                entries = complete_entries()
                self.summarize(entries + [copy.deepcopy(entries[index])])

    def test_noise_seed_protocol_requires_one_of_the_two_frozen_sets(self):
        for seeds in ((), (10, 11, 12, 13), (10, 11, 12, 13, 13),
                      (0, 1, 2, 3, 4), (10, 11, 12, 13, 104),
                      (10., 11, 12, 13, 14), ('10', 11, 12, 13, 14),
                      (True, 11, 12, 13, 14), None):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                self.summarize(complete_entries(), seeds)

    def test_observation_noise_seed_must_match_condition_and_protocol(self):
        for index, bad_seed in ((0, 10), (0, False), (1, None), (1, 100),
                                (1, 15), (1, 10.), (1, '10'), (1, True)):
            entries = complete_entries()
            entries[index]['noise_seed'] = bad_seed
            with self.subTest(index=index, seed=bad_seed), self.assertRaises(ValueError):
                self.summarize(entries)
        with self.assertRaises(ValueError):
            self.summarize(complete_entries(TEST_SEEDS))

    def test_invalid_training_seeds_families_and_conditions_are_rejected(self):
        for field, bad in (('training_seed', None), ('training_seed', 3),
                           ('training_seed', -1), ('training_seed', True),
                           ('training_seed', 0.), ('training_seed', '0'),
                           ('family', 'cnn'), ('family', None), ('family', []),
                           ('condition', 'Clean'), ('condition', '40'),
                           ('condition', 30), ('condition', [])):
            entries = complete_entries()
            entries[0][field] = bad
            with self.subTest(field=field, value=bad), self.assertRaises(ValueError):
                self.summarize(entries)
        for seed in (0, 1, 2, False):
            entries = complete_entries()
            entries[-1]['training_seed'] = seed
            with self.subTest(svm_seed=seed), self.assertRaises(ValueError):
                self.summarize(entries)

    def test_same_run_id_cannot_change_family_or_training_seed(self):
        for field, bad in (('family', 'resnet_clean'), ('training_seed', 1)):
            entries = complete_entries()
            entries[1][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(entries)
        entries = complete_entries()
        for entry in entries:
            if entry['family'] == 'svm':
                entry['run_id'] = 'cnn_clean_s0'
        with self.assertRaises(ValueError):
            self.summarize(entries)

    def test_multiple_run_ids_cannot_claim_one_training_identity(self):
        entries = complete_entries()
        entries[1]['run_id'] = 'second_model'
        with self.assertRaises(ValueError):
            self.summarize(entries)
        full = complete_entries()
        for source in ('cnn_clean_s0', 'svm'):
            extras = [dict(e, run_id='extra_model') for e in full if e['run_id'] == source]
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.summarize(full + extras)

    def test_invalid_run_ids_and_missing_fields_are_rejected(self):
        for bad in (None, '', '  ', 1, []):
            entries = complete_entries()
            entries[0]['run_id'] = bad
            with self.subTest(run_id=bad), self.assertRaises(ValueError):
                self.summarize(entries)
        for field in ('run_id', 'family', 'training_seed', 'condition', 'noise_seed', 'metrics'):
            entries = complete_entries()
            del entries[0][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.summarize(entries)

    def test_metrics_must_be_finite_numeric_rates_in_unit_interval(self):
        for bad in (float('nan'), float('inf'), -float('inf'), -.001, 1.001,
                    '0.5', None, True, [], {}, complex(.5, 0)):
            entries = complete_entries()
            entries[0]['metrics']['accuracy'] = bad
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                self.summarize(entries)
        entries = complete_entries()
        entries[-1]['metrics']['balanced_accuracy'] = float('nan')
        with self.assertRaises(ValueError):
            self.summarize(entries)

    def test_scalar_metric_schema_must_be_complete_and_consistent(self):
        for bad in (None, [], {}, {'accuracy': .5}, {'macro_f1': .5},
                    {'accuracy': .5, 'macro_f1': .5},
                    {'accuracy': .5, 'macro_f1': .5, 'balanced_accuracy': .5, 'extra': .5}):
            entries = complete_entries()
            entries[0]['metrics'] = bad
            with self.subTest(metrics=bad), self.assertRaises(ValueError):
                self.summarize(entries)
        for bad_key in ('', ' ', 7):
            entries = complete_entries()
            for entry in entries:
                entry['metrics'][bad_key] = .5
            with self.subTest(key=bad_key), self.assertRaises(ValueError):
                self.summarize(entries)
        for missing in ('accuracy', 'macro_f1'):
            entries = complete_entries()
            for entry in entries:
                del entry['metrics'][missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.summarize(entries)

    def test_malformed_entry_containers_raise_value_error(self):
        for bad in (None, [None], [1], ['entry'], {'run_id': 'cnn_clean_s0'}):
            with self.subTest(entries=bad), self.assertRaises(ValueError):
                self.summarize(bad)


if __name__ == '__main__':
    unittest.main()
