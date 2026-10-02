"""Pure scalar-metric summaries for the frozen formal_v1 evaluation protocol.

Noise repeats describe variation for one fixed model. Between-training variation
is computed from the three model means, never from 15 pooled observations.
This module reads no files, runs no inference, and does not release sealed test
classification; the evaluator is responsible for those gates.
"""
from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real
from statistics import mean, stdev


_DEEP_FAMILIES = ('cnn_clean', 'resnet_clean', 'resnet_aug')
_FAMILIES = (*_DEEP_FAMILIES, 'svm')
_CONDITIONS = ('clean', '30', '20', '10')
_VALIDATION_SEEDS = (10, 11, 12, 13, 14)
_TEST_SEEDS = (100, 101, 102, 103, 104)
_REQUIRED_FIELDS = frozenset(('run_id', 'family', 'training_seed', 'condition',
                              'noise_seed', 'metrics'))


def _is_integer(value):
    # bool is an Integral, but is not a seed in this protocol.
    return isinstance(value, Integral) and not isinstance(value, bool)


def summarize_metrics(entries: Iterable[Mapping], noise_seeds: Iterable[int]) -> dict:
    """Validate and summarize all 160 scalar records for one evaluation split.

    ``noise_seeds`` must contain exactly validation 10..14 or test 100..104,
    in any order. Each deep family must have one run for each training seed
    0/1/2; SVM must have one run with training_seed=None. A run_id is an opaque,
    nonblank string identifying exactly one (family, training_seed), with the
    same identity on every record. Each run requires one clean record with
    noise_seed=None and five records for each of '30', '20', '10'.

    Metrics must have identical nonblank string keys on every record, including
    accuracy and macro_f1. All values must be finite real scalars in [0, 1].
    Pass scalar rates extracted from evaluator metrics; confusion matrices,
    counts and nested per-class/group reports do not belong in this mapping.

    Returns a JSON-serializable dict. per_run[run_id] contains family,
    training_seed, and conditions[condition][metric] with mean/noise_sample_sd.
    families[family] contains run_ids, training_seeds, and
    conditions[condition][metric] with mean, training_sample_sd,
    n_training_seeds, and within_training_seed_noise_sd (a list of
    {training_seed, noise_sample_sd}). Clean noise SD and SVM training SD are
    None. All sample SDs use ddof=1. statistical_scope records the fixed-split
    interpretation. Invalid or incomplete input raises ValueError. Inputs are
    not modified; records and seeds may be one-shot iterables.
    """
    try:
        seeds = tuple(noise_seeds)
    except TypeError as exc:
        raise ValueError('noise_seeds must be a frozen five-seed iterable') from exc
    if (len(seeds) != 5 or not all(_is_integer(seed) for seed in seeds)
            or set(seeds) not in (set(_VALIDATION_SEEDS), set(_TEST_SEEDS))):
        raise ValueError('noise_seeds must be exactly 10..14 or 100..104 without duplicates')
    seeds = tuple(sorted(int(seed) for seed in seeds))

    try:
        records = iter(entries)
    except TypeError as exc:
        raise ValueError('entries must be an iterable of metric records') from exc

    runs = {}
    identity_to_run = {}
    metric_names = None
    for index, entry in enumerate(records):
        if not isinstance(entry, Mapping) or not _REQUIRED_FIELDS <= entry.keys():
            raise ValueError(f'Entry {index} must contain all required record fields')
        run_id, family = entry['run_id'], entry['family']
        training_seed = entry['training_seed']
        condition, noise_seed = entry['condition'], entry['noise_seed']
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f'Entry {index} requires a nonblank string run_id')
        if not isinstance(family, str) or family not in _FAMILIES:
            raise ValueError(f'Invalid family for run {run_id}: {family!r}')
        if family == 'svm':
            if training_seed is not None:
                raise ValueError('SVM training_seed must be None')
        elif not _is_integer(training_seed) or training_seed not in (0, 1, 2):
            raise ValueError(f'Invalid training_seed for run {run_id}: {training_seed!r}')
        else:
            training_seed = int(training_seed)

        identity = (family, training_seed)
        if run_id in runs and runs[run_id]['identity'] != identity:
            raise ValueError(f'Inconsistent family/training_seed identity for run {run_id}')
        if identity in identity_to_run and identity_to_run[identity] != run_id:
            raise ValueError(f'Multiple run_ids claim model identity {identity!r}')
        identity_to_run[identity] = run_id

        if not isinstance(condition, str) or condition not in _CONDITIONS:
            raise ValueError(f'Invalid condition for run {run_id}: {condition!r}')
        if condition == 'clean':
            if noise_seed is not None:
                raise ValueError(f'Clean noise_seed must be None for run {run_id}')
        elif not _is_integer(noise_seed) or noise_seed not in seeds:
            raise ValueError(f'Invalid noise_seed for run {run_id}: {noise_seed!r}')
        else:
            noise_seed = int(noise_seed)

        metrics = entry['metrics']
        if (not isinstance(metrics, Mapping)
                or not all(isinstance(key, str) and key.strip() for key in metrics)
                or not {'accuracy', 'macro_f1'} <= metrics.keys()):
            raise ValueError(f'Run {run_id} requires a scalar metric mapping with accuracy/macro_f1')
        names = tuple(sorted(metrics))
        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            raise ValueError(f'Inconsistent metric keys for run {run_id}')
        values = {}
        for name, value in metrics.items():
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not 0 <= value <= 1 or not math.isfinite(value)):
                raise ValueError(f'Metric {name!r} for run {run_id} must be finite in [0, 1]')
            values[name] = float(value)

        observations = runs.setdefault(run_id, {'identity': identity, 'observations': {}})['observations']
        observation = (condition, noise_seed)
        if observation in observations:
            raise ValueError(f'Duplicate observation for run {run_id}: {observation!r}')
        observations[observation] = values

    expected_identities = {(family, seed) for family in _DEEP_FAMILIES for seed in (0, 1, 2)}
    expected_identities.add(('svm', None))
    if set(identity_to_run) != expected_identities:
        raise ValueError('Incomplete formal_v1 coverage: require all nine deep runs and one SVM')
    expected_observations = {('clean', None)}
    expected_observations.update((condition, seed) for condition in _CONDITIONS[1:] for seed in seeds)

    per_run = {}
    for run_id in sorted(runs):
        run = runs[run_id]
        if set(run['observations']) != expected_observations:
            raise ValueError(f'Incomplete observations for run {run_id}: require 1 Clean + 3 x 5 noise')
        family, training_seed = run['identity']
        conditions = {}
        for condition in _CONDITIONS:
            condition_seeds = (None,) if condition == 'clean' else seeds
            metrics = {}
            for name in metric_names:
                values = [run['observations'][(condition, seed)][name] for seed in condition_seeds]
                metrics[name] = {'mean': mean(values),
                                 'noise_sample_sd': None if condition == 'clean' else stdev(values)}
            conditions[condition] = metrics
        per_run[run_id] = {'family': family, 'training_seed': training_seed, 'conditions': conditions}

    families = {}
    for family in _FAMILIES:
        training_seeds = [None] if family == 'svm' else [0, 1, 2]
        run_ids = [identity_to_run[(family, seed)] for seed in training_seeds]
        conditions = {}
        for condition in _CONDITIONS:
            metrics = {}
            for name in metric_names:
                run_metrics = [per_run[run_id]['conditions'][condition][name] for run_id in run_ids]
                run_means = [metric['mean'] for metric in run_metrics]
                metrics[name] = {
                    'mean': mean(run_means),
                    'training_sample_sd': stdev(run_means) if len(run_means) > 1 else None,
                    'n_training_seeds': len(run_means),
                    'within_training_seed_noise_sd': [
                        {'training_seed': seed, 'noise_sample_sd': metric['noise_sample_sd']}
                        for seed, metric in zip(training_seeds, run_metrics)],
                }
            conditions[condition] = metrics
        families[family] = {'run_ids': run_ids, 'training_seeds': training_seeds, 'conditions': conditions}

    return {
        'families': families,
        'per_run': per_run,
        'statistical_scope': {
            'protocol': 'formal_v1',
            'split': 'validation' if seeds == _VALIDATION_SEEDS else 'test',
            'noise_seeds': list(seeds),
            'sample_sd_ddof': 1,
            'clean_observations_per_run': 1,
            'noise_observations_per_condition_per_run': 5,
            'noise_repeats_are_independent_training_runs': False,
            'population_confidence_interval': False,
            'interpretation': ('Fixed split: average noise repeats within each model, then summarize '
                               'the three training-seed means; SVM is one deterministic model. '
                               'Variation is not a sampling-population confidence interval.'),
        },
    }
