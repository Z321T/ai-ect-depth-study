"""Identity-keyed noise in raw units, calibrated to joint I/Q AC power.

RNG keys use compact ASCII JSON of [namespace, seed, sample_id, condition],
SHA256, and the full digest interpreted as an unsigned big-endian PCG64 seed.
SNR conditions use float.hex(), with signed zero normalized. Training uses
namespace ``training`` and includes epoch in both selection and noise keys.
No Python hash or global NumPy random state is used.
"""
import hashlib
import json
from numbers import Integral

import numpy as np


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def _inputs(signals, sample_ids, seed):
    try:
        x = np.asarray(signals)
        if x.dtype.kind not in 'fiu':
            raise ValueError('signals must be real numeric values')
        x = np.asarray(x, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError('signals must be real numeric values') from exc
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] < 2 or x.shape[2] != 2:
        raise ValueError('signals must have nonempty shape [B,T,2] with T >= 2')
    if not np.isfinite(x).all():
        raise ValueError('signals must contain only finite values')
    seed = _nonnegative_integer(seed, 'seed')
    if isinstance(sample_ids, (str, bytes)):
        raise ValueError('sample_ids must be a sequence with one identity per sample')
    try:
        ids = list(sample_ids)
    except TypeError as exc:
        raise ValueError('sample_ids must be a sequence') from exc
    if len(ids) != len(x):
        raise ValueError('sample_ids length must equal B')
    normalized = []
    for identity in ids:
        if isinstance(identity, str) and identity.strip():
            normalized.append(str(identity))
        elif isinstance(identity, Integral) and not isinstance(identity, (bool, np.bool_)):
            normalized.append(int(identity))
        else:
            raise ValueError('sample identities must be nonempty strings or integers')
    return x, normalized, seed


def _snrs(snr_db, batch_size):
    try:
        values = np.asarray(snr_db)
        if values.dtype.kind not in 'fiu':
            raise ValueError('snr_db must be real numeric values')
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError('snr_db must be a finite scalar or vector of length B') from exc
    if values.ndim == 0:
        values = np.full(batch_size, float(values), dtype=np.float64)
    elif values.shape != (batch_size,):
        raise ValueError('snr_db must be a scalar or vector of length B')
    if not np.isfinite(values).all():
        raise ValueError('snr_db must be finite')
    return values.tolist()


def _rng(namespace, seed, identity, condition):
    key = json.dumps([namespace, seed, identity, condition], ensure_ascii=True,
                     separators=(',', ':'), allow_nan=False).encode('ascii')
    entropy = int.from_bytes(hashlib.sha256(key).digest(), 'big')
    return np.random.Generator(np.random.PCG64(entropy))


def _snr_condition(snr):
    return ['snr_db', (0. if snr == 0 else float(snr)).hex()]


def _apply(x, ids, requested, seed, namespace, conditions):
    # Subtract the first point before averaging: exact constants stay exactly
    # zero, and large DC offsets do not contaminate the AC variance.
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            shifted = x - x[:, :1, :]
            ac = shifted - shifted.mean(axis=1, keepdims=True)
            ac_power = np.mean(ac ** 2, axis=(1, 2))
            noisy = x.copy()
            applicable = []
            for i, (identity, snr, condition) in enumerate(zip(ids, requested, conditions)):
                apply_noise = snr is not None and ac_power[i] > 0
                applicable.append(bool(apply_noise))
                if not apply_noise:
                    continue
                noise = _rng(namespace, seed, identity, condition).standard_normal(x.shape[1:])
                noise -= noise.mean(axis=0, keepdims=True)
                # One scale for both channels preserves joint power calibration.
                target_power = ac_power[i] * np.power(10., -snr / 10.)
                if not np.isfinite(target_power) or target_power <= 0:
                    raise ValueError('requested noise power is not representable in float64')
                noise *= np.sqrt(target_power / np.mean(noise ** 2))
                noisy[i] += noise
            # Report the actual difference after float64 addition, including
            # rounding, rather than just the requested theoretical noise power.
            noise_power = np.mean((noisy - x) ** 2, axis=(1, 2))
    except FloatingPointError as exc:
        raise ValueError('noise calculation exceeded float64 range') from exc
    if not np.isfinite(noisy).all() or not np.isfinite(noise_power).all():
        raise ValueError('noise calculation produced nonfinite values')
    if any(active and power == 0 for active, power in zip(applicable, noise_power)):
        raise ValueError('requested noise is too small to represent on these signals')
    return noisy, {'ac_power': ac_power.tolist(), 'noise_power': noise_power.tolist(),
                   'snr_applicable': applicable, 'snr_db': list(requested)}


def add_ac_noise(signals, sample_ids, snr_db, seed, namespace):
    """Return (float64 [B,T,2], info) without modifying the clean input.

    ``snr_db`` is a finite scalar or length-B vector. Identities are nonempty
    strings or integers; duplicates intentionally share their random stream.
    Each channel's generated noise has zero time mean; one joint energy scale
    sets SNR against mean squared, channel-centered AC. Input DC is retained.
    Exact zero-AC rows are unchanged and have ``snr_applicable=False``.
    Info values are per-sample lists; ``noise_power`` measures returned residuals.
    Energy calibration is exact up to float64 arithmetic and addition rounding.
    """
    x, ids, seed = _inputs(signals, sample_ids, seed)
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError('namespace must be a nonempty string')
    requested = _snrs(snr_db, len(x))
    return _apply(x, ids, requested, seed, namespace,
                  [_snr_condition(snr) for snr in requested])


def augment_training(signals, sample_ids, seed, epoch):
    """Select 50% clean / 50% uniform [10,30) dB by identity and epoch.

    ``info['augmented']`` records the Bernoulli choice even for zero-AC rows.
    ``snrs`` and ``snr_db`` contain requested dB, or None for clean rows;
    ``snr_applicable`` is true only when noise was added to nonzero AC.
    Every row reports clean AC and actual residual noise power.
    """
    x, ids, seed = _inputs(signals, sample_ids, seed)
    epoch = _nonnegative_integer(epoch, 'epoch')
    requested, augmented, conditions = [], [], []
    for identity in ids:
        rng = _rng('training', seed, identity, ['augmentation', epoch])
        selected = bool(rng.random() < .5)
        snr = float(rng.uniform(10., 30.)) if selected else None
        augmented.append(selected)
        requested.append(snr)
        conditions.append(['noise', epoch, _snr_condition(snr)] if selected else None)
    noisy, info = _apply(x, ids, requested, seed, 'training', conditions)
    info.update(augmented=augmented, snrs=list(requested))
    return noisy, info
