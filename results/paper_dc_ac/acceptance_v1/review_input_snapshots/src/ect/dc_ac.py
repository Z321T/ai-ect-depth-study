"""Train-only DC/AC scales for complete 250-point, two-channel waveforms.

All fitting and decomposition use float64. Transform the full waveform to
float32 before applying any crop; coefficients remain fixed across samples.
This module uses only NumPy and does not own input memmap lifetimes.
"""
import math

import numpy as np


_EPSILON = 1e-12
_SIGNAL_LENGTH = 250
_CHANNELS = 2
_PROTOCOL = "dc_ac_full250_v1"
_STAT_KEYS = frozenset({
    "schema_version", "protocol", "fitted_split", "sample_count",
    "points_per_channel", "epsilon", "mu_d", "sigma_d", "sigma_a",
    "scale_d", "scale_a", "floor_d", "floor_a", "channels", "signal_length",
})


def _finite_number(value):
    """Require a JSON number, excluding booleans and NumPy scalar objects."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_stats(stats):
    """Validate the exact v1 JSON schema and coefficient coherence.

    Raises ValueError for missing/extra fields, invalid types, nonfinite
    coefficients, inconsistent counts, floors, or scales. Returns None.
    A floor flag is true exactly when its unfloored sigma is below epsilon.
    """
    if type(stats) is not dict or stats.keys() != _STAT_KEYS:
        raise ValueError("DC/AC statistics require the exact full v1 schema")
    for key, expected in (("schema_version", 1), ("channels", _CHANNELS),
                          ("signal_length", _SIGNAL_LENGTH)):
        if type(stats[key]) is not int or stats[key] != expected:
            raise ValueError(f"Invalid DC/AC statistics {key}")
    if type(stats["protocol"]) is not str or stats["protocol"] != _PROTOCOL:
        raise ValueError("Invalid DC/AC statistics protocol")
    if type(stats["fitted_split"]) is not str or stats["fitted_split"] != "train":
        raise ValueError("DC/AC statistics must be fitted on train")
    if type(stats["sample_count"]) is not int or stats["sample_count"] <= 0:
        raise ValueError("DC/AC sample_count must be a positive integer")
    if (type(stats["points_per_channel"]) is not int
            or stats["points_per_channel"] != stats["sample_count"] * _SIGNAL_LENGTH):
        raise ValueError("Inconsistent DC/AC points_per_channel")
    if not _finite_number(stats["epsilon"]) or stats["epsilon"] != _EPSILON:
        raise ValueError("DC/AC epsilon must be 1e-12")
    for key in ("mu_d", "sigma_d", "sigma_a", "scale_d", "scale_a"):
        values = stats[key]
        if (type(values) is not list or len(values) != _CHANNELS
                or not all(_finite_number(value) for value in values)):
            raise ValueError(f"DC/AC {key} must be a finite two-number JSON list")
    for component in ("d", "a"):
        flags = stats[f"floor_{component}"]
        if (type(flags) is not list or len(flags) != _CHANNELS
                or any(type(flag) is not bool for flag in flags)):
            raise ValueError(f"DC/AC floor_{component} must be a two-boolean JSON list")
        for sigma, scale, floored in zip(stats[f"sigma_{component}"],
                                         stats[f"scale_{component}"], flags):
            if sigma < 0 or scale <= 0:
                raise ValueError("DC/AC sigmas must be nonnegative and scales positive")
            if scale != max(sigma, _EPSILON) or floored != (sigma < _EPSILON):
                raise ValueError("Incoherent DC/AC scale or floor flag")


def _raw_array(raw):
    # No whole-array dtype conversion or finite scan: fitting reads one batch
    # at a time, including for read-only NumPy memmaps.
    data = np.asarray(raw)
    if data.ndim != 3 or data.shape[1:] != (_SIGNAL_LENGTH, _CHANNELS):
        raise ValueError("DC/AC input must have shape [B,250,2]")
    if data.shape[0] == 0:
        raise ValueError("DC/AC input must not be empty")
    if data.dtype.kind not in "fiu":
        raise ValueError("DC/AC input must contain real numeric values")
    return data


def _decompose(data):
    wave = np.asarray(data, dtype=np.float64)
    if not np.isfinite(wave).all():
        raise ValueError("DC/AC input must be finite in float64")
    dc = wave.mean(axis=1, dtype=np.float64)
    return dc, wave - dc[:, None, :]


def fit_dc_ac(raw, batch_size=256, fitted_split="train"):
    """Fit JSON-serializable per-channel scales on clean train waveforms.

    Merge central moments of wave means in float64. A common origin avoids
    loss of small between-wave variance when DC offsets are very large.
    AC power uses the full N*250 residual points in each channel. Memory
    use for NumPy arrays/memmaps is bounded by batch_size, not dataset size.
    """
    if type(fitted_split) is not str or fitted_split != "train":
        raise ValueError("DC/AC fitting is allowed only for train")
    if (isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0):
        raise ValueError("batch_size must be a positive integer")
    data = _raw_array(raw)
    count = 0
    mean = np.zeros(_CHANNELS, dtype=np.float64)
    m2 = np.zeros(_CHANNELS, dtype=np.float64)
    ac_square_sum = np.zeros(_CHANNELS, dtype=np.float64)
    origin = None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for start in range(0, len(data), int(batch_size)):
                dc, ac = _decompose(data[start:start + batch_size])
                if origin is None:
                    origin = dc[0].copy()
                centered = dc - origin
                n = len(dc)
                batch_mean = centered.mean(axis=0, dtype=np.float64)
                batch_m2 = np.sum((centered - batch_mean) ** 2, axis=0, dtype=np.float64)
                if count == 0:
                    mean = batch_mean
                    m2 = batch_m2
                else:
                    delta = batch_mean - mean
                    total = count + n
                    m2 += batch_m2 + delta ** 2 * (count * n / total)
                    mean += delta * (n / total)
                count += n
                ac_square_sum += np.sum(ac ** 2, axis=(0, 1), dtype=np.float64)
            mu_d = origin + mean
            sigma_d = np.sqrt(m2 / count)
            sigma_a = np.sqrt(ac_square_sum / (count * _SIGNAL_LENGTH))
    except FloatingPointError as error:
        raise ValueError("DC/AC input exceeds finite float64 fitting range") from error
    stats = {
        "schema_version": 1, "protocol": _PROTOCOL, "fitted_split": "train",
        "sample_count": count, "points_per_channel": count * _SIGNAL_LENGTH,
        "epsilon": _EPSILON, "mu_d": mu_d.tolist(),
        "sigma_d": sigma_d.tolist(), "sigma_a": sigma_a.tolist(),
        "scale_d": np.maximum(sigma_d, _EPSILON).tolist(),
        "scale_a": np.maximum(sigma_a, _EPSILON).tolist(),
        "floor_d": (sigma_d < _EPSILON).tolist(),
        "floor_a": (sigma_a < _EPSILON).tolist(),
        "channels": _CHANNELS, "signal_length": _SIGNAL_LENGTH,
    }
    validate_stats(stats)
    return stats


def transform_dc_ac(raw, stats):
    """Return finite float32 [B,250,2] using fixed clean-train coefficients.

    z = ((d-mu_d)/scale_d + (x-d)/scale_a)/sqrt(2). No fitting, crops,
    or per-wave AC rescaling occur here; input data and stats are untouched.
    """
    validate_stats(stats)
    data = _raw_array(raw)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            dc, ac = _decompose(data)
            dc_scaled = (dc - np.asarray(stats["mu_d"], dtype=np.float64)) / stats["scale_d"]
            z = (dc_scaled[:, None, :] + ac / stats["scale_a"]) / math.sqrt(2)
            result = z.astype(np.float32)
    except FloatingPointError as error:
        raise ValueError("DC/AC transform exceeds finite float32 range") from error
    if not np.isfinite(result).all():
        raise ValueError("DC/AC transform must be finite")
    return result
