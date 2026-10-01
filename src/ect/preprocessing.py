"""Fixed anti-alias filtering and train-only channel normalization."""
import numpy as np
from scipy.signal import firwin, resample_poly

FIR_COEFFICIENTS = firwin(101, 250, fs=2500, window=("kaiser", 5.0), scale=True)


def downsample_iq(signals):
    data = np.asarray(signals)
    if data.ndim not in (2, 3) or data.shape[-2:] != (1250, 2):
        raise ValueError("Signals must end in 1250,2 with an optional batch axis")
    if not np.isfinite(data).all():
        raise ValueError("Signals must be finite")
    result = resample_poly(data.astype(np.float64), up=1, down=5, axis=-2,
                           window=FIR_COEFFICIENTS, padtype="line")
    return result.astype(np.float32)


def fit_channel_stats(batches):
    """Parallel central-moment merge; count refers to scalar points per channel."""
    count = 0
    mean = np.zeros(2, dtype=np.float64)
    m2 = np.zeros(2, dtype=np.float64)
    for batch in batches:
        data = np.asarray(batch, dtype=np.float64)
        if data.ndim != 3 or data.shape[-1] != 2 or not np.isfinite(data).all():
            raise ValueError("Statistics require finite B,T,2 arrays")
        flat = data.reshape(-1, 2)
        n = len(flat)
        if not n:
            continue
        # Center before summation: exact constants remain exact even in float64.
        batch_mean = flat[0] + (flat - flat[0]).mean(axis=0)
        batch_m2 = np.sum((flat - batch_mean) ** 2, axis=0)
        delta = batch_mean - mean
        total = count + n
        m2 += batch_m2 + delta ** 2 * (count * n / total)
        mean += delta * (n / total)
        count = total
    if not count:
        raise ValueError("Cannot fit statistics on an empty training set")
    std = np.sqrt(m2 / count)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Training channel variance must be positive and finite")
    return {"count_per_channel": count, "mean": mean.tolist(), "std": std.tolist(),
            "ddof": 0, "fitted_split": "train"}


def standardize(signals, stats):
    data = np.asarray(signals, dtype=np.float64)
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    if data.ndim not in (2, 3) or data.shape[-1] != 2 or not np.isfinite(data).all():
        raise ValueError("Normalization requires finite T,2 or B,T,2 signals")
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid channel statistics")
    return ((data - mean) / std).astype(np.float32)
