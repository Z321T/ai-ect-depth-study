"""Fixed raw-unit statistics and I/Q trajectory descriptors (features_v1)."""
import numpy as np

CHANNEL_FEATURES = ('mean', 'std', 'ptp', 'q10', 'q50', 'q90', 'mean_abs_ac',
                    'diff_rms', 'skew', 'excess_kurtosis')
FEATURE_NAMES = tuple(f'{channel}_{name}' for channel in ('I', 'Q') for name in CHANNEL_FEATURES) + (
    'IQ_covariance', 'IQ_correlation', 'trajectory_length', 'endpoint_distance', 'signed_closed_area')


def extract_features(signals):
    """Return float64 [B,25], retaining DC means; never fit or modify inputs."""
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 3 or not len(x) or x.shape[1] < 2 or x.shape[2] != 2 or not np.isfinite(x).all():
        raise ValueError('Features require finite, nonempty B,T,2 signals with T>=2')
    mean = x[:, 0] + (x - x[:, :1]).mean(axis=1)
    ac = x - mean[:, None]
    variance = np.mean(ac ** 2, axis=1)
    std = np.sqrt(variance)
    unit = np.divide(ac, std[:, None], out=np.zeros_like(ac), where=std[:, None] > 0)
    skew = np.mean(unit ** 3, axis=1)
    kurtosis = np.where(std > 0, np.mean(unit ** 4, axis=1) - 3, 0)
    q10, q50, q90 = np.quantile(x, [0.1, 0.5, 0.9], axis=1)
    delta = np.diff(x, axis=1)
    channel = np.stack((mean, std, np.ptp(x, axis=1), q10, q50, q90,
                        np.mean(np.abs(ac), axis=1), np.sqrt(np.mean(delta ** 2, axis=1)),
                        skew, kurtosis), axis=-1).reshape(len(x), 20)
    covariance = np.mean(ac[:, :, 0] * ac[:, :, 1], axis=1)
    denominator = std[:, 0] * std[:, 1]
    correlation = np.divide(covariance, denominator, out=np.zeros(len(x)), where=denominator > 0)
    length = np.linalg.norm(delta, axis=-1).sum(axis=1)
    endpoint = np.linalg.norm(x[:, -1] - x[:, 0], axis=-1)
    following = np.roll(ac, -1, axis=1)
    area = 0.5 * np.sum(ac[:, :, 0] * following[:, :, 1] - ac[:, :, 1] * following[:, :, 0], axis=1)
    result = np.column_stack((channel, covariance, correlation, length, endpoint, area))
    if not np.isfinite(result).all():
        raise ValueError('Feature calculation overflowed')
    return result
