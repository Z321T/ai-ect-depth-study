"""Exhaustive, label-free train/validation waveform candidate diagnostics.

Rows refer to the supplied arrays (zero based). Undefined distances are NaN,
with index -1 and no candidates. Thresholds are inclusive. Candidate counts
count all reference pairs within each threshold, not just the nearest pair.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial.distance import cdist


def _signals(values):
    values = np.asarray(values)
    if (values.ndim != 3 or values.shape[2] != 2 or min(values.shape[:2]) < 1
            or values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
        raise ValueError("Finite real signals with nonempty shape [N,T,2] required")
    return np.asarray(values, dtype=np.float64)


def center_iq(signals):
    """Per-channel time centering in float64, removing large DC before summing."""
    values = _signals(signals)
    shifted = values - values[:, :1, :]
    centered = shifted - shifted.mean(axis=1, keepdims=True, dtype=np.float64)
    if not np.isfinite(centered).all():
        raise ValueError("Signal range exceeds float64 centering capacity")
    return centered


def _unit_ac(values):
    centered = center_iq(values).reshape(len(values), -1)
    norms = np.sqrt(np.sum(centered * centered, axis=1))
    if not np.isfinite(norms).all():
        raise ValueError("AC norm exceeds float64 capacity")
    valid = norms > 0
    unit = np.zeros_like(centered)
    unit[valid] = centered[valid] / norms[valid, None]
    return unit, norms, valid


@dataclass
class NearestNeighbors:
    raw_index: np.ndarray
    raw_distance: np.ndarray
    shape_index: np.ndarray
    shape_distance: np.ndarray
    raw_candidate_count: np.ndarray
    shape_candidate_count: np.ndarray
    query_ac_norm: np.ndarray
    reference_ac_norm: np.ndarray
    elapsed_seconds: float


def nearest_neighbors(queries, references, *, query_block=256, reference_block=2048,
                      raw_threshold=.001, shape_threshold=.01, progress=None):
    """Search every reference without labels, shift, sign or channel scaling.

    Raw: direct Euclidean difference in input units / joint query AC norm.
    Shape: joint unit L2 after separate time means in the two channels.
    Dot products only screen shape pairs; all possible block minima (including
    rounding ties) and threshold candidates are evaluated by direct differences.
    Exact computed ties select the smallest original reference row. A callback,
    if supplied, receives (queries_completed, query_count, elapsed_seconds).
    Memory is O((N+M)*T + query_block*reference_block), never O(N*M).
    """
    started = time.perf_counter()
    for size in (query_block, reference_block):
        if type(size) is not int or size < 1:
            raise ValueError("Positive integer block sizes required")
    for threshold in (raw_threshold, shape_threshold):
        if not np.isscalar(threshold) or not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Finite nonnegative distance thresholds required")
    q, r = _signals(queries), _signals(references)
    if q.shape[1:] != r.shape[1:]:
        raise ValueError("Query/reference time and channel shapes must match")
    uq, nq, vq = _unit_ac(q)
    ur, nr, vr = _unit_ac(r)
    qflat, rflat = q.reshape(len(q), -1), r.reshape(len(r), -1)
    raw_index = np.full(len(q), -1, dtype=np.int64)
    shape_index = raw_index.copy()
    raw_distance = np.full(len(q), np.inf)
    shape_distance = raw_distance.copy()
    raw_counts = np.zeros(len(q), dtype=np.int64)
    shape_counts = raw_counts.copy()
    # Conservative O(d*eps) absolute error allowance for the squared distance
    # dot formula on unit vectors, including normalization and summation error.
    screen_error = 32 * np.finfo(np.float64).eps * uq.shape[1]
    q2 = np.sum(uq * uq, axis=1)
    r2 = np.sum(ur * ur, axis=1)
    for qstart in range(0, len(q), query_block):
        qstop = min(qstart + query_block, len(q))
        active = np.flatnonzero(vq[qstart:qstop]) + qstart
        if len(active):
            for rstart in range(0, len(r), reference_block):
                rstop = min(rstart + reference_block, len(r))
                # cdist euclidean accumulates actual coordinate differences;
                # no ||q||^2+||r||^2-2*q.r cancellation in large DC signals.
                raw = cdist(qflat[active], rflat[rstart:rstop], metric="euclidean")
                raw /= nq[active, None]
                if not np.isfinite(raw).all():
                    raise ValueError("Raw relative distance exceeds float64 capacity")
                indices = np.argmin(raw, axis=1)
                distances = raw[np.arange(len(active)), indices]
                improved = distances < raw_distance[active]
                raw_distance[active[improved]] = distances[improved]
                raw_index[active[improved]] = indices[improved] + rstart
                raw_counts[active] += np.count_nonzero(raw <= raw_threshold, axis=1)

                valid_rows = np.flatnonzero(vr[rstart:rstop]) + rstart
                if not len(valid_rows):
                    continue
                approx = (q2[active, None] + r2[valid_rows][None, :]
                          - 2 * (uq[active] @ ur[valid_rows].T))
                minima = np.min(approx, axis=1)
                for local, qi in enumerate(active):
                    # Any actual block minimum can be at most 2*error above
                    # the approximate minimum. Also screen all threshold pairs.
                    cutoff = max(minima[local] + 2 * screen_error,
                                 shape_threshold ** 2 + screen_error)
                    candidates = valid_rows[approx[local] <= cutoff]
                    delta = uq[qi] - ur[candidates]
                    exact = np.sqrt(np.sum(delta * delta, axis=1))
                    best = int(np.argmin(exact))
                    if exact[best] < shape_distance[qi]:
                        shape_distance[qi] = exact[best]
                        shape_index[qi] = candidates[best]
                    shape_counts[qi] += np.count_nonzero(exact <= shape_threshold)
        if progress is not None:
            progress(qstop, len(q), time.perf_counter() - started)
    raw_distance[raw_index < 0] = np.nan
    shape_distance[shape_index < 0] = np.nan
    return NearestNeighbors(raw_index, raw_distance, shape_index, shape_distance,
                            raw_counts, shape_counts, nq, nr,
                            time.perf_counter() - started)
