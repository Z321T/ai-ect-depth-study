"""Identity-keyed random 250-to-224 crops for paper component reconstruction.

The project RNG protocol is compact ASCII JSON [namespace, seed, sample_id,
epoch], SHA256, and the full digest interpreted as a big-endian PCG64 seed.
Only ``training`` (one crop, nonnegative epoch) and ``validation`` (ten crops,
epoch=None) are supported. CPU NumPy draws uniform integer starts in [0, 26]
with replacement, independent of batch order, partitioning and model device.

Averaging ten per-crop softmax probabilities is a project engineering
assumption: it is not a claim about the original authors' unpublished code.
The identity RNG protocol is likewise a project reproducibility choice.
"""
import copy
import hashlib
import json
from numbers import Integral

import numpy as np
import torch
from torch import nn

from .preprocessing import standardize


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _identities(sample_ids):
    if isinstance(sample_ids, (str, bytes)):
        raise ValueError('sample_ids must be a nonempty sequence of strings')
    try:
        ids = list(sample_ids)
    except TypeError as exc:
        raise ValueError('sample_ids must be a nonempty sequence of strings') from exc
    if not ids or any(not isinstance(identity, str) or not identity.strip() for identity in ids):
        raise ValueError('sample_ids must contain nonblank strings')
    return [str(identity) for identity in ids]


def _raw_inputs(raw, sample_ids):
    try:
        values = np.asarray(raw)
        if values.dtype.kind not in 'fiu':
            raise ValueError('raw must contain real numeric values')
        if values.ndim != 3 or values.shape[1:] != (250, 2) or not len(values):
            raise ValueError('raw must have nonempty shape [B,250,2]')
        if not np.isfinite(values).all():
            raise ValueError('raw must contain finite values')
        with np.errstate(over='raise', invalid='raise'):
            values = values.astype(np.float32, copy=False)
    except (TypeError, OverflowError, FloatingPointError) as exc:
        raise ValueError('raw must be finite and representable as float32') from exc
    if not np.isfinite(values).all():
        raise ValueError('raw must be finite and representable as float32')
    ids = _identities(sample_ids)
    if len(ids) != len(values):
        raise ValueError('sample_ids length must equal B')
    return values, ids


def crop_starts(sample_ids, seed, namespace, epoch=None):
    """Return int64 [B,1] training or [B,10] validation starts in [0,26].

    Seed is an unsigned 32-bit integer. Duplicate identities deliberately
    share starts; string contents are retained without trimming or hashing
    with Python's process-dependent hash. There is no test namespace.
    """
    ids = _identities(sample_ids)
    seed = _integer(seed, 'seed')
    if seed >= 2**32:
        raise ValueError('seed must be uint32')
    if not isinstance(namespace, str) or namespace not in ('training', 'validation'):
        raise ValueError('namespace must be training or validation')
    if namespace == 'training':
        epoch = _integer(epoch, 'epoch')
        count = 1
    else:
        if epoch is not None:
            raise ValueError('validation requires epoch=None')
        count = 10
    starts = np.empty((len(ids), count), dtype=np.int64)
    for row, identity in enumerate(ids):
        key = json.dumps([namespace, seed, identity, epoch], ensure_ascii=True,
                         separators=(',', ':'), allow_nan=False).encode('ascii')
        entropy = int.from_bytes(hashlib.sha256(key).digest(), 'big')
        rng = np.random.Generator(np.random.PCG64(entropy))
        starts[row] = rng.integers(0, 27, size=count, dtype=np.int64)
    return starts


def _extract_crops(values, starts):
    points = starts[:, :, None] + np.arange(224)
    # Advanced indexing produces a copy, including for readonly/mapped input.
    return values[np.arange(len(values))[:, None, None], points, :]


def training_crops(raw, sample_ids, seed, epoch):
    """Return float32 [B,224,2], without altering raw or removing its DC."""
    values, ids = _raw_inputs(raw, sample_ids)
    starts = crop_starts(ids, seed, 'training', epoch)
    return _extract_crops(values, starts)[:, 0]


def validation_crops(raw, sample_ids, seed=10):
    """Return float32 [B,10,224,2] random crops; repeats are permitted."""
    values, ids = _raw_inputs(raw, sample_ids)
    starts = crop_starts(ids, seed, 'validation')
    return _extract_crops(values, starts)


def _model_device(model):
    # Honor an explicit device property; ordinary nn.Modules expose the device
    # through parameters/buffers. Neither route probes hardware availability.
    if hasattr(model, 'device'):
        return torch.device(model.device)
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    return torch.device('cpu')


def ten_crop_probabilities(model, raw, sample_ids, normalization, seed=10, batch_size=128):
    """Return NumPy [B,C] means of ten per-crop softmax probabilities.

    A deep copy runs in eval mode under inference_mode on the model's device.
    The original's parameters, buffers, gradients and all submodule modes
    remain intact, also when validation or inference fails. Existing channel
    standardization is applied before conversion to float32 [N,2,224].
    ``batch_size`` caps the number of crops in each model forward call.
    This validation component neither accepts labels nor exposes test access.
    """
    batch_size = _integer(batch_size, 'batch_size', minimum=1)
    values, ids = _raw_inputs(raw, sample_ids)
    starts = crop_starts(ids, seed, 'validation')
    if not isinstance(model, nn.Module):
        raise ValueError('model must be a torch.nn.Module')
    cloned = copy.deepcopy(model).eval()
    for tensor in list(cloned.parameters()) + list(cloned.buffers()):
        if not torch.isfinite(tensor).all():
            raise ValueError('model parameters and buffers must be finite')
    device = _model_device(cloned)
    result = None
    with torch.inference_mode():
        for begin in range(0, len(values), batch_size):
            end = min(begin + batch_size, len(values))
            crops = _extract_crops(values[begin:end], starts[begin:end]).reshape(-1, 224, 2)
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    normalized = standardize(crops, normalization)
            except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
                raise ValueError('Invalid normalization or nonfinite standardized crops') from exc
            if not np.isfinite(normalized).all():
                raise ValueError('Standardized crops must be finite')
            crop_probabilities = []
            for offset in range(0, len(crops), batch_size):
                batch = normalized[offset:offset + batch_size]
                inputs = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 2, 1))).to(device)
                logits = cloned(inputs)
                if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
                        or logits.shape[0] != len(batch) or logits.shape[1] < 2
                        or not logits.is_floating_point()):
                    raise ValueError('Model must return floating classification logits [N,C], C>=2')
                if not torch.isfinite(logits).all():
                    raise ValueError('Model logits must be finite')
                if result is None:
                    result = np.empty((len(values), logits.shape[1]), dtype=np.float32)
                if logits.shape[1] != result.shape[1]:
                    raise ValueError('Model class count changed across batches')
                probabilities = torch.softmax(logits, dim=1)
                if not torch.isfinite(probabilities).all():
                    raise ValueError('Crop probabilities must be finite')
                crop_probabilities.append(probabilities.to(dtype=torch.float32).cpu().numpy())
            probabilities = np.concatenate(crop_probabilities).reshape(end - begin, 10, -1)
            result[begin:end] = probabilities.mean(axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Averaged probabilities must be finite')
    return result
