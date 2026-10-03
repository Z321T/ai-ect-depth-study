"""Device selection and deterministic seeds, without a hardware survey."""

import os
import random

import numpy as np
import torch


def resolve_device(request: str) -> torch.device:
    """Select cpu/cuda from an explicit request or CUDA availability."""
    if request not in ('auto', 'cpu', 'cuda'):
        raise ValueError("device request must be 'auto', 'cpu' or 'cuda'")
    if request == 'cpu':
        return torch.device('cpu')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if torch.cuda.is_available():
        return torch.device('cuda')
    if request == 'cuda':
        raise RuntimeError('CUDA requested but CUDA is not available')
    return torch.device('cpu')


def seed_everything(seed: int, threads: int = 1) -> None:
    """Seed Python/NumPy/Torch and configure deterministic training.

    Call before creating CUDA tensors or initializing CUDA. The NumPy seed
    range is used for all three RNGs. CPU/GPU results need not be bitwise equal.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError('seed must be an integer in [0, 2**32 - 1]')
    if not 0 <= seed < 2 ** 32:
        raise ValueError('seed must be an integer in [0, 2**32 - 1]')
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise TypeError('threads must be an integer >= 1')
    if threads < 1:
        raise ValueError('threads must be an integer >= 1')

    # torch.manual_seed also schedules lazy CUDA seeds, so configure cuBLAS first.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
