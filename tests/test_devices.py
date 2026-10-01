import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


class DeviceTests(unittest.TestCase):
    def module(self):
        from src.ect import devices
        return devices

    def setUp(self):
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state()
        self.threads = torch.get_num_threads()
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.benchmark = torch.backends.cudnn.benchmark
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        self.matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.cudnn_tf32 = torch.backends.cudnn.allow_tf32
        self.workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')

    def tearDown(self):
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_rng)
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(self.deterministic, warn_only=self.warn_only)
        torch.backends.cudnn.benchmark = self.benchmark
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = self.matmul_tf32
        torch.backends.cudnn.allow_tf32 = self.cudnn_tf32
        if self.workspace is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = self.workspace

    def test_cpu_request_does_not_probe_or_initialize_cuda(self):
        module = self.module()
        with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CPU probed CUDA')):
            device = module.resolve_device('cpu')
        self.assertIsInstance(device, torch.device)
        self.assertEqual(device, torch.device('cpu'))

    def test_auto_selects_cpu_when_cuda_unavailable(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            self.assertEqual(self.module().resolve_device('auto'), torch.device('cpu'))

    def test_cuda_requests_select_generic_cuda_when_available(self):
        with patch.object(torch.cuda, 'is_available', return_value=True):
            for request in ('auto', 'cuda'):
                with self.subTest(request=request):
                    device = self.module().resolve_device(request)
                    self.assertEqual(device, torch.device('cuda'))
                    self.assertIsNone(device.index)

    def test_explicit_unavailable_cuda_raises_clear_error(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, r'CUDA.*(unavailable|not available)'):
                self.module().resolve_device('cuda')

    def test_invalid_requests_are_rejected_before_cuda_probe(self):
        with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('Invalid request probed CUDA')):
            for request in ('gpu', 'CUDA', 'cuda:0', '', None, 0):
                with self.subTest(request=request), self.assertRaises(ValueError):
                    self.module().resolve_device(request)

    def test_seeding_repeats_real_python_numpy_and_torch_draws(self):
        module = self.module()
        def draw():
            return random.random(), np.random.normal(size=8), torch.randn(8)
        module.seed_everything(123)
        first = draw()
        module.seed_everything(123)
        repeated = draw()
        self.assertEqual(first[0], repeated[0])
        np.testing.assert_array_equal(first[1], repeated[1])
        torch.testing.assert_close(first[2], repeated[2], rtol=0, atol=0)
        module.seed_everything(124)
        different = draw()
        self.assertNotEqual(first[0], different[0])
        self.assertFalse(np.array_equal(first[1], different[1]))
        self.assertFalse(torch.equal(first[2], different[2]))

    def test_seed_enforces_threads_and_determinism_flags(self):
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        self.module().seed_everything(0, threads=2)
        self.assertEqual(torch.get_num_threads(), 2)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.module().seed_everything(0)
        self.assertEqual(torch.get_num_threads(), 1)

    def test_workspace_config_precedes_cuda_seeding_in_fresh_process(self):
        # torch.manual_seed itself schedules a lazy CUDA seed. Observe that
        # boundary while preserving the real call and avoiding GPU initialization.
        code = '''
import json, os, torch
from unittest.mock import patch
from src.ect.devices import seed_everything
os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
observed = []
original = torch.cuda.manual_seed_all
def observe(seed):
    observed.append(os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    return original(seed)
with patch.object(torch.cuda, 'is_available', return_value=False):
    with patch.object(torch.cuda, 'manual_seed_all', side_effect=observe):
        seed_everything(13)
print(json.dumps({'observed': observed, 'initialized': torch.cuda.is_initialized()}))
'''
        completed = subprocess.run([sys.executable, '-c', code],
                                   cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, check=True, timeout=60)
        result = json.loads(completed.stdout)
        self.assertTrue(result['observed'])
        self.assertTrue(all(value == ':4096:8' for value in result['observed']))
        self.assertFalse(result['initialized'])

    def test_cpu_only_seeding_does_not_initialize_cuda(self):
        module = self.module()
        # Torch's CPU seed API may queue CUDA seeds lazily; it must not initialize CUDA.
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('Initialized CUDA')):
                module.seed_everything(9)
                self.assertEqual(torch.initial_seed(), 9)

    def test_invalid_seed_and_threads_are_rejected(self):
        for seed in (True, 1.0, '1', None):
            with self.subTest(seed=seed), self.assertRaises(TypeError):
                self.module().seed_everything(seed)
        for seed in (-1, 2 ** 32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.module().seed_everything(seed)
        for threads in (True, 1.0, '1', None):
            with self.subTest(threads=threads), self.assertRaises(TypeError):
                self.module().seed_everything(0, threads=threads)
        for threads in (0, -1):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                self.module().seed_everything(0, threads=threads)

    @unittest.skipUnless(torch.cuda.is_available(), 'Real CUDA unavailable; GPU acceptance not verified')
    def test_real_cuda_random_draws_are_reproducible(self):
        module = self.module()
        module.seed_everything(123)
        first = torch.randn(16, device=module.resolve_device('cuda'))
        module.seed_everything(123)
        torch.testing.assert_close(torch.randn(16, device='cuda'), first, rtol=0, atol=0)
