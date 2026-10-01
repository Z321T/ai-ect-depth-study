import io
import unittest

import torch
from torch import nn


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def build(self, *args, **kwargs):
        from src.ect.models import build_model
        return build_model(*args, **kwargs)

    def test_logits_shape_dtype_and_unbounded_scores(self):
        for name in ('cnn', 'resnet'):
            for classes in (2, 7, 20):
                with self.subTest(name=name, classes=classes):
                    model = self.build(name=name, num_classes=classes).eval()
                    self.assertIsInstance(model, nn.Module)
                    head = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
                    with torch.no_grad():
                        head.weight.zero_()
                        head.bias.copy_(torch.linspace(-3, 3, classes))
                        output = model(torch.randn(3, 2, 250))
                    self.assertEqual(output.shape, (3, classes))
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(output).all())
                    torch.testing.assert_close(output, head.bias.expand(3, -1))

    def test_default_factory_is_twenty_class_cnn(self):
        model = self.build()
        self.assertEqual(model(torch.randn(1, 2, 250)).shape, (1, 20))
        self.assertEqual(sum(isinstance(m, nn.Conv1d) for m in model.modules()), 3)

    def test_cpu_backward_and_optimizer_really_update_every_parameter(self):
        for name in ('cnn', 'resnet'):
            with self.subTest(name=name):
                torch.manual_seed(20261001)
                model = self.build(name, num_classes=5).cpu().train()
                before = {key: p.detach().clone() for key, p in model.named_parameters()}
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                inputs = torch.randn(4, 2, 250, requires_grad=True)
                logits = model(inputs)
                loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
                self.assertTrue(torch.isfinite(loss))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertGreater(inputs.grad.abs().sum().item(), 0)
                for key, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, key)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, key)
                optimizer.step()
                for key, parameter in model.named_parameters():
                    self.assertFalse(torch.equal(before[key], parameter.detach()), key)

    def test_cpu_eval_state_dict_roundtrip_after_training(self):
        for name in ('cnn', 'resnet'):
            with self.subTest(name=name):
                torch.manual_seed(17)
                model = self.build(name, num_classes=7).train()
                inputs = torch.randn(4, 2, 250)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1, 2, 3])).backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    expected = model(inputs)
                payload = io.BytesIO()
                torch.save({key: value.cpu() for key, value in model.state_dict().items()}, payload)
                payload.seek(0)
                state = torch.load(payload, weights_only=True, map_location='cpu')
                restored = self.build(name, num_classes=7).cpu().eval()
                restored.load_state_dict(state, strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
                self.assertTrue(all(value.device.type == 'cpu' for value in state.values()))

    def test_cnn_has_frozen_blocks_and_time_lengths(self):
        model = self.build('cnn').eval()
        children = [m for m in model.modules() if not list(m.children())]
        self.assertEqual([type(m) for m in children],
                         [nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.MaxPool1d] * 3
                         + [nn.AdaptiveAvgPool1d, nn.Linear])
        convolutions = [m for m in children if isinstance(m, nn.Conv1d)]
        self.assertEqual([(m.in_channels, m.out_channels) for m in convolutions],
                         [(2, 32), (32, 64), (64, 128)])
        for conv in convolutions:
            self.assertEqual((conv.kernel_size, conv.padding, conv.stride), ((5,), (2,), (1,)))
            self.assertIsNone(conv.bias)
        lengths = []
        handles = [m.register_forward_hook(lambda m, args, out: lengths.append(out.shape[-1]))
                   for m in children if isinstance(m, nn.MaxPool1d)]
        try:
            model(torch.randn(2, 2, 250))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(lengths, [125, 62, 31])

    def test_resnet_convolutions_and_stage_lengths(self):
        model = self.build('resnet').eval()
        convolutions = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
        stem = [m for m in convolutions if m.kernel_size == (7,)]
        self.assertEqual(len(stem), 1)
        self.assertEqual((stem[0].in_channels, stem[0].out_channels, stem[0].padding,
                          stem[0].stride), (2, 32, (3,), (1,)))
        main = [m for m in convolutions if m.kernel_size == (3,)]
        self.assertEqual([(m.in_channels, m.out_channels, m.stride) for m in main], [
            (32, 32, (1,)), (32, 32, (1,)), (32, 32, (1,)), (32, 32, (1,)),
            (32, 64, (2,)), (64, 64, (1,)), (64, 64, (1,)), (64, 64, (1,)),
            (64, 128, (2,)), (128, 128, (1,)), (128, 128, (1,)), (128, 128, (1,)),
        ])
        self.assertTrue(all(m.padding == (1,) for m in main))
        shortcuts = [m for m in convolutions if m.kernel_size == (1,)]
        self.assertEqual([(m.in_channels, m.out_channels, m.stride) for m in shortcuts],
                         [(32, 64, (2,)), (64, 128, (2,))])
        self.assertTrue(all(m.bias is None for m in convolutions))
        norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        self.assertEqual([m.num_features for m in norms].count(32), 5)
        self.assertEqual([m.num_features for m in norms].count(64), 5)
        self.assertEqual([m.num_features for m in norms].count(128), 5)
        observed = []
        handles = [m.register_forward_hook(
            lambda m, args, out: observed.append((out.shape[1], out.shape[2]))) for m in main]
        try:
            model(torch.randn(2, 2, 250))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(observed, [(32, 250)] * 4 + [(64, 125)] * 4 + [(128, 63)] * 4)

    def test_residual_identity_paths_survive_zero_main_convolutions(self):
        model = self.build('resnet').eval()
        # Recognize basic blocks by their two direct main convolutions, without
        # requiring a private class name or a particular stage container name.
        blocks = [m for m in model.modules()
                  if len([c for c in m.children() if isinstance(c, nn.Conv1d)
                          and c.kernel_size == (3,)]) == 2]
        self.assertEqual(len(blocks), 6)
        for block in blocks:
            main = [m for m in block.children() if isinstance(m, nn.Conv1d)]
            if main[0].in_channels != main[0].out_channels or main[0].stride != (1,):
                continue
            with torch.no_grad():
                for conv in main:
                    conv.weight.zero_()
                inputs = torch.rand(2, main[0].in_channels, 19)
                torch.testing.assert_close(block(inputs), inputs, rtol=0, atol=0)

    def test_rejects_wrong_input_layout(self):
        for name in ('cnn', 'resnet'):
            model = self.build(name)
            for shape in ((250, 2), (2, 250, 2), (2, 1, 250), (2, 2, 249), (2, 2, 250, 1)):
                with self.subTest(name=name, shape=shape), self.assertRaisesRegex(ValueError, '250'):
                    model(torch.zeros(shape))

    def test_rejects_non_float32_input(self):
        for name in ('cnn', 'resnet'):
            model = self.build(name)
            for dtype in (torch.float64, torch.float16, torch.int64):
                with self.subTest(name=name, dtype=dtype), self.assertRaisesRegex(TypeError, 'float32'):
                    model(torch.zeros(2, 2, 250, dtype=dtype))

    def test_rejects_invalid_factory_arguments(self):
        for name in ('CNN', 'resnext', '', None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.build(name)
        for classes in (True, False, 2.0, '20', None):
            with self.subTest(classes=classes), self.assertRaises(TypeError):
                self.build('cnn', classes)
        for classes in (-1, 0, 1):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                self.build('resnet', classes)

    @unittest.skipUnless(torch.cuda.is_available(), 'Real CUDA unavailable; GPU acceptance not verified')
    def test_real_cuda_backward_and_cpu_weight_reload(self):
        from src.ect.devices import resolve_device, seed_everything
        seed_everything(29)
        for name in ('cnn', 'resnet'):
            with self.subTest(name=name):
                model = self.build(name, 5).to(resolve_device('cuda')).train()
                inputs = torch.randn(4, 2, 250, device='cuda')
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                before = {key: p.detach().clone() for key, p in model.named_parameters()}
                nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1, 2, 3], device='cuda')).backward()
                optimizer.step()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                self.assertTrue(any(not torch.equal(before[key], p) for key, p in model.named_parameters()))
                model.eval()
                with torch.no_grad():
                    expected = model(inputs).cpu()
                restored = self.build(name, 5).cpu().eval()
                restored.load_state_dict({key: value.cpu() for key, value in model.state_dict().items()})
                with torch.no_grad():
                    torch.testing.assert_close(restored(inputs.cpu()), expected, rtol=1e-4, atol=1e-5)
