"""CPU acceptance of the registered paper_components_v1 network blueprint."""

import io
import math
import unittest

import torch
from torch import nn


class PaperModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def build(self, **kwargs):
        from src.ect.paper_models import PaperResNeXt1D
        return PaperResNeXt1D(**kwargs).cpu()

    @staticmethod
    def main_convs(block):
        return [m for m in block.main if isinstance(m, nn.Conv1d)]

    def test_default_interface_and_parameter_counts(self):
        for blocks, expected in ((3, 134654), (2, 93754)):
            with self.subTest(blocks=blocks):
                model = self.build(blocks_per_stage=blocks)
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
                self.assertEqual(len(model.stages), 4)
                self.assertTrue(all(len(stage) == blocks for stage in model.stages))
        self.assertEqual(sum(p.numel() for p in self.build().parameters()), 134654)

    def test_forward_returns_float32_unbounded_logits(self):
        for blocks in (2, 3):
            for classes in (2, 7, 20):
                with self.subTest(blocks=blocks, classes=classes):
                    model = self.build(num_classes=classes, blocks_per_stage=blocks).eval()
                    with torch.no_grad():
                        model.classifier.weight.zero_()
                        model.classifier.bias.copy_(torch.linspace(-3, 3, classes))
                        output = model(torch.randn(3, 2, 224))
                    self.assertEqual(output.shape, (3, classes))
                    self.assertEqual(output.dtype, torch.float32)
                    torch.testing.assert_close(output, model.classifier.bias.expand(3, -1))

    def test_stem_and_stage_dimensions(self):
        for blocks in (2, 3):
            with self.subTest(blocks=blocks):
                model = self.build(blocks_per_stage=blocks).eval()
                observed = []
                modules = [model.stem, model.stem_pool, *model.stages]
                handles = [m.register_forward_hook(
                    lambda m, args, out: observed.append(tuple(out.shape))) for m in modules]
                try:
                    with torch.no_grad():
                        model(torch.randn(2, 2, 224))
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(observed, [(2, 6, 224), (2, 6, 112),
                                            (2, 20, 112), (2, 40, 56),
                                            (2, 80, 28), (2, 160, 14)])

    def test_blueprint_channels_groups_strides_and_preactivation_order(self):
        for blocks in (2, 3):
            model = self.build(blocks_per_stage=blocks)
            self.assertEqual(len(model.stem), 3)
            self.assertIsInstance(model.stem[0], nn.Conv1d)
            self.assertIsInstance(model.stem[1], nn.BatchNorm1d)
            self.assertIsInstance(model.stem[2], nn.ReLU)
            stem = model.stem[0]
            self.assertEqual((stem.in_channels, stem.out_channels, stem.kernel_size,
                              stem.stride, stem.groups), (2, 6, (3,), (1,), 1))
            self.assertIsInstance(model.stem_pool, nn.MaxPool1d)
            self.assertEqual((model.stem_pool.kernel_size, model.stem_pool.stride), (3, 2))
            previous = 6
            for s, (stage, width) in enumerate(zip(model.stages, (10, 20, 40, 80))):
                for b, block in enumerate(stage):
                    stride = 2 if s > 0 and b == 0 else 1
                    prefix = [] if (s, b) == (0, 0) else [nn.BatchNorm1d, nn.ReLU]
                    types = prefix + [nn.Conv1d, nn.BatchNorm1d, nn.ReLU,
                                      nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.Conv1d]
                    self.assertEqual(len(block.main), len(types))
                    for module, expected_type in zip(block.main, types):
                        self.assertIsInstance(module, expected_type)
                    if prefix:
                        self.assertEqual(block.main[0].num_features, previous)
                    convs = self.main_convs(block)
                    self.assertEqual([(c.in_channels, c.out_channels, c.kernel_size,
                                       c.stride, c.groups) for c in convs], [
                        (previous, width, (1,), (1,), 1),
                        (width, width, (3,), (stride,), 5),
                        (width, 2 * width, (1,), (1,), 1)])
                    norms = [m.num_features for m in block.main
                             if isinstance(m, nn.BatchNorm1d)]
                    self.assertEqual(norms, ([previous] if prefix else []) + [width, width])
                    if b == 0:
                        self.assertIsInstance(block.shortcut, nn.Conv1d)
                        self.assertEqual((block.shortcut.in_channels, block.shortcut.out_channels,
                                          block.shortcut.kernel_size, block.shortcut.stride),
                                         (previous, 2 * width, (1,), (stride,)))
                    else:
                        self.assertIsInstance(block.shortcut, nn.Identity)
                    previous = 2 * width
            self.assertEqual(len(model.final), 2)
            self.assertIsInstance(model.final[0], nn.BatchNorm1d)
            self.assertEqual(model.final[0].num_features, 160)
            self.assertIsInstance(model.final[1], nn.ReLU)
            self.assertIsInstance(model.pool, nn.AdaptiveAvgPool1d)
            self.assertEqual(model.pool.output_size, 1)
            self.assertEqual((model.classifier.in_features, model.classifier.out_features), (160, 20))

    def test_runtime_main_layer_order_and_raw_input_projection(self):
        model = self.build().eval()
        for stage in model.stages:
            for b, block in enumerate(stage):
                observed = []
                projection_inputs = []
                handles = [module.register_forward_hook(
                    lambda m, args, out, index=i: observed.append(index))
                    for i, module in enumerate(block.main)]
                if b == 0:
                    handles.append(block.shortcut.register_forward_pre_hook(
                        lambda m, args: projection_inputs.append(args[0].detach().clone())))
                cin = self.main_convs(block)[0].in_channels
                inputs = -torch.rand(2, cin, 8)
                # Ensure preactivation differs materially from the raw input.
                if isinstance(block.main[0], nn.BatchNorm1d):
                    with torch.no_grad():
                        block.main[0].bias.fill_(2)
                try:
                    with torch.no_grad():
                        block(inputs)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(observed, list(range(len(block.main))))
                if b == 0:
                    self.assertEqual(len(projection_inputs), 1)
                    torch.testing.assert_close(projection_inputs[0], inputs, rtol=0, atol=0)

    def test_residual_addition_preserves_negative_identity_and_projection(self):
        model = self.build().eval()
        for stage in model.stages:
            for b, block in enumerate(stage):
                cin = self.main_convs(block)[0].in_channels
                inputs = -torch.arange(1, 1 + 2 * cin * 8, dtype=torch.float32).reshape(2, cin, 8)
                with torch.no_grad():
                    for conv in self.main_convs(block):
                        conv.weight.zero_()
                        conv.bias.zero_()
                    if b == 0:
                        block.shortcut.weight.zero_()
                        block.shortcut.bias.zero_()
                        block.shortcut.weight[:, 0, 0] = 1
                        stride = block.shortcut.stride[0]
                        expected = inputs[:, :1, ::stride].expand(-1, block.shortcut.out_channels, -1)
                    else:
                        expected = inputs
                    output = block(inputs)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                self.assertTrue((output < 0).all())

    def test_same_stem_conv_hand_computed_boundary_alignment(self):
        conv = self.build().stem[0]
        with torch.no_grad():
            conv.weight.zero_()
            conv.bias.zero_()
            conv.weight[0, 0] = torch.tensor([1., 10., 100.])
            inputs = torch.tensor([[[1., 2., 3., 4.], [0., 0., 0., 0.]]])
            # Windows [0,1,2], [1,2,3], [2,3,4], [3,4,0].
            actual = conv(inputs)[0, 0]
        torch.testing.assert_close(actual, torch.tensor([210., 321., 432., 43.]), rtol=0, atol=0)

    def test_same_group_conv_stride_two_hand_computed_even_and_odd(self):
        conv = self.main_convs(self.build().stages[1][0])[1]
        with torch.no_grad():
            conv.weight.zero_()
            conv.bias.zero_()
            conv.weight[0, 0] = torch.tensor([1., 10., 100.])
            # Even: [1,2,3], [3,4,0]; odd: [0,1,2], [2,3,4], [4,5,0].
            for signal, expected in (([1., 2., 3., 4.], [321., 43.]),
                                     ([1., 2., 3., 4., 5.], [210., 432., 54.]),
                                     ([7.], [70.])):
                inputs = torch.zeros(1, conv.in_channels, len(signal))
                inputs[0, 0] = torch.tensor(signal)
                torch.testing.assert_close(conv(inputs)[0, 0], torch.tensor(expected), rtol=0, atol=0)

    def test_same_pool_hand_computed_negative_boundary_values(self):
        pool = self.build().stem_pool
        # Negative edges detect zero padding; odd lengths need left padding.
        for signal, expected in (([-5., -4., -3., -2.], [-3., -2.]),
                                 ([-5., -4., -3., -2., -1.], [-4., -2., -1.]),
                                 ([-7.], [-7.])):
            with self.subTest(signal=signal):
                actual = pool(torch.tensor([[signal]]))
                torch.testing.assert_close(actual, torch.tensor([[expected]]), rtol=0, atol=0)

    def test_projection_stride_two_samples_raw_even_positions(self):
        conv = self.build().stages[1][0].shortcut
        with torch.no_grad():
            conv.weight.zero_()
            conv.bias.zero_()
            conv.weight[:, 0, 0] = 1
            for signal, expected in (([1., 2., 3., 4.], [1., 3.]),
                                     ([1., 2., 3., 4., 5.], [1., 3., 5.])):
                inputs = torch.zeros(1, conv.in_channels, len(signal))
                inputs[0, 0] = torch.tensor(signal)
                output = conv(inputs)
                torch.testing.assert_close(output, torch.tensor([[expected]]).expand_as(output),
                                           rtol=0, atol=0)

    def test_bias_xavier_bounds_and_batch_norm_initialization(self):
        for module in self.build().modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                self.assertIsNotNone(module.bias)
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
                shape = module.weight.shape
                receptive = math.prod(shape[2:])
                bound = math.sqrt(6 / ((shape[0] + shape[1]) * receptive))
                self.assertLessEqual(module.weight.abs().max().item(), bound)
                self.assertGreater(module.weight.std().item(), 0)
            elif isinstance(module, nn.BatchNorm1d):
                self.assertEqual(module.eps, 1e-3)
                self.assertEqual(module.momentum, 0.01)
                self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
                self.assertTrue(torch.equal(module.running_mean, torch.zeros_like(module.running_mean)))
                self.assertTrue(torch.equal(module.running_var, torch.ones_like(module.running_var)))

    def test_invalid_class_count_is_rejected(self):
        for value in (True, False, 2.0, '20', None):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, 'num_classes'):
                self.build(num_classes=value)
        for value in (-1, 0, 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'num_classes'):
                self.build(num_classes=value)

    def test_invalid_blocks_per_stage_is_rejected(self):
        for value in (True, False, 2.0, '3', None):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, 'blocks_per_stage'):
                self.build(blocks_per_stage=value)
        for value in (-1, 0, 1, 4):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'blocks_per_stage'):
                self.build(blocks_per_stage=value)

    def test_invalid_input_shapes_are_rejected(self):
        model = self.build()
        for shape in ((224, 2), (2, 224, 2), (2, 1, 224), (2, 2, 223),
                      (2, 2, 250), (2, 2, 224, 1), (0, 2, 224)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, '224'):
                model(torch.zeros(shape))

    def test_non_float32_inputs_are_rejected(self):
        model = self.build()
        for dtype in (torch.float64, torch.float16, torch.bfloat16, torch.int64, torch.bool):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(TypeError, 'float32'):
                model(torch.zeros(2, 2, 224, dtype=dtype))

    def test_non_tensor_inputs_are_rejected(self):
        model = self.build()
        for value in (None, [], 1, 'signals'):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, 'Tensor'):
                model(value)

    def test_nonfinite_inputs_rejected_before_batch_norm_state_changes(self):
        model = self.build().train()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for value in (float('nan'), float('inf'), -float('inf')):
            inputs = torch.zeros(2, 2, 224)
            inputs[1, 1, -1] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite'):
                model(inputs)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)

    def test_cpu_gradients_and_optimizer_update_main_and_shortcut_weights(self):
        for blocks in (2, 3):
            with self.subTest(blocks=blocks):
                torch.manual_seed(20261002)
                model = self.build(num_classes=5, blocks_per_stage=blocks).train()
                before = {k: v.detach().clone() for k, v in model.named_parameters()}
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                inputs = torch.randn(4, 2, 224, requires_grad=True)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1, 2, 3]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertGreater(inputs.grad.abs().sum().item(), 0)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if parameter.ndim > 1:
                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                optimizer.step()
                for name, parameter in model.named_parameters():
                    if parameter.ndim > 1:
                        self.assertFalse(torch.equal(parameter.detach(), before[name]), name)

    def test_cpu_state_roundtrip_after_real_training(self):
        for blocks in (2, 3):
            with self.subTest(blocks=blocks):
                torch.manual_seed(17)
                model = self.build(num_classes=7, blocks_per_stage=blocks).train()
                inputs = torch.randn(4, 2, 224)
                optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1, 2, 3])).backward()
                optimizer.step()
                norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
                self.assertTrue(all(m.num_batches_tracked.item() == 1 for m in norms))
                model.eval()
                with torch.no_grad():
                    expected = model(inputs)
                payload = io.BytesIO()
                torch.save(model.state_dict(), payload)
                payload.seek(0)
                state = torch.load(payload, map_location='cpu', weights_only=True)
                restored = self.build(num_classes=7, blocks_per_stage=blocks).eval()
                restored.load_state_dict(state, strict=True)
                for key, value in restored.state_dict().items():
                    self.assertEqual(value.device.type, 'cpu')
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
                with torch.no_grad():
                    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
