"""Synthetic CPU acceptance tests for the maintained Wang 100-filter CNN-RC."""

import copy
import math
import unittest

import numpy as np
import torch

from src.cnn_rc import (
    CNNRC,
    CONTRACT_ID,
    architecture_settings,
    reverse_complement,
)


SEQUENCES = ("ACGTTGCAAGCTAC", "TGCATACCGGATCA", "GATCTTACGCGATA")
TOLERANCES = {
    torch.float32: {"atol": 1e-6, "rtol": 1e-5},
    torch.float64: {"atol": 1e-10, "rtol": 1e-8},
}
SPATIAL_STAGES = (
    "convolution", "first_relu", "batch_normalization", "second_relu", "pooled",
)
INVARIANT_STAGES = ("dense", "dense_relu", "logit", "output")


def _one_hot(sequences=SEQUENCES, dtype=torch.float32):
    values = torch.zeros((len(sequences), 14, 4), dtype=dtype)
    for row, sequence in enumerate(sequences):
        for position, base in enumerate(sequence):
            values[row, position, "ACGT".index(base)] = 1
    return values


def _numpy_reference(model, values, training):
    """Express the fixed algebra independently, using explicit channel pairs."""
    parameters = {}
    for name, parameter in model.named_parameters():
        parameters[name] = parameter.detach().numpy().copy()
    running_mean = model.running_mean.detach().numpy().copy()
    running_var = model.running_var.detach().numpy().copy()
    batch_size = values.shape[0]
    convolution = np.empty((batch_size, 200, 14), dtype=np.float64)
    padded = np.pad(values, ((0, 0), (5, 5), (0, 0)))
    for channel in range(100):
        for position in range(14):
            patch = padded[:, position:position + 11, :]
            direct_kernel = parameters["W"][channel].T
            partner_kernel = parameters["W"][channel, ::-1, ::-1].T
            convolution[:, channel, position] = (
                np.sum(patch * direct_kernel, axis=(1, 2))
                + parameters["b"][channel]
            )
            convolution[:, 199 - channel, position] = (
                np.sum(patch * partner_kernel, axis=(1, 2))
                + parameters["b"][channel]
            )
    first_relu = np.maximum(convolution, 0)
    normalized = np.empty_like(convolution)
    for channel in range(100):
        mean = running_mean[channel]
        variance = running_var[channel]
        if training:
            observations = np.concatenate((
                first_relu[:, channel, :].ravel(),
                first_relu[:, 199 - channel, :].ravel(),
            ))
            mean = np.sum(observations) / observations.size
            variance = np.sum((observations - mean) ** 2) / observations.size
            running_mean[channel] = 0.99 * running_mean[channel] + 0.01 * mean
            running_var[channel] = 0.99 * running_var[channel] + 0.01 * variance
        for oriented_channel in (channel, 199 - channel):
            normalized[:, oriented_channel, :] = (
                parameters["gamma"][channel]
                * (first_relu[:, oriented_channel, :] - mean)
                / math.sqrt(variance + 0.001)
                + parameters["beta"][channel]
            )
    second_relu = np.maximum(normalized, 0)
    pooled = np.maximum(second_relu[:, :, 0::2], second_relu[:, :, 1::2])
    weighted_sum = np.empty((batch_size, 200), dtype=np.float64)
    for channel in range(100):
        weighted_sum[:, channel] = np.sum(
            pooled[:, channel, :] * parameters["A"][:, channel], axis=1,
        )
        weighted_sum[:, 199 - channel] = np.sum(
            pooled[:, 199 - channel, :] * parameters["A"][::-1, channel], axis=1,
        )
    dense = np.empty((batch_size, 512), dtype=np.float64)
    for unit in range(512):
        direct = np.sum(weighted_sum[:, :100] * parameters["V"][:, unit], axis=1)
        partner = np.sum(
            weighted_sum[:, 199:99:-1] * parameters["V"][:, unit], axis=1,
        )
        dense[:, unit] = direct + partner + parameters["d"][unit]
    dense_relu = np.maximum(dense, 0)
    logit = (
        np.sum(dense_relu * parameters["q"][:, 0], axis=1, keepdims=True)
        + parameters["e"][0]
    )
    stages = {
        "input": values.copy(),
        "convolution": convolution,
        "first_relu": first_relu,
        "batch_normalization": normalized,
        "second_relu": second_relu,
        "pooled": pooled,
        "weighted_sum": weighted_sum,
        "dense": dense,
        "dense_relu": dense_relu,
        "logit": logit,
        "output": 1 / (1 + np.exp(-logit)),
    }
    return stages, running_mean, running_var


def _smooth_model(dtype):
    """Keep every ReLU away from zero and retain nonconstant pooling inputs."""
    model = CNNRC(seed=841)
    if dtype == torch.float64:
        model.double()
    with torch.no_grad():
        model.b.fill_(1)
        model.gamma.fill_(0.1)
        model.beta.fill_(1)
        model.V.mul_(0.03)
        model.d.fill_(2)
        model.q.mul_(0.05)
    return model


class CNNRCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_thread_count)

    def assert_state_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for name in first:
            if isinstance(first[name], torch.Tensor):
                self.assertTrue(torch.equal(first[name], second[name]), name)
            else:
                self.assertEqual(first[name], second[name], name)

    def assert_stages_transformed(self, direct, transformed, selected, dtype):
        tolerances = TOLERANCES[dtype]
        for name in direct:
            expected = direct[name].clone()
            if name == "input":
                expected[selected] = direct[name][selected].flip((1, 2))
            elif name in SPATIAL_STAGES:
                expected[selected] = direct[name][selected].flip((1, 2))
            elif name == "weighted_sum":
                expected[selected] = direct[name][selected].flip((1,))
            else:
                self.assertIn(name, INVARIANT_STAGES)
            torch.testing.assert_close(transformed[name], expected, **tolerances)

    def test_input_validation_rejects_shapes_types_and_encodings(self):
        valid = _one_hot()
        malformed = {
            "rank_two": valid[0],
            "wrong_length": valid[:, :13],
            "wrong_channels": valid[:, :, :3],
            "channels_first": valid.transpose(1, 2),
            "extra_dimension": valid.unsqueeze(0),
            "empty_batch": valid[:0],
            "integer": valid.to(torch.int64),
            "boolean": valid.bool(),
            "complex": valid.to(torch.complex64),
            "numpy": valid.numpy(),
            "none": None,
        }
        for name, replacement in (
            ("no_active_channel", [0, 0, 0, 0]),
            ("two_active_channels", [1, 1, 0, 0]),
            ("probabilities", [0.5, 0.5, 0, 0]),
            ("negative", [-1, 1, 1, 0]),
            ("out_of_range", [2, 0, 0, 0]),
            ("nan", [float("nan"), 0, 0, 0]),
            ("infinity", [float("inf"), 0, 0, 0]),
        ):
            values = valid.clone()
            values[0, 0] = torch.tensor(replacement, dtype=values.dtype)
            malformed[name] = values
        model = CNNRC(seed=1)
        before = copy.deepcopy(model.state_dict())
        for name, values in malformed.items():
            with self.subTest(case=name):
                with self.assertRaises((TypeError, ValueError)):
                    model(values)
                with self.assertRaises((TypeError, ValueError)):
                    reverse_complement(values)
        with self.assertRaises((TypeError, ValueError)):
            model(valid.double())
        self.assert_state_equal(before, model.state_dict())
        self.assertTrue(torch.equal(valid, _one_hot()))

    def test_rc_involution_palindrome_preservation_and_singleton(self):
        for dtype in TOLERANCES:
            values = _one_hot(dtype=dtype)
            before = values.clone()
            transformed = reverse_complement(values)
            self.assertEqual(transformed.dtype, dtype)
            self.assertEqual(transformed.device, values.device)
            self.assertTrue(torch.equal(transformed, values.flip((1, 2))))
            self.assertTrue(torch.equal(reverse_complement(transformed), values))
            self.assertTrue(torch.equal(before, values))
            palindrome = _one_hot(("AAAAAAATTTTTTT",), dtype=dtype)
            self.assertTrue(torch.equal(reverse_complement(palindrome), palindrome))
            model = CNNRC(seed=2).to(dtype=dtype)
            training_prediction = model(palindrome)
            self.assertEqual(tuple(training_prediction.shape), (1, 1))
            self.assertTrue(torch.isfinite(training_prediction).all().item())
            model.eval()
            prediction = model(palindrome)
            self.assertEqual(tuple(prediction.shape), (1, 1))
            self.assertEqual(prediction.dtype, dtype)
            self.assertTrue(torch.equal(model(reverse_complement(palindrome)), prediction))

    def test_every_graph_shape_parameter_shape_and_buffer_count(self):
        model = CNNRC(seed=3)
        expected_parameters = {
            "W": (100, 4, 11), "b": (100,), "gamma": (100,), "beta": (100,),
            "A": (7, 100), "V": (100, 512), "d": (512,), "q": (512, 1), "e": (1,),
        }
        parameters = dict(model.named_parameters())
        self.assertEqual(set(parameters), set(expected_parameters))
        for name, shape in expected_parameters.items():
            self.assertEqual(tuple(parameters[name].shape), shape, name)
            self.assertTrue(parameters[name].requires_grad, name)
            self.assertEqual(parameters[name].device.type, "cpu")
            self.assertEqual(parameters[name].dtype, torch.float32)
        self.assertEqual(sum(parameter.numel() for parameter in parameters.values()), 57625)
        buffers = dict(model.named_buffers())
        self.assertEqual(set(buffers), {"running_mean", "running_var"})
        self.assertEqual(sum(buffer.numel() for buffer in buffers.values()), 200)
        for buffer in buffers.values():
            self.assertEqual(tuple(buffer.shape), (100,))
            self.assertFalse(buffer.requires_grad)
            self.assertEqual(buffer.dtype, torch.float32)
        expected_shapes = {
            "input": (3, 14, 4), "convolution": (3, 200, 14),
            "first_relu": (3, 200, 14), "batch_normalization": (3, 200, 14),
            "second_relu": (3, 200, 14), "pooled": (3, 200, 7),
            "weighted_sum": (3, 200), "dense": (3, 512),
            "dense_relu": (3, 512), "logit": (3, 1), "output": (3, 1),
        }
        direct_model = copy.deepcopy(model)
        stages = model.forward_intermediates(_one_hot())
        self.assertTrue(torch.equal(direct_model(_one_hot()), stages["output"]))
        self.assert_state_equal(model.state_dict(), direct_model.state_dict())
        self.assertEqual(set(stages), set(expected_shapes))
        for name, shape in expected_shapes.items():
            self.assertEqual(tuple(stages[name].shape), shape, name)
        for module in model.modules():
            self.assertNotIsInstance(module, torch.nn.modules.dropout._DropoutNd)

    def test_asymmetric_indexed_convolution_and_shared_bias_ordering(self):
        model = CNNRC(seed=4).double()
        values = torch.zeros((1, 4, 14), dtype=torch.float64)
        values[0, 0, 0] = 1
        values[0, 3, 13] = 2
        values[0, 1, 4] = 3
        values[0, 2, 9] = 4
        with torch.no_grad():
            model.W.zero_()
            model.W[0, 0, 0] = 2
            model.W[0, 3, 10] = 3
            model.W[12, 1, 2] = 5
            model.W[99, 2, 8] = 7
            model.b.copy_(torch.arange(100, dtype=torch.float64) + 0.25)
        expected = np.zeros((1, 200, 14))
        kernels = model.W.detach().numpy()
        inputs = values.numpy()
        for channel in range(100):
            for position in range(14):
                expected[0, channel, position] = channel + 0.25
                expected[0, 199 - channel, position] = channel + 0.25
                for base in range(4):
                    for offset in range(11):
                        source_position = position + offset - 5
                        if 0 <= source_position < 14:
                            observed = inputs[0, base, source_position]
                            expected[0, channel, position] += (
                                kernels[channel, base, offset] * observed
                            )
                            expected[0, 199 - channel, position] += (
                                kernels[channel, 3 - base, 10 - offset] * observed
                            )
        actual = model.rc_convolution(values)
        np.testing.assert_array_equal(actual.detach().numpy(), expected)
        self.assertEqual(actual[0, 12, 7].item(), 27.25)
        self.assertEqual(actual[0, 187, 6].item(), 32.25)
        self.assertEqual(actual[0, 99, 6].item(), 127.25)
        self.assertEqual(actual[0, 100, 7].item(), 120.25)

    def test_convolution_partner_gradients_reach_only_independent_parameters(self):
        model = CNNRC(seed=5).double()
        values = torch.ones((2, 4, 14), dtype=torch.float64)
        model.rc_convolution(values).sum().backward()
        expected_kernel_gradient = torch.empty_like(model.W)
        for offset in range(11):
            expected_kernel_gradient[:, :, offset] = 4 * (14 - abs(offset - 5))
        self.assertTrue(torch.equal(model.W.grad, expected_kernel_gradient))
        self.assertTrue(torch.equal(model.b.grad, torch.full_like(model.b, 56)))
        for name, parameter in model.named_parameters():
            if name not in ("W", "b"):
                self.assertIsNone(parameter.grad, name)

    def test_population_bn_hand_moments_and_two_ema_updates(self):
        model = CNNRC(seed=6).double()
        values = torch.zeros((1, 200, 14), dtype=torch.float64)
        values[0, 0] = 1
        values[0, 199] = 3
        values[0, 1] = torch.arange(14, dtype=torch.float64)
        values[0, 198] = torch.arange(14, 28, dtype=torch.float64)
        with torch.no_grad():
            model.gamma[0] = 2
            model.beta[0] = 0.5
        normalized = model.shared_batch_normalization(values)
        torch.testing.assert_close(
            normalized[0, 0], torch.full((14,), 0.5 - 2 / math.sqrt(1.001), dtype=torch.float64),
            **TOLERANCES[torch.float64],
        )
        torch.testing.assert_close(
            normalized[0, 199], torch.full((14,), 0.5 + 2 / math.sqrt(1.001), dtype=torch.float64),
            **TOLERANCES[torch.float64],
        )
        torch.testing.assert_close(
            normalized[0, 1], (torch.arange(14, dtype=torch.float64) - 13.5) / math.sqrt(65.251),
            **TOLERANCES[torch.float64],
        )
        self.assertAlmostEqual(model.running_mean[0].item(), 0.02, places=14)
        self.assertAlmostEqual(model.running_var[0].item(), 1, places=14)
        self.assertNotAlmostEqual(model.running_var[0].item(), 0.99 + 0.01 * 28 / 27, places=10)
        self.assertAlmostEqual(model.running_mean[1].item(), 0.135, places=14)
        self.assertAlmostEqual(model.running_var[1].item(), 1.6425, places=14)
        self.assertAlmostEqual(model.running_var[2].item(), 0.99, places=14)
        values[0, 0] = 3
        values[0, 199] = 5
        values[0, 1].mul_(2)
        values[0, 198].mul_(2)
        model.shared_batch_normalization(values)
        self.assertAlmostEqual(model.running_mean[0].item(), 0.0598, places=14)
        self.assertAlmostEqual(model.running_var[0].item(), 1, places=14)
        self.assertAlmostEqual(model.running_mean[1].item(), 0.40365, places=14)
        self.assertAlmostEqual(model.running_var[1].item(), 4.236075, places=14)
        self.assertAlmostEqual(model.running_var[2].item(), 0.9801, places=14)

    def test_bn_epsilon_inside_root_and_fixed_eval_statistics(self):
        model = CNNRC(seed=7).double()
        values = torch.zeros((1, 200, 14), dtype=torch.float64)
        values[:, 199] = 0.002
        normalized = model.shared_batch_normalization(values)
        expected = -0.001 / math.sqrt(0.000001 + 0.001)
        self.assertAlmostEqual(normalized[0, 0, 0].item(), expected, places=14)
        self.assertNotAlmostEqual(normalized[0, 0, 0].item(), -0.001 / (0.001 + 0.001), places=10)
        self.assertAlmostEqual(model.running_var[0].item(), 0.99000001, places=14)
        model.eval()
        with torch.no_grad():
            model.running_mean.fill_(2)
            model.running_var.fill_(4)
            model.gamma.fill_(3)
            model.beta.fill_(-1)
        state_before = copy.deepcopy(model.state_dict())
        actual = model.shared_batch_normalization(torch.ones_like(values))
        torch.testing.assert_close(
            actual, torch.full_like(values, -3 / math.sqrt(4.001) - 1),
            **TOLERANCES[torch.float64],
        )
        self.assert_state_equal(state_before, model.state_dict())

    def test_training_updates_without_grad_and_eval_changes_no_state_with_grad(self):
        model = CNNRC(seed=8).double()
        values = _one_hot(dtype=torch.float64)
        _, expected_mean, expected_var = _numpy_reference(model, values.numpy(), True)
        with torch.no_grad():
            model(values)
        np.testing.assert_allclose(model.running_mean.numpy(), expected_mean, **TOLERANCES[torch.float64])
        np.testing.assert_allclose(model.running_var.numpy(), expected_var, **TOLERANCES[torch.float64])
        model.eval()
        state_before = copy.deepcopy(model.state_dict())
        prediction = model(values)
        self.assertTrue(prediction.requires_grad)
        prediction.sum().backward()
        self.assert_state_equal(state_before, model.state_dict())

    def test_both_relu_operations_and_dense_relu_are_applied(self):
        model = CNNRC(seed=9).double()
        model.eval()
        with torch.no_grad():
            model.W.zero_()
            model.b.fill_(-1)
            model.beta.fill_(-2)
            model.d.fill_(-1)
        stages = model.forward_intermediates(_one_hot(dtype=torch.float64))
        self.assertTrue(torch.equal(stages["convolution"], torch.full_like(stages["convolution"], -1)))
        self.assertEqual(torch.count_nonzero(stages["first_relu"]).item(), 0)
        self.assertTrue(torch.equal(stages["batch_normalization"], torch.full_like(stages["batch_normalization"], -2)))
        for name in ("second_relu", "pooled", "weighted_sum", "dense_relu", "logit"):
            self.assertEqual(torch.count_nonzero(stages[name]).item(), 0, name)
        self.assertTrue(torch.equal(stages["dense"], torch.full_like(stages["dense"], -1)))
        self.assertTrue(torch.equal(stages["output"], torch.full((3, 1), 0.5, dtype=torch.float64)))

    def test_asymmetric_signed_positional_weights_and_partner_gradients(self):
        model = CNNRC(seed=10).double()
        values = torch.zeros((1, 200, 7), dtype=torch.float64)
        positions = torch.arange(1, 8, dtype=torch.float64)
        values[0, 0] = positions
        values[0, 199] = 2 * positions
        values[0, 99] = positions.square()
        values[0, 100] = positions + 2
        with torch.no_grad():
            model.A.zero_()
            model.A[:, 0] = torch.arange(-3, 4, dtype=torch.float64)
            model.A[:, 99] = torch.tensor([2, -1, 4, -3, 5, -6, 7], dtype=torch.float64)
        actual = model.positional_weighted_sum(values)
        self.assertEqual(actual[0, 0].item(), 28)
        self.assertEqual(actual[0, 199].item(), -56)
        self.assertEqual(actual[0, 99].item(), 238)
        self.assertEqual(actual[0, 100].item(), 42)
        self.assertEqual(torch.count_nonzero(actual).item(), 4)
        actual.sum().backward()
        self.assertTrue(torch.equal(model.A.grad[:, 0], positions + 2 * positions.flip((0,))))
        self.assertTrue(torch.equal(model.A.grad[:, 99], positions.square() + (positions + 2).flip((0,))))

    def test_dense_partner_sum_and_bias_once_with_tied_gradients(self):
        model = CNNRC(seed=11).double()
        values = torch.zeros((1, 200), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            values[0, 3] = 2
            values[0, 196] = 5
            model.V.zero_()
            model.V[3, 0] = 3
            model.d.fill_(11)
        actual = model.rc_dense(values)
        self.assertEqual(actual[0, 0].item(), 32)
        self.assertTrue(torch.equal(actual[0, 1:], torch.full((511,), 11, dtype=torch.float64)))
        actual[0, 0].backward()
        self.assertEqual(model.V.grad[3, 0].item(), 7)
        self.assertEqual(model.d.grad[0].item(), 1)
        self.assertEqual(values.grad[0, 3].item(), 3)
        self.assertEqual(values.grad[0, 196].item(), 3)

    def test_independent_numpy_full_graph_and_running_statistics(self):
        for training in (False, True):
            with self.subTest(training=training):
                model = CNNRC(seed=12).double()
                model.train(training)
                with torch.no_grad():
                    model.gamma.copy_(torch.linspace(-0.7, 1.3, 100, dtype=torch.float64))
                    model.beta.copy_(torch.linspace(-0.2, 0.4, 100, dtype=torch.float64))
                    model.running_mean.copy_(torch.linspace(-0.1, 0.3, 100, dtype=torch.float64))
                    model.running_var.copy_(torch.linspace(0.2, 1.2, 100, dtype=torch.float64))
                    model.b.copy_(torch.linspace(-0.08, 0.1, 100, dtype=torch.float64))
                    model.d.copy_(torch.linspace(-0.1, 0.1, 512, dtype=torch.float64))
                    model.e.fill_(0.2)
                values = _one_hot(SEQUENCES[:2], dtype=torch.float64)
                expected, expected_mean, expected_var = _numpy_reference(model, values.numpy(), training)
                actual = model.forward_intermediates(values)
                for name in expected:
                    np.testing.assert_allclose(actual[name].detach().numpy(), expected[name], **TOLERANCES[torch.float64], err_msg=name)
                np.testing.assert_allclose(model.running_mean.numpy(), expected_mean, **TOLERANCES[torch.float64])
                np.testing.assert_allclose(model.running_var.numpy(), expected_var, **TOLERANCES[torch.float64])

    def test_rc_invariance_of_unaveraged_graph_and_updated_buffers(self):
        for dtype in TOLERANCES:
            for training in (False, True):
                for selected in ([0, 1, 2], [0, 2]):
                    with self.subTest(dtype=dtype, training=training, selected=selected):
                        first = CNNRC(seed=13).to(dtype=dtype)
                        first.train(training)
                        second = copy.deepcopy(first)
                        values = _one_hot(dtype=dtype)
                        transformed_values = values.clone()
                        transformed_values[selected] = values[selected].flip((1, 2))
                        first_stages = first.forward_intermediates(values)
                        second_stages = second.forward_intermediates(transformed_values)
                        self.assert_stages_transformed(first_stages, second_stages, selected, dtype)
                        torch.testing.assert_close(first.running_mean, second.running_mean, **TOLERANCES[dtype])
                        torch.testing.assert_close(first.running_var, second.running_var, **TOLERANCES[dtype])
                        direct_forward = copy.deepcopy(first).eval()
                        torch.testing.assert_close(
                            direct_forward(values), direct_forward(transformed_values),
                            **TOLERANCES[dtype],
                        )

    def test_tie_free_nonzero_margin_rc_gradient_symmetry(self):
        for dtype in TOLERANCES:
            for selected in ([0, 1, 2], [0, 2]):
                with self.subTest(dtype=dtype, selected=selected):
                    first = _smooth_model(dtype)
                    second = copy.deepcopy(first)
                    first_input = _one_hot(dtype=dtype).requires_grad_(True)
                    second_input = first_input.detach().clone()
                    second_input[selected] = second_input[selected].flip((1, 2))
                    second_input.requires_grad_(True)
                    first_stages = first.forward_intermediates(first_input)
                    second_stages = second.forward_intermediates(second_input)
                    for stages in (first_stages, second_stages):
                        for name in ("convolution", "batch_normalization", "dense"):
                            self.assertGreater(stages[name].detach().abs().min().item(), 0.1, name)
                        pairs = stages["second_relu"]
                        pool_margin = (pairs[:, :, 0::2] - pairs[:, :, 1::2]).abs().min().item()
                        self.assertGreater(pool_margin, 1e-7)
                        self.assertTrue(torch.all(stages["output"] > 0).item())
                        self.assertTrue(torch.all(stages["output"] < 1).item())
                    row_weights = torch.tensor([[0.3], [0.7], [1.1]], dtype=dtype)
                    (first_stages["output"].square() * row_weights).sum().backward()
                    (second_stages["output"].square() * row_weights).sum().backward()
                    for name, parameter in first.named_parameters():
                        other = dict(second.named_parameters())[name]
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                        self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0, name)
                        torch.testing.assert_close(parameter.grad, other.grad, **TOLERANCES[dtype])
                    expected_input_gradient = first_input.grad.clone()
                    expected_input_gradient[selected] = first_input.grad[selected].flip((1, 2))
                    self.assertTrue(torch.isfinite(first_input.grad).all().item())
                    torch.testing.assert_close(second_input.grad, expected_input_gradient, **TOLERANCES[dtype])

    def test_relu_boundaries_and_max_pool_ties_are_forward_only(self):
        for activation in (0, 1):
            with self.subTest(activation=activation):
                model = CNNRC(seed=14).double().eval()
                with torch.no_grad():
                    model.W.zero_()
                    model.b.fill_(activation)
                    model.beta.fill_(activation)
                    model.V.zero_()
                    model.d.zero_()
                values = _one_hot(dtype=torch.float64)
                stages = model.forward_intermediates(values)
                expected_activation = activation / math.sqrt(1.001) + activation
                self.assertTrue(torch.equal(stages["second_relu"][:, :, 0::2], stages["second_relu"][:, :, 1::2]))
                torch.testing.assert_close(stages["pooled"], torch.full_like(stages["pooled"], expected_activation), **TOLERANCES[torch.float64])
                self.assertEqual(torch.count_nonzero(stages["dense"]).item(), 0)
                self.assertTrue(torch.equal(stages["output"], torch.full((3, 1), 0.5, dtype=torch.float64)))
                transformed = model.forward_intermediates(values.flip((1, 2)))
                self.assert_stages_transformed(stages, transformed, [0, 1, 2], torch.float64)

    def test_convolution_only_regularization_value_gradient_and_coefficients(self):
        model = CNNRC(seed=15).double()
        with torch.no_grad():
            model.W.copy_(torch.arange(4400, dtype=torch.float64).remainder(5).reshape(100, 4, 11) - 2)
        penalty = model.convolution_kernel_penalty(l1=0.3, l2=0.2)
        self.assertAlmostEqual(penalty.item(), 0.3 * 880 * 6 + 0.2 * 880 * 10, places=10)
        penalty.backward()
        torch.testing.assert_close(model.W.grad, 0.3 * model.W.sign() + 0.4 * model.W, **TOLERANCES[torch.float64])
        for name, parameter in model.named_parameters():
            if name != "W":
                self.assertIsNone(parameter.grad, name)
        model.zero_grad(set_to_none=True)
        zero_penalty = model.convolution_kernel_penalty()
        self.assertEqual(zero_penalty.item(), 0)
        zero_penalty.backward()
        self.assertEqual(torch.count_nonzero(model.W.grad).item(), 0)
        for coefficient in (-0.1, float("nan"), float("inf"), float("-inf")):
            for name in ("l1", "l2"):
                with self.subTest(name=name, coefficient=coefficient):
                    with self.assertRaises((TypeError, ValueError)):
                        model.convolution_kernel_penalty(**{name: coefficient})

    def test_initialization_bounds_exact_local_rng_order_and_full_seeded_state(self):
        global_rng_before = torch.random.get_rng_state().clone()
        model = CNNRC(seed=163)
        repeated = CNNRC(seed=163)
        different = CNNRC(seed=164)
        self.assertTrue(torch.equal(global_rng_before, torch.random.get_rng_state()))
        self.assert_state_equal(model.state_dict(), repeated.state_dict())
        self.assertFalse(torch.equal(model.W, different.W))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(163)
        draws = (
            ("W", (100, 4, 11), math.sqrt(6 / 4800)),
            ("A", (7, 100), math.sqrt(3 / 700)),
            ("V", (100, 512), math.sqrt(6 / 612)),
            ("q", (512, 1), math.sqrt(6 / 513)),
        )
        for name, shape, bound in draws:
            parameter = getattr(model, name)
            expected = torch.empty(shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator)
            self.assertTrue(torch.equal(parameter, expected), name)
            self.assertTrue(torch.all(parameter >= -bound).item(), name)
            self.assertTrue(torch.all(parameter <= bound).item(), name)
            self.assertLess(parameter.min().item(), 0, name)
            self.assertGreater(parameter.max().item(), 0, name)
        for name in ("b", "d", "e", "beta", "running_mean"):
            self.assertEqual(torch.count_nonzero(getattr(model, name)).item(), 0, name)
        for name in ("gamma", "running_var"):
            self.assertTrue(torch.equal(getattr(model, name), torch.ones(100)), name)
        with self.assertRaises(TypeError):
            CNNRC()
        for invalid_seed in (None, "163", 1.5, True):
            with self.subTest(seed=invalid_seed):
                with self.assertRaises((TypeError, ValueError)):
                    CNNRC(seed=invalid_seed)

    def test_initialization_ignores_global_default_dtype(self):
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = CNNRC(seed=16)
        finally:
            torch.set_default_dtype(original_dtype)
        for parameter in model.parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertEqual(parameter.device.type, "cpu")
        for buffer in model.buffers():
            self.assertEqual(buffer.dtype, torch.float32)
            self.assertEqual(buffer.device.type, "cpu")

    def test_strict_in_memory_state_round_trip_and_contract_metadata(self):
        self.assertEqual(CONTRACT_ID, "cnn_rc_wang100_kundaje122_population_bn.v1")
        settings = architecture_settings()
        self.assertEqual(settings["wang_reference_commit"], "9e6d6ef0355558c98855b83a9c21fe11999f65d9")
        self.assertEqual(settings["kundaje_keras122_reference_commit"], "0a49220049716163db08b285149927e2fda19cc2")
        self.assertEqual(settings["initialization"]["W_choice"], "kundaje_keras122_th_fans")
        self.assertEqual(settings["initialization"]["W_fans"], [400, 4400])
        self.assertEqual(settings["initialization"]["A_choice"], "fanintimesfanouttimestwo")
        self.assertEqual(settings["cpu_acceptance_tolerances"]["float32"], TOLERANCES[torch.float32])
        self.assertEqual(settings["cpu_acceptance_tolerances"]["float64"], TOLERANCES[torch.float64])
        self.assertIs(settings["legacy_runtime_or_bit_exact_parity"], False)
        self.assertIs(settings["forward_rc_prediction_averaging"], False)
        model = CNNRC(seed=17)
        model(_one_hot())
        original_state = copy.deepcopy(model.state_dict())
        self.assertEqual(original_state["_extra_state"]["contract_id"], CONTRACT_ID)
        self.assertEqual(original_state["_extra_state"]["architecture_settings"], architecture_settings())
        restored = CNNRC(seed=18)
        restored.load_state_dict(original_state, strict=True)
        self.assert_state_equal(original_state, restored.state_dict())
        model.eval()
        restored.eval()
        self.assertTrue(torch.equal(model(_one_hot()), restored(_one_hot())))
        missing = copy.deepcopy(original_state)
        del missing["_extra_state"]
        with self.assertRaises((RuntimeError, ValueError)):
            CNNRC(seed=19).load_state_dict(missing, strict=True)
        malformed_metadata = (
            None,
            {},
            {"contract_id": "wrong", "architecture_settings": architecture_settings()},
            {"contract_id": CONTRACT_ID},
            {"contract_id": CONTRACT_ID, "architecture_settings": {}},
        )
        for metadata in malformed_metadata:
            with self.subTest(metadata=metadata):
                malformed = copy.deepcopy(original_state)
                malformed["_extra_state"] = metadata
                with self.assertRaises((TypeError, ValueError, RuntimeError)):
                    CNNRC(seed=20).load_state_dict(malformed, strict=True)
        nested_mutations = (
            ("batch_normalization", "epsilon_inside_sqrt", 0.01),
            ("convolution", "stride", True),
            ("rc_dense", "bias_additions", 1.0),
            ("initialization", "draw_order", ["A", "W", "V", "q"]),
        )
        for section, field, value in nested_mutations:
            with self.subTest(section=section, field=field):
                malformed = copy.deepcopy(original_state)
                malformed["_extra_state"]["architecture_settings"][section][field] = value
                with self.assertRaises((TypeError, ValueError, RuntimeError)):
                    CNNRC(seed=20).load_state_dict(malformed, strict=True)
        for name, value in original_state.items():
            if isinstance(value, torch.Tensor):
                with self.subTest(shape=name):
                    malformed = copy.deepcopy(original_state)
                    malformed[name] = value[:0]
                    with self.assertRaises(RuntimeError):
                        CNNRC(seed=21).load_state_dict(malformed, strict=True)


if __name__ == "__main__":
    unittest.main()
