"""Maintained paper-architecture-faithful Wang100 CNN-RC.

The fixed contract preserves the approved Kundaje Keras 1.2.2 mathematics,
including population-variance BN. It does not establish bit-exact legacy
parity or recover the historical Wang runtime. Initialization uses a local
CPU generator: W, A, V, then q are drawn in their stored shapes in float32.
This maintained RNG ordering is not a claim of legacy RNG parity.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional


CONTRACT_ID = "cnn_rc_wang100_kundaje122_population_bn.v1"
WANG_REFERENCE_COMMIT = "9e6d6ef0355558c98855b83a9c21fe11999f65d9"
KUNDAJE_REFERENCE_COMMIT = "0a49220049716163db08b285149927e2fda19cc2"


def architecture_settings() -> dict[str, Any]:
    """Return a fresh description of every fixed architectural choice."""
    return {
        "implementation": "maintained_paper_architecture_faithful",
        "wang_reference_commit": WANG_REFERENCE_COMMIT,
        "kundaje_keras122_reference_commit": KUNDAJE_REFERENCE_COMMIT,
        "legacy_runtime_or_bit_exact_parity": False,
        "input": {
            "shape": ["B>=1", 14, 4],
            "channels": ["A", "C", "G", "T"],
            "encoding": "finite_exact_one_hot_floating",
            "internal_layout": "B,4,14",
            "reverse_complement": "x[b,13-p,3-a]",
        },
        "graph": [
            "rc_convolution_with_bias", "relu", "shared_rc_bn", "relu",
            "max_pool", "rc_positional_weighted_sum", "rc_dense_with_bias",
            "relu", "output_dense_with_bias", "sigmoid",
        ],
        "convolution": {
            "independent_weight_shape": [100, 4, 11],
            "independent_bias_shape": [100],
            "expanded_channels": 200,
            "partner": "199-c",
            "partner_kernel": "W[c,3-a,10-t]",
            "partner_bias": "b[c]",
            "operation": "cross_correlation",
            "stride": 1,
            "dilation": 1,
            "zero_padding": 5,
            "output_shape": ["B", 200, 14],
        },
        "batch_normalization": {
            "affine_shapes": {"gamma": [100], "beta": [100]},
            "buffer_shapes": {"running_mean": [100], "running_var": [100]},
            "observations": "R[:,c,:] and R[:,199-c,:]; M=2*B*14",
            "variance": "population_in_forward_and_ema",
            "epsilon_inside_sqrt": 0.001,
            "old_state_retention": 0.99,
            "new_statistic_weight": 0.01,
            "updates": "once_per_training_forward_independent_of_grad_mode",
            "evaluation": "running_statistics_without_updates",
            "batch_counter": False,
        },
        "pool": {"kind": "max", "width": 2, "stride": 2, "padding": 0},
        "positional_weighted_sum": {
            "weight_shape": [7, 100],
            "partner_weight": "A[6-p,c]",
            "reduction": "sum",
            "weights": "unrestricted_signed",
            "bias": False,
            "normalization": False,
            "activation": False,
            "output_shape": ["B", 200],
        },
        "rc_dense": {
            "weight_shape": [100, 512],
            "bias_shape": [512],
            "input": "z[:,c]+z[:,199-c]",
            "bias_additions": 1,
            "activation": "relu",
        },
        "output": {
            "weight_shape": [512, 1], "bias_shape": [1],
            "activation": "sigmoid", "shape": ["B", 1],
        },
        "initialization": {
            "generator": "local_torch_cpu_generator_with_explicit_seed",
            "draw_order": ["W", "A", "V", "q"],
            "draw_dtype": "float32",
            "draw_shapes": "declared_independent_stored_shapes",
            "distribution": "independent_uniform_symmetric",
            "W_choice": "kundaje_keras122_th_fans",
            "W_fans": [400, 4400],
            "W_bound": math.sqrt(6.0 / 4800.0),
            "A_choice": "fanintimesfanouttimestwo",
            "A_bound": math.sqrt(3.0 / 700.0),
            "V_bound": math.sqrt(6.0 / (100.0 + 512.0)),
            "q_bound": math.sqrt(6.0 / (512.0 + 1.0)),
            "zeros": ["b", "d", "e", "beta", "running_mean"],
            "ones": ["gamma", "running_var"],
        },
        "regularization": "l1*sum(abs(W))+l2*sum(W**2); independent_W_only",
        "eventual_data_objective": "mean_squared_error_plus_kernel_penalty",
        "trainable_scalars": 57625,
        "floating_buffer_scalars": 200,
        "forward_rc_prediction_averaging": False,
        "cpu_acceptance_tolerances": {
            "float64": {"atol": 1e-10, "rtol": 1e-8},
            "float32": {"atol": 1e-6, "rtol": 1e-5},
        },
    }


def _validate_input(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    if x.ndim != 3 or tuple(x.shape[1:]) != (14, 4) or x.shape[0] < 1:
        raise ValueError("Input must have shape (B,14,4), with B >= 1.")
    if not x.is_floating_point():
        raise TypeError("Input must have a floating dtype.")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Input must be finite.")
    if not bool(((x == 0) | (x == 1)).all()):
        raise ValueError("Input must contain exact zero/one values.")
    if not bool((x.sum(dim=2) == 1).all()):
        raise ValueError("Each input row must contain exactly one active channel.")


def reverse_complement(x: torch.Tensor) -> torch.Tensor:
    """Reverse positions and A,C,G,T channels without casting or mutation."""
    _validate_input(x)
    return x.flip(dims=(1, 2))


def _metadata_matches(actual: Any, expected: Any) -> bool:
    """Compare only the exact plain types used by the fixed metadata schema."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return False
        for key in expected:
            if not _metadata_matches(actual[key], expected[key]):
                return False
        return True
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        for actual_item, expected_item in zip(actual, expected):
            if not _metadata_matches(actual_item, expected_item):
                return False
        return True
    return actual == expected


class CNNRC(nn.Module):
    """Fixed Wang100 CNN-RC; construction is always CPU float32.

    Use ``model.double()`` and float64 inputs for float64 CPU validation.
    ``forward_intermediates`` performs one full forward, including one BN
    update in training mode. Layer helpers take the internal shapes declared
    below; validation of public one-hot input occurs at the graph boundary.
    """

    contract_id = CONTRACT_ID

    def __init__(self, seed: int) -> None:
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("An explicit integer initialization seed is required.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        # Empty tensors avoid hidden module initialization and global RNG use.
        self.W = nn.Parameter(
            self._uniform((100, 4, 11), math.sqrt(6.0 / 4800.0), generator)
        )
        self.A = nn.Parameter(
            self._uniform((7, 100), math.sqrt(3.0 / 700.0), generator)
        )
        self.V = nn.Parameter(
            self._uniform((100, 512), math.sqrt(6.0 / 612.0), generator)
        )
        self.q = nn.Parameter(
            self._uniform((512, 1), math.sqrt(6.0 / 513.0), generator)
        )
        self.b = nn.Parameter(torch.zeros(100, dtype=torch.float32, device="cpu"))
        self.d = nn.Parameter(torch.zeros(512, dtype=torch.float32, device="cpu"))
        self.e = nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cpu"))
        self.gamma = nn.Parameter(torch.ones(100, dtype=torch.float32, device="cpu"))
        self.beta = nn.Parameter(torch.zeros(100, dtype=torch.float32, device="cpu"))
        self.register_buffer(
            "running_mean", torch.zeros(100, dtype=torch.float32, device="cpu")
        )
        self.register_buffer(
            "running_var", torch.ones(100, dtype=torch.float32, device="cpu")
        )

    @staticmethod
    def _uniform(
        shape: tuple[int, ...], bound: float, generator: torch.Generator
    ) -> torch.Tensor:
        weight = torch.empty(shape, dtype=torch.float32, device="cpu")
        return weight.uniform_(-bound, bound, generator=generator)

    def get_extra_state(self) -> dict[str, Any]:
        """Bind in-memory model state to the complete fixed contract."""
        return {
            "contract_id": CONTRACT_ID,
            "architecture_settings": architecture_settings(),
        }

    def set_extra_state(self, state: Any) -> None:
        """Reject missing fields, changed settings and malformed metadata."""
        if not _metadata_matches(state, self.get_extra_state()):
            raise RuntimeError(
                "CNN-RC contract metadata is missing, malformed or incompatible."
            )

    def rc_convolution(self, internal_input: torch.Tensor) -> torch.Tensor:
        """Cross-correlate (B,4,14) with differentiably tied RC kernels."""
        full_weight = torch.cat((self.W, self.W.flip(dims=(0, 1, 2))), dim=0)
        full_bias = torch.cat((self.b, self.b.flip(dims=(0,))), dim=0)
        return functional.conv1d(
            internal_input, full_weight, full_bias, stride=1, padding=5, dilation=1
        )

    def shared_batch_normalization(self, activations: torch.Tensor) -> torch.Tensor:
        """Normalize (B,200,14) with pair-shared population moments."""
        if self.training:
            paired = torch.cat(
                (activations[:, :100, :], activations[:, 100:, :].flip(dims=(1,))),
                dim=2,
            )
            mean = paired.mean(dim=(0, 2))
            centered = paired - mean[None, :, None]
            variance = centered.square().mean(dim=(0, 2))
            # Detaching statistics keeps buffer updates independent of autograd.
            self.running_mean.mul_(0.99).add_(mean.detach(), alpha=0.01)
            self.running_var.mul_(0.99).add_(variance.detach(), alpha=0.01)
        else:
            mean = self.running_mean
            variance = self.running_var
        full_mean = torch.cat((mean, mean.flip(dims=(0,))))
        full_variance = torch.cat((variance, variance.flip(dims=(0,))))
        full_gamma = torch.cat((self.gamma, self.gamma.flip(dims=(0,))))
        full_beta = torch.cat((self.beta, self.beta.flip(dims=(0,))))
        denominator = torch.sqrt(full_variance[None, :, None] + 0.001)
        normalized = (activations - full_mean[None, :, None]) / denominator
        return full_gamma[None, :, None] * normalized + full_beta[None, :, None]

    def positional_weighted_sum(self, pooled: torch.Tensor) -> torch.Tensor:
        """Reduce (B,200,7) using unrestricted signed RC positional weights."""
        weights = self.A.transpose(0, 1)
        full_weights = torch.cat((weights, weights.flip(dims=(0, 1))), dim=0)
        return (pooled * full_weights[None, :, :]).sum(dim=2)

    def rc_dense(self, weighted: torch.Tensor) -> torch.Tensor:
        """Map (B,200) to (B,512) with partner SUM and one bias addition."""
        paired = weighted[:, :100] + weighted[:, 100:].flip(dims=(1,))
        return paired @ self.V + self.d

    def forward_intermediates(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the unaveraged graph once and expose its differentiable stages."""
        _validate_input(x)
        if x.dtype != self.W.dtype or x.device != self.W.device:
            raise ValueError(
                "Input dtype and device must match the model; "
                "no implicit conversion is performed."
            )
        convolution = self.rc_convolution(x.transpose(1, 2))
        first_relu = functional.relu(convolution)
        normalized = self.shared_batch_normalization(first_relu)
        second_relu = functional.relu(normalized)
        pooled = functional.max_pool1d(
            second_relu, kernel_size=2, stride=2, padding=0
        )
        weighted = self.positional_weighted_sum(pooled)
        dense = self.rc_dense(weighted)
        dense_relu = functional.relu(dense)
        logit = dense_relu @ self.q + self.e
        output = torch.sigmoid(logit)
        return {
            "input": x,
            "convolution": convolution,
            "first_relu": first_relu,
            "batch_normalization": normalized,
            "second_relu": second_relu,
            "pooled": pooled,
            "weighted_sum": weighted,
            "dense": dense,
            "dense_relu": dense_relu,
            "logit": logit,
            "output": output,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_intermediates(x)["output"]

    def convolution_kernel_penalty(
        self, l1: float = 0.0, l2: float = 0.0
    ) -> torch.Tensor:
        """Apply finite nonnegative L1/L2 coefficients once to independent W."""
        for coefficient in (l1, l2):
            if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
                raise TypeError("Penalty coefficients must be real numbers.")
            if not math.isfinite(coefficient) or coefficient < 0:
                raise ValueError("Penalty coefficients must be finite and nonnegative.")
        return l1 * self.W.abs().sum() + l2 * self.W.square().sum()
