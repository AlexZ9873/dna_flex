"""Pure regression metrics for the complete supplied rows of one TF assay."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validated_values(values: Any, name: str) -> np.ndarray:
    """Validate original numeric values before making a float64 copy."""

    original = np.asarray(values)
    if original.ndim != 1 or original.size == 0:
        raise ValueError("{0} must be a nonempty one-dimensional array.".format(name))
    if original.dtype.kind not in ("i", "u", "f"):
        raise ValueError("{0} must contain real numeric values.".format(name))

    # Inspect sequence elements before NumPy can promote mixed bool/numeric
    # inputs. Object and string arrays are deliberately not coerced to numbers.
    original_elements = np.asarray(values, dtype=object)
    for value in original_elements:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError("{0} must contain real numeric values.".format(name))
    if not np.all(np.isfinite(original)):
        raise ValueError("{0} must contain only finite values.".format(name))
    if np.any(original < 0) or np.any(original > 1):
        raise ValueError("{0} values must be in [0, 1].".format(name))
    return original.astype(np.float64, copy=True)


def _validated_group_count(group_ids: Any, sample_count: int) -> int:
    """Count validated string identities without changing row weights."""

    groups = np.asarray(group_ids, dtype=object)
    if groups.ndim != 1 or groups.size != sample_count:
        raise ValueError("rc_group_ids must have one ID per supplied row.")
    for group_id in groups:
        if not isinstance(group_id, str) or len(group_id) == 0:
            raise ValueError("rc_group_ids must contain nonempty strings.")
    return len(set(groups.tolist()))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign one-based average ranks to exact ties."""

    _, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    cumulative_counts = np.cumsum(counts)
    average_ranks = cumulative_counts - (counts - 1) / 2.0
    return average_ranks[inverse]


def _scaled_centered(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Center after a translation and scaling to retain tiny differences."""

    offsets = values - values[0]
    scale = float(np.max(np.abs(offsets)))
    scaled_offsets = offsets / scale
    return scaled_offsets - np.mean(scaled_offsets), scale


def _pearson(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute centered cosine similarity for two nonconstant arrays."""

    # Translation/scaling cancel algebraically; no epsilon is introduced.
    centered_a, _ = _scaled_centered(values_a)
    centered_b, _ = _scaled_centered(values_b)
    numerator = np.dot(centered_a, centered_b)
    denominator = np.sqrt(np.dot(centered_a, centered_a)) * np.sqrt(
        np.dot(centered_b, centered_b)
    )
    return float(numerator / denominator)


def _r_squared(targets: np.ndarray, residuals: np.ndarray) -> float:
    """Evaluate 1-SSE/SST as a squared norm ratio without tiny squares."""

    residual_scale = float(np.max(np.abs(residuals)))
    if residual_scale == 0.0:
        return 1.0
    scaled_targets, target_scale = _scaled_centered(targets)
    scaled_residuals = residuals / residual_scale
    residual_norm = float(np.sqrt(np.dot(scaled_residuals, scaled_residuals)))
    target_norm = float(np.sqrt(np.dot(scaled_targets, scaled_targets)))
    norm_ratio = (residual_scale / target_scale) * (residual_norm / target_norm)
    return 1.0 - norm_ratio * norm_ratio


def compute_regression_metrics(
    y_true: Any,
    y_pred: Any,
    rc_group_ids: Any = None,
) -> dict[str, int | float | None | dict[str, str]]:
    """Measure one TF assay using every supplied row, with no grouping.

    Targets and sigmoid predictions must be finite real numeric vectors in
    [0, 1]. Calculations use float64 copies and never mutate caller inputs.
    Optional nonempty string RC identities supply only a unique-group count.
    Undefined R2/correlations are None, with reasons in ``undefined_reasons``;
    precedence is insufficient_samples, constant_targets, constant_predictions.
    Supply the full assay arrays: averaging separate batch metrics is invalid.
    """

    targets = _validated_values(y_true, "y_true")
    predictions = _validated_values(y_pred, "y_pred")
    if targets.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have equal lengths.")
    sample_count = int(targets.size)
    unique_group_count = None
    if rc_group_ids is not None:
        unique_group_count = _validated_group_count(rc_group_ids, sample_count)

    residuals = predictions - targets
    squared_error = float(np.dot(residuals, residuals))
    mse = squared_error / sample_count
    undefined_reasons: dict[str, str] = {}
    result: dict[str, int | float | None | dict[str, str]] = {
        "sample_count": sample_count,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": None,
        "pearson": None,
        "spearman": None,
        "undefined_reasons": undefined_reasons,
    }
    if unique_group_count is not None:
        result["unique_rc_group_count"] = unique_group_count

    if sample_count < 2:
        for metric in ("r2", "pearson", "spearman"):
            undefined_reasons[metric] = "insufficient_samples"
        return result
    if np.all(targets == targets[0]):
        for metric in ("r2", "pearson", "spearman"):
            undefined_reasons[metric] = "constant_targets"
        return result

    result["r2"] = _r_squared(targets, residuals)
    if np.all(predictions == predictions[0]):
        for metric in ("pearson", "spearman"):
            undefined_reasons[metric] = "constant_predictions"
        return result

    result["pearson"] = _pearson(targets, predictions)
    result["spearman"] = _pearson(
        _average_ranks(targets), _average_ranks(predictions)
    )
    return result
