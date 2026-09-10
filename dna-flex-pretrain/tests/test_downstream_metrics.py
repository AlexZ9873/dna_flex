"""Synthetic hand-calculated tests for pure one-assay regression metrics."""

import unittest

import numpy as np

from src.downstream_metrics import compute_regression_metrics


class DownstreamMetricsTests(unittest.TestCase):
    def test_hand_calculated_compressed_predictions(self) -> None:
        result = compute_regression_metrics([0, 0.5, 1], [0.25, 0.5, 0.75])

        self.assertEqual(result["sample_count"], 3)
        self.assertAlmostEqual(result["mse"], 1.0 / 24.0)
        self.assertAlmostEqual(result["rmse"], np.sqrt(1.0 / 24.0))
        self.assertAlmostEqual(result["r2"], 0.75)
        self.assertAlmostEqual(result["pearson"], 1.0)
        self.assertAlmostEqual(result["spearman"], 1.0)
        self.assertEqual(result["undefined_reasons"], {})
        self.assertNotIn("unique_rc_group_count", result)

    def test_hand_calculated_reversed_predictions_preserve_negative_r2(self) -> None:
        result = compute_regression_metrics([0, 0.5, 1], [1, 0.5, 0])

        self.assertAlmostEqual(result["mse"], 2.0 / 3.0)
        self.assertAlmostEqual(result["r2"], -3.0)
        self.assertAlmostEqual(result["pearson"], -1.0)
        self.assertAlmostEqual(result["spearman"], -1.0)

    def test_hand_calculated_average_tie_ranks(self) -> None:
        # Target ranks [1, 2.5, 2.5, 4]; prediction ranks [1, 2, 3.5, 3.5].
        # The centered rank dot product is 3.75 and both squared norms are 4.5.
        result = compute_regression_metrics([0, 0.5, 0.5, 1], [0, 0.5, 1, 1])

        self.assertAlmostEqual(result["spearman"], 5.0 / 6.0)
        self.assertAlmostEqual(result["mse"], 1.0 / 16.0)
        self.assertAlmostEqual(result["r2"], 0.5)

    def test_constant_predictions_keep_r2_defined(self) -> None:
        result = compute_regression_metrics([0, 0.5, 1], [0.25, 0.25, 0.25])

        self.assertAlmostEqual(result["mse"], 11.0 / 48.0)
        self.assertAlmostEqual(result["r2"], -0.375)
        self.assertIsNone(result["pearson"])
        self.assertIsNone(result["spearman"])
        self.assertEqual(
            result["undefined_reasons"],
            {"pearson": "constant_predictions", "spearman": "constant_predictions"},
        )

    def test_constant_targets_exact_inexact_and_precedence(self) -> None:
        cases = (
            ([0.5, 0.5, 0.5], 0.0),
            ([0.25, 0.25, 0.25], 1.0 / 16.0),
            ([0.0, 0.5, 1.0], 1.0 / 6.0),
        )
        for predictions, expected_mse in cases:
            with self.subTest(predictions=predictions):
                result = compute_regression_metrics([0.5, 0.5, 0.5], predictions)
                self.assertAlmostEqual(result["mse"], expected_mse)
                self.assertAlmostEqual(result["rmse"], np.sqrt(expected_mse))
                for metric in ("r2", "pearson", "spearman"):
                    self.assertIsNone(result[metric])
                    self.assertEqual(
                        result["undefined_reasons"][metric], "constant_targets"
                    )

    def test_singleton_retains_mse_rmse_with_reason_precedence(self) -> None:
        result = compute_regression_metrics([0.25], [0.75], ["group-a"])

        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["unique_rc_group_count"], 1)
        self.assertEqual(result["mse"], 0.25)
        self.assertEqual(result["rmse"], 0.5)
        for metric in ("r2", "pearson", "spearman"):
            self.assertIsNone(result[metric])
            self.assertEqual(
                result["undefined_reasons"][metric], "insufficient_samples"
            )

    def test_optional_groups_count_without_collapse_or_reweighting(self) -> None:
        targets = [0.0, 0.0, 1.0]
        predictions = [0.0, 0.0, 0.0]
        groups = ["duplicate", "duplicate", "other"]

        ungrouped = compute_regression_metrics(targets, predictions)
        grouped = compute_regression_metrics(targets, predictions, groups)

        self.assertEqual(grouped.pop("unique_rc_group_count"), 2)
        self.assertEqual(grouped, ungrouped)
        self.assertEqual(grouped["sample_count"], 3)
        self.assertAlmostEqual(grouped["mse"], 1.0 / 3.0)
        self.assertAlmostEqual(grouped["r2"], -0.5)

    def test_joint_permutation_invariance_and_float64_input_preservation(self) -> None:
        targets = np.array([0, 0.25, 0.5, 0.5, 1], dtype=np.float32)
        predictions = np.array([0.125, 0.375, 0.875, 0.625, 0.75], dtype=np.float64)
        groups = np.array(["a", "b", "c", "c", "d"], dtype=object)
        original_targets = targets.copy()
        original_predictions = predictions.copy()
        original_groups = groups.copy()
        targets.setflags(write=False)
        predictions.setflags(write=False)
        groups.setflags(write=False)

        expected = compute_regression_metrics(targets, predictions, groups)
        permutation = np.array([4, 2, 0, 3, 1])
        observed = compute_regression_metrics(
            targets[permutation], predictions[permutation], groups[permutation]
        )

        self.assertEqual(expected["undefined_reasons"], observed["undefined_reasons"])
        for metric in ("sample_count", "unique_rc_group_count", "mse", "rmse",
                       "r2", "pearson", "spearman"):
            self.assertAlmostEqual(expected[metric], observed[metric])
        np.testing.assert_array_equal(targets, original_targets)
        np.testing.assert_array_equal(predictions, original_predictions)
        np.testing.assert_array_equal(groups, original_groups)
        self.assertEqual(targets.dtype, np.dtype("float32"))
        self.assertEqual(predictions.dtype, np.dtype("float64"))
        self.assertEqual(groups.dtype, np.dtype("object"))
        self.assertFalse(targets.flags.writeable)
        self.assertFalse(predictions.flags.writeable)
        self.assertFalse(groups.flags.writeable)

    def test_real_integer_arrays_and_list_inputs_are_not_mutated(self) -> None:
        targets = [0, 1]
        predictions = np.array([0, 1], dtype=np.int16)
        result = compute_regression_metrics(targets, predictions)

        self.assertEqual(targets, [0, 1])
        np.testing.assert_array_equal(predictions, np.array([0, 1], dtype=np.int16))
        self.assertEqual(predictions.dtype, np.dtype("int16"))
        self.assertEqual(result["r2"], 1.0)
        for metric in ("mse", "rmse", "r2", "pearson", "spearman"):
            self.assertIsInstance(result[metric], float)

    def test_small_nonconstant_values_remain_defined_without_epsilon(self) -> None:
        targets = np.array([0.0, 0.5, 1.0]) * 1e-200
        predictions = np.array([0.25, 0.5, 0.75]) * 1e-200

        result = compute_regression_metrics(targets, predictions)
        exact = compute_regression_metrics(targets, targets)

        self.assertEqual(result["undefined_reasons"], {})
        self.assertAlmostEqual(result["r2"], 0.75)
        self.assertAlmostEqual(result["pearson"], 1.0)
        self.assertAlmostEqual(result["spearman"], 1.0)
        self.assertEqual(exact["r2"], 1.0)
        self.assertEqual(exact["mse"], 0.0)

        constant_predictions = compute_regression_metrics([0.0, 1e-200], [0.0, 0.0])
        self.assertAlmostEqual(constant_predictions["r2"], -1.0)
        self.assertEqual(
            constant_predictions["undefined_reasons"],
            {"pearson": "constant_predictions", "spearman": "constant_predictions"},
        )

    def test_adjacent_and_subnormal_values_retain_centered_correlations(self) -> None:
        cases = (
            np.array([0.0, np.nextafter(0.0, 1.0)]),
            np.array([np.nextafter(1.0, 0.0), 1.0]),
        )
        for targets in cases:
            with self.subTest(targets=targets):
                result = compute_regression_metrics(targets, targets[::-1])
                self.assertEqual(result["undefined_reasons"], {})
                self.assertAlmostEqual(result["r2"], -3.0)
                self.assertAlmostEqual(result["pearson"], -1.0)
                self.assertAlmostEqual(result["spearman"], -1.0)

    def test_full_assay_metrics_are_not_averages_of_batch_metrics(self) -> None:
        targets = [0.0, 0.25, 0.75, 1.0]
        predictions = [0.0, 0.0, 1.0, 1.0]
        full = compute_regression_metrics(targets, predictions)
        first = compute_regression_metrics(targets[:2], predictions[:2])
        second = compute_regression_metrics(targets[2:], predictions[2:])

        self.assertAlmostEqual(full["r2"], 0.8)
        self.assertAlmostEqual((first["r2"] + second["r2"]) / 2.0, -1.0)

    def test_empty_scalar_multidimensional_and_mismatched_inputs_rejected(self) -> None:
        malformed = (
            [],
            0.5,
            np.array(0.5),
            [[0.0], [1.0]],
            np.zeros((2, 1, 1)),
            [0.0, 0.5, 1.0],
        )
        for values in malformed:
            with self.subTest(values=values):
                self.assertRaises(ValueError, compute_regression_metrics, values, [0, 1])
                self.assertRaises(ValueError, compute_regression_metrics, [0, 1], values)

    def test_nonreal_and_nonnumeric_inputs_rejected_before_conversion(self) -> None:
        malformed = (
            [0.0 + 0.0j, 1.0 + 0.0j],
            ["0", "1"],
            [0.0, "1"],
            [False, True],
            [0.0, True],
            np.array([0.0, 1.0], dtype=object),
            [None, 1.0],
            [object(), 1.0],
            np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
        )
        for values in malformed:
            with self.subTest(values=values):
                self.assertRaises(ValueError, compute_regression_metrics, values, [0, 1])
                self.assertRaises(ValueError, compute_regression_metrics, [0, 1], values)

    def test_nonfinite_and_out_of_range_values_rejected_without_clipping(self) -> None:
        malformed = (
            np.array([np.nan, 0.5]),
            np.array([np.inf, 0.5]),
            np.array([-np.inf, 0.5]),
            np.array([-np.finfo(np.float64).eps, 0.5]),
            np.array([0.5, np.nextafter(1.0, 2.0)]),
        )
        for values in malformed:
            original = values.copy()
            with self.subTest(values=values):
                self.assertRaises(ValueError, compute_regression_metrics, values, [0, 1])
                self.assertRaises(ValueError, compute_regression_metrics, [0, 1], values)
                np.testing.assert_array_equal(values, original)

    def test_malformed_optional_group_ids_rejected(self) -> None:
        malformed = (
            "ab", [], ["a"], ["a", "b", "c"], [["a"], ["b"]],
            ["a", ""], ["a", None], ["a", 1], ["a", True],
            [b"a", b"b"],
        )
        for group_ids in malformed:
            with self.subTest(group_ids=group_ids):
                self.assertRaises(
                    ValueError, compute_regression_metrics, [0, 1], [0, 1], group_ids
                )


if __name__ == "__main__":
    unittest.main()
