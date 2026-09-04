"""Synthetic tests for exact/RC-safe Exd-Hox split construction."""

from collections import Counter
from dataclasses import replace
import inspect
import unittest

import numpy as np

from src.coordinates import reverse_complement
from src.exd_hox_splits import (
    ExdHoxSplitError,
    SPLIT_NAMES,
    assign_global_rc_groups,
    audit_primary_split_leakage,
    build_global_rc_groups,
    build_nested_subset_ordering,
    build_tie_preserving_affinity_bins,
    float32_big_endian_hex,
    largest_remainder_counts,
    reconcile_source_occurrences,
    rc_orientation_rows,
    resolve_nested_subset_levels,
    subset_membership_ids,
)
from src.selex_hdf5 import SelexHdf5File


def _record(
    transcription_factor: str,
    supplied_split: str,
    sequences,
    targets,
) -> SelexHdf5File:
    return SelexHdf5File(
        transcription_factor=transcription_factor,
        supplied_split=supplied_split,
        logical_path=(
            "data/raw/synthetic/{0}/{0}_{1}.h5".format(
                transcription_factor,
                supplied_split,
            )
        ),
        sequences=tuple(sequences),
        targets=tuple(targets),
        inventory={},
    )


def _records_for_tf(
    transcription_factor: str,
    training_sequences,
    training_targets,
    test_sequences=(),
    test_targets=(),
):
    return {
        "train": _record(
            transcription_factor,
            "train",
            training_sequences,
            training_targets,
        ),
        "test": _record(
            transcription_factor,
            "test",
            test_sequences,
            test_targets,
        ),
    }


def _sequence_from_index(index: int) -> str:
    bases = "ACGT"
    digits = []
    remaining = index
    for _ in range(14):
        digits.append(bases[remaining % 4])
        remaining //= 4
    return "".join(reversed(digits))


def _single_tf_examples(count: int):
    sequences = tuple(_sequence_from_index(index) for index in range(count))
    targets = tuple((index + 1) / (count + 1) for index in range(count))
    records = {
        "AbdA": _records_for_tf("AbdA", sequences, targets),
    }
    reconciliation = reconcile_source_occurrences(records)
    return reconciliation.logical_examples


class ExdHoxReconciliationTests(unittest.TestCase):
    def test_identical_labeled_occurrences_collapse_with_full_provenance(
        self,
    ) -> None:
        duplicate_sequence = "ACGTTGCAAAAAAA"
        unique_sequence = "CCCCCCCAAAAAAA"
        records = {
            "AbdA": _records_for_tf(
                "AbdA",
                (duplicate_sequence, unique_sequence),
                (0.25, 0.5),
                (duplicate_sequence,),
                (0.25,),
            )
        }

        result = reconcile_source_occurrences(records)

        self.assertEqual(len(result.source_occurrences), 3)
        self.assertEqual(len(result.logical_examples), 2)
        duplicate_examples = [
            example
            for example in result.logical_examples
            if example.sequence == duplicate_sequence
        ]
        self.assertEqual(len(duplicate_examples), 1)
        self.assertEqual(len(duplicate_examples[0].source_occurrence_ids), 2)
        self.assertEqual(
            len(
                {
                    occurrence.source_occurrence_id
                    for occurrence in result.source_occurrences
                }
            ),
            3,
        )
        self.assertEqual(
            result.audit_rows[0]["collapsed_duplicate_occurrence_count"],
            1,
        )
        self.assertEqual(
            duplicate_examples[0].target_bits_big_endian_hex,
            float32_big_endian_hex(np.float32(0.25)),
        )

    def test_exact_sequence_conflict_uses_float32_bits(self) -> None:
        sequence = "ACGTTGCAAAAAAA"
        next_target = float(
            np.nextafter(np.float32(0.25), np.float32(1.0))
        )
        self.assertNotEqual(
            float32_big_endian_hex(0.25),
            float32_big_endian_hex(next_target),
        )
        records = {
            "AbdA": _records_for_tf(
                "AbdA",
                (sequence,),
                (0.25,),
                (sequence,),
                (next_target,),
            )
        }

        with self.assertRaisesRegex(
            ExdHoxSplitError,
            "exact-sequence target-bit conflict",
        ):
            reconcile_source_occurrences(records)

    def test_reverse_complement_label_conflict_fails_without_collapsing(self) -> None:
        sequence = "ACGTTGCAAAAAAA"
        reverse = reverse_complement(sequence)
        records = {
            "AbdA": _records_for_tf(
                "AbdA",
                (sequence,),
                (0.25,),
                (reverse,),
                (0.5,),
            )
        }

        with self.assertRaisesRegex(ExdHoxSplitError, "RC-group target-bit"):
            reconcile_source_occurrences(records)

    def test_equal_label_reverse_complements_group_but_do_not_collapse(
        self,
    ) -> None:
        sequence = "ACGTTGCAAAAAAA"
        reverse = reverse_complement(sequence)
        records = {
            "AbdA": _records_for_tf(
                "AbdA",
                (sequence, reverse),
                (0.25, 0.25),
            )
        }

        reconciliation = reconcile_source_occurrences(records)
        groups = build_global_rc_groups(reconciliation.logical_examples)

        self.assertEqual(len(reconciliation.source_occurrences), 2)
        self.assertEqual(len(reconciliation.logical_examples), 2)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].logical_example_ids), 2)
        self.assertEqual(
            reconciliation.audit_rows[0][
                "collapsed_duplicate_occurrence_count"
            ],
            0,
        )

    def test_cross_tf_labels_share_one_group_without_becoming_conflicts(
        self,
    ) -> None:
        sequence = "ACGTTGCAAAAAAA"
        reverse = reverse_complement(sequence)
        records = {
            "AbdA": _records_for_tf("AbdA", (sequence,), (0.1,)),
            "Ubx": _records_for_tf("Ubx", (reverse,), (0.9,)),
        }

        reconciliation = reconcile_source_occurrences(records)
        groups = build_global_rc_groups(reconciliation.logical_examples)

        self.assertEqual(len(reconciliation.logical_examples), 2)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].transcription_factors, ("AbdA", "Ubx"))
        self.assertEqual(groups[0].transcription_factor_degree, 2)
        self.assertEqual(len(groups[0].logical_example_ids), 2)

    def test_reverse_complement_palindrome_forms_one_ordinary_group(self) -> None:
        palindrome = "AAAAAAATTTTTTT"
        self.assertEqual(reverse_complement(palindrome), palindrome)
        records = {
            "AbdA": _records_for_tf("AbdA", (palindrome,), (0.25,)),
        }

        reconciliation = reconcile_source_occurrences(records)
        groups = build_global_rc_groups(reconciliation.logical_examples)

        self.assertEqual(len(reconciliation.logical_examples), 1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].logical_example_ids[0], (
            reconciliation.logical_examples[0].logical_example_id
        ))


class ExdHoxAffinityBinTests(unittest.TestCase):
    def test_float32_ties_remain_in_one_empirical_midrank_bin(self) -> None:
        sequences = tuple(_sequence_from_index(index) for index in range(8))
        targets = (0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        records = {
            "AbdA": _records_for_tf("AbdA", sequences, targets),
        }
        examples = reconcile_source_occurrences(records).logical_examples

        bins = build_tie_preserving_affinity_bins(
            examples,
            bin_count=4,
            minimum_distinct_groups_per_bin=1,
        )

        tied_bins = {
            bins[example.logical_example_id]
            for example in examples
            if example.target_bits_big_endian_hex
            == float32_big_endian_hex(0.1)
        }
        self.assertEqual(len(tied_bins), 1)

    def test_small_bins_merge_with_lower_neighbor_on_equal_distance(self) -> None:
        examples = _single_tf_examples(6)

        bins = build_tie_preserving_affinity_bins(
            examples,
            bin_count=4,
            minimum_distinct_groups_per_bin=2,
        )
        counts = Counter(bins.values())

        self.assertEqual(counts, Counter({0: 4, 3: 2}))

    def test_tf_with_fewer_than_seven_groups_cannot_support_all_splits(
        self,
    ) -> None:
        examples = _single_tf_examples(6)

        with self.assertRaisesRegex(ExdHoxSplitError, "too few distinct"):
            build_tie_preserving_affinity_bins(
                examples,
                minimum_distinct_groups_per_bin=7,
            )

    def test_80_10_10_stratum_needs_seven_groups_for_all_splits(self) -> None:
        proportions = {"training": 0.8, "validation": 0.1, "test": 0.1}

        six_group_counts = largest_remainder_counts(
            6,
            proportions,
            SPLIT_NAMES,
            ("minimum_supported_stratum",),
        )
        seven_group_counts = largest_remainder_counts(
            7,
            proportions,
            SPLIT_NAMES,
            ("minimum_supported_stratum",),
        )

        self.assertIn(0, six_group_counts.values())
        self.assertTrue(all(count > 0 for count in seven_group_counts.values()))

    def test_largest_remainder_is_exact_and_deterministic(self) -> None:
        proportions = {"training": 0.8, "validation": 0.1, "test": 0.1}

        first = largest_remainder_counts(
            11,
            proportions,
            SPLIT_NAMES,
            ("synthetic", "31001"),
        )
        second = largest_remainder_counts(
            11,
            dict(reversed(tuple(proportions.items()))),
            SPLIT_NAMES,
            ("synthetic", "31001"),
        )

        self.assertEqual(first, second)
        self.assertEqual(first, {"training": 9, "validation": 1, "test": 1})
        self.assertEqual(sum(first.values()), 11)


class ExdHoxGlobalAssignmentTests(unittest.TestCase):
    def test_assignment_is_deterministic_exact_and_model_independent(self) -> None:
        examples = _single_tf_examples(10)
        groups = build_global_rc_groups(examples)
        bins = build_tie_preserving_affinity_bins(
            examples,
            minimum_distinct_groups_per_bin=1,
        )
        proportions = {"training": 0.8, "validation": 0.1, "test": 0.1}

        first = assign_global_rc_groups(
            examples,
            groups,
            bins,
            proportions,
            seed=31001,
        )
        second = assign_global_rc_groups(
            examples,
            tuple(reversed(groups)),
            bins,
            proportions,
            seed=31001,
        )

        self.assertEqual(first, second)
        split_counts = Counter(first[0].values())
        self.assertEqual(
            split_counts,
            Counter({"training": 8, "validation": 1, "test": 1}),
        )
        self.assertEqual(set(first[0]), {group.global_rc_group_id for group in groups})
        signature = inspect.signature(assign_global_rc_groups)
        prohibited_parameters = {
            "model_family",
            "tokenizer",
            "model_seed",
            "checkpoint",
            "physical_features",
            "model_output",
        }
        self.assertTrue(prohibited_parameters.isdisjoint(signature.parameters))

    def test_valid_global_assignment_has_zero_cross_split_leakage(self) -> None:
        examples = _single_tf_examples(10)
        groups = build_global_rc_groups(examples)
        bins = build_tie_preserving_affinity_bins(
            examples,
            minimum_distinct_groups_per_bin=1,
        )
        assignments, unused_order, unused_hashes = assign_global_rc_groups(
            examples,
            groups,
            bins,
            {"training": 0.8, "validation": 0.1, "test": 0.1},
            seed=31001,
        )
        del unused_order
        del unused_hashes

        leakage_rows = audit_primary_split_leakage(examples, assignments)

        self.assertEqual(len(leakage_rows), 3)
        for row in leakage_rows:
            self.assertEqual(row["exact_sequence_overlap_group_count"], 0)
            self.assertEqual(
                row["reverse_complement_equivalent_overlap_group_count"],
                0,
            )
            self.assertEqual(row["reverse_complement_only_overlap_group_count"], 0)
            self.assertEqual(row["logical_example_overlap_count"], 0)

    def test_leakage_audit_distinguishes_exact_and_rc_only_groups(self) -> None:
        exact_sequence = "ACGTTGCAAAAAAA"
        rc_sequence = "CCCCCCCAAAAAAA"
        records = {
            "AbdA": _records_for_tf(
                "AbdA",
                (exact_sequence, rc_sequence),
                (0.1, 0.2),
            ),
            "Ubx": _records_for_tf(
                "Ubx",
                (exact_sequence, reverse_complement(rc_sequence)),
                (0.8, 0.9),
            ),
        }
        examples = list(
            reconcile_source_occurrences(records).logical_examples
        )
        malformed_examples = []
        assignments = {}
        for index, example in enumerate(examples):
            malformed_group_id = "malformed_group_{0}".format(index)
            malformed_examples.append(
                replace(example, global_rc_group_id=malformed_group_id)
            )
            if example.transcription_factor == "AbdA":
                assignments[malformed_group_id] = "training"
            else:
                assignments[malformed_group_id] = "test"

        leakage_rows = audit_primary_split_leakage(
            malformed_examples,
            assignments,
        )
        training_test = next(
            row
            for row in leakage_rows
            if row["left_split"] == "training" and row["right_split"] == "test"
        )

        self.assertEqual(training_test["exact_sequence_overlap_group_count"], 1)
        self.assertEqual(
            training_test["reverse_complement_equivalent_overlap_group_count"],
            2,
        )
        self.assertEqual(
            training_test["reverse_complement_only_overlap_group_count"],
            1,
        )


class ExdHoxNestedSubsetTests(unittest.TestCase):
    def test_ordering_uses_only_primary_training_groups(self) -> None:
        examples = _single_tf_examples(20)
        assignments = {}
        training_ids = set()
        for index, example in enumerate(examples):
            if index < 16:
                assignments[example.global_rc_group_id] = "training"
                training_ids.add(example.logical_example_id)
            elif index < 18:
                assignments[example.global_rc_group_id] = "validation"
            else:
                assignments[example.global_rc_group_id] = "test"

        first = build_nested_subset_ordering(
            examples,
            assignments,
            seed=32001,
            affinity_bin_count=4,
            minimum_distinct_groups_per_bin=1,
        )
        second = build_nested_subset_ordering(
            tuple(reversed(examples)),
            assignments,
            seed=32001,
            affinity_bin_count=4,
            minimum_distinct_groups_per_bin=1,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(
            {row["logical_example_id"] for row in first},
            training_ids,
        )
        self.assertEqual(
            {row["rank_one_based"] for row in first},
            set(range(1, 17)),
        )

    def test_levels_are_prefixes_and_fractional_aliases_keep_anchors(self) -> None:
        ordering_rows = tuple(
            {
                "transcription_factor": "AbdA",
                "rank_one_based": rank,
                "global_rc_group_id": "group_{0}".format(rank),
                "logical_example_id": "example_{0}".format(rank),
                "training_affinity_bin": rank % 10,
                "deterministic_order_sha256": "{0:064x}".format(rank),
            }
            for rank in range(1, 1001)
        )

        levels = resolve_nested_subset_levels(
            ordering_rows,
            absolute_counts=(128, 256, 512),
            fractional_levels=("0.01", "0.1285", "0.25", "0.50", "1.00"),
            absolute_alias_tolerance="0.05",
            minimum_primary_count=128,
        )

        fractional_rows = {
            row["request_value"]: row
            for row in levels
            if row["request_type"] == "fractional"
        }
        self.assertNotIn("0.01", fractional_rows)
        rounded_alias = fractional_rows["0.1285"]
        self.assertEqual(
            rounded_alias["unaliased_requested_logical_example_count"],
            129,
        )
        self.assertEqual(rounded_alias["alias_absolute_anchor"], 128)
        self.assertEqual(
            rounded_alias["canonical_requested_logical_example_count"],
            128,
        )
        self.assertEqual(fractional_rows["0.25"]["alias_absolute_anchor"], 256)
        self.assertEqual(fractional_rows["0.5"]["alias_absolute_anchor"], 512)

        by_canonical_count = {}
        for level in levels:
            canonical_count = level[
                "canonical_requested_logical_example_count"
            ]
            by_canonical_count.setdefault(canonical_count, level)
        previous_membership = ()
        for canonical_count in sorted(by_canonical_count):
            membership = subset_membership_ids(
                ordering_rows,
                by_canonical_count[canonical_count],
            )
            self.assertEqual(
                membership[: len(previous_membership)],
                previous_membership,
            )
            previous_membership = membership

    def test_level_reports_logical_and_group_counts_at_group_boundary(self) -> None:
        ordering_rows = (
            {
                "transcription_factor": "AbdA",
                "rank_one_based": 1,
                "global_rc_group_id": "group_1",
                "logical_example_id": "example_1a",
            },
            {
                "transcription_factor": "AbdA",
                "rank_one_based": 1,
                "global_rc_group_id": "group_1",
                "logical_example_id": "example_1b",
            },
            {
                "transcription_factor": "AbdA",
                "rank_one_based": 2,
                "global_rc_group_id": "group_2",
                "logical_example_id": "example_2",
            },
        )

        levels = resolve_nested_subset_levels(
            ordering_rows,
            absolute_counts=(1, 3),
            fractional_levels=(),
            minimum_primary_count=1,
        )
        first_level = levels[0]

        self.assertEqual(
            first_level["canonical_requested_logical_example_count"],
            1,
        )
        self.assertEqual(first_level["actual_logical_example_count"], 2)
        self.assertEqual(first_level["actual_rc_group_count"], 1)
        self.assertEqual(
            subset_membership_ids(ordering_rows, first_level),
            ("example_1a", "example_1b"),
        )

    def test_rc_orientations_do_not_create_additional_labeled_examples(
        self,
    ) -> None:
        non_palindrome = "ACGTTGCAAAAAAA"
        palindrome = "AAAAAAATTTTTTT"
        records = {
            "AbdA": _records_for_tf(
                "AbdA",
                (non_palindrome, palindrome),
                (0.25, 0.5),
            )
        }
        examples = reconcile_source_occurrences(records).logical_examples
        example_by_sequence = {example.sequence: example for example in examples}

        non_palindrome_rows = rc_orientation_rows(
            example_by_sequence[non_palindrome]
        )
        palindrome_rows = rc_orientation_rows(example_by_sequence[palindrome])

        self.assertEqual(len(non_palindrome_rows), 2)
        self.assertEqual(len(palindrome_rows), 1)
        self.assertEqual(
            {row["logical_example_id"] for row in non_palindrome_rows},
            {example_by_sequence[non_palindrome].logical_example_id},
        )
        self.assertEqual(
            {row["global_rc_group_id"] for row in non_palindrome_rows},
            {example_by_sequence[non_palindrome].global_rc_group_id},
        )


if __name__ == "__main__":
    unittest.main()
