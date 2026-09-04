"""Deterministic exact/RC-safe Exd-Hox splits and nested subsets.

This module contains no model, tokenizer, checkpoint, or test-target access
logic.  The only HDF5 entry point validates the immutable Milestone 3C source
before constructing in-memory logical examples.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

from src.coordinates import reverse_complement
from src.data_fingerprints import reverse_complement_canonical_sequence
from src.downstream_fingerprints import (
    build_hashed_manifest,
    fingerprint_file,
    hash_file_bytes,
    hash_logical_content,
    repository_relative_path,
    validate_hashed_manifest,
    validate_repository_relative_path,
    write_json_exclusive,
    write_tsv_exclusive,
    write_tsv_gzip_exclusive,
)
from src.selex_hdf5 import (
    SelexHdf5File,
    audit_supplied_split,
    read_validate_selex_hdf5,
)
from src.sealed_test_access import validate_test_access_policy


CONFIG_SCHEMA_VERSION = "exd_hox_primary_split_config.v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "exd_hox_primary_split_manifest.v1"
SUBSET_MANIFEST_SCHEMA_VERSION = "exd_hox_subset_set_manifest.v1"
SEALED_TARGET_MANIFEST_SCHEMA_VERSION = (
    "exd_hox_sealed_test_target_manifest.v1"
)
SPLIT_POLICY_IDENTIFIER = "global_rc_affinity_stratified_80_10_10.v1"
SPLIT_NAMES = ("training", "validation", "test")
SUPPLIED_SPLIT_NAMES = ("train", "test")

LOGICAL_EXAMPLES_FILENAME = "exd_hox_logical_examples_v1.tsv.gz"
PROVENANCE_FILENAME = "exd_hox_source_occurrence_provenance_v1.tsv.gz"
GLOBAL_GROUPS_FILENAME = "exd_hox_global_rc_groups_v1.tsv.gz"
ASSIGNMENTS_FILENAME = "exd_hox_primary_split_assignments_v1.tsv.gz"
RECONCILIATION_AUDIT_FILENAME = "exd_hox_reconciliation_audit_v1.tsv"
COUNT_SUMMARY_FILENAME = "exd_hox_primary_split_count_summary_v1.tsv"
LEAKAGE_AUDIT_FILENAME = "exd_hox_primary_split_leakage_audit_v1.tsv"
AFFINITY_HISTOGRAM_FILENAME = (
    "exd_hox_primary_split_affinity_histogram_v1.tsv"
)
PUBLIC_TEST_INPUTS_FILENAME = "exd_hox_public_test_inputs_v1.tsv.gz"
SPLIT_MANIFEST_FILENAME = "exd_hox_primary_split_manifest_v1.json"
SEALED_TARGET_FILENAME = "exd_hox_sealed_test_targets_v1.tsv.gz"
SEALED_TARGET_MANIFEST_FILENAME = (
    "exd_hox_sealed_test_target_manifest_v1.json"
)

SUBSET_ORDERING_FILENAME = "exd_hox_nested_subset_ordering_v1.tsv.gz"
SUBSET_LEVELS_FILENAME = "exd_hox_nested_subset_levels_v1.tsv"
SUBSET_MANIFEST_FILENAME = "exd_hox_subset_set_manifest_v1.json"

LOGICAL_EXAMPLE_FIELDS = (
    "logical_example_id",
    "transcription_factor",
    "sequence",
    "sequence_sha256",
    "reverse_complement_canonical_sequence",
    "reverse_complement_canonical_sha256",
    "global_rc_group_id",
    "primary_split",
    "target_value_float32",
    "target_bits_big_endian_hex",
    "target_commitment_sha256",
    "source_occurrence_count",
)
PROVENANCE_FIELDS = (
    "source_occurrence_id",
    "logical_example_id",
    "transcription_factor",
    "supplied_split",
    "source_path",
    "source_row_index_zero_based",
    "sequence",
    "sequence_sha256",
    "reverse_complement_canonical_sha256",
    "global_rc_group_id",
    "primary_split",
    "target_value_float32",
    "target_bits_big_endian_hex",
    "target_commitment_sha256",
    "is_collapsed_duplicate",
)
GLOBAL_GROUP_FIELDS = (
    "global_rc_group_id",
    "reverse_complement_canonical_sequence",
    "reverse_complement_canonical_sha256",
    "transcription_factor_degree",
    "transcription_factors",
    "logical_example_count",
    "primary_split",
)
ASSIGNMENT_FIELDS = (
    "global_rc_group_id",
    "primary_split",
    "assignment_order_index_zero_based",
    "deterministic_order_sha256",
)
RECONCILIATION_AUDIT_FIELDS = (
    "transcription_factor",
    "source_occurrence_count",
    "logical_example_count",
    "collapsed_duplicate_occurrence_count",
    "exact_labeled_duplicate_group_count",
    "exact_sequence_label_conflict_count",
    "reverse_complement_label_conflict_count",
)
COUNT_SUMMARY_FIELDS = (
    "protocol",
    "transcription_factor",
    "split",
    "row_count",
    "logical_example_count",
    "global_rc_group_count",
    "exact_cross_split_overlap_occurrence_count",
)
LEAKAGE_AUDIT_FIELDS = (
    "comparison",
    "left_split",
    "right_split",
    "exact_sequence_overlap_group_count",
    "reverse_complement_equivalent_overlap_group_count",
    "reverse_complement_only_overlap_group_count",
    "logical_example_overlap_count",
)
AFFINITY_HISTOGRAM_FIELDS = (
    "transcription_factor",
    "split",
    "bin_index",
    "bin_left",
    "bin_right",
    "logical_example_count",
)
PUBLIC_TEST_INPUT_FIELDS = (
    "logical_example_id",
    "transcription_factor",
    "sequence",
    "sequence_sha256",
    "reverse_complement_canonical_sequence",
    "reverse_complement_canonical_sha256",
    "global_rc_group_id",
    "target_commitment_sha256",
)
SEALED_TARGET_FIELDS = (
    "logical_example_id",
    "target_value_float32",
    "target_bits_big_endian_hex",
    "target_commitment_sha256",
)
SUBSET_ORDERING_FIELDS = (
    "transcription_factor",
    "rank_one_based",
    "global_rc_group_id",
    "logical_example_id",
    "training_affinity_bin",
    "deterministic_order_sha256",
)
SUBSET_LEVEL_FIELDS = (
    "transcription_factor",
    "level_id",
    "request_type",
    "request_value",
    "unaliased_requested_logical_example_count",
    "alias_absolute_anchor",
    "canonical_requested_logical_example_count",
    "actual_logical_example_count",
    "actual_rc_group_count",
    "inclusive_maximum_rank",
)


class ExdHoxSplitError(ValueError):
    """Raised when split inputs or deterministic invariants fail closed."""


@dataclass(frozen=True, slots=True)
class SourceOccurrence:
    """One source HDF5 row retained for reconciliation provenance."""

    source_occurrence_id: str
    transcription_factor: str
    supplied_split: str
    source_path: str
    source_row_index_zero_based: int
    sequence: str
    sequence_sha256: str
    reverse_complement_canonical_sequence: str
    reverse_complement_canonical_sha256: str
    target_value: float
    target_bits_big_endian_hex: str
    logical_example_id: str


@dataclass(frozen=True, slots=True)
class LogicalExample:
    """One exact TF/sequence/float32-bit logical labeled example."""

    logical_example_id: str
    transcription_factor: str
    sequence: str
    sequence_sha256: str
    reverse_complement_canonical_sequence: str
    reverse_complement_canonical_sha256: str
    global_rc_group_id: str
    target_value: float
    target_bits_big_endian_hex: str
    target_commitment_sha256: str
    source_occurrence_ids: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GlobalRcGroup:
    """A cross-assay reverse-complement-equivalence group."""

    global_rc_group_id: str
    reverse_complement_canonical_sequence: str
    reverse_complement_canonical_sha256: str
    transcription_factors: Tuple[str, ...]
    logical_example_ids: Tuple[str, ...]

    @property
    def transcription_factor_degree(self) -> int:
        """Return the number of distinct TF assays represented."""

        return len(self.transcription_factors)


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Bit-exact logical examples plus complete occurrence provenance."""

    logical_examples: Tuple[LogicalExample, ...]
    source_occurrences: Tuple[SourceOccurrence, ...]
    audit_rows: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PrimarySplitResult:
    """Complete deterministic primary split held in memory."""

    reconciliation: ReconciliationResult
    global_groups: Tuple[GlobalRcGroup, ...]
    assignments: Mapping[str, str]
    assignment_order: Tuple[str, ...]
    assignment_hashes: Mapping[str, str]
    affinity_bins: Mapping[str, int]


def float32_big_endian_hex(value: float) -> str:
    """Return the exact big-endian IEEE-754 float32 bytes as hex."""

    return struct.pack(">f", float(value)).hex()


def float32_from_big_endian_hex(value: str) -> float:
    """Decode one exact big-endian IEEE-754 float32 bit pattern."""

    raw = bytes.fromhex(value)
    if len(raw) != 4:
        raise ExdHoxSplitError("A float32 target bit pattern must contain 4 bytes.")
    return float(struct.unpack(">f", raw)[0])


def float32_text(value: float) -> str:
    """Return a short decimal that round-trips to the same float32 bits."""

    normalized = float32_from_big_endian_hex(float32_big_endian_hex(value))
    return format(normalized, ".9g")


def target_commitment(
    logical_example_id: str,
    target_bits_big_endian_hex: str,
) -> str:
    """Commit to a target without exposing its decimal value or raw bits."""

    return _domain_hash(
        "exd_hox_target_commitment.v1",
        logical_example_id,
        target_bits_big_endian_hex,
    )


def reconcile_source_occurrences(
    records_by_tf: Mapping[str, Mapping[str, SelexHdf5File]],
) -> ReconciliationResult:
    """Collapse only identical TF/sequence/float32-bit source occurrences."""

    occurrences = []
    occurrence_ids_by_logical = defaultdict(list)
    target_bits_by_exact_sequence = defaultdict(set)
    target_bits_by_rc_group = defaultdict(set)
    target_value_by_bits = {}
    source_count_by_tf = Counter()
    logical_keys_by_tf = defaultdict(set)

    for transcription_factor in sorted(records_by_tf):
        split_records = records_by_tf[transcription_factor]
        if set(split_records) != set(SUPPLIED_SPLIT_NAMES):
            raise ExdHoxSplitError(
                "Each TF must contain exactly supplied train and test records."
            )
        for supplied_split in SUPPLIED_SPLIT_NAMES:
            record = split_records[supplied_split]
            if record.transcription_factor != transcription_factor:
                raise ExdHoxSplitError("Source record TF identity mismatch.")
            if record.supplied_split != supplied_split:
                raise ExdHoxSplitError("Source record split identity mismatch.")
            if len(record.sequences) != len(record.targets):
                raise ExdHoxSplitError("Source sequence and target counts differ.")
            for row_index, (sequence, target_value) in enumerate(
                zip(record.sequences, record.targets)
            ):
                target_bits = float32_big_endian_hex(target_value)
                canonical_sequence = reverse_complement_canonical_sequence(
                    sequence
                )
                sequence_hash = _sequence_hash(sequence)
                canonical_hash = _sequence_hash(canonical_sequence)
                logical_example_id = _stable_identifier(
                    "lex",
                    "exd_hox_logical_example.v1",
                    transcription_factor,
                    sequence,
                    target_bits,
                )
                occurrence_id = _stable_identifier(
                    "occ",
                    "exd_hox_source_occurrence.v1",
                    transcription_factor,
                    supplied_split,
                    record.logical_path,
                    str(row_index),
                    sequence,
                    target_bits,
                )
                occurrence = SourceOccurrence(
                    source_occurrence_id=occurrence_id,
                    transcription_factor=transcription_factor,
                    supplied_split=supplied_split,
                    source_path=record.logical_path,
                    source_row_index_zero_based=row_index,
                    sequence=sequence,
                    sequence_sha256=sequence_hash,
                    reverse_complement_canonical_sequence=canonical_sequence,
                    reverse_complement_canonical_sha256=canonical_hash,
                    target_value=float32_from_big_endian_hex(target_bits),
                    target_bits_big_endian_hex=target_bits,
                    logical_example_id=logical_example_id,
                )
                occurrences.append(occurrence)
                occurrence_ids_by_logical[logical_example_id].append(
                    occurrence_id
                )
                exact_key = (transcription_factor, sequence)
                rc_key = (transcription_factor, canonical_sequence)
                target_bits_by_exact_sequence[exact_key].add(target_bits)
                target_bits_by_rc_group[rc_key].add(target_bits)
                target_value_by_bits[target_bits] = occurrence.target_value
                source_count_by_tf[transcription_factor] += 1
                logical_keys_by_tf[transcription_factor].add(logical_example_id)

    exact_conflicts = _conflicting_keys(target_bits_by_exact_sequence)
    if exact_conflicts:
        transcription_factor, sequence = exact_conflicts[0]
        message = "Within-TF exact-sequence target-bit conflict: {0} {1}."
        raise ExdHoxSplitError(message.format(transcription_factor, sequence))
    rc_conflicts = _conflicting_keys(target_bits_by_rc_group)
    if rc_conflicts:
        transcription_factor, canonical_sequence = rc_conflicts[0]
        message = "Within-TF RC-group target-bit conflict: {0} {1}."
        raise ExdHoxSplitError(
            message.format(transcription_factor, canonical_sequence)
        )

    occurrence_by_id = {
        occurrence.source_occurrence_id: occurrence
        for occurrence in occurrences
    }
    logical_examples = []
    for logical_example_id in sorted(occurrence_ids_by_logical):
        occurrence_ids = tuple(sorted(occurrence_ids_by_logical[logical_example_id]))
        representative = occurrence_by_id[occurrence_ids[0]]
        group_id = _stable_identifier(
            "rcg",
            "exd_hox_global_rc_group.v1",
            representative.reverse_complement_canonical_sequence,
        )
        commitment = target_commitment(
            logical_example_id,
            representative.target_bits_big_endian_hex,
        )
        logical_examples.append(
            LogicalExample(
                logical_example_id=logical_example_id,
                transcription_factor=representative.transcription_factor,
                sequence=representative.sequence,
                sequence_sha256=representative.sequence_sha256,
                reverse_complement_canonical_sequence=(
                    representative.reverse_complement_canonical_sequence
                ),
                reverse_complement_canonical_sha256=(
                    representative.reverse_complement_canonical_sha256
                ),
                global_rc_group_id=group_id,
                target_value=representative.target_value,
                target_bits_big_endian_hex=(
                    representative.target_bits_big_endian_hex
                ),
                target_commitment_sha256=commitment,
                source_occurrence_ids=occurrence_ids,
            )
        )

    audit_rows = []
    for transcription_factor in sorted(records_by_tf):
        logical_ids = logical_keys_by_tf[transcription_factor]
        duplicate_groups = 0
        for logical_example_id in logical_ids:
            if len(occurrence_ids_by_logical[logical_example_id]) > 1:
                duplicate_groups += 1
        source_count = source_count_by_tf[transcription_factor]
        logical_count = len(logical_ids)
        audit_rows.append(
            {
                "transcription_factor": transcription_factor,
                "source_occurrence_count": source_count,
                "logical_example_count": logical_count,
                "collapsed_duplicate_occurrence_count": (
                    source_count - logical_count
                ),
                "exact_labeled_duplicate_group_count": duplicate_groups,
                "exact_sequence_label_conflict_count": 0,
                "reverse_complement_label_conflict_count": 0,
            }
        )

    occurrences.sort(
        key=lambda row: (
            row.transcription_factor,
            SUPPLIED_SPLIT_NAMES.index(row.supplied_split),
            row.source_path,
            row.source_row_index_zero_based,
        )
    )
    logical_examples.sort(
        key=lambda row: (
            row.transcription_factor,
            row.sequence,
            row.target_bits_big_endian_hex,
        )
    )
    return ReconciliationResult(
        logical_examples=tuple(logical_examples),
        source_occurrences=tuple(occurrences),
        audit_rows=tuple(audit_rows),
    )


def build_global_rc_groups(
    logical_examples: Sequence[LogicalExample],
) -> Tuple[GlobalRcGroup, ...]:
    """Build global cross-TF RC groups from logical examples."""

    examples_by_group = defaultdict(list)
    for example in logical_examples:
        examples_by_group[example.global_rc_group_id].append(example)

    groups = []
    for group_id in sorted(examples_by_group):
        examples = examples_by_group[group_id]
        canonical_sequences = {
            example.reverse_complement_canonical_sequence for example in examples
        }
        canonical_hashes = {
            example.reverse_complement_canonical_sha256 for example in examples
        }
        if len(canonical_sequences) != 1 or len(canonical_hashes) != 1:
            raise ExdHoxSplitError("Global RC-group identity collision.")
        transcription_factors = tuple(
            sorted({example.transcription_factor for example in examples})
        )
        logical_ids = tuple(
            sorted(example.logical_example_id for example in examples)
        )
        groups.append(
            GlobalRcGroup(
                global_rc_group_id=group_id,
                reverse_complement_canonical_sequence=(
                    next(iter(canonical_sequences))
                ),
                reverse_complement_canonical_sha256=(
                    next(iter(canonical_hashes))
                ),
                transcription_factors=transcription_factors,
                logical_example_ids=logical_ids,
            )
        )
    groups.sort(
        key=lambda row: row.reverse_complement_canonical_sequence
    )
    return tuple(groups)


def build_tie_preserving_affinity_bins(
    logical_examples: Sequence[LogicalExample],
    bin_count: int = 10,
    minimum_distinct_groups_per_bin: int = 7,
) -> Dict[str, int]:
    """Assign exact float32 ties to empirical-midrank affinity bins per TF."""

    if bin_count <= 0:
        raise ExdHoxSplitError("Affinity bin count must be positive.")
    if minimum_distinct_groups_per_bin < 1:
        raise ExdHoxSplitError("Minimum groups per affinity bin must be positive.")
    examples_by_tf = defaultdict(list)
    for example in logical_examples:
        examples_by_tf[example.transcription_factor].append(example)

    bins_by_example = {}
    for transcription_factor in sorted(examples_by_tf):
        examples = examples_by_tf[transcription_factor]
        groups_by_target = defaultdict(list)
        value_by_target = {}
        for example in examples:
            groups_by_target[example.target_bits_big_endian_hex].append(example)
            value_by_target[example.target_bits_big_endian_hex] = example.target_value
        ordered_targets = sorted(
            groups_by_target,
            key=lambda target_bits: (
                value_by_target[target_bits],
                target_bits,
            ),
        )
        raw_bin_by_example = {}
        cumulative_count = 0
        total_count = len(examples)
        for target_bits in ordered_targets:
            tied_examples = groups_by_target[target_bits]
            tied_count = len(tied_examples)
            numerator = bin_count * (2 * cumulative_count + tied_count)
            denominator = 2 * total_count
            raw_bin = min(bin_count - 1, numerator // denominator)
            for example in tied_examples:
                raw_bin_by_example[example.logical_example_id] = raw_bin
            cumulative_count += tied_count

        merged_bin_by_raw = _merged_affinity_bin_mapping(
            examples,
            raw_bin_by_example,
            minimum_distinct_groups_per_bin,
        )
        for example in examples:
            raw_bin = raw_bin_by_example[example.logical_example_id]
            bins_by_example[example.logical_example_id] = merged_bin_by_raw[
                raw_bin
            ]
    return bins_by_example


def largest_remainder_counts(
    total_count: int,
    proportions: Mapping[str, Any],
    ordered_names: Sequence[str],
    tie_context: Sequence[str] = (),
) -> Dict[str, int]:
    """Allocate an integer total by deterministic Hamilton apportionment."""

    if total_count < 0:
        raise ExdHoxSplitError("Largest-remainder total must be nonnegative.")
    if set(proportions) != set(ordered_names):
        raise ExdHoxSplitError("Proportion names must match the ordered names.")
    rational = {}
    for name in ordered_names:
        value = Fraction(str(proportions[name]))
        if value < 0:
            raise ExdHoxSplitError("Split proportions must be nonnegative.")
        rational[name] = value
    proportion_total = sum(rational.values(), Fraction(0, 1))
    if proportion_total <= 0:
        raise ExdHoxSplitError("Split proportions must have a positive sum.")

    floors = {}
    remainders = {}
    allocated = 0
    for name in ordered_names:
        exact = Fraction(total_count, 1) * rational[name] / proportion_total
        floor_value = exact.numerator // exact.denominator
        floors[name] = floor_value
        remainders[name] = exact - floor_value
        allocated += floor_value
    remaining = total_count - allocated
    tie_hashes = {}
    for name in ordered_names:
        tie_hashes[name] = _domain_hash(
            "exd_hox_largest_remainder_tie.v1",
            *tuple(tie_context),
            name,
        )
    ranked_names = sorted(
        ordered_names,
        key=lambda name: (
            -remainders[name],
            tie_hashes[name],
            ordered_names.index(name),
        ),
    )
    for index in range(remaining):
        floors[ranked_names[index]] += 1
    return floors


def assign_global_rc_groups(
    logical_examples: Sequence[LogicalExample],
    global_groups: Sequence[GlobalRcGroup],
    affinity_bins: Mapping[str, int],
    proportions: Mapping[str, Any],
    seed: int,
    policy_identifier: str = SPLIT_POLICY_IDENTIFIER,
    required_per_tf_split_counts: Optional[
        Mapping[str, Mapping[str, int]]
    ] = None,
) -> Tuple[Dict[str, str], Tuple[str, ...], Dict[str, str]]:
    """Assign global RC groups with hard quotas and stratum balancing.

    Groups are processed in a seeded SHA-256 order.  For each group, the
    candidate split that first avoids cell-target overshoot and then most
    reduces normalized squared per-TF/bin deficits is selected.  Every tie is
    resolved by a domain-separated SHA-256 value.
    """

    if tuple(proportions) != SPLIT_NAMES:
        raise ExdHoxSplitError(
            "Split proportions must use training, validation, test order."
        )
    example_by_id = {
        example.logical_example_id: example for example in logical_examples
    }
    if set(affinity_bins) != set(example_by_id):
        raise ExdHoxSplitError("Affinity bins must cover every logical example.")

    contributions_by_group = {}
    cell_totals = Counter()
    order_hashes = {}
    group_by_id = {}
    for group in global_groups:
        if group.global_rc_group_id in group_by_id:
            raise ExdHoxSplitError("Duplicate global RC-group identifier.")
        group_by_id[group.global_rc_group_id] = group
        contributions = Counter()
        for logical_example_id in group.logical_example_ids:
            if logical_example_id not in example_by_id:
                raise ExdHoxSplitError(
                    "Global RC group refers to an unknown logical example."
                )
            example = example_by_id[logical_example_id]
            cell = (
                example.transcription_factor,
                affinity_bins[logical_example_id],
            )
            contributions[cell] += 1
            cell_totals[cell] += 1
        contributions_by_group[group.global_rc_group_id] = contributions
        order_hashes[group.global_rc_group_id] = _domain_hash(
            "exd_hox_primary_group_order.v1",
            policy_identifier,
            str(seed),
            group.reverse_complement_canonical_sequence,
        )

    global_targets = largest_remainder_counts(
        len(global_groups),
        proportions,
        SPLIT_NAMES,
        (policy_identifier, str(seed), "global_group_quota"),
    )
    cell_targets = {}
    for cell in sorted(cell_totals):
        transcription_factor, affinity_bin = cell
        counts = largest_remainder_counts(
            cell_totals[cell],
            proportions,
            SPLIT_NAMES,
            (
                policy_identifier,
                str(seed),
                transcription_factor,
                str(affinity_bin),
            ),
        )
        for split in SPLIT_NAMES:
            cell_targets[(cell, split)] = counts[split]

    ordered_groups = sorted(
        global_groups,
        key=lambda group: (
            order_hashes[group.global_rc_group_id],
            group.global_rc_group_id,
        ),
    )
    assigned_global_counts = Counter()
    assigned_cell_counts = Counter()
    assignments = {}

    for group in ordered_groups:
        group_id = group.global_rc_group_id
        contributions = contributions_by_group[group_id]
        candidate_scores = []
        for split in SPLIT_NAMES:
            if assigned_global_counts[split] >= global_targets[split]:
                continue
            overshoot_cells = 0
            overshoot_amount = 0
            normalized_residual = 0.0
            absolute_residual = 0
            fulfilled_deficit = 0
            for cell, contribution in contributions.items():
                target = cell_targets[(cell, split)]
                before = assigned_cell_counts[(cell, split)]
                after = before + contribution
                if after > target:
                    overshoot_cells += 1
                    overshoot_amount += after - target
                residual = target - after
                denominator = max(target, 1)
                normalized_residual += (residual / denominator) ** 2
                absolute_residual += abs(residual)
                fulfilled_deficit += min(max(target - before, 0), contribution)
            remaining_fraction = (
                global_targets[split] - assigned_global_counts[split]
            ) / max(global_targets[split], 1)
            tie_hash = _domain_hash(
                "exd_hox_primary_assignment_tie.v1",
                policy_identifier,
                str(seed),
                group_id,
                split,
            )
            score = (
                overshoot_cells,
                overshoot_amount,
                normalized_residual,
                absolute_residual,
                -fulfilled_deficit,
                -remaining_fraction,
                tie_hash,
                SPLIT_NAMES.index(split),
            )
            candidate_scores.append((score, split))
        if not candidate_scores:
            raise ExdHoxSplitError("No global split quota remains for a group.")
        candidate_scores.sort(key=lambda item: item[0])
        selected_split = candidate_scores[0][1]
        assignments[group_id] = selected_split
        assigned_global_counts[selected_split] += 1
        for cell, contribution in contributions.items():
            assigned_cell_counts[(cell, selected_split)] += contribution

    if dict(assigned_global_counts) != global_targets:
        raise ExdHoxSplitError("Hard global RC-group quotas were not achieved.")
    assignment_order = tuple(
        group.global_rc_group_id for group in ordered_groups
    )
    if required_per_tf_split_counts is not None:
        assignments = _repair_approved_tf_margins(
            global_groups=global_groups,
            contributions_by_group=contributions_by_group,
            assignments=assignments,
            order_hashes=order_hashes,
            required_per_tf_split_counts=required_per_tf_split_counts,
            seed=seed,
            policy_identifier=policy_identifier,
        )
    return assignments, assignment_order, order_hashes


def build_primary_split(
    records_by_tf: Mapping[str, Mapping[str, SelexHdf5File]],
    proportions: Mapping[str, Any],
    seed: int,
    affinity_bin_count: int = 10,
    minimum_distinct_groups_per_bin: int = 7,
    policy_identifier: str = SPLIT_POLICY_IDENTIFIER,
    required_per_tf_split_counts: Optional[
        Mapping[str, Mapping[str, int]]
    ] = None,
) -> PrimarySplitResult:
    """Construct the complete deterministic primary split in memory."""

    reconciliation = reconcile_source_occurrences(records_by_tf)
    global_groups = build_global_rc_groups(reconciliation.logical_examples)
    affinity_bins = build_tie_preserving_affinity_bins(
        reconciliation.logical_examples,
        bin_count=affinity_bin_count,
        minimum_distinct_groups_per_bin=minimum_distinct_groups_per_bin,
    )
    assignments, assignment_order, order_hashes = assign_global_rc_groups(
        reconciliation.logical_examples,
        global_groups,
        affinity_bins,
        proportions,
        seed,
        policy_identifier,
        required_per_tf_split_counts,
    )
    return PrimarySplitResult(
        reconciliation=reconciliation,
        global_groups=global_groups,
        assignments=assignments,
        assignment_order=assignment_order,
        assignment_hashes=order_hashes,
        affinity_bins=affinity_bins,
    )


def audit_primary_split_leakage(
    logical_examples: Sequence[LogicalExample],
    assignments: Mapping[str, str],
) -> Tuple[Dict[str, Any], ...]:
    """Audit exact, RC-equivalent, and RC-only leakage between split pairs."""

    by_split = {}
    for split in SPLIT_NAMES:
        selected = [
            example
            for example in logical_examples
            if assignments[example.global_rc_group_id] == split
        ]
        exact_sequences = defaultdict(set)
        canonical_sequences = defaultdict(set)
        logical_ids = set()
        for example in selected:
            exact_sequences[example.sequence].add(example.sequence)
            canonical_sequences[
                example.reverse_complement_canonical_sequence
            ].add(example.sequence)
            logical_ids.add(example.logical_example_id)
        by_split[split] = {
            "exact": exact_sequences,
            "canonical": canonical_sequences,
            "logical": logical_ids,
        }

    rows = []
    pairs = (
        ("training", "validation"),
        ("training", "test"),
        ("validation", "test"),
    )
    for left_split, right_split in pairs:
        left = by_split[left_split]
        right = by_split[right_split]
        exact_overlap = set(left["exact"]) & set(right["exact"])
        canonical_overlap = set(left["canonical"]) & set(right["canonical"])
        rc_only = 0
        for canonical_sequence in canonical_overlap:
            left_sequences = left["canonical"][canonical_sequence]
            right_sequences = right["canonical"][canonical_sequence]
            if left_sequences.isdisjoint(right_sequences):
                rc_only += 1
        rows.append(
            {
                "comparison": "primary",
                "left_split": left_split,
                "right_split": right_split,
                "exact_sequence_overlap_group_count": len(exact_overlap),
                "reverse_complement_equivalent_overlap_group_count": len(
                    canonical_overlap
                ),
                "reverse_complement_only_overlap_group_count": rc_only,
                "logical_example_overlap_count": len(
                    left["logical"] & right["logical"]
                ),
            }
        )
    return tuple(rows)


def primary_split_counts(
    logical_examples: Sequence[LogicalExample],
    assignments: Mapping[str, str],
) -> Dict[str, Dict[str, int]]:
    """Return per-TF logical-example counts for each primary split."""

    counts = defaultdict(Counter)
    for example in logical_examples:
        split = assignments[example.global_rc_group_id]
        counts[example.transcription_factor][split] += 1
    result = {}
    for transcription_factor in sorted(counts):
        result[transcription_factor] = {
            split: counts[transcription_factor][split]
            for split in SPLIT_NAMES
        }
    return result


def build_nested_subset_ordering(
    logical_examples: Sequence[LogicalExample],
    assignments: Mapping[str, str],
    seed: int,
    affinity_bin_count: int = 10,
    minimum_distinct_groups_per_bin: int = 7,
) -> Tuple[Dict[str, Any], ...]:
    """Build one training-only, affinity-balanced RC-group order per TF."""

    training_examples = []
    for example in logical_examples:
        if example.global_rc_group_id not in assignments:
            raise ExdHoxSplitError("Subset input is missing a group assignment.")
        if assignments[example.global_rc_group_id] == "training":
            training_examples.append(example)
    if not training_examples:
        raise ExdHoxSplitError("Nested subsets require primary-training examples.")

    training_bins = build_tie_preserving_affinity_bins(
        training_examples,
        bin_count=affinity_bin_count,
        minimum_distinct_groups_per_bin=minimum_distinct_groups_per_bin,
    )
    examples_by_tf_group = defaultdict(list)
    for example in training_examples:
        key = (example.transcription_factor, example.global_rc_group_id)
        examples_by_tf_group[key].append(example)

    groups_by_tf_bin = defaultdict(lambda: defaultdict(list))
    for (transcription_factor, group_id), examples in examples_by_tf_group.items():
        observed_bins = {
            training_bins[example.logical_example_id] for example in examples
        }
        if len(observed_bins) != 1:
            raise ExdHoxSplitError(
                "One within-TF RC group spans multiple training affinity bins."
            )
        affinity_bin = next(iter(observed_bins))
        order_hash = _domain_hash(
            "exd_hox_nested_subset_group_order.v1",
            str(seed),
            transcription_factor,
            group_id,
        )
        groups_by_tf_bin[transcription_factor][affinity_bin].append(
            (order_hash, group_id)
        )

    rows = []
    for transcription_factor in sorted(groups_by_tf_bin):
        ranked_groups = []
        for affinity_bin in sorted(groups_by_tf_bin[transcription_factor]):
            bin_groups = groups_by_tf_bin[transcription_factor][affinity_bin]
            bin_groups.sort()
            bin_count = len(bin_groups)
            for index, (order_hash, group_id) in enumerate(bin_groups):
                balanced_position = Fraction(2 * index + 1, 2 * bin_count)
                ranked_groups.append(
                    (
                        balanced_position,
                        order_hash,
                        affinity_bin,
                        group_id,
                    )
                )
        ranked_groups.sort()
        for rank_index, (
            balanced_position,
            order_hash,
            affinity_bin,
            group_id,
        ) in enumerate(ranked_groups):
            del balanced_position
            examples = examples_by_tf_group[(transcription_factor, group_id)]
            examples.sort(key=lambda example: example.logical_example_id)
            for example in examples:
                rows.append(
                    {
                        "transcription_factor": transcription_factor,
                        "rank_one_based": rank_index + 1,
                        "global_rc_group_id": group_id,
                        "logical_example_id": example.logical_example_id,
                        "training_affinity_bin": affinity_bin,
                        "deterministic_order_sha256": order_hash,
                    }
                )
    return tuple(rows)


def resolve_nested_subset_levels(
    ordering_rows: Sequence[Mapping[str, Any]],
    absolute_counts: Sequence[int],
    fractional_levels: Sequence[Any],
    absolute_alias_tolerance: Any = "0.05",
    minimum_primary_count: int = 128,
) -> Tuple[Dict[str, Any], ...]:
    """Resolve absolute/fractional requests to immutable ordering prefixes."""

    if minimum_primary_count <= 0:
        raise ExdHoxSplitError("Minimum primary subset count must be positive.")
    anchors = tuple(sorted({int(value) for value in absolute_counts}))
    if not anchors or any(value < minimum_primary_count for value in anchors):
        raise ExdHoxSplitError(
            "Absolute anchors must be unique counts at or above the minimum."
        )
    tolerance = Decimal(str(absolute_alias_tolerance))
    if tolerance < 0:
        raise ExdHoxSplitError("Alias tolerance must be nonnegative.")

    rows_by_tf = defaultdict(list)
    for row in ordering_rows:
        rows_by_tf[str(row["transcription_factor"])].append(row)
    result = []
    for transcription_factor in sorted(rows_by_tf):
        tf_rows = rows_by_tf[transcription_factor]
        tf_rows.sort(
            key=lambda row: (
                int(row["rank_one_based"]),
                str(row["logical_example_id"]),
            )
        )
        total_logical_examples = len(tf_rows)
        logical_count_by_rank = Counter(
            int(row["rank_one_based"]) for row in tf_rows
        )
        ordered_ranks = sorted(logical_count_by_rank)
        if ordered_ranks != list(range(1, len(ordered_ranks) + 1)):
            raise ExdHoxSplitError("Subset ordering ranks must be contiguous.")
        cumulative_count_by_rank = {}
        cumulative_count = 0
        for rank in ordered_ranks:
            cumulative_count += logical_count_by_rank[rank]
            cumulative_count_by_rank[rank] = cumulative_count

        requests = []
        for anchor in anchors:
            if anchor <= total_logical_examples:
                requests.append(
                    {
                        "request_type": "absolute",
                        "request_value": str(anchor),
                        "unaliased_count": anchor,
                        "alias_anchor": None,
                        "canonical_count": anchor,
                    }
                )
        for fraction_value in fractional_levels:
            fraction = Decimal(str(fraction_value))
            if fraction <= 0 or fraction > 1:
                raise ExdHoxSplitError(
                    "Fractional subset levels must lie in (0, 1]."
                )
            requested_decimal = Decimal(total_logical_examples) * fraction
            requested_count = int(
                requested_decimal.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            )
            if requested_count < minimum_primary_count:
                continue
            requested_count = min(requested_count, total_logical_examples)
            eligible_anchors = []
            for anchor in anchors:
                if anchor > total_logical_examples:
                    continue
                distance = abs(requested_count - anchor)
                threshold = Decimal(anchor) * tolerance
                if Decimal(distance) <= threshold:
                    eligible_anchors.append((distance, anchor))
            alias_anchor = None
            canonical_count = requested_count
            if eligible_anchors:
                eligible_anchors.sort()
                alias_anchor = eligible_anchors[0][1]
                canonical_count = alias_anchor
            requests.append(
                {
                    "request_type": "fractional",
                    "request_value": _decimal_text(fraction),
                    "unaliased_count": requested_count,
                    "alias_anchor": alias_anchor,
                    "canonical_count": canonical_count,
                }
            )

        request_type_order = {"absolute": 0, "fractional": 1}
        requests.sort(
            key=lambda request: (
                request["canonical_count"],
                request_type_order[request["request_type"]],
                request["request_value"],
            )
        )
        for request in requests:
            canonical_count = request["canonical_count"]
            inclusive_rank = None
            actual_count = None
            for rank in ordered_ranks:
                if cumulative_count_by_rank[rank] >= canonical_count:
                    inclusive_rank = rank
                    actual_count = cumulative_count_by_rank[rank]
                    break
            if inclusive_rank is None or actual_count is None:
                raise ExdHoxSplitError("Subset prefix cannot meet a request.")
            level_id = _stable_identifier(
                "lvl",
                "exd_hox_nested_subset_level.v1",
                transcription_factor,
                str(canonical_count),
                request["request_type"],
                request["request_value"],
            )
            result.append(
                {
                    "transcription_factor": transcription_factor,
                    "level_id": level_id,
                    "request_type": request["request_type"],
                    "request_value": request["request_value"],
                    "unaliased_requested_logical_example_count": request[
                        "unaliased_count"
                    ],
                    "alias_absolute_anchor": (
                        ""
                        if request["alias_anchor"] is None
                        else request["alias_anchor"]
                    ),
                    "canonical_requested_logical_example_count": canonical_count,
                    "actual_logical_example_count": actual_count,
                    "actual_rc_group_count": inclusive_rank,
                    "inclusive_maximum_rank": inclusive_rank,
                }
            )
    return tuple(result)


def subset_membership_ids(
    ordering_rows: Sequence[Mapping[str, Any]],
    level_row: Mapping[str, Any],
) -> Tuple[str, ...]:
    """Return the exact logical-example prefix represented by one level."""

    transcription_factor = str(level_row["transcription_factor"])
    maximum_rank = int(level_row["inclusive_maximum_rank"])
    selected = []
    for row in ordering_rows:
        if str(row["transcription_factor"]) != transcription_factor:
            continue
        if int(row["rank_one_based"]) <= maximum_rank:
            selected.append(
                (
                    int(row["rank_one_based"]),
                    str(row["logical_example_id"]),
                )
            )
    selected.sort()
    return tuple(logical_example_id for rank, logical_example_id in selected)


def rc_orientation_rows(
    logical_example: LogicalExample,
) -> Tuple[Dict[str, str], ...]:
    """Return inference orientations without creating new labeled examples."""

    reverse_sequence = reverse_complement(logical_example.sequence)
    sequences = [logical_example.sequence]
    if reverse_sequence != logical_example.sequence:
        sequences.append(reverse_sequence)
    rows = []
    for orientation_index, sequence in enumerate(sequences):
        rows.append(
            {
                "logical_example_id": logical_example.logical_example_id,
                "global_rc_group_id": logical_example.global_rc_group_id,
                "orientation_index": str(orientation_index),
                "sequence": sequence,
            }
        )
    return tuple(rows)


def _decimal_text(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." not in text:
        return "{0}.0".format(text)
    return text


def _repair_approved_tf_margins(
    global_groups: Sequence[GlobalRcGroup],
    contributions_by_group: Mapping[str, Counter],
    assignments: Mapping[str, str],
    order_hashes: Mapping[str, str],
    required_per_tf_split_counts: Mapping[str, Mapping[str, int]],
    seed: int,
    policy_identifier: str,
) -> Dict[str, str]:
    """Deterministically exchange group signatures to meet approved margins."""

    transcription_factors = tuple(sorted(required_per_tf_split_counts))
    tf_index = {
        transcription_factor: index
        for index, transcription_factor in enumerate(transcription_factors)
    }
    required = {}
    for transcription_factor in transcription_factors:
        split_counts = required_per_tf_split_counts[transcription_factor]
        if set(split_counts) != set(SPLIT_NAMES):
            raise ExdHoxSplitError(
                "Approved per-TF counts must cover every primary split."
            )
        for split in SPLIT_NAMES:
            required[(transcription_factor, split)] = int(split_counts[split])

    buckets = {
        split: defaultdict(list) for split in SPLIT_NAMES
    }
    current = Counter()
    for group in global_groups:
        group_id = group.global_rc_group_id
        signature_values = [0] * len(transcription_factors)
        for (transcription_factor, affinity_bin), count in (
            contributions_by_group[group_id].items()
        ):
            del affinity_bin
            signature_values[tf_index[transcription_factor]] += count
        signature = tuple(signature_values)
        split = assignments[group_id]
        buckets[split][signature].append(group_id)
        for index, transcription_factor in enumerate(transcription_factors):
            current[(transcription_factor, split)] += signature[index]

    for split in SPLIT_NAMES:
        for signature in buckets[split]:
            buckets[split][signature].sort(
                key=lambda group_id: (
                    order_hashes[group_id],
                    group_id,
                )
            )

    total_by_tf = Counter()
    required_total_by_tf = Counter()
    for transcription_factor in transcription_factors:
        for split in SPLIT_NAMES:
            total_by_tf[transcription_factor] += current[
                (transcription_factor, split)
            ]
            required_total_by_tf[transcription_factor] += required[
                (transcription_factor, split)
            ]
    if total_by_tf != required_total_by_tf:
        raise ExdHoxSplitError(
            "Approved per-TF counts do not conserve logical examples."
        )

    repaired_assignments = dict(assignments)
    maximum_iterations = 1024
    iteration = 0
    current_objective = _tf_margin_objective(current, required)
    while current_objective > 0 and iteration < maximum_iterations:
        best_move = None
        for left_index, left_split in enumerate(SPLIT_NAMES):
            for right_split in SPLIT_NAMES[left_index + 1 :]:
                left_signatures = sorted(buckets[left_split])
                right_signatures = sorted(buckets[right_split])
                for left_signature in left_signatures:
                    left_ids = buckets[left_split][left_signature]
                    if not left_ids:
                        continue
                    for right_signature in right_signatures:
                        right_ids = buckets[right_split][right_signature]
                        if not right_ids or left_signature == right_signature:
                            continue
                        delta = tuple(
                            right_signature[index] - left_signature[index]
                            for index in range(len(transcription_factors))
                        )
                        maximum_move_count = min(
                            len(left_ids),
                            len(right_ids),
                        )
                        candidate_counts = _margin_move_candidate_counts(
                            current=current,
                            required=required,
                            transcription_factors=transcription_factors,
                            left_split=left_split,
                            right_split=right_split,
                            delta=delta,
                            maximum_count=maximum_move_count,
                        )
                        old_pair_objective = _tf_margin_pair_objective(
                            current,
                            required,
                            transcription_factors,
                            left_split,
                            right_split,
                        )
                        for move_count in candidate_counts:
                            new_pair_objective = 0
                            for index, transcription_factor in enumerate(
                                transcription_factors
                            ):
                                left_value = (
                                    current[(transcription_factor, left_split)]
                                    + move_count * delta[index]
                                )
                                right_value = (
                                    current[(transcription_factor, right_split)]
                                    - move_count * delta[index]
                                )
                                new_pair_objective += abs(
                                    left_value
                                    - required[(transcription_factor, left_split)]
                                )
                                new_pair_objective += abs(
                                    right_value
                                    - required[(transcription_factor, right_split)]
                                )
                            improvement = (
                                old_pair_objective - new_pair_objective
                            )
                            if improvement <= 0:
                                continue
                            tie_hash = _domain_hash(
                                "exd_hox_approved_margin_exchange_tie.v1",
                                policy_identifier,
                                str(seed),
                                str(iteration),
                                left_split,
                                right_split,
                                ",".join(str(value) for value in left_signature),
                                ",".join(str(value) for value in right_signature),
                                str(move_count),
                            )
                            key = (
                                -improvement,
                                tie_hash,
                                left_split,
                                right_split,
                                left_signature,
                                right_signature,
                                move_count,
                            )
                            if best_move is None or key < best_move[0]:
                                best_move = (
                                    key,
                                    left_split,
                                    right_split,
                                    left_signature,
                                    right_signature,
                                    move_count,
                                    delta,
                                )
        if best_move is None:
            raise ExdHoxSplitError(
                "Deterministic group exchanges cannot meet approved TF margins."
            )
        (
            move_key,
            left_split,
            right_split,
            left_signature,
            right_signature,
            move_count,
            delta,
        ) = best_move
        del move_key
        left_ids = buckets[left_split][left_signature]
        right_ids = buckets[right_split][right_signature]
        moved_left = left_ids[-move_count:]
        moved_right = right_ids[-move_count:]
        del left_ids[-move_count:]
        del right_ids[-move_count:]
        buckets[left_split][right_signature].extend(moved_right)
        buckets[right_split][left_signature].extend(moved_left)
        for group_id in moved_left:
            repaired_assignments[group_id] = right_split
        for group_id in moved_right:
            repaired_assignments[group_id] = left_split
        for index, transcription_factor in enumerate(transcription_factors):
            current[(transcription_factor, left_split)] += (
                move_count * delta[index]
            )
            current[(transcription_factor, right_split)] -= (
                move_count * delta[index]
            )
        new_objective = _tf_margin_objective(current, required)
        if new_objective >= current_objective:
            raise ExdHoxSplitError(
                "Approved-margin exchange failed to reduce its objective."
            )
        current_objective = new_objective
        iteration += 1

    if current_objective != 0:
        raise ExdHoxSplitError(
            "Approved per-TF margins were not reached within the iteration bound."
        )
    return repaired_assignments


def _margin_move_candidate_counts(
    current: Mapping[Tuple[str, str], int],
    required: Mapping[Tuple[str, str], int],
    transcription_factors: Sequence[str],
    left_split: str,
    right_split: str,
    delta: Sequence[int],
    maximum_count: int,
) -> Tuple[int, ...]:
    candidates = {1, maximum_count}
    for index, transcription_factor in enumerate(transcription_factors):
        if delta[index] == 0:
            continue
        for split, sign in ((left_split, 1), (right_split, -1)):
            error = current[(transcription_factor, split)] - required[
                (transcription_factor, split)
            ]
            quotient = abs(error) // abs(delta[index])
            for candidate in (quotient, quotient + 1):
                if 1 <= candidate <= maximum_count:
                    candidates.add(candidate)
    return tuple(sorted(candidates))


def _tf_margin_objective(
    current: Mapping[Tuple[str, str], int],
    required: Mapping[Tuple[str, str], int],
) -> int:
    return sum(abs(current[key] - required[key]) for key in required)


def _tf_margin_pair_objective(
    current: Mapping[Tuple[str, str], int],
    required: Mapping[Tuple[str, str], int],
    transcription_factors: Sequence[str],
    left_split: str,
    right_split: str,
) -> int:
    value = 0
    for transcription_factor in transcription_factors:
        value += abs(
            current[(transcription_factor, left_split)]
            - required[(transcription_factor, left_split)]
        )
        value += abs(
            current[(transcription_factor, right_split)]
            - required[(transcription_factor, right_split)]
        )
    return value


def _merged_affinity_bin_mapping(
    examples: Sequence[LogicalExample],
    raw_bin_by_example: Mapping[str, int],
    minimum_distinct_groups_per_bin: int,
) -> Dict[int, int]:
    groups_by_raw_bin = defaultdict(set)
    for example in examples:
        raw_bin = raw_bin_by_example[example.logical_example_id]
        groups_by_raw_bin[raw_bin].add(example.global_rc_group_id)
    all_group_ids = set()
    for group_ids in groups_by_raw_bin.values():
        all_group_ids.update(group_ids)
    if len(all_group_ids) < minimum_distinct_groups_per_bin:
        raise ExdHoxSplitError(
            "A TF has too few distinct RC groups for train/validation/test."
        )

    clusters = []
    for raw_bin in sorted(groups_by_raw_bin):
        clusters.append(
            {
                "raw_bins": [raw_bin],
                "group_ids": set(groups_by_raw_bin[raw_bin]),
            }
        )
    merging_required = True
    while merging_required and len(clusters) > 1:
        merging_required = False
        undersized_index = None
        for index, cluster in enumerate(clusters):
            if len(cluster["group_ids"]) < minimum_distinct_groups_per_bin:
                undersized_index = index
                break
        if undersized_index is not None:
            merging_required = True
            if undersized_index == 0:
                neighbor_index = 1
            elif undersized_index == len(clusters) - 1:
                neighbor_index = undersized_index - 1
            else:
                current_min = min(clusters[undersized_index]["raw_bins"])
                current_max = max(clusters[undersized_index]["raw_bins"])
                left_distance = current_min - max(
                    clusters[undersized_index - 1]["raw_bins"]
                )
                right_distance = min(
                    clusters[undersized_index + 1]["raw_bins"]
                ) - current_max
                if left_distance <= right_distance:
                    neighbor_index = undersized_index - 1
                else:
                    neighbor_index = undersized_index + 1
            left_index = min(undersized_index, neighbor_index)
            right_index = max(undersized_index, neighbor_index)
            merged = {
                "raw_bins": sorted(
                    clusters[left_index]["raw_bins"]
                    + clusters[right_index]["raw_bins"]
                ),
                "group_ids": (
                    clusters[left_index]["group_ids"]
                    | clusters[right_index]["group_ids"]
                ),
            }
            clusters[left_index : right_index + 1] = [merged]

    mapping = {}
    for cluster in clusters:
        merged_identifier = min(cluster["raw_bins"])
        for raw_bin in cluster["raw_bins"]:
            mapping[raw_bin] = merged_identifier
    return mapping


def _conflicting_keys(values_by_key: Mapping[Any, set]) -> list:
    conflicts = []
    for key, values in values_by_key.items():
        if len(values) > 1:
            conflicts.append(key)
    return sorted(conflicts)


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _domain_hash(domain: str, *parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()


def _stable_identifier(prefix: str, domain: str, *parts: str) -> str:
    return "{0}_{1}".format(prefix, _domain_hash(domain, *parts))


def load_split_config(path: Path | str) -> Dict[str, Any]:
    """Load and validate the versioned primary split configuration."""

    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    if not isinstance(payload, Mapping):
        raise ExdHoxSplitError("Primary split config must be a mapping.")
    config = dict(payload)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ExdHoxSplitError("Unsupported Exd-Hox primary split config schema.")
    required_sections = (
        "study",
        "inputs",
        "dataset",
        "split_policy",
        "subset_policy",
        "outputs",
        "test_access",
        "expected",
    )
    for section in required_sections:
        if not isinstance(config.get(section), Mapping):
            raise ExdHoxSplitError(
                "Config section {0} must be a mapping.".format(section)
            )
    for key in (
        "source_manifest_path",
        "audit_manifest_path",
    ):
        validate_repository_relative_path(str(config["inputs"][key]))
    for key in (
        "split_directory",
        "subset_directory",
        "sealed_target_directory",
        "plot_directory",
        "test_access_policy_path",
    ):
        validate_repository_relative_path(str(config["outputs"][key]))
    fixed_output_paths = {
        "split_directory": "data/processed/exd_hox_primary_split_v1",
        "subset_directory": "data/processed/exd_hox_nested_subsets_v1",
        "sealed_target_directory": (
            "data/sealed/exd_hox_primary_test_targets_v1"
        ),
        "plot_directory": "plots/exd_hox_primary_split_v1",
        "test_access_policy_path": "configs/exd_hox_test_access_policy_v1.yaml",
    }
    for key, value in fixed_output_paths.items():
        if config["outputs"].get(key) != value:
            raise ExdHoxSplitError("Primary v1 outputs.{0} differs.".format(key))
    for section, keys in (
        (
            config["inputs"],
            (
                "source_manifest_hash",
                "source_manifest_file_sha256",
                "audit_manifest_hash",
                "audit_manifest_file_sha256",
            ),
        ),
        (
            config["test_access"],
            ("policy_manifest_hash", "policy_file_sha256"),
        ),
    ):
        for key in keys:
            _validate_sha256_text(str(section[key]), key)
    transcription_factors = tuple(
        str(value) for value in config["dataset"]["transcription_factors"]
    )
    if not transcription_factors or len(set(transcription_factors)) != len(
        transcription_factors
    ):
        raise ExdHoxSplitError("Config TF names must be nonempty and unique.")
    if int(config["dataset"]["sequence_length"]) <= 0:
        raise ExdHoxSplitError("Configured sequence length must be positive.")
    policy = config["split_policy"]
    if policy.get("identifier") != SPLIT_POLICY_IDENTIFIER:
        raise ExdHoxSplitError("Unsupported primary split policy identifier.")
    if tuple(policy.get("split_order", ())) != SPLIT_NAMES:
        raise ExdHoxSplitError("Primary split order must be fixed v1 order.")
    proportions = policy.get("proportions")
    if not isinstance(proportions, Mapping) or tuple(proportions) != SPLIT_NAMES:
        raise ExdHoxSplitError("Primary split proportions use an invalid order.")
    normalized_proportions = {
        split: Fraction(str(proportions[split])) for split in SPLIT_NAMES
    }
    if normalized_proportions != {
        "training": Fraction(4, 5),
        "validation": Fraction(1, 10),
        "test": Fraction(1, 10),
    }:
        raise ExdHoxSplitError("Primary v1 must use exact 80/10/10 proportions.")
    largest_remainder_counts(
        1,
        proportions,
        SPLIT_NAMES,
        (SPLIT_POLICY_IDENTIFIER, "config_validation"),
    )
    if int(policy.get("seed", -1)) != 31001:
        raise ExdHoxSplitError("Primary v1 split seed must be 31001.")
    if int(policy.get("affinity_bin_count", 0)) != 10:
        raise ExdHoxSplitError("Primary v1 must use ten affinity bins.")
    if int(policy.get("minimum_distinct_groups_per_bin", 0)) != 7:
        raise ExdHoxSplitError(
            "Primary v1 affinity strata need seven groups to support all splits."
        )
    fixed_policy_values = {
        "equal_distance_bin_merge": "lower",
        "quota_method": "deterministic_largest_remainder",
        "group_order": "seeded_sha256",
        "assignment_objective": (
            "lexicographic_stratum_deficit_then_approved_margin_signature_exchange"
        ),
    }
    for key, value in fixed_policy_values.items():
        if policy.get(key) != value:
            raise ExdHoxSplitError("Primary v1 {0} differs.".format(key))
    subset_policy = config["subset_policy"]
    if int(subset_policy.get("seed", -1)) != 32001:
        raise ExdHoxSplitError("Primary v1 subset seed must be 32001.")
    anchors = tuple(int(value) for value in subset_policy["absolute_counts"])
    if anchors != (128, 256, 512):
        raise ExdHoxSplitError("Primary v1 absolute subset anchors differ.")
    fractions = tuple(
        Decimal(str(value)) for value in subset_policy["fractional_levels"]
    )
    if fractions != tuple(
        Decimal(value)
        for value in ("0.01", "0.02", "0.05", "0.10", "0.25", "0.50", "1.00")
    ):
        raise ExdHoxSplitError("Primary v1 fractional subset levels differ.")
    if Decimal(str(subset_policy.get("absolute_alias_tolerance"))) != Decimal(
        "0.05"
    ):
        raise ExdHoxSplitError("Primary v1 subset alias tolerance differs.")
    if subset_policy.get("fractional_rounding") != "round_half_up":
        raise ExdHoxSplitError("Primary v1 subset rounding rule differs.")
    if int(subset_policy.get("minimum_primary_count", 0)) != 128:
        raise ExdHoxSplitError("Primary v1 minimum subset count differs.")
    downstream_seeds = tuple(
        int(value)
        for value in subset_policy["downstream_initialization_seeds"]
    )
    if downstream_seeds != (33001, 33002, 33003, 33004, 33005):
        raise ExdHoxSplitError("Reserved downstream seeds differ from v1.")
    return config


def load_validated_source_records(
    config: Mapping[str, Any],
    repository_root: Path | str,
) -> Dict[str, Dict[str, SelexHdf5File]]:
    """Revalidate both 3C manifests and all HDF5 bytes before reading rows."""

    root = Path(repository_root).resolve()
    inputs = config["inputs"]
    source_path = _resolve_repository_path(
        root,
        str(inputs["source_manifest_path"]),
    )
    audit_path = _resolve_repository_path(
        root,
        str(inputs["audit_manifest_path"]),
    )
    _require_file_identity(
        source_path,
        None,
        str(inputs["source_manifest_file_sha256"]),
        "Milestone 3C source manifest",
    )
    _require_file_identity(
        audit_path,
        None,
        str(inputs["audit_manifest_file_sha256"]),
        "Milestone 3C audit manifest",
    )
    source_manifest = _load_json_mapping(source_path, "source manifest")
    audit_manifest = _load_json_mapping(audit_path, "audit manifest")
    validate_hashed_manifest(source_manifest)
    validate_hashed_manifest(audit_manifest)
    if source_manifest.get("schema_version") != "exd_hox_source_manifest.v1":
        raise ExdHoxSplitError("Unexpected Milestone 3C source schema.")
    if audit_manifest.get("schema_version") != "exd_hox_audit_manifest.v1":
        raise ExdHoxSplitError("Unexpected Milestone 3C audit schema.")
    if source_manifest["manifest_hash"] != inputs["source_manifest_hash"]:
        raise ExdHoxSplitError("Milestone 3C source-manifest identity differs.")
    if audit_manifest["manifest_hash"] != inputs["audit_manifest_hash"]:
        raise ExdHoxSplitError("Milestone 3C audit-manifest identity differs.")
    if audit_manifest["source_manifest_hash"] != source_manifest["manifest_hash"]:
        raise ExdHoxSplitError("Audit manifest binds a different source manifest.")
    if source_manifest["dataset_identifier"] != config["study"][
        "dataset_identifier"
    ]:
        raise ExdHoxSplitError("Source dataset identifier differs from config.")
    if audit_manifest.get("dataset_identifier") != config["study"][
        "dataset_identifier"
    ]:
        raise ExdHoxSplitError("Audit dataset identifier differs from config.")
    if source_manifest.get("source_commit") != config["study"][
        "external_source_commit"
    ]:
        raise ExdHoxSplitError("Pinned external source commit differs from config.")

    for artifact in audit_manifest["artifacts"]:
        artifact_path = _resolve_repository_path(root, str(artifact["path"]))
        _require_file_identity(
            artifact_path,
            int(artifact["byte_size"]),
            str(artifact["sha256"]),
            "Milestone 3C audit artifact",
        )

    transcription_factors = tuple(config["dataset"]["transcription_factors"])
    expected_pairs = {
        (transcription_factor, supplied_split)
        for transcription_factor in transcription_factors
        for supplied_split in SUPPLIED_SPLIT_NAMES
    }
    observed_pairs = set()
    records = {}
    for item in source_manifest["files"]:
        transcription_factor = str(item["transcription_factor"])
        supplied_split = str(item["supplied_split"])
        pair = (transcription_factor, supplied_split)
        if pair in observed_pairs:
            raise ExdHoxSplitError("Source manifest repeats an HDF5 identity.")
        observed_pairs.add(pair)
        raw_path = _resolve_repository_path(
            root,
            str(item["imported_raw_path"]),
        )
        if raw_path.is_symlink():
            raise ExdHoxSplitError("Imported canonical HDF5 files must not be symlinks.")
        _require_file_identity(
            raw_path,
            int(item["byte_size"]),
            str(item["sha256"]),
            "Imported canonical HDF5",
        )
        record = read_validate_selex_hdf5(
            raw_path,
            transcription_factor=transcription_factor,
            supplied_split=supplied_split,
            logical_path=str(item["imported_raw_path"]),
            sequence_length=int(config["dataset"]["sequence_length"]),
        )
        records.setdefault(transcription_factor, {})[supplied_split] = record
    if observed_pairs != expected_pairs:
        raise ExdHoxSplitError("Source manifest does not contain the exact TF file set.")
    return records


def _load_json_mapping(path: Path, description: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise ExdHoxSplitError("{0} must be a mapping.".format(description))
    return dict(payload)


def _require_file_identity(
    path: Path,
    expected_size: Optional[int],
    expected_sha256: str,
    description: str,
) -> None:
    if not path.is_file():
        raise ExdHoxSplitError("{0} is missing: {1}".format(description, path))
    observed_size = path.stat().st_size
    observed_sha256 = hash_file_bytes(path)
    size_differs = expected_size is not None and observed_size != expected_size
    if size_differs or observed_sha256 != expected_sha256:
        raise ExdHoxSplitError("{0} byte identity differs.".format(description))


def _validate_sha256_text(value: str, field_name: str) -> None:
    if len(value) != 64:
        raise ExdHoxSplitError("{0} must be a full SHA-256.".format(field_name))
    try:
        int(value, 16)
    except ValueError as error:
        raise ExdHoxSplitError(
            "{0} must be lowercase hexadecimal.".format(field_name)
        ) from error
    if value != value.lower():
        raise ExdHoxSplitError(
            "{0} must be lowercase hexadecimal.".format(field_name)
        )


def _resolve_repository_path(root: Path, logical_path: str) -> Path:
    normalized = validate_repository_relative_path(logical_path)
    candidate = root / normalized
    current = root
    for component in Path(normalized).parts:
        current = current / component
        if current.is_symlink():
            raise ExdHoxSplitError(
                "Repository paths must not contain symlink components."
            )
    resolved = candidate.resolve()
    repository_relative_path(resolved, root)
    return resolved


def _resolve_config_path(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    repository_relative_path(resolved, root)
    return resolved


def _require_disjoint_directories(
    first: Path,
    second: Path,
    description: str,
) -> None:
    if first == second or first in second.parents or second in first.parents:
        raise ExdHoxSplitError(
            "{0} directories must be disjoint.".format(description)
        )


def _validate_test_access_policy_identity(
    config: Mapping[str, Any],
    repository_root: Path,
) -> Dict[str, Any]:
    """Validate the public policy bytes and hashed logical identity."""

    policy_path = _resolve_repository_path(
        repository_root,
        str(config["outputs"]["test_access_policy_path"]),
    )
    _require_file_identity(
        policy_path,
        None,
        str(config["test_access"]["policy_file_sha256"]),
        "Test-access policy",
    )
    try:
        with open(policy_path, "r", encoding="utf-8") as input_file:
            payload = yaml.safe_load(input_file)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ExdHoxSplitError("Test-access policy cannot be read.") from error
    if not isinstance(payload, Mapping):
        raise ExdHoxSplitError("Test-access policy must be a mapping.")
    policy = dict(payload)
    try:
        validate_hashed_manifest(policy)
    except (KeyError, TypeError, ValueError) as error:
        raise ExdHoxSplitError("Test-access policy hash mismatch.") from error
    if policy.get("schema_version") != "exd_hox_test_access_policy.v1":
        raise ExdHoxSplitError("Unsupported test-access policy schema.")
    if policy.get("manifest_hash") != config["test_access"][
        "policy_manifest_hash"
    ]:
        raise ExdHoxSplitError("Configured test-access policy identity differs.")
    try:
        validate_test_access_policy(policy)
    except ValueError as error:
        raise ExdHoxSplitError("Test-access policy semantics differ.") from error
    if policy["sealed_target_directory"] != config["outputs"][
        "sealed_target_directory"
    ]:
        raise ExdHoxSplitError("Test-access policy sealed directory differs.")
    if policy["sealed_target_manifest_directory"] != config["outputs"][
        "split_directory"
    ]:
        raise ExdHoxSplitError("Test-access policy split directory differs.")
    return policy


def build_primary_split_artifacts(
    config_path: Path | str,
    repository_root: Path | str,
    output_directory: Optional[Path | str] = None,
    sealed_target_directory: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Build immutable public split artifacts and separately sealed targets."""

    if (output_directory is None) != (sealed_target_directory is None):
        raise ExdHoxSplitError(
            "Primary physical output overrides must be supplied together."
        )
    root = Path(repository_root).resolve()
    config_file = _resolve_config_path(root, config_path)
    config_logical_path = repository_relative_path(config_file, root)
    config = load_split_config(config_file)
    test_access_policy = _validate_test_access_policy_identity(config, root)
    outputs = config["outputs"]
    split_logical_directory = str(outputs["split_directory"])
    sealed_logical_directory = str(outputs["sealed_target_directory"])
    if output_directory is None:
        split_directory = _resolve_repository_path(
            root,
            split_logical_directory,
        )
        sealed_directory = _resolve_repository_path(
            root,
            sealed_logical_directory,
        )
    else:
        split_directory = Path(output_directory).resolve()
        sealed_directory = Path(sealed_target_directory).resolve()
    _require_disjoint_directories(
        split_directory,
        sealed_directory,
        "Public split and sealed target",
    )
    _require_new_directory(split_directory, "primary split")
    _require_new_directory(sealed_directory, "sealed target")

    records_by_tf = load_validated_source_records(config, root)
    policy = config["split_policy"]
    result = build_primary_split(
        records_by_tf=records_by_tf,
        proportions=policy["proportions"],
        seed=int(policy["seed"]),
        affinity_bin_count=int(policy["affinity_bin_count"]),
        minimum_distinct_groups_per_bin=int(
            policy["minimum_distinct_groups_per_bin"]
        ),
        policy_identifier=str(policy["identifier"]),
        required_per_tf_split_counts=config["expected"][
            "per_tf_split_counts"
        ],
    )
    leakage_rows = audit_primary_split_leakage(
        result.reconciliation.logical_examples,
        result.assignments,
    )
    _validate_expected_primary_result(
        config,
        result,
        leakage_rows,
    )

    split_directory.parent.mkdir(parents=True, exist_ok=True)
    sealed_directory.parent.mkdir(parents=True, exist_ok=True)
    (
        split_staging_context,
        sealed_staging_context,
        split_staging,
        sealed_staging,
    ) = _prepare_primary_staging(split_directory, sealed_directory)
    split_was_published = False
    sealed_target_was_published = False
    try:
        example_by_id = {
            example.logical_example_id: example
            for example in result.reconciliation.logical_examples
        }
        write_tsv_gzip_exclusive(
            split_staging / LOGICAL_EXAMPLES_FILENAME,
            LOGICAL_EXAMPLE_FIELDS,
            _logical_example_rows(result),
        )
        write_tsv_gzip_exclusive(
            split_staging / PROVENANCE_FILENAME,
            PROVENANCE_FIELDS,
            _provenance_rows(result, example_by_id),
        )
        write_tsv_gzip_exclusive(
            split_staging / GLOBAL_GROUPS_FILENAME,
            GLOBAL_GROUP_FIELDS,
            _global_group_rows(result),
        )
        write_tsv_gzip_exclusive(
            split_staging / ASSIGNMENTS_FILENAME,
            ASSIGNMENT_FIELDS,
            _assignment_rows(result),
        )
        write_tsv_exclusive(
            split_staging / RECONCILIATION_AUDIT_FILENAME,
            RECONCILIATION_AUDIT_FIELDS,
            result.reconciliation.audit_rows,
        )
        count_rows = _build_split_count_summary_rows(
            records_by_tf,
            result,
        )
        write_tsv_exclusive(
            split_staging / COUNT_SUMMARY_FILENAME,
            COUNT_SUMMARY_FIELDS,
            count_rows,
        )
        supplied_leakage = _build_supplied_leakage_row(records_by_tf)
        all_leakage_rows = tuple(leakage_rows) + (supplied_leakage,)
        write_tsv_exclusive(
            split_staging / LEAKAGE_AUDIT_FILENAME,
            LEAKAGE_AUDIT_FIELDS,
            all_leakage_rows,
        )
        affinity_histogram_rows = _build_public_affinity_histogram_rows(result)
        write_tsv_exclusive(
            split_staging / AFFINITY_HISTOGRAM_FILENAME,
            AFFINITY_HISTOGRAM_FIELDS,
            affinity_histogram_rows,
        )
        write_tsv_gzip_exclusive(
            split_staging / PUBLIC_TEST_INPUTS_FILENAME,
            PUBLIC_TEST_INPUT_FIELDS,
            _public_test_input_rows(result),
        )

        assignment_fingerprint = fingerprint_file(
            split_staging / ASSIGNMENTS_FILENAME,
            Path(split_logical_directory, ASSIGNMENTS_FILENAME).as_posix(),
        ).to_dict()
        split_identity_hash = hash_logical_content(
            {
                "schema_version": "exd_hox_primary_split_identity.v1",
                "dataset_identifier": config["study"]["dataset_identifier"],
                "source_manifest_hash": config["inputs"][
                    "source_manifest_hash"
                ],
                "policy": dict(policy),
                "assignment_artifact": assignment_fingerprint,
            }
        )

        sealed_rows = tuple(_sealed_target_rows(result))
        write_tsv_gzip_exclusive(
            sealed_staging / SEALED_TARGET_FILENAME,
            SEALED_TARGET_FIELDS,
            sealed_rows,
        )
        sealed_fingerprint = fingerprint_file(
            sealed_staging / SEALED_TARGET_FILENAME,
            Path(sealed_logical_directory, SEALED_TARGET_FILENAME).as_posix(),
        ).to_dict()
        commitment_digest = _target_commitment_digest(sealed_rows)
        sealed_manifest = build_hashed_manifest(
            SEALED_TARGET_MANIFEST_SCHEMA_VERSION,
            {
                "sealed_target_path": sealed_fingerprint["path"],
                "sealed_target_byte_size": sealed_fingerprint["byte_size"],
                "sealed_target_sha256": sealed_fingerprint["sha256"],
                "test_logical_example_count": len(sealed_rows),
                "target_commitment_digest_sha256": commitment_digest,
                "split_identity_hash": split_identity_hash,
                "split_policy_identifier": policy["identifier"],
            },
        )
        write_json_exclusive(
            split_staging / SEALED_TARGET_MANIFEST_FILENAME,
            sealed_manifest,
        )

        artifact_filenames = (
            LOGICAL_EXAMPLES_FILENAME,
            PROVENANCE_FILENAME,
            GLOBAL_GROUPS_FILENAME,
            ASSIGNMENTS_FILENAME,
            RECONCILIATION_AUDIT_FILENAME,
            COUNT_SUMMARY_FILENAME,
            LEAKAGE_AUDIT_FILENAME,
            AFFINITY_HISTOGRAM_FILENAME,
            PUBLIC_TEST_INPUTS_FILENAME,
            SEALED_TARGET_MANIFEST_FILENAME,
        )
        artifacts = _fingerprint_staged_files(
            split_staging,
            split_logical_directory,
            artifact_filenames,
        )
        manifest = build_hashed_manifest(
            SPLIT_MANIFEST_SCHEMA_VERSION,
            {
                "study_identifier": config["study"]["identifier"],
                "dataset_identifier": config["study"]["dataset_identifier"],
                "config_path": config_logical_path,
                "config_sha256": hash_file_bytes(config_file),
                "source_manifest_hash": config["inputs"][
                    "source_manifest_hash"
                ],
                "audit_manifest_hash": config["inputs"][
                    "audit_manifest_hash"
                ],
                "split_identity_hash": split_identity_hash,
                "split_directory": split_logical_directory,
                "policy": dict(policy),
                "excluded_identity_axes": [
                    "model_family",
                    "tokenizer",
                    "model_seed",
                    "pretrained_checkpoint",
                    "physical_features",
                    "model_output",
                    "downstream_performance",
                ],
                "artifacts": artifacts,
                "sealed_target_manifest_hash": sealed_manifest["manifest_hash"],
                "test_access_policy_manifest_hash": test_access_policy[
                    "manifest_hash"
                ],
                "counts": _primary_manifest_counts(result),
                "test_target_handling": (
                    "Plaintext test targets exist only in the separately "
                    "ignored sealed-target artifact."
                ),
            },
        )
        write_json_exclusive(
            split_staging / SPLIT_MANIFEST_FILENAME,
            manifest,
        )

        _require_new_directory(split_directory, "primary split")
        if not sealed_directory.is_dir() or sealed_directory.is_symlink():
            raise ExdHoxSplitError("Reserved sealed-target directory changed.")
        final_sealed_target = sealed_directory / SEALED_TARGET_FILENAME
        if final_sealed_target.exists() or final_sealed_target.is_symlink():
            raise FileExistsError("Refusing to overwrite a sealed target.")
        try:
            os.rename(
                sealed_staging / SEALED_TARGET_FILENAME,
                final_sealed_target,
            )
            sealed_target_was_published = True
            os.rename(split_staging, split_directory)
            split_was_published = True
        except OSError:
            if sealed_target_was_published and not split_was_published:
                os.rename(
                    final_sealed_target,
                    sealed_staging / SEALED_TARGET_FILENAME,
                )
                sealed_target_was_published = False
            raise
    finally:
        sealed_staging_context.cleanup()
        split_staging_context.cleanup()
        if not split_was_published:
            sealed_directory.rmdir()
    return manifest


def _validate_expected_primary_result(
    config: Mapping[str, Any],
    result: PrimarySplitResult,
    leakage_rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = config["expected"]
    reconciliation = result.reconciliation
    _require_expected_value(
        len(reconciliation.source_occurrences),
        expected["source_occurrences"],
        "source occurrences",
    )
    _require_expected_value(
        len(reconciliation.logical_examples),
        expected["logical_examples"],
        "logical examples",
    )
    collapsed = len(reconciliation.source_occurrences) - len(
        reconciliation.logical_examples
    )
    _require_expected_value(
        collapsed,
        expected["collapsed_duplicate_occurrences"],
        "collapsed duplicate occurrences",
    )
    audit_by_tf = {
        row["transcription_factor"]: row
        for row in reconciliation.audit_rows
    }
    for transcription_factor, expected_count in expected[
        "collapsed_duplicate_occurrences_by_tf"
    ].items():
        _require_expected_value(
            audit_by_tf[transcription_factor][
                "collapsed_duplicate_occurrence_count"
            ],
            expected_count,
            "{0} collapsed duplicate occurrences".format(
                transcription_factor
            ),
        )
    _require_expected_value(
        len(result.global_groups),
        expected["global_rc_groups"],
        "global RC groups",
    )
    shared_count = sum(
        group.transcription_factor_degree >= 2 for group in result.global_groups
    )
    _require_expected_value(
        shared_count,
        expected["groups_shared_across_at_least_two_tfs"],
        "shared global RC groups",
    )
    observed_degrees = Counter(
        group.transcription_factor_degree for group in result.global_groups
    )
    expected_degrees = {
        int(degree): int(count)
        for degree, count in expected[
            "global_rc_group_degree_distribution"
        ].items()
    }
    if dict(sorted(observed_degrees.items())) != dict(sorted(expected_degrees.items())):
        raise ExdHoxSplitError("Global RC-group degree distribution differs.")
    observed_global_counts = Counter(result.assignments.values())
    for split in SPLIT_NAMES:
        _require_expected_value(
            observed_global_counts[split],
            expected["global_rc_group_counts"][split],
            "{0} global RC groups".format(split),
        )
    observed_tf_counts = primary_split_counts(
        reconciliation.logical_examples,
        result.assignments,
    )
    observed_tf_totals = {
        transcription_factor: sum(split_counts.values())
        for transcription_factor, split_counts in observed_tf_counts.items()
    }
    expected_tf_totals = {
        str(transcription_factor): int(count)
        for transcription_factor, count in expected[
            "per_tf_logical_example_counts"
        ].items()
    }
    if observed_tf_totals != expected_tf_totals:
        raise ExdHoxSplitError("Per-TF logical-example totals differ.")
    for transcription_factor, split_counts in expected[
        "per_tf_split_counts"
    ].items():
        if observed_tf_counts[transcription_factor] != {
            split: int(split_counts[split]) for split in SPLIT_NAMES
        }:
            raise ExdHoxSplitError(
                "Approved per-TF split counts differ for {0}.".format(
                    transcription_factor
                )
            )
    for row in leakage_rows:
        for key in (
            "exact_sequence_overlap_group_count",
            "reverse_complement_equivalent_overlap_group_count",
            "reverse_complement_only_overlap_group_count",
            "logical_example_overlap_count",
        ):
            if int(row[key]) != 0:
                raise ExdHoxSplitError("Primary split leakage audit is nonzero.")


def _require_expected_value(
    observed: Any,
    expected: Any,
    description: str,
) -> None:
    if int(observed) != int(expected):
        message = "Expected {0} {1}; observed {2}."
        raise ExdHoxSplitError(message.format(expected, description, observed))


def _logical_example_rows(
    result: PrimarySplitResult,
) -> Iterable[Dict[str, Any]]:
    examples = sorted(
        result.reconciliation.logical_examples,
        key=lambda example: example.logical_example_id,
    )
    for example in examples:
        split = result.assignments[example.global_rc_group_id]
        target_value = ""
        target_bits = ""
        if split != "test":
            target_value = float32_text(example.target_value)
            target_bits = example.target_bits_big_endian_hex
        yield {
            "logical_example_id": example.logical_example_id,
            "transcription_factor": example.transcription_factor,
            "sequence": example.sequence,
            "sequence_sha256": example.sequence_sha256,
            "reverse_complement_canonical_sequence": (
                example.reverse_complement_canonical_sequence
            ),
            "reverse_complement_canonical_sha256": (
                example.reverse_complement_canonical_sha256
            ),
            "global_rc_group_id": example.global_rc_group_id,
            "primary_split": split,
            "target_value_float32": target_value,
            "target_bits_big_endian_hex": target_bits,
            "target_commitment_sha256": example.target_commitment_sha256,
            "source_occurrence_count": len(example.source_occurrence_ids),
        }


def _provenance_rows(
    result: PrimarySplitResult,
    example_by_id: Mapping[str, LogicalExample],
) -> Iterable[Dict[str, Any]]:
    representative_by_example = {}
    for example in example_by_id.values():
        representative_by_example[example.logical_example_id] = min(
            example.source_occurrence_ids
        )
    occurrences = sorted(
        result.reconciliation.source_occurrences,
        key=lambda occurrence: occurrence.source_occurrence_id,
    )
    for occurrence in occurrences:
        example = example_by_id[occurrence.logical_example_id]
        split = result.assignments[example.global_rc_group_id]
        target_value = ""
        target_bits = ""
        if split != "test":
            target_value = float32_text(occurrence.target_value)
            target_bits = occurrence.target_bits_big_endian_hex
        yield {
            "source_occurrence_id": occurrence.source_occurrence_id,
            "logical_example_id": occurrence.logical_example_id,
            "transcription_factor": occurrence.transcription_factor,
            "supplied_split": occurrence.supplied_split,
            "source_path": occurrence.source_path,
            "source_row_index_zero_based": (
                occurrence.source_row_index_zero_based
            ),
            "sequence": occurrence.sequence,
            "sequence_sha256": occurrence.sequence_sha256,
            "reverse_complement_canonical_sha256": (
                occurrence.reverse_complement_canonical_sha256
            ),
            "global_rc_group_id": example.global_rc_group_id,
            "primary_split": split,
            "target_value_float32": target_value,
            "target_bits_big_endian_hex": target_bits,
            "target_commitment_sha256": example.target_commitment_sha256,
            "is_collapsed_duplicate": int(
                occurrence.source_occurrence_id
                != representative_by_example[occurrence.logical_example_id]
            ),
        }


def _global_group_rows(
    result: PrimarySplitResult,
) -> Iterable[Dict[str, Any]]:
    groups = sorted(
        result.global_groups,
        key=lambda group: group.global_rc_group_id,
    )
    for group in groups:
        yield {
            "global_rc_group_id": group.global_rc_group_id,
            "reverse_complement_canonical_sequence": (
                group.reverse_complement_canonical_sequence
            ),
            "reverse_complement_canonical_sha256": (
                group.reverse_complement_canonical_sha256
            ),
            "transcription_factor_degree": group.transcription_factor_degree,
            "transcription_factors": ",".join(group.transcription_factors),
            "logical_example_count": len(group.logical_example_ids),
            "primary_split": result.assignments[group.global_rc_group_id],
        }


def _assignment_rows(
    result: PrimarySplitResult,
) -> Iterable[Dict[str, Any]]:
    order_index = {
        group_id: index for index, group_id in enumerate(result.assignment_order)
    }
    for group_id in sorted(result.assignments):
        yield {
            "global_rc_group_id": group_id,
            "primary_split": result.assignments[group_id],
            "assignment_order_index_zero_based": order_index[group_id],
            "deterministic_order_sha256": result.assignment_hashes[group_id],
        }


def _public_test_input_rows(
    result: PrimarySplitResult,
) -> Iterable[Dict[str, Any]]:
    examples = []
    for example in result.reconciliation.logical_examples:
        if result.assignments[example.global_rc_group_id] == "test":
            examples.append(example)
    examples.sort(key=lambda example: example.logical_example_id)
    for example in examples:
        yield {
            "logical_example_id": example.logical_example_id,
            "transcription_factor": example.transcription_factor,
            "sequence": example.sequence,
            "sequence_sha256": example.sequence_sha256,
            "reverse_complement_canonical_sequence": (
                example.reverse_complement_canonical_sequence
            ),
            "reverse_complement_canonical_sha256": (
                example.reverse_complement_canonical_sha256
            ),
            "global_rc_group_id": example.global_rc_group_id,
            "target_commitment_sha256": example.target_commitment_sha256,
        }


def _sealed_target_rows(
    result: PrimarySplitResult,
) -> Iterable[Dict[str, Any]]:
    examples = []
    for example in result.reconciliation.logical_examples:
        if result.assignments[example.global_rc_group_id] == "test":
            examples.append(example)
    examples.sort(key=lambda example: example.logical_example_id)
    for example in examples:
        yield {
            "logical_example_id": example.logical_example_id,
            "target_value_float32": float32_text(example.target_value),
            "target_bits_big_endian_hex": example.target_bits_big_endian_hex,
            "target_commitment_sha256": example.target_commitment_sha256,
        }


def _target_commitment_digest(
    sealed_rows: Sequence[Mapping[str, Any]],
) -> str:
    targets = []
    previous_id = None
    for row in sealed_rows:
        logical_example_id = str(row["logical_example_id"])
        if previous_id is not None and logical_example_id <= previous_id:
            raise ExdHoxSplitError("Sealed target rows must be uniquely ID-sorted.")
        targets.append(
            {
                "logical_example_id": logical_example_id,
                "target_commitment_sha256": row[
                    "target_commitment_sha256"
                ],
            }
        )
        previous_id = logical_example_id
    if not targets:
        raise ExdHoxSplitError("The primary test split must not be empty.")
    return hash_logical_content(
        {
            "schema_version": "exd_hox_target_commitment_set.v1",
            "targets": targets,
        }
    )


def _build_split_count_summary_rows(
    records_by_tf: Mapping[str, Mapping[str, SelexHdf5File]],
    result: PrimarySplitResult,
) -> Tuple[Dict[str, Any], ...]:
    primary_counts = primary_split_counts(
        result.reconciliation.logical_examples,
        result.assignments,
    )
    groups_by_tf_split = defaultdict(set)
    for example in result.reconciliation.logical_examples:
        split = result.assignments[example.global_rc_group_id]
        groups_by_tf_split[(example.transcription_factor, split)].add(
            example.global_rc_group_id
        )
    overlap_by_tf = {}
    for transcription_factor in sorted(records_by_tf):
        split_records = records_by_tf[transcription_factor]
        supplied_summary, unused_details = audit_supplied_split(
            split_records["train"],
            split_records["test"],
        )
        del unused_details
        overlap_by_tf[transcription_factor] = int(
            supplied_summary["exact_labeled_row_overlap_count"]
        )
    rows = []
    for transcription_factor in sorted(records_by_tf):
        for split in SPLIT_NAMES:
            logical_count = primary_counts[transcription_factor][split]
            rows.append(
                {
                    "protocol": "primary",
                    "transcription_factor": transcription_factor,
                    "split": split,
                    "row_count": logical_count,
                    "logical_example_count": logical_count,
                    "global_rc_group_count": len(
                        groups_by_tf_split[(transcription_factor, split)]
                    ),
                    "exact_cross_split_overlap_occurrence_count": 0,
                }
            )
        for supplied_split in SUPPLIED_SPLIT_NAMES:
            record = records_by_tf[transcription_factor][supplied_split]
            supplied_groups = {
                reverse_complement_canonical_sequence(sequence)
                for sequence in record.sequences
            }
            rows.append(
                {
                    "protocol": "paper_split_reproduction",
                    "transcription_factor": transcription_factor,
                    "split": supplied_split,
                    "row_count": record.row_count,
                    "logical_example_count": record.row_count,
                    "global_rc_group_count": len(supplied_groups),
                    "exact_cross_split_overlap_occurrence_count": overlap_by_tf[
                        transcription_factor
                    ],
                }
            )
    protocol_order = {"primary": 0, "paper_split_reproduction": 1}
    split_order = {
        "training": 0,
        "validation": 1,
        "test": 2,
        "train": 0,
    }
    rows.sort(
        key=lambda row: (
            row["transcription_factor"],
            protocol_order[row["protocol"]],
            split_order[row["split"]],
        )
    )
    return tuple(rows)


def _build_supplied_leakage_row(
    records_by_tf: Mapping[str, Mapping[str, SelexHdf5File]],
) -> Dict[str, Any]:
    totals = Counter()
    for transcription_factor in sorted(records_by_tf):
        split_records = records_by_tf[transcription_factor]
        summary, details = audit_supplied_split(
            split_records["train"],
            split_records["test"],
        )
        totals["exact_sequence_overlap_group_count"] += int(
            summary["exact_sequence_overlap_group_count"]
        )
        totals["reverse_complement_equivalent_overlap_group_count"] += int(
            summary["reverse_complement_equivalent_overlap_group_count"]
        )
        totals["reverse_complement_only_overlap_group_count"] += int(
            summary["reverse_complement_only_overlap_group_count"]
        )
        totals["logical_example_overlap_count"] += len(details)
    return {
        "comparison": "paper_split_reproduction",
        "left_split": "train",
        "right_split": "test",
        "exact_sequence_overlap_group_count": totals[
            "exact_sequence_overlap_group_count"
        ],
        "reverse_complement_equivalent_overlap_group_count": totals[
            "reverse_complement_equivalent_overlap_group_count"
        ],
        "reverse_complement_only_overlap_group_count": totals[
            "reverse_complement_only_overlap_group_count"
        ],
        "logical_example_overlap_count": totals[
            "logical_example_overlap_count"
        ],
    }


def _build_public_affinity_histogram_rows(
    result: PrimarySplitResult,
    bin_count: int = 50,
) -> Tuple[Dict[str, Any], ...]:
    if bin_count <= 0:
        raise ExdHoxSplitError("Plot affinity bin count must be positive.")
    counts = Counter()
    transcription_factors = sorted(
        {
            example.transcription_factor
            for example in result.reconciliation.logical_examples
        }
    )
    for example in result.reconciliation.logical_examples:
        split = result.assignments[example.global_rc_group_id]
        if split == "test":
            continue
        value = example.target_value
        bin_index = min(bin_count - 1, int(value * bin_count))
        counts[(example.transcription_factor, split, bin_index)] += 1
    rows = []
    for transcription_factor in transcription_factors:
        for split in ("training", "validation"):
            for bin_index in range(bin_count):
                rows.append(
                    {
                        "transcription_factor": transcription_factor,
                        "split": split,
                        "bin_index": bin_index,
                        "bin_left": format(bin_index / bin_count, ".9g"),
                        "bin_right": format((bin_index + 1) / bin_count, ".9g"),
                        "logical_example_count": counts[
                            (transcription_factor, split, bin_index)
                        ],
                    }
                )
    return tuple(rows)


def _primary_manifest_counts(result: PrimarySplitResult) -> Dict[str, Any]:
    global_counts = Counter(result.assignments.values())
    degree_distribution = Counter(
        group.transcription_factor_degree for group in result.global_groups
    )
    collapsed_by_tf = {
        row["transcription_factor"]: int(
            row["collapsed_duplicate_occurrence_count"]
        )
        for row in result.reconciliation.audit_rows
    }
    per_tf_split_counts = primary_split_counts(
        result.reconciliation.logical_examples,
        result.assignments,
    )
    return {
        "source_occurrences": len(result.reconciliation.source_occurrences),
        "logical_examples": len(result.reconciliation.logical_examples),
        "collapsed_duplicate_occurrences": (
            len(result.reconciliation.source_occurrences)
            - len(result.reconciliation.logical_examples)
        ),
        "collapsed_duplicate_occurrences_by_tf": collapsed_by_tf,
        "global_rc_groups": len(result.global_groups),
        "groups_shared_across_at_least_two_tfs": sum(
            group.transcription_factor_degree >= 2
            for group in result.global_groups
        ),
        "global_rc_group_degree_distribution": {
            str(degree): degree_distribution[degree]
            for degree in sorted(degree_distribution)
        },
        "global_rc_group_counts": {
            split: global_counts[split] for split in SPLIT_NAMES
        },
        "per_tf_logical_example_counts": {
            transcription_factor: sum(split_counts.values())
            for transcription_factor, split_counts in per_tf_split_counts.items()
        },
        "per_tf_split_counts": per_tf_split_counts,
    }


def _fingerprint_staged_files(
    staging_directory: Path,
    logical_directory: str,
    filenames: Sequence[str],
) -> list:
    artifacts = []
    for filename in filenames:
        artifacts.append(
            fingerprint_file(
                staging_directory / filename,
                Path(logical_directory, filename).as_posix(),
            ).to_dict()
        )
    artifacts.sort(key=lambda artifact: artifact["path"])
    return artifacts


def _require_new_directory(path: Path, description: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite existing {0} directory: {1}".format(
                description,
                path,
            )
        )


def _prepare_primary_staging(
    split_directory: Path,
    sealed_directory: Path,
) -> Tuple[
    tempfile.TemporaryDirectory,
    tempfile.TemporaryDirectory,
    Path,
    Path,
]:
    """Reserve the ignored seal and create cleanup-safe staging roots."""

    sealed_directory.mkdir()
    split_context = None
    sealed_context = None
    try:
        split_context = tempfile.TemporaryDirectory(
            prefix=".exd_hox_primary_split_staging_",
            dir=split_directory.parent,
        )
        sealed_context = tempfile.TemporaryDirectory(
            prefix=".exd_hox_sealed_target_staging_",
            dir=sealed_directory,
        )
        split_staging = Path(split_context.name) / "split"
        split_staging.mkdir()
    except OSError:
        if sealed_context is not None:
            sealed_context.cleanup()
        if split_context is not None:
            split_context.cleanup()
        sealed_directory.rmdir()
        raise
    return (
        split_context,
        sealed_context,
        split_staging,
        Path(sealed_context.name),
    )


def _require_exact_regular_files(
    directory: Path,
    expected_filenames: set[str],
    description: str,
) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise ExdHoxSplitError("{0} directory is not regular.".format(description))
    observed_filenames = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ExdHoxSplitError(
                "{0} contains a non-regular entry.".format(description)
            )
        observed_filenames.add(path.name)
    if observed_filenames != expected_filenames:
        raise ExdHoxSplitError("{0} file set differs from v1.".format(description))


def _validate_artifact_fingerprint_entry(
    artifact: Mapping[str, Any],
    expected_logical_path: str,
    physical_path: Path,
    description: str,
) -> None:
    if set(artifact) != {"path", "byte_size", "sha256"}:
        raise ExdHoxSplitError("{0} fingerprint fields differ.".format(description))
    logical_path = validate_repository_relative_path(str(artifact["path"]))
    if logical_path != expected_logical_path:
        raise ExdHoxSplitError("{0} logical path differs.".format(description))
    _validate_sha256_text(str(artifact["sha256"]), "artifact sha256")
    _require_file_identity(
        physical_path,
        int(artifact["byte_size"]),
        str(artifact["sha256"]),
        description,
    )


def build_nested_subset_artifacts(
    config_path: Path | str,
    repository_root: Path | str,
    split_directory: Optional[Path | str] = None,
    output_directory: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Build immutable nested subsets from finalized public training rows."""

    if (split_directory is None) != (output_directory is None):
        raise ExdHoxSplitError(
            "Subset physical input/output overrides must be supplied together."
        )
    root = Path(repository_root).resolve()
    config_file = _resolve_config_path(root, config_path)
    config_logical_path = repository_relative_path(config_file, root)
    config = load_split_config(config_file)
    outputs = config["outputs"]
    split_logical_directory = str(outputs["split_directory"])
    subset_logical_directory = str(outputs["subset_directory"])
    if split_directory is None:
        physical_split_directory = _resolve_repository_path(
            root,
            split_logical_directory,
        )
        subset_directory = _resolve_repository_path(
            root,
            subset_logical_directory,
        )
    else:
        physical_split_directory = Path(split_directory).resolve()
        subset_directory = Path(output_directory).resolve()
    _require_disjoint_directories(
        physical_split_directory,
        subset_directory,
        "Primary split and nested subset",
    )
    _require_new_directory(subset_directory, "nested subset")
    split_manifest_path = physical_split_directory / SPLIT_MANIFEST_FILENAME
    split_manifest = validate_split_artifacts(
        split_manifest_path,
        root,
        split_directory=physical_split_directory,
    )
    if split_manifest["config_sha256"] != hash_file_bytes(config_file):
        raise ExdHoxSplitError("Split manifest was built from a different config.")

    logical_rows = _read_tsv_gzip_strict(
        physical_split_directory / LOGICAL_EXAMPLES_FILENAME,
        LOGICAL_EXAMPLE_FIELDS,
    )
    logical_examples, assignments = _training_examples_from_public_rows(
        logical_rows
    )
    subset_policy = config["subset_policy"]
    ordering_rows = build_nested_subset_ordering(
        logical_examples=logical_examples,
        assignments=assignments,
        seed=int(subset_policy["seed"]),
        affinity_bin_count=int(config["split_policy"]["affinity_bin_count"]),
        minimum_distinct_groups_per_bin=int(
            config["split_policy"]["minimum_distinct_groups_per_bin"]
        ),
    )
    level_rows = resolve_nested_subset_levels(
        ordering_rows=ordering_rows,
        absolute_counts=subset_policy["absolute_counts"],
        fractional_levels=subset_policy["fractional_levels"],
        absolute_alias_tolerance=subset_policy["absolute_alias_tolerance"],
        minimum_primary_count=int(subset_policy["minimum_primary_count"]),
    )

    subset_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_context = tempfile.TemporaryDirectory(
        prefix=".exd_hox_nested_subset_staging_",
        dir=subset_directory.parent,
    )
    staging_directory = Path(staging_context.name) / "subsets"
    staging_directory.mkdir()
    try:
        write_tsv_gzip_exclusive(
            staging_directory / SUBSET_ORDERING_FILENAME,
            SUBSET_ORDERING_FIELDS,
            ordering_rows,
        )
        write_tsv_exclusive(
            staging_directory / SUBSET_LEVELS_FILENAME,
            SUBSET_LEVEL_FIELDS,
            level_rows,
        )
        artifacts = _fingerprint_staged_files(
            staging_directory,
            subset_logical_directory,
            (SUBSET_ORDERING_FILENAME, SUBSET_LEVELS_FILENAME),
        )
        manifest = build_hashed_manifest(
            SUBSET_MANIFEST_SCHEMA_VERSION,
            {
                "study_identifier": config["study"]["identifier"],
                "dataset_identifier": config["study"]["dataset_identifier"],
                "config_path": config_logical_path,
                "config_sha256": hash_file_bytes(config_file),
                "split_manifest_hash": split_manifest["manifest_hash"],
                "split_identity_hash": split_manifest["split_identity_hash"],
                "subset_directory": subset_logical_directory,
                "policy": dict(subset_policy),
                "shared_model_conditions": [
                    "cnn_rc",
                    "random_transformer",
                    "s0_transformer",
                    "s1_transformer",
                ],
                "artifacts": artifacts,
                "level_row_count": len(level_rows),
                "ordering_logical_example_count": len(ordering_rows),
            },
        )
        write_json_exclusive(
            staging_directory / SUBSET_MANIFEST_FILENAME,
            manifest,
        )
        _require_new_directory(subset_directory, "nested subset")
        os.rename(staging_directory, subset_directory)
    finally:
        staging_context.cleanup()
    return manifest


def validate_split_artifacts(
    manifest_path: Path | str,
    repository_root: Path | str,
    split_directory: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Validate public split artifacts without opening sealed target bytes."""

    root = Path(repository_root).resolve()
    path = Path(manifest_path)
    manifest = _load_json_mapping(path, "primary split manifest")
    try:
        validate_hashed_manifest(manifest)
    except (KeyError, ValueError) as error:
        raise ExdHoxSplitError("Primary split manifest hash mismatch.") from error
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ExdHoxSplitError("Unsupported primary split manifest schema.")
    required_manifest_fields = {
        "schema_version",
        "study_identifier",
        "dataset_identifier",
        "config_path",
        "config_sha256",
        "source_manifest_hash",
        "audit_manifest_hash",
        "split_identity_hash",
        "split_directory",
        "policy",
        "excluded_identity_axes",
        "artifacts",
        "sealed_target_manifest_hash",
        "test_access_policy_manifest_hash",
        "counts",
        "test_target_handling",
        "manifest_hash",
    }
    if set(manifest) != required_manifest_fields:
        raise ExdHoxSplitError("Primary split manifest fields differ from v1.")
    split_logical_directory = validate_repository_relative_path(
        str(manifest["split_directory"])
    )
    physical_directory = (
        Path(split_directory).resolve()
        if split_directory is not None
        else path.resolve().parent
    )
    if split_directory is None:
        expected_directory = _resolve_repository_path(
            root,
            split_logical_directory,
        )
        if physical_directory != expected_directory:
            raise ExdHoxSplitError("Primary split directory differs from manifest.")
    if path.resolve() != physical_directory / SPLIT_MANIFEST_FILENAME:
        raise ExdHoxSplitError("Primary split manifest filename or directory differs.")
    expected_filenames = {
        LOGICAL_EXAMPLES_FILENAME,
        PROVENANCE_FILENAME,
        GLOBAL_GROUPS_FILENAME,
        ASSIGNMENTS_FILENAME,
        RECONCILIATION_AUDIT_FILENAME,
        COUNT_SUMMARY_FILENAME,
        LEAKAGE_AUDIT_FILENAME,
        AFFINITY_HISTOGRAM_FILENAME,
        PUBLIC_TEST_INPUTS_FILENAME,
        SEALED_TARGET_MANIFEST_FILENAME,
    }
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ExdHoxSplitError("Primary split artifacts must be a list.")
    if len(artifacts) != len(expected_filenames):
        raise ExdHoxSplitError("Primary split artifact count differs from v1.")
    artifacts_by_path = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ExdHoxSplitError("Primary split artifact must be a mapping.")
        logical_path = validate_repository_relative_path(str(artifact.get("path")))
        if logical_path in artifacts_by_path:
            raise ExdHoxSplitError("Primary split repeats an artifact path.")
        artifacts_by_path[logical_path] = artifact
    expected_logical_paths = {
        Path(split_logical_directory, filename).as_posix()
        for filename in expected_filenames
    }
    if set(artifacts_by_path) != expected_logical_paths:
        raise ExdHoxSplitError("Primary split logical artifact set differs from v1.")
    for filename in sorted(expected_filenames):
        logical_path = Path(split_logical_directory, filename).as_posix()
        _validate_artifact_fingerprint_entry(
            artifacts_by_path[logical_path],
            logical_path,
            physical_directory / filename,
            "Public split artifact",
        )
    _require_exact_regular_files(
        physical_directory,
        expected_filenames | {SPLIT_MANIFEST_FILENAME},
        "Primary split",
    )
    descriptor = _load_json_mapping(
        physical_directory / SEALED_TARGET_MANIFEST_FILENAME,
        "sealed-target descriptor",
    )
    try:
        validate_hashed_manifest(descriptor)
    except (KeyError, TypeError, ValueError) as error:
        raise ExdHoxSplitError("Sealed-target descriptor hash mismatch.") from error
    required_descriptor_fields = {
        "schema_version",
        "sealed_target_path",
        "sealed_target_byte_size",
        "sealed_target_sha256",
        "test_logical_example_count",
        "target_commitment_digest_sha256",
        "split_identity_hash",
        "split_policy_identifier",
        "manifest_hash",
    }
    if set(descriptor) != required_descriptor_fields:
        raise ExdHoxSplitError("Sealed-target descriptor fields differ from v1.")
    if descriptor.get("schema_version") != SEALED_TARGET_MANIFEST_SCHEMA_VERSION:
        raise ExdHoxSplitError("Unsupported sealed-target descriptor schema.")
    if descriptor["manifest_hash"] != manifest["sealed_target_manifest_hash"]:
        raise ExdHoxSplitError("Split manifest binds a different target descriptor.")
    if descriptor["split_identity_hash"] != manifest["split_identity_hash"]:
        raise ExdHoxSplitError("Target descriptor split identity differs.")
    _validate_public_test_redaction(physical_directory, descriptor)
    return manifest


def validate_subset_artifacts(
    manifest_path: Path | str,
    repository_root: Path | str,
    subset_directory: Optional[Path | str] = None,
    expected_split_manifest_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate the immutable nested-subset artifact set."""

    root = Path(repository_root).resolve()
    path = Path(manifest_path)
    manifest = _load_json_mapping(path, "subset-set manifest")
    try:
        validate_hashed_manifest(manifest)
    except (KeyError, ValueError) as error:
        raise ExdHoxSplitError("Subset-set manifest hash mismatch.") from error
    if manifest.get("schema_version") != SUBSET_MANIFEST_SCHEMA_VERSION:
        raise ExdHoxSplitError("Unsupported subset-set manifest schema.")
    required_manifest_fields = {
        "schema_version",
        "study_identifier",
        "dataset_identifier",
        "config_path",
        "config_sha256",
        "split_manifest_hash",
        "split_identity_hash",
        "subset_directory",
        "policy",
        "shared_model_conditions",
        "artifacts",
        "level_row_count",
        "ordering_logical_example_count",
        "manifest_hash",
    }
    if set(manifest) != required_manifest_fields:
        raise ExdHoxSplitError("Subset-set manifest fields differ from v1.")
    if (
        expected_split_manifest_hash is not None
        and manifest["split_manifest_hash"] != expected_split_manifest_hash
    ):
        raise ExdHoxSplitError("Subset set binds a different primary split.")
    subset_logical_directory = validate_repository_relative_path(
        str(manifest["subset_directory"])
    )
    physical_directory = (
        Path(subset_directory).resolve()
        if subset_directory is not None
        else path.resolve().parent
    )
    if subset_directory is None:
        expected_directory = _resolve_repository_path(
            root,
            subset_logical_directory,
        )
        if physical_directory != expected_directory:
            raise ExdHoxSplitError("Nested subset directory differs from manifest.")
    if path.resolve() != physical_directory / SUBSET_MANIFEST_FILENAME:
        raise ExdHoxSplitError("Subset-set manifest filename or directory differs.")
    expected_filenames = {SUBSET_ORDERING_FILENAME, SUBSET_LEVELS_FILENAME}
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ExdHoxSplitError("Nested subset artifacts must be a list.")
    if len(artifacts) != len(expected_filenames):
        raise ExdHoxSplitError("Nested subset artifact count differs from v1.")
    artifacts_by_path = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ExdHoxSplitError("Nested subset artifact must be a mapping.")
        logical_path = validate_repository_relative_path(str(artifact.get("path")))
        if logical_path in artifacts_by_path:
            raise ExdHoxSplitError("Nested subset repeats an artifact path.")
        artifacts_by_path[logical_path] = artifact
    expected_logical_paths = {
        Path(subset_logical_directory, filename).as_posix()
        for filename in expected_filenames
    }
    if set(artifacts_by_path) != expected_logical_paths:
        raise ExdHoxSplitError("Nested subset logical artifact set differs from v1.")
    for filename in sorted(expected_filenames):
        logical_path = Path(subset_logical_directory, filename).as_posix()
        _validate_artifact_fingerprint_entry(
            artifacts_by_path[logical_path],
            logical_path,
            physical_directory / filename,
            "Nested subset artifact",
        )
    _require_exact_regular_files(
        physical_directory,
        expected_filenames | {SUBSET_MANIFEST_FILENAME},
        "Nested subset",
    )
    return manifest


def _read_tsv_gzip_strict(
    path: Path,
    expected_fields: Sequence[str],
) -> Tuple[Dict[str, str], ...]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            raise ExdHoxSplitError(
                "Compressed TSV schema differs for {0}.".format(path.name)
            )
        return tuple(dict(row) for row in reader)


def _read_tsv_strict(
    path: Path,
    expected_fields: Sequence[str],
) -> Tuple[Dict[str, str], ...]:
    with open(path, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            raise ExdHoxSplitError(
                "TSV schema differs for {0}.".format(path.name)
            )
        return tuple(dict(row) for row in reader)


def _training_examples_from_public_rows(
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Tuple[LogicalExample, ...], Dict[str, str]]:
    examples = []
    assignments = {}
    observed_ids = set()
    for row in rows:
        logical_example_id = str(row["logical_example_id"])
        if logical_example_id in observed_ids:
            raise ExdHoxSplitError("Public logical-example IDs must be unique.")
        observed_ids.add(logical_example_id)
        split = str(row["primary_split"])
        if split not in SPLIT_NAMES:
            raise ExdHoxSplitError("Public logical row has an invalid split.")
        group_id = str(row["global_rc_group_id"])
        previous_split = assignments.get(group_id)
        if previous_split is not None and previous_split != split:
            raise ExdHoxSplitError("One public RC group spans multiple splits.")
        assignments[group_id] = split
        if split == "test":
            if row["target_value_float32"] or row["target_bits_big_endian_hex"]:
                raise ExdHoxSplitError("Public test targets are not redacted.")
            continue
        if split != "training":
            continue
        target_bits = str(row["target_bits_big_endian_hex"])
        target_value = float32_from_big_endian_hex(target_bits)
        if float32_text(target_value) != str(row["target_value_float32"]):
            raise ExdHoxSplitError("Public training target decimal is noncanonical.")
        expected_commitment = target_commitment(logical_example_id, target_bits)
        if expected_commitment != row["target_commitment_sha256"]:
            raise ExdHoxSplitError("Public training target commitment differs.")
        sequence = str(row["sequence"])
        canonical_sequence = str(row["reverse_complement_canonical_sequence"])
        if reverse_complement_canonical_sequence(sequence) != canonical_sequence:
            raise ExdHoxSplitError("Public logical row RC canonicalization differs.")
        examples.append(
            LogicalExample(
                logical_example_id=logical_example_id,
                transcription_factor=str(row["transcription_factor"]),
                sequence=sequence,
                sequence_sha256=str(row["sequence_sha256"]),
                reverse_complement_canonical_sequence=canonical_sequence,
                reverse_complement_canonical_sha256=str(
                    row["reverse_complement_canonical_sha256"]
                ),
                global_rc_group_id=group_id,
                target_value=target_value,
                target_bits_big_endian_hex=target_bits,
                target_commitment_sha256=str(
                    row["target_commitment_sha256"]
                ),
                source_occurrence_ids=(),
            )
        )
    examples.sort(key=lambda example: example.logical_example_id)
    return tuple(examples), assignments


def _validate_public_test_redaction(
    split_directory: Path,
    sealed_target_descriptor: Mapping[str, Any],
) -> None:
    logical_rows = _read_tsv_gzip_strict(
        split_directory / LOGICAL_EXAMPLES_FILENAME,
        LOGICAL_EXAMPLE_FIELDS,
    )
    test_ids = set()
    commitments = {}
    for row in logical_rows:
        if row["primary_split"] == "test":
            if row["target_value_float32"] or row["target_bits_big_endian_hex"]:
                raise ExdHoxSplitError("Public logical test target is not redacted.")
            test_ids.add(row["logical_example_id"])
            commitments[row["logical_example_id"]] = row[
                "target_commitment_sha256"
            ]
    provenance_rows = _read_tsv_gzip_strict(
        split_directory / PROVENANCE_FILENAME,
        PROVENANCE_FIELDS,
    )
    for row in provenance_rows:
        if row["primary_split"] == "test":
            if row["target_value_float32"] or row["target_bits_big_endian_hex"]:
                raise ExdHoxSplitError(
                    "Public provenance test target is not redacted."
                )
    public_test_rows = _read_tsv_gzip_strict(
        split_directory / PUBLIC_TEST_INPUTS_FILENAME,
        PUBLIC_TEST_INPUT_FIELDS,
    )
    public_test_ids = set()
    for row in public_test_rows:
        logical_example_id = row["logical_example_id"]
        if logical_example_id in public_test_ids:
            raise ExdHoxSplitError("Public test input IDs must be unique.")
        public_test_ids.add(logical_example_id)
        if commitments.get(logical_example_id) != row[
            "target_commitment_sha256"
        ]:
            raise ExdHoxSplitError("Public test commitment differs.")
    if public_test_ids != test_ids:
        raise ExdHoxSplitError("Public test inputs do not cover all test IDs.")
    if int(sealed_target_descriptor["test_logical_example_count"]) != len(
        public_test_rows
    ):
        raise ExdHoxSplitError("Sealed descriptor test count differs from public IDs.")
    public_commitment_digest = _target_commitment_digest(public_test_rows)
    if public_commitment_digest != sealed_target_descriptor[
        "target_commitment_digest_sha256"
    ]:
        raise ExdHoxSplitError(
            "Sealed descriptor commitment digest differs from public commitments."
        )
    affinity_rows = _read_tsv_strict(
        split_directory / AFFINITY_HISTOGRAM_FILENAME,
        AFFINITY_HISTOGRAM_FIELDS,
    )
    if any(row["split"] == "test" for row in affinity_rows):
        raise ExdHoxSplitError("Public output contains a test affinity histogram.")
