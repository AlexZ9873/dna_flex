"""Coordinate-preserving genomic pretraining split generation and auditing."""

from __future__ import annotations

from bisect import bisect_right
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from Bio import SeqIO
import yaml

from src.coordinates import reverse_complement
from src.data_fingerprints import (
    SOURCE_FINGERPRINT_SCHEMA_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    build_pretraining_split_manifest,
    fingerprint_sequence_file,
    hash_file_bytes,
    hash_logical_content,
    load_split_manifest,
    repository_relative_source_path,
    reverse_complement_canonical_sequence,
    save_split_manifest,
)


CONFIG_SCHEMA_VERSION = "genomic_pretraining_split_config.v1"
LOGICAL_REFERENCE_SCHEMA_VERSION = "fasta_logical_reference.v1"
DRY_RUN_SCHEMA_VERSION = "genomic_pretraining_split_dry_run.v1"
GENOMIC_MANIFEST_SCHEMA_VERSION = "genomic_pretraining_split_manifest.v1"
GENOMIC_AUDIT_SCHEMA_VERSION = "genomic_pretraining_split_audit.v1"
RECORD_ID_SCHEMA_VERSION = "genomic_window_record_id.v1"
NORMALIZATION_ARTIFACT_SCHEMA_VERSION = "biophysical_normalization.v2"
POLICY_ID = "hg38_whole_chromosome_holdout.v1"
CREATION_ENTRY_POINT = (
    "python -m scripts.data_prep.build_hg38_pretraining_split"
)
CHILD_MANIFEST_MAXIMUM_EXAMPLES = 10
SPLIT_NAMES = ("training", "validation", "test")
AUDIT_MODES = ("report", "strict")
VALID_BASE_PATTERN = re.compile(r"[ACGT]+")
VALID_SEQUENCE_PATTERN = re.compile(r"^[ACGT]+$")

APPROVED_REFERENCE_PATH = "data/raw/hg38.fa"
APPROVED_REFERENCE_SHA256 = (
    "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51"
)
APPROVED_REFERENCE_IDENTIFIER = "ucsc_hg38"
APPROVED_ELIGIBLE_CHROMOSOMES = tuple(
    "chr{0}".format(index) for index in range(1, 23)
)
APPROVED_TRAINING_CHROMOSOMES = tuple(
    "chr{0}".format(index) for index in range(1, 21)
)
APPROVED_VALIDATION_CHROMOSOMES = ("chr21",)
APPROVED_TEST_CHROMOSOMES = ("chr22",)
APPROVED_WINDOW_LENGTH = 256
APPROVED_SAMPLING_SEED = 42
APPROVED_TRAINING_TARGET_COUNT = 180000
APPROVED_VALIDATION_TARGET_COUNT = 10000
APPROVED_TEST_TARGET_COUNT = 10000
APPROVED_MAXIMUM_CANDIDATE_MULTIPLIER = 2
APPROVED_OUTPUT_DIRECTORY = "data/generated/hg38_pretraining_split_v1"
APPROVED_TRAINING_RECORDS_PATH = (
    "data/generated/hg38_pretraining_split_v1/"
    "hg38_training_records_v1.tsv"
)
APPROVED_VALIDATION_RECORDS_PATH = (
    "data/generated/hg38_pretraining_split_v1/"
    "hg38_validation_records_v1.tsv"
)
APPROVED_TEST_RECORDS_PATH = (
    "data/generated/hg38_pretraining_split_v1/"
    "hg38_test_records_v1.tsv"
)
APPROVED_TRAINING_SEQUENCES_PATH = (
    "data/generated/hg38_pretraining_split_v1/"
    "hg38_training_sequences_v1.txt"
)
APPROVED_VALIDATION_SEQUENCES_PATH = (
    "data/generated/hg38_pretraining_split_v1/"
    "hg38_validation_sequences_v1.txt"
)
APPROVED_TEST_SEQUENCES_PATH = (
    "data/generated/hg38_pretraining_split_v1/"
    "hg38_test_sequences_v1.txt"
)
APPROVED_REJECTION_RECORDS_PATH = (
    "data/generated/hg38_pretraining_split_v1/hg38_rejections_v1.tsv"
)
APPROVED_CHILD_MANIFEST_PATH = (
    "data/generated/hg38_pretraining_split_v1/"
    "hg38_train_validation_manifest_v2.json"
)
APPROVED_GENOMIC_MANIFEST_PATH = (
    "data/generated/hg38_pretraining_split_v1/"
    "hg38_coordinate_split_manifest_v1.json"
)

REJECTION_REASONS = (
    "cross_split_exact_sequence_group",
    "cross_split_reverse_complement_only_group",
    "blacklisted_cross_split_equivalence_group",
)
PRIMARY_REJECTION_REASONS = REJECTION_REASONS[:2]
RESOLUTION_FIELD_NAMES = (
    "resolution_iterations",
    "unique_cross_split_equivalence_groups_rejected",
    "unique_exact_groups_rejected",
    "unique_rc_only_groups_rejected",
    "rejected_candidate_record_count",
    "consumed_candidate_counts",
    "maximum_candidate_counts",
)
GENOMIC_MANIFEST_FIELD_NAMES = (
    "schema_version",
    "creation_entry_point",
    "reference",
    "policy",
    "capacity_and_exclusions",
    "outputs",
    "rejections",
    "audits",
    "compatibility",
    "manifest_hash",
)
COMPATIBILITY_FIELD_NAMES = (
    "source_fingerprint_schema_version",
    "child_manifest_schema_version",
    "child_manifest_path",
    "child_manifest_file_sha256",
    "child_manifest_hash",
    "normalization_artifact_schema_version",
    "test_source_fingerprint",
)

RECORD_FIELD_NAMES = (
    "record_id",
    "reference_id",
    "chromosome",
    "start",
    "end",
    "strand",
    "sequence",
    "split",
    "sequence_sha256",
    "rc_canonical_sha256",
    "block_id",
    "selection_rank",
    "split_policy_version",
)
REJECTION_FIELD_NAMES = RECORD_FIELD_NAMES + (
    "canonical_sequence_sha256",
    "rejection_reason",
)


class GenomicSplitError(ValueError):
    """Base error for invalid genomic split inputs or outputs."""


class CandidateBudgetError(GenomicSplitError):
    """Raised when leakage-free targets exceed the candidate budget."""


class GenomicSplitAuditError(GenomicSplitError):
    """Raised when strict coordinate or sequence auditing fails."""


@dataclass(frozen=True)
class EligibleStartInterval:
    """A half-open range of valid genomic window-start coordinates."""

    chromosome: str
    start: int
    stop: int
    split: str

    @property
    def capacity(self) -> int:
        """Return the number of eligible starts in this interval."""

        return self.stop - self.start


@dataclass(frozen=True)
class ReferenceContigSummary:
    """Logical identity and eligibility counts for one FASTA contig."""

    identifier: str
    length: int
    sequence_sha256: str
    eligible: bool
    split: Optional[str]
    acgt_base_count: int
    n_base_count: int
    other_symbol_count: int
    lowercase_base_count: int
    total_possible_window_starts: int
    eligible_window_start_count: int
    invalid_window_start_count: int

    def logical_identity_dict(self) -> Dict[str, Any]:
        """Return wrapping- and case-independent contig identity."""

        return {
            "identifier": self.identifier,
            "length": self.length,
            "sequence_sha256": self.sequence_sha256,
        }

    def report_dict(self) -> Dict[str, Any]:
        """Return complete deterministic contig metadata."""

        return {
            "identifier": self.identifier,
            "length": self.length,
            "sequence_sha256": self.sequence_sha256,
            "eligible": self.eligible,
            "split": self.split,
            "acgt_base_count": self.acgt_base_count,
            "n_base_count": self.n_base_count,
            "other_symbol_count": self.other_symbol_count,
            "lowercase_base_count": self.lowercase_base_count,
            "total_possible_window_starts": (
                self.total_possible_window_starts
            ),
            "eligible_window_start_count": (
                self.eligible_window_start_count
            ),
            "invalid_window_start_count": self.invalid_window_start_count,
        }


@dataclass(frozen=True)
class ReferenceScan:
    """One deterministic scan of a FASTA reference."""

    reference_path: str
    source_file_size_bytes: int
    source_file_mtime_ns: int
    raw_file_sha256: str
    logical_reference_sha256: str
    contigs: Tuple[ReferenceContigSummary, ...]
    eligible_intervals: Tuple[EligibleStartInterval, ...]
    excluded_contig_identifiers: Tuple[str, ...]

    def split_capacity(self, split: str) -> int:
        """Return the eligible window-start capacity for one split."""

        capacity = 0
        for interval in self.eligible_intervals:
            if interval.split == split:
                capacity += interval.capacity
        return capacity

    def intervals_for_split(
        self,
        split: str,
    ) -> Tuple[EligibleStartInterval, ...]:
        """Return eligible intervals for one split in stable order."""

        intervals = []
        for interval in self.eligible_intervals:
            if interval.split == split:
                intervals.append(interval)
        return tuple(intervals)

    def contig_by_identifier(self) -> Mapping[str, ReferenceContigSummary]:
        """Return contig summaries keyed by identifier."""

        return {contig.identifier: contig for contig in self.contigs}


@dataclass(frozen=True)
class CandidateCoordinate:
    """One sampled coordinate in a deterministic per-split stream."""

    chromosome: str
    start: int
    end: int
    split: str
    selection_rank: int


@dataclass(frozen=True)
class GenomicWindowRecord:
    """One coordinate-preserving reference-forward sequence record."""

    record_id: str
    reference_id: str
    chromosome: str
    start: int
    end: int
    strand: str
    sequence: str
    split: str
    sequence_sha256: str
    rc_canonical_sha256: str
    block_id: str
    selection_rank: int
    split_policy_version: str

    @classmethod
    def create(
        cls,
        reference_id: str,
        chromosome: str,
        start: int,
        end: int,
        sequence: str,
        split: str,
        selection_rank: int,
        split_policy_version: str,
    ) -> "GenomicWindowRecord":
        """Create and hash one validated reference-forward record."""

        normalized = sequence.upper()
        if VALID_SEQUENCE_PATTERN.fullmatch(normalized) is None:
            raise GenomicSplitError(
                "Generated genomic windows must contain only A, C, G, and T."
            )
        if end <= start:
            raise GenomicSplitError("Genomic window end must exceed its start.")
        if start < 0:
            raise GenomicSplitError(
                "Genomic window start must be non-negative."
            )
        if selection_rank < 0:
            raise GenomicSplitError(
                "Selection rank must be non-negative."
            )
        if not reference_id or not chromosome:
            raise GenomicSplitError(
                "Reference and chromosome identifiers must not be empty."
            )
        if len(normalized) != end - start:
            raise GenomicSplitError(
                "Genomic sequence length does not match its coordinate span."
            )
        if split not in SPLIT_NAMES:
            raise GenomicSplitError("Unknown genomic split: {0}".format(split))

        locus_content = {
            "schema_version": RECORD_ID_SCHEMA_VERSION,
            "reference_id": reference_id,
            "chromosome": chromosome,
            "start": start,
            "end": end,
            "strand": "+",
        }
        sequence_sha256 = hashlib.sha256(
            normalized.encode("ascii")
        ).hexdigest()
        canonical = reverse_complement_canonical_sequence(normalized)
        canonical_sha256 = hashlib.sha256(
            canonical.encode("ascii")
        ).hexdigest()
        return cls(
            record_id=hash_logical_content(locus_content),
            reference_id=reference_id,
            chromosome=chromosome,
            start=start,
            end=end,
            strand="+",
            sequence=normalized,
            split=split,
            sequence_sha256=sequence_sha256,
            rc_canonical_sha256=canonical_sha256,
            block_id="",
            selection_rank=selection_rank,
            split_policy_version=split_policy_version,
        )

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, str],
    ) -> "GenomicWindowRecord":
        """Parse and validate one TSV row."""

        record = cls(
            record_id=row["record_id"],
            reference_id=row["reference_id"],
            chromosome=row["chromosome"],
            start=int(row["start"]),
            end=int(row["end"]),
            strand=row["strand"],
            sequence=row["sequence"],
            split=row["split"],
            sequence_sha256=row["sequence_sha256"],
            rc_canonical_sha256=row["rc_canonical_sha256"],
            block_id=row["block_id"],
            selection_rank=int(row["selection_rank"]),
            split_policy_version=row["split_policy_version"],
        )
        expected = cls.create(
            reference_id=record.reference_id,
            chromosome=record.chromosome,
            start=record.start,
            end=record.end,
            sequence=record.sequence,
            split=record.split,
            selection_rank=record.selection_rank,
            split_policy_version=record.split_policy_version,
        )
        if record.to_row() != expected.to_row():
            raise GenomicSplitError(
                "Genomic record metadata or integrity hash mismatch."
            )
        return record

    @property
    def locus_key(self) -> Tuple[str, str, int, int]:
        """Return a strand-independent genomic-locus key."""

        return (
            self.reference_id,
            self.chromosome,
            self.start,
            self.end,
        )

    @property
    def canonical_sequence(self) -> str:
        """Return the exact strand-independent sequence key."""

        return reverse_complement_canonical_sequence(self.sequence)

    def to_row(self) -> Dict[str, str]:
        """Return fixed-order TSV-compatible values."""

        return {
            "record_id": self.record_id,
            "reference_id": self.reference_id,
            "chromosome": self.chromosome,
            "start": str(self.start),
            "end": str(self.end),
            "strand": self.strand,
            "sequence": self.sequence,
            "split": self.split,
            "sequence_sha256": self.sequence_sha256,
            "rc_canonical_sha256": self.rc_canonical_sha256,
            "block_id": self.block_id,
            "selection_rank": str(self.selection_rank),
            "split_policy_version": self.split_policy_version,
        }

    def logical_dict(self) -> Dict[str, Any]:
        """Return typed logical record content for deterministic hashing."""

        return {
            "record_id": self.record_id,
            "reference_id": self.reference_id,
            "chromosome": self.chromosome,
            "start": self.start,
            "end": self.end,
            "strand": self.strand,
            "sequence": self.sequence,
            "split": self.split,
            "sequence_sha256": self.sequence_sha256,
            "rc_canonical_sha256": self.rc_canonical_sha256,
            "block_id": self.block_id,
            "selection_rank": self.selection_rank,
            "split_policy_version": self.split_policy_version,
        }


@dataclass(frozen=True)
class RejectedGenomicRecord:
    """One candidate rejected by the cross-split equivalence policy."""

    record: GenomicWindowRecord
    rejection_reason: str

    def to_row(self) -> Dict[str, str]:
        """Return fixed-order rejection-table values."""

        row = self.record.to_row()
        row["canonical_sequence_sha256"] = (
            self.record.rc_canonical_sha256
        )
        row["rejection_reason"] = self.rejection_reason
        return row


@dataclass(frozen=True)
class WholeChromosomeSplitConfig:
    """Strict configuration for the approved whole-chromosome policy."""

    schema_version: str
    reference_path: str
    expected_raw_sha256: str
    reference_identifier: str
    policy_id: str
    eligible_chromosomes: Tuple[str, ...]
    training_chromosomes: Tuple[str, ...]
    validation_chromosomes: Tuple[str, ...]
    test_chromosomes: Tuple[str, ...]
    window_length: int
    sampling_seed: int
    training_target_count: int
    validation_target_count: int
    test_target_count: int
    maximum_candidate_multiplier: int
    strand: str
    output_directory: str
    training_records_path: str
    validation_records_path: str
    test_records_path: str
    training_sequences_path: str
    validation_sequences_path: str
    test_sequences_path: str
    rejection_records_path: str
    child_manifest_path: str
    genomic_manifest_path: str

    @classmethod
    def from_yaml(cls, path: str) -> "WholeChromosomeSplitConfig":
        """Load a strict configuration without applying side effects."""

        with open(path, "r", encoding="utf-8") as config_file:
            payload = yaml.safe_load(config_file)
        if not isinstance(payload, dict):
            raise GenomicSplitError("Genomic split config must be a mapping.")
        _require_exact_keys(
            payload,
            ("schema_version", "reference", "policy", "outputs"),
            "config",
        )
        reference = _require_mapping(payload["reference"], "reference")
        policy = _require_mapping(payload["policy"], "policy")
        outputs = _require_mapping(payload["outputs"], "outputs")
        _require_exact_keys(
            reference,
            ("path", "expected_raw_sha256", "identifier"),
            "reference",
        )
        _require_exact_keys(
            policy,
            (
                "id",
                "eligible_chromosomes",
                "chromosomes",
                "window_length",
                "sampling_seed",
                "target_counts",
                "maximum_candidate_multiplier",
                "strand",
            ),
            "policy",
        )
        chromosomes = _require_mapping(
            policy["chromosomes"],
            "policy.chromosomes",
        )
        target_counts = _require_mapping(
            policy["target_counts"],
            "policy.target_counts",
        )
        _require_exact_keys(chromosomes, SPLIT_NAMES, "policy.chromosomes")
        _require_exact_keys(target_counts, SPLIT_NAMES, "policy.target_counts")
        _require_exact_keys(
            outputs,
            (
                "directory",
                "records",
                "sequences",
                "rejections",
                "child_manifest",
                "genomic_manifest",
            ),
            "outputs",
        )
        record_paths = _require_mapping(outputs["records"], "outputs.records")
        sequence_paths = _require_mapping(
            outputs["sequences"],
            "outputs.sequences",
        )
        _require_exact_keys(record_paths, SPLIT_NAMES, "outputs.records")
        _require_exact_keys(sequence_paths, SPLIT_NAMES, "outputs.sequences")

        config = cls(
            schema_version=str(payload["schema_version"]),
            reference_path=str(reference["path"]),
            expected_raw_sha256=str(reference["expected_raw_sha256"]),
            reference_identifier=str(reference["identifier"]),
            policy_id=str(policy["id"]),
            eligible_chromosomes=_string_tuple(
                policy["eligible_chromosomes"],
                "policy.eligible_chromosomes",
            ),
            training_chromosomes=_string_tuple(
                chromosomes["training"],
                "policy.chromosomes.training",
            ),
            validation_chromosomes=_string_tuple(
                chromosomes["validation"],
                "policy.chromosomes.validation",
            ),
            test_chromosomes=_string_tuple(
                chromosomes["test"],
                "policy.chromosomes.test",
            ),
            window_length=_require_integer(
                policy["window_length"],
                "policy.window_length",
            ),
            sampling_seed=_require_integer(
                policy["sampling_seed"],
                "policy.sampling_seed",
            ),
            training_target_count=_require_integer(
                target_counts["training"],
                "policy.target_counts.training",
            ),
            validation_target_count=_require_integer(
                target_counts["validation"],
                "policy.target_counts.validation",
            ),
            test_target_count=_require_integer(
                target_counts["test"],
                "policy.target_counts.test",
            ),
            maximum_candidate_multiplier=_require_integer(
                policy["maximum_candidate_multiplier"],
                "policy.maximum_candidate_multiplier",
            ),
            strand=str(policy["strand"]),
            output_directory=str(outputs["directory"]),
            training_records_path=str(record_paths["training"]),
            validation_records_path=str(record_paths["validation"]),
            test_records_path=str(record_paths["test"]),
            training_sequences_path=str(sequence_paths["training"]),
            validation_sequences_path=str(sequence_paths["validation"]),
            test_sequences_path=str(sequence_paths["test"]),
            rejection_records_path=str(outputs["rejections"]),
            child_manifest_path=str(outputs["child_manifest"]),
            genomic_manifest_path=str(outputs["genomic_manifest"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate scientific and output-layout invariants."""

        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise GenomicSplitError(
                "Unsupported genomic split config schema: {0}".format(
                    self.schema_version
                )
            )
        integer_fields = (
            ("policy.window_length", self.window_length),
            ("policy.sampling_seed", self.sampling_seed),
            (
                "policy.target_counts.training",
                self.training_target_count,
            ),
            (
                "policy.target_counts.validation",
                self.validation_target_count,
            ),
            ("policy.target_counts.test", self.test_target_count),
            (
                "policy.maximum_candidate_multiplier",
                self.maximum_candidate_multiplier,
            ),
        )
        for label, value in integer_fields:
            _require_integer(value, label)
        if self.window_length <= 0:
            raise GenomicSplitError("Window length must be positive.")
        if self.maximum_candidate_multiplier < 1:
            raise GenomicSplitError(
                "Maximum candidate multiplier must be at least one."
            )
        if self.strand != "+":
            raise GenomicSplitError(
                "Whole-chromosome policy v1 supports '+' records only."
            )
        if not self.reference_identifier:
            raise GenomicSplitError("Reference identifier must not be empty.")
        if len(self.expected_raw_sha256) != 64:
            raise GenomicSplitError(
                "Expected reference SHA-256 must contain 64 hexadecimal digits."
            )
        try:
            int(self.expected_raw_sha256, 16)
        except ValueError as error:
            raise GenomicSplitError(
                "Expected reference SHA-256 is not hexadecimal."
            ) from error

        eligible = self.eligible_chromosomes
        if not eligible or len(set(eligible)) != len(eligible):
            raise GenomicSplitError(
                "Eligible chromosome identifiers must be unique and non-empty."
            )
        assigned = (
            self.training_chromosomes
            + self.validation_chromosomes
            + self.test_chromosomes
        )
        if len(set(assigned)) != len(assigned):
            raise GenomicSplitError(
                "Chromosome assignments must be pairwise disjoint."
            )
        if set(assigned) != set(eligible):
            raise GenomicSplitError(
                "Every eligible chromosome must have exactly one split."
            )
        for split in SPLIT_NAMES:
            if not self.chromosomes_for_split(split):
                raise GenomicSplitError(
                    "Every split must contain at least one chromosome."
                )
            if self.target_count(split) <= 0:
                raise GenomicSplitError(
                    "Every split target count must be positive."
                )

        all_paths = (
            self.reference_path,
            self.output_directory,
            self.training_records_path,
            self.validation_records_path,
            self.test_records_path,
            self.training_sequences_path,
            self.validation_sequences_path,
            self.test_sequences_path,
            self.rejection_records_path,
            self.child_manifest_path,
            self.genomic_manifest_path,
        )
        for configured_path in all_paths:
            if Path(configured_path).is_absolute():
                raise GenomicSplitError(
                    "Configured paths must be repository-relative: {0}".format(
                        configured_path
                    )
                )
        output_paths = all_paths[2:]
        if len(set(output_paths)) != len(output_paths):
            raise GenomicSplitError("Configured output paths must be unique.")
        output_directory = Path(self.output_directory)
        for output_path in output_paths:
            try:
                Path(output_path).relative_to(output_directory)
            except ValueError as error:
                raise GenomicSplitError(
                    "Every production output must be inside output directory."
                ) from error
        self._validate_policy_contract()

    def _validate_policy_contract(self) -> None:
        """Require the exact approved hg38 whole-chromosome v1 contract."""

        expected_fields = (
            ("policy.id", self.policy_id, POLICY_ID),
            (
                "reference.path",
                self.reference_path,
                APPROVED_REFERENCE_PATH,
            ),
            (
                "reference.expected_raw_sha256",
                self.expected_raw_sha256,
                APPROVED_REFERENCE_SHA256,
            ),
            (
                "reference.identifier",
                self.reference_identifier,
                APPROVED_REFERENCE_IDENTIFIER,
            ),
            (
                "policy.eligible_chromosomes",
                self.eligible_chromosomes,
                APPROVED_ELIGIBLE_CHROMOSOMES,
            ),
            (
                "policy.chromosomes.training",
                self.training_chromosomes,
                APPROVED_TRAINING_CHROMOSOMES,
            ),
            (
                "policy.chromosomes.validation",
                self.validation_chromosomes,
                APPROVED_VALIDATION_CHROMOSOMES,
            ),
            (
                "policy.chromosomes.test",
                self.test_chromosomes,
                APPROVED_TEST_CHROMOSOMES,
            ),
            (
                "policy.window_length",
                self.window_length,
                APPROVED_WINDOW_LENGTH,
            ),
            (
                "policy.sampling_seed",
                self.sampling_seed,
                APPROVED_SAMPLING_SEED,
            ),
            (
                "policy.target_counts.training",
                self.training_target_count,
                APPROVED_TRAINING_TARGET_COUNT,
            ),
            (
                "policy.target_counts.validation",
                self.validation_target_count,
                APPROVED_VALIDATION_TARGET_COUNT,
            ),
            (
                "policy.target_counts.test",
                self.test_target_count,
                APPROVED_TEST_TARGET_COUNT,
            ),
            (
                "policy.maximum_candidate_multiplier",
                self.maximum_candidate_multiplier,
                APPROVED_MAXIMUM_CANDIDATE_MULTIPLIER,
            ),
            ("policy.strand", self.strand, "+"),
            (
                "outputs.directory",
                self.output_directory,
                APPROVED_OUTPUT_DIRECTORY,
            ),
            (
                "outputs.records.training",
                self.training_records_path,
                APPROVED_TRAINING_RECORDS_PATH,
            ),
            (
                "outputs.records.validation",
                self.validation_records_path,
                APPROVED_VALIDATION_RECORDS_PATH,
            ),
            (
                "outputs.records.test",
                self.test_records_path,
                APPROVED_TEST_RECORDS_PATH,
            ),
            (
                "outputs.sequences.training",
                self.training_sequences_path,
                APPROVED_TRAINING_SEQUENCES_PATH,
            ),
            (
                "outputs.sequences.validation",
                self.validation_sequences_path,
                APPROVED_VALIDATION_SEQUENCES_PATH,
            ),
            (
                "outputs.sequences.test",
                self.test_sequences_path,
                APPROVED_TEST_SEQUENCES_PATH,
            ),
            (
                "outputs.rejections",
                self.rejection_records_path,
                APPROVED_REJECTION_RECORDS_PATH,
            ),
            (
                "outputs.child_manifest",
                self.child_manifest_path,
                APPROVED_CHILD_MANIFEST_PATH,
            ),
            (
                "outputs.genomic_manifest",
                self.genomic_manifest_path,
                APPROVED_GENOMIC_MANIFEST_PATH,
            ),
        )
        for field_name, observed_value, expected_value in expected_fields:
            if observed_value != expected_value:
                message = (
                    "Approved hg38 whole-chromosome v1 contract mismatch for "
                    "{0}: expected {1!r}, observed {2!r}."
                )
                raise GenomicSplitError(
                    message.format(
                        field_name,
                        expected_value,
                        observed_value,
                    )
                )

    def chromosomes_for_split(self, split: str) -> Tuple[str, ...]:
        """Return the configured chromosome tuple for one split."""

        if split == "training":
            return self.training_chromosomes
        if split == "validation":
            return self.validation_chromosomes
        if split == "test":
            return self.test_chromosomes
        raise GenomicSplitError("Unknown genomic split: {0}".format(split))

    def split_for_chromosome(self, chromosome: str) -> str:
        """Return the split assigned before any sequence is sampled."""

        for split in SPLIT_NAMES:
            if chromosome in self.chromosomes_for_split(split):
                return split
        raise GenomicSplitError(
            "Chromosome is not eligible for this policy: {0}".format(
                chromosome
            )
        )

    def target_count(self, split: str) -> int:
        """Return the requested record count for one split."""

        if split == "training":
            return self.training_target_count
        if split == "validation":
            return self.validation_target_count
        if split == "test":
            return self.test_target_count
        raise GenomicSplitError("Unknown genomic split: {0}".format(split))

    def records_path(self, split: str) -> str:
        """Return one configured coordinate-record path."""

        if split == "training":
            return self.training_records_path
        if split == "validation":
            return self.validation_records_path
        if split == "test":
            return self.test_records_path
        raise GenomicSplitError("Unknown genomic split: {0}".format(split))

    def sequences_path(self, split: str) -> str:
        """Return one configured sequence-projection path."""

        if split == "training":
            return self.training_sequences_path
        if split == "validation":
            return self.validation_sequences_path
        if split == "test":
            return self.test_sequences_path
        raise GenomicSplitError("Unknown genomic split: {0}".format(split))

    def output_paths(self) -> Tuple[str, ...]:
        """Return every production output file in deterministic order."""

        paths = []
        for split in SPLIT_NAMES:
            paths.append(self.records_path(split))
        for split in SPLIT_NAMES:
            paths.append(self.sequences_path(split))
        paths.extend(
            (
                self.rejection_records_path,
                self.child_manifest_path,
                self.genomic_manifest_path,
            )
        )
        return tuple(paths)


def scan_fasta_reference(
    config: WholeChromosomeSplitConfig,
    repository_root: str,
) -> ReferenceScan:
    """Fingerprint a FASTA and enumerate valid starts one contig at a time."""

    config.validate()
    reference_path = _resolve_repository_path(
        repository_root,
        config.reference_path,
    )
    if not reference_path.is_file():
        raise FileNotFoundError(
            "Reference FASTA does not exist: {0}".format(reference_path)
        )
    initial_stat = reference_path.stat()
    raw_file_sha256 = hash_file_bytes(str(reference_path))
    if raw_file_sha256 != config.expected_raw_sha256:
        message = "Reference FASTA SHA-256 mismatch: expected {0}, observed {1}."
        raise GenomicSplitError(
            message.format(config.expected_raw_sha256, raw_file_sha256)
        )

    eligible_set = set(config.eligible_chromosomes)
    contigs = []
    excluded_identifiers = []
    seen_identifiers = set()
    intervals_by_chromosome: Dict[str, list[EligibleStartInterval]] = {}
    for chromosome in config.eligible_chromosomes:
        intervals_by_chromosome[chromosome] = []

    with open(reference_path, "r", encoding="ascii", newline=None) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            identifier = str(record.id)
            if not identifier:
                raise GenomicSplitError("FASTA contig identifier is empty.")
            if identifier in seen_identifiers:
                raise GenomicSplitError(
                    "Duplicate FASTA contig identifier: {0}".format(identifier)
                )
            seen_identifiers.add(identifier)
            source_sequence = str(record.seq)
            normalized_sequence = source_sequence.upper()
            try:
                sequence_bytes = normalized_sequence.encode("ascii")
            except UnicodeEncodeError as error:
                raise GenomicSplitError(
                    "FASTA sequences must contain ASCII symbols."
                ) from error

            length = len(normalized_sequence)
            sequence_sha256 = hashlib.sha256(sequence_bytes).hexdigest()
            is_eligible = identifier in eligible_set
            split = None
            acgt_count = 0
            n_count = 0
            other_count = 0
            lowercase_count = 0
            total_starts = 0
            eligible_starts = 0
            invalid_starts = 0
            if is_eligible:
                split = config.split_for_chromosome(identifier)
                acgt_count = 0
                for base in "ACGT":
                    acgt_count += normalized_sequence.count(base)
                n_count = normalized_sequence.count("N")
                other_count = length - acgt_count - n_count
                for base in "acgtn":
                    lowercase_count += source_sequence.count(base)
                total_starts = max(length - config.window_length + 1, 0)

                for match in VALID_BASE_PATTERN.finditer(normalized_sequence):
                    run_length = match.end() - match.start()
                    if run_length >= config.window_length:
                        start = match.start()
                        stop = match.end() - config.window_length + 1
                        interval = EligibleStartInterval(
                            chromosome=identifier,
                            start=start,
                            stop=stop,
                            split=split,
                        )
                        intervals_by_chromosome[identifier].append(interval)
                        eligible_starts += interval.capacity
                invalid_starts = total_starts - eligible_starts
            else:
                excluded_identifiers.append(identifier)

            contigs.append(
                ReferenceContigSummary(
                    identifier=identifier,
                    length=length,
                    sequence_sha256=sequence_sha256,
                    eligible=is_eligible,
                    split=split,
                    acgt_base_count=acgt_count,
                    n_base_count=n_count,
                    other_symbol_count=other_count,
                    lowercase_base_count=lowercase_count,
                    total_possible_window_starts=total_starts,
                    eligible_window_start_count=eligible_starts,
                    invalid_window_start_count=invalid_starts,
                )
            )

    missing = []
    for chromosome in config.eligible_chromosomes:
        if chromosome not in seen_identifiers:
            missing.append(chromosome)
    if missing:
        raise GenomicSplitError(
            "Eligible FASTA contigs are missing: {0}".format(
                ", ".join(missing)
            )
        )
    if not contigs:
        raise GenomicSplitError("Reference FASTA contains no records.")

    final_stat = reference_path.stat()
    if (
        final_stat.st_size != initial_stat.st_size
        or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
    ):
        raise GenomicSplitError(
            "Reference FASTA changed while it was being scanned."
        )

    logical_payload = {
        "schema_version": LOGICAL_REFERENCE_SCHEMA_VERSION,
        "contigs": [
            contig.logical_identity_dict()
            for contig in contigs
        ],
    }
    ordered_intervals = []
    for chromosome in config.eligible_chromosomes:
        chromosome_intervals = intervals_by_chromosome[chromosome]
        chromosome_intervals.sort(key=lambda value: value.start)
        ordered_intervals.extend(chromosome_intervals)

    return ReferenceScan(
        reference_path=repository_relative_source_path(
            str(reference_path),
            repository_root,
        ),
        source_file_size_bytes=initial_stat.st_size,
        source_file_mtime_ns=initial_stat.st_mtime_ns,
        raw_file_sha256=raw_file_sha256,
        logical_reference_sha256=hash_logical_content(logical_payload),
        contigs=tuple(contigs),
        eligible_intervals=tuple(ordered_intervals),
        excluded_contig_identifiers=tuple(excluded_identifiers),
    )


def build_dry_run_report(
    config: WholeChromosomeSplitConfig,
    scan: ReferenceScan,
) -> Dict[str, Any]:
    """Build a deterministic no-write capacity and exclusion report."""

    chromosome_summaries = []
    contig_map = scan.contig_by_identifier()
    for chromosome in config.eligible_chromosomes:
        chromosome_summaries.append(contig_map[chromosome].report_dict())

    split_capacities = {}
    target_counts = {}
    capacity_is_sufficient = {}
    for split in SPLIT_NAMES:
        capacity = scan.split_capacity(split)
        target = config.target_count(split)
        split_capacities[split] = capacity
        target_counts[split] = target
        capacity_is_sufficient[split] = capacity >= target

    eligible_contigs = [
        contig for contig in scan.contigs if contig.eligible
    ]
    excluded_contigs = [
        contig for contig in scan.contigs if not contig.eligible
    ]
    return {
        "schema_version": DRY_RUN_SCHEMA_VERSION,
        "policy_id": config.policy_id,
        "reference": {
            "source_path": scan.reference_path,
            "source_file_size_bytes": scan.source_file_size_bytes,
            "raw_file_sha256": scan.raw_file_sha256,
            "logical_reference_schema_version": (
                LOGICAL_REFERENCE_SCHEMA_VERSION
            ),
            "logical_reference_sha256": (
                scan.logical_reference_sha256
            ),
            "reference_identifier": config.reference_identifier,
            "total_contig_count": len(scan.contigs),
        },
        "coordinate_convention": "zero_based_half_open",
        "strand_convention": "reference_forward_only",
        "window_length": config.window_length,
        "sampling": {
            "method": (
                "uniform_without_replacement_over_eligible_start_ordinals"
            ),
            "seed": config.sampling_seed,
            "maximum_candidate_multiplier": (
                config.maximum_candidate_multiplier
            ),
            "tokenizer_independent": True,
        },
        "chromosome_assignments": {
            split: list(config.chromosomes_for_split(split))
            for split in SPLIT_NAMES
        },
        "target_counts": target_counts,
        "eligible_window_start_capacity": split_capacities,
        "capacity_is_sufficient": capacity_is_sufficient,
        "all_capacities_sufficient": all(
            capacity_is_sufficient.values()
        ),
        "eligible_chromosome_summaries": chromosome_summaries,
        "exclusions": {
            "excluded_contig_count": len(excluded_contigs),
            "excluded_contig_identifiers": list(
                scan.excluded_contig_identifiers
            ),
            "excluded_contig_base_count": sum(
                contig.length for contig in excluded_contigs
            ),
            "eligible_acgt_base_count": sum(
                contig.acgt_base_count for contig in eligible_contigs
            ),
            "eligible_n_base_count": sum(
                contig.n_base_count for contig in eligible_contigs
            ),
            "eligible_other_symbol_count": sum(
                contig.other_symbol_count for contig in eligible_contigs
            ),
            "eligible_lowercase_base_count": sum(
                contig.lowercase_base_count for contig in eligible_contigs
            ),
            "eligible_total_possible_window_starts": sum(
                contig.total_possible_window_starts
                for contig in eligible_contigs
            ),
            "invalid_window_start_count": sum(
                contig.invalid_window_start_count
                for contig in eligible_contigs
            ),
        },
        "production_outputs_created": False,
    }


class _Sha256RandomSource:
    """Version-stable SHA-256 counter stream with unbiased randbelow."""

    def __init__(self, domain: str) -> None:
        self.domain = domain.encode("utf-8")
        self.counter = 0

    def randbelow(self, upper_bound: int) -> int:
        """Return an unbiased deterministic integer below upper_bound."""

        if upper_bound <= 0:
            raise ValueError("randbelow upper bound must be positive.")
        full_range = 1 << 256
        acceptance_limit = full_range - (full_range % upper_bound)
        accepted_value = None
        while accepted_value is None:
            counter_bytes = self.counter.to_bytes(16, byteorder="big")
            digest = hashlib.sha256(
                self.domain + b"\0" + counter_bytes
            ).digest()
            self.counter += 1
            value = int.from_bytes(digest, byteorder="big")
            if value < acceptance_limit:
                accepted_value = value % upper_bound
        return accepted_value


def sample_candidate_ordinals(
    capacity: int,
    count: int,
    seed: int,
    namespace: str,
) -> Tuple[int, ...]:
    """Sample an ordered uniform prefix without replacement."""

    if capacity < 0 or count < 0:
        raise ValueError("Capacity and count must be non-negative.")
    if count > capacity:
        raise CandidateBudgetError(
            "Requested candidate count exceeds eligible capacity."
        )
    random_source = _Sha256RandomSource(
        "genomic_start_sampling.v1|{0}|{1}".format(seed, namespace)
    )
    virtual_swaps: Dict[int, int] = {}
    sampled = []
    for index in range(count):
        selected_index = index + random_source.randbelow(capacity - index)
        selected_value = virtual_swaps.get(selected_index, selected_index)
        index_value = virtual_swaps.get(index, index)
        virtual_swaps[selected_index] = index_value
        if index in virtual_swaps:
            del virtual_swaps[index]
        sampled.append(selected_value)
    return tuple(sampled)


def sample_candidate_coordinates(
    config: WholeChromosomeSplitConfig,
    scan: ReferenceScan,
) -> Mapping[str, Tuple[CandidateCoordinate, ...]]:
    """Sample deterministic proportional-capacity coordinate streams."""

    candidate_counts = {}
    for split in SPLIT_NAMES:
        capacity = scan.split_capacity(split)
        target = config.target_count(split)
        if capacity < target:
            message = (
                "Split {0} has capacity {1}, below target {2}."
            )
            raise CandidateBudgetError(
                message.format(split, capacity, target)
            )
        requested_budget = target * config.maximum_candidate_multiplier
        candidate_counts[split] = min(capacity, requested_budget)
    return _sample_candidate_coordinate_counts(
        config=config,
        scan=scan,
        candidate_counts=candidate_counts,
    )


def _sample_candidate_coordinate_counts(
    config: WholeChromosomeSplitConfig,
    scan: ReferenceScan,
    candidate_counts: Mapping[str, int],
) -> Mapping[str, Tuple[CandidateCoordinate, ...]]:
    """Regenerate deterministic candidate-stream prefixes by split."""

    _require_exact_keys(
        candidate_counts,
        SPLIT_NAMES,
        "candidate_counts",
    )
    sampled_by_split = {}
    for split in SPLIT_NAMES:
        intervals = scan.intervals_for_split(split)
        capacity = scan.split_capacity(split)
        target = config.target_count(split)
        if capacity < target:
            message = (
                "Split {0} has capacity {1}, below target {2}."
            )
            raise CandidateBudgetError(
                message.format(split, capacity, target)
            )
        requested_budget = target * config.maximum_candidate_multiplier
        maximum_count = min(capacity, requested_budget)
        candidate_count = _require_nonnegative_integer(
            candidate_counts[split],
            "{0} candidate count".format(split),
        )
        if candidate_count > maximum_count:
            raise CandidateBudgetError(
                "Requested candidate prefix exceeds the configured budget."
            )
        namespace = "{0}|{1}|{2}|{3}".format(
            config.policy_id,
            scan.logical_reference_sha256,
            split,
            config.window_length,
        )
        ordinals = sample_candidate_ordinals(
            capacity=capacity,
            count=candidate_count,
            seed=config.sampling_seed,
            namespace=namespace,
        )
        cumulative_ends = []
        cumulative = 0
        for interval in intervals:
            cumulative += interval.capacity
            cumulative_ends.append(cumulative)

        candidates = []
        for selection_rank, ordinal in enumerate(ordinals):
            interval_index = bisect_right(cumulative_ends, ordinal)
            interval = intervals[interval_index]
            previous_end = 0
            if interval_index > 0:
                previous_end = cumulative_ends[interval_index - 1]
            start = interval.start + (ordinal - previous_end)
            candidates.append(
                CandidateCoordinate(
                    chromosome=interval.chromosome,
                    start=start,
                    end=start + config.window_length,
                    split=split,
                    selection_rank=selection_rank,
                )
            )
        sampled_by_split[split] = tuple(candidates)
    return sampled_by_split


def extract_candidate_records(
    config: WholeChromosomeSplitConfig,
    scan: ReferenceScan,
    repository_root: str,
    candidates_by_split: Mapping[
        str,
        Sequence[CandidateCoordinate],
    ],
) -> Mapping[str, Tuple[GenomicWindowRecord, ...]]:
    """Extract sampled windows during a one-contig-at-a-time FASTA pass."""

    by_chromosome: Dict[str, list[CandidateCoordinate]] = {}
    for split in SPLIT_NAMES:
        for candidate in candidates_by_split[split]:
            if candidate.split != split:
                raise GenomicSplitError("Candidate split metadata mismatch.")
            by_chromosome.setdefault(candidate.chromosome, []).append(candidate)
    for candidates in by_chromosome.values():
        candidates.sort(key=lambda value: (value.start, value.selection_rank))

    reference_id = "{0}:sha256:{1}".format(
        config.reference_identifier,
        scan.logical_reference_sha256,
    )
    extracted: Dict[str, list[GenomicWindowRecord]] = {
        split: [] for split in SPLIT_NAMES
    }
    reference_path = _resolve_repository_path(
        repository_root,
        config.reference_path,
    )
    expected_contigs = scan.contigs
    observed_count = 0
    with open(reference_path, "r", encoding="ascii", newline=None) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if observed_count >= len(expected_contigs):
                raise GenomicSplitError(
                    "Reference FASTA gained contigs after its initial scan."
                )
            expected = expected_contigs[observed_count]
            observed_count += 1
            identifier = str(record.id)
            sequence = str(record.seq).upper()
            sequence_sha256 = hashlib.sha256(
                sequence.encode("ascii")
            ).hexdigest()
            if (
                identifier != expected.identifier
                or len(sequence) != expected.length
                or sequence_sha256 != expected.sequence_sha256
            ):
                raise GenomicSplitError(
                    "Reference FASTA changed between scan and extraction."
                )

            for candidate in by_chromosome.get(identifier, ()):
                window = sequence[candidate.start:candidate.end]
                if len(window) != config.window_length:
                    raise GenomicSplitError(
                        "Sampled coordinate extends beyond its chromosome."
                    )
                if VALID_SEQUENCE_PATTERN.fullmatch(window) is None:
                    raise GenomicSplitError(
                        "Sampled coordinate unexpectedly contains invalid bases."
                    )
                extracted[candidate.split].append(
                    GenomicWindowRecord.create(
                        reference_id=reference_id,
                        chromosome=identifier,
                        start=candidate.start,
                        end=candidate.end,
                        sequence=window,
                        split=candidate.split,
                        selection_rank=candidate.selection_rank,
                        split_policy_version=config.policy_id,
                    )
                )
    if observed_count != len(expected_contigs):
        raise GenomicSplitError(
            "Reference FASTA lost contigs after its initial scan."
        )
    final_stat = reference_path.stat()
    if (
        final_stat.st_size != scan.source_file_size_bytes
        or final_stat.st_mtime_ns != scan.source_file_mtime_ns
    ):
        raise GenomicSplitError(
            "Reference FASTA changed between scan and extraction."
        )

    final_records = {}
    for split in SPLIT_NAMES:
        records = extracted[split]
        records.sort(key=lambda value: value.selection_rank)
        expected_count = len(candidates_by_split[split])
        if len(records) != expected_count:
            message = "Failed to extract every {0} candidate: {1} of {2}."
            raise GenomicSplitError(
                message.format(split, len(records), expected_count)
            )
        _ensure_unique_loci(records)
        final_records[split] = tuple(records)
    return final_records


def resolve_cross_split_equivalence(
    candidate_records: Mapping[
        str,
        Sequence[GenomicWindowRecord],
    ],
    target_counts: Mapping[str, int],
) -> Tuple[
    Mapping[str, Tuple[GenomicWindowRecord, ...]],
    Tuple[RejectedGenomicRecord, ...],
    Mapping[str, Any],
]:
    """Reject and refill every cross-split exact or RC-equivalent group."""

    selected: Dict[str, list[GenomicWindowRecord]] = {
        split: [] for split in SPLIT_NAMES
    }
    next_candidate_index = {split: 0 for split in SPLIT_NAMES}
    blacklisted_keys = set()
    rejected = []
    rejected_loci = set()
    conflict_keys = set()
    exact_conflict_keys = set()
    rc_only_conflict_keys = set()

    for split in SPLIT_NAMES:
        _refill_selected_records(
            split=split,
            selected=selected,
            candidate_records=candidate_records,
            next_candidate_index=next_candidate_index,
            target_count=target_counts[split],
            blacklisted_keys=blacklisted_keys,
            rejected=rejected,
            rejected_loci=rejected_loci,
        )

    iteration = 0
    maximum_iterations = sum(
        len(candidate_records[split]) for split in SPLIT_NAMES
    ) + 1
    conflicts_remain = True
    while conflicts_remain and iteration < maximum_iterations:
        iteration += 1
        grouped = _group_selected_by_canonical_sequence(selected)
        conflicting_groups = {}
        for canonical_key in sorted(grouped):
            group_records = grouped[canonical_key]
            group_splits = {record.split for record in group_records}
            if len(group_splits) > 1:
                conflicting_groups[canonical_key] = group_records
        if not conflicting_groups:
            conflicts_remain = False
        else:
            for canonical_key in sorted(conflicting_groups):
                group_records = conflicting_groups[canonical_key]
                conflict_keys.add(canonical_key)
                is_exact = _group_has_cross_split_exact_match(group_records)
                if is_exact:
                    exact_conflict_keys.add(canonical_key)
                    reason = "cross_split_exact_sequence_group"
                else:
                    rc_only_conflict_keys.add(canonical_key)
                    reason = "cross_split_reverse_complement_only_group"
                blacklisted_keys.add(canonical_key)
                for record in group_records:
                    if record.locus_key not in rejected_loci:
                        rejected.append(
                            RejectedGenomicRecord(
                                record=record,
                                rejection_reason=reason,
                            )
                        )
                        rejected_loci.add(record.locus_key)

            for split in SPLIT_NAMES:
                retained = []
                for record in selected[split]:
                    if record.canonical_sequence not in blacklisted_keys:
                        retained.append(record)
                selected[split] = retained

            for split in SPLIT_NAMES:
                _refill_selected_records(
                    split=split,
                    selected=selected,
                    candidate_records=candidate_records,
                    next_candidate_index=next_candidate_index,
                    target_count=target_counts[split],
                    blacklisted_keys=blacklisted_keys,
                    rejected=rejected,
                    rejected_loci=rejected_loci,
                )

    if conflicts_remain:
        raise CandidateBudgetError(
            "Cross-split equivalence resolution did not converge."
        )

    final_selected = {}
    for split in SPLIT_NAMES:
        records = selected[split]
        records.sort(key=lambda value: value.selection_rank)
        if len(records) != target_counts[split]:
            raise CandidateBudgetError(
                "Leakage-free target was not reached for {0}.".format(split)
            )
        _ensure_unique_loci(records)
        final_selected[split] = tuple(records)

    rejected.sort(
        key=lambda value: (
            SPLIT_NAMES.index(value.record.split),
            value.record.selection_rank,
        )
    )
    statistics = {
        "resolution_iterations": iteration,
        "unique_cross_split_equivalence_groups_rejected": len(conflict_keys),
        "unique_exact_groups_rejected": len(exact_conflict_keys),
        "unique_rc_only_groups_rejected": len(rc_only_conflict_keys),
        "rejected_candidate_record_count": len(rejected),
        "consumed_candidate_counts": {
            split: next_candidate_index[split] for split in SPLIT_NAMES
        },
        "maximum_candidate_counts": {
            split: len(candidate_records[split]) for split in SPLIT_NAMES
        },
    }
    return final_selected, tuple(rejected), statistics


def audit_genomic_records(
    records_by_split: Mapping[
        str,
        Sequence[GenomicWindowRecord],
    ],
    config: Optional[WholeChromosomeSplitConfig] = None,
) -> Dict[str, Any]:
    """Audit exact, RC, locus, and interval leakage across all split pairs."""

    for split in SPLIT_NAMES:
        if split not in records_by_split:
            raise GenomicSplitError(
                "Missing records for split: {0}".format(split)
            )
        records = records_by_split[split]
        _ensure_unique_loci(records)
        if config is not None and len(records) != config.target_count(split):
            message = "Record count for {0} is {1}; expected {2}."
            raise GenomicSplitError(
                message.format(
                    split,
                    len(records),
                    config.target_count(split),
                )
            )
        previous_rank = -1
        for record in records:
            if record.split != split:
                raise GenomicSplitError("Record split metadata mismatch.")
            if record.selection_rank <= previous_rank:
                raise GenomicSplitError(
                    "Records must be ordered by increasing selection rank."
                )
            previous_rank = record.selection_rank
            if config is not None:
                if record.split_policy_version != config.policy_id:
                    raise GenomicSplitError(
                        "Record split-policy version mismatch."
                    )
                if record.chromosome not in config.chromosomes_for_split(split):
                    raise GenomicSplitError(
                        "Record chromosome violates whole-chromosome assignment."
                    )
                if record.end - record.start != config.window_length:
                    raise GenomicSplitError("Record window length mismatch.")
                if record.block_id:
                    raise GenomicSplitError(
                        "Whole-chromosome records must have empty block_id."
                    )

    pairwise = {}
    strict_pass = True
    pair_names = (
        ("training", "validation"),
        ("training", "test"),
        ("validation", "test"),
    )
    for first_split, second_split in pair_names:
        audit = _audit_pair(
            records_by_split[first_split],
            records_by_split[second_split],
        )
        key = "{0}_vs_{1}".format(first_split, second_split)
        pairwise[key] = audit
        if (
            audit["exact_sequence_overlap_group_count"] > 0
            or audit["reverse_complement_equivalent_group_count"] > 0
            or audit["same_locus_count"] > 0
            or audit["interval_overlap_pair_count"] > 0
        ):
            strict_pass = False

    within_split = {}
    for split in SPLIT_NAMES:
        within_split[split] = _within_split_repeat_summary(
            records_by_split[split]
        )

    return {
        "schema_version": GENOMIC_AUDIT_SCHEMA_VERSION,
        "pairwise": pairwise,
        "within_split_repeated_sequences": within_split,
        "strict_pass": strict_pass,
    }


def validate_records_against_reference(
    records_by_split: Mapping[
        str,
        Sequence[GenomicWindowRecord],
    ],
    config: WholeChromosomeSplitConfig,
    scan: ReferenceScan,
    repository_root: str,
) -> None:
    """Verify that every stored record equals its hg38 reference interval."""

    records_by_chromosome: Dict[str, list[GenomicWindowRecord]] = {}
    expected_reference_id = "{0}:sha256:{1}".format(
        config.reference_identifier,
        scan.logical_reference_sha256,
    )
    for split in SPLIT_NAMES:
        for record in records_by_split[split]:
            if record.reference_id != expected_reference_id:
                raise GenomicSplitAuditError(
                    "Record reference identifier does not match the FASTA."
                )
            records_by_chromosome.setdefault(
                record.chromosome,
                [],
            ).append(record)
    for records in records_by_chromosome.values():
        records.sort(key=lambda value: (value.start, value.end))

    reference_path = _resolve_repository_path(
        repository_root,
        config.reference_path,
    )
    initial_stat = reference_path.stat()
    if (
        initial_stat.st_size != scan.source_file_size_bytes
        or initial_stat.st_mtime_ns != scan.source_file_mtime_ns
    ):
        raise GenomicSplitAuditError(
            "Reference FASTA changed between scan and audit validation."
        )
    expected_contigs = scan.contigs
    observed_contig_count = 0
    matched_record_count = 0
    with open(reference_path, "r", encoding="ascii", newline=None) as handle:
        for fasta_record in SeqIO.parse(handle, "fasta"):
            if observed_contig_count >= len(expected_contigs):
                raise GenomicSplitAuditError(
                    "Reference FASTA gained contigs after its audit scan."
                )
            expected_contig = expected_contigs[observed_contig_count]
            observed_contig_count += 1
            identifier = str(fasta_record.id)
            sequence = str(fasta_record.seq).upper()
            sequence_sha256 = hashlib.sha256(
                sequence.encode("ascii")
            ).hexdigest()
            if (
                identifier != expected_contig.identifier
                or len(sequence) != expected_contig.length
                or sequence_sha256 != expected_contig.sequence_sha256
            ):
                raise GenomicSplitAuditError(
                    "Reference FASTA changed between scan and audit validation."
                )
            for record in records_by_chromosome.get(identifier, ()):
                observed = sequence[record.start:record.end]
                if observed != record.sequence:
                    message = (
                        "Record {0} does not match reference interval "
                        "{1}:{2}-{3}."
                    )
                    raise GenomicSplitAuditError(
                        message.format(
                            record.record_id,
                            identifier,
                            record.start,
                            record.end,
                        )
                    )
                matched_record_count += 1
    if observed_contig_count != len(expected_contigs):
        raise GenomicSplitAuditError(
            "Reference FASTA lost contigs after its audit scan."
        )
    final_stat = reference_path.stat()
    if (
        final_stat.st_size != scan.source_file_size_bytes
        or final_stat.st_mtime_ns != scan.source_file_mtime_ns
    ):
        raise GenomicSplitAuditError(
            "Reference FASTA changed between scan and audit validation."
        )
    expected_record_count = sum(
        len(records_by_split[split]) for split in SPLIT_NAMES
    )
    if matched_record_count != expected_record_count:
        raise GenomicSplitAuditError(
            "Not every genomic record could be found in the reference."
        )


def read_genomic_records(path: str) -> Tuple[GenomicWindowRecord, ...]:
    """Read a fixed-schema coordinate TSV."""

    records = []
    with open(path, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if tuple(reader.fieldnames or ()) != RECORD_FIELD_NAMES:
            raise GenomicSplitError(
                "Coordinate TSV header does not match the required schema."
            )
        for row in reader:
            records.append(GenomicWindowRecord.from_row(row))
    _ensure_unique_loci(records)
    return tuple(records)


def validate_generated_artifacts(
    config: WholeChromosomeSplitConfig,
    repository_root: str,
    mode: str = "strict",
) -> Mapping[str, Any]:
    """Validate the complete generated dataset and linked v2 artifacts."""

    _validate_audit_mode(mode)
    config.validate()
    genomic_manifest_path = _resolve_repository_path(
        repository_root,
        config.genomic_manifest_path,
    )
    parent_manifest = load_genomic_manifest(str(genomic_manifest_path))
    if (
        parent_manifest["schema_version"]
        != GENOMIC_MANIFEST_SCHEMA_VERSION
    ):
        raise GenomicSplitAuditError(
            "Generated parent manifest schema mismatch."
        )

    records_by_split = {}
    source_fingerprints = {}
    for split in SPLIT_NAMES:
        record_path = _resolve_repository_path(
            repository_root,
            config.records_path(split),
        )
        projection_path = _resolve_repository_path(
            repository_root,
            config.sequences_path(split),
        )
        output_metadata = parent_manifest["outputs"][split]
        _require_equal(
            output_metadata["records_path"],
            config.records_path(split),
            "{0} records path".format(split),
        )
        _require_equal(
            output_metadata["records_file_sha256"],
            hash_file_bytes(str(record_path)),
            "{0} records file hash".format(split),
        )
        _require_equal(
            output_metadata["sequence_projection_path"],
            config.sequences_path(split),
            "{0} projection path".format(split),
        )
        _require_equal(
            output_metadata["sequence_projection_file_sha256"],
            hash_file_bytes(str(projection_path)),
            "{0} projection file hash".format(split),
        )

        records = read_genomic_records(str(record_path))
        records_by_split[split] = records
        _require_equal(
            len(records),
            config.target_count(split),
            "{0} record count".format(split),
        )
        _require_equal(
            output_metadata["target_record_count"],
            config.target_count(split),
            "{0} manifest target count".format(split),
        )
        _require_equal(
            output_metadata["actual_record_count"],
            len(records),
            "{0} manifest actual count".format(split),
        )
        logical_records_payload = {
            "schema_version": "genomic_record_collection.v1",
            "records": [record.logical_dict() for record in records],
        }
        _require_equal(
            output_metadata["records_logical_sha256"],
            hash_logical_content(logical_records_payload),
            "{0} logical record hash".format(split),
        )
        _validate_sequence_projection(projection_path, records)
        source_fingerprint = fingerprint_sequence_file(
            str(projection_path),
            repository_root,
        )
        source_fingerprints[split] = source_fingerprint
        _require_equal(
            output_metadata["source_fingerprint"],
            source_fingerprint.to_dict(),
            "{0} source fingerprint".format(split),
        )

    rejection_path = _resolve_repository_path(
        repository_root,
        config.rejection_records_path,
    )
    rejection_metadata = _require_mapping(
        parent_manifest["rejections"],
        "parent_manifest.rejections",
    )
    _require_exact_keys(
        rejection_metadata,
        ("path", "file_sha256", "record_count", "resolution"),
        "parent_manifest.rejections",
    )
    _require_equal(
        rejection_metadata["path"],
        config.rejection_records_path,
        "rejection path",
    )
    _require_equal(
        rejection_metadata["file_sha256"],
        hash_file_bytes(str(rejection_path)),
        "rejection file hash",
    )
    rejected_records = _validate_rejection_table(rejection_path)
    rejection_count = len(rejected_records)
    stored_rejection_count = _require_nonnegative_integer(
        rejection_metadata["record_count"],
        "parent_manifest.rejections.record_count",
    )
    _require_equal(
        stored_rejection_count,
        rejection_count,
        "rejection record count",
    )

    child_path = _resolve_repository_path(
        repository_root,
        config.child_manifest_path,
    )
    compatibility = parent_manifest["compatibility"]
    _require_equal(
        compatibility["child_manifest_path"],
        config.child_manifest_path,
        "child manifest path",
    )
    _require_equal(
        compatibility["child_manifest_file_sha256"],
        hash_file_bytes(str(child_path)),
        "child manifest file hash",
    )
    child_manifest = load_split_manifest(str(child_path))
    _require_equal(
        compatibility["child_manifest_hash"],
        child_manifest.manifest_hash,
        "child manifest identity",
    )
    _require_equal(
        child_manifest.training_source.to_dict(),
        source_fingerprints["training"].to_dict(),
        "child training fingerprint",
    )
    _require_equal(
        child_manifest.validation_source.to_dict(),
        source_fingerprints["validation"].to_dict(),
        "child validation fingerprint",
    )
    _require_equal(
        compatibility["test_source_fingerprint"],
        source_fingerprints["test"].to_dict(),
        "parent test fingerprint",
    )
    rebuilt_child_manifest = build_pretraining_split_manifest(
        training_path=str(
            _resolve_repository_path(
                repository_root,
                config.training_sequences_path,
            )
        ),
        validation_path=str(
            _resolve_repository_path(
                repository_root,
                config.validation_sequences_path,
            )
        ),
        repository_root=repository_root,
        mode="report",
        maximum_examples=CHILD_MANIFEST_MAXIMUM_EXAMPLES,
    )
    _require_equal(
        child_manifest.to_dict(),
        rebuilt_child_manifest.to_dict(),
        "reconstructed child manifest",
    )

    scan = scan_fasta_reference(config, repository_root)
    reference_metadata = parent_manifest["reference"]
    _require_equal(
        reference_metadata,
        _reference_manifest_metadata(config, scan),
        "reference metadata",
    )
    _require_equal(
        parent_manifest["policy"],
        _policy_manifest_metadata(config, scan),
        "policy metadata",
    )
    _require_equal(
        parent_manifest["capacity_and_exclusions"],
        _capacity_manifest_metadata(scan),
        "capacity and exclusion metadata",
    )

    _validate_rejection_resolution(
        rejection_metadata=rejection_metadata,
        rejected_records=rejected_records,
        selected_records=records_by_split,
        config=config,
        scan=scan,
    )
    all_records_by_split = {}
    for split in SPLIT_NAMES:
        split_records = list(records_by_split[split])
        for rejected_record in rejected_records:
            if rejected_record.record.split == split:
                split_records.append(rejected_record.record)
        all_records_by_split[split] = tuple(split_records)
    validate_records_against_reference(
        records_by_split=all_records_by_split,
        config=config,
        scan=scan,
        repository_root=repository_root,
    )
    audit = audit_genomic_records(records_by_split, config=config)
    _require_equal(parent_manifest["audits"], audit, "stored genomic audit")
    _validate_child_audit_consistency(child_manifest, audit)
    _enforce_audit_mode(
        mode=mode,
        child_has_overlap=child_manifest.overlap_audit.has_overlap,
        audit=audit,
    )
    return audit


def build_hg38_pretraining_split(
    config: WholeChromosomeSplitConfig,
    repository_root: str,
    dry_run: bool,
) -> Mapping[str, Any]:
    """Run the approved dry-run or exclusively create the full split."""

    config.validate()
    _validate_configured_paths(config, repository_root)
    scan = scan_fasta_reference(config, repository_root)
    dry_run_report = build_dry_run_report(config, scan)
    if not dry_run_report["all_capacities_sufficient"]:
        raise CandidateBudgetError(
            "At least one split lacks enough eligible genomic starts."
        )
    if dry_run:
        return dry_run_report

    _preflight_production_outputs(config, repository_root)
    sampled_coordinates = sample_candidate_coordinates(config, scan)
    candidate_records = extract_candidate_records(
        config=config,
        scan=scan,
        repository_root=repository_root,
        candidates_by_split=sampled_coordinates,
    )
    target_counts = {
        split: config.target_count(split) for split in SPLIT_NAMES
    }
    selected_records, rejected_records, resolution_statistics = (
        resolve_cross_split_equivalence(
            candidate_records=candidate_records,
            target_counts=target_counts,
        )
    )
    audit = audit_genomic_records(selected_records, config=config)
    if not audit["strict_pass"]:
        raise GenomicSplitAuditError(
            "In-memory strict genomic split audit failed before writing."
        )
    _preflight_production_outputs(config, repository_root)
    _create_output_directory_exclusive(config, repository_root)

    for split in SPLIT_NAMES:
        _write_record_tsv_exclusive(
            _resolve_repository_path(
                repository_root,
                config.records_path(split),
            ),
            selected_records[split],
        )
        _write_sequence_projection_exclusive(
            _resolve_repository_path(
                repository_root,
                config.sequences_path(split),
            ),
            selected_records[split],
        )
    _write_rejections_exclusive(
        _resolve_repository_path(
            repository_root,
            config.rejection_records_path,
        ),
        rejected_records,
    )

    source_fingerprints = {}
    for split in SPLIT_NAMES:
        source_fingerprints[split] = fingerprint_sequence_file(
            str(
                _resolve_repository_path(
                    repository_root,
                    config.sequences_path(split),
                )
            ),
            repository_root,
        )
    child_manifest = build_pretraining_split_manifest(
        training_path=str(
            _resolve_repository_path(
                repository_root,
                config.training_sequences_path,
            )
        ),
        validation_path=str(
            _resolve_repository_path(
                repository_root,
                config.validation_sequences_path,
            )
        ),
        repository_root=repository_root,
        mode="strict",
        maximum_examples=CHILD_MANIFEST_MAXIMUM_EXAMPLES,
    )
    child_manifest_path = _resolve_repository_path(
        repository_root,
        config.child_manifest_path,
    )
    save_split_manifest(child_manifest, str(child_manifest_path))

    parent_manifest = _build_genomic_manifest(
        config=config,
        scan=scan,
        selected_records=selected_records,
        rejected_records=rejected_records,
        resolution_statistics=resolution_statistics,
        audit=audit,
        source_fingerprints=source_fingerprints,
        child_manifest=child_manifest,
        repository_root=repository_root,
    )
    _write_json_exclusive(
        _resolve_repository_path(
            repository_root,
            config.genomic_manifest_path,
        ),
        parent_manifest,
    )
    return parent_manifest


def validate_genomic_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the schema and integrity hash of a parent manifest."""

    manifest = _require_mapping(payload, "genomic manifest")
    _require_exact_keys(
        manifest,
        GENOMIC_MANIFEST_FIELD_NAMES,
        "genomic manifest",
    )
    if manifest["schema_version"] != GENOMIC_MANIFEST_SCHEMA_VERSION:
        raise GenomicSplitError(
            "Unsupported genomic pretraining manifest schema."
        )
    if manifest["creation_entry_point"] != CREATION_ENTRY_POINT:
        raise GenomicSplitError(
            "Unsupported genomic manifest creation entry point."
        )
    compatibility = _require_mapping(
        manifest["compatibility"],
        "genomic manifest compatibility",
    )
    _require_exact_keys(
        compatibility,
        COMPATIBILITY_FIELD_NAMES,
        "genomic manifest compatibility",
    )
    expected_compatibility_versions = (
        (
            "source_fingerprint_schema_version",
            SOURCE_FINGERPRINT_SCHEMA_VERSION,
        ),
        (
            "child_manifest_schema_version",
            SPLIT_MANIFEST_SCHEMA_VERSION,
        ),
        (
            "normalization_artifact_schema_version",
            NORMALIZATION_ARTIFACT_SCHEMA_VERSION,
        ),
    )
    for field_name, expected_value in expected_compatibility_versions:
        if compatibility[field_name] != expected_value:
            raise GenomicSplitError(
                "Unsupported genomic manifest compatibility value: {0}.".format(
                    field_name
                )
            )
    test_source_fingerprint = _require_mapping(
        compatibility["test_source_fingerprint"],
        "genomic manifest test source fingerprint",
    )
    if (
        test_source_fingerprint.get("schema_version")
        != SOURCE_FINGERPRINT_SCHEMA_VERSION
    ):
        raise GenomicSplitError(
            "Unsupported genomic manifest test source-fingerprint schema."
        )

    content = dict(manifest)
    stored_hash = str(content.pop("manifest_hash"))
    expected_hash = hash_logical_content(content)
    if stored_hash != expected_hash:
        raise GenomicSplitError("Genomic manifest integrity hash mismatch.")


def load_genomic_manifest(path: str) -> Mapping[str, Any]:
    """Load and validate a genomic parent manifest."""

    with open(path, "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    validate_genomic_manifest(payload)
    return payload


def _build_genomic_manifest(
    config: WholeChromosomeSplitConfig,
    scan: ReferenceScan,
    selected_records: Mapping[
        str,
        Sequence[GenomicWindowRecord],
    ],
    rejected_records: Sequence[RejectedGenomicRecord],
    resolution_statistics: Mapping[str, Any],
    audit: Mapping[str, Any],
    source_fingerprints: Mapping[str, Any],
    child_manifest: Any,
    repository_root: str,
) -> Dict[str, Any]:
    outputs = {}
    for split in SPLIT_NAMES:
        records_path = _resolve_repository_path(
            repository_root,
            config.records_path(split),
        )
        sequences_path = _resolve_repository_path(
            repository_root,
            config.sequences_path(split),
        )
        logical_records_payload = {
            "schema_version": "genomic_record_collection.v1",
            "records": [
                record.logical_dict()
                for record in selected_records[split]
            ],
        }
        outputs[split] = {
            "records_path": config.records_path(split),
            "records_file_sha256": hash_file_bytes(str(records_path)),
            "records_logical_sha256": hash_logical_content(
                logical_records_payload
            ),
            "sequence_projection_path": config.sequences_path(split),
            "sequence_projection_file_sha256": hash_file_bytes(
                str(sequences_path)
            ),
            "target_record_count": config.target_count(split),
            "actual_record_count": len(selected_records[split]),
            "source_fingerprint": source_fingerprints[split].to_dict(),
        }

    rejection_path = _resolve_repository_path(
        repository_root,
        config.rejection_records_path,
    )
    child_path = _resolve_repository_path(
        repository_root,
        config.child_manifest_path,
    )
    content = {
        "schema_version": GENOMIC_MANIFEST_SCHEMA_VERSION,
        "creation_entry_point": CREATION_ENTRY_POINT,
        "reference": _reference_manifest_metadata(config, scan),
        "policy": _policy_manifest_metadata(config, scan),
        "capacity_and_exclusions": _capacity_manifest_metadata(scan),
        "outputs": outputs,
        "rejections": {
            "path": config.rejection_records_path,
            "file_sha256": hash_file_bytes(str(rejection_path)),
            "record_count": len(rejected_records),
            "resolution": dict(resolution_statistics),
        },
        "audits": dict(audit),
        "compatibility": {
            "source_fingerprint_schema_version": (
                SOURCE_FINGERPRINT_SCHEMA_VERSION
            ),
            "child_manifest_schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "child_manifest_path": config.child_manifest_path,
            "child_manifest_file_sha256": hash_file_bytes(str(child_path)),
            "child_manifest_hash": child_manifest.manifest_hash,
            "normalization_artifact_schema_version": (
                NORMALIZATION_ARTIFACT_SCHEMA_VERSION
            ),
            "test_source_fingerprint": (
                source_fingerprints["test"].to_dict()
            ),
        },
    }
    payload = dict(content)
    payload["manifest_hash"] = hash_logical_content(content)
    return payload


def _reference_manifest_metadata(
    config: WholeChromosomeSplitConfig,
    scan: ReferenceScan,
) -> Dict[str, Any]:
    return {
        "source_path": scan.reference_path,
        "reference_identifier": config.reference_identifier,
        "source_file_size_bytes": scan.source_file_size_bytes,
        "raw_file_sha256": scan.raw_file_sha256,
        "logical_reference_schema_version": (
            LOGICAL_REFERENCE_SCHEMA_VERSION
        ),
        "logical_reference_sha256": scan.logical_reference_sha256,
        "ordered_contigs": [
            contig.logical_identity_dict() for contig in scan.contigs
        ],
    }


def _policy_manifest_metadata(
    config: WholeChromosomeSplitConfig,
    scan: ReferenceScan,
) -> Dict[str, Any]:
    return {
        "policy_id": config.policy_id,
        "coordinate_convention": "zero_based_half_open",
        "eligible_chromosomes": list(config.eligible_chromosomes),
        "chromosome_assignments": {
            split: list(config.chromosomes_for_split(split))
            for split in SPLIT_NAMES
        },
        "excluded_contig_identifiers": list(
            scan.excluded_contig_identifiers
        ),
        "window_length": config.window_length,
        "strand_convention": "reference_forward_only",
        "lowercase_rule": "uppercase_and_accept",
        "invalid_symbol_rule": "exclude_complete_window",
        "sampling_method": (
            "sha256_partial_fisher_yates_without_replacement_over_"
            "eligible_start_ordinals"
        ),
        "sampling_seed": config.sampling_seed,
        "maximum_candidate_multiplier": (
            config.maximum_candidate_multiplier
        ),
        "tokenizer_independent": True,
        "repeated_sequence_policy": (
            "reject_all_selected_cross_split_exact_or_reverse_"
            "complement_equivalent_group_members_and_refill"
        ),
        "test_usage_policy": (
            "never_use_for_normalization_fitting_selection_or_tuning"
        ),
    }


def _capacity_manifest_metadata(
    scan: ReferenceScan,
) -> Dict[str, Any]:
    eligible_contigs = [
        contig for contig in scan.contigs if contig.eligible
    ]
    excluded_contigs = [
        contig for contig in scan.contigs if not contig.eligible
    ]
    return {
        "eligible_window_start_capacity": {
            split: scan.split_capacity(split) for split in SPLIT_NAMES
        },
        "eligible_acgt_base_count": sum(
            contig.acgt_base_count for contig in eligible_contigs
        ),
        "eligible_n_base_count": sum(
            contig.n_base_count for contig in eligible_contigs
        ),
        "eligible_other_symbol_count": sum(
            contig.other_symbol_count for contig in eligible_contigs
        ),
        "eligible_lowercase_base_count": sum(
            contig.lowercase_base_count for contig in eligible_contigs
        ),
        "invalid_window_start_count": sum(
            contig.invalid_window_start_count
            for contig in eligible_contigs
        ),
        "excluded_contig_count": len(excluded_contigs),
        "excluded_contig_base_count": sum(
            contig.length for contig in excluded_contigs
        ),
    }


def _refill_selected_records(
    split: str,
    selected: Dict[str, list[GenomicWindowRecord]],
    candidate_records: Mapping[str, Sequence[GenomicWindowRecord]],
    next_candidate_index: Dict[str, int],
    target_count: int,
    blacklisted_keys: set[str],
    rejected: list[RejectedGenomicRecord],
    rejected_loci: set[Tuple[str, str, int, int]],
) -> None:
    candidates = candidate_records[split]
    while (
        len(selected[split]) < target_count
        and next_candidate_index[split] < len(candidates)
    ):
        candidate = candidates[next_candidate_index[split]]
        next_candidate_index[split] += 1
        if candidate.canonical_sequence in blacklisted_keys:
            if candidate.locus_key not in rejected_loci:
                rejected.append(
                    RejectedGenomicRecord(
                        record=candidate,
                        rejection_reason=(
                            "blacklisted_cross_split_equivalence_group"
                        ),
                    )
                )
                rejected_loci.add(candidate.locus_key)
        else:
            selected[split].append(candidate)
    if len(selected[split]) < target_count:
        message = (
            "Candidate budget exhausted for {0}: selected {1} of {2} from "
            "{3} candidates."
        )
        raise CandidateBudgetError(
            message.format(
                split,
                len(selected[split]),
                target_count,
                len(candidates),
            )
        )


def _group_selected_by_canonical_sequence(
    selected: Mapping[str, Sequence[GenomicWindowRecord]],
) -> Mapping[str, Tuple[GenomicWindowRecord, ...]]:
    grouped: Dict[str, list[GenomicWindowRecord]] = {}
    for split in SPLIT_NAMES:
        for record in selected[split]:
            grouped.setdefault(record.canonical_sequence, []).append(record)
    return {
        key: tuple(grouped[key]) for key in sorted(grouped)
    }


def _group_has_cross_split_exact_match(
    records: Sequence[GenomicWindowRecord],
) -> bool:
    sequence_splits: Dict[str, set[str]] = {}
    for record in records:
        sequence_splits.setdefault(record.sequence, set()).add(record.split)
    for splits in sequence_splits.values():
        if len(splits) > 1:
            return True
    return False


def _audit_pair(
    first_records: Sequence[GenomicWindowRecord],
    second_records: Sequence[GenomicWindowRecord],
) -> Dict[str, Any]:
    first_exact = _records_by_sequence(first_records)
    second_exact = _records_by_sequence(second_records)
    exact_keys = sorted(set(first_exact).intersection(second_exact))
    first_canonical = _records_by_canonical(first_records)
    second_canonical = _records_by_canonical(second_records)
    canonical_keys = sorted(
        set(first_canonical).intersection(second_canonical)
    )
    rc_only_count = 0
    for key in canonical_keys:
        first_sequences = {
            record.sequence for record in first_canonical[key]
        }
        second_sequences = {
            record.sequence for record in second_canonical[key]
        }
        if not first_sequences.intersection(second_sequences):
            rc_only_count += 1

    first_loci = {record.locus_key for record in first_records}
    second_loci = {record.locus_key for record in second_records}
    same_loci = first_loci.intersection(second_loci)
    interval_overlap_count, minimum_separation = _interval_overlap_summary(
        first_records,
        second_records,
    )
    return {
        "exact_sequence_overlap_group_count": len(exact_keys),
        "exact_first_record_count": sum(
            len(first_exact[key]) for key in exact_keys
        ),
        "exact_second_record_count": sum(
            len(second_exact[key]) for key in exact_keys
        ),
        "reverse_complement_equivalent_group_count": len(canonical_keys),
        "reverse_complement_only_group_count": rc_only_count,
        "same_locus_count": len(same_loci),
        "interval_overlap_pair_count": interval_overlap_count,
        "minimum_same_chromosome_separation_bp": minimum_separation,
    }


def _records_by_sequence(
    records: Sequence[GenomicWindowRecord],
) -> Mapping[str, Tuple[GenomicWindowRecord, ...]]:
    grouped: Dict[str, list[GenomicWindowRecord]] = {}
    for record in records:
        grouped.setdefault(record.sequence, []).append(record)
    return {key: tuple(grouped[key]) for key in grouped}


def _records_by_canonical(
    records: Sequence[GenomicWindowRecord],
) -> Mapping[str, Tuple[GenomicWindowRecord, ...]]:
    grouped: Dict[str, list[GenomicWindowRecord]] = {}
    for record in records:
        grouped.setdefault(record.canonical_sequence, []).append(record)
    return {key: tuple(grouped[key]) for key in grouped}


def _within_split_repeat_summary(
    records: Sequence[GenomicWindowRecord],
) -> Dict[str, int]:
    exact = _records_by_sequence(records)
    canonical = _records_by_canonical(records)
    exact_groups = 0
    canonical_groups = 0
    for group_records in exact.values():
        loci = {record.locus_key for record in group_records}
        if len(loci) > 1:
            exact_groups += 1
    for group_records in canonical.values():
        loci = {record.locus_key for record in group_records}
        if len(loci) > 1:
            canonical_groups += 1
    return {
        "exact_distinct_locus_group_count": exact_groups,
        "reverse_complement_equivalent_distinct_locus_group_count": (
            canonical_groups
        ),
    }


def _interval_overlap_summary(
    first_records: Sequence[GenomicWindowRecord],
    second_records: Sequence[GenomicWindowRecord],
) -> Tuple[int, Optional[int]]:
    first_by_chromosome: Dict[str, list[GenomicWindowRecord]] = {}
    second_by_chromosome: Dict[str, list[GenomicWindowRecord]] = {}
    for record in first_records:
        first_by_chromosome.setdefault(record.chromosome, []).append(record)
    for record in second_records:
        second_by_chromosome.setdefault(record.chromosome, []).append(record)

    overlap_count = 0
    minimum_separation = None
    shared_chromosomes = sorted(
        set(first_by_chromosome).intersection(second_by_chromosome)
    )
    for chromosome in shared_chromosomes:
        first_sorted = sorted(
            first_by_chromosome[chromosome],
            key=lambda value: (value.start, value.end),
        )
        second_sorted = sorted(
            second_by_chromosome[chromosome],
            key=lambda value: (value.start, value.end),
        )
        for first_record in first_sorted:
            for second_record in second_sorted:
                if second_record.start >= first_record.end:
                    separation = second_record.start - first_record.end
                    minimum_separation = _minimum_optional(
                        minimum_separation,
                        separation,
                    )
                    break
                if second_record.end <= first_record.start:
                    separation = first_record.start - second_record.end
                    minimum_separation = _minimum_optional(
                        minimum_separation,
                        separation,
                    )
                else:
                    overlap_count += 1
                    minimum_separation = 0
    return overlap_count, minimum_separation


def _minimum_optional(
    current: Optional[int],
    candidate: int,
) -> int:
    if current is None or candidate < current:
        return candidate
    return current


def _ensure_unique_loci(
    records: Sequence[GenomicWindowRecord],
) -> None:
    seen_loci = set()
    seen_record_ids = set()
    for record in records:
        if record.locus_key in seen_loci:
            raise GenomicSplitError(
                "Duplicate genomic locus selected: {0}:{1}-{2}".format(
                    record.chromosome,
                    record.start,
                    record.end,
                )
            )
        if record.record_id in seen_record_ids:
            raise GenomicSplitError(
                "Duplicate genomic record identifier selected."
            )
        seen_loci.add(record.locus_key)
        seen_record_ids.add(record.record_id)


def _validate_configured_paths(
    config: WholeChromosomeSplitConfig,
    repository_root: str,
) -> None:
    root = Path(repository_root).resolve()
    reference_path = _resolve_repository_path(
        repository_root,
        config.reference_path,
    )
    output_directory = _resolve_repository_path(
        repository_root,
        config.output_directory,
    )
    if output_directory == root:
        raise GenomicSplitError(
            "Production output directory must not be the repository root."
        )
    for output_path in config.output_paths():
        resolved = _resolve_repository_path(repository_root, output_path)
        try:
            resolved.relative_to(output_directory)
        except ValueError as error:
            raise GenomicSplitError(
                "Production output escaped the configured output directory."
            ) from error
        if resolved == reference_path:
            raise GenomicSplitError(
                "Production output path must differ from the reference."
            )


def _preflight_production_outputs(
    config: WholeChromosomeSplitConfig,
    repository_root: str,
) -> None:
    output_directory = _resolve_repository_path(
        repository_root,
        config.output_directory,
    )
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(
            "Production output directory already exists: {0}".format(
                output_directory
            )
        )
    for output_path in config.output_paths():
        resolved = _resolve_repository_path(repository_root, output_path)
        if resolved.exists() or resolved.is_symlink():
            raise FileExistsError(
                "Production output already exists: {0}".format(resolved)
            )


def _create_output_directory_exclusive(
    config: WholeChromosomeSplitConfig,
    repository_root: str,
) -> None:
    output_directory = _resolve_repository_path(
        repository_root,
        config.output_directory,
    )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(exist_ok=False)


def _write_record_tsv_exclusive(
    path: Path,
    records: Sequence[GenomicWindowRecord],
) -> None:
    with open(path, "x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=RECORD_FIELD_NAMES,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def _write_sequence_projection_exclusive(
    path: Path,
    records: Sequence[GenomicWindowRecord],
) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(record.sequence)
            output_file.write("\n")


def _write_rejections_exclusive(
    path: Path,
    rejected_records: Sequence[RejectedGenomicRecord],
) -> None:
    with open(path, "x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=REJECTION_FIELD_NAMES,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for rejected_record in rejected_records:
            writer.writerow(rejected_record.to_row())


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    with open(path, "x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(serialized)
        output_file.write("\n")


def _validate_sequence_projection(
    path: Path,
    records: Sequence[GenomicWindowRecord],
) -> None:
    record_index = 0
    with open(path, "r", encoding="utf-8", newline=None) as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if record_index >= len(records):
                raise GenomicSplitAuditError(
                    "Sequence projection contains extra rows."
                )
            sequence = line.rstrip("\r\n")
            if sequence != records[record_index].sequence:
                message = "Sequence projection mismatch at line {0}."
                raise GenomicSplitAuditError(
                    message.format(line_number)
                )
            record_index += 1
    if record_index != len(records):
        raise GenomicSplitAuditError(
            "Sequence projection is missing record rows."
        )


def _validate_rejection_table(
    path: Path,
) -> Tuple[RejectedGenomicRecord, ...]:
    rejected_records = []
    with open(path, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if tuple(reader.fieldnames or ()) != REJECTION_FIELD_NAMES:
            raise GenomicSplitAuditError(
                "Rejection TSV header does not match the required schema."
            )
        for row in reader:
            record = GenomicWindowRecord.from_row(row)
            if (
                row["canonical_sequence_sha256"]
                != record.rc_canonical_sha256
            ):
                raise GenomicSplitAuditError(
                    "Rejection canonical-sequence hash mismatch."
                )
            if not row["rejection_reason"]:
                raise GenomicSplitAuditError(
                    "Rejected record is missing its reason."
                )
            if row["rejection_reason"] not in REJECTION_REASONS:
                raise GenomicSplitAuditError(
                    "Rejected record has an unsupported reason."
                )
            rejected_records.append(
                RejectedGenomicRecord(
                    record=record,
                    rejection_reason=row["rejection_reason"],
                )
            )
    _ensure_unique_loci(
        tuple(value.record for value in rejected_records)
    )
    return tuple(rejected_records)


def _validate_rejection_resolution(
    rejection_metadata: Mapping[str, Any],
    rejected_records: Sequence[RejectedGenomicRecord],
    selected_records: Mapping[str, Sequence[GenomicWindowRecord]],
    config: WholeChromosomeSplitConfig,
    scan: ReferenceScan,
) -> None:
    """Reconcile stored rejection/refill provenance with generated records."""

    resolution = _require_mapping(
        rejection_metadata["resolution"],
        "parent_manifest.rejections.resolution",
    )
    _require_exact_keys(
        resolution,
        RESOLUTION_FIELD_NAMES,
        "parent_manifest.rejections.resolution",
    )
    resolution_iterations = _require_nonnegative_integer(
        resolution["resolution_iterations"],
        "resolution_iterations",
    )
    if resolution_iterations < 1:
        raise GenomicSplitAuditError(
            "Rejection resolution iterations must be positive."
        )
    unique_group_count = _require_nonnegative_integer(
        resolution[
            "unique_cross_split_equivalence_groups_rejected"
        ],
        "unique_cross_split_equivalence_groups_rejected",
    )
    unique_exact_count = _require_nonnegative_integer(
        resolution["unique_exact_groups_rejected"],
        "unique_exact_groups_rejected",
    )
    unique_rc_only_count = _require_nonnegative_integer(
        resolution["unique_rc_only_groups_rejected"],
        "unique_rc_only_groups_rejected",
    )
    rejected_candidate_count = _require_nonnegative_integer(
        resolution["rejected_candidate_record_count"],
        "rejected_candidate_record_count",
    )
    _require_equal(
        rejected_candidate_count,
        len(rejected_records),
        "resolution rejected candidate record count",
    )

    consumed_counts = _validate_split_count_mapping(
        resolution["consumed_candidate_counts"],
        "consumed_candidate_counts",
    )
    maximum_counts = _validate_split_count_mapping(
        resolution["maximum_candidate_counts"],
        "maximum_candidate_counts",
    )

    rejected_groups: Dict[str, list[RejectedGenomicRecord]] = {}
    rejected_loci = set()
    for rejected_record in rejected_records:
        record = rejected_record.record
        if record.split_policy_version != config.policy_id:
            raise GenomicSplitAuditError(
                "Rejected record split-policy version mismatch."
            )
        if record.chromosome not in config.chromosomes_for_split(record.split):
            raise GenomicSplitAuditError(
                "Rejected record chromosome violates split assignment."
            )
        if record.end - record.start != config.window_length:
            raise GenomicSplitAuditError(
                "Rejected record window length mismatch."
            )
        if record.block_id:
            raise GenomicSplitAuditError(
                "Rejected whole-chromosome record has a block identifier."
            )
        rejected_groups.setdefault(
            record.canonical_sequence,
            [],
        ).append(rejected_record)
        rejected_loci.add(record.locus_key)

    selected_loci = set()
    selected_canonical_sequences = set()
    for split in SPLIT_NAMES:
        for record in selected_records[split]:
            selected_loci.add(record.locus_key)
            selected_canonical_sequences.add(record.canonical_sequence)
    if selected_loci.intersection(rejected_loci):
        raise GenomicSplitAuditError(
            "A rejected genomic locus also appears in a final split."
        )
    if selected_canonical_sequences.intersection(rejected_groups):
        raise GenomicSplitAuditError(
            "A rejected equivalence group also appears in a final split."
        )

    observed_exact_count = 0
    observed_rc_only_count = 0
    for canonical_sequence in sorted(rejected_groups):
        group = rejected_groups[canonical_sequence]
        primary_reasons = {
            value.rejection_reason
            for value in group
            if value.rejection_reason in PRIMARY_REJECTION_REASONS
        }
        if len(primary_reasons) != 1:
            raise GenomicSplitAuditError(
                "Rejected equivalence group lacks one primary classification."
            )
        primary_records = [
            value.record
            for value in group
            if value.rejection_reason in PRIMARY_REJECTION_REASONS
        ]
        primary_splits = {record.split for record in primary_records}
        if len(primary_splits) < 2:
            raise GenomicSplitAuditError(
                "Primary rejected group does not span multiple splits."
            )
        has_exact_match = _group_has_cross_split_exact_match(primary_records)
        if has_exact_match:
            expected_reason = "cross_split_exact_sequence_group"
            observed_exact_count += 1
        else:
            expected_reason = (
                "cross_split_reverse_complement_only_group"
            )
            observed_rc_only_count += 1
        if primary_reasons != {expected_reason}:
            raise GenomicSplitAuditError(
                "Rejected equivalence-group classification mismatch."
            )

    _require_equal(
        unique_group_count,
        len(rejected_groups),
        "resolution unique rejected group count",
    )
    _require_equal(
        unique_exact_count,
        observed_exact_count,
        "resolution unique exact group count",
    )
    _require_equal(
        unique_rc_only_count,
        observed_rc_only_count,
        "resolution unique reverse-complement-only group count",
    )
    _require_equal(
        unique_group_count,
        unique_exact_count + unique_rc_only_count,
        "resolution classified group total",
    )
    if unique_group_count == 0:
        _require_equal(
            resolution_iterations,
            1,
            "resolution iteration count without conflicts",
        )
    elif (
        resolution_iterations < 2
        or resolution_iterations > unique_group_count + 1
    ):
        raise GenomicSplitAuditError(
            "Rejection resolution iteration count is inconsistent."
        )

    for split in SPLIT_NAMES:
        rejected_split_records = [
            value.record
            for value in rejected_records
            if value.record.split == split
        ]
        expected_consumed_count = (
            len(selected_records[split]) + len(rejected_split_records)
        )
        _require_equal(
            consumed_counts[split],
            expected_consumed_count,
            "{0} consumed candidate count".format(split),
        )
        all_ranks = [
            record.selection_rank for record in selected_records[split]
        ]
        all_ranks.extend(
            record.selection_rank for record in rejected_split_records
        )
        if sorted(all_ranks) != list(range(consumed_counts[split])):
            raise GenomicSplitAuditError(
                "{0} consumed candidate ranks are not a complete prefix.".format(
                    split
                )
            )
        expected_maximum = min(
            scan.split_capacity(split),
            config.target_count(split)
            * config.maximum_candidate_multiplier,
        )
        _require_equal(
            maximum_counts[split],
            expected_maximum,
            "{0} maximum candidate count".format(split),
        )
        if consumed_counts[split] > maximum_counts[split]:
            raise GenomicSplitAuditError(
                "{0} consumed candidate count exceeds its budget.".format(
                    split
                )
            )

    expected_candidates = _sample_candidate_coordinate_counts(
        config=config,
        scan=scan,
        candidate_counts=consumed_counts,
    )
    for split in SPLIT_NAMES:
        observed_records = list(selected_records[split])
        observed_records.extend(
            value.record
            for value in rejected_records
            if value.record.split == split
        )
        observed_records.sort(key=lambda value: value.selection_rank)
        for observed, expected in zip(
            observed_records,
            expected_candidates[split],
        ):
            observed_coordinate = (
                observed.split,
                observed.chromosome,
                observed.start,
                observed.end,
                observed.selection_rank,
            )
            expected_coordinate = (
                expected.split,
                expected.chromosome,
                expected.start,
                expected.end,
                expected.selection_rank,
            )
            if observed_coordinate != expected_coordinate:
                message = (
                    "{0} candidate at selection rank {1} does not match "
                    "the deterministic sampler."
                )
                raise GenomicSplitAuditError(
                    message.format(split, expected.selection_rank)
                )


def _validate_split_count_mapping(
    value: Any,
    label: str,
) -> Mapping[str, int]:
    mapping = _require_mapping(value, label)
    _require_exact_keys(mapping, SPLIT_NAMES, label)
    counts = {}
    for split in SPLIT_NAMES:
        counts[split] = _require_nonnegative_integer(
            mapping[split],
            "{0}.{1}".format(label, split),
        )
    return counts


def _validate_child_audit_consistency(
    child_manifest: Any,
    audit: Mapping[str, Any],
) -> None:
    training_validation = audit["pairwise"]["training_vs_validation"]
    overlap_audit = child_manifest.overlap_audit
    _require_equal(
        overlap_audit.exact_sequence_overlap.group_count,
        training_validation["exact_sequence_overlap_group_count"],
        "child exact overlap group count",
    )
    _require_equal(
        overlap_audit.exact_sequence_overlap.training_row_count,
        training_validation["exact_first_record_count"],
        "child exact training record count",
    )
    _require_equal(
        overlap_audit.exact_sequence_overlap.validation_row_count,
        training_validation["exact_second_record_count"],
        "child exact validation record count",
    )
    _require_equal(
        overlap_audit.reverse_complement_equivalent_overlap.group_count,
        training_validation[
            "reverse_complement_equivalent_group_count"
        ],
        "child reverse-complement overlap group count",
    )
    _require_equal(
        overlap_audit.reverse_complement_only_group_count,
        training_validation["reverse_complement_only_group_count"],
        "child reverse-complement-only group count",
    )
    _require_equal(
        overlap_audit.reverse_complement_overlap_includes_exact_matches,
        True,
        "child reverse-complement inclusive-counting rule",
    )


def _validate_audit_mode(mode: str) -> None:
    if mode not in AUDIT_MODES:
        raise ValueError(
            "Genomic split audit mode must be 'report' or 'strict'."
        )


def _enforce_audit_mode(
    mode: str,
    child_has_overlap: bool,
    audit: Mapping[str, Any],
) -> None:
    if mode == "strict" and child_has_overlap:
        raise GenomicSplitAuditError(
            "Child v2 manifest no longer passes strict overlap auditing."
        )
    if mode == "strict" and not audit["strict_pass"]:
        raise GenomicSplitAuditError(
            "Generated artifacts fail strict genomic split auditing."
        )


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise GenomicSplitAuditError(
            "{0} must be a non-negative integer.".format(label)
        )
    return value


def _require_integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise GenomicSplitError(
            "{0} must be an integer.".format(label)
        )
    return value


def _require_equal(
    actual: Any,
    expected: Any,
    label: str,
) -> None:
    if actual != expected:
        raise GenomicSplitAuditError(
            "Generated artifact {0} mismatch.".format(label)
        )


def _resolve_repository_path(
    repository_root: str,
    relative_path: str,
) -> Path:
    configured = Path(relative_path)
    if configured.is_absolute():
        raise GenomicSplitError(
            "Persisted paths must be repository-relative: {0}".format(
                relative_path
            )
        )
    root = Path(repository_root).resolve()
    resolved = (root / configured).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise GenomicSplitError(
            "Configured path escapes repository root: {0}".format(
                relative_path
            )
        ) from error
    return resolved


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GenomicSplitError("{0} must be a mapping.".format(label))
    return value


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected_keys: Iterable[str],
    label: str,
) -> None:
    expected = set(expected_keys)
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        message = "{0} keys mismatch; missing={1}, unexpected={2}."
        raise GenomicSplitError(
            message.format(label, missing, unexpected)
        )


def _string_tuple(value: Any, label: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise GenomicSplitError("{0} must be a list.".format(label))
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise GenomicSplitError(
            "{0} contains an empty identifier.".format(label)
        )
    return result
