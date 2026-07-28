"""Tests for coordinate-preserving, leakage-resistant genomic splits."""

import copy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from scripts.data_prep.audit_genomic_pretraining_splits import (
    main as audit_main,
)
from scripts.data_prep.build_hg38_pretraining_split import (
    main as build_main,
)
from src.data_fingerprints import (
    SOURCE_FINGERPRINT_SCHEMA_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    fingerprint_sequence_file,
    hash_file_bytes,
    load_split_manifest,
    hash_logical_content,
)
from src.genomic_splits import (
    CandidateBudgetError,
    CONFIG_SCHEMA_VERSION,
    GENOMIC_MANIFEST_SCHEMA_VERSION,
    GenomicSplitError,
    GenomicWindowRecord,
    RECORD_FIELD_NAMES,
    SPLIT_NAMES,
    WholeChromosomeSplitConfig,
    _enforce_audit_mode,
    _validate_child_audit_consistency,
    _validate_rejection_resolution,
    audit_genomic_records,
    build_hg38_pretraining_split,
    extract_candidate_records,
    load_genomic_manifest,
    read_genomic_records,
    resolve_cross_split_equivalence,
    sample_candidate_coordinates,
    sample_candidate_ordinals,
    scan_fasta_reference,
    validate_generated_artifacts,
    validate_records_against_reference,
    validate_genomic_manifest,
)


SYNTHETIC_POLICY_ID = "synthetic_whole_chromosome_holdout.test.v1"


class SyntheticWholeChromosomeSplitConfig(WholeChromosomeSplitConfig):
    """Test-only flexible policy that cannot carry the production policy ID."""

    def _validate_policy_contract(self) -> None:
        if self.policy_id != SYNTHETIC_POLICY_ID:
            raise GenomicSplitError(
                "Synthetic fixtures must use the test-only policy ID."
            )


class GenomicSplitTests(unittest.TestCase):
    """Synthetic tests for the Milestone 2.5B policy and artifacts."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repository_root = Path(self.temporary_directory.name)
        (self.repository_root / "data" / "raw").mkdir(parents=True)
        (self.repository_root / "configs").mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_fasta(
        self,
        records,
        name="reference.fa",
        line_width=None,
        descriptions=None,
    ) -> Path:
        path = self.repository_root / "data" / "raw" / name
        lines = []
        for index, (identifier, sequence) in enumerate(records):
            description = ""
            if descriptions is not None:
                description = " " + descriptions[index]
            lines.append(">{0}{1}".format(identifier, description))
            if line_width is None:
                lines.append(sequence)
            else:
                for start in range(0, len(sequence), line_width):
                    lines.append(sequence[start:start + line_width])
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path

    def _config_payload(
        self,
        fasta_path: Path,
        window_length=4,
        target_counts=None,
        candidate_multiplier=4,
        eligible=None,
        chromosome_assignments=None,
    ):
        if target_counts is None:
            target_counts = {
                "training": 2,
                "validation": 1,
                "test": 1,
            }
        if eligible is None:
            eligible = ["chr1", "chr21", "chr22"]
        if chromosome_assignments is None:
            chromosome_assignments = {
                "training": ["chr1"],
                "validation": ["chr21"],
                "test": ["chr22"],
            }
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "reference": {
                "path": fasta_path.relative_to(
                    self.repository_root
                ).as_posix(),
                "expected_raw_sha256": hash_file_bytes(str(fasta_path)),
                "identifier": "synthetic_hg38",
            },
            "policy": {
                "id": SYNTHETIC_POLICY_ID,
                "eligible_chromosomes": eligible,
                "chromosomes": chromosome_assignments,
                "window_length": window_length,
                "sampling_seed": 42,
                "target_counts": target_counts,
                "maximum_candidate_multiplier": candidate_multiplier,
                "strand": "+",
            },
            "outputs": {
                "directory": (
                    "data/generated/hg38_pretraining_split_v1"
                ),
                "records": {
                    "training": (
                        "data/generated/hg38_pretraining_split_v1/"
                        "hg38_training_records_v1.tsv"
                    ),
                    "validation": (
                        "data/generated/hg38_pretraining_split_v1/"
                        "hg38_validation_records_v1.tsv"
                    ),
                    "test": (
                        "data/generated/hg38_pretraining_split_v1/"
                        "hg38_test_records_v1.tsv"
                    ),
                },
                "sequences": {
                    "training": (
                        "data/generated/hg38_pretraining_split_v1/"
                        "hg38_training_sequences_v1.txt"
                    ),
                    "validation": (
                        "data/generated/hg38_pretraining_split_v1/"
                        "hg38_validation_sequences_v1.txt"
                    ),
                    "test": (
                        "data/generated/hg38_pretraining_split_v1/"
                        "hg38_test_sequences_v1.txt"
                    ),
                },
                "rejections": (
                    "data/generated/hg38_pretraining_split_v1/"
                    "hg38_rejections_v1.tsv"
                ),
                "child_manifest": (
                    "data/generated/hg38_pretraining_split_v1/"
                    "hg38_train_validation_manifest_v2.json"
                ),
                "genomic_manifest": (
                    "data/generated/hg38_pretraining_split_v1/"
                    "hg38_coordinate_split_manifest_v1.json"
                ),
            },
        }

    def _write_config(self, payload, name="split.yaml") -> Path:
        path = self.repository_root / "configs" / name
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def _standard_reference(self) -> Path:
        return self._write_fasta(
            (
                ("chr1", "aaaaaaa"),
                ("chr21", "CCCCCCC"),
                ("chr22", "ACACACA"),
                ("chrX", "NNRNN"),
                ("chr1_KI270706v1_random", "AAAA"),
            )
        )

    def _standard_config(
        self,
    ) -> tuple[SyntheticWholeChromosomeSplitConfig, Path]:
        fasta_path = self._standard_reference()
        config_path = self._write_config(
            self._config_payload(fasta_path)
        )
        return SyntheticWholeChromosomeSplitConfig.from_yaml(
            str(config_path)
        ), config_path

    def _record(
        self,
        sequence,
        split,
        rank,
        chromosome,
        start,
    ) -> GenomicWindowRecord:
        return GenomicWindowRecord.create(
            reference_id="synthetic:sha256:reference",
            chromosome=chromosome,
            start=start,
            end=start + len(sequence),
            sequence=sequence,
            split=split,
            selection_rank=rank,
            split_policy_version=SYNTHETIC_POLICY_ID,
        )

    def _run_build_cli(self, arguments):
        target = (
            "scripts.data_prep.build_hg38_pretraining_split."
            "WholeChromosomeSplitConfig"
        )
        with mock.patch(
            target,
            SyntheticWholeChromosomeSplitConfig,
        ):
            return build_main(arguments)

    def _run_audit_cli(self, arguments):
        target = (
            "scripts.data_prep.audit_genomic_pretraining_splits."
            "WholeChromosomeSplitConfig"
        )
        with mock.patch(
            target,
            SyntheticWholeChromosomeSplitConfig,
        ):
            return audit_main(arguments)

    def _read_parent_manifest_payload(
        self,
        config: WholeChromosomeSplitConfig,
    ):
        path = self.repository_root / config.genomic_manifest_path
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_parent_manifest_payload(
        self,
        config: WholeChromosomeSplitConfig,
        payload,
    ) -> None:
        content = dict(payload)
        content.pop("manifest_hash", None)
        payload["manifest_hash"] = hash_logical_content(content)
        serialized = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        path = self.repository_root / config.genomic_manifest_path
        path.write_text(serialized + "\n", encoding="utf-8")

    def _write_child_manifest_payload(
        self,
        config: WholeChromosomeSplitConfig,
        payload,
    ) -> None:
        content = dict(payload)
        content.pop("manifest_hash", None)
        payload["manifest_hash"] = hash_logical_content(content)
        serialized = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        child_path = self.repository_root / config.child_manifest_path
        child_path.write_text(serialized + "\n", encoding="utf-8")

        parent = self._read_parent_manifest_payload(config)
        compatibility = parent["compatibility"]
        compatibility["child_manifest_file_sha256"] = hash_file_bytes(
            str(child_path)
        )
        compatibility["child_manifest_hash"] = payload["manifest_hash"]
        self._write_parent_manifest_payload(config, parent)

    def _production_config_payload(self):
        repository_root = Path(__file__).resolve().parents[1]
        config_path = (
            repository_root / "configs" / "hg38_pretraining_split_v1.yaml"
        )
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def _set_nested_payload_value(
        self,
        payload,
        key_path,
        value,
    ) -> None:
        current = payload
        for key in key_path[:-1]:
            current = current[key]
        current[key_path[-1]] = value

    def _tree_snapshot(self):
        snapshot = {}
        for path in sorted(self.repository_root.rglob("*")):
            relative = path.relative_to(self.repository_root).as_posix()
            if path.is_file():
                snapshot[relative] = path.read_bytes()
            else:
                snapshot[relative] = None
        return snapshot

    def test_logical_reference_hash_ignores_case_wrapping_and_description(
        self,
    ) -> None:
        first_fasta = self._write_fasta(
            (
                ("chr1", "AAAACCCC"),
                ("chr21", "GGGG"),
                ("chr22", "TTTT"),
            ),
            name="first.fa",
            descriptions=("first description", "second", "third"),
        )
        second_fasta = self._write_fasta(
            (
                ("chr1", "aaaacccc"),
                ("chr21", "gggg"),
                ("chr22", "tttt"),
            ),
            name="second.fa",
            line_width=2,
            descriptions=("changed", "changed", "changed"),
        )
        first_config_path = self._write_config(
            self._config_payload(first_fasta),
            name="first.yaml",
        )
        second_config_path = self._write_config(
            self._config_payload(second_fasta),
            name="second.yaml",
        )
        first_config = SyntheticWholeChromosomeSplitConfig.from_yaml(
            str(first_config_path)
        )
        second_config = SyntheticWholeChromosomeSplitConfig.from_yaml(
            str(second_config_path)
        )

        first_scan = scan_fasta_reference(
            first_config,
            str(self.repository_root),
        )
        second_scan = scan_fasta_reference(
            second_config,
            str(self.repository_root),
        )

        self.assertNotEqual(
            first_scan.raw_file_sha256,
            second_scan.raw_file_sha256,
        )
        self.assertEqual(
            first_scan.logical_reference_sha256,
            second_scan.logical_reference_sha256,
        )

    def test_valid_runs_exclude_n_and_other_iupac_boundaries(self) -> None:
        fasta_path = self._write_fasta(
            (
                ("chr1", "aaaaNRAAAA"),
                ("chr21", "CCCC"),
                ("chr22", "GGGG"),
            )
        )
        payload = self._config_payload(
            fasta_path,
            window_length=3,
            target_counts={
                "training": 1,
                "validation": 1,
                "test": 1,
            },
        )
        config = SyntheticWholeChromosomeSplitConfig.from_yaml(
            str(self._write_config(payload))
        )

        scan = scan_fasta_reference(config, str(self.repository_root))
        chr1 = scan.contig_by_identifier()["chr1"]
        intervals = scan.intervals_for_split("training")

        self.assertEqual(chr1.length, 10)
        self.assertEqual(chr1.acgt_base_count, 8)
        self.assertEqual(chr1.n_base_count, 1)
        self.assertEqual(chr1.other_symbol_count, 1)
        self.assertEqual(chr1.lowercase_base_count, 4)
        self.assertEqual(chr1.total_possible_window_starts, 8)
        self.assertEqual(chr1.eligible_window_start_count, 4)
        self.assertEqual(chr1.invalid_window_start_count, 4)
        self.assertEqual(
            tuple((value.start, value.stop) for value in intervals),
            ((0, 2), (6, 8)),
        )

    def test_allowed_and_excluded_contigs_and_assignments(self) -> None:
        config, unused_path = self._standard_config()
        scan = scan_fasta_reference(config, str(self.repository_root))

        self.assertEqual(config.split_for_chromosome("chr1"), "training")
        self.assertEqual(config.split_for_chromosome("chr21"), "validation")
        self.assertEqual(config.split_for_chromosome("chr22"), "test")
        self.assertEqual(
            scan.excluded_contig_identifiers,
            ("chrX", "chr1_KI270706v1_random"),
        )
        self.assertEqual(scan.split_capacity("training"), 4)
        self.assertEqual(scan.split_capacity("validation"), 4)
        self.assertEqual(scan.split_capacity("test"), 4)

    def test_sha256_sampling_is_deterministic_and_without_replacement(
        self,
    ) -> None:
        first = sample_candidate_ordinals(100, 100, 42, "training")
        second = sample_candidate_ordinals(100, 100, 42, "training")
        other = sample_candidate_ordinals(100, 100, 43, "training")

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(range(100)))
        self.assertEqual(len(first), len(set(first)))
        self.assertNotEqual(first, other)

    def test_coordinate_sampling_uses_one_split_wide_capacity_space(
        self,
    ) -> None:
        fasta_path = self._write_fasta(
            (
                ("chr1", "AAAAA"),
                ("chr2", "CCCCCCCC"),
                ("chr21", "GGGG"),
                ("chr22", "ACAC"),
            )
        )
        payload = self._config_payload(
            fasta_path,
            target_counts={
                "training": 7,
                "validation": 1,
                "test": 1,
            },
            candidate_multiplier=1,
            eligible=["chr1", "chr2", "chr21", "chr22"],
            chromosome_assignments={
                "training": ["chr1", "chr2"],
                "validation": ["chr21"],
                "test": ["chr22"],
            },
        )
        config = SyntheticWholeChromosomeSplitConfig.from_yaml(
            str(self._write_config(payload))
        )
        scan = scan_fasta_reference(config, str(self.repository_root))
        sampled = sample_candidate_coordinates(config, scan)
        training_loci = {
            (record.chromosome, record.start)
            for record in sampled["training"]
        }

        self.assertEqual(scan.split_capacity("training"), 7)
        self.assertEqual(
            training_loci,
            {
                ("chr1", 0),
                ("chr1", 1),
                ("chr2", 0),
                ("chr2", 1),
                ("chr2", 2),
                ("chr2", 3),
                ("chr2", 4),
            },
        )

    def test_record_coordinates_hashes_and_identity_are_stable(self) -> None:
        training = self._record("AACC", "training", 0, "chr1", 4)
        validation = self._record("AACC", "validation", 8, "chr1", 4)

        self.assertEqual(training.start, 4)
        self.assertEqual(training.end, 8)
        self.assertEqual(training.strand, "+")
        self.assertEqual(training.block_id, "")
        self.assertEqual(training.record_id, validation.record_id)
        self.assertNotEqual(training.selection_rank, validation.selection_rank)
        self.assertEqual(
            tuple(training.to_row()),
            RECORD_FIELD_NAMES,
        )

    def test_exact_groups_are_rejected_and_refilled(self) -> None:
        candidates = {
            "training": (
                self._record("AAAA", "training", 0, "chr1", 0),
                self._record("AACC", "training", 1, "chr1", 10),
            ),
            "validation": (
                self._record("AAAA", "validation", 0, "chr21", 0),
                self._record("ACAC", "validation", 1, "chr21", 10),
            ),
            "test": (
                self._record("CGCG", "test", 0, "chr22", 0),
            ),
        }
        targets = {split: 1 for split in SPLIT_NAMES}

        selected, rejected, statistics = resolve_cross_split_equivalence(
            candidates,
            targets,
        )

        self.assertEqual(selected["training"][0].sequence, "AACC")
        self.assertEqual(selected["validation"][0].sequence, "ACAC")
        self.assertEqual(len(rejected), 2)
        self.assertEqual(statistics["unique_exact_groups_rejected"], 1)
        self.assertEqual(statistics["unique_rc_only_groups_rejected"], 0)
        self.assertTrue(audit_genomic_records(selected)["strict_pass"])

    def test_rejection_resolution_metadata_accepts_real_conflict(
        self,
    ) -> None:
        fasta_path = self._write_fasta(
            (
                ("chr1", "AAAAAC"),
                ("chr21", "AAAAAG"),
                ("chr22", "CCCCCC"),
            )
        )
        payload = self._config_payload(
            fasta_path,
            target_counts={split: 1 for split in SPLIT_NAMES},
            candidate_multiplier=3,
        )
        config = SyntheticWholeChromosomeSplitConfig.from_yaml(
            str(self._write_config(payload))
        )
        scan = scan_fasta_reference(config, str(self.repository_root))
        sampled_coordinates = sample_candidate_coordinates(config, scan)
        candidates = extract_candidate_records(
            config=config,
            scan=scan,
            repository_root=str(self.repository_root),
            candidates_by_split=sampled_coordinates,
        )
        selected, rejected, statistics = resolve_cross_split_equivalence(
            candidates,
            {split: 1 for split in SPLIT_NAMES},
        )

        _validate_rejection_resolution(
            rejection_metadata={"resolution": statistics},
            rejected_records=rejected,
            selected_records=selected,
            config=config,
            scan=scan,
        )

        self.assertEqual(len(rejected), 3)
        self.assertEqual(
            statistics["unique_exact_groups_rejected"],
            1,
        )
        for split in SPLIT_NAMES:
            consumed_ranks = [
                record.selection_rank for record in selected[split]
            ]
            consumed_ranks.extend(
                value.record.selection_rank
                for value in rejected
                if value.record.split == split
            )
            self.assertEqual(
                sorted(consumed_ranks),
                list(
                    range(
                        statistics["consumed_candidate_counts"][split]
                    )
                ),
            )

    def test_rc_only_groups_are_rejected_and_refilled(self) -> None:
        candidates = {
            "training": (
                self._record("AACC", "training", 0, "chr1", 0),
                self._record("AAAA", "training", 1, "chr1", 10),
            ),
            "validation": (
                self._record("GGTT", "validation", 0, "chr21", 0),
                self._record("CCCC", "validation", 1, "chr21", 10),
            ),
            "test": (
                self._record("ACAC", "test", 0, "chr22", 0),
            ),
        }
        targets = {split: 1 for split in SPLIT_NAMES}

        selected, rejected, statistics = resolve_cross_split_equivalence(
            candidates,
            targets,
        )

        self.assertEqual(len(rejected), 2)
        self.assertEqual(statistics["unique_exact_groups_rejected"], 0)
        self.assertEqual(statistics["unique_rc_only_groups_rejected"], 1)
        self.assertTrue(audit_genomic_records(selected)["strict_pass"])

    def test_palindrome_is_exact_not_rc_only(self) -> None:
        candidates = {
            "training": (
                self._record("ATAT", "training", 0, "chr1", 0),
                self._record("AAAA", "training", 1, "chr1", 10),
            ),
            "validation": (
                self._record("ATAT", "validation", 0, "chr21", 0),
                self._record("CCCC", "validation", 1, "chr21", 10),
            ),
            "test": (
                self._record("ACAC", "test", 0, "chr22", 0),
            ),
        }
        targets = {split: 1 for split in SPLIT_NAMES}

        unused_selected, unused_rejected, statistics = (
            resolve_cross_split_equivalence(candidates, targets)
        )

        self.assertEqual(statistics["unique_exact_groups_rejected"], 1)
        self.assertEqual(statistics["unique_rc_only_groups_rejected"], 0)

    def test_cascading_refill_conflicts_remove_previous_records(self) -> None:
        candidates = {
            "training": (
                self._record("AAAA", "training", 0, "chr1", 0),
                self._record("CCCC", "training", 1, "chr1", 10),
                self._record("AACC", "training", 2, "chr1", 20),
            ),
            "validation": (
                self._record("AAAA", "validation", 0, "chr21", 0),
                self._record("GGGG", "validation", 1, "chr21", 10),
                self._record("ACAC", "validation", 2, "chr21", 20),
            ),
            "test": (
                self._record("CGCG", "test", 0, "chr22", 0),
            ),
        }
        targets = {split: 1 for split in SPLIT_NAMES}

        selected, rejected, statistics = resolve_cross_split_equivalence(
            candidates,
            targets,
        )

        self.assertEqual(selected["training"][0].sequence, "AACC")
        self.assertEqual(selected["validation"][0].sequence, "ACAC")
        self.assertEqual(len(rejected), 4)
        self.assertEqual(
            statistics["unique_cross_split_equivalence_groups_rejected"],
            2,
        )

    def test_candidate_budget_exhaustion_fails_clearly(self) -> None:
        candidates = {
            "training": (
                self._record("AAAA", "training", 0, "chr1", 0),
            ),
            "validation": (
                self._record("AAAA", "validation", 0, "chr21", 0),
            ),
            "test": (
                self._record("ACAC", "test", 0, "chr22", 0),
            ),
        }

        with self.assertRaisesRegex(
            CandidateBudgetError,
            "Candidate budget exhausted",
        ):
            resolve_cross_split_equivalence(
                candidates,
                {split: 1 for split in SPLIT_NAMES},
            )

    def test_same_split_repeated_loci_remain_and_are_reported(self) -> None:
        selected = {
            "training": (
                self._record("AAAA", "training", 0, "chr1", 0),
                self._record("AAAA", "training", 1, "chr1", 10),
            ),
            "validation": (
                self._record("CCCC", "validation", 0, "chr21", 0),
            ),
            "test": (
                self._record("ACAC", "test", 0, "chr22", 0),
            ),
        }

        audit = audit_genomic_records(selected)

        self.assertTrue(audit["strict_pass"])
        self.assertEqual(
            audit["within_split_repeated_sequences"]["training"][
                "exact_distinct_locus_group_count"
            ],
            1,
        )

    def test_duplicate_loci_are_rejected(self) -> None:
        duplicate_one = self._record("AAAA", "training", 0, "chr1", 0)
        duplicate_two = self._record("AAAA", "training", 1, "chr1", 0)
        records = {
            "training": (duplicate_one, duplicate_two),
            "validation": (
                self._record("CCCC", "validation", 0, "chr21", 0),
            ),
            "test": (
                self._record("ACAC", "test", 0, "chr22", 0),
            ),
        }

        with self.assertRaisesRegex(
            GenomicSplitError,
            "Duplicate genomic locus",
        ):
            audit_genomic_records(records)

    def test_half_open_touching_intervals_do_not_overlap(self) -> None:
        records = {
            "training": (
                self._record("AAAA", "training", 0, "chr1", 0),
            ),
            "validation": (
                self._record("CCCC", "validation", 0, "chr1", 4),
            ),
            "test": (
                self._record("ACAC", "test", 0, "chr22", 0),
            ),
        }

        audit = audit_genomic_records(records)
        pair = audit["pairwise"]["training_vs_validation"]

        self.assertEqual(pair["interval_overlap_pair_count"], 0)
        self.assertEqual(
            pair["minimum_same_chromosome_separation_bp"],
            0,
        )
        self.assertTrue(audit["strict_pass"])

    def test_partial_and_same_locus_cross_split_overlap_are_audited(
        self,
    ) -> None:
        records = {
            "training": (
                self._record("AAAA", "training", 0, "chr1", 0),
            ),
            "validation": (
                self._record("CCCC", "validation", 0, "chr1", 2),
            ),
            "test": (
                self._record("AAAA", "test", 0, "chr1", 0),
            ),
        }

        audit = audit_genomic_records(records)

        self.assertFalse(audit["strict_pass"])
        self.assertEqual(
            audit["pairwise"]["training_vs_validation"][
                "interval_overlap_pair_count"
            ],
            1,
        )
        self.assertEqual(
            audit["pairwise"]["training_vs_test"]["same_locus_count"],
            1,
        )

    def test_dry_run_creates_no_files_or_directories(self) -> None:
        config, config_path = self._standard_config()
        before = self._tree_snapshot()
        output = io.StringIO()

        with redirect_stdout(output):
            result = self._run_build_cli(
                (
                    "--config",
                    str(config_path),
                    "--repository-root",
                    str(self.repository_root),
                    "--dry-run",
                )
            )

        self.assertEqual(before, self._tree_snapshot())
        self.assertFalse(
            (self.repository_root / config.output_directory).exists()
        )
        self.assertTrue(result["all_capacities_sufficient"])
        self.assertFalse(result["production_outputs_created"])

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        config, unused_config_path = self._standard_config()
        output_directory = self.repository_root / config.output_directory
        output_directory.mkdir(parents=True)
        sentinel = output_directory / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            build_hg38_pretraining_split(
                config=config,
                repository_root=str(self.repository_root),
                dry_run=False,
            )

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertEqual(tuple(output_directory.iterdir()), (sentinel,))

    def test_unknown_tokenizer_configuration_is_rejected(self) -> None:
        fasta_path = self._standard_reference()
        payload = self._config_payload(fasta_path)
        payload["policy"]["tokenizer"] = {"k": 6}
        config_path = self._write_config(payload)

        with self.assertRaisesRegex(
            GenomicSplitError,
            "unexpected=.*tokenizer",
        ):
            WholeChromosomeSplitConfig.from_yaml(str(config_path))

    def test_checked_in_hg38_config_matches_approved_contract(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config_path = (
            repository_root / "configs" / "hg38_pretraining_split_v1.yaml"
        )

        config = WholeChromosomeSplitConfig.from_yaml(str(config_path))

        self.assertEqual(
            config.policy_id,
            "hg38_whole_chromosome_holdout.v1",
        )

    def test_integer_policy_fields_reject_yaml_numeric_coercions(self) -> None:
        base_payload = self._production_config_payload()
        field_cases = (
            (
                "policy.window_length",
                ("policy", "window_length"),
                256,
            ),
            (
                "policy.sampling_seed",
                ("policy", "sampling_seed"),
                42,
            ),
            (
                "policy.target_counts.training",
                ("policy", "target_counts", "training"),
                180000,
            ),
            (
                "policy.target_counts.validation",
                ("policy", "target_counts", "validation"),
                10000,
            ),
            (
                "policy.target_counts.test",
                ("policy", "target_counts", "test"),
                10000,
            ),
            (
                "policy.maximum_candidate_multiplier",
                ("policy", "maximum_candidate_multiplier"),
                2,
            ),
        )
        for field_label, key_path, approved_value in field_cases:
            invalid_values = (
                float(approved_value),
                float(approved_value) + 0.9,
                True,
                str(approved_value),
                None,
            )
            for index, invalid_value in enumerate(invalid_values):
                with self.subTest(
                    field=field_label,
                    value=invalid_value,
                ):
                    payload = copy.deepcopy(base_payload)
                    self._set_nested_payload_value(
                        payload,
                        key_path,
                        invalid_value,
                    )
                    config_path = self._write_config(
                        payload,
                        name="non_integer_{0}_{1}.yaml".format(
                            key_path[-1],
                            index,
                        ),
                    )
                    with self.assertRaisesRegex(
                        GenomicSplitError,
                        "{0} must be an integer".format(field_label),
                    ):
                        WholeChromosomeSplitConfig.from_yaml(
                            str(config_path)
                        )

        valid_path = self._write_config(
            copy.deepcopy(base_payload),
            name="valid_integer_256.yaml",
        )
        valid_config = WholeChromosomeSplitConfig.from_yaml(str(valid_path))
        self.assertEqual(valid_config.window_length, 256)
        self.assertIs(type(valid_config.window_length), int)

    def test_approved_contract_rejects_scientific_drift(self) -> None:
        base_payload = self._production_config_payload()
        simple_cases = (
            (
                "policy_id",
                ("policy", "id"),
                "hg38_whole_chromosome_holdout.v2",
            ),
            (
                "reference_path",
                ("reference", "path"),
                "data/raw/other.fa",
            ),
            (
                "reference_hash",
                ("reference", "expected_raw_sha256"),
                "0" * 64,
            ),
            (
                "reference_identifier",
                ("reference", "identifier"),
                "other_hg38",
            ),
            (
                "window_length",
                ("policy", "window_length"),
                128,
            ),
            (
                "sampling_seed",
                ("policy", "sampling_seed"),
                43,
            ),
            (
                "training_target",
                ("policy", "target_counts", "training"),
                179999,
            ),
            (
                "validation_target",
                ("policy", "target_counts", "validation"),
                9999,
            ),
            (
                "test_target",
                ("policy", "target_counts", "test"),
                9999,
            ),
            (
                "candidate_multiplier",
                ("policy", "maximum_candidate_multiplier"),
                3,
            ),
        )
        mutated_payloads = []
        for label, key_path, value in simple_cases:
            payload = copy.deepcopy(base_payload)
            self._set_nested_payload_value(payload, key_path, value)
            mutated_payloads.append((label, payload))

        payload = copy.deepcopy(base_payload)
        payload["policy"]["eligible_chromosomes"].reverse()
        mutated_payloads.append(("eligible_order", payload))

        payload = copy.deepcopy(base_payload)
        payload["policy"]["chromosomes"]["training"].reverse()
        mutated_payloads.append(("training_order", payload))

        payload = copy.deepcopy(base_payload)
        payload["policy"]["chromosomes"]["validation"] = ["chr22"]
        payload["policy"]["chromosomes"]["test"] = ["chr21"]
        mutated_payloads.append(("validation_test_swap", payload))

        for index, (label, payload) in enumerate(mutated_payloads):
            with self.subTest(field=label):
                path = self._write_config(
                    payload,
                    name="policy_drift_{0}.yaml".format(index),
                )
                with self.assertRaisesRegex(
                    GenomicSplitError,
                    "Approved hg38 whole-chromosome v1 contract mismatch",
                ):
                    WholeChromosomeSplitConfig.from_yaml(str(path))

    def test_approved_contract_rejects_alternate_output_directory(self) -> None:
        payload = self._production_config_payload()
        approved_directory = "data/generated/hg38_pretraining_split_v1"
        alternate_directory = "data/generated/hg38_pretraining_split_v2"
        outputs = payload["outputs"]
        outputs["directory"] = alternate_directory
        for split in SPLIT_NAMES:
            outputs["records"][split] = outputs["records"][split].replace(
                approved_directory,
                alternate_directory,
            )
            outputs["sequences"][split] = outputs["sequences"][split].replace(
                approved_directory,
                alternate_directory,
            )
        for key in ("rejections", "child_manifest", "genomic_manifest"):
            outputs[key] = outputs[key].replace(
                approved_directory,
                alternate_directory,
            )
        path = self._write_config(payload, name="alternate_outputs.yaml")

        with self.assertRaisesRegex(
            GenomicSplitError,
            "outputs.directory",
        ):
            WholeChromosomeSplitConfig.from_yaml(str(path))

    def test_approved_contract_rejects_unversioned_output_name(self) -> None:
        payload = self._production_config_payload()
        payload["outputs"]["records"]["training"] = (
            "data/generated/hg38_pretraining_split_v1/training.tsv"
        )
        path = self._write_config(payload, name="unversioned_output.yaml")

        with self.assertRaisesRegex(
            GenomicSplitError,
            "outputs.records.training",
        ):
            WholeChromosomeSplitConfig.from_yaml(str(path))

    def test_audit_mode_enforcement_is_report_or_strict(self) -> None:
        leaky_audit = {"strict_pass": False}

        _enforce_audit_mode(
            mode="report",
            child_has_overlap=True,
            audit=leaky_audit,
        )
        with self.assertRaisesRegex(
            GenomicSplitError,
            "Child v2 manifest",
        ):
            _enforce_audit_mode(
                mode="strict",
                child_has_overlap=True,
                audit=leaky_audit,
            )
        with self.assertRaisesRegex(
            GenomicSplitError,
            "strict genomic split",
        ):
            _enforce_audit_mode(
                mode="strict",
                child_has_overlap=False,
                audit=leaky_audit,
            )

    def test_complete_auditor_rejects_unknown_mode_before_io(self) -> None:
        config, unused_config_path = self._standard_config()

        with self.assertRaisesRegex(ValueError, "report.*strict"):
            validate_generated_artifacts(
                config=config,
                repository_root=str(self.repository_root),
                mode="permissive",
            )

    def test_audit_cli_selects_report_and_strict_modes(self) -> None:
        unused_config, config_path = self._standard_config()
        validator_target = (
            "scripts.data_prep.audit_genomic_pretraining_splits."
            "validate_generated_artifacts"
        )
        cases = (
            ("report", (), False),
            ("strict", ("--strict",), True),
        )
        for expected_mode, extra_arguments, strict_pass in cases:
            arguments = [
                "--config",
                str(config_path),
                "--repository-root",
                str(self.repository_root),
            ]
            arguments.extend(extra_arguments)
            with self.subTest(mode=expected_mode):
                with mock.patch(
                    validator_target,
                    return_value={"strict_pass": strict_pass},
                ) as validator:
                    with redirect_stdout(io.StringIO()):
                        result = self._run_audit_cli(tuple(arguments))
                self.assertEqual(result["strict_pass"], strict_pass)
                validator.assert_called_once_with(
                    mock.ANY,
                    str(self.repository_root.resolve()),
                    mode=expected_mode,
                )

    def test_full_synthetic_build_is_v2_compatible_and_auditable(self) -> None:
        config, unused_config_path = self._standard_config()

        parent = build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )

        self.assertEqual(
            parent["schema_version"],
            GENOMIC_MANIFEST_SCHEMA_VERSION,
        )
        self.assertTrue(parent["audits"]["strict_pass"])
        validate_genomic_manifest(parent)
        loaded_parent = load_genomic_manifest(
            str(self.repository_root / config.genomic_manifest_path)
        )
        self.assertEqual(parent, loaded_parent)

        records_by_split = {}
        for split in SPLIT_NAMES:
            record_path = self.repository_root / config.records_path(split)
            records = read_genomic_records(str(record_path))
            records_by_split[split] = records
            self.assertEqual(len(records), config.target_count(split))
            projection_path = (
                self.repository_root / config.sequences_path(split)
            )
            projection_sequences = projection_path.read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(
                projection_sequences,
                [record.sequence for record in records],
            )
            fingerprint = fingerprint_sequence_file(
                str(projection_path),
                str(self.repository_root),
            )
            self.assertEqual(
                fingerprint.schema_version,
                SOURCE_FINGERPRINT_SCHEMA_VERSION,
            )
            self.assertEqual(
                fingerprint.to_dict(),
                parent["outputs"][split]["source_fingerprint"],
            )

        child = load_split_manifest(
            str(self.repository_root / config.child_manifest_path)
        )
        self.assertEqual(
            child.schema_version,
            SPLIT_MANIFEST_SCHEMA_VERSION,
        )
        self.assertFalse(child.overlap_audit.has_overlap)
        self.assertEqual(
            child.manifest_hash,
            parent["compatibility"]["child_manifest_hash"],
        )
        inconsistent_overlap_audits = (
            replace(
                child.overlap_audit,
                reverse_complement_only_group_count=1,
            ),
            replace(
                child.overlap_audit,
                reverse_complement_overlap_includes_exact_matches=False,
            ),
        )
        for overlap_audit in inconsistent_overlap_audits:
            with self.subTest(overlap_audit=overlap_audit):
                inconsistent_child = replace(
                    child,
                    overlap_audit=overlap_audit,
                )
                with self.assertRaises(GenomicSplitError):
                    _validate_child_audit_consistency(
                        inconsistent_child,
                        parent["audits"],
                    )

        output = io.StringIO()
        config_path = self.repository_root / "configs" / "split.yaml"
        with redirect_stdout(output):
            audit = self._run_audit_cli(
                (
                    "--config",
                    str(config_path),
                    "--repository-root",
                    str(self.repository_root),
                    "--strict",
                )
            )
        self.assertTrue(audit["strict_pass"])
        self.assertTrue(audit_genomic_records(records_by_split)["strict_pass"])

    def test_configured_audit_rejects_a_truncated_split(self) -> None:
        config, unused_config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        records_by_split = {}
        for split in SPLIT_NAMES:
            records_by_split[split] = read_genomic_records(
                str(self.repository_root / config.records_path(split))
            )
        records_by_split["training"] = records_by_split["training"][:-1]

        with self.assertRaisesRegex(
            GenomicSplitError,
            "Record count for training",
        ):
            audit_genomic_records(records_by_split, config=config)

    def test_complete_auditor_rejects_projection_tampering(self) -> None:
        config, config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        projection_path = (
            self.repository_root / config.validation_sequences_path
        )
        projection_path.write_text("TTTT\n", encoding="utf-8")

        with self.assertRaisesRegex(
            GenomicSplitError,
            "projection file hash mismatch",
        ):
            self._run_audit_cli(
                (
                    "--config",
                    str(config_path),
                    "--repository-root",
                    str(self.repository_root),
                    "--strict",
                )
            )

    def test_complete_auditor_rejects_child_manifest_tampering(self) -> None:
        config, config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        child_path = self.repository_root / config.child_manifest_path
        child_path.write_bytes(child_path.read_bytes() + b" ")

        with self.assertRaisesRegex(
            GenomicSplitError,
            "child manifest file hash mismatch",
        ):
            self._run_audit_cli(
                (
                    "--config",
                    str(config_path),
                    "--repository-root",
                    str(self.repository_root),
                    "--strict",
                )
            )

    def test_parent_manifest_binds_every_compatibility_schema_claim(
        self,
    ) -> None:
        config, unused_config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        original_payload = self._read_parent_manifest_payload(config)
        cases = (
            (
                "parent_schema",
                ("schema_version",),
                "genomic_pretraining_split_manifest.v999",
                "manifest schema",
            ),
            (
                "creation_entry_point",
                ("creation_entry_point",),
                "python other_builder.py",
                "creation entry point",
            ),
            (
                "source_fingerprint_schema",
                (
                    "compatibility",
                    "source_fingerprint_schema_version",
                ),
                "sequence_source_fingerprint.v999",
                "source_fingerprint_schema_version",
            ),
            (
                "child_manifest_schema",
                ("compatibility", "child_manifest_schema_version"),
                "pretraining_split_manifest.v999",
                "child_manifest_schema_version",
            ),
            (
                "normalization_schema",
                (
                    "compatibility",
                    "normalization_artifact_schema_version",
                ),
                "biophysical_normalization.v999",
                "normalization_artifact_schema_version",
            ),
            (
                "nested_test_fingerprint_schema",
                (
                    "compatibility",
                    "test_source_fingerprint",
                    "schema_version",
                ),
                "sequence_source_fingerprint.v999",
                "test source-fingerprint schema",
            ),
        )
        for label, key_path, value, expected_message in cases:
            with self.subTest(field=label):
                payload = copy.deepcopy(original_payload)
                self._set_nested_payload_value(payload, key_path, value)
                self._write_parent_manifest_payload(config, payload)
                with self.assertRaisesRegex(
                    GenomicSplitError,
                    expected_message,
                ):
                    validate_generated_artifacts(
                        config=config,
                        repository_root=str(self.repository_root),
                        mode="report",
                    )

    def test_parent_manifest_rejects_unrecognized_compatibility_claim(
        self,
    ) -> None:
        config, unused_config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        payload = self._read_parent_manifest_payload(config)
        payload["compatibility"]["unknown_schema_version"] = "unknown.v1"
        self._write_parent_manifest_payload(config, payload)

        with self.assertRaisesRegex(
            GenomicSplitError,
            "compatibility keys mismatch",
        ):
            validate_generated_artifacts(
                config=config,
                repository_root=str(self.repository_root),
                mode="report",
            )

    def test_complete_auditor_rebuilds_unchecked_child_overlap_fields(
        self,
    ) -> None:
        config, unused_config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        child_path = self.repository_root / config.child_manifest_path
        original_payload = json.loads(child_path.read_text(encoding="utf-8"))
        example = {
            "group_key": "AAAA",
            "training_sequences": ["AAAA"],
            "validation_sequences": ["AAAA"],
            "training_row_count": 1,
            "validation_row_count": 1,
        }
        cases = (
            (
                "exact_representative_examples",
                (
                    "overlap_audit",
                    "exact_sequence_overlap",
                    "representative_examples",
                ),
                [example],
            ),
            (
                "rc_training_row_count",
                (
                    "overlap_audit",
                    "reverse_complement_equivalent_overlap",
                    "training_row_count",
                ),
                1,
            ),
            (
                "rc_validation_row_count",
                (
                    "overlap_audit",
                    "reverse_complement_equivalent_overlap",
                    "validation_row_count",
                ),
                1,
            ),
            (
                "rc_representative_examples",
                (
                    "overlap_audit",
                    "reverse_complement_equivalent_overlap",
                    "representative_examples",
                ),
                [example],
            ),
        )
        for label, key_path, value in cases:
            with self.subTest(field=label):
                payload = copy.deepcopy(original_payload)
                self._set_nested_payload_value(payload, key_path, value)
                self._write_child_manifest_payload(config, payload)
                with self.assertRaisesRegex(
                    GenomicSplitError,
                    "reconstructed child manifest mismatch",
                ):
                    validate_generated_artifacts(
                        config=config,
                        repository_root=str(self.repository_root),
                        mode="report",
                    )

    def test_complete_auditor_reconciles_child_duplicate_metadata(
        self,
    ) -> None:
        config, unused_config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        child_path = self.repository_root / config.child_manifest_path
        original_payload = json.loads(child_path.read_text(encoding="utf-8"))
        fields = (
            "exact_sequence_duplicate_count",
            "reverse_complement_canonical_duplicate_count",
        )
        for field_name in fields:
            with self.subTest(field=field_name):
                payload = copy.deepcopy(original_payload)
                training_source = payload["training_source"]
                training_source[field_name] += 1
                source_content = dict(training_source)
                source_content.pop("fingerprint_hash", None)
                training_source["fingerprint_hash"] = hash_logical_content(
                    source_content
                )
                self._write_child_manifest_payload(config, payload)
                with self.assertRaisesRegex(
                    GenomicSplitError,
                    "child training fingerprint mismatch",
                ):
                    validate_generated_artifacts(
                        config=config,
                        repository_root=str(self.repository_root),
                        mode="report",
                    )

    def test_complete_auditor_binds_the_sampling_seed(self) -> None:
        config, unused_config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        alternate_payload = yaml.safe_load(
            (
                self.repository_root / "configs" / "split.yaml"
            ).read_text(encoding="utf-8")
        )
        alternate_payload["policy"]["sampling_seed"] = 43
        alternate_path = self._write_config(
            alternate_payload,
            name="alternate_seed.yaml",
        )

        with self.assertRaisesRegex(
            GenomicSplitError,
            "policy metadata mismatch",
        ):
            self._run_audit_cli(
                (
                    "--config",
                    str(alternate_path),
                    "--repository-root",
                    str(self.repository_root),
                    "--strict",
                )
            )

    def test_complete_auditor_rejects_valid_window_outside_seeded_stream(
        self,
    ) -> None:
        fasta_path = self._write_fasta(
            (
                ("chr1", "AAAAAAAAA"),
                ("chr21", "CCCCCCC"),
                ("chr22", "ACACACA"),
                ("chrX", "NNRNN"),
            )
        )
        payload = self._config_payload(
            fasta_path,
            candidate_multiplier=2,
        )
        config = SyntheticWholeChromosomeSplitConfig.from_yaml(
            str(self._write_config(payload))
        )
        scan = scan_fasta_reference(config, str(self.repository_root))
        sampled = sample_candidate_coordinates(config, scan)
        sampled_training_starts = {
            candidate.start for candidate in sampled["training"]
        }
        eligible_training_starts = []
        for interval in scan.intervals_for_split("training"):
            eligible_training_starts.extend(
                range(interval.start, interval.stop)
            )
        unused_start = next(
            start
            for start in eligible_training_starts
            if start not in sampled_training_starts
        )
        changed_training = list(sampled["training"])
        changed_training[0] = replace(
            changed_training[0],
            start=unused_start,
            end=unused_start + config.window_length,
        )
        substituted_candidates = dict(sampled)
        substituted_candidates["training"] = tuple(changed_training)

        with mock.patch(
            "src.genomic_splits.sample_candidate_coordinates",
            return_value=substituted_candidates,
        ):
            build_hg38_pretraining_split(
                config=config,
                repository_root=str(self.repository_root),
                dry_run=False,
            )

        with self.assertRaisesRegex(
            GenomicSplitError,
            "does not match the deterministic sampler",
        ):
            validate_generated_artifacts(
                config=config,
                repository_root=str(self.repository_root),
                mode="report",
            )

    def test_complete_auditor_requires_rejection_resolution(self) -> None:
        config, unused_config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        payload = self._read_parent_manifest_payload(config)
        del payload["rejections"]["resolution"]
        self._write_parent_manifest_payload(config, payload)

        with self.assertRaisesRegex(
            GenomicSplitError,
            "missing=.*resolution",
        ):
            validate_generated_artifacts(
                config=config,
                repository_root=str(self.repository_root),
                mode="report",
            )

    def test_complete_auditor_reconciles_rejection_resolution_counts(
        self,
    ) -> None:
        config, unused_config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        original_payload = self._read_parent_manifest_payload(config)
        cases = (
            (
                "rejected_record_count",
                ("rejected_candidate_record_count",),
            ),
            (
                "consumed_training_count",
                ("consumed_candidate_counts", "training"),
            ),
            (
                "maximum_training_count",
                ("maximum_candidate_counts", "training"),
            ),
        )
        for label, key_path in cases:
            with self.subTest(field=label):
                payload = copy.deepcopy(original_payload)
                resolution = payload["rejections"]["resolution"]
                if len(key_path) == 1:
                    key = key_path[0]
                    resolution[key] += 1
                else:
                    mapping = resolution[key_path[0]]
                    mapping[key_path[1]] += 1
                self._write_parent_manifest_payload(config, payload)
                with self.assertRaises(GenomicSplitError):
                    validate_generated_artifacts(
                        config=config,
                        repository_root=str(self.repository_root),
                        mode="report",
                    )

    def test_reference_validation_rejects_foreign_reference_identity(
        self,
    ) -> None:
        config, unused_config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        records_by_split = {}
        for split in SPLIT_NAMES:
            records_by_split[split] = list(
                read_genomic_records(
                    str(self.repository_root / config.records_path(split))
                )
            )
        original = records_by_split["training"][0]
        records_by_split["training"][0] = GenomicWindowRecord.create(
            reference_id="foreign:sha256:reference",
            chromosome=original.chromosome,
            start=original.start,
            end=original.end,
            sequence=original.sequence,
            split=original.split,
            selection_rank=original.selection_rank,
            split_policy_version=original.split_policy_version,
        )
        scan = scan_fasta_reference(config, str(self.repository_root))

        with self.assertRaisesRegex(
            GenomicSplitError,
            "reference identifier",
        ):
            validate_records_against_reference(
                records_by_split=records_by_split,
                config=config,
                scan=scan,
                repository_root=str(self.repository_root),
            )

    def test_complete_auditor_detects_unselected_reference_change_between_passes(
        self,
    ) -> None:
        config, unused_config_path = self._standard_config()
        build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        reference_path = self.repository_root / config.reference_path
        original_scan = scan_fasta_reference

        def scan_then_mutate(scan_config, repository_root):
            scan = original_scan(scan_config, repository_root)
            reference_stat = reference_path.stat()
            reference_text = reference_path.read_text(encoding="ascii")
            changed_text = reference_text.replace("NNRNN", "NNANN")
            self.assertNotEqual(changed_text, reference_text)
            reference_path.write_text(changed_text, encoding="ascii")
            os.utime(
                reference_path,
                ns=(
                    reference_stat.st_atime_ns,
                    scan.source_file_mtime_ns,
                ),
            )
            self.assertEqual(
                reference_path.stat().st_size,
                scan.source_file_size_bytes,
            )
            return scan

        with mock.patch(
            "src.genomic_splits.scan_fasta_reference",
            side_effect=scan_then_mutate,
        ):
            with self.assertRaisesRegex(
                GenomicSplitError,
                "changed between scan and audit validation",
            ):
                validate_generated_artifacts(
                    config=config,
                    repository_root=str(self.repository_root),
                    mode="report",
                )

    def test_builds_are_byte_deterministic_across_repository_roots(self) -> None:
        first_config, unused_path = self._standard_config()
        first_parent = build_hg38_pretraining_split(
            config=first_config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        first_output = self.repository_root / first_config.output_directory
        first_bytes = {
            path.name: path.read_bytes()
            for path in sorted(first_output.iterdir())
        }

        with tempfile.TemporaryDirectory() as second_directory:
            second_root = Path(second_directory)
            (second_root / "data" / "raw").mkdir(parents=True)
            (second_root / "configs").mkdir()
            second_fasta = second_root / "data" / "raw" / "reference.fa"
            first_fasta = self.repository_root / "data" / "raw" / "reference.fa"
            second_fasta.write_bytes(first_fasta.read_bytes())
            payload = self._config_payload(first_fasta)
            payload["reference"]["path"] = "data/raw/reference.fa"
            payload["reference"]["expected_raw_sha256"] = hash_file_bytes(
                str(second_fasta)
            )
            second_config_path = second_root / "configs" / "split.yaml"
            second_config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            second_config = SyntheticWholeChromosomeSplitConfig.from_yaml(
                str(second_config_path)
            )
            second_parent = build_hg38_pretraining_split(
                config=second_config,
                repository_root=str(second_root),
                dry_run=False,
            )
            second_output = second_root / second_config.output_directory
            second_bytes = {
                path.name: path.read_bytes()
                for path in sorted(second_output.iterdir())
            }

        self.assertEqual(first_parent, second_parent)
        self.assertEqual(first_bytes, second_bytes)

    def test_manifest_tampering_fails_integrity_validation(self) -> None:
        config, unused_config_path = self._standard_config()
        parent = build_hg38_pretraining_split(
            config=config,
            repository_root=str(self.repository_root),
            dry_run=False,
        )
        tampered = copy.deepcopy(parent)
        tampered["policy"]["window_length"] = 999

        with self.assertRaisesRegex(
            GenomicSplitError,
            "integrity hash mismatch",
        ):
            validate_genomic_manifest(tampered)


if __name__ == "__main__":
    unittest.main()
