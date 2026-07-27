"""Tests for deterministic pretraining source and split fingerprints."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.data_fingerprints import (
    PretrainingSplitManifest,
    SourceFingerprint,
    SplitOverlapError,
    build_pretraining_split_manifest,
    fingerprint_sequence_file,
    load_split_manifest,
    save_split_manifest,
    validate_new_artifact_output_path,
)


class DataFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repository_root = Path(self.temporary_directory.name)
        self.data_directory = self.repository_root / "data"
        self.data_directory.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_sequences(self, name: str, sequences) -> Path:
        path = self.data_directory / name
        with open(path, "w", encoding="utf-8", newline="\n") as output_file:
            for sequence in sequences:
                output_file.write(sequence)
                output_file.write("\n")
        return path

    def test_source_and_manifest_hashes_are_deterministic(self) -> None:
        training_path = self._write_sequences(
            "training.txt",
            ("ACGT", "AAAA", "AAAA", "AANN"),
        )
        validation_path = self._write_sequences(
            "validation.txt",
            ("CCCC", "CGCG"),
        )

        first_fingerprint = fingerprint_sequence_file(
            str(training_path),
            str(self.repository_root),
        )
        second_fingerprint = fingerprint_sequence_file(
            str(training_path),
            str(self.repository_root),
        )
        first_manifest = build_pretraining_split_manifest(
            str(training_path),
            str(validation_path),
            str(self.repository_root),
        )
        second_manifest = build_pretraining_split_manifest(
            str(training_path),
            str(validation_path),
            str(self.repository_root),
        )

        self.assertEqual(
            first_fingerprint.to_dict(),
            second_fingerprint.to_dict(),
        )
        self.assertEqual(first_manifest.to_dict(), second_manifest.to_dict())
        self.assertEqual(
            first_manifest.manifest_hash,
            second_manifest.manifest_hash,
        )

        first_output = self.repository_root / "first_manifest.json"
        second_output = self.repository_root / "second_manifest.json"
        save_split_manifest(first_manifest, str(first_output))
        save_split_manifest(second_manifest, str(second_output))
        self.assertEqual(first_output.read_bytes(), second_output.read_bytes())

    def test_source_fingerprint_records_duplicates_lengths_and_n(self) -> None:
        source_path = self._write_sequences(
            "source.txt",
            ("AAAA", "AAAA", "TTTT", "ACN"),
        )

        fingerprint = fingerprint_sequence_file(
            str(source_path),
            str(self.repository_root),
        )

        self.assertEqual(fingerprint.source_path, "data/source.txt")
        self.assertEqual(fingerprint.total_rows, 4)
        self.assertEqual(fingerprint.unique_sequences, 3)
        self.assertEqual(fingerprint.minimum_sequence_length, 3)
        self.assertEqual(fingerprint.maximum_sequence_length, 4)
        self.assertEqual(fingerprint.observed_sequence_lengths, (3, 4))
        self.assertEqual(fingerprint.sequences_containing_n, 1)
        self.assertEqual(fingerprint.exact_sequence_duplicate_count, 1)
        self.assertEqual(
            fingerprint.reverse_complement_canonical_duplicate_count,
            2,
        )

    def test_raw_and_logical_sequence_hashes_have_distinct_meanings(
        self,
    ) -> None:
        first_path = self.data_directory / "first.txt"
        second_path = self.data_directory / "second.txt"
        first_path.write_bytes(b"acgt\n  aaaa  \n")
        second_path.write_bytes(b"ACGT\r\nAAAA\r\n")

        first = fingerprint_sequence_file(
            str(first_path),
            str(self.repository_root),
        )
        second = fingerprint_sequence_file(
            str(second_path),
            str(self.repository_root),
        )

        self.assertNotEqual(first.source_file_sha256, second.source_file_sha256)
        self.assertEqual(
            first.source_file_sha256,
            hashlib.sha256(first_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            second.source_file_sha256,
            hashlib.sha256(second_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            first.logical_sequence_sha256,
            second.logical_sequence_sha256,
        )
        self.assertNotEqual(first.fingerprint_hash, second.fingerprint_hash)

    def test_old_schema_versions_fail_before_v2_fields_are_read(self) -> None:
        training_path = self._write_sequences("training.txt", ("AAAA",))
        validation_path = self._write_sequences("validation.txt", ("CCCC",))
        manifest = build_pretraining_split_manifest(
            str(training_path),
            str(validation_path),
            str(self.repository_root),
        )

        source_payload = manifest.training_source.to_dict()
        source_payload["schema_version"] = "sequence_source_fingerprint.v1"
        source_payload.pop("logical_sequence_sha256")
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported source-fingerprint schema version",
        ):
            SourceFingerprint.from_dict(source_payload)

        manifest_payload = manifest.to_dict()
        manifest_payload["schema_version"] = "pretraining_split_manifest.v1"
        manifest_payload["training_source"].pop("logical_sequence_sha256")
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported split-manifest schema version",
        ):
            PretrainingSplitManifest.from_dict(manifest_payload)

    def test_clean_train_validation_split_has_no_overlap(self) -> None:
        training_path = self._write_sequences(
            "training.txt",
            ("AAAA", "CCCC"),
        )
        validation_path = self._write_sequences(
            "validation.txt",
            ("ACAC", "AGAG"),
        )

        manifest = build_pretraining_split_manifest(
            str(training_path),
            str(validation_path),
            str(self.repository_root),
            mode="strict",
        )

        self.assertFalse(manifest.overlap_audit.has_overlap)
        self.assertEqual(
            manifest.overlap_audit.exact_sequence_overlap.group_count,
            0,
        )
        self.assertEqual(
            manifest.overlap_audit.reverse_complement_equivalent_overlap.group_count,
            0,
        )

    def test_report_mode_records_exact_and_reverse_complement_overlap(
        self,
    ) -> None:
        training_path = self._write_sequences(
            "training.txt",
            ("AAAA", "AGTC", "CCCC"),
        )
        validation_path = self._write_sequences(
            "validation.txt",
            ("AAAA", "GACT", "ACAC"),
        )

        manifest = build_pretraining_split_manifest(
            str(training_path),
            str(validation_path),
            str(self.repository_root),
            mode="report",
        )
        audit = manifest.overlap_audit

        self.assertEqual(audit.exact_sequence_overlap.group_count, 1)
        self.assertEqual(
            audit.reverse_complement_equivalent_overlap.group_count,
            2,
        )
        self.assertEqual(audit.reverse_complement_only_group_count, 1)
        self.assertEqual(
            audit.exact_sequence_overlap.representative_examples[0].group_key,
            "AAAA",
        )
        reverse_examples = (
            audit.reverse_complement_equivalent_overlap.representative_examples
        )
        self.assertEqual(reverse_examples[1].training_sequences, ("AGTC",))
        self.assertEqual(reverse_examples[1].validation_sequences, ("GACT",))

    def test_duplicate_rows_and_palindromes_do_not_inflate_group_counts(
        self,
    ) -> None:
        training_path = self._write_sequences(
            "training.txt",
            ("AAAA", "AAAA", "ATAT"),
        )
        validation_path = self._write_sequences(
            "validation.txt",
            ("AAAA", "ATAT"),
        )

        manifest = build_pretraining_split_manifest(
            str(training_path),
            str(validation_path),
            str(self.repository_root),
            mode="report",
        )
        audit = manifest.overlap_audit

        self.assertEqual(audit.exact_sequence_overlap.group_count, 2)
        self.assertEqual(audit.exact_sequence_overlap.training_row_count, 3)
        self.assertEqual(audit.exact_sequence_overlap.validation_row_count, 2)
        self.assertEqual(
            audit.reverse_complement_equivalent_overlap.group_count,
            2,
        )
        self.assertEqual(audit.reverse_complement_only_group_count, 0)

    def test_strict_mode_rejects_cross_split_overlap(self) -> None:
        training_path = self._write_sequences(
            "training.txt",
            ("AAAA", "AGTC"),
        )
        validation_path = self._write_sequences(
            "validation.txt",
            ("GACT",),
        )

        with self.assertRaisesRegex(
            SplitOverlapError,
            "reverse-complement-equivalent",
        ):
            build_pretraining_split_manifest(
                str(training_path),
                str(validation_path),
                str(self.repository_root),
                mode="strict",
            )

    def test_manifest_round_trip_validates_hashes(self) -> None:
        training_path = self._write_sequences("training.txt", ("AAAA",))
        validation_path = self._write_sequences("validation.txt", ("CCCC",))
        manifest = build_pretraining_split_manifest(
            str(training_path),
            str(validation_path),
            str(self.repository_root),
        )
        output_path = self.repository_root / "manifest.json"
        save_split_manifest(manifest, str(output_path))

        loaded = load_split_manifest(str(output_path))

        self.assertEqual(loaded.to_dict(), manifest.to_dict())

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["training_source"]["total_rows"] = 99
        output_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "fingerprint hash mismatch"):
            load_split_manifest(str(output_path))

    def test_audit_does_not_modify_input_files(self) -> None:
        training_path = self._write_sequences(
            "training.txt",
            ("AAAA", "CCCC"),
        )
        validation_path = self._write_sequences(
            "validation.txt",
            ("ACAC",),
        )
        training_before = training_path.read_bytes()
        validation_before = validation_path.read_bytes()

        build_pretraining_split_manifest(
            str(training_path),
            str(validation_path),
            str(self.repository_root),
        )

        self.assertEqual(training_path.read_bytes(), training_before)
        self.assertEqual(validation_path.read_bytes(), validation_before)

    def test_artifact_output_paths_are_restricted_and_versioned(self) -> None:
        valid_path = self.repository_root / "logs" / "manifest_v1.json"
        validated = validate_new_artifact_output_path(
            str(valid_path),
            str(self.repository_root),
            allowed_relative_directories=("logs",),
        )
        self.assertEqual(validated, str(valid_path.resolve()))

        with self.assertRaisesRegex(ValueError, "contain a version"):
            validate_new_artifact_output_path(
                str(self.repository_root / "logs" / "manifest.json"),
                str(self.repository_root),
                allowed_relative_directories=("logs",),
            )
        with self.assertRaisesRegex(ValueError, "under one of"):
            validate_new_artifact_output_path(
                str(self.repository_root / "data" / "manifest_v1.json"),
                str(self.repository_root),
                allowed_relative_directories=("logs",),
            )
        with self.assertRaisesRegex(ValueError, "outside repository"):
            validate_new_artifact_output_path(
                str(self.repository_root.parent / "manifest_v1.json"),
                str(self.repository_root),
                allowed_relative_directories=("logs",),
            )


if __name__ == "__main__":
    unittest.main()
