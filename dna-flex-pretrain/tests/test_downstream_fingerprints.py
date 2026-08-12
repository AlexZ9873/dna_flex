"""Tests for deterministic downstream artifact fingerprints."""

import json
from pathlib import Path
import tempfile
import unittest

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
)


class DownstreamFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_logical_hash_and_manifest_are_deterministic(self) -> None:
        first_content = {"b": [2, 3], "a": 1}
        second_content = {"a": 1, "b": [2, 3]}

        self.assertEqual(
            hash_logical_content(first_content),
            hash_logical_content(second_content),
        )
        first_manifest = build_hashed_manifest(
            "downstream_test_manifest.v1",
            first_content,
        )
        second_manifest = build_hashed_manifest(
            "downstream_test_manifest.v1",
            second_content,
        )
        self.assertEqual(first_manifest, second_manifest)
        validate_hashed_manifest(first_manifest)

        tampered = dict(first_manifest)
        tampered["a"] = 2
        with self.assertRaisesRegex(ValueError, "Manifest hash mismatch"):
            validate_hashed_manifest(tampered)

    def test_file_fingerprint_records_only_logical_relative_path(self) -> None:
        artifact = self.root / "data" / "artifact.tsv"
        artifact.parent.mkdir()
        artifact.write_bytes(b"a\tb\n1\t2\n")

        fingerprint = fingerprint_file(
            artifact,
            "data/artifact.tsv",
        )

        self.assertEqual(fingerprint.path, "data/artifact.tsv")
        self.assertEqual(fingerprint.byte_size, len(artifact.read_bytes()))
        self.assertEqual(fingerprint.sha256, hash_file_bytes(artifact))
        self.assertNotIn(str(self.root), json.dumps(fingerprint.to_dict()))
        self.assertEqual(
            repository_relative_path(artifact, self.root),
            "data/artifact.tsv",
        )

    def test_absolute_and_parent_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            validate_repository_relative_path("/tmp/output.json")
        with self.assertRaisesRegex(ValueError, "escape"):
            validate_repository_relative_path("../output.json")

    def test_exclusive_json_and_tsv_writers_refuse_overwrite(self) -> None:
        json_path = self.root / "manifest_v1.json"
        tsv_path = self.root / "table_v1.tsv"
        write_json_exclusive(json_path, {"schema_version": "test.v1"})
        write_tsv_exclusive(
            tsv_path,
            ("name", "count"),
            ({"name": "A", "count": 1},),
        )

        with self.assertRaises(FileExistsError):
            write_json_exclusive(json_path, {"schema_version": "test.v1"})
        with self.assertRaises(FileExistsError):
            write_tsv_exclusive(
                tsv_path,
                ("name", "count"),
                ({"name": "A", "count": 1},),
            )


if __name__ == "__main__":
    unittest.main()
