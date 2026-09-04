"""Tests for the Milestone 3D-B split and subset command wrappers."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import yaml

from scripts.data_prep import build_exd_hox_nested_subsets
from scripts.data_prep import build_exd_hox_primary_split
from src.downstream_fingerprints import (
    build_hashed_manifest,
    hash_file_bytes,
    write_json_exclusive,
)
from src.exd_hox_splits import (
    ASSIGNMENTS_FILENAME,
    ExdHoxSplitError,
    SPLIT_MANIFEST_FILENAME,
    SUBSET_MANIFEST_FILENAME,
    SUBSET_LEVELS_FILENAME,
    validate_split_artifacts,
    validate_subset_artifacts,
)
from src.sealed_test_access import build_test_access_policy


def _one_hot(sequences) -> np.ndarray:
    channel_by_base = {"A": 0, "C": 1, "G": 2, "T": 3}
    values = np.zeros((len(sequences), 14, 4), dtype=np.int8)
    for row_index, sequence in enumerate(sequences):
        for base_index, base in enumerate(sequence):
            values[row_index, base_index, channel_by_base[base]] = 1
    return values


def _sequence_from_index(index: int) -> str:
    bases = "ACGT"
    digits = []
    remaining = index
    for _ in range(14):
        digits.append(bases[remaining % 4])
        remaining //= 4
    return "".join(reversed(digits))


def _write_hdf5(path: Path, sequences, targets) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sequence_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as output_file:
        data_group = output_file.create_group("data")
        target_group = output_file.create_group("targets")
        data_group.create_dataset(
            "sequence",
            data=np.asarray(sequences, dtype=object),
            dtype=sequence_dtype,
        )
        data_group.create_dataset("s_x", data=_one_hot(sequences))
        data_group.create_dataset(
            "c0_y",
            data=np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        )
        target_group.create_dataset(
            "id",
            data=np.asarray([b"c0"], dtype="S16"),
        )
        target_group.create_dataset(
            "name",
            data=np.asarray([b"dummy"], dtype="S16"),
        )


def _files_by_relative_path(directory: Path):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _fingerprints_by_relative_path(directory: Path):
    return {
        path.relative_to(directory).as_posix(): (
            path.stat().st_size,
            hash_file_bytes(path),
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "exd_hox_primary_split_v1.yaml"
REAL_HDF5_SENTINEL = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "exd_hox_selex_canonical_v1"
    / "AbdA"
    / "AbdA_train.h5"
)


class ExdHoxSplitCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_synthetic_repository(self, repository_root: Path) -> Path:
        config_path = repository_root / "configs" / "synthetic_split_v1.yaml"
        policy_path = (
            repository_root / "configs" / "exd_hox_test_access_policy_v1.yaml"
        )
        source_manifest_path = (
            repository_root
            / "data"
            / "processed"
            / "synthetic_audit_v1"
            / "source_manifest_v1.json"
        )
        audit_manifest_path = source_manifest_path.parent / "audit_manifest_v1.json"
        sequences = tuple(_sequence_from_index(index) for index in range(10))
        targets = tuple((index + 1) / 20 for index in range(10))
        source_files = []
        split_payloads = (
            ("train", sequences[:8], targets[:8]),
            (
                "test",
                (sequences[0], sequences[8], sequences[9]),
                (targets[0], targets[8], targets[9]),
            ),
        )
        for supplied_split, split_sequences, split_targets in split_payloads:
            logical_path = Path(
                "data",
                "raw",
                "synthetic_exd_hox_v1",
                "AbdA",
                "AbdA_{0}.h5".format(supplied_split),
            ).as_posix()
            physical_path = repository_root / logical_path
            _write_hdf5(physical_path, split_sequences, split_targets)
            source_files.append(
                {
                    "transcription_factor": "AbdA",
                    "supplied_split": supplied_split,
                    "imported_raw_path": logical_path,
                    "byte_size": physical_path.stat().st_size,
                    "sha256": hash_file_bytes(physical_path),
                }
            )
        source_files.sort(key=lambda row: row["supplied_split"])
        source_manifest = build_hashed_manifest(
            "exd_hox_source_manifest.v1",
            {
                "dataset_identifier": "synthetic_exd_hox.v1",
                "source_commit": "b" * 40,
                "files": source_files,
            },
        )
        write_json_exclusive(source_manifest_path, source_manifest)
        audit_manifest = build_hashed_manifest(
            "exd_hox_audit_manifest.v1",
            {
                "dataset_identifier": "synthetic_exd_hox.v1",
                "source_manifest_hash": source_manifest["manifest_hash"],
                "artifacts": [],
            },
        )
        write_json_exclusive(audit_manifest_path, audit_manifest)

        policy = build_test_access_policy()
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(
            yaml.safe_dump(policy, sort_keys=False),
            encoding="utf-8",
        )

        config = {
            "schema_version": "exd_hox_primary_split_config.v1",
            "study": {
                "identifier": "synthetic_exd_hox_low_data.v1",
                "dataset_identifier": "synthetic_exd_hox.v1",
                "project_commit": "a" * 40,
                "external_source_commit": "b" * 40,
            },
            "inputs": {
                "source_manifest_path": source_manifest_path.relative_to(
                    repository_root
                ).as_posix(),
                "source_manifest_hash": source_manifest["manifest_hash"],
                "source_manifest_file_sha256": hash_file_bytes(
                    source_manifest_path
                ),
                "audit_manifest_path": audit_manifest_path.relative_to(
                    repository_root
                ).as_posix(),
                "audit_manifest_hash": audit_manifest["manifest_hash"],
                "audit_manifest_file_sha256": hash_file_bytes(
                    audit_manifest_path
                ),
            },
            "dataset": {
                "sequence_length": 14,
                "transcription_factors": ["AbdA"],
            },
            "split_policy": {
                "identifier": "global_rc_affinity_stratified_80_10_10.v1",
                "seed": 31001,
                "split_order": ["training", "validation", "test"],
                "proportions": {
                    "training": 0.8,
                    "validation": 0.1,
                    "test": 0.1,
                },
                "affinity_bin_count": 10,
                "minimum_distinct_groups_per_bin": 7,
                "equal_distance_bin_merge": "lower",
                "quota_method": "deterministic_largest_remainder",
                "group_order": "seeded_sha256",
                "assignment_objective": "lexicographic_stratum_deficit_then_approved_margin_signature_exchange",
            },
            "subset_policy": {
                "seed": 32001,
                "absolute_counts": [128, 256, 512],
                "fractional_levels": [
                    0.01,
                    0.02,
                    0.05,
                    0.10,
                    0.25,
                    0.50,
                    1.00,
                ],
                "absolute_alias_tolerance": 0.05,
                "fractional_rounding": "round_half_up",
                "minimum_primary_count": 128,
                "downstream_initialization_seeds": [
                    33001,
                    33002,
                    33003,
                    33004,
                    33005,
                ],
            },
            "outputs": {
                "split_directory": "data/processed/exd_hox_primary_split_v1",
                "subset_directory": "data/processed/exd_hox_nested_subsets_v1",
                "sealed_target_directory": (
                    "data/sealed/exd_hox_primary_test_targets_v1"
                ),
                "plot_directory": "plots/exd_hox_primary_split_v1",
                "test_access_policy_path": (
                    "configs/exd_hox_test_access_policy_v1.yaml"
                ),
            },
            "test_access": {
                "policy_manifest_hash": policy["manifest_hash"],
                "policy_file_sha256": hash_file_bytes(policy_path),
            },
            "expected": {
                "source_occurrences": 11,
                "logical_examples": 10,
                "collapsed_duplicate_occurrences": 1,
                "collapsed_duplicate_occurrences_by_tf": {"AbdA": 1},
                "global_rc_groups": 10,
                "groups_shared_across_at_least_two_tfs": 0,
                "global_rc_group_degree_distribution": {1: 10},
                "global_rc_group_counts": {
                    "training": 8,
                    "validation": 1,
                    "test": 1,
                },
                "per_tf_logical_example_counts": {"AbdA": 10},
                "per_tf_split_counts": {
                    "AbdA": {
                        "training": 8,
                        "validation": 1,
                        "test": 1,
                    }
                },
            },
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        return config_path

    def _copy_synthetic_inputs(
        self,
        source_root: Path,
        destination_root: Path,
    ) -> Path:
        relative_paths = (
            "configs/synthetic_split_v1.yaml",
            "configs/exd_hox_test_access_policy_v1.yaml",
            "data/processed/synthetic_audit_v1/source_manifest_v1.json",
            "data/processed/synthetic_audit_v1/audit_manifest_v1.json",
            "data/raw/synthetic_exd_hox_v1/AbdA/AbdA_train.h5",
            "data/raw/synthetic_exd_hox_v1/AbdA/AbdA_test.h5",
        )
        for relative_path in relative_paths:
            source_path = source_root / relative_path
            destination_path = destination_root / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination_path)
        return destination_root / relative_paths[0]

    def test_primary_cli_defaults_to_checked_in_config(self) -> None:
        arguments = build_exd_hox_primary_split.parse_arguments([])

        self.assertEqual(
            arguments.config,
            "configs/exd_hox_primary_split_v1.yaml",
        )
        self.assertEqual(arguments.repository_root, ".")
        self.assertIsNone(arguments.output_directory)
        self.assertIsNone(arguments.sealed_target_directory)

        subset_arguments = build_exd_hox_nested_subsets.parse_arguments([])
        self.assertEqual(
            subset_arguments.config,
            "configs/exd_hox_primary_split_v1.yaml",
        )
        self.assertEqual(subset_arguments.repository_root, ".")
        self.assertIsNone(subset_arguments.split_directory)
        self.assertIsNone(subset_arguments.output_directory)

    def test_primary_cli_requires_both_physical_output_overrides(self) -> None:
        malformed_arguments = (
            ("--output-directory", str(self.root / "public")),
            (
                "--sealed-target-directory",
                str(self.root / "sealed"),
            ),
        )
        for arguments in malformed_arguments:
            with self.subTest(arguments=arguments):
                standard_error = io.StringIO()
                with redirect_stderr(standard_error):
                    with self.assertRaises(SystemExit) as raised:
                        build_exd_hox_primary_split.parse_arguments(arguments)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("must be provided together", standard_error.getvalue())

    def test_subset_cli_requires_both_physical_directory_overrides(self) -> None:
        malformed_arguments = (
            ("--split-directory", str(self.root / "split")),
            ("--output-directory", str(self.root / "subsets")),
        )
        for arguments in malformed_arguments:
            with self.subTest(arguments=arguments):
                standard_error = io.StringIO()
                with redirect_stderr(standard_error):
                    with self.assertRaises(SystemExit) as raised:
                        build_exd_hox_nested_subsets.parse_arguments(arguments)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("must be provided together", standard_error.getvalue())

    def test_primary_main_passes_temporary_roots_and_prints_manifest(self) -> None:
        public_directory = self.root / "temporary_public"
        sealed_directory = self.root / "temporary_sealed"
        manifest = {
            "schema_version": "exd_hox_primary_split_manifest.v1",
            "manifest_hash": "a" * 64,
        }
        standard_output = io.StringIO()

        with mock.patch.object(
            build_exd_hox_primary_split,
            "build_primary_split_artifacts",
            return_value=manifest,
        ) as build:
            with redirect_stdout(standard_output):
                result = build_exd_hox_primary_split.main(
                    (
                        "--config",
                        "configs/synthetic.yaml",
                        "--repository-root",
                        str(self.root),
                        "--output-directory",
                        str(public_directory),
                        "--sealed-target-directory",
                        str(sealed_directory),
                    )
                )

        self.assertEqual(result, manifest)
        build.assert_called_once_with(
            config_path="configs/synthetic.yaml",
            repository_root=str(self.root),
            output_directory=str(public_directory),
            sealed_target_directory=str(sealed_directory),
        )
        self.assertEqual(
            standard_output.getvalue(),
            json.dumps(
                manifest,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
        )
        self.assertFalse(
            (self.root / "data" / "processed" / "exd_hox_primary_split_v1").exists()
        )
        self.assertFalse(
            (
                self.root
                / "data"
                / "sealed"
                / "exd_hox_primary_test_targets_v1"
            ).exists()
        )

    def test_subset_main_passes_temporary_roots_and_prints_manifest(self) -> None:
        split_directory = self.root / "temporary_split"
        subset_directory = self.root / "temporary_subsets"
        manifest = {
            "schema_version": "exd_hox_subset_set_manifest.v1",
            "manifest_hash": "b" * 64,
        }
        standard_output = io.StringIO()

        with mock.patch.object(
            build_exd_hox_nested_subsets,
            "build_nested_subset_artifacts",
            return_value=manifest,
        ) as build:
            with redirect_stdout(standard_output):
                result = build_exd_hox_nested_subsets.main(
                    (
                        "--config",
                        "configs/synthetic.yaml",
                        "--repository-root",
                        str(self.root),
                        "--split-directory",
                        str(split_directory),
                        "--output-directory",
                        str(subset_directory),
                    )
                )

        self.assertEqual(result, manifest)
        build.assert_called_once_with(
            config_path="configs/synthetic.yaml",
            repository_root=str(self.root),
            split_directory=str(split_directory),
            output_directory=str(subset_directory),
        )
        self.assertEqual(
            standard_output.getvalue(),
            json.dumps(
                manifest,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
        )
        self.assertFalse(
            (
                self.root
                / "data"
                / "processed"
                / "exd_hox_nested_subsets_v1"
            ).exists()
        )

    def test_cli_has_no_force_option_and_does_not_retry_exclusive_failure(
        self,
    ) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_exd_hox_primary_split.parse_arguments(("--force",))
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_exd_hox_nested_subsets.parse_arguments(("--force",))

        with mock.patch.object(
            build_exd_hox_primary_split,
            "build_primary_split_artifacts",
            side_effect=FileExistsError("exclusive destination exists"),
        ) as build:
            with self.assertRaisesRegex(FileExistsError, "exclusive"):
                build_exd_hox_primary_split.main(
                    (
                        "--output-directory",
                        str(self.root / "public"),
                        "--sealed-target-directory",
                        str(self.root / "sealed"),
                    )
                )
        build.assert_called_once()

    def test_synthetic_artifacts_are_exclusive_temporary_and_deterministic(
        self,
    ) -> None:
        first_root = self.root / "first_repository"
        second_root = self.root / "second_repository"
        first_config = self._write_synthetic_repository(first_root)
        second_config = self._copy_synthetic_inputs(first_root, second_root)
        run_directories = []
        for repository_root, config_path in (
            (first_root, first_config),
            (second_root, second_config),
        ):
            artifact_root = repository_root / "temporary_artifacts"
            public_directory = artifact_root / "public"
            sealed_directory = artifact_root / "sealed"
            subset_directory = artifact_root / "subsets"
            with redirect_stdout(io.StringIO()):
                split_manifest = build_exd_hox_primary_split.main(
                    (
                        "--config",
                        "configs/synthetic_split_v1.yaml",
                        "--repository-root",
                        str(repository_root),
                        "--output-directory",
                        str(public_directory),
                        "--sealed-target-directory",
                        str(sealed_directory),
                    )
                )
                subset_manifest = build_exd_hox_nested_subsets.main(
                    (
                        "--config",
                        "configs/synthetic_split_v1.yaml",
                        "--repository-root",
                        str(repository_root),
                        "--split-directory",
                        str(public_directory),
                        "--output-directory",
                        str(subset_directory),
                    )
                )
            validated_split = validate_split_artifacts(
                public_directory / SPLIT_MANIFEST_FILENAME,
                repository_root,
                split_directory=public_directory,
            )
            validated_subsets = validate_subset_artifacts(
                subset_directory / SUBSET_MANIFEST_FILENAME,
                repository_root,
                subset_directory=subset_directory,
                expected_split_manifest_hash=split_manifest["manifest_hash"],
            )
            self.assertEqual(validated_split, split_manifest)
            self.assertEqual(validated_subsets, subset_manifest)
            self.assertFalse(
                (
                    repository_root
                    / "data"
                    / "processed"
                    / "exd_hox_primary_split_v1"
                ).exists()
            )
            self.assertFalse(
                (
                    repository_root
                    / "data"
                    / "processed"
                    / "exd_hox_nested_subsets_v1"
                ).exists()
            )
            self.assertFalse(
                (
                    repository_root
                    / "data"
                    / "sealed"
                    / "exd_hox_primary_test_targets_v1"
                ).exists()
            )
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    build_exd_hox_primary_split.main(
                        (
                            "--config",
                            "configs/synthetic_split_v1.yaml",
                            "--repository-root",
                            str(repository_root),
                            "--output-directory",
                            str(public_directory),
                            "--sealed-target-directory",
                            str(sealed_directory),
                        )
                    )
                with self.assertRaises(FileExistsError):
                    build_exd_hox_nested_subsets.main(
                        (
                            "--config",
                            "configs/synthetic_split_v1.yaml",
                            "--repository-root",
                            str(repository_root),
                            "--split-directory",
                            str(public_directory),
                            "--output-directory",
                            str(subset_directory),
                        )
                    )
            run_directories.append(
                (public_directory, sealed_directory, subset_directory)
            )

        first_directories, second_directories = run_directories
        for first_directory, second_directory in zip(
            first_directories,
            second_directories,
        ):
            self.assertEqual(
                _files_by_relative_path(first_directory),
                _files_by_relative_path(second_directory),
            )

        public_directory = first_directories[0]
        assignment_path = public_directory / ASSIGNMENTS_FILENAME
        assignment_path.write_bytes(assignment_path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ExdHoxSplitError, "byte identity"):
            validate_split_artifacts(
                public_directory / SPLIT_MANIFEST_FILENAME,
                first_root,
                split_directory=public_directory,
            )

    def test_primary_publish_rolls_back_sealed_target_on_second_rename_failure(
        self,
    ) -> None:
        repository_root = self.root / "rollback_repository"
        self._write_synthetic_repository(repository_root)
        public_directory = self.root / "rollback_artifacts" / "public"
        sealed_directory = self.root / "rollback_artifacts" / "sealed"
        original_rename = os.rename
        call_count = 0

        def fail_second_rename(source, destination):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("simulated public publication failure")
            return original_rename(source, destination)

        with mock.patch(
            "src.exd_hox_splits.os.rename",
            side_effect=fail_second_rename,
        ):
            with self.assertRaisesRegex(OSError, "simulated public"):
                build_exd_hox_primary_split.main(
                    (
                        "--config",
                        "configs/synthetic_split_v1.yaml",
                        "--repository-root",
                        str(repository_root),
                        "--output-directory",
                        str(public_directory),
                        "--sealed-target-directory",
                        str(sealed_directory),
                    )
                )

        self.assertEqual(call_count, 3)
        self.assertFalse(public_directory.exists())
        self.assertFalse(sealed_directory.exists())

    def test_public_validator_rejects_unexpected_files(self) -> None:
        repository_root = self.root / "unexpected_file_repository"
        self._write_synthetic_repository(repository_root)
        public_directory = self.root / "unexpected_artifacts" / "public"
        sealed_directory = self.root / "unexpected_artifacts" / "sealed"
        with redirect_stdout(io.StringIO()):
            build_exd_hox_primary_split.main(
                (
                    "--config",
                    "configs/synthetic_split_v1.yaml",
                    "--repository-root",
                    str(repository_root),
                    "--output-directory",
                    str(public_directory),
                    "--sealed-target-directory",
                    str(sealed_directory),
                )
            )
        (public_directory / "unexpected.tsv").write_text(
            "plaintext_target\n0.75\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ExdHoxSplitError, "file set differs"):
            validate_split_artifacts(
                public_directory / SPLIT_MANIFEST_FILENAME,
                repository_root,
                split_directory=public_directory,
            )

    def test_subset_output_cannot_contaminate_finalized_split(self) -> None:
        repository_root = self.root / "disjoint_repository"
        self._write_synthetic_repository(repository_root)
        public_directory = self.root / "disjoint_artifacts" / "public"
        sealed_directory = self.root / "disjoint_artifacts" / "sealed"
        with redirect_stdout(io.StringIO()):
            build_exd_hox_primary_split.main(
                (
                    "--config",
                    "configs/synthetic_split_v1.yaml",
                    "--repository-root",
                    str(repository_root),
                    "--output-directory",
                    str(public_directory),
                    "--sealed-target-directory",
                    str(sealed_directory),
                )
            )

        with self.assertRaisesRegex(ExdHoxSplitError, "must be disjoint"):
            build_exd_hox_nested_subsets.main(
                (
                    "--config",
                    "configs/synthetic_split_v1.yaml",
                    "--repository-root",
                    str(repository_root),
                    "--split-directory",
                    str(public_directory),
                    "--output-directory",
                    str(public_directory / "subsets"),
                )
            )
        self.assertFalse((public_directory / "subsets").exists())

    @unittest.skipUnless(
        REAL_HDF5_SENTINEL.is_file(),
        "ignored Milestone 3C HDF5 inputs are unavailable",
    )
    def test_real_hdf5_two_temporary_builds_are_exact_and_byte_identical(
        self,
    ) -> None:
        production_paths = (
            REPOSITORY_ROOT
            / "data"
            / "processed"
            / "exd_hox_primary_split_v1",
            REPOSITORY_ROOT
            / "data"
            / "processed"
            / "exd_hox_nested_subsets_v1",
            REPOSITORY_ROOT
            / "data"
            / "sealed"
            / "exd_hox_primary_test_targets_v1",
        )
        for production_path in production_paths:
            self.assertFalse(production_path.exists())
            self.assertFalse(production_path.is_symlink())

        config = yaml.safe_load(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        expected = config["expected"]
        run_directories = []
        split_manifests = []
        subset_manifests = []
        for run_name in ("first_real_run", "second_real_run"):
            run_root = self.root / run_name
            public_directory = run_root / "public"
            sealed_directory = run_root / "sealed"
            subset_directory = run_root / "subsets"
            with redirect_stdout(io.StringIO()):
                split_manifest = build_exd_hox_primary_split.main(
                    (
                        "--config",
                        str(REAL_CONFIG_PATH),
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                        "--output-directory",
                        str(public_directory),
                        "--sealed-target-directory",
                        str(sealed_directory),
                    )
                )
                subset_manifest = build_exd_hox_nested_subsets.main(
                    (
                        "--config",
                        str(REAL_CONFIG_PATH),
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                        "--split-directory",
                        str(public_directory),
                        "--output-directory",
                        str(subset_directory),
                    )
                )
            validate_split_artifacts(
                public_directory / SPLIT_MANIFEST_FILENAME,
                REPOSITORY_ROOT,
                split_directory=public_directory,
            )
            validate_subset_artifacts(
                subset_directory / SUBSET_MANIFEST_FILENAME,
                REPOSITORY_ROOT,
                subset_directory=subset_directory,
                expected_split_manifest_hash=split_manifest["manifest_hash"],
            )
            run_directories.append(
                (public_directory, sealed_directory, subset_directory)
            )
            split_manifests.append(split_manifest)
            subset_manifests.append(subset_manifest)

        counts = split_manifests[0]["counts"]
        for key in (
            "source_occurrences",
            "logical_examples",
            "collapsed_duplicate_occurrences",
            "collapsed_duplicate_occurrences_by_tf",
            "global_rc_groups",
            "groups_shared_across_at_least_two_tfs",
            "global_rc_group_counts",
            "per_tf_logical_example_counts",
            "per_tf_split_counts",
        ):
            self.assertEqual(counts[key], expected[key])
        expected_degree_distribution = {
            str(degree): count
            for degree, count in expected[
                "global_rc_group_degree_distribution"
            ].items()
        }
        self.assertEqual(
            counts["global_rc_group_degree_distribution"],
            expected_degree_distribution,
        )
        expected_training_count = sum(
            split_counts["training"]
            for split_counts in expected["per_tf_split_counts"].values()
        )
        self.assertEqual(
            subset_manifests[0]["ordering_logical_example_count"],
            expected_training_count,
        )
        self.assertEqual(subset_manifests[0]["level_row_count"], 80)

        expected_fractional_counts = {
            "AbdA": (201, 401, 1003, 2005, 5013, 10025, 20050),
            "AbdB": (332, 663, 1659, 3317, 8293, 16585, 33170),
            "Antp": (256, 512, 1231, 2461, 6153, 12305, 24610),
            "Dfd": (213, 426, 1066, 2131, 5328, 10656, 21311),
            "Lab": (256, 512, 1302, 2605, 6512, 13025, 26049),
            "Pb": (1824, 3649, 9121, 18243, 45607, 91214, 182428),
            "Scr": (276, 552, 1379, 2759, 6897, 13794, 27588),
            "Ubx": (340, 679, 1699, 3397, 8493, 16986, 33971),
        }
        fraction_order = ("0.01", "0.02", "0.05", "0.1", "0.25", "0.5", "1.0")
        level_path = run_directories[0][2] / SUBSET_LEVELS_FILENAME
        with open(level_path, "r", encoding="utf-8", newline="") as input_file:
            level_rows = tuple(csv.DictReader(input_file, delimiter="\t"))
        for transcription_factor, expected_counts in (
            expected_fractional_counts.items()
        ):
            fractional_by_value = {
                row["request_value"]: row
                for row in level_rows
                if row["transcription_factor"] == transcription_factor
                and row["request_type"] == "fractional"
            }
            observed_counts = tuple(
                int(
                    fractional_by_value[value][
                        "canonical_requested_logical_example_count"
                    ]
                )
                for value in fraction_order
            )
            self.assertEqual(observed_counts, expected_counts)
        for row in level_rows:
            self.assertEqual(
                int(row["actual_logical_example_count"]),
                int(row["canonical_requested_logical_example_count"]),
            )
            self.assertEqual(
                int(row["actual_rc_group_count"]),
                int(row["actual_logical_example_count"]),
            )

        for first_directory, second_directory in zip(
            run_directories[0],
            run_directories[1],
        ):
            self.assertEqual(
                _fingerprints_by_relative_path(first_directory),
                _fingerprints_by_relative_path(second_directory),
            )
        self.assertEqual(split_manifests[0], split_manifests[1])
        self.assertEqual(subset_manifests[0], subset_manifests[1])
        for production_path in production_paths:
            self.assertFalse(production_path.exists())
            self.assertFalse(production_path.is_symlink())


if __name__ == "__main__":
    unittest.main()
