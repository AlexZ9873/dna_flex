"""Tests for the v2 Exd-Hox primary-split plot provenance contract."""

import builtins
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import yaml

import scripts.plot.plot_exd_hox_primary_split as plot_module
from scripts.plot.plot_exd_hox_primary_split import (
    AFFINITY_INPUT_FIELDS,
    AFFINITY_INPUT_FILENAME,
    AFFINITY_SOURCE_FILENAME,
    COUNT_INPUT_FIELDS,
    COUNT_INPUT_FILENAME,
    COUNT_SOURCE_FILENAME,
    COMPARISON_SOURCE_FILENAME,
    EXTERNAL_SOURCE_COMMIT,
    LEAKAGE_INPUT_FIELDS,
    LEAKAGE_INPUT_FILENAME,
    LEAKAGE_SOURCE_FILENAME,
    OUTPUT_FILENAMES,
    PLOT_CONFIG_LOGICAL_PATH,
    PLOT_LOGICAL_DIRECTORY,
    PLOT_MANIFEST_FILENAME,
    PLOTTING_ENTRY_POINT,
    PRIMARY_CONFIG_LOGICAL_PATH,
    RAW_FILE_SHA256,
    SOURCE_FOUNDATION_COMMIT,
    SPLIT_MANIFEST_FILENAME,
    SPLIT_PIPELINE_COMMIT,
    SUBSET_INPUT_FIELDS,
    SUBSET_INPUT_FILENAME,
    SUBSET_MANIFEST_FILENAME,
    SUBSET_SOURCE_FILENAME,
    main,
    plot_primary_split_tables,
    validate_primary_split_plot_manifest,
)
from src.downstream_fingerprints import (
    build_hashed_manifest,
    fingerprint_file,
    hash_file_bytes,
    write_json_exclusive,
    write_tsv_exclusive,
)


TRANSCRIPTION_FACTORS = ("AbdA", "Ubx")
SYNTHETIC_STUDY_ID = "synthetic_exd_hox_study.v1"
SYNTHETIC_DATASET_ID = "synthetic_exd_hox_primary.v1"
SYNTHETIC_SPLIT_ID = "a" * 64
SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExdHoxPrimarySplitPlotTests(unittest.TestCase):
    """Exercise generation and later validation in nested temporary Git repos."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.template_temporary_directory = tempfile.TemporaryDirectory()
        template_parent = Path(cls.template_temporary_directory.name)
        source_git_root = cls._command_output(
            ("git", "-C", str(SOURCE_PROJECT_ROOT), "rev-parse", "--show-toplevel"),
            SOURCE_PROJECT_ROOT,
        )
        cls.project_prefix = SOURCE_PROJECT_ROOT.relative_to(
            Path(source_git_root).resolve()
        )
        cls.template_git_root = template_parent / "template_git"
        cls._run_command(
            (
                "git",
                "clone",
                "--shared",
                "--quiet",
                source_git_root,
                str(cls.template_git_root),
            ),
            template_parent,
        )
        cls.template_root = cls.template_git_root / cls.project_prefix
        cls._configure_git_identity(cls.template_git_root)

        producer_source = SOURCE_PROJECT_ROOT / PLOTTING_ENTRY_POINT
        producer_destination = cls.template_root / PLOTTING_ENTRY_POINT
        producer_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(producer_source, producer_destination)

        primary_config_path = cls.template_root / PRIMARY_CONFIG_LOGICAL_PATH
        cls._write_primary_config(primary_config_path)
        finalized = cls._write_finalized_inputs(cls.template_root)
        plot_config_path = cls.template_root / PLOT_CONFIG_LOGICAL_PATH
        cls._write_plot_config(plot_config_path, finalized)

        cls.template_sentinel = cls.template_git_root / "tracked_scope_sentinel.txt"
        cls.template_sentinel.write_text("clean\n", encoding="utf-8")
        tracked_paths = (
            cls.project_prefix / PLOTTING_ENTRY_POINT,
            cls.project_prefix / PRIMARY_CONFIG_LOGICAL_PATH,
            cls.project_prefix / PLOT_CONFIG_LOGICAL_PATH,
            Path("tracked_scope_sentinel.txt"),
        )
        command = ["git", "add", "--force", "--"]
        for tracked_path in tracked_paths:
            command.append(tracked_path.as_posix())
        cls._run_command(tuple(command), cls.template_git_root)
        cls._commit_at_fixed_time(
            cls.template_git_root,
            "Synthetic v2 plot generator",
            "2026-09-04T12:00:00+00:00",
        )
        cls.template_generator_commit = cls._command_output(
            ("git", "rev-parse", "HEAD"),
            cls.template_git_root,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.template_temporary_directory.cleanup()

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary_directory.name)
        self.git_root = temporary_root / "flex"
        self._run_command(
            (
                "git",
                "clone",
                "--shared",
                "--quiet",
                str(self.template_git_root),
                str(self.git_root),
            ),
            temporary_root,
        )
        self.root = self.git_root / self.project_prefix
        self._configure_git_identity(self.git_root)
        self.finalized = self._write_finalized_inputs(self.root)
        self.config_path = self.root / PLOT_CONFIG_LOGICAL_PATH
        self.generator_commit = self._head()
        self.plot_directory = self.root / PLOT_LOGICAL_DIRECTORY
        self.manifest_path = self.plot_directory / PLOT_MANIFEST_FILENAME
        self._write_forbidden_sentinels()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _run_command(arguments, cwd, environment=None):
        result = subprocess.run(
            arguments,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace")
            raise AssertionError(
                "Command failed ({0}): {1}".format(result.returncode, message)
            )
        return result

    @classmethod
    def _command_output(cls, arguments, cwd):
        result = cls._run_command(arguments, cwd)
        return result.stdout.decode("utf-8").strip()

    @classmethod
    def _configure_git_identity(cls, git_root):
        cls._run_command(
            ("git", "config", "user.name", "Plot Contract Test"),
            git_root,
        )
        cls._run_command(
            ("git", "config", "user.email", "plot-contract@example.invalid"),
            git_root,
        )

    @classmethod
    def _commit_at_fixed_time(cls, git_root, message, timestamp):
        environment = os.environ.copy()
        environment["GIT_AUTHOR_DATE"] = timestamp
        environment["GIT_COMMITTER_DATE"] = timestamp
        cls._run_command(
            ("git", "commit", "--quiet", "-m", message),
            git_root,
            environment,
        )

    def _git(self, *arguments):
        command = ("git",) + tuple(arguments)
        return self._run_command(command, self.git_root)

    def _head(self):
        return self._command_output(("git", "rev-parse", "HEAD"), self.git_root)

    def _commit(self, message):
        timestamp = "2026-09-04T12:01:00+00:00"
        self._commit_at_fixed_time(self.git_root, message, timestamp)

    @classmethod
    def _write_primary_config(cls, path):
        payload = {
            "schema_version": "exd_hox_primary_split_config.v1",
            "study": {
                "identifier": SYNTHETIC_STUDY_ID,
                "dataset_identifier": SYNTHETIC_DATASET_ID,
                "project_commit": SOURCE_FOUNDATION_COMMIT,
                "external_source_commit": EXTERNAL_SOURCE_COMMIT,
            },
            "dataset": {
                "transcription_factors": list(TRANSCRIPTION_FACTORS),
            },
            "outputs": {
                "split_directory": "data/processed/exd_hox_primary_split_v1",
                "subset_directory": "data/processed/exd_hox_nested_subsets_v1",
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def _write_finalized_inputs(cls, root):
        primary_config_path = root / PRIMARY_CONFIG_LOGICAL_PATH
        primary_config_sha256 = hash_file_bytes(primary_config_path)
        split_directory = root / "data/processed/exd_hox_primary_split_v1"
        subset_directory = root / "data/processed/exd_hox_nested_subsets_v1"
        split_directory.mkdir(parents=True, exist_ok=True)
        subset_directory.mkdir(parents=True, exist_ok=True)

        count_path = split_directory / COUNT_INPUT_FILENAME
        affinity_path = split_directory / AFFINITY_INPUT_FILENAME
        leakage_path = split_directory / LEAKAGE_INPUT_FILENAME
        subset_path = subset_directory / SUBSET_INPUT_FILENAME
        write_tsv_exclusive(count_path, COUNT_INPUT_FIELDS, cls._count_rows())
        write_tsv_exclusive(
            affinity_path,
            AFFINITY_INPUT_FIELDS,
            cls._affinity_rows(),
        )
        write_tsv_exclusive(
            leakage_path,
            LEAKAGE_INPUT_FIELDS,
            cls._leakage_rows(),
        )
        write_tsv_exclusive(subset_path, SUBSET_INPUT_FIELDS, cls._subset_rows())

        split_artifacts = []
        for path in (count_path, affinity_path, leakage_path):
            logical_path = path.relative_to(root).as_posix()
            split_artifacts.append(fingerprint_file(path, logical_path).to_dict())
        split_artifacts.sort(key=lambda row: row["path"])
        split_manifest = build_hashed_manifest(
            "exd_hox_primary_split_manifest.v1",
            {
                "study_identifier": SYNTHETIC_STUDY_ID,
                "dataset_identifier": SYNTHETIC_DATASET_ID,
                "config_path": PRIMARY_CONFIG_LOGICAL_PATH,
                "config_sha256": primary_config_sha256,
                "split_directory": split_directory.relative_to(root).as_posix(),
                "split_identity_hash": SYNTHETIC_SPLIT_ID,
                "policy": {"identifier": "synthetic_primary_split.v1"},
                "artifacts": split_artifacts,
            },
        )
        split_manifest_path = split_directory / SPLIT_MANIFEST_FILENAME
        write_json_exclusive(split_manifest_path, split_manifest)

        subset_artifact = fingerprint_file(
            subset_path,
            subset_path.relative_to(root).as_posix(),
        ).to_dict()
        subset_manifest = build_hashed_manifest(
            "exd_hox_subset_set_manifest.v1",
            {
                "study_identifier": SYNTHETIC_STUDY_ID,
                "dataset_identifier": SYNTHETIC_DATASET_ID,
                "config_path": PRIMARY_CONFIG_LOGICAL_PATH,
                "config_sha256": primary_config_sha256,
                "subset_directory": subset_directory.relative_to(root).as_posix(),
                "split_identity_hash": SYNTHETIC_SPLIT_ID,
                "split_manifest_hash": split_manifest["manifest_hash"],
                "policy": {"identifier": "synthetic_nested_subsets.v1"},
                "artifacts": [subset_artifact],
            },
        )
        subset_manifest_path = subset_directory / SUBSET_MANIFEST_FILENAME
        write_json_exclusive(subset_manifest_path, subset_manifest)
        return {
            "primary_config_sha256": primary_config_sha256,
            "split_manifest": split_manifest,
            "split_manifest_path": split_manifest_path,
            "subset_manifest": subset_manifest,
            "subset_manifest_path": subset_manifest_path,
            "count_path": count_path,
            "affinity_path": affinity_path,
            "leakage_path": leakage_path,
            "subset_path": subset_path,
        }

    @classmethod
    def _write_plot_config(cls, path, finalized):
        split_manifest = finalized["split_manifest"]
        subset_manifest = finalized["subset_manifest"]
        payload = {
            "schema_version": "exd_hox_primary_split_plot_config.v2",
            "study": {
                "identifier": SYNTHETIC_STUDY_ID,
                "dataset_identifier": SYNTHETIC_DATASET_ID,
            },
            "provenance": {
                "external_source_commit": EXTERNAL_SOURCE_COMMIT,
                "source_foundation_commit": SOURCE_FOUNDATION_COMMIT,
                "split_pipeline_commit": SPLIT_PIPELINE_COMMIT,
                "plotting_entry_point": PLOTTING_ENTRY_POINT,
            },
            "inputs": {
                "primary_split_config_path": PRIMARY_CONFIG_LOGICAL_PATH,
                "primary_split_config_sha256": finalized[
                    "primary_config_sha256"
                ],
                "primary_split_id": SYNTHETIC_SPLIT_ID,
                "primary_split_manifest_path": finalized[
                    "split_manifest_path"
                ].relative_to(path.parents[1]).as_posix(),
                "primary_split_manifest_hash": split_manifest["manifest_hash"],
                "primary_split_manifest_file_sha256": hash_file_bytes(
                    finalized["split_manifest_path"]
                ),
                "subset_set_id": subset_manifest["manifest_hash"],
                "subset_set_manifest_path": finalized[
                    "subset_manifest_path"
                ].relative_to(path.parents[1]).as_posix(),
                "subset_set_manifest_hash": subset_manifest["manifest_hash"],
                "subset_set_manifest_file_sha256": hash_file_bytes(
                    finalized["subset_manifest_path"]
                ),
                "count_summary_path": finalized["count_path"].relative_to(
                    path.parents[1]
                ).as_posix(),
                "affinity_histogram_path": finalized[
                    "affinity_path"
                ].relative_to(path.parents[1]).as_posix(),
                "leakage_audit_path": finalized["leakage_path"].relative_to(
                    path.parents[1]
                ).as_posix(),
                "subset_levels_path": finalized["subset_path"].relative_to(
                    path.parents[1]
                ).as_posix(),
                "raw_file_sha256": dict(RAW_FILE_SHA256),
            },
            "outputs": {"plot_directory": PLOT_LOGICAL_DIRECTORY},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

    @staticmethod
    def _count_rows():
        rows = []
        primary_counts = {
            "AbdA": {"training": 8, "validation": 1, "test": 1},
            "Ubx": {"training": 12, "validation": 2, "test": 2},
        }
        paper_counts = {
            "AbdA": {"train": 8, "test": 2},
            "Ubx": {"train": 13, "test": 3},
        }
        for transcription_factor in TRANSCRIPTION_FACTORS:
            for split in ("training", "validation", "test"):
                count = primary_counts[transcription_factor][split]
                rows.append(
                    {
                        "protocol": "primary",
                        "transcription_factor": transcription_factor,
                        "split": split,
                        "row_count": count,
                        "logical_example_count": count,
                        "global_rc_group_count": count,
                        "exact_cross_split_overlap_occurrence_count": 0,
                    }
                )
            for split in ("train", "test"):
                count = paper_counts[transcription_factor][split]
                rows.append(
                    {
                        "protocol": "paper_split_reproduction",
                        "transcription_factor": transcription_factor,
                        "split": split,
                        "row_count": count,
                        "logical_example_count": count,
                        "global_rc_group_count": count,
                        "exact_cross_split_overlap_occurrence_count": (
                            2 if transcription_factor == "AbdA" else 4
                        ),
                    }
                )
        return tuple(rows)

    @staticmethod
    def _affinity_rows():
        rows = []
        for transcription_factor in TRANSCRIPTION_FACTORS:
            for split in ("training", "validation"):
                rows.extend(
                    (
                        {
                            "transcription_factor": transcription_factor,
                            "split": split,
                            "bin_index": 0,
                            "bin_left": "0",
                            "bin_right": "0.5",
                            "logical_example_count": 1,
                        },
                        {
                            "transcription_factor": transcription_factor,
                            "split": split,
                            "bin_index": 1,
                            "bin_left": "0.5",
                            "bin_right": "1",
                            "logical_example_count": 1,
                        },
                    )
                )
        return tuple(rows)

    @staticmethod
    def _leakage_rows():
        rows = []
        for left_split, right_split in (
            ("training", "validation"),
            ("training", "test"),
            ("validation", "test"),
        ):
            rows.append(
                {
                    "comparison": "primary",
                    "left_split": left_split,
                    "right_split": right_split,
                    "exact_sequence_overlap_group_count": 0,
                    "reverse_complement_equivalent_overlap_group_count": 0,
                    "reverse_complement_only_overlap_group_count": 0,
                    "logical_example_overlap_count": 0,
                }
            )
        return tuple(rows)

    @staticmethod
    def _subset_rows():
        rows = []
        for transcription_factor in TRANSCRIPTION_FACTORS:
            for level_id, requested, rank in (
                ("absolute_2", 2, 1),
                ("fraction_100pct", 8, 7),
            ):
                actual = requested
                if transcription_factor == "Ubx" and requested == 8:
                    actual = 12
                rows.append(
                    {
                        "transcription_factor": transcription_factor,
                        "level_id": level_id,
                        "request_type": (
                            "absolute"
                            if level_id.startswith("absolute")
                            else "fractional"
                        ),
                        "request_value": str(requested),
                        "unaliased_requested_logical_example_count": requested,
                        "alias_absolute_anchor": "",
                        "canonical_requested_logical_example_count": requested,
                        "actual_logical_example_count": actual,
                        "actual_rc_group_count": actual,
                        "inclusive_maximum_rank": rank,
                    }
                )
        return tuple(rows)

    def _write_forbidden_sentinels(self):
        split_directory = self.finalized["split_manifest_path"].parent
        self.public_test_path = (
            split_directory / "exd_hox_public_test_inputs_v1.tsv.gz"
        )
        self.public_test_path.write_bytes(b"public-input-secret-0.75")
        self.sealed_manifest_path = (
            split_directory / "exd_hox_sealed_test_target_manifest_v1.json"
        )
        self.sealed_manifest_path.write_text(
            '{"sealed": "manifest-secret-0.75"}\n',
            encoding="utf-8",
        )
        self.sealed_target_path = (
            self.root / "data/sealed/exd_hox_primary_test_targets_v1/targets.tsv"
        )
        self.sealed_target_path.parent.mkdir(parents=True, exist_ok=True)
        self.sealed_target_path.write_text(
            "logical_example_id\ttarget\nsecret\t0.75\n",
            encoding="utf-8",
        )
        first_raw_path = next(iter(RAW_FILE_SHA256))
        self.raw_hdf5_path = self.root / first_raw_path
        self.raw_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_hdf5_path.write_bytes(b"hdf5-secret-0.75")

    def _generate(self):
        return plot_primary_split_tables(
            self.config_path,
            self.root,
            self.generator_commit,
        )

    def _read_manifest(self):
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def _rehash_manifest(payload):
        content = copy.deepcopy(payload)
        schema_version = content.pop("schema_version")
        content.pop("manifest_hash", None)
        return build_hashed_manifest(schema_version, content)

    def _write_manifest(self, payload):
        self.manifest_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _validate(self, expected_commit=None):
        commit = self.generator_commit
        if expected_commit is not None:
            commit = expected_commit
        return validate_primary_split_plot_manifest(
            self.manifest_path,
            self.root,
            commit,
        )

    @staticmethod
    def _byte_map(directory):
        result = {}
        for path in sorted(directory.iterdir()):
            result[path.name] = path.read_bytes()
        return result

    def _guarded_open(self, original_open):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(os.fspath(file)))
                lowered_name = path.name.lower()
                lowered_parts = tuple(part.lower() for part in path.parts)
                if path.suffix.lower() in (".h5", ".hdf5"):
                    raise AssertionError("Plotter attempted HDF5 access.")
                if "public_test_inputs" in lowered_name:
                    raise AssertionError("Plotter attempted public test-row access.")
                if "sealed_test_target_manifest" in lowered_name:
                    raise AssertionError("Plotter attempted sealed-manifest access.")
                if any("sealed" in part for part in lowered_parts):
                    raise AssertionError("Plotter attempted sealed-target access.")
            return original_open(file, *args, **kwargs)

        return guarded

    def test_tracked_v2_config_pins_the_accepted_non_circular_contract(self):
        config_path = SOURCE_PROJECT_ROOT / PLOT_CONFIG_LOGICAL_PATH
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["schema_version"],
            "exd_hox_primary_split_plot_config.v2",
        )
        self.assertEqual(
            config["provenance"],
            {
                "external_source_commit": (
                    "9e6d6ef0355558c98855b83a9c21fe11999f65d9"
                ),
                "source_foundation_commit": (
                    "62a99688cbd3af97f081df500223ba6f55cd0fe0"
                ),
                "split_pipeline_commit": (
                    "65c6bddc0f4570e7a6cf5a90f5f6ef5801e01d27"
                ),
                "plotting_entry_point": (
                    "scripts/plot/plot_exd_hox_primary_split.py"
                ),
            },
        )
        expected_inputs = {
            "primary_split_config_path": "configs/exd_hox_primary_split_v1.yaml",
            "primary_split_config_sha256": (
                "5894d3c1eac27ddf47a2792c46107b9ff00dfbb4a609e135c78ccee4fe9800b6"
            ),
            "primary_split_id": (
                "a684fd4fd863709d4a59e8925a2f76d95255e0f33a9996216fd896ce098c393f"
            ),
            "primary_split_manifest_path": (
                "data/processed/exd_hox_primary_split_v1/"
                "exd_hox_primary_split_manifest_v1.json"
            ),
            "primary_split_manifest_hash": (
                "fb595729defc1f140637f0a75d2beb78694a0a36e0fa04446727483bc121e564"
            ),
            "primary_split_manifest_file_sha256": (
                "be00a42d4d80d31d7f99567731f53536c897be4b28b0cdb64e574b520a3b8f8d"
            ),
            "subset_set_id": (
                "ce75331e6bf5db939ff70df06a1cda07028e31664ef7a8d579c9363d00fc4125"
            ),
            "subset_set_manifest_path": (
                "data/processed/exd_hox_nested_subsets_v1/"
                "exd_hox_subset_set_manifest_v1.json"
            ),
            "subset_set_manifest_hash": (
                "ce75331e6bf5db939ff70df06a1cda07028e31664ef7a8d579c9363d00fc4125"
            ),
            "subset_set_manifest_file_sha256": (
                "645cbea698a3b576b0f8cfcbb87d518c0950337fd580703a933da3bf64a94a5c"
            ),
            "count_summary_path": (
                "data/processed/exd_hox_primary_split_v1/"
                "exd_hox_primary_split_count_summary_v1.tsv"
            ),
            "affinity_histogram_path": (
                "data/processed/exd_hox_primary_split_v1/"
                "exd_hox_primary_split_affinity_histogram_v1.tsv"
            ),
            "leakage_audit_path": (
                "data/processed/exd_hox_primary_split_v1/"
                "exd_hox_primary_split_leakage_audit_v1.tsv"
            ),
            "subset_levels_path": (
                "data/processed/exd_hox_nested_subsets_v1/"
                "exd_hox_nested_subset_levels_v1.tsv"
            ),
        }
        for field, expected_value in expected_inputs.items():
            self.assertEqual(config["inputs"][field], expected_value)
        source_manifest_path = (
            SOURCE_PROJECT_ROOT
            / "data/processed/exd_hox_selex_audit_v1/exd_hox_source_manifest_v1.json"
        )
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        accepted_raw_identities = {}
        for item in source_manifest["files"]:
            accepted_raw_identities[item["imported_raw_path"]] = item["sha256"]
        self.assertEqual(
            config["inputs"]["raw_file_sha256"],
            accepted_raw_identities,
        )
        self.assertEqual(
            config["outputs"]["plot_directory"],
            "plots/exd_hox_primary_split_v2",
        )
        self.assertNotIn("plot_generator_commit", json.dumps(config))
        self.assertEqual(
            LEAKAGE_INPUT_FIELDS,
            (
                "comparison",
                "left_split",
                "right_split",
                "exact_sequence_overlap_group_count",
                "reverse_complement_equivalent_overlap_group_count",
                "reverse_complement_only_overlap_group_count",
                "logical_example_overlap_count",
            ),
        )

    def test_happy_path_generation_has_exact_v2_manifest_fields(self):
        manifest = self._generate()
        expected_fields = {
            "schema_version",
            "study_identifier",
            "dataset_identifier",
            "external_source_commit",
            "source_foundation_commit",
            "split_pipeline_commit",
            "plot_generator_commit",
            "plot_generator_tracked_worktree_clean",
            "plotting_entry_point_path",
            "plotting_entry_point_byte_size",
            "plotting_entry_point_sha256",
            "plot_config_path",
            "plot_config_sha256",
            "primary_split_config_path",
            "primary_split_config_sha256",
            "primary_split_id",
            "primary_split_manifest_path",
            "primary_split_manifest_hash",
            "primary_split_manifest_file_sha256",
            "subset_set_id",
            "subset_set_manifest_path",
            "subset_set_manifest_hash",
            "subset_set_manifest_file_sha256",
            "plot_directory",
            "inputs",
            "outputs",
            "test_target_policy",
            "manifest_hash",
        }
        self.assertEqual(set(manifest), expected_fields)
        self.assertEqual(
            manifest["schema_version"],
            "exd_hox_primary_split_plot_manifest.v2",
        )
        self.assertEqual(manifest["plot_generator_commit"], self._head())
        self.assertEqual(manifest["plot_generator_commit"], self.generator_commit)
        self.assertIs(manifest["plot_generator_tracked_worktree_clean"], True)
        self.assertEqual(manifest["source_foundation_commit"], SOURCE_FOUNDATION_COMMIT)
        self.assertEqual(manifest["split_pipeline_commit"], SPLIT_PIPELINE_COMMIT)
        self.assertTrue(self.manifest_path.is_file())

    def test_manifest_has_six_inputs_and_fifteen_v2_outputs(self):
        manifest = self._generate()
        self.assertEqual(len(manifest["inputs"]), 6)
        self.assertEqual(len(manifest["outputs"]), 15)
        for entry in manifest["inputs"] + manifest["outputs"]:
            self.assertEqual(set(entry), {"path", "byte_size", "sha256"})
        expected_names = {
            "exd_hox_primary_split_counts_plot_source_v2.tsv",
            "exd_hox_primary_split_affinity_plot_source_v2.tsv",
            "exd_hox_nested_subset_counts_plot_source_v2.tsv",
            "exd_hox_primary_split_leakage_plot_source_v2.tsv",
            "exd_hox_paper_vs_primary_split_plot_source_v2.tsv",
            "exd_hox_primary_split_counts_v2.png",
            "exd_hox_primary_split_counts_v2.pdf",
            "exd_hox_primary_split_affinity_distributions_v2.png",
            "exd_hox_primary_split_affinity_distributions_v2.pdf",
            "exd_hox_nested_subset_counts_v2.png",
            "exd_hox_nested_subset_counts_v2.pdf",
            "exd_hox_primary_split_leakage_v2.png",
            "exd_hox_primary_split_leakage_v2.pdf",
            "exd_hox_paper_vs_primary_split_counts_v2.png",
            "exd_hox_paper_vs_primary_split_counts_v2.pdf",
        }
        observed_names = {Path(entry["path"]).name for entry in manifest["outputs"]}
        self.assertEqual(observed_names, expected_names)
        self.assertNotIn(PLOT_MANIFEST_FILENAME, observed_names)
        self.assertEqual(set(OUTPUT_FILENAMES), expected_names)

    def test_public_validation_uses_the_historical_producer_blob(self):
        self._generate()
        historical_commit = self.generator_commit
        producer_path = self.root / PLOTTING_ENTRY_POINT
        producer_path.write_bytes(producer_path.read_bytes() + b"\n# later checkout\n")
        self._git("add", "--", (self.project_prefix / PLOTTING_ENTRY_POINT).as_posix())
        self._commit("Later producer checkout")
        self.assertNotEqual(self._head(), historical_commit)

        validated = self._validate(historical_commit)
        self.assertEqual(validated["plot_generator_commit"], historical_commit)
        with mock.patch("builtins.print"):
            cli_result = main(
                (
                    "--repository-root",
                    str(self.root),
                    "--validate-manifest",
                    str(self.manifest_path),
                    "--expected-plot-generator-commit",
                    historical_commit,
                )
            )
        self.assertEqual(cli_result, validated)

    def test_historical_v1_artifacts_remain_byte_identical(self):
        v1_directory = self.root / "plots/exd_hox_primary_split_v1"
        v1_directory.mkdir(parents=True)
        for index, filename in enumerate(OUTPUT_FILENAMES + (PLOT_MANIFEST_FILENAME,)):
            v1_name = filename.replace("_v2", "_v1")
            (v1_directory / v1_name).write_bytes(
                "historical-v1-{0}\n".format(index).encode("ascii")
            )
        before = self._byte_map(v1_directory)
        self._generate()
        self.assertEqual(self._byte_map(v1_directory), before)

    def test_existing_v2_output_directory_refuses_overwrite(self):
        self._generate()
        before = self._byte_map(self.plot_directory)
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            self._generate()
        self.assertEqual(self._byte_map(self.plot_directory), before)

    def test_generation_rejects_expected_versus_actual_head_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match runtime HEAD"):
            plot_primary_split_tables(
                self.config_path,
                self.root,
                SPLIT_PIPELINE_COMMIT,
            )
        with self.assertRaisesRegex(ValueError, "full 40-character lowercase"):
            plot_primary_split_tables(
                self.config_path,
                self.root,
                "A" * 40,
            )
        self.assertFalse(self.plot_directory.exists())

    def test_generation_rejects_missing_or_unresolved_head(self):
        head_path = self.git_root / ".git/HEAD"
        original = head_path.read_bytes()
        cases = (None, b"ref: refs/heads/unresolved-plot-head\n")
        for replacement in cases:
            with self.subTest(replacement=replacement):
                if head_path.exists():
                    head_path.unlink()
                if replacement is not None:
                    head_path.write_bytes(replacement)
                try:
                    with self.assertRaisesRegex(ValueError, "resolve runtime Git HEAD"):
                        self._generate()
                finally:
                    head_path.write_bytes(original)
                self.assertFalse(self.plot_directory.exists())
        with mock.patch.object(
            plot_module,
            "_git_text",
            return_value="A" * 40,
        ):
            with self.assertRaisesRegex(ValueError, "Runtime Git HEAD.*lowercase"):
                self._generate()

    def test_generation_rejects_untracked_plotting_entry_point(self):
        git_path = (self.project_prefix / PLOTTING_ENTRY_POINT).as_posix()
        self._git("rm", "--cached", "--quiet", "--", git_path)
        self._commit("Remove producer from generator tree")
        self.generator_commit = self._head()
        with self.assertRaisesRegex(ValueError, "not tracked"):
            self._generate()
        self.assertFalse(self.plot_directory.exists())

    def test_generation_rejects_repository_wide_staged_tracked_changes(self):
        sentinel = self.git_root / "tracked_scope_sentinel.txt"
        sentinel.write_text("staged\n", encoding="utf-8")
        self._git("add", "--", "tracked_scope_sentinel.txt")
        with self.assertRaisesRegex(ValueError, "Staged tracked changes"):
            self._generate()
        self.assertFalse(self.plot_directory.exists())

    def test_generation_rejects_repository_wide_unstaged_tracked_changes(self):
        sentinel = self.git_root / "tracked_scope_sentinel.txt"
        sentinel.write_text("unstaged\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unstaged tracked changes"):
            self._generate()
        self.assertFalse(self.plot_directory.exists())

    def test_generation_allows_unrelated_untracked_artifacts(self):
        artifact = self.root / "data/processed/milestone_3d_c/unrelated.tsv"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("untracked\n", encoding="utf-8")
        manifest = self._generate()
        self.assertEqual(manifest["plot_generator_commit"], self.generator_commit)

    def test_validation_rejects_missing_commit_fields_one_at_a_time(self):
        valid_manifest = self._generate()
        commit_fields = (
            "external_source_commit",
            "source_foundation_commit",
            "split_pipeline_commit",
            "plot_generator_commit",
        )
        for field in commit_fields:
            with self.subTest(field=field):
                malformed = copy.deepcopy(valid_manifest)
                malformed.pop(field)
                self._write_manifest(self._rehash_manifest(malformed))
                with self.assertRaisesRegex(ValueError, "fields differ"):
                    self._validate()

    def test_validation_rejects_malformed_commit_fields_one_at_a_time(self):
        valid_manifest = self._generate()
        malformed_values = {
            "external_source_commit": "1" * 39,
            "source_foundation_commit": "g" * 40,
            "split_pipeline_commit": "A" * 40,
            "plot_generator_commit": "not-a-full-commit",
        }
        for field, malformed_value in malformed_values.items():
            with self.subTest(field=field):
                malformed = copy.deepcopy(valid_manifest)
                malformed[field] = malformed_value
                self._write_manifest(self._rehash_manifest(malformed))
                with self.assertRaisesRegex(ValueError, "full 40-character lowercase"):
                    self._validate()
        self._write_manifest(valid_manifest)
        with self.assertRaisesRegex(ValueError, "full 40-character lowercase"):
            self._validate("short")

    def test_validation_rejects_wrong_foundation_or_pipeline_commit(self):
        valid_manifest = self._generate()
        replacements = {
            "source_foundation_commit": SPLIT_PIPELINE_COMMIT,
            "split_pipeline_commit": SOURCE_FOUNDATION_COMMIT,
        }
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                malformed = copy.deepcopy(valid_manifest)
                malformed[field] = replacement
                self._write_manifest(self._rehash_manifest(malformed))
                with self.assertRaisesRegex(ValueError, "differs from bound config"):
                    self._validate()

        wrong_ancestry = copy.deepcopy(valid_manifest)
        wrong_ancestry["plot_generator_commit"] = SOURCE_FOUNDATION_COMMIT
        self._write_manifest(self._rehash_manifest(wrong_ancestry))
        with self.assertRaisesRegex(ValueError, "Required Git ancestry is absent"):
            self._validate(SOURCE_FOUNDATION_COMMIT)

        unresolved_generator = copy.deepcopy(valid_manifest)
        unresolved_generator["plot_generator_commit"] = "0" * 40
        self._write_manifest(self._rehash_manifest(unresolved_generator))
        with self.assertRaisesRegex(ValueError, "resolve plot-generator commit"):
            self._validate("0" * 40)

    def test_validation_rejects_wrong_expected_generator_commit(self):
        self._generate()
        with self.assertRaisesRegex(ValueError, "historical.*differs"):
            self._validate(SPLIT_PIPELINE_COMMIT)

    def test_all_commit_fields_participate_in_manifest_hash(self):
        valid_manifest = self._generate()
        commit_fields = (
            "external_source_commit",
            "source_foundation_commit",
            "split_pipeline_commit",
            "plot_generator_commit",
        )
        for field in commit_fields:
            with self.subTest(field=field):
                malformed = copy.deepcopy(valid_manifest)
                malformed[field] = "0" * 40
                self._write_manifest(malformed)
                with self.assertRaisesRegex(ValueError, "Manifest hash mismatch"):
                    self._validate()

    def test_rehashed_v1_manifest_is_rejected_when_v2_is_required(self):
        self._generate()
        rehashed_v1 = build_hashed_manifest(
            "exd_hox_primary_split_plot_manifest.v1",
            {
                "dataset_identifier": SYNTHETIC_DATASET_ID,
                "inputs": [],
                "outputs": [],
            },
        )
        self._write_manifest(rehashed_v1)
        with self.assertRaisesRegex(ValueError, "v2.*required"):
            self._validate()

    def test_every_input_and_output_fingerprint_is_validated(self):
        manifest = self._generate()
        for index, entry in enumerate(manifest["inputs"]):
            with self.subTest(input_path=entry["path"]):
                malformed = copy.deepcopy(manifest)
                malformed["inputs"][index]["sha256"] = "0" * 64
                self._write_manifest(self._rehash_manifest(malformed))
                with self.assertRaisesRegex(ValueError, "Input fingerprint mismatch"):
                    self._validate()
        self._write_manifest(manifest)
        for entry in manifest["outputs"]:
            output_path = self.root / entry["path"]
            original = output_path.read_bytes()
            with self.subTest(path=entry["path"]):
                output_path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError, "Output fingerprint mismatch"):
                    self._validate()
                output_path.write_bytes(original)

    def test_validation_rejects_input_config_and_entry_point_mismatch(self):
        valid_manifest = self._generate()
        valid_manifest_bytes = self.manifest_path.read_bytes()
        cases = (
            (self.config_path, "Plot config fingerprint mismatch"),
            (self.finalized["count_path"], "fingerprint mismatch"),
        )
        for path, message in cases:
            original = path.read_bytes()
            with self.subTest(path=path):
                path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError, message):
                    self._validate()
                path.write_bytes(original)
                self.manifest_path.write_bytes(valid_manifest_bytes)

        malformed = copy.deepcopy(valid_manifest)
        malformed["plotting_entry_point_sha256"] = "0" * 64
        self._write_manifest(self._rehash_manifest(malformed))
        with self.assertRaisesRegex(ValueError, "entry-point fingerprint differs"):
            self._validate()

    def test_validation_rejects_missing_extra_relocated_or_repointed_outputs(self):
        valid_manifest = self._generate()
        valid_manifest_bytes = self.manifest_path.read_bytes()
        first_entry = valid_manifest["outputs"][0]
        first_path = self.root / first_entry["path"]
        first_bytes = first_path.read_bytes()

        first_path.unlink()
        with self.assertRaises(ValueError):
            self._validate()
        first_path.write_bytes(first_bytes)

        extra_path = self.plot_directory / "legacy_extra_v1.tsv"
        extra_path.write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing, extra, or non-v2"):
            self._validate()
        extra_path.unlink()

        relocated_name = "relocated_output_v2.tsv"
        relocated_path = self.plot_directory / relocated_name
        first_path.rename(relocated_path)
        relocated_manifest = copy.deepcopy(valid_manifest)
        relocated_manifest["outputs"][0]["path"] = Path(
            PLOT_LOGICAL_DIRECTORY,
            relocated_name,
        ).as_posix()
        relocated_manifest["outputs"].sort(key=lambda row: row["path"])
        self._write_manifest(self._rehash_manifest(relocated_manifest))
        with self.assertRaisesRegex(ValueError, "path set differs"):
            self._validate()
        relocated_path.rename(first_path)
        self.manifest_path.write_bytes(valid_manifest_bytes)

        target_path = self.root / "repointed-output-copy.bin"
        target_path.write_bytes(first_bytes)
        first_path.unlink()
        first_path.symlink_to(target_path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self._validate()
        first_path.unlink()
        first_path.write_bytes(first_bytes)

    def test_late_snapshot_and_tracked_state_changes_abort_before_rename(self):
        initial_count_bytes = self.finalized["count_path"].read_bytes()
        original_entry_verifier = plot_module._verify_tracked_entry_point

        def verify_entry_then_mutate(*args, **kwargs):
            fingerprint = original_entry_verifier(*args, **kwargs)
            self.finalized["count_path"].write_bytes(
                initial_count_bytes + b"\n"
            )
            return fingerprint

        with mock.patch.object(
            plot_module,
            "_verify_tracked_entry_point",
            side_effect=verify_entry_then_mutate,
        ):
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                self._generate()
        self.finalized["count_path"].write_bytes(initial_count_bytes)
        self.assertFalse(self.plot_directory.exists())

        scenarios = (
            (self.finalized["count_path"], False, False),
            (self.config_path, False, True),
            (self.root / PRIMARY_CONFIG_LOGICAL_PATH, False, True),
            (self.root / PLOTTING_ENTRY_POINT, False, True),
            (self.git_root / "tracked_scope_sentinel.txt", True, False),
        )
        original_writer = plot_module._write_plot_manifest
        for mutation_path, stage_mutation, bypass_cleanliness in scenarios:
            with self.subTest(
                path=mutation_path,
                staged=stage_mutation,
                bypass_cleanliness=bypass_cleanliness,
            ):
                original_bytes = mutation_path.read_bytes()

                def write_then_mutate(*args, **kwargs):
                    manifest = original_writer(*args, **kwargs)
                    mutation_path.write_bytes(original_bytes + b"late mutation")
                    if stage_mutation:
                        self._git("add", "--", "tracked_scope_sentinel.txt")
                    return manifest

                with mock.patch.object(
                    plot_module,
                    "_write_plot_manifest",
                    side_effect=write_then_mutate,
                ):
                    if bypass_cleanliness:
                        with mock.patch.object(
                            plot_module,
                            "_require_clean_tracked_worktree",
                            return_value=None,
                        ):
                            with self.assertRaisesRegex(
                                ValueError,
                                "Snapshotted plot provenance file changed",
                            ):
                                self._generate()
                    else:
                        with self.assertRaises(ValueError):
                            self._generate()
                self.assertFalse(self.plot_directory.exists())
                mutation_path.write_bytes(original_bytes)
                if stage_mutation:
                    self._git("add", "--", "tracked_scope_sentinel.txt")
                staging_directories = tuple(
                    self.plot_directory.parent.glob(
                        ".exd_hox_primary_split_plot_staging_*"
                    )
                )
                self.assertEqual(staging_directories, ())

    def test_plotter_never_opens_hdf5_test_rows_or_sealed_targets(self):
        protected_paths = (
            self.public_test_path,
            self.sealed_manifest_path,
            self.sealed_target_path,
            self.raw_hdf5_path,
        )
        protected_bytes = {}
        for path in protected_paths:
            protected_bytes[path] = path.read_bytes()
        original_open = builtins.open
        original_io_open = io.open
        with mock.patch(
            "builtins.open",
            side_effect=self._guarded_open(original_open),
        ), mock.patch(
            "io.open",
            side_effect=self._guarded_open(original_io_open),
        ):
            self._generate()
        for path, before in protected_bytes.items():
            self.assertEqual(path.read_bytes(), before)

    def test_outputs_contain_no_plaintext_test_affinity(self):
        manifest = self._generate()
        affinity_path = self.plot_directory / AFFINITY_SOURCE_FILENAME
        with open(affinity_path, "r", encoding="utf-8", newline="") as input_file:
            rows = tuple(csv.DictReader(input_file, delimiter="\t"))
        test_rows = []
        for row in rows:
            if row["split"] == "test":
                test_rows.append(row)
        self.assertEqual(len(test_rows), len(TRANSCRIPTION_FACTORS))
        self.assertTrue(all(row["record_type"] == "test_count" for row in test_rows))
        self.assertTrue(all(row["bin_left"] == "" for row in test_rows))
        text_filenames = (
            COUNT_SOURCE_FILENAME,
            AFFINITY_SOURCE_FILENAME,
            SUBSET_SOURCE_FILENAME,
            LEAKAGE_SOURCE_FILENAME,
            COMPARISON_SOURCE_FILENAME,
            PLOT_MANIFEST_FILENAME,
        )
        for filename in text_filenames:
            serialized = (self.plot_directory / filename).read_text(encoding="utf-8")
            self.assertNotIn("0.75", serialized)
            self.assertNotIn("target_value", serialized)
            self.assertNotIn("target_float32_bits", serialized)
            self.assertNotIn("plaintext_test_targets", serialized)
        forbidden_bytes = (
            b"public-input-secret-0.75",
            b"manifest-secret-0.75",
            b"hdf5-secret-0.75",
            b"secret\t0.75",
        )
        for entry in manifest["outputs"]:
            output_bytes = (self.root / entry["path"]).read_bytes()
            for forbidden_value in forbidden_bytes:
                self.assertNotIn(forbidden_value, output_bytes)
        self.assertEqual(manifest["test_target_policy"], (
            "aggregate_test_counts_only_no_test_affinity_distribution"
        ))

    def test_equivalent_roots_generate_byte_identical_artifacts(self):
        second_git_root = Path(self.temporary_directory.name) / "second_flex"
        self._run_command(
            (
                "git",
                "clone",
                "--shared",
                "--quiet",
                str(self.template_git_root),
                str(second_git_root),
            ),
            Path(self.temporary_directory.name),
        )
        second_root = second_git_root / self.project_prefix
        self._write_finalized_inputs(second_root)
        first_manifest = self._generate()
        second_manifest = plot_primary_split_tables(
            second_root / PLOT_CONFIG_LOGICAL_PATH,
            second_root,
            self.generator_commit,
        )
        self.assertEqual(first_manifest, second_manifest)
        second_plot_directory = second_root / PLOT_LOGICAL_DIRECTORY
        self.assertEqual(
            self._byte_map(self.plot_directory),
            self._byte_map(second_plot_directory),
        )


if __name__ == "__main__":
    unittest.main()
