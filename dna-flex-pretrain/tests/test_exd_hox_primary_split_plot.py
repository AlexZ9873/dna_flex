"""Tests for v3 Exd-Hox plots and historical v2 validation."""

import builtins
import copy
import csv
import inspect
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings

import yaml

import scripts.plot.plot_exd_hox_primary_split as plot_module
from matplotlib.figure import Figure
from matplotlib.text import Text
from scripts.plot.plot_exd_hox_primary_split import (
    AFFINITY_INPUT_FIELDS,
    AFFINITY_INPUT_FILENAME,
    AFFINITY_SOURCE_FIELDS,
    AFFINITY_SOURCE_FILENAME,
    COUNT_INPUT_FIELDS,
    COUNT_INPUT_FILENAME,
    COUNT_SOURCE_FIELDS,
    COUNT_SOURCE_FILENAME,
    COMPARISON_SOURCE_FIELDS,
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
    V2_PLOT_CONTRACT,
    V3_PLOT_CONTRACT,
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
PRODUCTION_TRANSCRIPTION_FACTORS = (
    "AbdA",
    "AbdB",
    "Antp",
    "Dfd",
    "Lab",
    "Pb",
    "Scr",
    "Ubx",
)
SUBSET_CAPTION = (
    "Absolute levels are requested counts; percentage levels are fractions of "
    "the full primary training split. Percentage requests may alias an absolute "
    "canonical level."
)
LEAKAGE_CAPTION = (
    "RC-equivalent overlap is inclusive of exact-sequence matches; the three "
    "series are not additive. RC-only counts exclude exact matches."
)
COMPARISON_CAPTION = (
    "For each TF, supplied-split bars count labeled row occurrences in the "
    "supplied train/test files, whereas primary-split bars count reconciled "
    "logical labeled examples after exact/RC grouping. The two bar families "
    "therefore use different counting units."
)
SOURCE_IDENTITY_FIXTURES = (
    (
        "count",
        635,
        "5d0b1232da6cb01208cf02034cdb3e9e18ef3b93478ef1a2e7915313f5083c98",
    ),
    (
        "affinity",
        39134,
        "029fd16f124ad0f058545f219494659759531b7f73ff547cce707d85905d5ebe",
    ),
    (
        "subset",
        9278,
        "8aa891cd030b91cbd9bd2a8576a5d73d403e6d34f58f7964548b4caa354ecf67",
    ),
    (
        "leakage",
        338,
        "fb65bd894344018e70293d462f97849021f0271c6ff4971f8dd9faa173f1237b",
    ),
    (
        "comparison",
        1329,
        "1ac89f1bb6bd710fdc0b0d1014ec314022c87df302deb031474b38f94bbbe917",
    ),
)


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
            "Synthetic v3 plot generator",
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
    def _write_plot_config(cls, path, finalized, contract=V3_PLOT_CONTRACT):
        split_manifest = finalized["split_manifest"]
        subset_manifest = finalized["subset_manifest"]
        payload = {
            "schema_version": contract.config_schema_version,
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
            "outputs": {"plot_directory": contract.plot_logical_directory},
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
        rows = [
            {
                "comparison": "paper_split_reproduction",
                "left_split": "train",
                "right_split": "test",
                "exact_sequence_overlap_group_count": 2,
                "reverse_complement_equivalent_overlap_group_count": 2,
                "reverse_complement_only_overlap_group_count": 0,
                "logical_example_overlap_count": 2,
            }
        ]
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
            for level_id, request_type, request_value, requested, rank in (
                ("absolute_2", "absolute", "2", 2, 1),
                ("fraction_100pct", "fractional", "1.0", 8, 7),
            ):
                actual = requested
                if transcription_factor == "Ubx" and requested == 8:
                    actual = 12
                rows.append(
                    {
                        "transcription_factor": transcription_factor,
                        "level_id": level_id,
                        "request_type": request_type,
                        "request_value": request_value,
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

    @staticmethod
    def _read_tsv_fixture(path, expected_fields):
        with open(path, "r", encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file, delimiter="\t")
            if tuple(reader.fieldnames or ()) != tuple(expected_fields):
                raise AssertionError("Unexpected TSV fixture schema: {0}".format(path))
            return tuple(dict(row) for row in reader)

    @staticmethod
    def _figure_text_values(figure):
        values = []
        for artist in figure.findobj(match=lambda item: isinstance(item, Text)):
            if artist.get_visible() and artist.get_text():
                values.append(artist.get_text())
        return tuple(values)

    @staticmethod
    def _legend_text_values(axis):
        legend = axis.get_legend()
        if legend is None:
            return ()
        return tuple(text.get_text() for text in legend.get_texts())

    def _assert_visible_text_within_canvas(self, figure):
        renderer = figure.canvas.get_renderer()
        canvas = figure.bbox
        tolerance = 0.5
        checked = 0
        for artist in figure.findobj(match=lambda item: isinstance(item, Text)):
            if not artist.get_visible() or not artist.get_text():
                continue
            if artist.axes is not None and not artist.axes.get_visible():
                continue
            bounds = artist.get_window_extent(renderer=renderer)
            self.assertGreaterEqual(bounds.x0, canvas.x0 - tolerance, artist.get_text())
            self.assertGreaterEqual(bounds.y0, canvas.y0 - tolerance, artist.get_text())
            self.assertLessEqual(bounds.x1, canvas.x1 + tolerance, artist.get_text())
            self.assertLessEqual(bounds.y1, canvas.y1 + tolerance, artist.get_text())
            checked += 1
        self.assertGreater(checked, 0)

    def _capture_figure(self, renderer, *arguments):
        captured = []
        original_close = plot_module.plt.close

        def capture(figure, *unused_arguments, **unused_keywords):
            captured.append(figure)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            with mock.patch.object(
                plot_module,
                "_save_figure_pair",
                side_effect=capture,
            ), mock.patch.object(
                plot_module.plt,
                "close",
            ), mock.patch.object(
                Figure,
                "tight_layout",
                side_effect=AssertionError("tight_layout must not be called"),
            ):
                renderer(*arguments, V3_PLOT_CONTRACT)
                self.assertEqual(len(captured), 1)
                figure = captured[0]
                figure.canvas.draw()
        for warning in caught_warnings:
            message = str(warning.message).lower()
            self.assertNotIn("layout", message)
        self.assertTrue(figure.get_constrained_layout())
        self._assert_visible_text_within_canvas(figure)
        self.addCleanup(original_close, figure)
        return figure

    @staticmethod
    def _stress_subset_rows():
        full_counts = (12800, 25600, 1000, 25000, 6000, 40000, 9000, 18000)
        requests = (
            ("absolute", "128", 128),
            ("absolute", "256", 256),
            ("absolute", "512", 512),
            ("fractional", "0.01", 1),
            ("fractional", "0.02", 2),
            ("fractional", "0.05", 5),
            ("fractional", "0.1", 10),
            ("fractional", "0.25", 25),
            ("fractional", "0.5", 50),
            ("fractional", "1.0", 100),
        )
        rows = []
        for tf_index, transcription_factor in enumerate(
            PRODUCTION_TRANSCRIPTION_FACTORS
        ):
            full_count = full_counts[tf_index]
            for request_index, (request_type, request_value, amount) in enumerate(
                reversed(requests)
            ):
                if request_type == "absolute":
                    canonical_count = amount
                    unaliased_count = amount
                else:
                    unaliased_count = (full_count * amount + 99) // 100
                    canonical_count = unaliased_count
                opaque_number = (tf_index + 1) * 100 + request_index
                rows.append(
                    {
                        "transcription_factor": transcription_factor,
                        "level_id": "lvl_{0:064x}".format(opaque_number),
                        "request_type": request_type,
                        "request_value": request_value,
                        "unaliased_requested_logical_example_count": (
                            unaliased_count
                        ),
                        "alias_absolute_anchor": "",
                        "canonical_requested_logical_example_count": (
                            canonical_count
                        ),
                        "actual_logical_example_count": canonical_count,
                        "actual_rc_group_count": canonical_count,
                        "inclusive_maximum_rank": canonical_count,
                    }
                )
        return tuple(rows)

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

    def test_explicit_v2_v3_contracts_and_v3_default_aliases(self):
        self.assertEqual(
            V2_PLOT_CONTRACT.config_schema_version,
            "exd_hox_primary_split_plot_config.v2",
        )
        self.assertEqual(
            V2_PLOT_CONTRACT.config_logical_path,
            "configs/exd_hox_primary_split_plot_v2.yaml",
        )
        self.assertEqual(
            V2_PLOT_CONTRACT.manifest_schema_version,
            "exd_hox_primary_split_plot_manifest.v2",
        )
        self.assertEqual(
            V2_PLOT_CONTRACT.manifest_filename,
            "exd_hox_primary_split_plot_manifest_v2.json",
        )
        self.assertEqual(
            V2_PLOT_CONTRACT.plot_logical_directory,
            "plots/exd_hox_primary_split_v2",
        )
        self.assertEqual(
            V3_PLOT_CONTRACT.config_schema_version,
            "exd_hox_primary_split_plot_config.v3",
        )
        self.assertEqual(
            V3_PLOT_CONTRACT.config_logical_path,
            "configs/exd_hox_primary_split_plot_v3.yaml",
        )
        self.assertEqual(
            V3_PLOT_CONTRACT.manifest_schema_version,
            "exd_hox_primary_split_plot_manifest.v3",
        )
        self.assertEqual(
            V3_PLOT_CONTRACT.manifest_filename,
            "exd_hox_primary_split_plot_manifest_v3.json",
        )
        self.assertEqual(
            V3_PLOT_CONTRACT.plot_logical_directory,
            "plots/exd_hox_primary_split_v3",
        )
        expected_v3_sources = (
            "exd_hox_primary_split_counts_plot_source_v3.tsv",
            "exd_hox_primary_split_affinity_plot_source_v3.tsv",
            "exd_hox_nested_subset_counts_plot_source_v3.tsv",
            "exd_hox_primary_split_leakage_plot_source_v3.tsv",
            "exd_hox_paper_vs_primary_split_plot_source_v3.tsv",
        )
        self.assertEqual(
            (
                V3_PLOT_CONTRACT.count_source_filename,
                V3_PLOT_CONTRACT.affinity_source_filename,
                V3_PLOT_CONTRACT.subset_source_filename,
                V3_PLOT_CONTRACT.leakage_source_filename,
                V3_PLOT_CONTRACT.comparison_source_filename,
            ),
            expected_v3_sources,
        )
        self.assertEqual(V3_PLOT_CONTRACT.creator_metadata, (
            "dna-flex-pretrain Milestone 3D-B.2"
        ))
        self.assertEqual(PLOT_CONFIG_LOGICAL_PATH, V3_PLOT_CONTRACT.config_logical_path)
        self.assertEqual(
            PLOT_LOGICAL_DIRECTORY,
            V3_PLOT_CONTRACT.plot_logical_directory,
        )
        self.assertEqual(PLOT_MANIFEST_FILENAME, V3_PLOT_CONTRACT.manifest_filename)
        self.assertEqual(OUTPUT_FILENAMES, V3_PLOT_CONTRACT.output_filenames)

    def test_tracked_v3_config_pins_the_accepted_non_circular_contract(self):
        config_path = SOURCE_PROJECT_ROOT / PLOT_CONFIG_LOGICAL_PATH
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["schema_version"],
            "exd_hox_primary_split_plot_config.v3",
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
            "plots/exd_hox_primary_split_v3",
        )
        self.assertNotIn("plot_generator_commit", json.dumps(config))
        v2_config_path = SOURCE_PROJECT_ROOT / V2_PLOT_CONTRACT.config_logical_path
        v2_config = yaml.safe_load(v2_config_path.read_text(encoding="utf-8"))
        expected_v3_config = copy.deepcopy(v2_config)
        expected_v3_config["schema_version"] = V3_PLOT_CONTRACT.config_schema_version
        expected_v3_config["outputs"]["plot_directory"] = (
            V3_PLOT_CONTRACT.plot_logical_directory
        )
        self.assertEqual(config, expected_v3_config)
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

    def test_happy_path_generation_has_exact_v3_manifest_fields(self):
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
            "exd_hox_primary_split_plot_manifest.v3",
        )
        self.assertEqual(manifest["plot_generator_commit"], self._head())
        self.assertEqual(manifest["plot_generator_commit"], self.generator_commit)
        self.assertIs(manifest["plot_generator_tracked_worktree_clean"], True)
        self.assertEqual(manifest["source_foundation_commit"], SOURCE_FOUNDATION_COMMIT)
        self.assertEqual(manifest["split_pipeline_commit"], SPLIT_PIPELINE_COMMIT)
        self.assertTrue(self.manifest_path.is_file())

    def test_manifest_has_six_inputs_and_fifteen_v3_outputs(self):
        manifest = self._generate()
        self.assertEqual(len(manifest["inputs"]), 6)
        self.assertEqual(len(manifest["outputs"]), 15)
        for entry in manifest["inputs"] + manifest["outputs"]:
            self.assertEqual(set(entry), {"path", "byte_size", "sha256"})
        expected_names = {
            "exd_hox_primary_split_counts_plot_source_v3.tsv",
            "exd_hox_primary_split_affinity_plot_source_v3.tsv",
            "exd_hox_nested_subset_counts_plot_source_v3.tsv",
            "exd_hox_primary_split_leakage_plot_source_v3.tsv",
            "exd_hox_paper_vs_primary_split_plot_source_v3.tsv",
            "exd_hox_primary_split_counts_v3.png",
            "exd_hox_primary_split_counts_v3.pdf",
            "exd_hox_primary_split_affinity_distributions_v3.png",
            "exd_hox_primary_split_affinity_distributions_v3.pdf",
            "exd_hox_nested_subset_counts_v3.png",
            "exd_hox_nested_subset_counts_v3.pdf",
            "exd_hox_primary_split_leakage_v3.png",
            "exd_hox_primary_split_leakage_v3.pdf",
            "exd_hox_paper_vs_primary_split_counts_v3.png",
            "exd_hox_paper_vs_primary_split_counts_v3.pdf",
        }
        observed_names = {Path(entry["path"]).name for entry in manifest["outputs"]}
        self.assertEqual(observed_names, expected_names)
        for entry in manifest["outputs"]:
            self.assertEqual(
                Path(entry["path"]).parent.as_posix(),
                V3_PLOT_CONTRACT.plot_logical_directory,
            )
        self.assertNotIn(PLOT_MANIFEST_FILENAME, observed_names)
        self.assertEqual(set(OUTPUT_FILENAMES), expected_names)
        self.assertEqual(
            {path.name for path in self.plot_directory.iterdir()},
            expected_names | {V3_PLOT_CONTRACT.manifest_filename},
        )

    def test_manifest_schema_selects_contract_before_filename_validation(self):
        manifest = self._generate()

        wrong_schema = copy.deepcopy(manifest)
        wrong_schema["schema_version"] = V2_PLOT_CONTRACT.manifest_schema_version
        self._write_manifest(self._rehash_manifest(wrong_schema))
        with self.assertRaisesRegex(ValueError, "canonical"):
            self._validate()

        self._write_manifest(manifest)
        wrong_name_path = self.plot_directory / V2_PLOT_CONTRACT.manifest_filename
        self.manifest_path.rename(wrong_name_path)
        with self.assertRaisesRegex(ValueError, "missing, extra|canonical"):
            validate_primary_split_plot_manifest(
                wrong_name_path,
                self.root,
                self.generator_commit,
            )
        wrong_name_path.rename(self.manifest_path)

        unknown_schema = copy.deepcopy(manifest)
        unknown_schema["schema_version"] = (
            "exd_hox_primary_split_plot_manifest.unknown"
        )
        self._write_manifest(self._rehash_manifest(unknown_schema))
        with self.assertRaisesRegex(ValueError, "schema"):
            self._validate()

    def test_low_data_display_label_is_strict_and_ignores_level_id(self):
        expected_labels = (
            ("absolute", "128", "n=128"),
            ("absolute", "256", "n=256"),
            ("absolute", "512", "n=512"),
            ("fractional", "0.01", "1%"),
            ("fractional", "0.02", "2%"),
            ("fractional", "0.05", "5%"),
            ("fractional", "0.1", "10%"),
            ("fractional", "0.25", "25%"),
            ("fractional", "0.5", "50%"),
            ("fractional", "1.0", "100%"),
        )
        for request_type, request_value, expected in expected_labels:
            with self.subTest(
                request_type=request_type,
                request_value=request_value,
            ):
                self.assertEqual(
                    plot_module._low_data_display_label(
                        request_type,
                        request_value,
                    ),
                    expected,
                )
        with self.assertRaisesRegex(ValueError, "request type"):
            plot_module._low_data_display_label("opaque_level_id", "0.1")
        for invalid_fraction in (
            "",
            "0",
            "-0.01",
            "1.01",
            "nan",
            "inf",
            "not-a-number",
        ):
            with self.subTest(invalid_fraction=invalid_fraction):
                with self.assertRaises(ValueError):
                    plot_module._low_data_display_label(
                        "fractional",
                        invalid_fraction,
                    )

    def test_v3_exact_rendered_text_constrained_layout_and_bounding_boxes(self):
        output_directory = Path(self.temporary_directory.name) / "captured_figures"
        count_rows = self._count_rows()
        affinity_rows = self._affinity_rows()
        leakage_rows = self._leakage_rows()
        subset_rows = self._subset_rows()
        count_source_rows = plot_module._count_source_rows(
            TRANSCRIPTION_FACTORS,
            count_rows,
        )
        affinity_source_rows = plot_module._affinity_source_rows(
            TRANSCRIPTION_FACTORS,
            affinity_rows,
            count_rows,
        )
        leakage_source_rows = plot_module._leakage_source_rows(leakage_rows)
        subset_source_rows = plot_module._subset_source_rows(
            TRANSCRIPTION_FACTORS,
            subset_rows,
        )
        comparison_source_rows = plot_module._comparison_source_rows(
            TRANSCRIPTION_FACTORS,
            count_rows,
        )

        count_figure = self._capture_figure(
            plot_module._plot_primary_counts,
            output_directory,
            count_source_rows,
        )
        count_axis = count_figure.axes[0]
        self.assertEqual(count_axis.get_title(), "Exd-Hox primary split counts")
        self.assertEqual(
            count_axis.get_ylabel(),
            "Number of logical labeled examples",
        )
        self.assertEqual(count_axis.get_ylim()[0], 0.0)
        self.assertEqual(
            tuple(label.get_text() for label in count_axis.get_xticklabels()),
            TRANSCRIPTION_FACTORS,
        )
        self.assertEqual(
            self._legend_text_values(count_axis),
            ("Training", "Validation", "Test"),
        )

        affinity_figure = self._capture_figure(
            plot_module._plot_affinity_distributions,
            output_directory,
            TRANSCRIPTION_FACTORS,
            affinity_source_rows,
        )
        affinity_axes = [axis for axis in affinity_figure.axes if axis.get_visible()]
        self.assertEqual(len(affinity_axes), len(TRANSCRIPTION_FACTORS))
        for axis, transcription_factor in zip(
            affinity_axes,
            TRANSCRIPTION_FACTORS,
        ):
            self.assertEqual(axis.get_title(), transcription_factor)
            self.assertEqual(axis.get_xlabel(), "Relative binding affinity (0–1)")
            self.assertEqual(axis.get_ylabel(), "Normalized density")
            self.assertEqual(axis.get_xlim(), (0.0, 1.0))
            self.assertEqual(axis.get_ylim()[0], 0.0)
        self.assertEqual(
            self._legend_text_values(affinity_axes[0]),
            ("Training", "Validation"),
        )
        self.assertIn(
            "Training/validation affinity distributions; test targets sealed",
            self._figure_text_values(affinity_figure),
        )

        subset_figure = self._capture_figure(
            plot_module._plot_subset_counts,
            output_directory,
            TRANSCRIPTION_FACTORS,
            subset_source_rows,
        )
        subset_axes = [axis for axis in subset_figure.axes if axis.get_visible()]
        for axis, transcription_factor in zip(subset_axes, TRANSCRIPTION_FACTORS):
            self.assertEqual(axis.get_title(), transcription_factor)
            self.assertEqual(axis.get_xlabel(), "Low-data training level")
            self.assertEqual(
                axis.get_ylabel(),
                "Number of labeled training examples",
            )
            self.assertEqual(axis.get_ylim()[0], 0.0)
            self.assertTrue(all(label.get_visible() for label in axis.get_xticklabels()))
            self.assertEqual(
                tuple(label.get_text() for label in axis.get_xticklabels()),
                ("n=2", "100%"),
            )
        self.assertEqual(
            self._legend_text_values(subset_axes[0]),
            ("Canonical requested count", "Actual subset count"),
        )
        normalized_subset_text = {
            " ".join(value.split())
            for value in self._figure_text_values(subset_figure)
        }
        self.assertTrue(
            any(
                "Requested versus actual nested low-data counts" in value
                for value in normalized_subset_text
            )
        )
        self.assertTrue(
            any(SUBSET_CAPTION in value for value in normalized_subset_text)
        )

        leakage_figure = self._capture_figure(
            plot_module._plot_leakage,
            output_directory,
            leakage_source_rows,
        )
        leakage_axis = leakage_figure.axes[0]
        self.assertEqual(
            leakage_axis.get_title(),
            "Exact and reverse-complement leakage audit",
        )
        self.assertEqual(
            leakage_axis.get_ylabel(),
            "Cross-split overlap groups",
        )
        self.assertEqual(leakage_axis.get_ylim()[0], 0.0)
        self.assertEqual(
            self._legend_text_values(leakage_axis),
            (
                "Exact-sequence overlap groups",
                "RC-equivalent overlap groups (includes exact)",
                "RC-only overlap groups",
            ),
        )
        leakage_ticks = tuple(
            label.get_text() for label in leakage_axis.get_xticklabels()
        )
        self.assertIn("Supplied paper split\nTrain vs Test", leakage_ticks)
        self.assertNotIn("paper_split_reproduction", "\n".join(leakage_ticks))
        normalized_leakage_text = {
            " ".join(value.split())
            for value in self._figure_text_values(leakage_figure)
        }
        self.assertTrue(
            any(LEAKAGE_CAPTION in value for value in normalized_leakage_text)
        )

        comparison_figure = self._capture_figure(
            plot_module._plot_protocol_comparison,
            output_directory,
            TRANSCRIPTION_FACTORS,
            comparison_source_rows,
        )
        comparison_axis = comparison_figure.axes[0]
        self.assertEqual(
            comparison_axis.get_title(),
            "Supplied split row occurrences versus primary logical examples",
        )
        self.assertEqual(comparison_axis.get_ylabel(), "Count")
        self.assertEqual(comparison_axis.get_ylim()[0], 0.0)
        self.assertEqual(
            tuple(label.get_text() for label in comparison_axis.get_xticklabels()),
            TRANSCRIPTION_FACTORS,
        )
        self.assertEqual(
            self._legend_text_values(comparison_axis),
            (
                "Supplied train (row occurrences)",
                "Supplied test (row occurrences)",
                "Primary training (logical labeled examples)",
                "Primary validation (logical labeled examples)",
                "Primary test (logical labeled examples)",
            ),
        )
        normalized_comparison_text = {
            " ".join(value.split())
            for value in self._figure_text_values(comparison_figure)
        }
        self.assertTrue(
            any(COMPARISON_CAPTION in value for value in normalized_comparison_text)
        )

    def test_eight_tf_ten_level_layout_hides_all_internal_level_ids(self):
        rows = plot_module._subset_source_rows(
            PRODUCTION_TRANSCRIPTION_FACTORS,
            self._stress_subset_rows(),
        )
        figure = self._capture_figure(
            plot_module._plot_subset_counts,
            Path(self.temporary_directory.name) / "stress_figure",
            PRODUCTION_TRANSCRIPTION_FACTORS,
            rows,
        )
        axes_by_tf = {
            axis.get_title(): axis
            for axis in figure.axes
            if axis.get_visible() and axis.get_title()
        }
        self.assertEqual(set(axes_by_tf), set(PRODUCTION_TRANSCRIPTION_FACTORS))
        observed_orders = []
        all_visible_tick_text = []
        for transcription_factor in PRODUCTION_TRANSCRIPTION_FACTORS:
            axis = axes_by_tf[transcription_factor]
            self.assertTrue(axis.has_data())
            self.assertEqual(len(axis.lines), 2)
            selected = []
            for row in rows:
                if row["transcription_factor"] == transcription_factor:
                    selected.append(row)
            expected_ticks = tuple(
                plot_module._low_data_display_label(
                    row["request_type"],
                    row["request_value"],
                )
                for row in selected
            )
            observed_ticks = tuple(
                label.get_text() for label in axis.get_xticklabels()
            )
            self.assertEqual(observed_ticks, expected_ticks)
            self.assertEqual(len(observed_ticks), 10)
            self.assertTrue(all(label.get_visible() for label in axis.get_xticklabels()))
            self.assertEqual(axis.get_xlabel(), "Low-data training level")
            self.assertNotIn("lvl_", "\n".join(observed_ticks))
            self.assertIsNone(
                re.search(r"[0-9a-f]{24,}", "\n".join(observed_ticks))
            )
            for label in axis.get_xticklabels() + axis.get_yticklabels():
                if label.get_visible():
                    all_visible_tick_text.append(label.get_text())
            expected_requested = tuple(
                int(row["canonical_requested_logical_example_count"])
                for row in selected
            )
            self.assertEqual(expected_requested, tuple(sorted(expected_requested)))
            observed_requested = tuple(
                int(value) for value in axis.lines[0].get_ydata()
            )
            self.assertEqual(observed_requested, expected_requested)
            observed_orders.append(expected_ticks)
        self.assertGreater(len(set(observed_orders)), 1)
        combined_tick_text = "\n".join(all_visible_tick_text)
        self.assertNotIn("lvl_", combined_tick_text)
        self.assertIsNone(re.search(r"[0-9a-f]{24,}", combined_tick_text))

    def test_v3_source_table_serialization_matches_explicit_v2_byte_fixtures(
        self,
    ):
        fixture_transcription_factors = ("FixtureTF",)
        count_input_rows = (
            {
                "protocol": "paper_split_reproduction",
                "transcription_factor": "FixtureTF",
                "split": "test",
                "row_count": "4",
                "logical_example_count": "4",
                "global_rc_group_count": "4",
                "exact_cross_split_overlap_occurrence_count": "1",
            },
            {
                "protocol": "primary",
                "transcription_factor": "FixtureTF",
                "split": "validation",
                "row_count": "2",
                "logical_example_count": "2",
                "global_rc_group_count": "2",
                "exact_cross_split_overlap_occurrence_count": "0",
            },
            {
                "protocol": "primary",
                "transcription_factor": "FixtureTF",
                "split": "test",
                "row_count": "3",
                "logical_example_count": "3",
                "global_rc_group_count": "3",
                "exact_cross_split_overlap_occurrence_count": "0",
            },
            {
                "protocol": "paper_split_reproduction",
                "transcription_factor": "FixtureTF",
                "split": "train",
                "row_count": "8",
                "logical_example_count": "8",
                "global_rc_group_count": "8",
                "exact_cross_split_overlap_occurrence_count": "1",
            },
            {
                "protocol": "primary",
                "transcription_factor": "FixtureTF",
                "split": "training",
                "row_count": "7",
                "logical_example_count": "7",
                "global_rc_group_count": "6",
                "exact_cross_split_overlap_occurrence_count": "0",
            },
        )
        affinity_input_rows = (
            {
                "transcription_factor": "FixtureTF",
                "split": "validation",
                "bin_index": "0",
                "bin_left": "0.50",
                "bin_right": "1.00",
                "logical_example_count": "2",
            },
            {
                "transcription_factor": "FixtureTF",
                "split": "training",
                "bin_index": "1",
                "bin_left": "0.5",
                "bin_right": "1",
                "logical_example_count": "4",
            },
            {
                "transcription_factor": "FixtureTF",
                "split": "training",
                "bin_index": "0",
                "bin_left": "0",
                "bin_right": "0.5",
                "logical_example_count": "3",
            },
        )
        subset_input_rows = (
            {
                "transcription_factor": "FixtureTF",
                "level_id": "opaque_fraction_25",
                "request_type": "fractional",
                "request_value": "0.25",
                "unaliased_requested_logical_example_count": "26",
                "alias_absolute_anchor": "25",
                "canonical_requested_logical_example_count": "25",
                "actual_logical_example_count": "25",
                "actual_rc_group_count": "24",
                "inclusive_maximum_rank": "25",
            },
            {
                "transcription_factor": "FixtureTF",
                "level_id": "opaque_absolute_5",
                "request_type": "absolute",
                "request_value": "5",
                "unaliased_requested_logical_example_count": "5",
                "alias_absolute_anchor": "",
                "canonical_requested_logical_example_count": "5",
                "actual_logical_example_count": "5",
                "actual_rc_group_count": "5",
                "inclusive_maximum_rank": "5",
            },
        )
        leakage_input_rows = (
            {
                "comparison": "primary",
                "left_split": "training",
                "right_split": "validation",
                "exact_sequence_overlap_group_count": "0",
                "reverse_complement_equivalent_overlap_group_count": "2",
                "reverse_complement_only_overlap_group_count": "2",
                "logical_example_overlap_count": "3",
            },
            {
                "comparison": "paper_split_reproduction",
                "left_split": "train",
                "right_split": "test",
                "exact_sequence_overlap_group_count": "2",
                "reverse_complement_equivalent_overlap_group_count": "3",
                "reverse_complement_only_overlap_group_count": "1",
                "logical_example_overlap_count": "4",
            },
        )

        expected_rows = {
            "count": (
                {
                    "transcription_factor": "FixtureTF",
                    "split": "training",
                    "logical_example_count": 7,
                    "global_rc_group_count": 6,
                },
                {
                    "transcription_factor": "FixtureTF",
                    "split": "validation",
                    "logical_example_count": 2,
                    "global_rc_group_count": 2,
                },
                {
                    "transcription_factor": "FixtureTF",
                    "split": "test",
                    "logical_example_count": 3,
                    "global_rc_group_count": 3,
                },
            ),
            "affinity": (
                {
                    "record_type": "affinity_histogram",
                    "transcription_factor": "FixtureTF",
                    "split": "training",
                    "bin_index": 0,
                    "bin_left": "0",
                    "bin_right": "0.5",
                    "logical_example_count": 3,
                },
                {
                    "record_type": "affinity_histogram",
                    "transcription_factor": "FixtureTF",
                    "split": "training",
                    "bin_index": 1,
                    "bin_left": "0.5",
                    "bin_right": "1",
                    "logical_example_count": 4,
                },
                {
                    "record_type": "affinity_histogram",
                    "transcription_factor": "FixtureTF",
                    "split": "validation",
                    "bin_index": 0,
                    "bin_left": "0.50",
                    "bin_right": "1.00",
                    "logical_example_count": 2,
                },
                {
                    "record_type": "test_count",
                    "transcription_factor": "FixtureTF",
                    "split": "test",
                    "bin_index": "",
                    "bin_left": "",
                    "bin_right": "",
                    "logical_example_count": 3,
                },
            ),
            "subset": (
                {
                    "transcription_factor": "FixtureTF",
                    "level_id": "opaque_absolute_5",
                    "request_type": "absolute",
                    "request_value": "5",
                    "unaliased_requested_logical_example_count": "5",
                    "alias_absolute_anchor": "",
                    "canonical_requested_logical_example_count": "5",
                    "actual_logical_example_count": "5",
                    "actual_rc_group_count": "5",
                    "inclusive_maximum_rank": "5",
                },
                {
                    "transcription_factor": "FixtureTF",
                    "level_id": "opaque_fraction_25",
                    "request_type": "fractional",
                    "request_value": "0.25",
                    "unaliased_requested_logical_example_count": "26",
                    "alias_absolute_anchor": "25",
                    "canonical_requested_logical_example_count": "25",
                    "actual_logical_example_count": "25",
                    "actual_rc_group_count": "24",
                    "inclusive_maximum_rank": "25",
                },
            ),
            "leakage": (
                {
                    "comparison": "paper_split_reproduction",
                    "left_split": "train",
                    "right_split": "test",
                    "exact_sequence_overlap_group_count": "2",
                    "reverse_complement_equivalent_overlap_group_count": "3",
                    "reverse_complement_only_overlap_group_count": "1",
                    "logical_example_overlap_count": "4",
                },
                {
                    "comparison": "primary",
                    "left_split": "training",
                    "right_split": "validation",
                    "exact_sequence_overlap_group_count": "0",
                    "reverse_complement_equivalent_overlap_group_count": "2",
                    "reverse_complement_only_overlap_group_count": "2",
                    "logical_example_overlap_count": "3",
                },
            ),
            "comparison": (
                {
                    "protocol": "paper_split_reproduction",
                    "transcription_factor": "FixtureTF",
                    "split": "train",
                    "logical_example_count": 8,
                },
                {
                    "protocol": "paper_split_reproduction",
                    "transcription_factor": "FixtureTF",
                    "split": "test",
                    "logical_example_count": 4,
                },
                {
                    "protocol": "primary",
                    "transcription_factor": "FixtureTF",
                    "split": "training",
                    "logical_example_count": 7,
                },
                {
                    "protocol": "primary",
                    "transcription_factor": "FixtureTF",
                    "split": "validation",
                    "logical_example_count": 2,
                },
                {
                    "protocol": "primary",
                    "transcription_factor": "FixtureTF",
                    "split": "test",
                    "logical_example_count": 3,
                },
            ),
        }
        transformed_rows = {
            "count": plot_module._count_source_rows(
                fixture_transcription_factors,
                count_input_rows,
            ),
            "affinity": plot_module._affinity_source_rows(
                fixture_transcription_factors,
                affinity_input_rows,
                count_input_rows,
            ),
            "subset": plot_module._subset_source_rows(
                fixture_transcription_factors,
                subset_input_rows,
            ),
            "leakage": plot_module._leakage_source_rows(leakage_input_rows),
            "comparison": plot_module._comparison_source_rows(
                fixture_transcription_factors,
                count_input_rows,
            ),
        }

        explicit_fields = {
            "count": (
                "transcription_factor",
                "split",
                "logical_example_count",
                "global_rc_group_count",
            ),
            "affinity": (
                "record_type",
                "transcription_factor",
                "split",
                "bin_index",
                "bin_left",
                "bin_right",
                "logical_example_count",
            ),
            "subset": (
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
            ),
            "leakage": (
                "comparison",
                "left_split",
                "right_split",
                "exact_sequence_overlap_group_count",
                "reverse_complement_equivalent_overlap_group_count",
                "reverse_complement_only_overlap_group_count",
                "logical_example_overlap_count",
            ),
            "comparison": (
                "protocol",
                "transcription_factor",
                "split",
                "logical_example_count",
            ),
        }
        v2_fields = {
            "count": COUNT_SOURCE_FIELDS,
            "affinity": AFFINITY_SOURCE_FIELDS,
            "subset": SUBSET_INPUT_FIELDS,
            "leakage": LEAKAGE_INPUT_FIELDS,
            "comparison": COMPARISON_SOURCE_FIELDS,
        }
        v3_fields = {
            "count": COUNT_SOURCE_FIELDS,
            "affinity": AFFINITY_SOURCE_FIELDS,
            "subset": SUBSET_INPUT_FIELDS,
            "leakage": LEAKAGE_INPUT_FIELDS,
            "comparison": COMPARISON_SOURCE_FIELDS,
        }
        v2_names = {
            "count": V2_PLOT_CONTRACT.count_source_filename,
            "affinity": V2_PLOT_CONTRACT.affinity_source_filename,
            "subset": V2_PLOT_CONTRACT.subset_source_filename,
            "leakage": V2_PLOT_CONTRACT.leakage_source_filename,
            "comparison": V2_PLOT_CONTRACT.comparison_source_filename,
        }
        v3_names = {
            "count": V3_PLOT_CONTRACT.count_source_filename,
            "affinity": V3_PLOT_CONTRACT.affinity_source_filename,
            "subset": V3_PLOT_CONTRACT.subset_source_filename,
            "leakage": V3_PLOT_CONTRACT.leakage_source_filename,
            "comparison": V3_PLOT_CONTRACT.comparison_source_filename,
        }
        expected_v2_names = {
            "count": "exd_hox_primary_split_counts_plot_source_v2.tsv",
            "affinity": "exd_hox_primary_split_affinity_plot_source_v2.tsv",
            "subset": "exd_hox_nested_subset_counts_plot_source_v2.tsv",
            "leakage": "exd_hox_primary_split_leakage_plot_source_v2.tsv",
            "comparison": "exd_hox_paper_vs_primary_split_plot_source_v2.tsv",
        }
        expected_v3_names = {
            "count": "exd_hox_primary_split_counts_plot_source_v3.tsv",
            "affinity": "exd_hox_primary_split_affinity_plot_source_v3.tsv",
            "subset": "exd_hox_nested_subset_counts_plot_source_v3.tsv",
            "leakage": "exd_hox_primary_split_leakage_plot_source_v3.tsv",
            "comparison": "exd_hox_paper_vs_primary_split_plot_source_v3.tsv",
        }
        expected_bytes = {
            "count": (
                b"transcription_factor\tsplit\tlogical_example_count\t"
                b"global_rc_group_count\n"
                b"FixtureTF\ttraining\t7\t6\n"
                b"FixtureTF\tvalidation\t2\t2\n"
                b"FixtureTF\ttest\t3\t3\n"
            ),
            "affinity": (
                b"record_type\ttranscription_factor\tsplit\tbin_index\tbin_left\t"
                b"bin_right\tlogical_example_count\n"
                b"affinity_histogram\tFixtureTF\ttraining\t0\t0\t0.5\t3\n"
                b"affinity_histogram\tFixtureTF\ttraining\t1\t0.5\t1\t4\n"
                b"affinity_histogram\tFixtureTF\tvalidation\t0\t0.50\t1.00\t2\n"
                b"test_count\tFixtureTF\ttest\t\t\t\t3\n"
            ),
            "subset": (
                b"transcription_factor\tlevel_id\trequest_type\trequest_value\t"
                b"unaliased_requested_logical_example_count\t"
                b"alias_absolute_anchor\t"
                b"canonical_requested_logical_example_count\t"
                b"actual_logical_example_count\tactual_rc_group_count\t"
                b"inclusive_maximum_rank\n"
                b"FixtureTF\topaque_absolute_5\tabsolute\t5\t5\t\t5\t5\t5\t5\n"
                b"FixtureTF\topaque_fraction_25\tfractional\t0.25\t26\t25\t25\t"
                b"25\t24\t25\n"
            ),
            "leakage": (
                b"comparison\tleft_split\tright_split\t"
                b"exact_sequence_overlap_group_count\t"
                b"reverse_complement_equivalent_overlap_group_count\t"
                b"reverse_complement_only_overlap_group_count\t"
                b"logical_example_overlap_count\n"
                b"paper_split_reproduction\ttrain\ttest\t2\t3\t1\t4\n"
                b"primary\ttraining\tvalidation\t0\t2\t2\t3\n"
            ),
            "comparison": (
                b"protocol\ttranscription_factor\tsplit\tlogical_example_count\n"
                b"paper_split_reproduction\tFixtureTF\ttrain\t8\n"
                b"paper_split_reproduction\tFixtureTF\ttest\t4\n"
                b"primary\tFixtureTF\ttraining\t7\n"
                b"primary\tFixtureTF\tvalidation\t2\n"
                b"primary\tFixtureTF\ttest\t3\n"
            ),
        }

        self.assertEqual(v2_fields, v3_fields)
        self.assertEqual(v2_fields, explicit_fields)
        self.assertEqual(v3_fields, explicit_fields)
        self.assertEqual(v2_names, expected_v2_names)
        self.assertEqual(v3_names, expected_v3_names)
        self.assertEqual(transformed_rows, expected_rows)

        temporary_root = Path(self.temporary_directory.name)
        serialization_root = temporary_root / "source_table_serialization"
        v2_directory = serialization_root / "v2"
        v3_directory = serialization_root / "v3"
        v2_directory.mkdir(parents=True)
        v3_directory.mkdir(parents=True)
        self.assertEqual(serialization_root.parent, temporary_root)
        for name in ("count", "affinity", "subset", "leakage", "comparison"):
            with self.subTest(source=name):
                self.assertNotEqual(v2_names[name], v3_names[name])
                v2_path = v2_directory / v2_names[name]
                v3_path = v3_directory / v3_names[name]
                write_tsv_exclusive(
                    v2_path,
                    v2_fields[name],
                    transformed_rows[name],
                )
                write_tsv_exclusive(
                    v3_path,
                    v3_fields[name],
                    transformed_rows[name],
                )
                v2_bytes = v2_path.read_bytes()
                v3_bytes = v3_path.read_bytes()
                self.assertEqual(v2_bytes, expected_bytes[name])
                self.assertEqual(v3_bytes, expected_bytes[name])
                self.assertEqual(v2_bytes, v3_bytes)
                self.assertEqual(
                    self._read_tsv_fixture(v2_path, explicit_fields[name]),
                    self._read_tsv_fixture(v3_path, explicit_fields[name]),
                )
        self.assertEqual(
            {path.name for path in v2_directory.iterdir()},
            set(expected_v2_names.values()),
        )
        self.assertEqual(
            {path.name for path in v3_directory.iterdir()},
            set(expected_v3_names.values()),
        )

    def test_v3_figure_metadata_and_save_geometry_policy(self):
        self._generate()
        encoded_creator = V3_PLOT_CONTRACT.creator_metadata.encode("utf-8")
        for filename in V3_PLOT_CONTRACT.output_filenames:
            suffix = Path(filename).suffix
            if suffix not in (".png", ".pdf"):
                continue
            with self.subTest(filename=filename):
                output_bytes = (self.plot_directory / filename).read_bytes()
                self.assertIn(encoded_creator, output_bytes)
                self.assertNotIn(b"dna-flex-pretrain Milestone 3D-B.1", output_bytes)
        save_source = inspect.getsource(plot_module._save_figure_pair)
        self.assertNotIn("bbox_inches", save_source)

    def test_v3_validation_uses_the_historical_producer_blob(self):
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

    def test_historical_v2_validation_uses_v2_contract_from_later_checkout(self):
        v2_config_path = self.root / V2_PLOT_CONTRACT.config_logical_path
        v2_plot_directory = self.root / V2_PLOT_CONTRACT.plot_logical_directory
        v3_config_path = self.root / V3_PLOT_CONTRACT.config_logical_path
        v3_plot_directory = self.root / V3_PLOT_CONTRACT.plot_logical_directory
        self.assertTrue(v2_config_path.is_file())
        self.assertFalse(v2_plot_directory.exists())
        self.assertTrue(v3_config_path.is_file())
        self.assertFalse(v3_plot_directory.exists())
        self._write_plot_config(
            v2_config_path,
            self.finalized,
            V2_PLOT_CONTRACT,
        )
        self._git(
            "add",
            "--",
            (
                self.project_prefix / V2_PLOT_CONTRACT.config_logical_path
            ).as_posix(),
        )
        self._commit("Synthetic historical v2 config")
        v2_generator_commit = self._head()
        generated = plot_primary_split_tables(
            v2_config_path,
            self.root,
            v2_generator_commit,
        )
        v2_manifest_path = v2_plot_directory / V2_PLOT_CONTRACT.manifest_filename
        self.assertEqual(v2_manifest_path.parent, v2_plot_directory)
        self.assertEqual(v2_manifest_path.name, V2_PLOT_CONTRACT.manifest_filename)
        producer_path = self.root / PLOTTING_ENTRY_POINT
        producer_path.write_bytes(producer_path.read_bytes() + b"\n# later v2 checkout\n")
        v3_config_path.unlink()
        self._git(
            "add",
            "--",
            (self.project_prefix / PLOTTING_ENTRY_POINT).as_posix(),
            (
                self.project_prefix / V3_PLOT_CONTRACT.config_logical_path
            ).as_posix(),
        )
        self._commit("Later checkout after historical v2 generation")
        self.assertNotEqual(self._head(), v2_generator_commit)
        self.assertFalse(v3_config_path.exists())
        self.assertFalse(v3_plot_directory.exists())
        validated = validate_primary_split_plot_manifest(
            v2_manifest_path,
            self.root,
            v2_generator_commit,
        )
        self.assertEqual(validated, generated)
        self.assertEqual(validated["plot_generator_commit"], v2_generator_commit)
        self.assertEqual(
            validated["schema_version"],
            V2_PLOT_CONTRACT.manifest_schema_version,
        )
        self.assertNotEqual(
            validated["schema_version"],
            V3_PLOT_CONTRACT.manifest_schema_version,
        )
        self.assertEqual(
            validated["plot_config_path"],
            V2_PLOT_CONTRACT.config_logical_path,
        )
        self.assertEqual(
            validated["plot_directory"],
            V2_PLOT_CONTRACT.plot_logical_directory,
        )
        self.assertEqual(len(validated["inputs"]), 6)
        self.assertEqual(len(validated["outputs"]), 15)
        observed_names = {
            Path(entry["path"]).name for entry in validated["outputs"]
        }
        self.assertEqual(observed_names, set(V2_PLOT_CONTRACT.output_filenames))
        self.assertTrue(
            observed_names.isdisjoint(set(V3_PLOT_CONTRACT.output_filenames))
        )
        self.assertEqual(
            {path.name for path in v2_plot_directory.iterdir()},
            set(V2_PLOT_CONTRACT.output_filenames)
            | {V2_PLOT_CONTRACT.manifest_filename},
        )
        with mock.patch.dict(
            plot_module.PLOT_CONTRACTS_BY_MANIFEST_SCHEMA,
            {
                V2_PLOT_CONTRACT.manifest_schema_version: V3_PLOT_CONTRACT,
            },
        ):
            with self.assertRaisesRegex(ValueError, "Plot config path differs"):
                validate_primary_split_plot_manifest(
                    v2_manifest_path,
                    self.root,
                    v2_generator_commit,
                )

    def test_historical_v1_and_v2_artifacts_remain_byte_identical(self):
        v1_directory = self.root / "plots/exd_hox_primary_split_v1"
        v1_directory.mkdir(parents=True)
        for index, filename in enumerate(
            V2_PLOT_CONTRACT.output_filenames
            + (V2_PLOT_CONTRACT.manifest_filename,)
        ):
            v1_name = filename.replace("_v2", "_v1")
            (v1_directory / v1_name).write_bytes(
                "historical-v1-{0}\n".format(index).encode("ascii")
            )
        v2_directory = self.root / V2_PLOT_CONTRACT.plot_logical_directory
        v2_directory.mkdir(parents=True)
        for index, filename in enumerate(
            V2_PLOT_CONTRACT.output_filenames
            + (V2_PLOT_CONTRACT.manifest_filename,)
        ):
            (v2_directory / filename).write_bytes(
                "historical-v2-{0}\n".format(index).encode("ascii")
            )
        v1_before = self._byte_map(v1_directory)
        v2_before = self._byte_map(v2_directory)
        self._generate()
        self.assertEqual(self._byte_map(v1_directory), v1_before)
        self.assertEqual(self._byte_map(v2_directory), v2_before)

    def test_existing_v3_output_path_refuses_overwrite(self):
        self.plot_directory.parent.mkdir(parents=True, exist_ok=True)
        self.plot_directory.write_bytes(b"existing-v3-path\n")
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            self._generate()
        self.assertEqual(self.plot_directory.read_bytes(), b"existing-v3-path\n")
        self.plot_directory.unlink()

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

    def test_rehashed_v1_manifest_is_rejected_as_unsupported(self):
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
        with self.assertRaisesRegex(ValueError, "schema"):
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
        with self.assertRaisesRegex(ValueError, "missing, extra"):
            self._validate()
        extra_path.unlink()

        relocated_name = "relocated_output_v3.tsv"
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
