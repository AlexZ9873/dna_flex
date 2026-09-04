"""Tests for finalized-table-only Exd-Hox primary split plotting."""

import builtins
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from scripts.plot.plot_exd_hox_primary_split import (
    AFFINITY_INPUT_FIELDS,
    AFFINITY_INPUT_FILENAME,
    AFFINITY_SOURCE_FILENAME,
    COUNT_INPUT_FIELDS,
    COUNT_INPUT_FILENAME,
    COUNT_SOURCE_FILENAME,
    LEAKAGE_INPUT_FIELDS,
    LEAKAGE_INPUT_FILENAME,
    LEAKAGE_SOURCE_FILENAME,
    PLOT_MANIFEST_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    SUBSET_INPUT_FIELDS,
    SUBSET_INPUT_FILENAME,
    SUBSET_MANIFEST_FILENAME,
    plot_primary_split_tables,
)
from src.downstream_fingerprints import (
    build_hashed_manifest,
    fingerprint_file,
    write_json_exclusive,
    write_tsv_exclusive,
)


TRANSCRIPTION_FACTORS = ("AbdA", "Ubx")


class ExdHoxPrimarySplitPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self._write_synthetic_repository(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _config_payload(self):
        return {
            "schema_version": "exd_hox_primary_split_config.v1",
            "study": {
                "identifier": "synthetic_exd_hox_study.v1",
                "dataset_identifier": "synthetic_exd_hox_primary.v1",
            },
            "dataset": {
                "transcription_factors": list(TRANSCRIPTION_FACTORS),
            },
            "outputs": {
                "split_directory": "data/processed/exd_hox_split_v1",
                "subset_directory": "data/processed/exd_hox_subsets_v1",
                "plot_directory": "plots/exd_hox_primary_split_v1",
            },
        }

    def _write_synthetic_repository(
        self,
        root: Path,
        include_test_affinity: bool = False,
    ) -> Path:
        config = self._config_payload()
        split_directory = root / config["outputs"]["split_directory"]
        subset_directory = root / config["outputs"]["subset_directory"]
        split_directory.mkdir(parents=True)
        subset_directory.mkdir(parents=True)
        config_path = root / "configs" / "exd_hox_primary_split_v1.yaml"
        config_path.parent.mkdir()
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

        count_path = split_directory / COUNT_INPUT_FILENAME
        affinity_path = split_directory / AFFINITY_INPUT_FILENAME
        leakage_path = split_directory / LEAKAGE_INPUT_FILENAME
        subset_path = subset_directory / SUBSET_INPUT_FILENAME
        write_tsv_exclusive(
            count_path,
            COUNT_INPUT_FIELDS,
            self._count_rows(),
        )
        write_tsv_exclusive(
            affinity_path,
            AFFINITY_INPUT_FIELDS,
            self._affinity_rows(include_test_affinity),
        )
        write_tsv_exclusive(
            leakage_path,
            LEAKAGE_INPUT_FIELDS,
            self._leakage_rows(),
        )
        write_tsv_exclusive(
            subset_path,
            SUBSET_INPUT_FIELDS,
            self._subset_rows(),
        )

        split_artifacts = []
        for path in (count_path, affinity_path, leakage_path):
            logical_path = path.relative_to(root).as_posix()
            split_artifacts.append(
                fingerprint_file(path, logical_path).to_dict()
            )
        split_artifacts.sort(key=lambda row: row["path"])
        split_manifest = build_hashed_manifest(
            "exd_hox_primary_split_manifest.v1",
            {
                "policy": {"identifier": "synthetic_primary_split.v1"},
                "artifacts": split_artifacts,
            },
        )
        write_json_exclusive(
            split_directory / SPLIT_MANIFEST_FILENAME,
            split_manifest,
        )

        subset_artifact = fingerprint_file(
            subset_path,
            subset_path.relative_to(root).as_posix(),
        ).to_dict()
        subset_manifest = build_hashed_manifest(
            "exd_hox_subset_set_manifest.v1",
            {
                "split_manifest_hash": split_manifest["manifest_hash"],
                "policy": {"identifier": "synthetic_nested_subsets.v1"},
                "artifacts": [subset_artifact],
            },
        )
        write_json_exclusive(
            subset_directory / SUBSET_MANIFEST_FILENAME,
            subset_manifest,
        )

        sealed_directory = root / "data" / "sealed" / "targets_v1"
        sealed_directory.mkdir(parents=True)
        (sealed_directory / "plaintext_test_targets.tsv").write_text(
            "logical_example_id\ttarget\nsecret\t0.75\n",
            encoding="utf-8",
        )
        return config_path

    def _count_rows(self):
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

    def _affinity_rows(self, include_test_affinity):
        rows = []
        for transcription_factor in TRANSCRIPTION_FACTORS:
            splits = ["training", "validation"]
            if include_test_affinity:
                splits.append("test")
            for split in splits:
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

    def _leakage_rows(self):
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

    def _subset_rows(self):
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

    def _guarded_open(self, original_open):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(os.fspath(file)))
                lowered_parts = tuple(part.lower() for part in path.parts)
                if path.suffix.lower() in (".h5", ".hdf5"):
                    raise AssertionError("Plotter attempted HDF5 access.")
                if any("sealed" in part for part in lowered_parts):
                    raise AssertionError("Plotter attempted sealed-target access.")
            return original_open(file, *args, **kwargs)

        return guarded

    def test_plotter_uses_only_public_finalized_tables(self) -> None:
        sealed_path = (
            self.root
            / "data"
            / "sealed"
            / "targets_v1"
            / "plaintext_test_targets.tsv"
        )
        sealed_before = sealed_path.read_bytes()
        original_open = builtins.open
        with mock.patch(
            "builtins.open",
            side_effect=self._guarded_open(original_open),
        ):
            manifest = plot_primary_split_tables(self.config_path, self.root)

        plot_directory = self.root / "plots" / "exd_hox_primary_split_v1"
        self.assertEqual(
            manifest["schema_version"],
            "exd_hox_primary_split_plot_manifest.v1",
        )
        self.assertIn("split_manifest_hash", manifest)
        self.assertIn("subset_set_manifest_hash", manifest)
        self.assertEqual(
            manifest["test_target_policy"],
            "aggregate_test_counts_only_no_test_affinity_distribution",
        )
        self.assertEqual(len(manifest["outputs"]), 15)
        self.assertTrue((plot_directory / PLOT_MANIFEST_FILENAME).is_file())
        self.assertTrue((plot_directory / COUNT_SOURCE_FILENAME).is_file())
        self.assertTrue((plot_directory / LEAKAGE_SOURCE_FILENAME).is_file())
        self.assertEqual(sealed_path.read_bytes(), sealed_before)

        with open(
            plot_directory / AFFINITY_SOURCE_FILENAME,
            "r",
            encoding="utf-8",
            newline="",
        ) as input_file:
            rows = tuple(csv.DictReader(input_file, delimiter="\t"))
        test_rows = [row for row in rows if row["split"] == "test"]
        self.assertEqual(len(test_rows), len(TRANSCRIPTION_FACTORS))
        self.assertTrue(all(row["record_type"] == "test_count" for row in test_rows))
        self.assertTrue(all(row["bin_left"] == "" for row in test_rows))
        serialized_source = (plot_directory / AFFINITY_SOURCE_FILENAME).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("target_value", serialized_source)
        self.assertNotIn("target_float32_bits", serialized_source)
        self.assertNotIn("0.75", serialized_source)

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        plot_primary_split_tables(self.config_path, self.root)
        plot_directory = self.root / "plots" / "exd_hox_primary_split_v1"
        before = {}
        for path in sorted(plot_directory.iterdir()):
            before[path.name] = path.read_bytes()

        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            plot_primary_split_tables(self.config_path, self.root)

        after = {}
        for path in sorted(plot_directory.iterdir()):
            after[path.name] = path.read_bytes()
        self.assertEqual(after, before)

    def test_table_and_manifest_tampering_are_rejected(self) -> None:
        split_directory = self.root / "data" / "processed" / "exd_hox_split_v1"
        count_path = split_directory / COUNT_INPUT_FILENAME
        count_path.write_bytes(count_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            plot_primary_split_tables(self.config_path, self.root)

        with tempfile.TemporaryDirectory() as second_directory:
            second_root = Path(second_directory)
            second_config = self._write_synthetic_repository(second_root)
            manifest_path = (
                second_root
                / "data"
                / "processed"
                / "exd_hox_split_v1"
                / SPLIT_MANIFEST_FILENAME
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["policy"]["identifier"] = "tampered"
            manifest_path.write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Manifest hash mismatch"):
                plot_primary_split_tables(second_config, second_root)

    def test_test_affinity_distribution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as second_directory:
            second_root = Path(second_directory)
            second_config = self._write_synthetic_repository(
                second_root,
                include_test_affinity=True,
            )
            with self.assertRaisesRegex(
                ValueError,
                "only training and validation",
            ):
                plot_primary_split_tables(second_config, second_root)

    def test_outputs_are_byte_identical_across_repository_roots(self) -> None:
        plot_primary_split_tables(self.config_path, self.root)
        first_directory = self.root / "plots" / "exd_hox_primary_split_v1"
        first_bytes = {}
        for path in sorted(first_directory.iterdir()):
            first_bytes[path.name] = path.read_bytes()

        with tempfile.TemporaryDirectory() as second_directory:
            second_root = Path(second_directory)
            second_config = self._write_synthetic_repository(second_root)
            plot_primary_split_tables(second_config, second_root)
            second_plot_directory = (
                second_root / "plots" / "exd_hox_primary_split_v1"
            )
            second_bytes = {}
            for path in sorted(second_plot_directory.iterdir()):
                second_bytes[path.name] = path.read_bytes()

        self.assertEqual(first_bytes, second_bytes)


if __name__ == "__main__":
    unittest.main()
