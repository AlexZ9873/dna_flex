"""Tests for plotting finalized Exd-Hox audit tables only."""

from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.plot.plot_exd_hox_dataset_audit import plot_audit_tables
from src.downstream_fingerprints import write_tsv_exclusive


class ExdHoxAuditPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.audit_directory = self.root / "data" / "processed" / "audit_v1"
        self.plot_directory = self.root / "plots" / "audit_v1"
        self.audit_directory.mkdir(parents=True)
        self.config_path = self.root / "configs" / "import_v1.yaml"
        self.config_path.parent.mkdir()
        config = {
            "schema_version": "exd_hox_selex_import_config.v1",
            "dataset": {
                "identifier": "synthetic_exd_hox.v1",
                "transcription_factors": ["AbdA", "Ubx"],
            },
            "outputs": {
                "audit_directory": "data/processed/audit_v1",
                "plot_directory": "plots/audit_v1",
            },
        }
        self.config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        self._write_audit_tables()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_audit_tables(self) -> None:
        write_tsv_exclusive(
            self.audit_directory / "exd_hox_per_tf_count_summary_v1.tsv",
            (
                "transcription_factor",
                "supplied_training_rows",
                "supplied_test_rows",
            ),
            (
                {
                    "transcription_factor": "AbdA",
                    "supplied_training_rows": 8,
                    "supplied_test_rows": 2,
                },
                {
                    "transcription_factor": "Ubx",
                    "supplied_training_rows": 12,
                    "supplied_test_rows": 3,
                },
            ),
        )
        histogram_rows = []
        for transcription_factor in ("AbdA", "Ubx"):
            for supplied_split in ("train", "test"):
                histogram_rows.extend(
                    (
                        {
                            "transcription_factor": transcription_factor,
                            "supplied_split": supplied_split,
                            "bin_index": 0,
                            "bin_left": 0.0,
                            "bin_right": 0.5,
                            "row_count": 1,
                        },
                        {
                            "transcription_factor": transcription_factor,
                            "supplied_split": supplied_split,
                            "bin_index": 1,
                            "bin_left": 0.5,
                            "bin_right": 1.0,
                            "row_count": 1,
                        },
                    )
                )
        write_tsv_exclusive(
            self.audit_directory / "exd_hox_affinity_histogram_v1.tsv",
            (
                "transcription_factor",
                "supplied_split",
                "bin_index",
                "bin_left",
                "bin_right",
                "row_count",
            ),
            histogram_rows,
        )
        write_tsv_exclusive(
            self.audit_directory
            / "exd_hox_supplied_split_leakage_summary_v1.tsv",
            (
                "transcription_factor",
                "exact_labeled_row_overlap_count",
            ),
            (
                {
                    "transcription_factor": "AbdA",
                    "exact_labeled_row_overlap_count": 2,
                },
                {
                    "transcription_factor": "Ubx",
                    "exact_labeled_row_overlap_count": 4,
                },
            ),
        )

    def test_plots_use_saved_tables_and_refuse_overwrite(self) -> None:
        raw_directory = self.root / "data" / "raw"
        self.assertFalse(raw_directory.exists())

        manifest = plot_audit_tables(self.config_path, self.root)

        self.assertEqual(
            manifest["schema_version"],
            "exd_hox_dataset_audit_plot_manifest.v1",
        )
        self.assertTrue(
            (self.plot_directory / "exd_hox_supplied_counts_v1.png").is_file()
        )
        self.assertTrue(
            (
                self.plot_directory
                / "exd_hox_affinity_distributions_v1.pdf"
            ).is_file()
        )
        self.assertTrue(
            (
                self.plot_directory
                / "exd_hox_supplied_exact_overlaps_plot_source_v1.tsv"
            ).is_file()
        )
        self.assertFalse(raw_directory.exists())

        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            plot_audit_tables(self.config_path, self.root)


if __name__ == "__main__":
    unittest.main()
