"""Plot finalized Exd-Hox audit tables without reopening HDF5 source files.

Run with ``python -m scripts.plot.plot_exd_hox_dataset_audit``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dna_flex_pretrain_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "dna_flex_pretrain_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.downstream_fingerprints import (
    build_hashed_manifest,
    fingerprint_file,
    hash_file_bytes,
    repository_relative_path,
    validate_repository_relative_path,
    write_json_exclusive,
    write_tsv_exclusive,
)


CONFIG_SCHEMA_VERSION = "exd_hox_selex_import_config.v1"
COUNT_INPUT_FILENAME = "exd_hox_per_tf_count_summary_v1.tsv"
AFFINITY_INPUT_FILENAME = "exd_hox_affinity_histogram_v1.tsv"
OVERLAP_INPUT_FILENAME = "exd_hox_supplied_split_leakage_summary_v1.tsv"

COUNT_SOURCE_FILENAME = "exd_hox_supplied_counts_plot_source_v1.tsv"
AFFINITY_SOURCE_FILENAME = "exd_hox_affinity_plot_source_v1.tsv"
OVERLAP_SOURCE_FILENAME = "exd_hox_supplied_exact_overlaps_plot_source_v1.tsv"
PLOT_MANIFEST_FILENAME = "exd_hox_dataset_audit_plot_manifest_v1.json"

COUNT_PLOT_STEM = "exd_hox_supplied_counts_v1"
AFFINITY_PLOT_STEM = "exd_hox_affinity_distributions_v1"
OVERLAP_PLOT_STEM = "exd_hox_supplied_exact_overlaps_v1"


def parse_arguments(argv=None):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Plot the finalized versioned Exd-Hox audit tables."
    )
    parser.add_argument(
        "--config",
        default="configs/exd_hox_selex_import_v1.yaml",
    )
    parser.add_argument("--repository-root", default=".")
    return parser.parse_args(argv)


def plot_audit_tables(
    config_path: Path | str,
    repository_root: Path | str,
) -> Dict[str, Any]:
    """Create three immutable plot families from saved audit tables only."""

    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    config = _load_plot_config(config_file)
    outputs = config["outputs"]
    audit_directory = _resolve_repository_path(
        root,
        str(outputs["audit_directory"]),
    )
    plot_directory = _resolve_repository_path(
        root,
        str(outputs["plot_directory"]),
    )
    if not audit_directory.is_dir():
        raise FileNotFoundError(
            "Finalized audit directory does not exist: {0}".format(
                audit_directory
            )
        )
    if plot_directory.exists() or plot_directory.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite existing plot directory: {0}".format(
                plot_directory
            )
        )

    count_input = audit_directory / COUNT_INPUT_FILENAME
    affinity_input = audit_directory / AFFINITY_INPUT_FILENAME
    overlap_input = audit_directory / OVERLAP_INPUT_FILENAME
    count_rows = _read_tsv(count_input)
    affinity_rows = _read_tsv(affinity_input)
    overlap_rows = _read_tsv(overlap_input)
    transcription_factors = tuple(
        config["dataset"]["transcription_factors"]
    )
    _validate_plot_table_coverage(
        transcription_factors,
        count_rows,
        affinity_rows,
        overlap_rows,
    )

    plot_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_context = tempfile.TemporaryDirectory(
        prefix=".exd_hox_plot_staging_",
        dir=plot_directory.parent,
    )
    staging_directory = Path(staging_context.name) / "plots"
    staging_directory.mkdir()
    try:
        count_source_rows = _count_source_rows(
            transcription_factors,
            count_rows,
        )
        affinity_source_rows = _affinity_source_rows(
            transcription_factors,
            affinity_rows,
        )
        overlap_source_rows = _overlap_source_rows(
            transcription_factors,
            overlap_rows,
        )
        write_tsv_exclusive(
            staging_directory / COUNT_SOURCE_FILENAME,
            (
                "transcription_factor",
                "supplied_training_rows",
                "supplied_test_rows",
            ),
            count_source_rows,
        )
        write_tsv_exclusive(
            staging_directory / AFFINITY_SOURCE_FILENAME,
            (
                "transcription_factor",
                "supplied_split",
                "bin_index",
                "bin_left",
                "bin_right",
                "row_count",
            ),
            affinity_source_rows,
        )
        write_tsv_exclusive(
            staging_directory / OVERLAP_SOURCE_FILENAME,
            (
                "transcription_factor",
                "exact_labeled_row_overlap_count",
            ),
            overlap_source_rows,
        )

        _plot_counts(staging_directory, count_source_rows)
        _plot_affinity_distributions(
            staging_directory,
            transcription_factors,
            affinity_source_rows,
        )
        _plot_overlap_counts(staging_directory, overlap_source_rows)

        manifest = _write_plot_manifest(
            config=config,
            config_file=config_file,
            repository_root=root,
            audit_directory=audit_directory,
            plot_logical_directory=str(outputs["plot_directory"]),
            staging_directory=staging_directory,
        )
        os.rename(staging_directory, plot_directory)
    finally:
        staging_context.cleanup()
    return manifest


def _load_plot_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    if not isinstance(payload, Mapping):
        raise ValueError("Plot config must be a mapping.")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported Exd-Hox plot config schema.")
    if not isinstance(payload.get("outputs"), Mapping):
        raise ValueError("Plot config outputs must be a mapping.")
    if not isinstance(payload.get("dataset"), Mapping):
        raise ValueError("Plot config dataset must be a mapping.")
    for key in ("audit_directory", "plot_directory"):
        validate_repository_relative_path(str(payload["outputs"][key]))
    return dict(payload)


def _read_tsv(path: Path) -> Tuple[Dict[str, str], ...]:
    with open(path, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Audit table has no TSV header: {0}".format(path))
        return tuple(dict(row) for row in reader)


def _validate_plot_table_coverage(
    transcription_factors: Sequence[str],
    count_rows: Sequence[Mapping[str, str]],
    affinity_rows: Sequence[Mapping[str, str]],
    overlap_rows: Sequence[Mapping[str, str]],
) -> None:
    expected = set(transcription_factors)
    count_tfs = {row.get("transcription_factor") for row in count_rows}
    overlap_tfs = {row.get("transcription_factor") for row in overlap_rows}
    affinity_tfs = {row.get("transcription_factor") for row in affinity_rows}
    if count_tfs != expected:
        raise ValueError("Count table TF coverage does not match the config.")
    if overlap_tfs != expected:
        raise ValueError("Overlap table TF coverage does not match the config.")
    if affinity_tfs != expected:
        raise ValueError("Affinity table TF coverage does not match the config.")
    affinity_pairs = {
        (row.get("transcription_factor"), row.get("supplied_split"))
        for row in affinity_rows
    }
    expected_pairs = set()
    for transcription_factor in transcription_factors:
        expected_pairs.add((transcription_factor, "train"))
        expected_pairs.add((transcription_factor, "test"))
    if not expected_pairs.issubset(affinity_pairs):
        raise ValueError("Affinity table must contain train and test bins per TF.")


def _count_source_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    by_tf = {row["transcription_factor"]: row for row in rows}
    selected = []
    for transcription_factor in transcription_factors:
        row = by_tf[transcription_factor]
        selected.append(
            {
                "transcription_factor": transcription_factor,
                "supplied_training_rows": int(row["supplied_training_rows"]),
                "supplied_test_rows": int(row["supplied_test_rows"]),
            }
        )
    return tuple(selected)


def _affinity_source_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    tf_order = {
        transcription_factor: index
        for index, transcription_factor in enumerate(transcription_factors)
    }
    selected = []
    for row in rows:
        if row["supplied_split"] not in ("train", "test"):
            continue
        selected.append(
            {
                "transcription_factor": row["transcription_factor"],
                "supplied_split": row["supplied_split"],
                "bin_index": int(row["bin_index"]),
                "bin_left": row["bin_left"],
                "bin_right": row["bin_right"],
                "row_count": int(row["row_count"]),
            }
        )
    split_order = {"train": 0, "test": 1}
    selected.sort(
        key=lambda row: (
            tf_order[row["transcription_factor"]],
            split_order[row["supplied_split"]],
            row["bin_index"],
        )
    )
    return tuple(selected)


def _overlap_source_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    by_tf = {row["transcription_factor"]: row for row in rows}
    selected = []
    for transcription_factor in transcription_factors:
        row = by_tf[transcription_factor]
        selected.append(
            {
                "transcription_factor": transcription_factor,
                "exact_labeled_row_overlap_count": int(
                    row["exact_labeled_row_overlap_count"]
                ),
            }
        )
    return tuple(selected)


def _plot_counts(
    output_directory: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    labels = [row["transcription_factor"] for row in rows]
    training = np.asarray(
        [row["supplied_training_rows"] for row in rows],
        dtype=np.int64,
    )
    test = np.asarray(
        [row["supplied_test_rows"] for row in rows],
        dtype=np.int64,
    )
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.bar(positions - width / 2, training, width, label="Supplied train")
    axis.bar(positions + width / 2, test, width, label="Supplied test")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Row occurrences")
    axis.set_title("Wang et al. Exd-Hox supplied split counts")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure_pair(figure, output_directory, COUNT_PLOT_STEM)
    plt.close(figure)


def _plot_affinity_distributions(
    output_directory: Path,
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(10.0, 12.0), sharex=True)
    colors = {"train": "#3366AA", "test": "#CC6677"}
    for axis, transcription_factor in zip(axes.flat, transcription_factors):
        for supplied_split in ("train", "test"):
            selected = [
                row
                for row in rows
                if row["transcription_factor"] == transcription_factor
                and row["supplied_split"] == supplied_split
            ]
            left = np.asarray(
                [float(row["bin_left"]) for row in selected],
                dtype=np.float64,
            )
            right = np.asarray(
                [float(row["bin_right"]) for row in selected],
                dtype=np.float64,
            )
            counts = np.asarray(
                [int(row["row_count"]) for row in selected],
                dtype=np.float64,
            )
            centers = (left + right) / 2.0
            widths = right - left
            density = counts / np.sum(counts) / widths
            axis.plot(
                centers,
                density,
                color=colors[supplied_split],
                label=supplied_split.capitalize(),
                linewidth=1.5,
            )
        axis.set_title(transcription_factor)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel("Density")
    for axis in axes[-1, :]:
        axis.set_xlabel("Relative affinity")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Exd-Hox supplied-split affinity distributions", y=0.995)
    figure.tight_layout()
    _save_figure_pair(figure, output_directory, AFFINITY_PLOT_STEM)
    plt.close(figure)


def _plot_overlap_counts(
    output_directory: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    labels = [row["transcription_factor"] for row in rows]
    counts = [row["exact_labeled_row_overlap_count"] for row in rows]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    bars = axis.bar(positions, counts, color="#AA4499")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Exact labeled-row overlaps")
    axis.set_title("Supplied train/test leakage (91 total overlaps)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.bar_label(bars, padding=3)
    figure.tight_layout()
    _save_figure_pair(figure, output_directory, OVERLAP_PLOT_STEM)
    plt.close(figure)


def _save_figure_pair(
    figure: plt.Figure,
    output_directory: Path,
    stem: str,
) -> None:
    png_path = output_directory / "{0}.png".format(stem)
    pdf_path = output_directory / "{0}.pdf".format(stem)
    with open(png_path, "xb") as png_file:
        figure.savefig(
            png_file,
            format="png",
            dpi=180,
            metadata={"Software": "dna-flex-pretrain Milestone 3C"},
        )
    with open(pdf_path, "xb") as pdf_file:
        figure.savefig(
            pdf_file,
            format="pdf",
            metadata={
                "Creator": "dna-flex-pretrain Milestone 3C",
                "Producer": "matplotlib",
                "CreationDate": None,
                "ModDate": None,
            },
        )


def _write_plot_manifest(
    config: Mapping[str, Any],
    config_file: Path,
    repository_root: Path,
    audit_directory: Path,
    plot_logical_directory: str,
    staging_directory: Path,
) -> Dict[str, Any]:
    input_filenames = (
        COUNT_INPUT_FILENAME,
        AFFINITY_INPUT_FILENAME,
        OVERLAP_INPUT_FILENAME,
    )
    input_fingerprints = []
    for filename in input_filenames:
        physical_path = audit_directory / filename
        logical_path = repository_relative_path(physical_path, repository_root)
        input_fingerprints.append(
            fingerprint_file(physical_path, logical_path).to_dict()
        )
    input_fingerprints.sort(key=lambda row: row["path"])

    output_filenames = (
        COUNT_SOURCE_FILENAME,
        AFFINITY_SOURCE_FILENAME,
        OVERLAP_SOURCE_FILENAME,
        "{0}.png".format(COUNT_PLOT_STEM),
        "{0}.pdf".format(COUNT_PLOT_STEM),
        "{0}.png".format(AFFINITY_PLOT_STEM),
        "{0}.pdf".format(AFFINITY_PLOT_STEM),
        "{0}.png".format(OVERLAP_PLOT_STEM),
        "{0}.pdf".format(OVERLAP_PLOT_STEM),
    )
    output_fingerprints = []
    for filename in output_filenames:
        logical_path = Path(plot_logical_directory, filename).as_posix()
        output_fingerprints.append(
            fingerprint_file(
                staging_directory / filename,
                logical_path,
            ).to_dict()
        )
    output_fingerprints.sort(key=lambda row: row["path"])

    manifest = build_hashed_manifest(
        "exd_hox_dataset_audit_plot_manifest.v1",
        {
            "dataset_identifier": config["dataset"]["identifier"],
            "config_path": repository_relative_path(
                config_file,
                repository_root,
            ),
            "config_sha256": hash_file_bytes(config_file),
            "plot_directory": plot_logical_directory,
            "input_audit_tables": input_fingerprints,
            "outputs": output_fingerprints,
        },
    )
    write_json_exclusive(
        staging_directory / PLOT_MANIFEST_FILENAME,
        manifest,
    )
    return manifest


def _resolve_repository_path(
    repository_root: Path,
    relative_path: str,
) -> Path:
    normalized = validate_repository_relative_path(relative_path)
    resolved = (repository_root / normalized).resolve()
    repository_relative_path(resolved, repository_root)
    return resolved


def main(argv=None):
    """Generate the immutable audit plot artifacts."""

    arguments = parse_arguments(argv)
    manifest = plot_audit_tables(
        config_path=arguments.config,
        repository_root=arguments.repository_root,
    )
    print(
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return manifest


if __name__ == "__main__":
    main()
