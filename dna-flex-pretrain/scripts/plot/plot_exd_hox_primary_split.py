"""Plot finalized Exd-Hox primary split and subset tables only.

Run with ``python -m scripts.plot.plot_exd_hox_primary_split``.
The plotter deliberately has no HDF5, split-generation, or sealed-target
dependency.  Test affinities are represented only by aggregate test counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    validate_hashed_manifest,
    validate_repository_relative_path,
    write_json_exclusive,
    write_tsv_exclusive,
)


CONFIG_SCHEMA_VERSION = "exd_hox_primary_split_config.v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "exd_hox_primary_split_manifest.v1"
SUBSET_MANIFEST_SCHEMA_VERSION = "exd_hox_subset_set_manifest.v1"
PLOT_MANIFEST_SCHEMA_VERSION = "exd_hox_primary_split_plot_manifest.v1"

COUNT_INPUT_FILENAME = "exd_hox_primary_split_count_summary_v1.tsv"
AFFINITY_INPUT_FILENAME = "exd_hox_primary_split_affinity_histogram_v1.tsv"
LEAKAGE_INPUT_FILENAME = "exd_hox_primary_split_leakage_audit_v1.tsv"
SUBSET_INPUT_FILENAME = "exd_hox_nested_subset_levels_v1.tsv"
SPLIT_MANIFEST_FILENAME = "exd_hox_primary_split_manifest_v1.json"
SUBSET_MANIFEST_FILENAME = "exd_hox_subset_set_manifest_v1.json"

COUNT_SOURCE_FILENAME = "exd_hox_primary_split_counts_plot_source_v1.tsv"
AFFINITY_SOURCE_FILENAME = "exd_hox_primary_split_affinity_plot_source_v1.tsv"
SUBSET_SOURCE_FILENAME = "exd_hox_nested_subset_counts_plot_source_v1.tsv"
LEAKAGE_SOURCE_FILENAME = "exd_hox_primary_split_leakage_plot_source_v1.tsv"
COMPARISON_SOURCE_FILENAME = "exd_hox_paper_vs_primary_split_plot_source_v1.tsv"
PLOT_MANIFEST_FILENAME = "exd_hox_primary_split_plot_manifest_v1.json"

COUNT_PLOT_STEM = "exd_hox_primary_split_counts_v1"
AFFINITY_PLOT_STEM = "exd_hox_primary_split_affinity_distributions_v1"
SUBSET_PLOT_STEM = "exd_hox_nested_subset_counts_v1"
LEAKAGE_PLOT_STEM = "exd_hox_primary_split_leakage_v1"
COMPARISON_PLOT_STEM = "exd_hox_paper_vs_primary_split_counts_v1"

COUNT_INPUT_FIELDS = (
    "protocol",
    "transcription_factor",
    "split",
    "row_count",
    "logical_example_count",
    "global_rc_group_count",
    "exact_cross_split_overlap_occurrence_count",
)
AFFINITY_INPUT_FIELDS = (
    "transcription_factor",
    "split",
    "bin_index",
    "bin_left",
    "bin_right",
    "logical_example_count",
)
LEAKAGE_INPUT_FIELDS = (
    "comparison",
    "left_split",
    "right_split",
    "exact_sequence_overlap_group_count",
    "reverse_complement_equivalent_overlap_group_count",
    "reverse_complement_only_overlap_group_count",
    "logical_example_overlap_count",
)
SUBSET_INPUT_FIELDS = (
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
)

COUNT_SOURCE_FIELDS = (
    "transcription_factor",
    "split",
    "logical_example_count",
    "global_rc_group_count",
)
AFFINITY_SOURCE_FIELDS = (
    "record_type",
    "transcription_factor",
    "split",
    "bin_index",
    "bin_left",
    "bin_right",
    "logical_example_count",
)
COMPARISON_SOURCE_FIELDS = (
    "protocol",
    "transcription_factor",
    "split",
    "logical_example_count",
)

PRIMARY_SPLITS = ("training", "validation", "test")
PAPER_SPLITS = ("train", "test")
AFFINITY_SPLITS = ("training", "validation")
FORBIDDEN_PLAINTEXT_FIELDS = frozenset(
    (
        "affinity",
        "affinity_value",
        "plaintext_target",
        "target",
        "target_bits",
        "target_float32_bits",
        "target_value",
    )
)


def parse_arguments(argv=None):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Plot finalized Exd-Hox primary split and subset tables."
    )
    parser.add_argument(
        "--config",
        default="configs/exd_hox_primary_split_v1.yaml",
    )
    parser.add_argument("--repository-root", default=".")
    return parser.parse_args(argv)


def plot_primary_split_tables(
    config_path: Path | str,
    repository_root: Path | str,
) -> Dict[str, Any]:
    """Create five immutable plot families from finalized public tables."""

    root = Path(repository_root).resolve()
    config_candidate = Path(config_path)
    if not config_candidate.is_absolute():
        config_candidate = root / config_candidate
    config_file = config_candidate.resolve()
    config_logical_path = repository_relative_path(config_file, root)
    _reject_protected_input_path(config_logical_path)
    if config_file.is_symlink() or not config_file.is_file():
        raise FileNotFoundError("Plot config must be a regular public file.")
    config = _load_plot_config(config_file)

    split_directory = _resolve_input_directory(
        root,
        str(config["outputs"]["split_directory"]),
    )
    subset_directory = _resolve_input_directory(
        root,
        str(config["outputs"]["subset_directory"]),
    )
    plot_logical_directory = str(config["outputs"]["plot_directory"])
    plot_directory = _resolve_repository_path(root, plot_logical_directory)
    _require_finalized_directory(split_directory, "split")
    _require_finalized_directory(subset_directory, "subset")
    _require_new_plot_directory(plot_directory)

    count_path = split_directory / COUNT_INPUT_FILENAME
    affinity_path = split_directory / AFFINITY_INPUT_FILENAME
    leakage_path = split_directory / LEAKAGE_INPUT_FILENAME
    subset_path = subset_directory / SUBSET_INPUT_FILENAME
    split_manifest_path = split_directory / SPLIT_MANIFEST_FILENAME
    subset_manifest_path = subset_directory / SUBSET_MANIFEST_FILENAME
    for input_path in (
        count_path,
        affinity_path,
        leakage_path,
        subset_path,
        split_manifest_path,
        subset_manifest_path,
    ):
        _require_public_regular_file(input_path, root)

    split_manifest = _load_hashed_manifest(
        split_manifest_path,
        SPLIT_MANIFEST_SCHEMA_VERSION,
    )
    subset_manifest = _load_hashed_manifest(
        subset_manifest_path,
        SUBSET_MANIFEST_SCHEMA_VERSION,
    )
    if subset_manifest.get("split_manifest_hash") != split_manifest["manifest_hash"]:
        raise ValueError("Subset manifest does not bind the split manifest identity.")

    split_inputs = (count_path, affinity_path, leakage_path)
    subset_inputs = (subset_path,)
    for input_path in split_inputs:
        _validate_manifest_artifact(split_manifest, input_path, root)
    for input_path in subset_inputs:
        _validate_manifest_artifact(subset_manifest, input_path, root)

    input_paths = (
        count_path,
        affinity_path,
        leakage_path,
        subset_path,
        split_manifest_path,
        subset_manifest_path,
    )
    initial_input_fingerprints = _fingerprint_paths(input_paths, root)

    count_rows = _read_tsv(count_path, COUNT_INPUT_FIELDS)
    affinity_rows = _read_tsv(affinity_path, AFFINITY_INPUT_FIELDS)
    leakage_rows = _read_tsv(leakage_path, LEAKAGE_INPUT_FIELDS)
    subset_rows = _read_tsv(subset_path, SUBSET_INPUT_FIELDS)
    transcription_factors = tuple(config["dataset"]["transcription_factors"])

    _validate_count_rows(transcription_factors, count_rows)
    _validate_affinity_rows(transcription_factors, affinity_rows)
    _validate_leakage_rows(leakage_rows)
    _validate_subset_rows(transcription_factors, subset_rows)

    count_source_rows = _count_source_rows(transcription_factors, count_rows)
    affinity_source_rows = _affinity_source_rows(
        transcription_factors,
        affinity_rows,
        count_rows,
    )
    subset_source_rows = _subset_source_rows(transcription_factors, subset_rows)
    leakage_source_rows = _leakage_source_rows(leakage_rows)
    comparison_source_rows = _comparison_source_rows(
        transcription_factors,
        count_rows,
    )

    plot_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_context = tempfile.TemporaryDirectory(
        prefix=".exd_hox_primary_split_plot_staging_",
        dir=plot_directory.parent,
    )
    staging_directory = Path(staging_context.name) / "plots"
    staging_directory.mkdir()
    try:
        write_tsv_exclusive(
            staging_directory / COUNT_SOURCE_FILENAME,
            COUNT_SOURCE_FIELDS,
            count_source_rows,
        )
        write_tsv_exclusive(
            staging_directory / AFFINITY_SOURCE_FILENAME,
            AFFINITY_SOURCE_FIELDS,
            affinity_source_rows,
        )
        write_tsv_exclusive(
            staging_directory / SUBSET_SOURCE_FILENAME,
            SUBSET_INPUT_FIELDS,
            subset_source_rows,
        )
        write_tsv_exclusive(
            staging_directory / LEAKAGE_SOURCE_FILENAME,
            LEAKAGE_INPUT_FIELDS,
            leakage_source_rows,
        )
        write_tsv_exclusive(
            staging_directory / COMPARISON_SOURCE_FILENAME,
            COMPARISON_SOURCE_FIELDS,
            comparison_source_rows,
        )

        _plot_primary_counts(staging_directory, count_source_rows)
        _plot_affinity_distributions(
            staging_directory,
            transcription_factors,
            affinity_source_rows,
        )
        _plot_subset_counts(
            staging_directory,
            transcription_factors,
            subset_source_rows,
        )
        _plot_leakage(staging_directory, leakage_source_rows)
        _plot_protocol_comparison(
            staging_directory,
            transcription_factors,
            comparison_source_rows,
        )

        current_input_fingerprints = _fingerprint_paths(input_paths, root)
        if current_input_fingerprints != initial_input_fingerprints:
            raise ValueError("Finalized plot input changed while plotting.")
        manifest = _write_plot_manifest(
            config=config,
            config_file=config_file,
            repository_root=root,
            plot_logical_directory=plot_logical_directory,
            staging_directory=staging_directory,
            input_fingerprints=initial_input_fingerprints,
            split_manifest=split_manifest,
            subset_manifest=subset_manifest,
        )
        _require_new_plot_directory(plot_directory)
        os.rename(staging_directory, plot_directory)
    finally:
        staging_context.cleanup()
    return manifest


def _load_plot_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    if not isinstance(payload, Mapping):
        raise ValueError("Primary split plot config must be a mapping.")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported Exd-Hox primary split config schema.")
    outputs = payload.get("outputs")
    dataset = payload.get("dataset")
    if not isinstance(outputs, Mapping):
        raise ValueError("Primary split config outputs must be a mapping.")
    if not isinstance(dataset, Mapping):
        raise ValueError("Primary split config dataset must be a mapping.")
    for key in ("split_directory", "subset_directory", "plot_directory"):
        if key not in outputs:
            raise ValueError("Missing config outputs.{0}.".format(key))
        validate_repository_relative_path(str(outputs[key]))
    transcription_factors = dataset.get("transcription_factors")
    if not isinstance(transcription_factors, Sequence) or isinstance(
        transcription_factors,
        (str, bytes),
    ):
        raise ValueError("Dataset transcription_factors must be a sequence.")
    normalized_tfs = tuple(str(value) for value in transcription_factors)
    if not normalized_tfs or len(set(normalized_tfs)) != len(normalized_tfs):
        raise ValueError("Dataset transcription factors must be nonempty and unique.")
    study = payload.get("study")
    if not isinstance(study, Mapping):
        raise ValueError("Primary split config study must be a mapping.")
    if not study.get("dataset_identifier"):
        raise ValueError("Study dataset identifier must not be empty.")
    return dict(payload)


def _resolve_input_directory(repository_root: Path, relative_path: str) -> Path:
    normalized = validate_repository_relative_path(relative_path)
    _reject_protected_input_path(normalized)
    resolved = _resolve_repository_path(repository_root, normalized)
    resolved_logical_path = repository_relative_path(resolved, repository_root)
    _reject_protected_input_path(resolved_logical_path)
    return resolved


def _reject_protected_input_path(logical_path: str) -> None:
    path = Path(logical_path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        raise ValueError("Plot inputs must not use HDF5 files.")
    for component in path.parts:
        lowered = component.lower()
        if "sealed" in lowered or lowered == "test_targets":
            raise ValueError("Plot inputs must not use a sealed-target directory.")


def _require_public_regular_file(path: Path, repository_root: Path) -> None:
    logical_path = repository_relative_path(path, repository_root)
    _reject_protected_input_path(logical_path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            "Finalized plot input must be a regular public file: {0}".format(path)
        )


def _resolve_repository_path(repository_root: Path, relative_path: str) -> Path:
    normalized = validate_repository_relative_path(relative_path)
    resolved = (repository_root / normalized).resolve()
    repository_relative_path(resolved, repository_root)
    return resolved


def _require_finalized_directory(path: Path, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        message = "Finalized {0} directory does not exist: {1}"
        raise FileNotFoundError(message.format(label, path))


def _require_new_plot_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite existing plot directory: {0}".format(path)
        )


def _load_hashed_manifest(path: Path, expected_schema: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise ValueError("Finalized manifest must be a mapping.")
    if payload.get("schema_version") != expected_schema:
        raise ValueError("Unsupported finalized manifest schema.")
    validate_hashed_manifest(payload)
    if not isinstance(payload.get("policy"), Mapping):
        raise ValueError("Finalized manifest policy must be a mapping.")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, (Mapping, Sequence)) or isinstance(
        artifacts,
        (str, bytes),
    ):
        raise ValueError("Finalized manifest artifacts must be a collection.")
    return dict(payload)


def _artifact_entries(manifest: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    artifacts = manifest["artifacts"]
    if isinstance(artifacts, Mapping):
        candidates = tuple(artifacts.values())
    else:
        candidates = tuple(artifacts)
    entries = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Manifest artifact entry must be a mapping.")
        required = ("path", "byte_size", "sha256")
        if any(field not in candidate for field in required):
            raise ValueError("Manifest artifact entry is incomplete.")
        entries.append(candidate)
    return tuple(entries)


def _validate_manifest_artifact(
    manifest: Mapping[str, Any],
    physical_path: Path,
    repository_root: Path,
) -> None:
    logical_path = repository_relative_path(physical_path, repository_root)
    matches = []
    for artifact in _artifact_entries(manifest):
        if artifact["path"] == logical_path:
            matches.append(artifact)
    if len(matches) != 1:
        message = "Manifest must bind exactly one artifact for {0}."
        raise ValueError(message.format(logical_path))
    expected = matches[0]
    observed = fingerprint_file(physical_path, logical_path).to_dict()
    normalized_expected = {
        "path": str(expected["path"]),
        "byte_size": int(expected["byte_size"]),
        "sha256": str(expected["sha256"]),
    }
    if observed != normalized_expected:
        raise ValueError(
            "Finalized artifact fingerprint mismatch: {0}".format(logical_path)
        )


def _fingerprint_paths(
    paths: Sequence[Path],
    repository_root: Path,
) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for path in paths:
        logical_path = repository_relative_path(path, repository_root)
        rows.append(fingerprint_file(path, logical_path).to_dict())
    rows.sort(key=lambda row: row["path"])
    return tuple(rows)


def _read_tsv(
    path: Path,
    expected_fields: Sequence[str],
) -> Tuple[Dict[str, str], ...]:
    with open(path, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            message = "Unexpected finalized TSV schema for {0}."
            raise ValueError(message.format(path.name))
        forbidden = set(reader.fieldnames or ()).intersection(
            FORBIDDEN_PLAINTEXT_FIELDS
        )
        if forbidden:
            raise ValueError("Public plot input contains plaintext target fields.")
        return tuple(dict(row) for row in reader)


def _nonnegative_integer(value: str, label: str) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{0} must be an integer.".format(label)) from error
    if converted < 0:
        raise ValueError("{0} must be nonnegative.".format(label))
    return converted


def _finite_float(value: str, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{0} must be numeric.".format(label)) from error
    if not math.isfinite(converted):
        raise ValueError("{0} must be finite.".format(label))
    return converted


def _validate_count_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    expected_pairs = set()
    for transcription_factor in transcription_factors:
        for split in PRIMARY_SPLITS:
            expected_pairs.add(("primary", transcription_factor, split))
        for split in PAPER_SPLITS:
            expected_pairs.add(
                ("paper_split_reproduction", transcription_factor, split)
            )
    observed_pairs = set()
    for row in rows:
        key = (row["protocol"], row["transcription_factor"], row["split"])
        if key in observed_pairs:
            raise ValueError("Duplicate count-summary row.")
        observed_pairs.add(key)
        for field in COUNT_INPUT_FIELDS[3:]:
            _nonnegative_integer(row[field], "Count summary {0}".format(field))
    if observed_pairs != expected_pairs:
        raise ValueError("Count summary protocol, TF, or split coverage mismatch.")


def _validate_affinity_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    expected_tfs = set(transcription_factors)
    observed_pairs = set()
    indices_by_pair: Dict[Tuple[str, str], list[int]] = {}
    for row in rows:
        transcription_factor = row["transcription_factor"]
        split = row["split"]
        if transcription_factor not in expected_tfs:
            raise ValueError("Affinity histogram contains an unexpected TF.")
        if split not in AFFINITY_SPLITS:
            raise ValueError(
                "Affinity histogram may contain only training and validation rows."
            )
        pair = (transcription_factor, split)
        observed_pairs.add(pair)
        bin_index = _nonnegative_integer(row["bin_index"], "Affinity bin index")
        left = _finite_float(row["bin_left"], "Affinity bin left edge")
        right = _finite_float(row["bin_right"], "Affinity bin right edge")
        if right <= left:
            raise ValueError("Affinity histogram bins must have positive width.")
        _nonnegative_integer(
            row["logical_example_count"],
            "Affinity histogram logical-example count",
        )
        indices_by_pair.setdefault(pair, []).append(bin_index)
    expected_pairs = set()
    for transcription_factor in transcription_factors:
        for split in AFFINITY_SPLITS:
            expected_pairs.add((transcription_factor, split))
    if observed_pairs != expected_pairs:
        raise ValueError("Affinity histogram TF/split coverage mismatch.")
    for indices in indices_by_pair.values():
        ordered = sorted(indices)
        if ordered != list(range(len(ordered))):
            raise ValueError("Affinity histogram bin indices must be contiguous.")


def _validate_leakage_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ValueError("Leakage audit must not be empty.")
    seen = set()
    for row in rows:
        key = (row["comparison"], row["left_split"], row["right_split"])
        if key in seen:
            raise ValueError("Duplicate leakage-audit row.")
        seen.add(key)
        for field in LEAKAGE_INPUT_FIELDS[3:]:
            _nonnegative_integer(row[field], "Leakage {0}".format(field))


def _validate_subset_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    expected_tfs = set(transcription_factors)
    observed_tfs = set()
    seen = set()
    for row in rows:
        transcription_factor = row["transcription_factor"]
        if transcription_factor not in expected_tfs:
            raise ValueError("Subset levels contain an unexpected TF.")
        observed_tfs.add(transcription_factor)
        key = (transcription_factor, row["level_id"])
        if key in seen:
            raise ValueError("Duplicate subset level row.")
        seen.add(key)
        for field in (
            "unaliased_requested_logical_example_count",
            "canonical_requested_logical_example_count",
            "actual_logical_example_count",
            "actual_rc_group_count",
            "inclusive_maximum_rank",
        ):
            _nonnegative_integer(row[field], "Subset {0}".format(field))
    if observed_tfs != expected_tfs:
        raise ValueError("Subset-level TF coverage mismatch.")


def _count_source_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    by_key = {}
    for row in rows:
        by_key[(row["protocol"], row["transcription_factor"], row["split"])] = row
    selected = []
    for transcription_factor in transcription_factors:
        for split in PRIMARY_SPLITS:
            row = by_key[("primary", transcription_factor, split)]
            selected.append(
                {
                    "transcription_factor": transcription_factor,
                    "split": split,
                    "logical_example_count": int(row["logical_example_count"]),
                    "global_rc_group_count": int(row["global_rc_group_count"]),
                }
            )
    return tuple(selected)


def _affinity_source_rows(
    transcription_factors: Sequence[str],
    affinity_rows: Sequence[Mapping[str, str]],
    count_rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    tf_order = {}
    for index, transcription_factor in enumerate(transcription_factors):
        tf_order[transcription_factor] = index
    split_order = {"training": 0, "validation": 1}
    selected = []
    for row in affinity_rows:
        selected.append(
            {
                "record_type": "affinity_histogram",
                "transcription_factor": row["transcription_factor"],
                "split": row["split"],
                "bin_index": int(row["bin_index"]),
                "bin_left": row["bin_left"],
                "bin_right": row["bin_right"],
                "logical_example_count": int(row["logical_example_count"]),
            }
        )
    selected.sort(
        key=lambda row: (
            tf_order[row["transcription_factor"]],
            split_order[row["split"]],
            row["bin_index"],
        )
    )
    count_by_key = {}
    for row in count_rows:
        count_by_key[(row["protocol"], row["transcription_factor"], row["split"])] = row
    for transcription_factor in transcription_factors:
        test_row = count_by_key[("primary", transcription_factor, "test")]
        selected.append(
            {
                "record_type": "test_count",
                "transcription_factor": transcription_factor,
                "split": "test",
                "bin_index": "",
                "bin_left": "",
                "bin_right": "",
                "logical_example_count": int(test_row["logical_example_count"]),
            }
        )
    return tuple(selected)


def _subset_source_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    tf_order = {}
    for index, transcription_factor in enumerate(transcription_factors):
        tf_order[transcription_factor] = index
    selected = [dict(row) for row in rows]
    selected.sort(
        key=lambda row: (
            tf_order[row["transcription_factor"]],
            int(row["canonical_requested_logical_example_count"]),
            row["level_id"],
        )
    )
    return tuple(selected)


def _leakage_source_rows(
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    selected = [dict(row) for row in rows]
    selected.sort(
        key=lambda row: (
            row["comparison"],
            row["left_split"],
            row["right_split"],
        )
    )
    return tuple(selected)


def _comparison_source_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    by_key = {}
    for row in rows:
        by_key[(row["protocol"], row["transcription_factor"], row["split"])] = row
    selected = []
    for transcription_factor in transcription_factors:
        for protocol, splits in (
            ("paper_split_reproduction", PAPER_SPLITS),
            ("primary", PRIMARY_SPLITS),
        ):
            for split in splits:
                row = by_key[(protocol, transcription_factor, split)]
                selected.append(
                    {
                        "protocol": protocol,
                        "transcription_factor": transcription_factor,
                        "split": split,
                        "logical_example_count": int(
                            row["logical_example_count"]
                        ),
                    }
                )
    return tuple(selected)


def _plot_primary_counts(
    output_directory: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    labels = []
    for row in rows:
        if row["transcription_factor"] not in labels:
            labels.append(row["transcription_factor"])
    by_key = {}
    for row in rows:
        by_key[(row["transcription_factor"], row["split"])] = int(
            row["logical_example_count"]
        )
    positions = np.arange(len(labels))
    width = 0.25
    colors = {"training": "#3366AA", "validation": "#EE7733", "test": "#009988"}
    figure, axis = plt.subplots(figsize=(9.4, 5.4))
    for split_index, split in enumerate(PRIMARY_SPLITS):
        values = [by_key[(label, split)] for label in labels]
        offset = (split_index - 1) * width
        axis.bar(
            positions + offset,
            values,
            width,
            color=colors[split],
            label=split.capitalize(),
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Logical examples")
    axis.set_title("Exd-Hox primary split counts")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure_pair(figure, output_directory, COUNT_PLOT_STEM)
    plt.close(figure)


def _panel_layout(panel_count: int) -> Tuple[plt.Figure, np.ndarray]:
    column_count = 1 if panel_count == 1 else 2
    row_count = int(math.ceil(panel_count / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(10.0, max(4.0, 3.1 * row_count)),
        squeeze=False,
        sharex=False,
    )
    return figure, axes


def _plot_affinity_distributions(
    output_directory: Path,
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    figure, axes = _panel_layout(len(transcription_factors))
    colors = {"training": "#3366AA", "validation": "#EE7733"}
    flat_axes = axes.flat
    for axis, transcription_factor in zip(flat_axes, transcription_factors):
        for split in AFFINITY_SPLITS:
            selected = []
            for row in rows:
                if (
                    row["record_type"] == "affinity_histogram"
                    and row["transcription_factor"] == transcription_factor
                    and row["split"] == split
                ):
                    selected.append(row)
            left = np.asarray(
                [float(row["bin_left"]) for row in selected],
                dtype=np.float64,
            )
            right = np.asarray(
                [float(row["bin_right"]) for row in selected],
                dtype=np.float64,
            )
            counts = np.asarray(
                [int(row["logical_example_count"]) for row in selected],
                dtype=np.float64,
            )
            widths = right - left
            total = np.sum(counts)
            density = counts if total == 0 else counts / total / widths
            axis.plot(
                (left + right) / 2.0,
                density,
                color=colors[split],
                label=split.capitalize(),
                linewidth=1.5,
            )
        test_counts = []
        for row in rows:
            if (
                row["record_type"] == "test_count"
                and row["transcription_factor"] == transcription_factor
            ):
                test_counts.append(int(row["logical_example_count"]))
        axis.text(
            0.98,
            0.96,
            "Test count: {0}".format(test_counts[0]),
            transform=axis.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=9,
        )
        axis.set_title(transcription_factor)
        axis.set_xlabel("Relative affinity")
        axis.set_ylabel("Density")
        axis.spines[["top", "right"]].set_visible(False)
    for unused_index in range(len(transcription_factors), axes.size):
        axes.flat[unused_index].set_visible(False)
    axes.flat[0].legend(frameon=False)
    figure.suptitle(
        "Training/validation affinity distributions; test targets sealed",
        y=0.995,
    )
    figure.tight_layout()
    _save_figure_pair(figure, output_directory, AFFINITY_PLOT_STEM)
    plt.close(figure)


def _plot_subset_counts(
    output_directory: Path,
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    figure, axes = _panel_layout(len(transcription_factors))
    for axis, transcription_factor in zip(axes.flat, transcription_factors):
        selected = []
        for row in rows:
            if row["transcription_factor"] == transcription_factor:
                selected.append(row)
        positions = np.arange(len(selected))
        requested = [
            int(row["canonical_requested_logical_example_count"])
            for row in selected
        ]
        actual = [int(row["actual_logical_example_count"]) for row in selected]
        axis.plot(
            positions,
            requested,
            marker="o",
            linewidth=1.2,
            color="#777777",
            label="Requested",
        )
        axis.plot(
            positions,
            actual,
            marker="s",
            linewidth=1.2,
            color="#AA4499",
            label="Actual",
        )
        axis.set_xticks(
            positions,
            [row["level_id"] for row in selected],
            rotation=45,
            horizontalalignment="right",
        )
        axis.set_title(transcription_factor)
        axis.set_ylabel("Logical examples")
        axis.spines[["top", "right"]].set_visible(False)
    for unused_index in range(len(transcription_factors), axes.size):
        axes.flat[unused_index].set_visible(False)
    axes.flat[0].legend(frameon=False)
    figure.suptitle("Requested versus actual nested low-data counts", y=0.995)
    figure.tight_layout()
    _save_figure_pair(figure, output_directory, SUBSET_PLOT_STEM)
    plt.close(figure)


def _plot_leakage(
    output_directory: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    labels = [
        "{0}\n{1} vs {2}".format(
            row["comparison"],
            row["left_split"],
            row["right_split"],
        )
        for row in rows
    ]
    positions = np.arange(len(rows))
    width = 0.25
    series = (
        ("Exact", "exact_sequence_overlap_group_count", "#CC6677"),
        (
            "RC-equivalent",
            "reverse_complement_equivalent_overlap_group_count",
            "#4477AA",
        ),
        ("RC-only", "reverse_complement_only_overlap_group_count", "#228833"),
    )
    figure, axis = plt.subplots(figsize=(max(8.0, 2.1 * len(rows)), 5.4))
    for series_index, (label, field, color) in enumerate(series):
        values = [int(row[field]) for row in rows]
        offset = (series_index - 1) * width
        axis.bar(positions + offset, values, width, color=color, label=label)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Cross-split overlap groups")
    axis.set_title("Exact and reverse-complement leakage audit")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure_pair(figure, output_directory, LEAKAGE_PLOT_STEM)
    plt.close(figure)


def _plot_protocol_comparison(
    output_directory: Path,
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    by_key = {}
    for row in rows:
        by_key[(row["protocol"], row["transcription_factor"], row["split"])] = int(
            row["logical_example_count"]
        )
    positions = np.arange(len(transcription_factors))
    categories = (
        ("paper_split_reproduction", "train", "Paper train", "#88CCEE"),
        ("paper_split_reproduction", "test", "Paper test", "#CC6677"),
        ("primary", "training", "Primary train", "#3366AA"),
        ("primary", "validation", "Primary validation", "#EE7733"),
        ("primary", "test", "Primary test", "#009988"),
    )
    width = 0.16
    figure, axis = plt.subplots(figsize=(10.2, 5.6))
    center = (len(categories) - 1) / 2.0
    for category_index, (protocol, split, label, color) in enumerate(categories):
        values = [
            by_key[(protocol, transcription_factor, split)]
            for transcription_factor in transcription_factors
        ]
        offset = (category_index - center) * width
        axis.bar(positions + offset, values, width, color=color, label=label)
    axis.set_xticks(positions, transcription_factors)
    axis.set_ylabel("Logical examples")
    axis.set_title("Supplied paper split versus exact/RC-safe primary split")
    axis.legend(frameon=False, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure_pair(figure, output_directory, COMPARISON_PLOT_STEM)
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
            metadata={"Software": "dna-flex-pretrain Milestone 3D-B"},
        )
    with open(pdf_path, "xb") as pdf_file:
        figure.savefig(
            pdf_file,
            format="pdf",
            metadata={
                "Creator": "dna-flex-pretrain Milestone 3D-B",
                "Producer": "matplotlib",
                "CreationDate": None,
                "ModDate": None,
            },
        )


def _write_plot_manifest(
    config: Mapping[str, Any],
    config_file: Path,
    repository_root: Path,
    plot_logical_directory: str,
    staging_directory: Path,
    input_fingerprints: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    subset_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    output_filenames = (
        COUNT_SOURCE_FILENAME,
        AFFINITY_SOURCE_FILENAME,
        SUBSET_SOURCE_FILENAME,
        LEAKAGE_SOURCE_FILENAME,
        COMPARISON_SOURCE_FILENAME,
        "{0}.png".format(COUNT_PLOT_STEM),
        "{0}.pdf".format(COUNT_PLOT_STEM),
        "{0}.png".format(AFFINITY_PLOT_STEM),
        "{0}.pdf".format(AFFINITY_PLOT_STEM),
        "{0}.png".format(SUBSET_PLOT_STEM),
        "{0}.pdf".format(SUBSET_PLOT_STEM),
        "{0}.png".format(LEAKAGE_PLOT_STEM),
        "{0}.pdf".format(LEAKAGE_PLOT_STEM),
        "{0}.png".format(COMPARISON_PLOT_STEM),
        "{0}.pdf".format(COMPARISON_PLOT_STEM),
    )
    output_fingerprints = []
    for filename in output_filenames:
        logical_path = Path(plot_logical_directory, filename).as_posix()
        fingerprint = fingerprint_file(
            staging_directory / filename,
            logical_path,
        )
        output_fingerprints.append(fingerprint.to_dict())
    output_fingerprints.sort(key=lambda row: row["path"])

    manifest = build_hashed_manifest(
        PLOT_MANIFEST_SCHEMA_VERSION,
        {
            "dataset_identifier": config["study"]["dataset_identifier"],
            "config_path": repository_relative_path(config_file, repository_root),
            "config_sha256": hash_file_bytes(config_file),
            "split_manifest_hash": split_manifest["manifest_hash"],
            "subset_set_manifest_hash": subset_manifest["manifest_hash"],
            "plot_directory": plot_logical_directory,
            "inputs": list(input_fingerprints),
            "outputs": output_fingerprints,
            "test_target_policy": (
                "aggregate_test_counts_only_no_test_affinity_distribution"
            ),
        },
    )
    write_json_exclusive(staging_directory / PLOT_MANIFEST_FILENAME, manifest)
    return manifest


def main(argv=None):
    """Generate immutable primary-split plot artifacts."""

    arguments = parse_arguments(argv)
    manifest = plot_primary_split_tables(
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
