"""Import and audit the pinned Wang et al. Exd-Hox SELEX HDF5 files.

Run with ``python -m scripts.data_prep.import_audit_exd_hox_selex``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
from src.selex_hdf5 import (
    SelexHdf5File,
    audit_supplied_split,
    audit_within_tf,
    build_affinity_histogram_rows,
    build_affinity_summary_rows,
    build_count_summary_rows,
    build_cross_tf_sharing_rows,
    read_validate_selex_hdf5,
    validate_hdf5_file_set,
    verify_corresponding_file_identity,
)


CONFIG_SCHEMA_VERSION = "exd_hox_selex_import_config.v1"
SOURCE_MANIFEST_FILENAME = "exd_hox_source_manifest_v1.json"
HDF5_INVENTORY_FILENAME = "exd_hox_hdf5_inventory_v1.tsv"
COUNT_SUMMARY_FILENAME = "exd_hox_per_tf_count_summary_v1.tsv"
AFFINITY_SUMMARY_FILENAME = "exd_hox_affinity_summary_v1.tsv"
AFFINITY_HISTOGRAM_FILENAME = "exd_hox_affinity_histogram_v1.tsv"
LEAKAGE_SUMMARY_FILENAME = "exd_hox_supplied_split_leakage_summary_v1.tsv"
LEAKAGE_DETAIL_FILENAME = "exd_hox_supplied_split_exact_overlaps_v1.tsv"
WITHIN_TF_FILENAME = "exd_hox_within_tf_duplicate_summary_v1.tsv"
CROSS_TF_FILENAME = "exd_hox_cross_tf_sharing_summary_v1.tsv"
CROSS_TF_TOTALS_FILENAME = "exd_hox_cross_tf_sharing_totals_v1.json"
AUDIT_MANIFEST_FILENAME = "exd_hox_audit_manifest_v1.json"

INVENTORY_FIELDS = (
    "transcription_factor",
    "supplied_split",
    "raw_path",
    "row_count",
    "byte_size",
    "sha256",
    "sequence_shape",
    "sequence_dtype",
    "sequence_string_encoding",
    "one_hot_shape",
    "one_hot_dtype",
    "one_hot_channel_order",
    "target_shape",
    "target_dtype",
    "target_id_shape",
    "target_id_dtype",
    "target_id_value",
    "target_name_shape",
    "target_name_dtype",
    "target_name_value",
)
COUNT_FIELDS = (
    "transcription_factor",
    "supplied_training_rows",
    "supplied_test_rows",
    "total_row_occurrences",
    "unique_exact_sequences",
    "unique_reverse_complement_groups",
)
AFFINITY_FIELDS = (
    "transcription_factor",
    "supplied_split",
    "row_count",
    "mean",
    "population_standard_deviation",
    "minimum",
    "q01",
    "q05",
    "q25",
    "median",
    "q75",
    "q95",
    "q99",
    "maximum",
)
AFFINITY_HISTOGRAM_FIELDS = (
    "transcription_factor",
    "supplied_split",
    "bin_index",
    "bin_left",
    "bin_right",
    "row_count",
)
LEAKAGE_SUMMARY_FIELDS = (
    "transcription_factor",
    "exact_sequence_overlap_group_count",
    "exact_labeled_row_overlap_count",
    "reverse_complement_equivalent_overlap_group_count",
    "reverse_complement_only_overlap_group_count",
    "exact_conflicting_label_group_count",
    "reverse_complement_conflicting_label_group_count",
)
LEAKAGE_DETAIL_FIELDS = (
    "transcription_factor",
    "sequence",
    "target",
    "training_row_index_zero_based",
    "test_row_index_zero_based",
    "overlap_ordinal_within_labeled_row",
    "sequence_sha256",
    "reverse_complement_canonical_sequence",
    "reverse_complement_canonical_sha256",
)
WITHIN_TF_FIELDS = (
    "transcription_factor",
    "row_occurrences",
    "unique_exact_sequences",
    "exact_duplicate_sequence_group_count",
    "exact_duplicate_sequence_extra_occurrence_count",
    "unique_exact_labeled_rows",
    "exact_labeled_row_duplicate_group_count",
    "exact_labeled_row_duplicate_extra_occurrence_count",
    "unique_reverse_complement_groups",
    "reverse_complement_duplicate_group_count",
    "reverse_complement_duplicate_extra_occurrence_count",
    "reverse_complement_only_group_count",
    "self_reverse_complement_unique_sequence_count",
    "self_reverse_complement_row_occurrence_count",
    "exact_conflicting_label_group_count",
    "reverse_complement_conflicting_label_group_count",
)
CROSS_TF_FIELDS = (
    "group_type",
    "group_sequence",
    "group_sequence_sha256",
    "transcription_factor_count",
    "transcription_factors",
    "row_occurrences_by_transcription_factor",
    "total_row_occurrences",
)


def parse_arguments(argv=None):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Exclusively import and audit the pinned Exd-Hox SELEX source."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/exd_hox_selex_import_v1.yaml",
    )
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--source-checkout",
        help=(
            "Optional existing checkout at the exact pinned commit. The "
            "default is a shallow temporary sparse clone."
        ),
    )
    return parser.parse_args(argv)


def load_import_config(path: Path | str) -> Dict[str, Any]:
    """Load and validate the versioned import configuration."""

    with open(path, "r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    if not isinstance(payload, Mapping):
        raise ValueError("Import config must contain a mapping.")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported Exd-Hox import config schema.")
    required_sections = ("source", "dataset", "outputs", "audit", "expected")
    for section in required_sections:
        if not isinstance(payload.get(section), Mapping):
            raise ValueError("Config section '{0}' must be a mapping.".format(section))

    source = payload["source"]
    for key in (
        "repository_url",
        "commit",
        "canonical_directory",
        "comparison_directory",
    ):
        if not source.get(key):
            raise ValueError("Config source.{0} must not be empty.".format(key))
    commit = str(source["commit"])
    if len(commit) != 40:
        raise ValueError("Pinned source commit must be a full 40-character hash.")
    int(commit, 16)
    validate_repository_relative_path(str(source["canonical_directory"]))
    validate_repository_relative_path(str(source["comparison_directory"]))

    dataset = payload["dataset"]
    transcription_factors = tuple(dataset.get("transcription_factors", ()))
    if not transcription_factors:
        raise ValueError("At least one transcription factor is required.")
    if len(set(transcription_factors)) != len(transcription_factors):
        raise ValueError("Transcription-factor names must be unique.")
    if tuple(dataset.get("one_hot_channel_order", ())) != (
        "A",
        "C",
        "G",
        "T",
    ):
        raise ValueError("One-hot channel order must be A, C, G, T.")
    if int(dataset.get("sequence_length", 0)) != 14:
        raise ValueError("The active Exd-Hox dataset must use 14-mer sequences.")

    outputs = payload["outputs"]
    for key in ("raw_directory", "audit_directory", "plot_directory"):
        validate_repository_relative_path(str(outputs[key]))
    if int(payload["audit"].get("affinity_histogram_bin_count", 0)) <= 0:
        raise ValueError("Affinity histogram bin count must be positive.")

    expected_overlap_keys = set(
        payload["expected"]["exact_labeled_row_train_test_overlaps"]
    )
    if expected_overlap_keys != set(transcription_factors):
        raise ValueError(
            "Expected supplied-split overlap keys must match all TF names."
        )
    return dict(payload)


def run_import_audit(
    config_path: Path | str,
    repository_root: Path | str,
    source_checkout: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Perform one exclusive import and deterministic audit."""

    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_import_config(config_file)
    config_logical_path = repository_relative_path(config_file, root)
    outputs = config["outputs"]
    raw_directory = _resolve_output_path(root, outputs["raw_directory"])
    audit_directory = _resolve_output_path(root, outputs["audit_directory"])
    _preflight_output_directories(raw_directory, audit_directory)

    checkout_temporary_directory = None
    if source_checkout is None:
        checkout_temporary_directory = tempfile.TemporaryDirectory(
            prefix="exd_hox_source_"
        )
        checkout = _create_shallow_sparse_checkout(
            config["source"],
            Path(checkout_temporary_directory.name) / "source",
        )
    else:
        checkout = Path(source_checkout).resolve()
        _verify_source_commit(checkout, str(config["source"]["commit"]))

    try:
        result = _import_and_audit_from_checkout(
            config=config,
            config_path=config_file,
            config_logical_path=config_logical_path,
            repository_root=root,
            checkout=checkout,
            raw_directory=raw_directory,
            audit_directory=audit_directory,
        )
    finally:
        if checkout_temporary_directory is not None:
            checkout_temporary_directory.cleanup()
    return result


def _import_and_audit_from_checkout(
    config: Mapping[str, Any],
    config_path: Path,
    config_logical_path: str,
    repository_root: Path,
    checkout: Path,
    raw_directory: Path,
    audit_directory: Path,
) -> Dict[str, Any]:
    source = config["source"]
    dataset = config["dataset"]
    transcription_factors = tuple(dataset["transcription_factors"])
    canonical_directory = checkout / str(source["canonical_directory"])
    comparison_directory = checkout / str(source["comparison_directory"])
    relative_paths = validate_hdf5_file_set(
        canonical_directory,
        transcription_factors,
    )
    validate_hdf5_file_set(comparison_directory, transcription_factors)
    identity_rows = verify_corresponding_file_identity(
        canonical_directory,
        comparison_directory,
        relative_paths,
    )

    raw_directory.parent.mkdir(parents=True, exist_ok=True)
    audit_directory.parent.mkdir(parents=True, exist_ok=True)
    raw_staging_context = tempfile.TemporaryDirectory(
        prefix=".exd_hox_raw_staging_",
        dir=raw_directory.parent,
    )
    audit_staging_context = tempfile.TemporaryDirectory(
        prefix=".exd_hox_audit_staging_",
        dir=audit_directory.parent,
    )
    raw_staging = Path(raw_staging_context.name) / "raw"
    audit_staging = Path(audit_staging_context.name) / "audit"
    raw_staging.mkdir()
    audit_staging.mkdir()
    try:
        source_manifest_files = _copy_canonical_files(
            config,
            canonical_directory,
            identity_rows,
            raw_staging,
        )
        records_by_tf, inventory_rows = _read_imported_files(
            config,
            raw_staging,
        )
        table_payload = _build_audit_payloads(config, records_by_tf)
        _validate_expected_results(config, table_payload)
        source_manifest = _write_audit_artifacts(
            config=config,
            config_path=config_path,
            config_logical_path=config_logical_path,
            source_manifest_files=source_manifest_files,
            inventory_rows=inventory_rows,
            table_payload=table_payload,
            audit_staging=audit_staging,
            audit_logical_directory=str(config["outputs"]["audit_directory"]),
        )
        audit_manifest = _write_audit_manifest(
            config=config,
            config_path=config_path,
            config_logical_path=config_logical_path,
            source_manifest=source_manifest,
            table_payload=table_payload,
            audit_staging=audit_staging,
            audit_logical_directory=str(config["outputs"]["audit_directory"]),
        )
        os.rename(raw_staging, raw_directory)
        os.rename(audit_staging, audit_directory)
    finally:
        audit_staging_context.cleanup()
        raw_staging_context.cleanup()
    return audit_manifest


def _copy_canonical_files(
    config: Mapping[str, Any],
    canonical_directory: Path,
    identity_rows: Sequence[Mapping[str, Any]],
    raw_staging: Path,
) -> Tuple[Dict[str, Any], ...]:
    source = config["source"]
    raw_logical_directory = str(config["outputs"]["raw_directory"])
    manifest_files = []
    for identity in identity_rows:
        relative_path = str(identity["relative_path"])
        source_path = canonical_directory / relative_path
        destination_path = raw_staging / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)
        copied_hash = hash_file_bytes(destination_path)
        if copied_hash != identity["canonical_sha256"]:
            raise ValueError("Copied HDF5 bytes do not match the source hash.")
        path_parts = Path(relative_path).parts
        transcription_factor = path_parts[0]
        supplied_split = Path(path_parts[1]).stem.rsplit("_", 1)[1]
        manifest_files.append(
            {
                "transcription_factor": transcription_factor,
                "supplied_split": supplied_split,
                "canonical_source_path": Path(
                    str(source["canonical_directory"]),
                    relative_path,
                ).as_posix(),
                "comparison_source_path": Path(
                    str(source["comparison_directory"]),
                    relative_path,
                ).as_posix(),
                "imported_raw_path": Path(
                    raw_logical_directory,
                    relative_path,
                ).as_posix(),
                "byte_size": int(identity["byte_size"]),
                "sha256": str(identity["canonical_sha256"]),
                "comparison_sha256": str(identity["comparison_sha256"]),
                "comparison_byte_identical": bool(identity["byte_identical"]),
            }
        )
    manifest_files.sort(
        key=lambda row: (row["transcription_factor"], row["supplied_split"])
    )
    return tuple(manifest_files)


def _read_imported_files(
    config: Mapping[str, Any],
    raw_staging: Path,
) -> Tuple[Dict[str, Dict[str, SelexHdf5File]], Tuple[Dict[str, Any], ...]]:
    transcription_factors = tuple(config["dataset"]["transcription_factors"])
    sequence_length = int(config["dataset"]["sequence_length"])
    raw_logical_directory = str(config["outputs"]["raw_directory"])
    records_by_tf = {}
    inventory_rows = []
    for transcription_factor in transcription_factors:
        split_records = {}
        for supplied_split in ("train", "test"):
            relative_path = Path(
                transcription_factor,
                "{0}_{1}.h5".format(transcription_factor, supplied_split),
            )
            physical_path = raw_staging / relative_path
            logical_path = Path(
                raw_logical_directory,
                relative_path,
            ).as_posix()
            record = read_validate_selex_hdf5(
                physical_path,
                transcription_factor=transcription_factor,
                supplied_split=supplied_split,
                logical_path=logical_path,
                sequence_length=sequence_length,
            )
            split_records[supplied_split] = record
            inventory_rows.append(dict(record.inventory))
        records_by_tf[transcription_factor] = split_records
    inventory_rows.sort(
        key=lambda row: (row["transcription_factor"], row["supplied_split"])
    )
    return records_by_tf, tuple(inventory_rows)


def _build_audit_payloads(
    config: Mapping[str, Any],
    records_by_tf: Mapping[str, Mapping[str, SelexHdf5File]],
) -> Dict[str, Any]:
    count_rows = build_count_summary_rows(records_by_tf)
    affinity_rows = build_affinity_summary_rows(records_by_tf)
    affinity_histogram_rows = build_affinity_histogram_rows(
        records_by_tf,
        int(config["audit"]["affinity_histogram_bin_count"]),
    )
    leakage_rows = []
    leakage_detail_rows = []
    within_tf_rows = []
    for transcription_factor in sorted(records_by_tf):
        split_records = records_by_tf[transcription_factor]
        leakage, details = audit_supplied_split(
            split_records["train"],
            split_records["test"],
        )
        leakage_rows.append(leakage)
        leakage_detail_rows.extend(details)
        combined_sequences = (
            split_records["train"].sequences
            + split_records["test"].sequences
        )
        combined_targets = (
            split_records["train"].targets
            + split_records["test"].targets
        )
        within_tf_rows.append(
            audit_within_tf(
                transcription_factor,
                combined_sequences,
                combined_targets,
            )
        )
    leakage_detail_rows.sort(
        key=lambda row: (
            row["transcription_factor"],
            row["sequence"],
            row["target"],
            row["training_row_index_zero_based"],
            row["test_row_index_zero_based"],
        )
    )
    cross_tf_rows, cross_tf_summary = build_cross_tf_sharing_rows(records_by_tf)
    return {
        "count_rows": tuple(count_rows),
        "affinity_rows": tuple(affinity_rows),
        "affinity_histogram_rows": tuple(affinity_histogram_rows),
        "leakage_rows": tuple(leakage_rows),
        "leakage_detail_rows": tuple(leakage_detail_rows),
        "within_tf_rows": tuple(within_tf_rows),
        "cross_tf_rows": tuple(cross_tf_rows),
        "cross_tf_summary": cross_tf_summary,
    }


def _validate_expected_results(
    config: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    expected = config["expected"]
    count_rows = payload["count_rows"]
    training_rows = sum(int(row["supplied_training_rows"]) for row in count_rows)
    test_rows = sum(int(row["supplied_test_rows"]) for row in count_rows)
    total_rows = sum(int(row["total_row_occurrences"]) for row in count_rows)
    _require_expected(training_rows, expected["training_rows"], "training rows")
    _require_expected(test_rows, expected["test_rows"], "test rows")
    _require_expected(
        total_rows,
        expected["total_row_occurrences"],
        "total row occurrences",
    )

    expected_overlaps = expected["exact_labeled_row_train_test_overlaps"]
    for row in payload["leakage_rows"]:
        transcription_factor = row["transcription_factor"]
        _require_expected(
            row["exact_labeled_row_overlap_count"],
            expected_overlaps[transcription_factor],
            "{0} exact labeled-row overlaps".format(transcription_factor),
        )
    _require_expected(
        len(payload["leakage_detail_rows"]),
        expected["total_exact_labeled_row_train_test_overlaps"],
        "total exact labeled-row overlaps",
    )
    _require_expected(
        sum(
            int(row["reverse_complement_only_overlap_group_count"])
            for row in payload["leakage_rows"]
        ),
        expected[
            "total_reverse_complement_only_train_test_overlap_groups"
        ],
        "total RC-only supplied-split overlap groups",
    )
    _require_expected(
        sum(
            int(row["exact_conflicting_label_group_count"])
            for row in payload["leakage_rows"]
        ),
        expected["total_exact_train_test_conflicting_label_groups"],
        "total exact supplied-split conflicting-label groups",
    )
    _require_expected(
        sum(
            int(row["reverse_complement_conflicting_label_group_count"])
            for row in payload["leakage_rows"]
        ),
        expected[
            "total_reverse_complement_train_test_conflicting_label_groups"
        ],
        "total RC supplied-split conflicting-label groups",
    )
def _write_audit_artifacts(
    config: Mapping[str, Any],
    config_path: Path,
    config_logical_path: str,
    source_manifest_files: Sequence[Mapping[str, Any]],
    inventory_rows: Sequence[Mapping[str, Any]],
    table_payload: Mapping[str, Any],
    audit_staging: Path,
    audit_logical_directory: str,
) -> Dict[str, Any]:
    source = config["source"]
    source_manifest = build_hashed_manifest(
        "exd_hox_source_manifest.v1",
        {
            "dataset_identifier": config["dataset"]["identifier"],
            "source_repository_url": source["repository_url"],
            "source_commit": source["commit"],
            "canonical_source_directory": source["canonical_directory"],
            "comparison_source_directory": source["comparison_directory"],
            "comparison_role": "legacy_architecture_routing_duplicate_only",
            "imported_raw_directory": config["outputs"]["raw_directory"],
            "config_path": config_logical_path,
            "config_sha256": hash_file_bytes(config_path),
            "files": list(source_manifest_files),
        },
    )
    write_json_exclusive(
        audit_staging / SOURCE_MANIFEST_FILENAME,
        source_manifest,
    )
    write_tsv_exclusive(
        audit_staging / HDF5_INVENTORY_FILENAME,
        INVENTORY_FIELDS,
        inventory_rows,
    )
    write_tsv_exclusive(
        audit_staging / COUNT_SUMMARY_FILENAME,
        COUNT_FIELDS,
        table_payload["count_rows"],
    )
    write_tsv_exclusive(
        audit_staging / AFFINITY_SUMMARY_FILENAME,
        AFFINITY_FIELDS,
        table_payload["affinity_rows"],
    )
    write_tsv_exclusive(
        audit_staging / AFFINITY_HISTOGRAM_FILENAME,
        AFFINITY_HISTOGRAM_FIELDS,
        table_payload["affinity_histogram_rows"],
    )
    write_tsv_exclusive(
        audit_staging / LEAKAGE_SUMMARY_FILENAME,
        LEAKAGE_SUMMARY_FIELDS,
        table_payload["leakage_rows"],
    )
    write_tsv_exclusive(
        audit_staging / LEAKAGE_DETAIL_FILENAME,
        LEAKAGE_DETAIL_FIELDS,
        table_payload["leakage_detail_rows"],
    )
    write_tsv_exclusive(
        audit_staging / WITHIN_TF_FILENAME,
        WITHIN_TF_FIELDS,
        table_payload["within_tf_rows"],
    )
    write_tsv_exclusive(
        audit_staging / CROSS_TF_FILENAME,
        CROSS_TF_FIELDS,
        table_payload["cross_tf_rows"],
    )
    cross_tf_totals = build_hashed_manifest(
        "exd_hox_cross_tf_sharing_totals.v1",
        {
            "source_table_path": Path(
                audit_logical_directory,
                CROSS_TF_FILENAME,
            ).as_posix(),
            "group_types": table_payload["cross_tf_summary"],
        },
    )
    write_json_exclusive(
        audit_staging / CROSS_TF_TOTALS_FILENAME,
        cross_tf_totals,
    )
    return source_manifest


def _write_audit_manifest(
    config: Mapping[str, Any],
    config_path: Path,
    config_logical_path: str,
    source_manifest: Mapping[str, Any],
    table_payload: Mapping[str, Any],
    audit_staging: Path,
    audit_logical_directory: str,
) -> Dict[str, Any]:
    artifact_filenames = (
        SOURCE_MANIFEST_FILENAME,
        HDF5_INVENTORY_FILENAME,
        COUNT_SUMMARY_FILENAME,
        AFFINITY_SUMMARY_FILENAME,
        AFFINITY_HISTOGRAM_FILENAME,
        LEAKAGE_SUMMARY_FILENAME,
        LEAKAGE_DETAIL_FILENAME,
        WITHIN_TF_FILENAME,
        CROSS_TF_FILENAME,
        CROSS_TF_TOTALS_FILENAME,
    )
    artifacts = []
    for filename in artifact_filenames:
        fingerprint = fingerprint_file(
            audit_staging / filename,
            Path(audit_logical_directory, filename).as_posix(),
        )
        artifacts.append(fingerprint.to_dict())
    artifacts.sort(key=lambda artifact: artifact["path"])

    totals = {
        "training_rows": sum(
            int(row["supplied_training_rows"])
            for row in table_payload["count_rows"]
        ),
        "test_rows": sum(
            int(row["supplied_test_rows"])
            for row in table_payload["count_rows"]
        ),
        "total_row_occurrences": sum(
            int(row["total_row_occurrences"])
            for row in table_payload["count_rows"]
        ),
        "exact_labeled_row_train_test_overlaps": len(
            table_payload["leakage_detail_rows"]
        ),
        "cross_tf_sharing": table_payload["cross_tf_summary"],
    }
    manifest = build_hashed_manifest(
        "exd_hox_audit_manifest.v1",
        {
            "dataset_identifier": config["dataset"]["identifier"],
            "config_path": config_logical_path,
            "config_sha256": hash_file_bytes(config_path),
            "source_manifest_path": Path(
                audit_logical_directory,
                SOURCE_MANIFEST_FILENAME,
            ).as_posix(),
            "source_manifest_hash": source_manifest["manifest_hash"],
            "audit_directory": audit_logical_directory,
            "artifacts": artifacts,
            "totals": totals,
        },
    )
    write_json_exclusive(audit_staging / AUDIT_MANIFEST_FILENAME, manifest)
    return manifest


def _create_shallow_sparse_checkout(
    source: Mapping[str, Any],
    destination: Path,
) -> Path:
    _run_git(
        (
            "git",
            "clone",
            "--no-checkout",
            "--filter=blob:none",
            "--depth",
            "1",
            str(source["repository_url"]),
            str(destination),
        )
    )
    _run_git(
        (
            "git",
            "-C",
            str(destination),
            "fetch",
            "--depth",
            "1",
            "origin",
            str(source["commit"]),
        )
    )
    _run_git(
        (
            "git",
            "-C",
            str(destination),
            "sparse-checkout",
            "set",
            str(source["canonical_directory"]),
            str(source["comparison_directory"]),
        )
    )
    _run_git(
        (
            "git",
            "-C",
            str(destination),
            "checkout",
            "--detach",
            str(source["commit"]),
        )
    )
    _verify_source_commit(destination, str(source["commit"]))
    return destination


def _verify_source_commit(checkout: Path, expected_commit: str) -> None:
    result = _run_git(("git", "-C", str(checkout), "rev-parse", "HEAD"))
    observed_commit = result.stdout.strip()
    if observed_commit != expected_commit:
        message = "External source commit mismatch: expected {0}, observed {1}."
        raise ValueError(message.format(expected_commit, observed_commit))


def _run_git(arguments: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        tuple(arguments),
        check=True,
        capture_output=True,
        text=True,
    )


def _resolve_output_path(repository_root: Path, relative_path: str) -> Path:
    normalized = validate_repository_relative_path(str(relative_path))
    resolved = (repository_root / normalized).resolve()
    repository_relative_path(resolved, repository_root)
    return resolved


def _preflight_output_directories(
    raw_directory: Path,
    audit_directory: Path,
) -> None:
    for output_directory in (raw_directory, audit_directory):
        if output_directory.exists() or output_directory.is_symlink():
            message = "Refusing to overwrite existing output directory: {0}"
            raise FileExistsError(message.format(output_directory))


def _require_expected(actual: Any, expected: Any, label: str) -> None:
    if int(actual) != int(expected):
        message = "Expected {0} {1}; observed {2}."
        raise ValueError(message.format(expected, label, actual))


def main(argv=None):
    """Run the exclusive import and print its deterministic manifest."""

    arguments = parse_arguments(argv)
    source_checkout = arguments.source_checkout
    manifest = run_import_audit(
        config_path=arguments.config,
        repository_root=arguments.repository_root,
        source_checkout=source_checkout,
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
