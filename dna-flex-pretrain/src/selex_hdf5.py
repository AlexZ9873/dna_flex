"""Strict Wang et al. Exd-Hox SELEX HDF5 parsing and source audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Dict, Mapping, Sequence, Tuple

import h5py
import numpy as np

from src.coordinates import reverse_complement
from src.data_fingerprints import reverse_complement_canonical_sequence
from src.downstream_fingerprints import hash_file_bytes


EXPECTED_TOP_LEVEL_GROUPS = frozenset(("data", "targets"))
EXPECTED_DATA_DATASETS = frozenset(("sequence", "s_x", "c0_y"))
EXPECTED_TARGET_DATASETS = frozenset(("id", "name"))
EXPECTED_CHANNEL_ORDER = ("A", "C", "G", "T")
EXPECTED_TARGET_ID = "c0"
EXPECTED_TARGET_NAME = "dummy"
TARGET_DTYPE = np.dtype("float32")
ONE_HOT_DTYPE = np.dtype("int8")
TARGET_METADATA_DTYPE = np.dtype("S16")


class SelexHdf5ValidationError(ValueError):
    """Raised when a source HDF5 file violates the pinned schema."""


@dataclass(frozen=True)
class SelexHdf5File:
    """Validated logical rows and inventory for one supplied source file."""

    transcription_factor: str
    supplied_split: str
    logical_path: str
    sequences: Tuple[str, ...]
    targets: Tuple[float, ...]
    inventory: Mapping[str, Any]

    @property
    def row_count(self) -> int:
        """Return the number of row occurrences."""

        return len(self.sequences)


def expected_relative_hdf5_paths(
    transcription_factors: Sequence[str],
) -> Tuple[str, ...]:
    """Return the exact expected TF/split source file set."""

    paths = []
    for transcription_factor in transcription_factors:
        for supplied_split in ("train", "test"):
            filename = "{0}_{1}.h5".format(
                transcription_factor,
                supplied_split,
            )
            paths.append(
                Path(transcription_factor, filename).as_posix()
            )
    return tuple(sorted(paths))


def validate_hdf5_file_set(
    source_directory: Path | str,
    transcription_factors: Sequence[str],
) -> Tuple[str, ...]:
    """Fail unless a source folder has exactly the expected 16 HDF5 files."""

    directory = Path(source_directory)
    expected = expected_relative_hdf5_paths(transcription_factors)
    observed = []
    for path in directory.glob("*/*.h5"):
        if path.is_file():
            observed.append(path.relative_to(directory).as_posix())
    observed_tuple = tuple(sorted(observed))
    if observed_tuple != expected:
        message = (
            "Unexpected HDF5 source file set. Expected {0}; observed {1}."
        )
        raise SelexHdf5ValidationError(
            message.format(list(expected), list(observed_tuple))
        )
    return expected


def verify_corresponding_file_identity(
    canonical_directory: Path | str,
    comparison_directory: Path | str,
    relative_paths: Sequence[str],
) -> Tuple[Dict[str, Any], ...]:
    """Verify canonical and legacy RCmodel source bytes are identical."""

    canonical_root = Path(canonical_directory)
    comparison_root = Path(comparison_directory)
    rows = []
    for relative_path in sorted(relative_paths):
        canonical_path = canonical_root / relative_path
        comparison_path = comparison_root / relative_path
        canonical_hash = hash_file_bytes(canonical_path)
        comparison_hash = hash_file_bytes(comparison_path)
        canonical_size = canonical_path.stat().st_size
        comparison_size = comparison_path.stat().st_size
        identical = (
            canonical_hash == comparison_hash
            and canonical_size == comparison_size
        )
        if not identical:
            message = "SELEX_canonical and SELEX_RCmodel differ for {0}."
            raise SelexHdf5ValidationError(message.format(relative_path))
        rows.append(
            {
                "relative_path": relative_path,
                "canonical_sha256": canonical_hash,
                "comparison_sha256": comparison_hash,
                "byte_size": canonical_size,
                "byte_identical": True,
            }
        )
    return tuple(rows)


def read_validate_selex_hdf5(
    path: Path | str,
    transcription_factor: str,
    supplied_split: str,
    logical_path: str,
    sequence_length: int = 14,
) -> SelexHdf5File:
    """Read and strictly validate one canonical SELEX HDF5 source file."""

    if supplied_split not in ("train", "test"):
        raise ValueError("Supplied split must be 'train' or 'test'.")
    source_path = Path(path)
    with h5py.File(source_path, "r") as source_file:
        _validate_group_schema(source_file)
        sequence_dataset = _require_dataset(source_file, "data/sequence")
        one_hot_dataset = _require_dataset(source_file, "data/s_x")
        target_dataset = _require_dataset(source_file, "data/c0_y")
        target_id_dataset = _require_dataset(source_file, "targets/id")
        target_name_dataset = _require_dataset(source_file, "targets/name")

        _validate_dataset_dtypes(
            sequence_dataset,
            one_hot_dataset,
            target_dataset,
            target_id_dataset,
            target_name_dataset,
        )

        row_count = int(sequence_dataset.shape[0])
        if sequence_dataset.shape != (row_count,):
            raise SelexHdf5ValidationError(
                "data/sequence must have shape N."
            )
        expected_one_hot_shape = (row_count, sequence_length, 4)
        if one_hot_dataset.shape != expected_one_hot_shape:
            message = "data/s_x must have shape {0}; observed {1}."
            raise SelexHdf5ValidationError(
                message.format(expected_one_hot_shape, one_hot_dataset.shape)
            )
        if target_dataset.shape != (row_count, 1):
            message = "data/c0_y must have shape ({0}, 1)."
            raise SelexHdf5ValidationError(message.format(row_count))
        if target_id_dataset.shape != (1,):
            raise SelexHdf5ValidationError(
                "targets/id must have shape (1,)."
            )
        if target_name_dataset.shape != (1,):
            raise SelexHdf5ValidationError(
                "targets/name must have shape (1,)."
            )

        sequences = _decode_sequences(sequence_dataset)
        _validate_sequences(sequences, sequence_length)

        one_hot = np.asarray(one_hot_dataset[:])
        _validate_one_hot(one_hot, sequences)

        target_values = np.asarray(target_dataset[:, 0])
        _validate_targets(target_values)

        target_id = _decode_scalar_text(target_id_dataset[0], "targets/id")
        target_name = _decode_scalar_text(
            target_name_dataset[0],
            "targets/name",
        )
        if target_id != EXPECTED_TARGET_ID:
            message = "targets/id must be '{0}'; observed '{1}'."
            raise SelexHdf5ValidationError(
                message.format(EXPECTED_TARGET_ID, target_id)
            )
        if target_name != EXPECTED_TARGET_NAME:
            message = "targets/name must be '{0}'; observed '{1}'."
            raise SelexHdf5ValidationError(
                message.format(EXPECTED_TARGET_NAME, target_name)
            )

        string_metadata = h5py.check_string_dtype(sequence_dataset.dtype)
        inventory = {
            "transcription_factor": transcription_factor,
            "supplied_split": supplied_split,
            "raw_path": logical_path,
            "row_count": row_count,
            "byte_size": source_path.stat().st_size,
            "sha256": hash_file_bytes(source_path),
            "sequence_shape": _shape_text(sequence_dataset.shape),
            "sequence_dtype": str(sequence_dataset.dtype),
            "sequence_string_encoding": string_metadata.encoding,
            "one_hot_shape": _shape_text(one_hot_dataset.shape),
            "one_hot_dtype": str(one_hot_dataset.dtype),
            "one_hot_channel_order": ",".join(EXPECTED_CHANNEL_ORDER),
            "target_shape": _shape_text(target_dataset.shape),
            "target_dtype": str(target_dataset.dtype),
            "target_id_shape": _shape_text(target_id_dataset.shape),
            "target_id_dtype": str(target_id_dataset.dtype),
            "target_id_value": target_id,
            "target_name_shape": _shape_text(target_name_dataset.shape),
            "target_name_dtype": str(target_name_dataset.dtype),
            "target_name_value": target_name,
        }

    return SelexHdf5File(
        transcription_factor=transcription_factor,
        supplied_split=supplied_split,
        logical_path=logical_path,
        sequences=tuple(sequences),
        targets=tuple(float(value) for value in target_values),
        inventory=inventory,
    )


def build_count_summary_rows(
    records_by_tf: Mapping[str, Mapping[str, SelexHdf5File]],
) -> Tuple[Dict[str, Any], ...]:
    """Build deterministic per-TF supplied row counts."""

    rows = []
    for transcription_factor in sorted(records_by_tf):
        split_records = records_by_tf[transcription_factor]
        training_rows = split_records["train"].row_count
        test_rows = split_records["test"].row_count
        combined_sequences = (
            split_records["train"].sequences
            + split_records["test"].sequences
        )
        canonical_groups = set()
        for sequence in combined_sequences:
            canonical_groups.add(
                reverse_complement_canonical_sequence(sequence)
            )
        rows.append(
            {
                "transcription_factor": transcription_factor,
                "supplied_training_rows": training_rows,
                "supplied_test_rows": test_rows,
                "total_row_occurrences": training_rows + test_rows,
                "unique_exact_sequences": len(set(combined_sequences)),
                "unique_reverse_complement_groups": len(canonical_groups),
            }
        )
    return tuple(rows)


def build_affinity_summary_rows(
    records_by_tf: Mapping[str, Mapping[str, SelexHdf5File]],
) -> Tuple[Dict[str, Any], ...]:
    """Build per-TF, per-split finite affinity summaries."""

    rows = []
    quantiles = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    quantile_names = (
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
    for transcription_factor in sorted(records_by_tf):
        split_records = records_by_tf[transcription_factor]
        target_groups = {
            "train": split_records["train"].targets,
            "test": split_records["test"].targets,
            "combined": (
                split_records["train"].targets
                + split_records["test"].targets
            ),
        }
        for supplied_split in ("train", "test", "combined"):
            values = np.asarray(
                target_groups[supplied_split],
                dtype=np.float32,
            )
            calculated_quantiles = np.quantile(values, quantiles)
            row = {
                "transcription_factor": transcription_factor,
                "supplied_split": supplied_split,
                "row_count": int(values.size),
                "mean": _float_text(float(np.mean(values, dtype=np.float64))),
                "population_standard_deviation": _float_text(
                    float(np.std(values, dtype=np.float64, ddof=0))
                ),
            }
            for name, value in zip(quantile_names, calculated_quantiles):
                row[name] = _float_text(float(value))
            rows.append(row)
    return tuple(rows)


def build_affinity_histogram_rows(
    records_by_tf: Mapping[str, Mapping[str, SelexHdf5File]],
    bin_count: int,
) -> Tuple[Dict[str, Any], ...]:
    """Build fixed [0,1] histogram tables for plotting without HDF5 access."""

    if bin_count <= 0:
        raise ValueError("Affinity histogram bin count must be positive.")
    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
    rows = []
    for transcription_factor in sorted(records_by_tf):
        for supplied_split in ("train", "test"):
            values = np.asarray(
                records_by_tf[transcription_factor][supplied_split].targets,
                dtype=np.float32,
            )
            counts, observed_edges = np.histogram(values, bins=bin_edges)
            for bin_index, count in enumerate(counts):
                rows.append(
                    {
                        "transcription_factor": transcription_factor,
                        "supplied_split": supplied_split,
                        "bin_index": bin_index,
                        "bin_left": _float_text(observed_edges[bin_index]),
                        "bin_right": _float_text(
                            observed_edges[bin_index + 1]
                        ),
                        "row_count": int(count),
                    }
                )
    return tuple(rows)


def audit_supplied_split(
    training_record: SelexHdf5File,
    test_record: SelexHdf5File,
) -> Tuple[Dict[str, Any], Tuple[Dict[str, Any], ...]]:
    """Audit exact and RC-equivalent leakage in one TF's supplied split."""

    if training_record.transcription_factor != test_record.transcription_factor:
        raise ValueError("Train and test records must belong to the same TF.")
    transcription_factor = training_record.transcription_factor
    training = _row_group_maps(training_record.sequences, training_record.targets)
    test = _row_group_maps(test_record.sequences, test_record.targets)

    exact_sequences = sorted(
        set(training["exact_labels"]) & set(test["exact_labels"])
    )
    canonical_groups = sorted(
        set(training["canonical_labels"])
        & set(test["canonical_labels"])
    )

    exact_conflicts = 0
    for sequence in exact_sequences:
        labels = set(training["exact_labels"][sequence])
        labels.update(test["exact_labels"][sequence])
        if len(labels) > 1:
            exact_conflicts += 1

    reverse_complement_conflicts = 0
    reverse_complement_only_groups = 0
    for canonical_sequence in canonical_groups:
        labels = set(training["canonical_labels"][canonical_sequence])
        labels.update(test["canonical_labels"][canonical_sequence])
        if len(labels) > 1:
            reverse_complement_conflicts += 1
        training_sequences = training["canonical_sequences"][canonical_sequence]
        test_sequences = test["canonical_sequences"][canonical_sequence]
        if training_sequences.isdisjoint(test_sequences):
            reverse_complement_only_groups += 1

    detail_rows = []
    labeled_keys = sorted(
        set(training["labeled_indices"]) & set(test["labeled_indices"])
    )
    for sequence, target_key in labeled_keys:
        training_indices = training["labeled_indices"][(sequence, target_key)]
        test_indices = test["labeled_indices"][(sequence, target_key)]
        matched_count = min(len(training_indices), len(test_indices))
        canonical_sequence = reverse_complement_canonical_sequence(sequence)
        for match_index in range(matched_count):
            target_value = training["target_values"][target_key]
            detail_rows.append(
                {
                    "transcription_factor": transcription_factor,
                    "sequence": sequence,
                    "target": _float_text(target_value),
                    "training_row_index_zero_based": training_indices[match_index],
                    "test_row_index_zero_based": test_indices[match_index],
                    "overlap_ordinal_within_labeled_row": match_index + 1,
                    "sequence_sha256": _sequence_hash(sequence),
                    "reverse_complement_canonical_sequence": canonical_sequence,
                    "reverse_complement_canonical_sha256": _sequence_hash(
                        canonical_sequence
                    ),
                }
            )

    summary = {
        "transcription_factor": transcription_factor,
        "exact_sequence_overlap_group_count": len(exact_sequences),
        "exact_labeled_row_overlap_count": len(detail_rows),
        "reverse_complement_equivalent_overlap_group_count": len(
            canonical_groups
        ),
        "reverse_complement_only_overlap_group_count": (
            reverse_complement_only_groups
        ),
        "exact_conflicting_label_group_count": exact_conflicts,
        "reverse_complement_conflicting_label_group_count": (
            reverse_complement_conflicts
        ),
    }
    return summary, tuple(detail_rows)


def audit_within_tf(
    transcription_factor: str,
    sequences: Sequence[str],
    targets: Sequence[float],
) -> Dict[str, Any]:
    """Audit duplicates and label conflicts within one TF assay."""

    if len(sequences) != len(targets):
        raise ValueError("Sequence and target counts must match.")
    exact_counts = Counter(sequences)
    labeled_counts = Counter()
    exact_labels = defaultdict(set)
    canonical_counts = Counter()
    canonical_labels = defaultdict(set)
    canonical_sequences = defaultdict(set)
    self_reverse_complement_sequences = set()
    self_reverse_complement_occurrences = 0

    for sequence, target in zip(sequences, targets):
        target_key = _target_key(target)
        labeled_counts[(sequence, target_key)] += 1
        exact_labels[sequence].add(target_key)
        canonical_sequence = reverse_complement_canonical_sequence(sequence)
        canonical_counts[canonical_sequence] += 1
        canonical_labels[canonical_sequence].add(target_key)
        canonical_sequences[canonical_sequence].add(sequence)
        if sequence == reverse_complement(sequence):
            self_reverse_complement_sequences.add(sequence)
            self_reverse_complement_occurrences += 1

    return {
        "transcription_factor": transcription_factor,
        "row_occurrences": len(sequences),
        "unique_exact_sequences": len(exact_counts),
        "exact_duplicate_sequence_group_count": _duplicate_group_count(
            exact_counts
        ),
        "exact_duplicate_sequence_extra_occurrence_count": (
            _duplicate_extra_occurrence_count(exact_counts)
        ),
        "unique_exact_labeled_rows": len(labeled_counts),
        "exact_labeled_row_duplicate_group_count": _duplicate_group_count(
            labeled_counts
        ),
        "exact_labeled_row_duplicate_extra_occurrence_count": (
            _duplicate_extra_occurrence_count(labeled_counts)
        ),
        "unique_reverse_complement_groups": len(canonical_counts),
        "reverse_complement_duplicate_group_count": _duplicate_group_count(
            canonical_counts
        ),
        "reverse_complement_duplicate_extra_occurrence_count": (
            _duplicate_extra_occurrence_count(canonical_counts)
        ),
        "reverse_complement_only_group_count": sum(
            len(group_sequences) > 1
            for group_sequences in canonical_sequences.values()
        ),
        "self_reverse_complement_unique_sequence_count": len(
            self_reverse_complement_sequences
        ),
        "self_reverse_complement_row_occurrence_count": (
            self_reverse_complement_occurrences
        ),
        "exact_conflicting_label_group_count": sum(
            len(labels) > 1 for labels in exact_labels.values()
        ),
        "reverse_complement_conflicting_label_group_count": sum(
            len(labels) > 1 for labels in canonical_labels.values()
        ),
    }


def build_cross_tf_sharing_rows(
    records_by_tf: Mapping[str, Mapping[str, SelexHdf5File]],
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Any]]:
    """Audit exact and RC-equivalent sequence sharing across TF assays."""

    exact_counts = defaultdict(Counter)
    canonical_counts = defaultdict(Counter)
    for transcription_factor in sorted(records_by_tf):
        split_records = records_by_tf[transcription_factor]
        for supplied_split in ("train", "test"):
            for sequence in split_records[supplied_split].sequences:
                exact_counts[sequence][transcription_factor] += 1
                canonical_sequence = reverse_complement_canonical_sequence(
                    sequence
                )
                canonical_counts[canonical_sequence][transcription_factor] += 1

    rows = []
    rows.extend(_shared_group_rows("exact_sequence", exact_counts))
    rows.extend(
        _shared_group_rows("reverse_complement_equivalent", canonical_counts)
    )
    rows.sort(key=lambda row: (row["group_type"], row["group_sequence"]))

    summary = {}
    for group_type in ("exact_sequence", "reverse_complement_equivalent"):
        selected = [row for row in rows if row["group_type"] == group_type]
        tf_count_distribution = Counter(
            int(row["transcription_factor_count"]) for row in selected
        )
        summary[group_type] = {
            "shared_group_count": len(selected),
            "maximum_transcription_factor_count": max(
                (int(row["transcription_factor_count"]) for row in selected),
                default=0,
            ),
            "group_count_by_transcription_factor_count": {
                str(key): tf_count_distribution[key]
                for key in sorted(tf_count_distribution)
            },
            "row_occurrences_in_shared_groups": sum(
                int(row["total_row_occurrences"]) for row in selected
            ),
        }
    return tuple(rows), summary


def _validate_group_schema(source_file: h5py.File) -> None:
    observed_top_level = frozenset(source_file.keys())
    if observed_top_level != EXPECTED_TOP_LEVEL_GROUPS:
        message = "Expected top-level groups {0}; observed {1}."
        raise SelexHdf5ValidationError(
            message.format(
                sorted(EXPECTED_TOP_LEVEL_GROUPS),
                sorted(observed_top_level),
            )
        )
    data_group = source_file["data"]
    targets_group = source_file["targets"]
    if not isinstance(data_group, h5py.Group):
        raise SelexHdf5ValidationError("data must be an HDF5 group.")
    if not isinstance(targets_group, h5py.Group):
        raise SelexHdf5ValidationError("targets must be an HDF5 group.")
    if frozenset(data_group.keys()) != EXPECTED_DATA_DATASETS:
        raise SelexHdf5ValidationError(
            "data must contain exactly sequence, s_x, and c0_y."
        )
    if frozenset(targets_group.keys()) != EXPECTED_TARGET_DATASETS:
        raise SelexHdf5ValidationError(
            "targets must contain exactly id and name."
        )


def _require_dataset(source_file: h5py.File, path: str) -> h5py.Dataset:
    value = source_file[path]
    if not isinstance(value, h5py.Dataset):
        raise SelexHdf5ValidationError(
            "{0} must be an HDF5 dataset.".format(path)
        )
    return value


def _validate_dataset_dtypes(
    sequence_dataset: h5py.Dataset,
    one_hot_dataset: h5py.Dataset,
    target_dataset: h5py.Dataset,
    target_id_dataset: h5py.Dataset,
    target_name_dataset: h5py.Dataset,
) -> None:
    sequence_string = h5py.check_string_dtype(sequence_dataset.dtype)
    if (
        sequence_dataset.dtype.kind != "O"
        or sequence_string is None
        or sequence_string.encoding != "utf-8"
        or sequence_string.length is not None
    ):
        raise SelexHdf5ValidationError(
            "data/sequence must use variable-length UTF-8 strings."
        )
    if one_hot_dataset.dtype != ONE_HOT_DTYPE:
        raise SelexHdf5ValidationError("data/s_x must have dtype int8.")
    if target_dataset.dtype != TARGET_DTYPE:
        raise SelexHdf5ValidationError("data/c0_y must have dtype float32.")
    if target_id_dataset.dtype != TARGET_METADATA_DTYPE:
        raise SelexHdf5ValidationError("targets/id must have dtype S16.")
    if target_name_dataset.dtype != TARGET_METADATA_DTYPE:
        raise SelexHdf5ValidationError("targets/name must have dtype S16.")


def _decode_sequences(dataset: h5py.Dataset) -> Tuple[str, ...]:
    sequences = []
    for row_index, raw_value in enumerate(dataset[:]):
        try:
            sequence = _decode_text(raw_value)
        except (TypeError, UnicodeDecodeError) as error:
            message = "Invalid UTF-8 sequence at row {0}."
            raise SelexHdf5ValidationError(
                message.format(row_index)
            ) from error
        sequences.append(sequence)
    return tuple(sequences)


def _decode_scalar_text(value: Any, dataset_name: str) -> str:
    try:
        return _decode_text(value)
    except (TypeError, UnicodeDecodeError) as error:
        message = "{0} contains invalid text."
        raise SelexHdf5ValidationError(message.format(dataset_name)) from error


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError("Expected a string value.")


def _validate_sequences(
    sequences: Sequence[str],
    sequence_length: int,
) -> None:
    valid_bases = frozenset(EXPECTED_CHANNEL_ORDER)
    for row_index, sequence in enumerate(sequences):
        if len(sequence) != sequence_length:
            message = "Sequence row {0} must have length {1}."
            raise SelexHdf5ValidationError(
                message.format(row_index, sequence_length)
            )
        if any(base not in valid_bases for base in sequence):
            message = (
                "Sequence row {0} must contain only uppercase A, C, G, and T."
            )
            raise SelexHdf5ValidationError(message.format(row_index))


def _validate_one_hot(
    one_hot: np.ndarray,
    sequences: Sequence[str],
) -> None:
    if np.any((one_hot != 0) & (one_hot != 1)):
        raise SelexHdf5ValidationError(
            "data/s_x must contain only binary 0/1 values."
        )
    if np.any(np.sum(one_hot, axis=2) != 1):
        raise SelexHdf5ValidationError(
            "Every data/s_x base position must have exactly one active channel."
        )
    channel_array = np.asarray(EXPECTED_CHANNEL_ORDER)
    reconstructed = channel_array[np.argmax(one_hot, axis=2)]
    for row_index, sequence in enumerate(sequences):
        reconstructed_sequence = "".join(reconstructed[row_index].tolist())
        if reconstructed_sequence != sequence:
            message = (
                "data/s_x does not reconstruct data/sequence at row {0}."
            )
            raise SelexHdf5ValidationError(message.format(row_index))


def _validate_targets(targets: np.ndarray) -> None:
    if not np.all(np.isfinite(targets)):
        raise SelexHdf5ValidationError("data/c0_y must contain finite targets.")
    if np.any(targets < 0.0) or np.any(targets > 1.0):
        raise SelexHdf5ValidationError(
            "data/c0_y targets must be within [0, 1]."
        )


def _row_group_maps(
    sequences: Sequence[str],
    targets: Sequence[float],
) -> Dict[str, Any]:
    exact_labels = defaultdict(list)
    canonical_labels = defaultdict(list)
    canonical_sequences = defaultdict(set)
    labeled_indices = defaultdict(list)
    target_values = {}
    for row_index, (sequence, target) in enumerate(zip(sequences, targets)):
        target_key = _target_key(target)
        exact_labels[sequence].append(target_key)
        canonical_sequence = reverse_complement_canonical_sequence(sequence)
        canonical_labels[canonical_sequence].append(target_key)
        canonical_sequences[canonical_sequence].add(sequence)
        labeled_indices[(sequence, target_key)].append(row_index)
        target_values[target_key] = float(target)
    return {
        "exact_labels": exact_labels,
        "canonical_labels": canonical_labels,
        "canonical_sequences": canonical_sequences,
        "labeled_indices": labeled_indices,
        "target_values": target_values,
    }


def _target_key(value: float) -> str:
    return struct.pack(">f", float(np.float32(value))).hex()


def _float_text(value: float) -> str:
    return format(float(value), ".9g")


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _shape_text(shape: Sequence[int]) -> str:
    return json.dumps(list(shape), separators=(",", ":"))


def _duplicate_group_count(counts: Mapping[Any, int]) -> int:
    return sum(count > 1 for count in counts.values())


def _duplicate_extra_occurrence_count(counts: Mapping[Any, int]) -> int:
    return sum(count - 1 for count in counts.values() if count > 1)


def _shared_group_rows(
    group_type: str,
    counts_by_group: Mapping[str, Counter],
) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for group_sequence in sorted(counts_by_group):
        counts = counts_by_group[group_sequence]
        if len(counts) < 2:
            continue
        ordered_counts = {key: counts[key] for key in sorted(counts)}
        rows.append(
            {
                "group_type": group_type,
                "group_sequence": group_sequence,
                "group_sequence_sha256": _sequence_hash(group_sequence),
                "transcription_factor_count": len(ordered_counts),
                "transcription_factors": ",".join(ordered_counts),
                "row_occurrences_by_transcription_factor": json.dumps(
                    ordered_counts,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "total_row_occurrences": sum(ordered_counts.values()),
            }
        )
    return tuple(rows)
