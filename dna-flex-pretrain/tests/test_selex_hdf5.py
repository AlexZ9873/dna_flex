"""Tests for strict Exd-Hox SELEX HDF5 parsing and audits."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from src.coordinates import reverse_complement
from src.selex_hdf5 import (
    SelexHdf5File,
    SelexHdf5ValidationError,
    audit_supplied_split,
    audit_within_tf,
    build_cross_tf_sharing_rows,
    expected_relative_hdf5_paths,
    read_validate_selex_hdf5,
    verify_corresponding_file_identity,
)


CHANNEL_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def _one_hot(sequences):
    values = np.zeros((len(sequences), 14, 4), dtype=np.int8)
    for row_index, sequence in enumerate(sequences):
        for base_index, base in enumerate(sequence):
            values[row_index, base_index, CHANNEL_INDEX[base]] = 1
    return values


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


def _record(tf_name, supplied_split, sequences, targets):
    return SelexHdf5File(
        transcription_factor=tf_name,
        supplied_split=supplied_split,
        logical_path="data/{0}_{1}.h5".format(tf_name, supplied_split),
        sequences=tuple(sequences),
        targets=tuple(targets),
        inventory={},
    )


class SelexHdf5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.valid_sequences = (
            "ACGTTGCAAAAAAA",
            "AAAAAAATTTTTTT",
        )
        self.valid_targets = (0.125, 0.75)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_hdf5_parsing_and_inventory(self) -> None:
        path = self.root / "AbdA_train.h5"
        _write_hdf5(path, self.valid_sequences, self.valid_targets)

        record = read_validate_selex_hdf5(
            path,
            transcription_factor="AbdA",
            supplied_split="train",
            logical_path="data/raw/exd/AbdA/AbdA_train.h5",
        )

        self.assertEqual(record.sequences, self.valid_sequences)
        self.assertEqual(record.targets, self.valid_targets)
        self.assertEqual(record.inventory["row_count"], 2)
        self.assertEqual(record.inventory["sequence_shape"], "[2]")
        self.assertEqual(record.inventory["one_hot_shape"], "[2,14,4]")
        self.assertEqual(record.inventory["one_hot_dtype"], "int8")
        self.assertEqual(record.inventory["target_dtype"], "float32")
        self.assertEqual(record.inventory["one_hot_channel_order"], "A,C,G,T")

    def test_malformed_dataset_shapes_and_dtypes_are_rejected(self) -> None:
        sequence_dtype = h5py.string_dtype(encoding="utf-8")
        valid_one_hot = _one_hot(self.valid_sequences)
        short_one_hot = valid_one_hot[:, :13, :].copy()
        self.assertEqual(
            short_one_hot.shape,
            (len(self.valid_sequences), 13, 4),
        )
        self.assertEqual(short_one_hot.dtype, np.dtype("int8"))
        self.assertTrue(np.all((short_one_hot == 0) | (short_one_hot == 1)))
        self.assertTrue(np.all(np.sum(short_one_hot, axis=2) == 1))

        valid_targets = np.asarray(
            self.valid_targets,
            dtype=np.float32,
        ).reshape(-1, 1)
        malformed_targets = valid_targets.astype(np.float64)
        self.assertEqual(malformed_targets.shape, valid_targets.shape)
        np.testing.assert_array_equal(malformed_targets, valid_targets)
        self.assertNotEqual(malformed_targets.dtype, valid_targets.dtype)

        malformed_sequences = np.asarray(self.valid_sequences, dtype="S14")
        decoded_malformed_sequences = tuple(
            value.decode("ascii") for value in malformed_sequences
        )
        self.assertEqual(
            malformed_sequences.shape,
            (len(self.valid_sequences),),
        )
        self.assertEqual(decoded_malformed_sequences, self.valid_sequences)
        self.assertNotEqual(malformed_sequences.dtype, sequence_dtype)

        malformed_cases = (
            (
                "one_hot_wrong_sequence_dimension",
                "data/s_x",
                short_one_hot,
                None,
                r"data/s_x must have shape",
            ),
            (
                "target_wrong_rank",
                "data/c0_y",
                np.asarray(self.valid_targets, dtype=np.float32),
                None,
                r"data/c0_y must have shape",
            ),
            (
                "sequence_wrong_rank",
                "data/sequence",
                np.asarray(self.valid_sequences, dtype=object).reshape(-1, 1),
                sequence_dtype,
                r"data/sequence must have shape N",
            ),
            (
                "one_hot_wrong_dtype",
                "data/s_x",
                _one_hot(self.valid_sequences).astype(np.float32),
                None,
                r"data/s_x must have dtype int8",
            ),
            (
                "target_wrong_dtype",
                "data/c0_y",
                malformed_targets,
                None,
                r"data/c0_y must have dtype float32",
            ),
            (
                "sequence_wrong_dtype",
                "data/sequence",
                malformed_sequences,
                None,
                r"data/sequence must use variable-length UTF-8 strings",
            ),
        )

        for (
            case_name,
            dataset_path,
            replacement_data,
            replacement_dtype,
            expected_error,
        ) in malformed_cases:
            with self.subTest(case=case_name):
                path = self.root / "{0}.h5".format(case_name)
                _write_hdf5(path, self.valid_sequences, self.valid_targets)
                with h5py.File(path, "a") as output_file:
                    del output_file[dataset_path]
                    if replacement_dtype is None:
                        output_file.create_dataset(
                            dataset_path,
                            data=replacement_data,
                        )
                    else:
                        output_file.create_dataset(
                            dataset_path,
                            data=replacement_data,
                            dtype=replacement_dtype,
                        )
                    if case_name == "target_wrong_dtype":
                        stored_targets = np.asarray(output_file[dataset_path][:])
                        self.assertEqual(stored_targets.shape, valid_targets.shape)
                        np.testing.assert_array_equal(stored_targets, valid_targets)
                    if case_name == "sequence_wrong_dtype":
                        stored_sequences = output_file[dataset_path][:]
                        decoded_stored_sequences = tuple(
                            value.decode("ascii") for value in stored_sequences
                        )
                        self.assertEqual(
                            output_file[dataset_path].shape,
                            (len(self.valid_sequences),),
                        )
                        self.assertEqual(
                            decoded_stored_sequences,
                            self.valid_sequences,
                        )

                with self.assertRaisesRegex(
                    SelexHdf5ValidationError,
                    expected_error,
                ):
                    read_validate_selex_hdf5(
                        path,
                        "AbdA",
                        "train",
                        "data/{0}.h5".format(case_name),
                    )

    def test_malformed_group_schema_is_rejected(self) -> None:
        path = self.root / "malformed.h5"
        _write_hdf5(path, self.valid_sequences, self.valid_targets)
        with h5py.File(path, "a") as output_file:
            del output_file["targets/name"]

        with self.assertRaisesRegex(
            SelexHdf5ValidationError,
            "targets must contain exactly",
        ):
            read_validate_selex_hdf5(
                path,
                "AbdA",
                "train",
                "data/malformed.h5",
            )

    def test_incorrect_one_hot_reconstruction_is_rejected(self) -> None:
        path = self.root / "bad_one_hot.h5"
        _write_hdf5(path, self.valid_sequences, self.valid_targets)
        with h5py.File(path, "a") as output_file:
            values = output_file["data/s_x"][:]
            values[0, 0, :] = np.asarray([0, 1, 0, 0], dtype=np.int8)
            output_file["data/s_x"][:] = values

        with self.assertRaisesRegex(
            SelexHdf5ValidationError,
            "does not reconstruct",
        ):
            read_validate_selex_hdf5(
                path,
                "AbdA",
                "train",
                "data/bad_one_hot.h5",
            )

    def test_invalid_sequence_is_rejected(self) -> None:
        path = self.root / "bad_sequence.h5"
        sequences = ("acgttgcaaaaaaa",)
        path.parent.mkdir(parents=True, exist_ok=True)
        sequence_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path, "w") as output_file:
            data_group = output_file.create_group("data")
            targets_group = output_file.create_group("targets")
            data_group.create_dataset(
                "sequence",
                data=np.asarray(sequences, dtype=object),
                dtype=sequence_dtype,
            )
            data_group.create_dataset(
                "s_x",
                data=np.zeros((1, 14, 4), dtype=np.int8),
            )
            data_group.create_dataset(
                "c0_y",
                data=np.asarray([[0.5]], dtype=np.float32),
            )
            targets_group.create_dataset(
                "id", data=np.asarray([b"c0"], dtype="S16")
            )
            targets_group.create_dataset(
                "name", data=np.asarray([b"dummy"], dtype="S16")
            )

        with self.assertRaisesRegex(
            SelexHdf5ValidationError,
            "uppercase A, C, G, and T",
        ):
            read_validate_selex_hdf5(
                path,
                "AbdA",
                "train",
                "data/bad_sequence.h5",
            )

    def test_invalid_target_is_rejected(self) -> None:
        path = self.root / "bad_target.h5"
        _write_hdf5(path, self.valid_sequences, (0.5, 1.1))

        with self.assertRaisesRegex(
            SelexHdf5ValidationError,
            r"within \[0, 1\]",
        ):
            read_validate_selex_hdf5(
                path,
                "AbdA",
                "train",
                "data/bad_target.h5",
            )

    def test_exact_rc_and_self_rc_grouping(self) -> None:
        sequence = "ACGTTGCAAAAAAA"
        reverse = reverse_complement(sequence)
        self_reverse = "AAAAAAATTTTTTT"
        summary = audit_within_tf(
            "AbdA",
            (sequence, sequence, reverse, self_reverse, self_reverse),
            (0.1, 0.2, 0.1, 0.3, 0.3),
        )

        self.assertEqual(summary["exact_duplicate_sequence_group_count"], 2)
        self.assertEqual(summary["reverse_complement_only_group_count"], 1)
        self.assertEqual(
            summary["self_reverse_complement_unique_sequence_count"],
            1,
        )
        self.assertEqual(
            summary["self_reverse_complement_row_occurrence_count"],
            2,
        )
        self.assertEqual(summary["exact_conflicting_label_group_count"], 1)
        self.assertEqual(
            summary["reverse_complement_conflicting_label_group_count"],
            1,
        )

    def test_supplied_split_distinguishes_exact_and_rc_only_overlap(self) -> None:
        exact_sequence = "ACGTTGCAAAAAAA"
        rc_training_sequence = "CCCCCCCAAAAAAA"
        rc_test_sequence = reverse_complement(rc_training_sequence)
        training = _record(
            "AbdA",
            "train",
            (exact_sequence, rc_training_sequence),
            (0.25, 0.5),
        )
        test = _record(
            "AbdA",
            "test",
            (exact_sequence, rc_test_sequence),
            (0.25, 0.5),
        )

        summary, details = audit_supplied_split(training, test)

        self.assertEqual(summary["exact_sequence_overlap_group_count"], 1)
        self.assertEqual(summary["exact_labeled_row_overlap_count"], 1)
        self.assertEqual(
            summary["reverse_complement_equivalent_overlap_group_count"],
            2,
        )
        self.assertEqual(
            summary["reverse_complement_only_overlap_group_count"],
            1,
        )
        self.assertEqual(summary["exact_conflicting_label_group_count"], 0)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["sequence"], exact_sequence)

    def test_cross_tf_different_labels_are_sharing_not_conflicts(self) -> None:
        shared_sequence = "ACGTTGCAAAAAAA"
        records_by_tf = {
            "AbdA": {
                "train": _record("AbdA", "train", (shared_sequence,), (0.1,)),
                "test": _record("AbdA", "test", (), ()),
            },
            "Ubx": {
                "train": _record("Ubx", "train", (shared_sequence,), (0.9,)),
                "test": _record("Ubx", "test", (), ()),
            },
        }

        rows, summary = build_cross_tf_sharing_rows(records_by_tf)

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["group_type"] for row in rows},
            {"exact_sequence", "reverse_complement_equivalent"},
        )
        self.assertTrue(
            all(row["transcription_factor_count"] == 2 for row in rows)
        )
        self.assertTrue(all("conflict" not in row for row in rows))
        self.assertEqual(summary["exact_sequence"]["shared_group_count"], 1)

    def test_canonical_and_rcmodel_file_identity(self) -> None:
        canonical = self.root / "canonical"
        comparison = self.root / "comparison"
        relative_paths = expected_relative_hdf5_paths(("AbdA",))
        for relative_path in relative_paths:
            canonical_path = canonical / relative_path
            comparison_path = comparison / relative_path
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            canonical_path.write_bytes(relative_path.encode("ascii"))
            comparison_path.write_bytes(relative_path.encode("ascii"))

        identities = verify_corresponding_file_identity(
            canonical,
            comparison,
            relative_paths,
        )
        self.assertTrue(all(row["byte_identical"] for row in identities))

        (comparison / relative_paths[0]).write_bytes(b"changed")
        with self.assertRaisesRegex(
            SelexHdf5ValidationError,
            "differ",
        ):
            verify_corresponding_file_identity(
                canonical,
                comparison,
                relative_paths,
            )


if __name__ == "__main__":
    unittest.main()
