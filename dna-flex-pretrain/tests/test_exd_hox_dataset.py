"""Hermetic CPU tests for public Exd-Hox membership and exact encoding.

All biological rows, identities, metadata and stages in this module are
synthetic. Fixture hashes are calculated independently of production helpers.
"""

import ast
import builtins
import copy
from dataclasses import FrozenInstanceError, replace
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from src import exd_hox_dataset as dataset_module


LOGICAL_FIELDS = (
    "logical_example_id", "transcription_factor", "sequence", "sequence_sha256",
    "reverse_complement_canonical_sequence", "reverse_complement_canonical_sha256",
    "global_rc_group_id", "primary_split", "target_value_float32",
    "target_bits_big_endian_hex", "target_commitment_sha256",
    "source_occurrence_count",
)
ORDERING_FIELDS = (
    "transcription_factor", "rank_one_based", "global_rc_group_id",
    "logical_example_id", "training_affinity_bin", "deterministic_order_sha256",
)
LEVEL_FIELDS = (
    "transcription_factor", "level_id", "request_type", "request_value",
    "unaliased_requested_logical_example_count", "alias_absolute_anchor",
    "canonical_requested_logical_example_count", "actual_logical_example_count",
    "actual_rc_group_count", "inclusive_maximum_rank",
)
TF_NAMES = ("AbdA", "AbdB", "Antp", "Dfd", "Lab", "Pb", "Scr", "Ubx")
SPLIT_PATH = "data/processed/exd_hox_primary_split_v1"
SUBSET_PATH = "data/processed/exd_hox_nested_subsets_v1"
AUDIT_PATH = "data/processed/exd_hox_selex_audit_v1"
LOGICAL_PATH = SPLIT_PATH + "/exd_hox_logical_examples_v1.tsv.gz"
ORDERING_PATH = SUBSET_PATH + "/exd_hox_nested_subset_ordering_v1.tsv.gz"
LEVEL_PATH = SUBSET_PATH + "/exd_hox_nested_subset_levels_v1.tsv"
SPLIT_MANIFEST_PATH = SPLIT_PATH + "/exd_hox_primary_split_manifest_v1.json"
SUBSET_MANIFEST_PATH = SUBSET_PATH + "/exd_hox_subset_set_manifest_v1.json"
SOURCE_MANIFEST_PATH = AUDIT_PATH + "/exd_hox_source_manifest_v1.json"
AUDIT_MANIFEST_PATH = AUDIT_PATH + "/exd_hox_audit_manifest_v1.json"
PRIMARY_CONFIG_PATH = "configs/exd_hox_primary_split_v1.yaml"
ACCESS_CONFIG_PATH = "configs/exd_hox_test_access_policy_v1.yaml"
METADATA_PATHS = (
    PRIMARY_CONFIG_PATH, ACCESS_CONFIG_PATH, SPLIT_MANIFEST_PATH,
    SUBSET_MANIFEST_PATH, LEVEL_PATH, SOURCE_MANIFEST_PATH, AUDIT_MANIFEST_PATH,
)


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _domain(domain, *parts):
    return _digest("\0".join((domain,) + tuple(parts)).encode("ascii"))


def _manifest(schema, content):
    value = {"schema_version": schema, **content}
    value["manifest_hash"] = _digest(_json_bytes(value))
    return value


def _reverse(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _sequence(index):
    bases = []
    for shift in range(13, -1, -1):
        bases.append("ACGT"[(index // (4 ** shift)) % 4])
    return "".join(bases)


def _logical_row(sequence, split="training", tf="Ubx", bits="3f000000"):
    canonical = min(sequence, _reverse(sequence))
    logical_id = "lex_" + _domain("exd_hox_logical_example.v1", tf, sequence, bits)
    target = struct.unpack(">f", bytes.fromhex(bits))[0]
    return {
        "logical_example_id": logical_id,
        "transcription_factor": tf,
        "sequence": sequence,
        "sequence_sha256": _digest(sequence.encode("ascii")),
        "reverse_complement_canonical_sequence": canonical,
        "reverse_complement_canonical_sha256": _digest(canonical.encode("ascii")),
        "global_rc_group_id": "rcg_" + _domain("exd_hox_global_rc_group.v1", canonical),
        "primary_split": split,
        "target_value_float32": "" if split == "test" else format(target, ".9g"),
        "target_bits_big_endian_hex": "" if split == "test" else bits,
        "target_commitment_sha256": _domain(
            "exd_hox_target_commitment.v1", logical_id, bits,
        ),
        "source_occurrence_count": "1",
    }


def _ordering_row(logical, rank):
    return {
        "transcription_factor": logical["transcription_factor"],
        "rank_one_based": str(rank),
        "global_rc_group_id": logical["global_rc_group_id"],
        "logical_example_id": logical["logical_example_id"],
        "training_affinity_bin": "0",
        "deterministic_order_sha256": _domain(
            "exd_hox_nested_subset_group_order.v1", "32001",
            logical["transcription_factor"], logical["global_rc_group_id"],
        ),
    }


def _level(name, kind, request, canonical, count, rank, alias="", tf="Ubx"):
    return {
        "transcription_factor": tf,
        "level_id": "lvl_" + _digest(name.encode("ascii")),
        "request_type": kind,
        "request_value": request,
        "unaliased_requested_logical_example_count": str(canonical),
        "alias_absolute_anchor": str(alias),
        "canonical_requested_logical_example_count": str(canonical),
        "actual_logical_example_count": str(count),
        "actual_rc_group_count": str(rank),
        "inclusive_maximum_rank": str(rank),
    }


def _synthetic_rows(overshoot=False):
    """Make a repeated RC group and independent literal prefix expectations."""
    logical = []
    ordering = []
    repeated_rank = 128 if overshoot else 1
    for rank in range(1, 256):
        sequence = _sequence(rank)
        group_rows = [_logical_row(sequence)]
        if rank == repeated_rank:
            group_rows.append(_logical_row(_reverse(sequence)))
        group_rows.sort(key=lambda row: row["logical_example_id"])
        for row in group_rows:
            logical.append(row)
            ordering.append(_ordering_row(row, rank))
    logical.append(_logical_row(_sequence(1001), "validation", bits="3e800000"))
    logical.append(_logical_row(_sequence(1002), "validation", bits="3f400000"))
    logical.append(_logical_row(_sequence(1003), "test"))
    logical.sort(key=lambda row: row["logical_example_id"])
    prefix_count = 129 if overshoot else 128
    prefix_rank = 128 if overshoot else 127
    levels = [
        _level("n128", "absolute", "128", 128, prefix_count, prefix_rank),
        _level("half", "fractional", "0.5", 128, prefix_count, prefix_rank, 128),
        _level("n256", "absolute", "256", 256, 256, 255),
        _level("full", "fractional", "1.0", 256, 256, 255, 256),
    ]
    return logical, ordering, levels


def _table_bytes(fields, rows):
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row[field]) for field in fields))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _gzip_bytes(value):
    target = io.BytesIO()
    output = gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0)
    output.write(value)
    output.close()
    return target.getvalue()


def _stage_manifest(contract):
    content = {}
    for key in (
        "purpose", "staging_policy_identifier", "foundation_commit",
        "dataset_identifier", "split_identity_hash", "split_manifest_hash",
        "subset_set_manifest_hash", "allowed_dataset_selections",
    ):
        content[key] = contract[key]
    content["staging_config_path"] = "configs/carc_exd_hox_training_staging_v1.json"
    content["staging_config_sha256"] = _digest(_json_bytes(contract) + b"\n")
    content["files"] = sorted(
        contract["tracked_metadata"] + contract["transferred_public_payloads"],
        key=lambda record: record["path"],
    )
    return _manifest("exd_hox_training_stage_manifest.v1", content)


class SyntheticPublicFiles:
    """Own the complete nine-file synthetic input graph, never production data."""

    def __init__(self, root, overshoot=False):
        self.root = root
        self.logical, self.ordering, self.level_rows = _synthetic_rows(overshoot)
        self.logical_fields = LOGICAL_FIELDS
        self.ordering_fields = ORDERING_FIELDS
        self.level_fields = LEVEL_FIELDS
        self.overrides = {}
        self.metadata_changes = {}
        self.contract_changes = {}
        self.content = {}

    def replace_logical_row(self, original_id, replacement):
        """Update only the dependent IDs and serialization order of one row."""
        for position, row in enumerate(self.logical):
            if row["logical_example_id"] == original_id:
                self.logical[position] = replacement
        for row in self.ordering:
            if row["logical_example_id"] == original_id:
                row["logical_example_id"] = replacement["logical_example_id"]
        self.logical.sort(key=lambda row: row["logical_example_id"])
        self.ordering.sort(key=lambda row: (
            row["transcription_factor"], int(row["rank_one_based"]), row["logical_example_id"],
        ))

    def _store(self, logical_path, value):
        self.content[logical_path] = value
        output = self.root / logical_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(value)

    def _record(self, logical_path):
        value = self.content[logical_path]
        return {"path": logical_path, "byte_size": len(value), "sha256": _digest(value)}

    def _store_manifest(self, path, schema, content):
        content.update(self.metadata_changes.get(path, {}))
        value = _manifest(schema, content)
        self._store(path, _json_bytes(value))
        return value

    def write(self):
        for path, fields, rows, compressed in (
            (LOGICAL_PATH, self.logical_fields, self.logical, True),
            (ORDERING_PATH, self.ordering_fields, self.ordering, True),
            (LEVEL_PATH, self.level_fields, self.level_rows, False),
        ):
            value = _table_bytes(fields, rows)
            if compressed:
                value = _gzip_bytes(value)
            self._store(path, self.overrides.get(path, value))
        dataset_id = "wang_etal_exd_hox_selex_canonical.v1"
        source = self._store_manifest(
            SOURCE_MANIFEST_PATH, "exd_hox_source_manifest.v1", {
                "dataset_identifier": dataset_id,
                "source_commit": "9e6d6ef0355558c98855b83a9c21fe11999f65d9",
            },
        )
        audit = self._store_manifest(
            AUDIT_MANIFEST_PATH, "exd_hox_audit_manifest.v1", {
                "dataset_identifier": dataset_id,
                "source_manifest_hash": source["manifest_hash"],
                "totals": {"total_row_occurrences": 259},
            },
        )
        access = self._store_manifest(
            ACCESS_CONFIG_PATH, "exd_hox_test_access_policy.v1", {
                "authorized_operation": "final_primary_test_evaluation",
            },
        )
        primary = {
            "schema_version": "exd_hox_primary_split_config.v1",
            "dataset": {"sequence_length": 14, "transcription_factors": ["Ubx"]},
            "split_policy": {"seed": 31001},
            "subset_policy": {"seed": 32001},
            "inputs": {
                "source_manifest_hash": source["manifest_hash"],
                "audit_manifest_hash": audit["manifest_hash"],
                "source_manifest_file_sha256": self._record(SOURCE_MANIFEST_PATH)["sha256"],
                "audit_manifest_file_sha256": self._record(AUDIT_MANIFEST_PATH)["sha256"],
            },
            "test_access": {
                "policy_manifest_hash": access["manifest_hash"],
                "policy_file_sha256": self._record(ACCESS_CONFIG_PATH)["sha256"],
            },
        }
        primary.update(self.metadata_changes.get(PRIMARY_CONFIG_PATH, {}))
        self._store(PRIMARY_CONFIG_PATH, _json_bytes(primary))
        split_counts = {"training": 256, "validation": 2, "test": 1}
        group_counts = {"training": 255, "validation": 2, "test": 1}
        per_tf = {"Ubx": {}}
        for split in split_counts:
            per_tf["Ubx"][split] = {
                "logical_examples": split_counts[split],
                "rc_groups": group_counts[split],
            }
        split_identity = _digest(b"synthetic accepted split")
        split = self._store_manifest(
            SPLIT_MANIFEST_PATH, "exd_hox_primary_split_manifest.v1", {
                "dataset_identifier": dataset_id,
                "split_identity_hash": split_identity,
                "config_path": PRIMARY_CONFIG_PATH,
                "config_sha256": self._record(PRIMARY_CONFIG_PATH)["sha256"],
                "source_manifest_hash": source["manifest_hash"],
                "audit_manifest_hash": audit["manifest_hash"],
                "test_access_policy_manifest_hash": access["manifest_hash"],
                "policy": primary["split_policy"],
                "counts": {
                    "source_occurrences": 259,
                    "logical_examples": 259,
                    "global_rc_group_counts": group_counts,
                    "per_tf_split_counts": {"Ubx": split_counts},
                },
                "artifacts": [self._record(LOGICAL_PATH)],
            },
        )
        subset = self._store_manifest(
            SUBSET_MANIFEST_PATH, "exd_hox_subset_set_manifest.v1", {
                "dataset_identifier": dataset_id,
                "split_identity_hash": split_identity,
                "split_manifest_hash": split["manifest_hash"],
                "config_path": PRIMARY_CONFIG_PATH,
                "config_sha256": self._record(PRIMARY_CONFIG_PATH)["sha256"],
                "policy": primary["subset_policy"],
                "level_row_count": 4,
                "ordering_logical_example_count": 256,
                "artifacts": [self._record(ORDERING_PATH), self._record(LEVEL_PATH)],
            },
        )
        contract = {
            "schema_version": "carc_exd_hox_training_staging_config.v1",
            "purpose": "training_validation_only",
            "staging_policy_identifier": "exd_hox_public_training_stage.v1",
            "foundation_commit": "711e5b17e07cabf3d2e6a0ff35f46ea520fdad68",
            "dataset_identifier": dataset_id,
            "split_identity_hash": split_identity,
            "split_manifest_hash": split["manifest_hash"],
            "subset_set_manifest_hash": subset["manifest_hash"],
            "allowed_dataset_selections": ["training", "validation"],
            "transcription_factors": ["Ubx"],
            "tracked_metadata": [],
            "transferred_public_payloads": [],
            "forbidden_path_classes": [
                "data/sealed/**", "data/raw/**", "**/*.h5", "**/*.hdf5",
                "**/SELEX_canonical/**", "**/SELEX_RCmodel/**",
                "**/exd_hox_public_test_inputs_v1.tsv.gz",
                "**/exd_hox_source_occurrence_provenance_v1.tsv.gz",
                "**/exd_hox_global_rc_groups_v1.tsv.gz",
                "**/exd_hox_primary_split_assignments_v1.tsv.gz",
                "**/exd_hox_sealed_test_targets_v1.tsv.gz",
                "**/exd_hox_sealed_test_target_manifest_v1.json",
                "results/**", "plots/**", "checkpoints/**",
                "**/authorization*", "**/test_access_record*",
            ],
            "expected_counts": {
                "source_occurrences": 259,
                "logical_examples": 259,
                "split_logical_examples": split_counts,
                "global_rc_groups": group_counts,
                "ordering_logical_examples": 256,
                "level_rows": 4,
                "per_tf": per_tf,
            },
        }
        for path in sorted(METADATA_PATHS):
            contract["tracked_metadata"].append(self._record(path))
        for path in sorted((LOGICAL_PATH, ORDERING_PATH)):
            contract["transferred_public_payloads"].append(self._record(path))
        contract.update(self.contract_changes)
        return contract


class OneHotTests(unittest.TestCase):
    def test_exact_channel_order_shape_and_fresh_storage(self):
        sequence = "ACGTACGTACGTAC"
        encoded = dataset_module.encode_one_hot_14(sequence)
        self.assertEqual(encoded.shape, (14, 4))
        self.assertEqual(encoded.dtype, torch.float32)
        self.assertEqual(encoded.device.type, "cpu")
        expected = torch.eye(4, dtype=torch.float32)[
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        ]
        self.assertTrue(torch.equal(encoded, expected))
        encoded.zero_()
        self.assertTrue(torch.equal(dataset_module.encode_one_hot_14(sequence), expected))

    def test_invalid_sequence_never_normalized_or_cast(self):
        cases = (
            "ACGT", "A" * 13, "A" * 15, "a" * 14, "N" * 14,
            "ACGTACGTACGTA ", " ACGTACGTACGTA", "A" * 13 + "\n",
            b"A" * 14, ["A"] * 14, None, 14,
        )
        for sequence in cases:
            with self.subTest(sequence=sequence):
                with self.assertRaises((TypeError, ValueError)):
                    dataset_module.encode_one_hot_14(sequence)

    def test_declared_schemas_match_independent_literals(self):
        self.assertEqual(tuple(dataset_module.LOGICAL_FIELDS), LOGICAL_FIELDS)
        self.assertEqual(tuple(dataset_module.ORDERING_FIELDS), ORDERING_FIELDS)
        self.assertEqual(tuple(dataset_module.LEVEL_FIELDS), LEVEL_FIELDS)


class PublicDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.fixture = SyntheticPublicFiles(self.root)

    def _load(self, fixture=None):
        fixture = self.fixture if fixture is None else fixture
        contract = fixture.write()
        stage_id = "exd_hox_training_stage_" + _stage_manifest(contract)["manifest_hash"]
        return dataset_module._read_public_tf_data(
            fixture.root, contract, "Ubx", stage_id,
        )

    def _training(self, data, position=0):
        return data.dataset("training", level_id=self.fixture.level_rows[position]["level_id"])

    def _ids(self, data):
        return tuple(data[index].metadata.logical_example_id for index in range(len(data)))

    def _replace_target(self, fixture, bits):
        target_index = next(
            index for index, row in enumerate(fixture.logical)
            if row["primary_split"] == "validation"
        )
        original = fixture.logical[target_index]
        replacement = _logical_row(original["sequence"], "validation", bits=bits)
        fixture.logical[target_index] = replacement
        fixture.logical.sort(key=lambda row: row["logical_example_id"])

    def test_exact_n128_order_unique_groups_alias_and_full_membership(self):
        data = self._load()
        training = self._training(data)
        expected = tuple(row["logical_example_id"] for row in self.fixture.ordering[:128])
        self.assertEqual(len(training), 128)
        self.assertEqual(self._ids(training), expected)
        groups = set()
        for sample in training:
            groups.add(sample.metadata.global_rc_group_id)
        self.assertEqual(len(groups), 127)
        self.assertEqual(training.unique_rc_group_count, 127)
        self.assertEqual(self._ids(self._training(data, 1)), expected)
        self.assertEqual(len(self._training(data, 2)), 256)
        self.assertEqual(self._ids(self._training(data, 3)), self._ids(self._training(data, 2)))
        full_groups = set(sample.metadata.global_rc_group_id for sample in self._training(data, 3))
        self.assertEqual(len(full_groups), 255)
        self.assertEqual(len(data.levels()), 4)

    def test_whole_group_overshoot_retains_every_member(self):
        self.fixture = SyntheticPublicFiles(self.root, overshoot=True)
        data = self._load()
        training = self._training(data)
        self.assertEqual(len(training), 129)
        self.assertEqual(len(set(sample.metadata.global_rc_group_id for sample in training)), 128)
        self.assertEqual(self._ids(training), self._ids(self._training(data, 1)))
        self.assertEqual(self._ids(training), tuple(
            row["logical_example_id"] for row in self.fixture.ordering[:129]
        ))

    def test_validation_is_complete_sorted_and_sealed_paths_absent(self):
        data = self._load()
        validation = data.dataset("validation")
        expected = tuple(sorted(
            row["logical_example_id"] for row in self.fixture.logical
            if row["primary_split"] == "validation"
        ))
        self.assertEqual(len(validation), 2)
        self.assertEqual(self._ids(validation), expected)
        self.assertFalse((self.root / "data/sealed").exists())
        for sample in validation:
            self.assertEqual(sample.metadata.primary_split, "validation")

    def test_exact_sample_contract_alias_metadata_and_mutation_safety(self):
        data = self._load()
        training = self._training(data)
        sample = training[0]
        self.assertEqual(sample.x.shape, (14, 4))
        self.assertEqual(sample.y.shape, (1,))
        self.assertEqual(sample.x.dtype, torch.float32)
        self.assertEqual(sample.y.dtype, torch.float32)
        self.assertEqual(sample.x.device.type, "cpu")
        self.assertEqual(sample.y.device.type, "cpu")
        self.assertEqual(sample.y.item(), 0.5)
        self.assertEqual(sample.metadata.target_bits_big_endian_hex, "3f000000")
        self.assertEqual(sample.metadata.rank_one_based, 1)
        self.assertEqual(sample.metadata.level_id, self.fixture.level_rows[0]["level_id"])
        alias = self._training(data, 1)[0]
        self.assertEqual(alias.metadata.level_id, self.fixture.level_rows[1]["level_id"])
        self.assertEqual(alias.metadata.canonical_level_id, sample.metadata.level_id)
        before = sample.x.clone()
        sample.x.zero_()
        sample.y.zero_()
        fresh = training[0]
        self.assertTrue(torch.equal(fresh.x, before))
        self.assertEqual(fresh.y.item(), 0.5)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            fresh.metadata.sequence = "A" * 14

    def test_deterministic_repeated_open_and_no_rng_consumption(self):
        state = torch.random.get_rng_state().clone()
        first = self._training(self._load())
        second = self._training(self._load())
        self.assertEqual(self._ids(first), self._ids(second))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_selection_rejects_tests_unknowns_missing_or_wrong_levels(self):
        data = self._load()
        for selection in ("test", "primary_test", "supplied_paper_test", "paper_test", "train", "", None):
            with self.subTest(selection=selection):
                with self.assertRaises((TypeError, ValueError)):
                    data.dataset(selection)
        with self.assertRaises((TypeError, ValueError)):
            data.dataset("training")
        with self.assertRaises((TypeError, ValueError)):
            data.dataset("training", level_id="lvl_" + "f" * 64)
        with self.assertRaises((TypeError, ValueError)):
            data.dataset("validation", level_id=self.fixture.level_rows[0]["level_id"])

    def test_collate_shapes_metadata_and_independent_storage(self):
        data = self._training(self._load())
        samples = (data[0], data[1])
        batch = dataset_module.collate_exd_hox(samples)
        self.assertEqual(batch.x.shape, (2, 14, 4))
        self.assertEqual(batch.y.shape, (2, 1))
        self.assertEqual(batch.x.dtype, torch.float32)
        self.assertEqual(batch.y.dtype, torch.float32)
        self.assertEqual(batch.metadata, tuple(sample.metadata for sample in samples))
        batch.x.zero_()
        batch.y.zero_()
        self.assertEqual(samples[0].x.sum().item(), 14)
        self.assertEqual(samples[0].y.item(), 0.5)

    def test_collate_rejects_invalid_values_types_shapes_and_mixed_identity(self):
        data = self._load()
        sample = self._training(data)[0]
        invalid_x = sample.x.clone()
        invalid_x[0, :] = 0
        nonbinary_x = sample.x.clone()
        nonbinary_x[0, :] = 0.25
        variants = (
            (), (None,), (replace(sample, x=invalid_x),),
            (replace(sample, x=nonbinary_x),),
            (replace(sample, x=sample.x.double()),),
            (replace(sample, x=sample.x[:, :3]),),
            (replace(sample, x=sample.x.flip(dims=(1,))),),
            (replace(sample, y=sample.y.double()),),
            (replace(sample, y=torch.tensor([float("nan")])),),
            (replace(sample, y=torch.tensor([1.25])),),
            (replace(sample, y=torch.tensor([0.75])),),
            (replace(sample, y=torch.tensor(0.5)),),
            (sample, data.dataset("validation")[0]),
            (sample, self._training(data, 1)[0]),
        )
        for index, values in enumerate(variants):
            with self.subTest(case=index):
                with self.assertRaises((TypeError, ValueError)):
                    dataset_module.collate_exd_hox(values)

    def test_singleton_collate_rejects_forged_training_metadata(self):
        sample = self._training(self._load())[0]
        cases = (
            ("logical_example_id", "lex_" + "f" * 64),
            ("transcription_factor", "unknown"),
            ("global_rc_group_id", "rcg_" + "f" * 64),
            ("stage_id", "invalid-stage"),
            ("split_identity_hash", "invalid-split"),
            ("split_manifest_hash", "invalid-manifest"),
            ("subset_set_manifest_hash", "invalid-subset"),
            ("sample_rank_one_based", 0), ("sample_rank_one_based", True),
            ("rank_one_based", 0), ("rank_one_based", True),
            ("level_id", None), ("canonical_level_id", None),
            ("validation_id", "validation_" + "f" * 64),
            ("target_value_float32", 0.50000000001),
        )
        for key, value in cases:
            with self.subTest(field=key, value=value):
                metadata = replace(sample.metadata, **{key: value})
                changed = replace(sample, metadata=metadata)
                with self.assertRaises(ValueError):
                    dataset_module.collate_exd_hox((changed,))

    def test_singleton_collate_rejects_invalid_validation_identity_and_training_fields(self):
        sample = self._load().dataset("validation")[0]
        cases = (
            ("validation_id", "validation_" + "f" * 64),
            ("validation_id", None), ("level_id", "lvl_" + "f" * 64),
            ("canonical_level_id", "lvl_" + "f" * 64), ("rank_one_based", 1),
        )
        for key, value in cases:
            with self.subTest(field=key):
                metadata = replace(sample.metadata, **{key: value})
                with self.assertRaises(ValueError):
                    dataset_module.collate_exd_hox((replace(sample, metadata=metadata),))

    def test_logical_invalid_fields_fail_with_valid_byte_envelopes(self):
        cases = (
            ("transcription_factor", "unknown"), ("primary_split", "supplied_test"),
            ("sequence", "a" * 14), ("sequence", "A" * 13),
            ("sequence", "N" * 14), ("sequence_sha256", "f" * 64),
            ("reverse_complement_canonical_sha256", "f" * 64),
            ("reverse_complement_canonical_sequence", "T" * 14),
            ("global_rc_group_id", "rcg_" + "f" * 64),
            ("source_occurrence_count", "0"),
            ("target_value_float32", ""), ("target_value_float32", "0.5000"),
            ("target_bits_big_endian_hex", ""),
            ("target_bits_big_endian_hex", "3F000000"),
            ("target_bits_big_endian_hex", "3f 00 00 00"),
            ("target_bits_big_endian_hex", "3f00000"),
            ("target_commitment_sha256", "f" * 64),
        )
        for key, value in cases:
            with self.subTest(field=key, value=value):
                fixture = SyntheticPublicFiles(self.root)
                row = next(row for row in fixture.logical if row["primary_split"] == "training")
                original = copy.deepcopy(row)
                row[key] = value
                self.assertEqual(
                    {name: field for name, field in row.items() if name != key},
                    {name: field for name, field in original.items() if name != key},
                )
                with self.assertRaises(ValueError):
                    self._load(fixture)

    def test_nonfinite_and_out_of_range_bits_have_consistent_other_identities(self):
        for bits in ("7fc00000", "7f800000", "ff800000", "bf000000", "3fc00000"):
            with self.subTest(bits=bits):
                fixture = SyntheticPublicFiles(self.root)
                self._replace_target(fixture, bits)
                with self.assertRaises(ValueError):
                    self._load(fixture)

    def test_blank_validation_target_and_unredacted_public_test_are_rejected(self):
        for split in ("validation", "test"):
            fixture = SyntheticPublicFiles(self.root)
            row = next(row for row in fixture.logical if row["primary_split"] == split)
            row["target_value_float32"] = "0.5" if split == "test" else ""
            with self.subTest(split=split):
                with self.assertRaises(ValueError):
                    self._load(fixture)

    def test_duplicate_logical_ids_and_out_of_order_rows_are_rejected(self):
        duplicate = SyntheticPublicFiles(self.root)
        original_count = len(duplicate.logical)
        duplicate.logical[1] = dict(duplicate.logical[0])
        self.assertEqual(len(duplicate.logical), original_count)
        with self.assertRaises(ValueError):
            self._load(duplicate)
        reordered = SyntheticPublicFiles(self.root)
        reordered.logical[0], reordered.logical[1] = reordered.logical[1], reordered.logical[0]
        with self.assertRaises(ValueError):
            self._load(reordered)

    def test_header_field_order_missing_extra_and_row_width(self):
        for path, fields, rows, compressed in (
            (LOGICAL_PATH, LOGICAL_FIELDS, self.fixture.logical, True),
            (ORDERING_PATH, ORDERING_FIELDS, self.fixture.ordering, True),
            (LEVEL_PATH, LEVEL_FIELDS, self.fixture.level_rows, False),
        ):
            correct = _table_bytes(fields, rows)
            lines = correct.splitlines(keepends=True)
            swapped = list(fields)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            variants = (
                b"\t".join(field.encode("ascii") for field in fields[:-1]) + b"\n" + b"".join(lines[1:]),
                lines[0].rstrip(b"\n") + b"\textra\n" + b"".join(lines[1:]),
                "\t".join(swapped).encode("ascii") + b"\n" + b"".join(lines[1:]),
                lines[0] + lines[1].rstrip(b"\n") + b"\textra\n" + b"".join(lines[2:]),
                lines[0] + b"\t".join(lines[1].rstrip(b"\n").split(b"\t")[:-1]) + b"\n" + b"".join(lines[2:]),
            )
            for position, value in enumerate(variants):
                with self.subTest(path=path, case=position):
                    fixture = SyntheticPublicFiles(self.root)
                    fixture.overrides[path] = _gzip_bytes(value) if compressed else value
                    with self.assertRaises(ValueError):
                        self._load(fixture)

    def test_bad_gzip_and_utf8_fail_without_fallback(self):
        for value in (b"not gzip", _gzip_bytes(b"\xff\n"), _gzip_bytes(b"a\n")[:-5]):
            fixture = SyntheticPublicFiles(self.root)
            fixture.overrides[LOGICAL_PATH] = value
            with self.subTest(payload=value[:5]):
                with self.assertRaises(ValueError):
                    self._load(fixture)

    def test_ordering_errors_reject_missing_duplicate_wrong_group_and_rank(self):
        changes = (
            ("rank_one_based", "0"), ("rank_one_based", "3"),
            ("global_rc_group_id", "rcg_" + "f" * 64),
            ("logical_example_id", "lex_" + "f" * 64),
            ("transcription_factor", "AbdA"),
            ("training_affinity_bin", "-1"),
            ("deterministic_order_sha256", "f" * 64),
        )
        for key, value in changes:
            with self.subTest(field=key):
                fixture = SyntheticPublicFiles(self.root)
                fixture.ordering[0][key] = value
                with self.assertRaises(ValueError):
                    self._load(fixture)
        for duplicate in (False, True):
            fixture = SyntheticPublicFiles(self.root)
            if duplicate:
                fixture.ordering[1] = dict(fixture.ordering[0])
            else:
                fixture.ordering.pop(0)
            with self.subTest(duplicate=duplicate):
                with self.assertRaises(ValueError):
                    self._load(fixture)

    def test_group_cannot_have_two_ranks_and_rank_cannot_have_two_groups(self):
        for index, rank in ((1, "2"), (2, "1")):
            fixture = SyntheticPublicFiles(self.root)
            fixture.ordering[index]["rank_one_based"] = rank
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self._load(fixture)

    def test_same_tf_rc_group_cannot_have_conflicting_float32_targets(self):
        first_id = self.fixture.ordering[0]["logical_example_id"]
        original = next(row for row in self.fixture.logical if row["logical_example_id"] == first_id)
        changed = _logical_row(original["sequence"], bits="3e800000")
        self.assertEqual(changed["global_rc_group_id"], original["global_rc_group_id"])
        self.fixture.replace_logical_row(first_id, changed)
        with self.assertRaisesRegex(ValueError, "Conflicting within-TF"):
            self._load()

    def test_global_rc_group_cannot_cross_training_and_validation(self):
        training_id = self.fixture.ordering[2]["logical_example_id"]
        original = next(row for row in self.fixture.logical if row["logical_example_id"] == training_id)
        training = _logical_row(original["sequence"], bits="3e800000")
        self.fixture.replace_logical_row(training_id, training)
        self._load()
        validation = next(
            row for row in self.fixture.logical
            if row["primary_split"] == "validation" and row["target_bits_big_endian_hex"] == "3e800000"
        )
        changed = _logical_row(_reverse(training["sequence"]), "validation", bits="3e800000")
        self.assertEqual(changed["target_value_float32"], validation["target_value_float32"])
        self.fixture.replace_logical_row(validation["logical_example_id"], changed)
        with self.assertRaisesRegex(ValueError, "global RC group spans"):
            self._load()

    def test_levels_reject_cumulative_prefix_alias_and_full_mismatches(self):
        cases = (
            (0, "actual_logical_example_count", "127"),
            (0, "actual_rc_group_count", "128"),
            (0, "inclusive_maximum_rank", "126"),
            (0, "canonical_requested_logical_example_count", "127"),
            (1, "alias_absolute_anchor", "256"),
            (3, "inclusive_maximum_rank", "254"),
            (3, "actual_logical_example_count", "255"),
            (0, "transcription_factor", "AbdA"),
        )
        for index, key, value in cases:
            with self.subTest(index=index, field=key):
                fixture = SyntheticPublicFiles(self.root)
                fixture.level_rows[index][key] = value
                with self.assertRaises(ValueError):
                    self._load(fixture)

    def test_manifest_links_fail_even_when_semantic_and_byte_hashes_are_valid(self):
        cases = (
            (SPLIT_MANIFEST_PATH, "split_identity_hash", "f" * 64),
            (SPLIT_MANIFEST_PATH, "config_sha256", "f" * 64),
            (SUBSET_MANIFEST_PATH, "split_identity_hash", "f" * 64),
            (SUBSET_MANIFEST_PATH, "split_manifest_hash", "f" * 64),
            (SOURCE_MANIFEST_PATH, "source_commit", "f" * 40),
            (SUBSET_MANIFEST_PATH, "ordering_logical_example_count", 255),
        )
        for path, key, value in cases:
            with self.subTest(path=path, field=key):
                fixture = SyntheticPublicFiles(self.root)
                fixture.metadata_changes[path] = {key: value}
                with self.assertRaises(ValueError):
                    self._load(fixture)

    def test_payload_mutation_after_contract_creation_fails_hash_or_size(self):
        contract = self.fixture.write()
        stage_id = "exd_hox_training_stage_" + _stage_manifest(contract)["manifest_hash"]
        for path in (LOGICAL_PATH, ORDERING_PATH):
            original = (self.root / path).read_bytes()
            (self.root / path).write_bytes(original + b"x")
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    dataset_module._read_public_tf_data(
                        self.root, contract, "Ubx", stage_id,
                    )
            (self.root / path).write_bytes(original)

    def test_same_size_payload_corruption_fails_sha256_before_parsing(self):
        contract = self.fixture.write()
        stage_id = "exd_hox_training_stage_" + _stage_manifest(contract)["manifest_hash"]
        path = self.root / LOGICAL_PATH
        original = path.read_bytes()
        changed = bytearray(original)
        changed[-1] ^= 1
        path.write_bytes(changed)
        self.assertEqual(path.stat().st_size, len(original))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            dataset_module._read_public_tf_data(self.root, contract, "Ubx", stage_id)

    def test_source_state_change_during_parsing_is_detected(self):
        contract = self.fixture.write()
        record = next(
            record for record in contract["transferred_public_payloads"]
            if record["path"] == LOGICAL_PATH
        )
        rows = dataset_module._tsv_rows(self.root, record, LOGICAL_FIELDS)
        self.addCleanup(rows.close)
        self.assertEqual(len(next(rows)), len(LOGICAL_FIELDS))
        path = self.root / LOGICAL_PATH
        previous = path.stat()
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1000000))
        with self.assertRaisesRegex(ValueError, "changed during parsing"):
            tuple(rows)

    def test_other_tf_ordering_duplicate_ids_are_rejected_while_selecting_ubx(self):
        first = _ordering_row(_logical_row(_sequence(1100), tf="AbdA"), 1)
        second = _ordering_row(_logical_row(_sequence(1101), tf="AbdA"), 2)
        ordered = [first, second]
        path = self.root / ORDERING_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        contract = {
            "transcription_factors": ["AbdA", "Ubx"],
            "tracked_metadata": [],
            "transferred_public_payloads": [],
            "expected_counts": {
                "ordering_logical_examples": 2,
                "per_tf": {
                    "AbdA": {"training": {"logical_examples": 2, "rc_groups": 2}},
                    "Ubx": {"training": {"logical_examples": 0, "rc_groups": 0}},
                },
            },
        }

        def write_ordering():
            value = _gzip_bytes(_table_bytes(ORDERING_FIELDS, ordered))
            path.write_bytes(value)
            contract["transferred_public_payloads"] = [{
                "path": ORDERING_PATH, "byte_size": len(value), "sha256": _digest(value),
            }]

        write_ordering()
        selected, cumulative = dataset_module._scan_ordering(self.root, contract, "Ubx", {})
        self.assertEqual(selected, ())
        self.assertEqual(cumulative["AbdA"], {0: 0, 1: 1, 2: 2})
        original = dict(second)
        second["logical_example_id"] = first["logical_example_id"]
        self.assertEqual(
            {key: value for key, value in second.items() if key != "logical_example_id"},
            {key: value for key, value in original.items() if key != "logical_example_id"},
        )
        write_ordering()
        with self.assertRaisesRegex(ValueError, "Duplicate ordering logical ID"):
            dataset_module._scan_ordering(self.root, contract, "Ubx", {})

    def test_load_contract_rejects_bytes_outside_production_pin(self):
        with patch.object(dataset_module, "_read_small", return_value=b"{}\n"):
            with self.assertRaisesRegex(ValueError, "config byte identity differs"):
                dataset_module._load_contract()

    def test_strict_json_rejects_duplicate_keys_nonfinite_values_and_wrong_root(self):
        self.assertEqual(dataset_module._strict_json(b'{"value":1}'), {"value": 1})
        for value in (b'{"value":1,"value":2}', b'{"value":NaN}', b'[]', b'\xff'):
            with self.subTest(payload=value):
                with self.assertRaises(ValueError):
                    dataset_module._strict_json(value)

    def _stage(self):
        contract = self.fixture.write()
        manifest = _stage_manifest(contract)
        (self.root / "exd_hox_training_stage_manifest_v1.json").write_bytes(
            _json_bytes(manifest) + b"\n",
        )
        return contract, "exd_hox_training_stage_" + manifest["manifest_hash"]

    def _assert_stage_field_rejected(self, field, value, message, *, remove=False, rehash=True):
        """Change one field, preserving the envelope for the intended rejection."""
        contract, stage_id = self._stage()
        original = _stage_manifest(contract)
        changed = copy.deepcopy(original)
        if remove:
            del changed[field]
        else:
            changed[field] = value
        content = dict(changed)
        content.pop("manifest_hash")
        if rehash:
            changed["manifest_hash"] = _digest(_json_bytes(content))
            dataset_module._semantic_manifest(changed, dataset_module.STAGE_SCHEMA)
        else:
            self.assertEqual(changed["manifest_hash"], original["manifest_hash"])
            self.assertNotEqual(_digest(_json_bytes(content)), changed["manifest_hash"])
        self.assertEqual(
            {key: item for key, item in changed.items() if key not in (field, "manifest_hash")},
            {key: item for key, item in original.items() if key not in (field, "manifest_hash")},
        )
        manifest_path = self.root / dataset_module.STAGE_MANIFEST_FILENAME
        manifest_path.write_bytes(_json_bytes(changed) + b"\n")
        with patch.object(dataset_module, "_open_regular", wraps=dataset_module._open_regular) as opened:
            with self.assertRaisesRegex(dataset_module.ExdHoxDatasetError, message):
                dataset_module._verify_stage(self.root, contract, stage_id)
        # A self-declared path must never be opened, even when its hash is valid.
        self.assertEqual(
            [entry.args for entry in opened.call_args_list],
            [(self.root, dataset_module.STAGE_MANIFEST_FILENAME)],
        )

    def test_stage_manifest_config_provenance_participates_in_hash_and_stage_id(self):
        contract, stage_id = self._stage()
        manifest = dataset_module._build_stage_manifest(contract)
        self.assertEqual(manifest, _stage_manifest(contract))
        self.assertEqual(set(manifest), {
            "schema_version", "purpose", "staging_policy_identifier", "foundation_commit",
            "dataset_identifier", "split_identity_hash", "split_manifest_hash",
            "subset_set_manifest_hash", "allowed_dataset_selections",
            "staging_config_path", "staging_config_sha256", "files", "manifest_hash",
        })
        self.assertEqual(
            manifest["staging_config_path"], "configs/carc_exd_hox_training_staging_v1.json",
        )
        self.assertEqual(manifest["staging_config_sha256"], _digest(_json_bytes(contract) + b"\n"))
        content = dict(manifest)
        content.pop("manifest_hash")
        corrected_hash = _digest(_json_bytes(content))
        self.assertEqual(manifest["manifest_hash"], corrected_hash)
        self.assertEqual(dataset_module._stage_id(manifest), "exd_hox_training_stage_" + corrected_hash)
        self.assertEqual(stage_id, dataset_module._stage_id(manifest))
        for field, replacement in (
            ("staging_config_path", "configs/another_staging_config.json"),
            ("staging_config_sha256", _digest(b"different config bytes\n")),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(content)
                changed[field] = replacement
                self.assertNotEqual(_digest(_json_bytes(changed)), corrected_hash)
                del changed[field]
                self.assertNotEqual(_digest(_json_bytes(changed)), corrected_hash)
        paths = [record["path"] for record in manifest["files"]]
        self.assertEqual(len(paths), 9)
        self.assertNotIn(manifest["staging_config_path"], paths)
        self.assertNotIn(dataset_module.STAGE_MANIFEST_FILENAME, paths)
        dataset_module._verify_stage(self.root, contract, stage_id)

    def test_stage_manifest_requires_config_path_and_rejects_unknown_field(self):
        for field, value, remove in (
            ("staging_config_path", None, True),
            ("unknown_staging_config_path", "configs/another_staging_config.json", False),
        ):
            with self.subTest(field=field):
                self._assert_stage_field_rejected(
                    field, value, "Stage manifest fields differ", remove=remove,
                )

    def test_stage_manifest_rejects_rehashed_untrusted_config_path(self):
        self._assert_stage_field_rejected(
            "staging_config_path", "configs/another_staging_config.json",
            "staging_config_path differs from trusted CONFIG_PATH",
        )

    def test_stage_manifest_rejects_rehashed_nonstring_config_path(self):
        for value in (None, True, 7, [], {}):
            with self.subTest(value=value):
                self._assert_stage_field_rejected(
                    "staging_config_path", value, "Logical path must be a string",
                )

    def test_stage_manifest_rejects_rehashed_noncanonical_config_path(self):
        canonical = "configs/carc_exd_hox_training_staging_v1.json"
        cases = (
            ("/" + canonical, "Noncanonical or escaping logical path"),
            ("../" + canonical, "Noncanonical or escaping logical path"),
            ("configs/../" + canonical, "Noncanonical or escaping logical path"),
            ("./" + canonical, "Noncanonical or escaping logical path"),
            (canonical.replace("configs/", "configs/./"), "Noncanonical or escaping logical path"),
            (canonical.replace("configs/", "configs//"), "Noncanonical or escaping logical path"),
            (canonical + "/", "Noncanonical or escaping logical path"),
            (canonical.replace("/", "\\"), "Invalid logical path characters"),
            (canonical + "\n", "Invalid logical path characters"),
        )
        for value, message in cases:
            with self.subTest(value=value):
                self._assert_stage_field_rejected("staging_config_path", value, message)

    def test_stage_manifest_config_path_change_without_rehash_fails_integrity(self):
        self._assert_stage_field_rejected(
            "staging_config_path", "configs/another_staging_config.json",
            "Semantic manifest hash mismatch", rehash=False,
        )

    def test_stage_manifest_config_sha_remains_bound_to_trusted_bytes(self):
        self._assert_stage_field_rejected(
            "staging_config_sha256", _digest(b"different config bytes\n"),
            "staging_config_sha256 differs from trusted configuration bytes",
        )

    def test_public_constructor_requires_exact_stage_manifest_and_id(self):
        contract, stage_id = self._stage()
        with patch.object(dataset_module, "_load_contract", return_value=contract):
            data = dataset_module.open_public_tf_data(
                self.root, transcription_factor="Ubx", expected_stage_id=stage_id,
            )
            self.assertEqual(len(data.dataset("validation")), 2)
            with self.assertRaises(ValueError):
                dataset_module.open_public_tf_data(
                    self.root, transcription_factor="Ubx",
                    expected_stage_id="exd_hox_training_stage_" + "f" * 64,
                )
            (self.root / "exd_hox_training_stage_manifest_v1.json").write_bytes(b"{}\n")
            with self.assertRaises(ValueError):
                dataset_module.open_public_tf_data(
                    self.root, transcription_factor="Ubx", expected_stage_id=stage_id,
                )

    def test_public_constructor_rejects_pooled_and_unknown_tf(self):
        contract, stage_id = self._stage()
        with patch.object(dataset_module, "_load_contract", return_value=contract):
            for factor in ("all", "pooled", "Ubx,AbdA", ["Ubx"], ("Ubx", "AbdA"), None):
                with self.subTest(factor=factor):
                    with self.assertRaises((TypeError, ValueError)):
                        dataset_module.open_public_tf_data(
                            self.root, transcription_factor=factor, expected_stage_id=stage_id,
                        )

    def test_open_boundaries_never_follow_inert_sealed_or_raw_references(self):
        self.fixture.metadata_changes[SPLIT_MANIFEST_PATH] = {
            "inert_provenance": {
                "sealed_target_path": "data/sealed/forbidden.tsv.gz",
                "source_path": "data/raw/forbidden.h5",
            },
        }
        contract, stage_id = self._stage()
        observed = []

        def guard(original):
            def checked(path, *args, **kwargs):
                if not isinstance(path, int):
                    text = os.fsdecode(path)
                    observed.append(text)
                    self.assertNotIn("sealed", text)
                    self.assertNotIn("forbidden", text)
                    self.assertFalse(text.endswith((".h5", ".hdf5")))
                return original(path, *args, **kwargs)
            return checked

        with patch.object(dataset_module, "_load_contract", return_value=contract):
            with patch("builtins.open", guard(builtins.open)):
                with patch("io.open", guard(io.open)):
                    with patch("os.open", guard(os.open)):
                        data = dataset_module.open_public_tf_data(
                            self.root, transcription_factor="Ubx", expected_stage_id=stage_id,
                        )
                        self.assertEqual(len(self._training(data)), 128)
        self.assertTrue(observed)
        self.assertFalse((self.root / "data/sealed").exists())


class ImportIsolationTests(unittest.TestCase):
    def test_source_has_no_forbidden_imports(self):
        source = Path(dataset_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
                if node.module == "src":
                    names.extend("src." + alias.name for alias in node.names)
        self.assertNotIn("src.sealed_test_access", names)
        self.assertNotIn("src.exd_hox_splits", names)

    def test_fresh_interpreter_denies_direct_and_transitive_sealed_imports(self):
        script = """
import builtins
import sys
blocked = ('src.sealed_test_access', 'src.exd_hox_splits')
original = builtins.__import__
def checked(name, globals=None, locals=None, fromlist=(), level=0):
    if name in blocked:
        raise AssertionError('Forbidden import: ' + name)
    if name == 'src':
        for entry in fromlist:
            if 'src.' + entry in blocked:
                raise AssertionError('Forbidden from-import: ' + entry)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = checked
import src.exd_hox_dataset
assert all(name not in sys.modules for name in blocked)
"""
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1], env=environment,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
