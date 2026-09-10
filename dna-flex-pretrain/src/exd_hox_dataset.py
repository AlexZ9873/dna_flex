"""Pinned public Exd-Hox membership; no test selection or split construction.

The public constructor requires an immutable verified stage. Private readers
also support bounded read-only acceptance of the nine original public inputs.
Metadata references outside that inventory are inert and never followed.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
from typing import Any, BinaryIO, Iterator, Sequence, TYPE_CHECKING
import zlib

from src.downstream_fingerprints import canonical_json_bytes, hash_logical_content

if TYPE_CHECKING:
    import torch


CONFIG_PATH = "configs/carc_exd_hox_training_staging_v1.json"
CONFIG_SHA256 = "50a7ba5ae1bd7219b20cdc71b511dfbce8b38855a1b065ac4777eca71a18833b"
STAGE_MANIFEST_FILENAME = "exd_hox_training_stage_manifest_v1.json"
CONFIG_SCHEMA = "carc_exd_hox_training_staging_config.v1"
STAGE_SCHEMA = "exd_hox_training_stage_manifest.v1"
TF_NAMES = ("AbdA", "AbdB", "Antp", "Dfd", "Lab", "Pb", "Scr", "Ubx")
SPLITS = ("training", "validation", "test")
PRIMARY_CONFIG = "configs/exd_hox_primary_split_v1.yaml"
ACCESS_POLICY = "configs/exd_hox_test_access_policy_v1.yaml"
SPLIT_DIRECTORY = "data/processed/exd_hox_primary_split_v1/"
SUBSET_DIRECTORY = "data/processed/exd_hox_nested_subsets_v1/"
AUDIT_DIRECTORY = "data/processed/exd_hox_selex_audit_v1/"
SPLIT_MANIFEST = SPLIT_DIRECTORY + "exd_hox_primary_split_manifest_v1.json"
SUBSET_MANIFEST = SUBSET_DIRECTORY + "exd_hox_subset_set_manifest_v1.json"
LEVELS_PATH = SUBSET_DIRECTORY + "exd_hox_nested_subset_levels_v1.tsv"
SOURCE_MANIFEST = AUDIT_DIRECTORY + "exd_hox_source_manifest_v1.json"
AUDIT_MANIFEST = AUDIT_DIRECTORY + "exd_hox_audit_manifest_v1.json"
LOGICAL_PATH = SPLIT_DIRECTORY + "exd_hox_logical_examples_v1.tsv.gz"
ORDERING_PATH = SUBSET_DIRECTORY + "exd_hox_nested_subset_ordering_v1.tsv.gz"
METADATA_PATHS = tuple(sorted((PRIMARY_CONFIG, ACCESS_POLICY, SPLIT_MANIFEST,
                              SUBSET_MANIFEST, LEVELS_PATH, SOURCE_MANIFEST,
                              AUDIT_MANIFEST)))
PAYLOAD_PATHS = tuple(sorted((LOGICAL_PATH, ORDERING_PATH)))
LOGICAL_FIELDS = (
    "logical_example_id", "transcription_factor", "sequence", "sequence_sha256",
    "reverse_complement_canonical_sequence", "reverse_complement_canonical_sha256",
    "global_rc_group_id", "primary_split", "target_value_float32",
    "target_bits_big_endian_hex", "target_commitment_sha256", "source_occurrence_count",
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
_FORBIDDEN = (
    "data/sealed/**", "data/raw/**", "**/*.h5", "**/*.hdf5",
    "**/SELEX_canonical/**", "**/SELEX_RCmodel/**",
    "**/exd_hox_public_test_inputs_v1.tsv.gz",
    "**/exd_hox_source_occurrence_provenance_v1.tsv.gz",
    "**/exd_hox_global_rc_groups_v1.tsv.gz",
    "**/exd_hox_primary_split_assignments_v1.tsv.gz",
    "**/exd_hox_sealed_test_targets_v1.tsv.gz",
    "**/exd_hox_sealed_test_target_manifest_v1.json",
    "results/**", "plots/**", "checkpoints/**", "**/authorization*",
    "**/test_access_record*",
)
FORBIDDEN_PATH_CLASSES = _FORBIDDEN


class ExdHoxDatasetError(ValueError):
    """Public data or staging input violates the accepted contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExdHoxDatasetError(message)


def _exact_keys(value: Any, keys: Sequence[str], description: str) -> None:
    _require(type(value) is dict and set(value) == set(keys), description + " fields differ.")


def _hex(value: Any, length: int = 64) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _strict_json(data: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key.")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ExdHoxDatasetError("Nonfinite JSON number.")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExdHoxDatasetError("Malformed JSON metadata.") from error
    _require(type(value) is dict, "JSON metadata must be an object.")
    return value


def _logical_path(value: Any) -> str:
    _require(type(value) is str and bool(value), "Logical path must be a string.")
    parts = value.split("/")
    _require(not value.startswith("/") and all(part not in ("", ".", "..") for part in parts),
             "Noncanonical or escaping logical path.")
    _require("\\" not in value and all(ord(character) >= 32 for character in value),
             "Invalid logical path characters.")
    return value


def _records(contract: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(sorted(contract["tracked_metadata"] + contract["transferred_public_payloads"],
                        key=lambda record: record["path"]))


def _validate_contract(contract: dict[str, Any]) -> None:
    """Validate schema independently of the public constructor's byte pin."""
    _exact_keys(contract, (
        "schema_version", "purpose", "staging_policy_identifier", "foundation_commit",
        "dataset_identifier", "split_identity_hash", "split_manifest_hash",
        "subset_set_manifest_hash", "allowed_dataset_selections", "transcription_factors",
        "tracked_metadata", "transferred_public_payloads", "expected_counts",
        "forbidden_path_classes",
    ), "Staging config")
    _require(contract["schema_version"] == CONFIG_SCHEMA, "Unsupported staging config schema.")
    _require(contract["purpose"] == "training_validation_only", "Invalid staging purpose.")
    _require(contract["staging_policy_identifier"] == "exd_hox_public_training_stage.v1",
             "Invalid staging policy.")
    _require(contract["dataset_identifier"] == "wang_etal_exd_hox_selex_canonical.v1",
             "Invalid dataset identifier.")
    _require(_hex(contract["foundation_commit"], 40), "Invalid foundation commit.")
    for key in ("split_identity_hash", "split_manifest_hash", "subset_set_manifest_hash"):
        _require(_hex(contract[key]), "Invalid scientific identity hash.")
    _require(contract["allowed_dataset_selections"] == ["training", "validation"],
             "Invalid allowed dataset selections.")
    factors = contract["transcription_factors"]
    _require(type(factors) is list and bool(factors)
             and all(type(tf) is str and tf in TF_NAMES for tf in factors), "Invalid TF list.")
    _require(factors == sorted(set(factors)), "TF list must be unique and sorted.")
    _require(contract["forbidden_path_classes"] == list(_FORBIDDEN), "Forbidden path classes differ.")
    for key, expected_paths in (("tracked_metadata", METADATA_PATHS),
                                ("transferred_public_payloads", PAYLOAD_PATHS)):
        records = contract[key]
        _require(type(records) is list, "File records must be a list.")
        paths = []
        for record in records:
            _exact_keys(record, ("path", "byte_size", "sha256"), "File record")
            paths.append(_logical_path(record["path"]))
            _require(_integer(record["byte_size"], 1) and _hex(record["sha256"]),
                     "Invalid file fingerprint.")
        _require(tuple(paths) == expected_paths, "Exact input path inventory differs.")
    counts = contract["expected_counts"]
    _exact_keys(counts, ("source_occurrences", "logical_examples", "ordering_logical_examples",
                        "level_rows", "split_logical_examples", "global_rc_groups", "per_tf"),
                "Expected counts")
    for key in ("source_occurrences", "logical_examples", "ordering_logical_examples", "level_rows"):
        _require(_integer(counts[key], 1), "Invalid expected count.")
    _exact_keys(counts["per_tf"], factors, "Per-TF counts")
    for key in ("split_logical_examples", "global_rc_groups"):
        _exact_keys(counts[key], SPLITS, "Split counts")
        _require(all(_integer(value) for value in counts[key].values()), "Invalid split count.")
    for tf in factors:
        _exact_keys(counts["per_tf"][tf], SPLITS, "TF split counts")
        for split in SPLITS:
            item = counts["per_tf"][tf][split]
            _exact_keys(item, ("logical_examples", "rc_groups"), "TF membership counts")
            _require(all(_integer(value) for value in item.values()), "Invalid membership count.")
            _require(item["rc_groups"] <= item["logical_examples"], "Group count exceeds rows.")
    for split in SPLITS:
        total = sum(counts["per_tf"][tf][split]["logical_examples"] for tf in factors)
        _require(total == counts["split_logical_examples"][split], "Per-TF totals differ.")
    _require(sum(counts["split_logical_examples"].values()) == counts["logical_examples"],
             "Logical total differs.")
    _require(counts["source_occurrences"] >= counts["logical_examples"], "Source total differs.")
    _require(counts["ordering_logical_examples"] == counts["split_logical_examples"]["training"],
             "Training ordering count differs.")


@contextmanager
def _root_directory(root: Path) -> Iterator[int]:
    """Anchor traversal at an absolute directory, rejecting every symlink."""
    absolute = Path(os.path.abspath(os.fspath(root)))
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _open_regular(root: Path, logical_path: str) -> Iterator[BinaryIO]:
    """Open a confined regular file; contexts are necessary for descriptor cleanup."""
    parts = _logical_path(logical_path).split("/")
    with _root_directory(root) as root_descriptor:
        parent = os.dup(root_descriptor)
        try:
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = child
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                _require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Input is not a regular file.")
            except BaseException:
                os.close(descriptor)
                raise
            with os.fdopen(descriptor, "rb") as handle:
                yield handle
        finally:
            os.close(parent)


def _file_state(handle: BinaryIO) -> tuple[int, ...]:
    value = os.fstat(handle.fileno())
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _require_inventory(root: Path, allowed_files: Sequence[str]) -> None:
    allowed = set(_logical_path(path) for path in allowed_files)
    _require(len(allowed) == len(allowed_files), "Duplicate inventory path.")
    directories = set()
    for name in allowed:
        for parent in PurePosixPath(name).parents:
            if str(parent) != ".":
                directories.add(str(parent))
    observed = set()

    def visit(descriptor: int, prefix: str) -> None:
        for name in sorted(os.listdir(descriptor)):
            logical = prefix + name
            details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(details.st_mode):
                _require(logical in directories, "Unexpected input directory.")
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                try:
                    visit(child, logical + "/")
                finally:
                    os.close(child)
            else:
                _require(stat.S_ISREG(details.st_mode) and logical in allowed,
                         "Unexpected, symlink, or nonregular input.")
                observed.add(logical)

    with _root_directory(root) as descriptor:
        visit(descriptor, "")
    _require(observed == allowed, "Missing input file.")


def _verify_records(root: Path, records: Sequence[dict[str, Any]]) -> None:
    for record in records:
        with _open_regular(root, record["path"]) as handle:
            before = _file_state(handle)
            _require(before[2] == record["byte_size"], "Input byte size mismatch.")
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            _require(_file_state(handle) == before, "Input changed during fingerprinting.")
            _require(digest.hexdigest() == record["sha256"], "Input SHA-256 mismatch.")
            with _open_regular(root, record["path"]) as current:
                _require(_file_state(current) == before, "Input replaced during fingerprinting.")


def _read_small(root: Path, logical_path: str) -> bytes:
    with _open_regular(root, logical_path) as handle:
        data = handle.read(2 * 1024 * 1024 + 1)
    _require(len(data) <= 2 * 1024 * 1024, "Metadata size limit exceeded.")
    return data


def _load_contract() -> dict[str, Any]:
    data = _read_small(Path(__file__).resolve().parents[1], CONFIG_PATH)
    _require(hashlib.sha256(data).hexdigest() == CONFIG_SHA256, "Installed staging config byte identity differs.")
    contract = _strict_json(data)
    _validate_contract(contract)
    _require(data == canonical_json_bytes(contract) + b"\n", "Staging config is not canonical JSON.")
    return contract


def _build_stage_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    _validate_contract(contract)
    result = {"schema_version": STAGE_SCHEMA}
    for key in ("purpose", "staging_policy_identifier", "foundation_commit", "dataset_identifier",
                "split_identity_hash", "split_manifest_hash", "subset_set_manifest_hash",
                "allowed_dataset_selections"):
        result[key] = contract[key]
    result["staging_config_path"] = CONFIG_PATH
    result["staging_config_sha256"] = hashlib.sha256(canonical_json_bytes(contract) + b"\n").hexdigest()
    result["files"] = list(_records(contract))
    result["manifest_hash"] = hash_logical_content(result)
    return result


def _stage_id(manifest: dict[str, Any]) -> str:
    return "exd_hox_training_stage_" + manifest["manifest_hash"]


def _verify_stage(root: Path, contract: dict[str, Any], expected_stage_id: str) -> None:
    expected = _build_stage_manifest(contract)
    _require(type(expected_stage_id) is str and expected_stage_id == _stage_id(expected),
             "Expected stage identity differs from pinned contract.")
    _require_inventory(root, [record["path"] for record in _records(contract)] + [STAGE_MANIFEST_FILENAME])
    data = _read_small(root, STAGE_MANIFEST_FILENAME)
    actual = _strict_json(data)
    _exact_keys(actual, tuple(expected), "Stage manifest")
    _semantic_manifest(actual, STAGE_SCHEMA)
    _require(_logical_path(actual["staging_config_path"]) == CONFIG_PATH,
             "Stage staging_config_path differs from trusted CONFIG_PATH.")
    _require(actual["staging_config_sha256"] == expected["staging_config_sha256"],
             "Stage staging_config_sha256 differs from trusted configuration bytes.")
    _require(data == canonical_json_bytes(expected) + b"\n" and actual == expected,
             "Stage manifest identity or canonical bytes differ.")
    _verify_records(root, _records(contract))


def _semantic_manifest(value: dict[str, Any], schema: str) -> None:
    _require(value.get("schema_version") == schema, "Metadata schema differs.")
    content = dict(value)
    stored = content.pop("manifest_hash", None)
    _require(_hex(stored) and hash_logical_content(content) == stored,
             "Semantic manifest hash mismatch.")


def _yaml_metadata(data: bytes) -> dict[str, Any]:
    import yaml

    class UniqueSafeLoader(yaml.SafeLoader):
        pass

    def mapping(loader: Any, node: Any) -> dict[str, Any]:
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            _require(isinstance(key, (str, int)) and key not in result,
                     "Invalid or duplicate YAML key.")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        result = yaml.load(data.decode("utf-8"), Loader=UniqueSafeLoader)
    except (UnicodeError, yaml.YAMLError) as error:
        raise ExdHoxDatasetError("Malformed YAML metadata.") from error
    _require(type(result) is dict, "YAML metadata must be a mapping.")
    return result


def _read_metadata(root: Path, contract: dict[str, Any]) -> None:
    """Validate only the accepted public dependency edges; never follow others."""
    records = {record["path"]: record for record in _records(contract)}
    metadata = {}
    for path in METADATA_PATHS:
        if path != LEVELS_PATH:
            data = _read_small(root, path)
            record = records[path]
            _require(len(data) == record["byte_size"] and hashlib.sha256(data).hexdigest() == record["sha256"],
                     "Metadata byte identity differs.")
            metadata[path] = _yaml_metadata(data) if path.endswith(".yaml") else _strict_json(data)
    split = metadata[SPLIT_MANIFEST]
    subset = metadata[SUBSET_MANIFEST]
    source = metadata[SOURCE_MANIFEST]
    audit = metadata[AUDIT_MANIFEST]
    policy = metadata[ACCESS_POLICY]
    config = metadata[PRIMARY_CONFIG]
    for value, schema in ((split, "exd_hox_primary_split_manifest.v1"),
                          (subset, "exd_hox_subset_set_manifest.v1"),
                          (source, "exd_hox_source_manifest.v1"),
                          (audit, "exd_hox_audit_manifest.v1"),
                          (policy, "exd_hox_test_access_policy.v1")):
        _semantic_manifest(value, schema)
    _require(config.get("schema_version") == "exd_hox_primary_split_config.v1", "Primary config schema differs.")
    _require(policy.get("authorized_operation") == "final_primary_test_evaluation", "Test policy operation differs.")
    _require(source.get("source_commit") == "9e6d6ef0355558c98855b83a9c21fe11999f65d9", "External source commit differs.")
    for value in (split, subset, source, audit):
        _require(value.get("dataset_identifier") == contract["dataset_identifier"], "Metadata dataset identity differs.")
    _require(split["manifest_hash"] == contract["split_manifest_hash"], "Accepted split manifest identity differs.")
    _require(subset["manifest_hash"] == contract["subset_set_manifest_hash"], "Accepted subset-set identity differs.")
    for value in (split, subset):
        _require(value.get("split_identity_hash") == contract["split_identity_hash"], "Split identity differs.")
        _require(value.get("config_path") == PRIMARY_CONFIG
                 and value.get("config_sha256") == records[PRIMARY_CONFIG]["sha256"], "Config fingerprint link differs.")
    _require(subset.get("split_manifest_hash") == split["manifest_hash"], "Subset split manifest link differs.")
    _require(split.get("source_manifest_hash") == source["manifest_hash"]
             and audit.get("source_manifest_hash") == source["manifest_hash"]
             and split.get("audit_manifest_hash") == audit["manifest_hash"], "Source/audit manifest links differ.")
    _require(split.get("test_access_policy_manifest_hash") == policy["manifest_hash"], "Policy manifest link differs.")
    try:
        _require(config["dataset"]["sequence_length"] == 14
                 and type(config["dataset"]["sequence_length"]) is int
                 and config["dataset"]["transcription_factors"] == contract["transcription_factors"],
                 "Dataset configuration differs.")
        for config_key, value, seed in (("split_policy", split, 31001), ("subset_policy", subset, 32001)):
            _require(type(config[config_key]["seed"]) is int and config[config_key]["seed"] == seed,
                     "Accepted seed differs.")
            _require(canonical_json_bytes(value["policy"]) == canonical_json_bytes(config[config_key]),
                     "Policy configuration differs.")
        for name, path, manifest in (("source", SOURCE_MANIFEST, source), ("audit", AUDIT_MANIFEST, audit)):
            _require(config["inputs"][name + "_manifest_hash"] == manifest["manifest_hash"]
                     and config["inputs"][name + "_manifest_file_sha256"] == records[path]["sha256"],
                     "Config source/audit links differ.")
        _require(config["test_access"]["policy_manifest_hash"] == policy["manifest_hash"]
                 and config["test_access"]["policy_file_sha256"] == records[ACCESS_POLICY]["sha256"],
                 "Config test policy link differs.")
        expected = contract["expected_counts"]
        for key in ("source_occurrences", "logical_examples"):
            _require(type(split["counts"][key]) is int and split["counts"][key] == expected[key], "Manifest count differs.")
        _require(split["counts"]["global_rc_group_counts"] == expected["global_rc_groups"], "Global group counts differ.")
        for tf in contract["transcription_factors"]:
            for selection in SPLITS:
                observed = split["counts"]["per_tf_split_counts"][tf][selection]
                _require(type(observed) is int and observed == expected["per_tf"][tf][selection]["logical_examples"],
                         "TF manifest count differs.")
        _require(audit["totals"]["total_row_occurrences"] == expected["source_occurrences"], "Audit source total differs.")
        _require(subset["level_row_count"] == expected["level_rows"]
                 and subset["ordering_logical_example_count"] == expected["ordering_logical_examples"],
                 "Subset manifest counts differ.")
        for value, paths in ((split, (LOGICAL_PATH,)), (subset, (LEVELS_PATH, ORDERING_PATH))):
            entries = value["artifacts"]
            _require(type(entries) is list, "Artifact records must be a list.")
            for path in paths:
                matches = [entry for entry in entries if type(entry) is dict and entry.get("path") == path]
                _require(len(matches) == 1 and canonical_json_bytes(matches[0]) == canonical_json_bytes(records[path]),
                         "Public artifact fingerprint link differs.")
    except (KeyError, TypeError) as error:
        raise ExdHoxDatasetError("Missing or malformed public metadata field.") from error


def _tsv_rows(root: Path, record: dict[str, Any], fields: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    """Read exact unquoted v1 TSV rows from the same descriptor whose bytes pass."""
    try:
        with _open_regular(root, record["path"]) as handle:
            before = _file_state(handle)
            _require(before[2] == record["byte_size"], "Table byte size mismatch.")
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            _require(digest.hexdigest() == record["sha256"], "Table SHA-256 mismatch.")
            handle.seek(0)
            stream = gzip.GzipFile(fileobj=handle, mode="rb") if record["path"].endswith(".gz") else handle
            with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
                _require(text.readline() == "\t".join(fields) + "\n", "TSV header or field order differs.")
                for line in text:
                    _require(line.endswith("\n") and "\r" not in line, "TSV line ending differs.")
                    row = tuple(line[:-1].split("\t"))
                    _require(len(row) == len(fields), "TSV row field count differs.")
                    yield row
                # Plain text wrappers would close handle, so compare before exit.
                _require(_file_state(handle) == before, "Table changed during parsing.")
            with _open_regular(root, record["path"]) as current:
                _require(_file_state(current) == before, "Table replaced during parsing.")
    except (UnicodeError, EOFError, gzip.BadGzipFile, zlib.error) as error:
        raise ExdHoxDatasetError("Malformed public TSV/gzip.") from error


def _domain_hash(domain: str, *parts: str) -> str:
    digest = hashlib.sha256(domain.encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def _text_integer(text: str, minimum: int = 0) -> int:
    _require(re.fullmatch("0|[1-9][0-9]*", text) is not None, "Noncanonical integer field.")
    value = int(text)
    _require(value >= minimum, "Integer field below minimum.")
    return value


def _sequence(sequence: Any) -> str:
    _require(type(sequence) is str and re.fullmatch("[ACGT]{14}", sequence) is not None,
             "Sequence must be exactly 14 uppercase A/C/G/T bases.")
    return sequence


def _target(bits: str, text: str) -> float:
    _require(_hex(bits, 8), "Target bits must contain eight lowercase hex characters.")
    value = struct.unpack(">f", bytes.fromhex(bits))[0]
    _require(math.isfinite(value) and 0 <= value <= 1, "Target must be finite in [0,1].")
    _require(format(value, ".9g") == text, "Target decimal is not canonical for float32 bits.")
    return value


@dataclass(frozen=True, slots=True)
class _PublicRow:
    logical_example_id: str
    transcription_factor: str
    sequence: str
    global_rc_group_id: str
    primary_split: str
    target_value_float32: float
    target_bits_big_endian_hex: str


@dataclass(frozen=True, slots=True)
class LevelSpec:
    transcription_factor: str
    level_id: str
    request_type: str
    request_value: str
    unaliased_requested_logical_example_count: int
    alias_absolute_anchor: int | None
    canonical_requested_logical_example_count: int
    actual_logical_example_count: int
    actual_rc_group_count: int
    inclusive_maximum_rank: int
    canonical_level_id: str


@dataclass(frozen=True, slots=True)
class SampleMetadata:
    logical_example_id: str
    transcription_factor: str
    sequence: str
    global_rc_group_id: str
    primary_split: str
    target_value_float32: float
    target_bits_big_endian_hex: str
    stage_id: str
    split_identity_hash: str
    split_manifest_hash: str
    subset_set_manifest_hash: str
    level_id: str | None
    canonical_level_id: str | None
    validation_id: str | None
    rank_one_based: int | None
    sample_rank_one_based: int


@dataclass(frozen=True, slots=True)
class ExdHoxSample:
    x: torch.Tensor
    y: torch.Tensor
    metadata: SampleMetadata


@dataclass(frozen=True, slots=True)
class ExdHoxBatch:
    x: torch.Tensor
    y: torch.Tensor
    metadata: tuple[SampleMetadata, ...]


def encode_one_hot_14(sequence: str) -> torch.Tensor:
    """Construct exact CPU float32 A,C,G,T channels without normalization."""
    import torch

    _sequence(sequence)
    result = torch.zeros((14, 4), dtype=torch.float32, device="cpu")
    for position, base in enumerate(sequence):
        result[position, "ACGT".index(base)] = 1
    return result


@dataclass(frozen=True, slots=True, init=False)
class ExdHoxDataset:
    """Immutable materialized membership; no open files or sampler state."""

    _metadata: tuple[SampleMetadata, ...]
    unique_rc_group_count: int

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, index: int) -> ExdHoxSample:
        import torch

        _require(type(index) is int, "Dataset index must be an integer.")
        metadata = self._metadata[index]
        return ExdHoxSample(encode_one_hot_14(metadata.sequence),
                           torch.tensor([metadata.target_value_float32], dtype=torch.float32, device="cpu"),
                           metadata)


@dataclass(frozen=True, slots=True, init=False)
class PublicTFData:
    """Validated one-TF handle reusable across accepted levels in one process."""

    transcription_factor: str
    stage_id: str
    _identities: tuple[str, str, str]
    _training: tuple[tuple[_PublicRow, int], ...]
    _validation: tuple[_PublicRow, ...]
    _levels: tuple[LevelSpec, ...]

    def levels(self) -> tuple[LevelSpec, ...]:
        return self._levels

    def dataset(self, selection: str, *, level_id: str | None = None) -> ExdHoxDataset:
        _require(type(selection) is str and selection in ("training", "validation"),
                 "Only training and validation selections are supported.")
        level = None
        validation_id = None
        if selection == "training":
            _require(type(level_id) is str, "Training requires an accepted level ID.")
            matches = [item for item in self._levels if item.level_id == level_id]
            _require(len(matches) == 1, "Unknown or wrong-TF level ID.")
            level = matches[0]
            rows = self._training[:level.actual_logical_example_count]
        else:
            _require(level_id is None, "Validation always uses the full fixed membership.")
            validation_id = "validation_" + _domain_hash(
                "exd_hox_fixed_validation.v1", self._identities[0], self.transcription_factor)
            rows = tuple((row, None) for row in self._validation)
        metadata = []
        for sample_rank, (row, rank) in enumerate(rows, 1):
            metadata.append(SampleMetadata(
                row.logical_example_id, row.transcription_factor, row.sequence,
                row.global_rc_group_id, row.primary_split, row.target_value_float32,
                row.target_bits_big_endian_hex, self.stage_id, *self._identities,
                level_id, None if level is None else level.canonical_level_id,
                validation_id, rank, sample_rank,
            ))
        result = object.__new__(ExdHoxDataset)
        object.__setattr__(result, "_metadata", tuple(metadata))
        object.__setattr__(result, "unique_rc_group_count", len({row.global_rc_group_id for row in metadata}))
        return result


def _validate_sample_metadata(item: SampleMetadata) -> None:
    """Reject forged or inconsistent metadata even in a singleton batch."""
    _require(type(item.transcription_factor) is str and item.transcription_factor in TF_NAMES,
             "Invalid sample TF.")
    _require(type(item.primary_split) is str and item.primary_split in ("training", "validation"),
             "Unsupported sample split.")
    _sequence(item.sequence)
    _require(type(item.target_value_float32) is float, "Invalid sample target type.")
    value = _target(item.target_bits_big_endian_hex, format(item.target_value_float32, ".9g"))
    _require(value == item.target_value_float32, "Metadata target differs from exact float32 value.")
    _require(item.logical_example_id == "lex_" + _domain_hash(
        "exd_hox_logical_example.v1", item.transcription_factor, item.sequence, item.target_bits_big_endian_hex),
        "Invalid sample logical identity.")
    canonical = min(item.sequence, item.sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1])
    _require(item.global_rc_group_id == "rcg_" + _domain_hash("exd_hox_global_rc_group.v1", canonical),
             "Invalid sample RC group.")
    for value in (item.split_identity_hash, item.split_manifest_hash, item.subset_set_manifest_hash):
        _require(_hex(value), "Invalid sample manifest identity.")
    prefix = "exd_hox_training_stage_"
    _require(type(item.stage_id) is str and item.stage_id.startswith(prefix) and _hex(item.stage_id[len(prefix):]),
             "Invalid sample stage identity.")
    _require(_integer(item.sample_rank_one_based, 1), "Invalid sample membership rank.")
    if item.primary_split == "training":
        for value in (item.level_id, item.canonical_level_id):
            _require(type(value) is str and value.startswith("lvl_") and _hex(value[4:]),
                     "Missing or invalid training level identity.")
        _require(item.validation_id is None and _integer(item.rank_one_based, 1)
                 and item.rank_one_based <= item.sample_rank_one_based, "Invalid training rank or validation identity.")
    else:
        _require(item.level_id is None and item.canonical_level_id is None and item.rank_one_based is None,
                 "Validation cannot carry training level/rank metadata.")
        _require(item.validation_id == "validation_" + _domain_hash(
            "exd_hox_fixed_validation.v1", item.split_identity_hash, item.transcription_factor),
            "Invalid fixed validation identity.")


def collate_exd_hox(samples: Sequence[ExdHoxSample]) -> ExdHoxBatch:
    """Stack validated samples without shuffling, casting or changing orientation."""
    import torch

    _require(isinstance(samples, (tuple, list)) and bool(samples), "Collation requires a nonempty sample sequence.")
    identity = None
    metadata = []
    inputs = []
    targets = []
    for sample in samples:
        _require(type(sample) is ExdHoxSample and type(sample.metadata) is SampleMetadata, "Invalid sample type.")
        item = sample.metadata
        _validate_sample_metadata(item)
        current = (item.stage_id, item.split_identity_hash, item.split_manifest_hash,
                   item.subset_set_manifest_hash, item.transcription_factor, item.primary_split,
                   item.level_id, item.canonical_level_id, item.validation_id)
        _require(identity is None or identity == current, "Mixed dataset identities in batch.")
        identity = current
        _require(item.primary_split in ("training", "validation"), "Unsupported sample split.")
        _require(type(sample.x) is torch.Tensor and sample.x.dtype == torch.float32
                 and sample.x.device.type == "cpu" and tuple(sample.x.shape) == (14, 4)
                 and torch.equal(sample.x, encode_one_hot_14(item.sequence)), "Invalid one-hot sample.")
        _require(type(sample.y) is torch.Tensor and sample.y.dtype == torch.float32
                 and sample.y.device.type == "cpu" and tuple(sample.y.shape) == (1,), "Invalid target tensor.")
        value = _target(item.target_bits_big_endian_hex, format(item.target_value_float32, ".9g"))
        _require(sample.y.item() == value and struct.pack(">f", sample.y.item()).hex() == item.target_bits_big_endian_hex,
                 "Target tensor differs from source bits.")
        inputs.append(sample.x)
        targets.append(sample.y)
        metadata.append(item)
    return ExdHoxBatch(torch.stack(inputs), torch.stack(targets), tuple(metadata))


def _scan_logical(
    root: Path, contract: dict[str, Any], transcription_factor: str,
) -> tuple[dict[str, _PublicRow], tuple[_PublicRow, ...]]:
    records = {record["path"]: record for record in _records(contract)}
    factors = contract["transcription_factors"]
    factor_bits = {tf: 1 << index for index, tf in enumerate(factors)}
    counts = Counter()
    group_counts = Counter()
    global_groups = {}
    selected_labels = {}
    training = {}
    validation = []
    previous_id = ""
    source_count = 0
    complement = str.maketrans("ACGT", "TGCA")
    for fields in _tsv_rows(root, records[LOGICAL_PATH], LOGICAL_FIELDS):
        (logical_id, tf, sequence, sequence_hash, canonical, canonical_hash, group_id,
         selection, target_text, bits, commitment, occurrences) = fields
        _require(tf in factors and selection in SPLITS, "Invalid table TF or split.")
        _require(logical_id.startswith("lex_") and _hex(logical_id[4:]) and previous_id < logical_id,
                 "Duplicate, unsorted or malformed logical ID.")
        previous_id = logical_id
        _sequence(sequence)
        expected_canonical = min(sequence, sequence.translate(complement)[::-1])
        _require(canonical == expected_canonical, "RC canonical sequence mismatch.")
        _require(hashlib.sha256(sequence.encode("ascii")).hexdigest() == sequence_hash,
                 "Sequence hash mismatch.")
        _require(hashlib.sha256(canonical.encode("ascii")).hexdigest() == canonical_hash,
                 "RC canonical hash mismatch.")
        _require(group_id == "rcg_" + _domain_hash("exd_hox_global_rc_group.v1", canonical),
                 "RC group identity mismatch.")
        source_count += _text_integer(occurrences, 1)
        counts[(tf, selection)] += 1
        old_split, seen_factors = global_groups.get(group_id, (selection, 0))
        _require(old_split == selection, "One global RC group spans multiple splits.")
        if not seen_factors & factor_bits[tf]:
            group_counts[(tf, selection)] += 1
        global_groups[group_id] = (selection, seen_factors | factor_bits[tf])
        _require(_hex(commitment), "Malformed target commitment.")
        if selection == "test":
            _require(target_text == "" and bits == "", "Public test targets must be redacted.")
        else:
            target = _target(bits, target_text)
            _require(logical_id == "lex_" + _domain_hash("exd_hox_logical_example.v1", tf, sequence, bits),
                     "Logical identity mismatch.")
            _require(commitment == _domain_hash("exd_hox_target_commitment.v1", logical_id, bits),
                     "Target commitment mismatch.")
            if tf == transcription_factor:
                _require(group_id not in selected_labels or selected_labels[group_id] == bits,
                         "Conflicting within-TF RC-group targets.")
                selected_labels[group_id] = bits
                row = _PublicRow(logical_id, tf, sequence, group_id, selection, target, bits)
                if selection == "training":
                    training[logical_id] = row
                else:
                    validation.append(row)
    expected = contract["expected_counts"]
    _require(source_count == expected["source_occurrences"], "Source occurrence count mismatch.")
    _require(sum(counts.values()) == expected["logical_examples"], "Logical example count mismatch.")
    observed_global = Counter(value[0] for value in global_groups.values())
    for selection in SPLITS:
        _require(observed_global[selection] == expected["global_rc_groups"][selection], "Global RC-group count mismatch.")
        for tf in factors:
            item = expected["per_tf"][tf][selection]
            _require(counts[(tf, selection)] == item["logical_examples"]
                     and group_counts[(tf, selection)] == item["rc_groups"], "TF split membership count mismatch.")
    return training, tuple(validation)


def _scan_ordering(
    root: Path, contract: dict[str, Any], transcription_factor: str,
    training: dict[str, _PublicRow],
) -> tuple[tuple[tuple[_PublicRow, int], ...], dict[str, dict[int, int]]]:
    records = {record["path"]: record for record in _records(contract)}
    factors = contract["transcription_factors"]
    ranks = {tf: 0 for tf in factors}
    cumulative = {tf: {0: 0} for tf in factors}
    row_counts = Counter()
    groups = {tf: set() for tf in factors}
    last_group = {}
    previous_key = None
    observed_ids = set()
    selected_ids = set()
    selected = []
    for fields in _tsv_rows(root, records[ORDERING_PATH], ORDERING_FIELDS):
        tf, rank_text, group_id, logical_id, bin_text, order_hash = fields
        _require(tf in factors, "Invalid ordering TF.")
        rank = _text_integer(rank_text, 1)
        affinity_bin = _text_integer(bin_text)
        _require(affinity_bin < 10, "Invalid training affinity bin.")
        _require(group_id.startswith("rcg_") and _hex(group_id[4:])
                 and logical_id.startswith("lex_") and _hex(logical_id[4:]), "Malformed ordering ID.")
        _require(logical_id not in observed_ids, "Duplicate ordering logical ID.")
        observed_ids.add(logical_id)
        key = (tf, rank, logical_id)
        _require(previous_key is None or previous_key < key, "Duplicate or unsorted ordering row.")
        previous_key = key
        _require(order_hash == _domain_hash("exd_hox_nested_subset_group_order.v1", "32001", tf, group_id),
                 "Ordering digest mismatch.")
        if rank == ranks[tf]:
            _require(last_group[tf] == (group_id, affinity_bin, order_hash), "Rank has inconsistent group or bin.")
        else:
            _require(rank == ranks[tf] + 1, "Ordering rank gap.")
            _require(group_id not in groups[tf], "Ordered group appears under multiple ranks.")
            groups[tf].add(group_id)
            ranks[tf] = rank
            last_group[tf] = (group_id, affinity_bin, order_hash)
        row_counts[tf] += 1
        cumulative[tf][rank] = row_counts[tf]
        if tf == transcription_factor:
            _require(logical_id in training and logical_id not in selected_ids, "Missing or duplicate training ordering member.")
            row = training[logical_id]
            _require(row.global_rc_group_id == group_id, "Ordering group differs from training membership.")
            selected_ids.add(logical_id)
            selected.append((row, rank))
    expected = contract["expected_counts"]
    _require(selected_ids == set(training), "Ordering does not equal full training membership.")
    _require(sum(row_counts.values()) == expected["ordering_logical_examples"], "Ordering row count mismatch.")
    for tf in factors:
        item = expected["per_tf"][tf]["training"]
        _require(row_counts[tf] == item["logical_examples"] and ranks[tf] == item["rc_groups"],
                 "Ordering TF row/group count mismatch.")
    return tuple(selected), cumulative


def _read_levels(
    root: Path, contract: dict[str, Any], cumulative: dict[str, dict[int, int]],
    transcription_factor: str,
) -> tuple[LevelSpec, ...]:
    records = {record["path"]: record for record in _records(contract)}
    levels = []
    identifiers = set()
    requests = set()
    for fields in _tsv_rows(root, records[LEVELS_PATH], LEVEL_FIELDS):
        tf, level_id, request_type, request_value, unaliased, alias_text, canonical, actual, groups, terminal = fields
        _require(tf in contract["transcription_factors"], "Invalid level TF.")
        _require(level_id.startswith("lvl_") and _hex(level_id[4:]) and level_id not in identifiers,
                 "Duplicate or malformed level ID.")
        identifiers.add(level_id)
        _require(request_type in ("absolute", "fractional"), "Invalid level request type.")
        unaliased_count = _text_integer(unaliased, 1)
        canonical_count = _text_integer(canonical, 1)
        actual_count = _text_integer(actual, 1)
        group_count = _text_integer(groups, 1)
        terminal_rank = _text_integer(terminal, 1)
        alias = None if alias_text == "" else _text_integer(alias_text, 1)
        if request_type == "absolute":
            _require(_text_integer(request_value, 1) == unaliased_count == canonical_count and alias is None,
                     "Absolute level counts differ.")
        else:
            _require(re.fullmatch(r"0\.[0-9]+|1\.0", request_value) is not None, "Invalid fractional request text.")
            try:
                fraction = Decimal(request_value)
            except InvalidOperation as error:
                raise ExdHoxDatasetError("Invalid fractional request.") from error
            _require(fraction.is_finite() and 0 < fraction <= 1, "Fraction is outside (0,1].")
            _require(canonical_count == (unaliased_count if alias is None else alias), "Alias canonical count differs.")
        request_key = (tf, request_type, request_value)
        _require(request_key not in requests, "Duplicate level request.")
        requests.add(request_key)
        prefix = cumulative[tf]
        _require(terminal_rank in prefix and terminal_rank - 1 in prefix, "Level terminal group is missing.")
        _require(prefix[terminal_rank] == actual_count and group_count == terminal_rank,
                 "Level cumulative row or unique-group count mismatch.")
        _require(prefix[terminal_rank - 1] < canonical_count <= actual_count,
                 "Level membership differs from exact whole-group prefix.")
        if request_type == "fractional" and request_value == "1.0":
            full = contract["expected_counts"]["per_tf"][tf]["training"]
            _require(unaliased_count == canonical_count == actual_count == full["logical_examples"]
                     and group_count == full["rc_groups"], "100% level is not exact full training membership.")
        levels.append(LevelSpec(tf, level_id, request_type, request_value, unaliased_count, alias,
                                canonical_count, actual_count, group_count, terminal_rank, level_id))
    _require(len(levels) == contract["expected_counts"]["level_rows"], "Level row count mismatch.")
    for tf in contract["transcription_factors"]:
        _require((tf, "fractional", "1.0") in requests, "Missing 100% level.")
    resolved = []
    for level in levels:
        if level.alias_absolute_anchor is not None:
            anchors = [candidate for candidate in levels
                       if candidate.transcription_factor == level.transcription_factor
                       and candidate.request_type == "absolute"
                       and candidate.request_value == str(level.alias_absolute_anchor)]
            _require(len(anchors) == 1, "Missing or ambiguous alias anchor.")
            anchor = anchors[0]
            _require((level.canonical_requested_logical_example_count, level.actual_logical_example_count,
                      level.actual_rc_group_count, level.inclusive_maximum_rank)
                     == (anchor.canonical_requested_logical_example_count, anchor.actual_logical_example_count,
                         anchor.actual_rc_group_count, anchor.inclusive_maximum_rank), "Alias membership differs from anchor.")
            level = replace(level, canonical_level_id=anchor.level_id)
        if level.transcription_factor == transcription_factor:
            resolved.append(level)
    return tuple(resolved)


def _read_public_tf_data(
    root: Path, contract: dict[str, Any], transcription_factor: str, stage_id: str,
) -> PublicTFData:
    """Private verified-input reader for stage loading and read-only acceptance.

    No directory scan is made here: the original accepted repository contains
    unrelated artifacts. Only the nine exact public inputs may be opened.
    """
    _validate_contract(contract)
    _require(type(transcription_factor) is str and transcription_factor in contract["transcription_factors"],
             "Exactly one supported TF is required.")
    _require(stage_id == _stage_id(_build_stage_manifest(contract)), "Scientific stage identity differs.")
    _verify_records(root, _records(contract))
    _read_metadata(root, contract)
    training, validation = _scan_logical(root, contract, transcription_factor)
    ordered, cumulative = _scan_ordering(root, contract, transcription_factor, training)
    levels = _read_levels(root, contract, cumulative, transcription_factor)
    result = object.__new__(PublicTFData)
    object.__setattr__(result, "transcription_factor", transcription_factor)
    object.__setattr__(result, "stage_id", stage_id)
    object.__setattr__(result, "_identities", (contract["split_identity_hash"], contract["split_manifest_hash"],
                                            contract["subset_set_manifest_hash"]))
    object.__setattr__(result, "_training", ordered)
    object.__setattr__(result, "_validation", validation)
    object.__setattr__(result, "_levels", levels)
    return result


def open_public_tf_data(
    stage_root: Path, *, transcription_factor: str, expected_stage_id: str,
) -> PublicTFData:
    """Open exactly one TF from the pinned immutable public training stage."""
    contract = _load_contract()
    _require(type(transcription_factor) is str and transcription_factor in contract["transcription_factors"],
             "Exactly one supported TF is required.")
    _verify_stage(Path(stage_root), contract, expected_stage_id)
    return _read_public_tf_data(Path(stage_root), contract, transcription_factor, expected_stage_id)
