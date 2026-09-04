"""Fail-closed software access control for sealed Exd-Hox test targets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Dict, Mapping, Sequence, Tuple

import yaml

from src.downstream_fingerprints import (
    build_hashed_manifest,
    hash_logical_content,
    validate_hashed_manifest,
    validate_repository_relative_path,
    write_json_exclusive,
)


POLICY_SCHEMA_VERSION = "exd_hox_test_access_policy.v1"
AUTHORIZATION_SCHEMA_VERSION = "exd_hox_test_access_authorization.v1"
ACCESS_RECORD_SCHEMA_VERSION = "exd_hox_test_access_record.v1"
TARGET_COMMITMENT_SCHEMA_VERSION = "exd_hox_target_commitment.v1"
SEALED_TARGET_MANIFEST_SCHEMA_VERSION = (
    "exd_hox_sealed_test_target_manifest.v1"
)
TARGET_COMMITMENT_SET_SCHEMA_VERSION = "exd_hox_target_commitment_set.v1"

POLICY_NAME = "exd_hox_primary_test_access.v1"
AUTHORIZED_OPERATION = "final_primary_test_evaluation"
PRIMARY_SPLIT_POLICY_IDENTIFIER = (
    "global_rc_affinity_stratified_80_10_10.v1"
)
SEALED_TARGET_DIRECTORY = "data/sealed/exd_hox_primary_test_targets_v1"
SEALED_TARGET_MANIFEST_DIRECTORY = "data/processed/exd_hox_primary_split_v1"
ACCESS_RECORD_DIRECTORY = "results/exd_hox_test_access_records_v1"
SEALED_TARGET_FILENAME_SUFFIX = ".tsv.gz"
SEALED_TARGET_MANIFEST_FILENAME = "exd_hox_sealed_test_target_manifest_v1.json"
DEFAULT_AUTHORIZATION_MAX_VALIDITY_SECONDS = 86400

IDENTITY_FIELDS = (
    "split_manifest_hash",
    "split_identity_hash",
    "subset_set_manifest_hash",
    "config_hash",
    "model_hash",
    "checkpoint_hash",
)
SEALED_TARGET_FIELDS = (
    "logical_example_id",
    "target_value_float32",
    "target_bits_big_endian_hex",
    "target_commitment_sha256",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FLOAT32_BITS_PATTERN = re.compile(r"^[0-9a-f]{8}$")
_UTC_TIMESTAMP_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
                                    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class SealedTestAccessError(ValueError):
    """Raised when sealed test-target access cannot be authorized safely."""


@dataclass(frozen=True)
class SealedTestTarget:
    """One validated plaintext target returned after authorized access."""

    logical_example_id: str
    target_value_float32: float
    target_bits_big_endian_hex: str
    target_commitment_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        """Return the target using the deterministic sealed-table schema."""

        return {
            "logical_example_id": self.logical_example_id,
            "target_value_float32": self.target_value_float32,
            "target_bits_big_endian_hex": self.target_bits_big_endian_hex,
            "target_commitment_sha256": self.target_commitment_sha256,
        }


@dataclass(frozen=True)
class AuthorizedSealedTestTargets:
    """Validated targets and the exclusive record proving their access."""

    targets: Tuple[SealedTestTarget, ...]
    access_record: Mapping[str, Any]
    access_record_path: Path


def build_test_access_policy(
    authorization_max_validity_seconds: int = (
        DEFAULT_AUTHORIZATION_MAX_VALIDITY_SECONDS
    ),
) -> Dict[str, Any]:
    """Build the fixed version-one test-access policy."""

    if authorization_max_validity_seconds <= 0:
        raise ValueError("Authorization maximum validity must be positive.")
    return build_hashed_manifest(
        POLICY_SCHEMA_VERSION,
        {
            "policy_name": POLICY_NAME,
            "authorized_operation": AUTHORIZED_OPERATION,
            "sealed_target_directory": SEALED_TARGET_DIRECTORY,
            "sealed_target_filename_suffix": SEALED_TARGET_FILENAME_SUFFIX,
            "sealed_target_manifest_filename": SEALED_TARGET_MANIFEST_FILENAME,
            "sealed_target_manifest_directory": (
                SEALED_TARGET_MANIFEST_DIRECTORY
            ),
            "access_record_directory": ACCESS_RECORD_DIRECTORY,
            "authorization_max_validity_seconds": (
                authorization_max_validity_seconds
            ),
            "required_identity_fields": list(IDENTITY_FIELDS),
            "sealed_target_fields": list(SEALED_TARGET_FIELDS),
        },
    )


def load_test_access_policy(path: Path | str) -> Dict[str, Any]:
    """Load and validate a hashed YAML test-access policy."""

    policy_path = Path(path)
    if not policy_path.is_file():
        raise SealedTestAccessError("Test-access policy is missing.")
    try:
        payload = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise SealedTestAccessError("Test-access policy cannot be read.") from error
    if not isinstance(payload, Mapping):
        raise SealedTestAccessError("Test-access policy must be a mapping.")
    policy = dict(payload)
    validate_test_access_policy(policy)
    return policy


def validate_test_access_policy(policy: Mapping[str, Any]) -> None:
    """Validate the policy hash and its fixed, narrow access boundary."""

    required_keys = {
        "schema_version",
        "policy_name",
        "authorized_operation",
        "sealed_target_directory",
        "sealed_target_filename_suffix",
        "sealed_target_manifest_filename",
        "sealed_target_manifest_directory",
        "access_record_directory",
        "authorization_max_validity_seconds",
        "required_identity_fields",
        "sealed_target_fields",
        "manifest_hash",
    }
    _require_exact_keys(policy, required_keys, "Test-access policy")
    _validate_manifest_hash(policy, "Test-access policy")
    if policy["schema_version"] != POLICY_SCHEMA_VERSION:
        raise SealedTestAccessError("Unsupported test-access policy schema.")
    if policy["policy_name"] != POLICY_NAME:
        raise SealedTestAccessError("Unexpected test-access policy name.")
    if policy["authorized_operation"] != AUTHORIZED_OPERATION:
        raise SealedTestAccessError("Unexpected authorized test operation.")
    if policy["sealed_target_directory"] != SEALED_TARGET_DIRECTORY:
        raise SealedTestAccessError(
            "Policy must use the fixed versioned sealed-target directory."
        )
    if policy["sealed_target_filename_suffix"] != SEALED_TARGET_FILENAME_SUFFIX:
        raise SealedTestAccessError("Unexpected sealed-target filename suffix.")
    if policy["sealed_target_manifest_filename"] != (
        SEALED_TARGET_MANIFEST_FILENAME
    ):
        raise SealedTestAccessError("Unexpected sealed-target manifest filename.")
    if policy["sealed_target_manifest_directory"] != (
        SEALED_TARGET_MANIFEST_DIRECTORY
    ):
        raise SealedTestAccessError(
            "Unexpected sealed-target manifest directory."
        )
    if policy["access_record_directory"] != ACCESS_RECORD_DIRECTORY:
        raise SealedTestAccessError("Unexpected test-access record directory.")
    maximum_validity = policy["authorization_max_validity_seconds"]
    if not isinstance(maximum_validity, int) or isinstance(
        maximum_validity,
        bool,
    ):
        raise SealedTestAccessError(
            "Authorization maximum validity must be an integer."
        )
    if maximum_validity <= 0:
        raise SealedTestAccessError(
            "Authorization maximum validity must be positive."
        )
    required_identity_fields = policy["required_identity_fields"]
    if not isinstance(required_identity_fields, list):
        raise SealedTestAccessError("Policy identity fields must be a list.")
    if tuple(required_identity_fields) != IDENTITY_FIELDS:
        raise SealedTestAccessError("Policy identity fields do not match v1.")
    sealed_target_fields = policy["sealed_target_fields"]
    if not isinstance(sealed_target_fields, list):
        raise SealedTestAccessError("Policy sealed-target fields must be a list.")
    if tuple(sealed_target_fields) != SEALED_TARGET_FIELDS:
        raise SealedTestAccessError("Policy sealed-target fields do not match v1.")
    _validate_sha256(str(policy["manifest_hash"]), "Policy manifest hash")


def target_commitment_sha256(
    logical_example_id: str,
    target_bits_big_endian_hex: str,
) -> str:
    """Commit one target's exact float32 bits to its logical-example ID."""

    if not logical_example_id:
        raise ValueError("Logical-example ID must not be empty.")
    _validate_float32_bits(target_bits_big_endian_hex)
    digest = hashlib.sha256()
    digest.update(TARGET_COMMITMENT_SCHEMA_VERSION.encode("utf-8"))
    for part in (logical_example_id, target_bits_big_endian_hex):
        digest.update(b"\x00")
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def target_commitment_digest_sha256(
    rows: Sequence[Mapping[str, Any] | SealedTestTarget],
) -> str:
    """Hash the public ordered set of target commitments without target values."""

    commitments = []
    previous_id = None
    observed_ids = set()
    for row in rows:
        if isinstance(row, SealedTestTarget):
            logical_example_id = row.logical_example_id
            commitment = row.target_commitment_sha256
        else:
            logical_example_id = str(row["logical_example_id"])
            commitment = str(row["target_commitment_sha256"])
        if not logical_example_id:
            raise ValueError("Logical-example ID must not be empty.")
        if logical_example_id in observed_ids:
            raise ValueError("Target commitment IDs must be unique.")
        if previous_id is not None and logical_example_id <= previous_id:
            raise ValueError("Target commitments must be strictly ID-sorted.")
        _validate_sha256(commitment, "Target commitment")
        commitments.append(
            {
                "logical_example_id": logical_example_id,
                "target_commitment_sha256": commitment,
            }
        )
        observed_ids.add(logical_example_id)
        previous_id = logical_example_id
    if not commitments:
        raise ValueError("Target commitments must not be empty.")
    return hash_logical_content(
        {
            "schema_version": TARGET_COMMITMENT_SET_SCHEMA_VERSION,
            "targets": commitments,
        }
    )


def build_test_access_authorization(
    *,
    policy: Mapping[str, Any],
    authorization_id: str,
    issued_at_utc: datetime | str,
    expires_at_utc: datetime | str,
    identities: Mapping[str, str],
    sealed_target_manifest_path: str,
    sealed_target_manifest_hash: str,
) -> Dict[str, Any]:
    """Build a hashed authorization bound to one target and run identity."""

    validate_test_access_policy(policy)
    _validate_sha256(authorization_id, "Authorization ID")
    normalized_identities = _validate_identities(identities)
    normalized_manifest_path = _validate_sealed_target_manifest_logical_path(
        sealed_target_manifest_path,
        policy,
    )
    _validate_sha256(
        sealed_target_manifest_hash,
        "Sealed-target manifest hash",
    )
    issued_text = _normalize_utc_timestamp(issued_at_utc)
    expires_text = _normalize_utc_timestamp(expires_at_utc)
    issued = _parse_utc_timestamp(issued_text, "issued_at_utc")
    expires = _parse_utc_timestamp(expires_text, "expires_at_utc")
    _validate_authorization_interval(issued, expires, policy)
    return build_hashed_manifest(
        AUTHORIZATION_SCHEMA_VERSION,
        {
            "authorization_id": authorization_id,
            "policy_manifest_hash": policy["manifest_hash"],
            "authorized_operation": AUTHORIZED_OPERATION,
            "issued_at_utc": issued_text,
            "expires_at_utc": expires_text,
            "identities": normalized_identities,
            "sealed_target_manifest_path": normalized_manifest_path,
            "sealed_target_manifest_hash": sealed_target_manifest_hash,
        },
    )


def load_authorized_sealed_test_targets(
    *,
    repository_root: Path | str,
    policy_path: Path | str,
    authorization_path: Path | str,
    expected_identities: Mapping[str, str],
) -> AuthorizedSealedTestTargets:
    """Authorize one access, write its record exclusively, and load targets."""

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise SealedTestAccessError("Repository root is missing.")
    policy = load_test_access_policy(policy_path)
    authorization = _load_test_access_authorization(authorization_path)
    access_time = _current_utc_time()
    normalized_identities = _validate_identities(expected_identities)
    validate_test_access_authorization(
        authorization=authorization,
        policy=policy,
        expected_identities=normalized_identities,
        now_utc=access_time,
    )

    target_manifest = _load_sealed_target_manifest(
        repository_root=root,
        logical_path=str(authorization["sealed_target_manifest_path"]),
        policy=policy,
    )
    if target_manifest["manifest_hash"] != (
        authorization["sealed_target_manifest_hash"]
    ):
        raise SealedTestAccessError(
            "Authorization sealed-target manifest identity mismatch."
        )
    if target_manifest["split_identity_hash"] != (
        normalized_identities["split_identity_hash"]
    ):
        raise SealedTestAccessError(
            "Sealed-target descriptor split identity mismatch."
        )

    access_record = _build_access_record(
        authorization=authorization,
        policy=policy,
        target_manifest=target_manifest,
        identities=normalized_identities,
        accessed_at=access_time,
    )
    try:
        record_path = _access_record_path(root, policy, authorization)
        write_json_exclusive(record_path, access_record)
    except FileExistsError as error:
        raise SealedTestAccessError(
            "Authorization has already been used."
        ) from error
    except OSError as error:
        raise SealedTestAccessError(
            "Exclusive test-access record cannot be written."
        ) from error

    target_path = _resolve_sealed_target_path(
        repository_root=root,
        logical_path=str(target_manifest["sealed_target_path"]),
        policy=policy,
    )
    try:
        target_bytes = target_path.read_bytes()
    except OSError as error:
        raise SealedTestAccessError("Sealed target cannot be read.") from error
    _validate_target_file_identity(target_bytes, target_manifest)

    targets = _parse_sealed_target_bytes(target_bytes)
    _validate_loaded_targets_against_manifest(targets, target_manifest)
    return AuthorizedSealedTestTargets(
        targets=targets,
        access_record=access_record,
        access_record_path=record_path,
    )


def load_sealed_test_targets(
    *,
    repository_root: Path | str,
    policy_path: Path | str,
    authorization_path: Path | str,
    expected_identities: Mapping[str, str],
) -> AuthorizedSealedTestTargets:
    """Load sealed targets through the authorization-only access path."""

    return load_authorized_sealed_test_targets(
        repository_root=repository_root,
        policy_path=policy_path,
        authorization_path=authorization_path,
        expected_identities=expected_identities,
    )


def validate_test_access_authorization(
    *,
    authorization: Mapping[str, Any],
    policy: Mapping[str, Any],
    expected_identities: Mapping[str, str],
    now_utc: datetime,
) -> None:
    """Validate one authorization against current immutable identities."""

    required_keys = {
        "schema_version",
        "authorization_id",
        "policy_manifest_hash",
        "authorized_operation",
        "issued_at_utc",
        "expires_at_utc",
        "identities",
        "sealed_target_manifest_path",
        "sealed_target_manifest_hash",
        "manifest_hash",
    }
    _require_exact_keys(authorization, required_keys, "Test authorization")
    _validate_manifest_hash(authorization, "Test authorization")
    if authorization["schema_version"] != AUTHORIZATION_SCHEMA_VERSION:
        raise SealedTestAccessError("Unsupported test authorization schema.")
    _validate_sha256(str(authorization["authorization_id"]), "Authorization ID")
    if authorization["policy_manifest_hash"] != policy["manifest_hash"]:
        raise SealedTestAccessError("Authorization policy identity mismatch.")
    if authorization["authorized_operation"] != AUTHORIZED_OPERATION:
        raise SealedTestAccessError("Authorization operation mismatch.")
    authorized_identities = _validate_identities(authorization["identities"])
    current_identities = _validate_identities(expected_identities)
    for field in IDENTITY_FIELDS:
        if authorized_identities[field] != current_identities[field]:
            message = "Authorization {0} identity mismatch."
            raise SealedTestAccessError(message.format(field))

    issued = _parse_utc_timestamp(
        str(authorization["issued_at_utc"]),
        "issued_at_utc",
    )
    expires = _parse_utc_timestamp(
        str(authorization["expires_at_utc"]),
        "expires_at_utc",
    )
    _validate_authorization_interval(issued, expires, policy)
    if _access_time_before_issue(now_utc, issued):
        raise SealedTestAccessError("Authorization is not yet valid.")
    if now_utc >= expires:
        raise SealedTestAccessError("Authorization is stale or expired.")
    _validate_sealed_target_manifest_logical_path(
        str(authorization["sealed_target_manifest_path"]),
        policy,
    )
    _validate_sha256(
        str(authorization["sealed_target_manifest_hash"]),
        "Sealed-target manifest hash",
    )


def _access_time_before_issue(access_time: datetime, issued: datetime) -> bool:
    """Return whether an aware access time precedes authorization issuance."""

    if access_time.tzinfo is None:
        raise SealedTestAccessError("Access time must include a timezone.")
    return access_time.astimezone(timezone.utc) < issued


def _load_test_access_authorization(path: Path | str) -> Dict[str, Any]:
    authorization_path = Path(path)
    if not authorization_path.is_file():
        raise SealedTestAccessError("Test authorization is missing.")
    try:
        payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedTestAccessError("Test authorization cannot be read.") from error
    if not isinstance(payload, Mapping):
        raise SealedTestAccessError("Test authorization must be a mapping.")
    return dict(payload)


def _validate_manifest_hash(payload: Mapping[str, Any], label: str) -> None:
    try:
        validate_hashed_manifest(payload)
    except (KeyError, TypeError, ValueError) as error:
        message = "{0} hash is missing or invalid."
        raise SealedTestAccessError(message.format(label)) from error


def _require_exact_keys(
    payload: Mapping[str, Any],
    required_keys: set[str],
    label: str,
) -> None:
    observed_keys = set(payload)
    if observed_keys != required_keys:
        missing = sorted(required_keys - observed_keys)
        unexpected = sorted(observed_keys - required_keys)
        message = "{0} fields differ; missing={1}, unexpected={2}."
        raise SealedTestAccessError(
            message.format(label, missing, unexpected)
        )


def _validate_identities(identities: Mapping[str, str]) -> Dict[str, str]:
    if not isinstance(identities, Mapping):
        raise SealedTestAccessError("Run identities must be a mapping.")
    required_fields = set(IDENTITY_FIELDS)
    _require_exact_keys(identities, required_fields, "Run identities")
    normalized = {}
    for field in IDENTITY_FIELDS:
        value = identities[field]
        if not isinstance(value, str):
            message = "{0} must be a string SHA-256 identity."
            raise SealedTestAccessError(message.format(field))
        _validate_sha256(value, field)
        normalized[field] = value
    return normalized


def _validate_sha256(value: str, label: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        message = "{0} must be a full lowercase SHA-256 identity."
        raise SealedTestAccessError(message.format(label))


def _validate_float32_bits(value: str) -> None:
    if _FLOAT32_BITS_PATTERN.fullmatch(value) is None:
        raise SealedTestAccessError(
            "Target bits must be eight lowercase big-endian hexadecimal digits."
        )


def _normalize_utc_timestamp(value: datetime | str) -> str:
    if isinstance(value, str):
        _parse_utc_timestamp(value, "timestamp")
        return value
    if value.tzinfo is None:
        raise ValueError("Authorization timestamps must include a timezone.")
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_timestamp(value: str, field: str) -> datetime:
    if _UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        message = "Authorization {0} must use YYYY-MM-DDTHH:MM:SSZ."
        raise SealedTestAccessError(message.format(field))
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        message = "Authorization {0} is not a valid UTC timestamp."
        raise SealedTestAccessError(message.format(field)) from error
    return parsed.replace(tzinfo=timezone.utc)


def _validate_authorization_interval(
    issued: datetime,
    expires: datetime,
    policy: Mapping[str, Any],
) -> None:
    if expires <= issued:
        raise SealedTestAccessError(
            "Authorization expiration must follow issuance."
        )
    maximum_seconds = int(policy["authorization_max_validity_seconds"])
    validity_seconds = int((expires - issued).total_seconds())
    if validity_seconds > maximum_seconds:
        raise SealedTestAccessError(
            "Authorization validity exceeds the policy maximum."
        )


def _current_utc_time() -> datetime:
    """Read the production access clock; tests may patch this private seam."""

    return datetime.now(timezone.utc).replace(microsecond=0)


def _validate_sealed_target_logical_path(
    path: str,
    policy: Mapping[str, Any],
) -> str:
    try:
        normalized = validate_repository_relative_path(path)
    except ValueError as error:
        raise SealedTestAccessError("Invalid sealed-target path.") from error
    candidate = Path(normalized)
    sealed_directory = Path(str(policy["sealed_target_directory"]))
    try:
        relative_target = candidate.relative_to(sealed_directory)
    except ValueError as error:
        raise SealedTestAccessError(
            "Sealed target is outside the versioned sealed directory."
        ) from error
    if relative_target == Path("."):
        raise SealedTestAccessError("Sealed-target path must name a file.")
    suffix = str(policy["sealed_target_filename_suffix"])
    if not candidate.as_posix().endswith(suffix):
        raise SealedTestAccessError("Sealed target must use the .tsv.gz suffix.")
    return candidate.as_posix()


def _validate_sealed_target_manifest_logical_path(
    path: str,
    policy: Mapping[str, Any],
) -> str:
    try:
        normalized = validate_repository_relative_path(path)
    except ValueError as error:
        raise SealedTestAccessError(
            "Invalid sealed-target manifest path."
        ) from error
    expected_filename = str(policy["sealed_target_manifest_filename"])
    expected_directory = Path(
        str(policy["sealed_target_manifest_directory"])
    )
    candidate = Path(normalized)
    if candidate.parent != expected_directory:
        raise SealedTestAccessError(
            "Sealed-target manifest is outside the finalized split directory."
        )
    if candidate.name != expected_filename:
        raise SealedTestAccessError(
            "Unexpected sealed-target manifest filename."
        )
    return normalized


def _load_sealed_target_manifest(
    *,
    repository_root: Path,
    logical_path: str,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized = _validate_sealed_target_manifest_logical_path(
        logical_path,
        policy,
    )
    manifest_path = repository_root / normalized
    _reject_symlink_components(repository_root, manifest_path)
    if not manifest_path.is_file():
        raise SealedTestAccessError("Sealed-target manifest is missing.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedTestAccessError(
            "Sealed-target manifest cannot be read."
        ) from error
    if not isinstance(payload, Mapping):
        raise SealedTestAccessError(
            "Sealed-target manifest must be a mapping."
        )
    manifest = dict(payload)
    _validate_sealed_target_manifest(manifest, policy)
    return manifest


def _validate_sealed_target_manifest(
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    required_keys = {
        "schema_version",
        "split_identity_hash",
        "split_policy_identifier",
        "sealed_target_path",
        "sealed_target_byte_size",
        "sealed_target_sha256",
        "test_logical_example_count",
        "target_commitment_digest_sha256",
        "manifest_hash",
    }
    _require_exact_keys(
        manifest,
        required_keys,
        "Sealed-target manifest",
    )
    _validate_manifest_hash(manifest, "Sealed-target manifest")
    if manifest["schema_version"] != SEALED_TARGET_MANIFEST_SCHEMA_VERSION:
        raise SealedTestAccessError(
            "Unsupported sealed-target manifest schema."
        )
    _validate_sha256(
        str(manifest["split_identity_hash"]),
        "Split identity hash",
    )
    if manifest["split_policy_identifier"] != PRIMARY_SPLIT_POLICY_IDENTIFIER:
        raise SealedTestAccessError(
            "Sealed-target manifest split policy mismatch."
        )
    _validate_sealed_target_logical_path(
        str(manifest["sealed_target_path"]),
        policy,
    )
    byte_size = manifest["sealed_target_byte_size"]
    if not isinstance(byte_size, int) or isinstance(byte_size, bool):
        raise SealedTestAccessError(
            "Sealed-target byte size must be an integer."
        )
    if byte_size <= 0:
        raise SealedTestAccessError("Sealed-target byte size must be positive.")
    _validate_sha256(
        str(manifest["sealed_target_sha256"]),
        "Sealed-target SHA-256",
    )
    test_count = manifest["test_logical_example_count"]
    if not isinstance(test_count, int) or isinstance(test_count, bool):
        raise SealedTestAccessError(
            "Test logical-example count must be an integer."
        )
    if test_count <= 0:
        raise SealedTestAccessError(
            "Test logical-example count must be positive."
        )
    _validate_sha256(
        str(manifest["target_commitment_digest_sha256"]),
        "Target commitment digest",
    )


def _resolve_sealed_target_path(
    *,
    repository_root: Path,
    logical_path: str,
    policy: Mapping[str, Any],
) -> Path:
    normalized = _validate_sealed_target_logical_path(logical_path, policy)
    sealed_directory = repository_root / str(policy["sealed_target_directory"])
    target_path = repository_root / normalized
    _reject_symlink_components(repository_root, target_path)
    if not target_path.is_file():
        raise SealedTestAccessError("Sealed target is missing.")
    try:
        resolved_directory = sealed_directory.resolve(strict=True)
        resolved_target = target_path.resolve(strict=True)
        resolved_target.relative_to(resolved_directory)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SealedTestAccessError(
            "Sealed target does not resolve inside its fixed directory."
        ) from error
    return resolved_target


def _reject_symlink_components(root: Path, target: Path) -> None:
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise SealedTestAccessError("Path is outside the repository root.") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SealedTestAccessError(
                "Sealed-target paths must not contain symbolic links."
            )


def _validate_target_file_identity(
    target_bytes: bytes,
    target_manifest: Mapping[str, Any],
) -> None:
    expected_size = int(target_manifest["sealed_target_byte_size"])
    if len(target_bytes) != expected_size:
        raise SealedTestAccessError("Sealed-target byte-size mismatch.")
    actual_hash = hashlib.sha256(target_bytes).hexdigest()
    if actual_hash != target_manifest["sealed_target_sha256"]:
        raise SealedTestAccessError("Sealed-target SHA-256 mismatch.")


def _validate_loaded_targets_against_manifest(
    targets: Sequence[SealedTestTarget],
    target_manifest: Mapping[str, Any],
) -> None:
    expected_count = int(target_manifest["test_logical_example_count"])
    if len(targets) != expected_count:
        raise SealedTestAccessError(
            "Sealed-target logical-example count mismatch."
        )
    actual_digest = target_commitment_digest_sha256(targets)
    if actual_digest != target_manifest["target_commitment_digest_sha256"]:
        raise SealedTestAccessError(
            "Sealed-target commitment digest mismatch."
        )


def _build_access_record(
    *,
    authorization: Mapping[str, Any],
    policy: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    identities: Mapping[str, str],
    accessed_at: datetime,
) -> Dict[str, Any]:
    return build_hashed_manifest(
        ACCESS_RECORD_SCHEMA_VERSION,
        {
            "authorization_id": authorization["authorization_id"],
            "authorization_manifest_hash": authorization["manifest_hash"],
            "policy_manifest_hash": policy["manifest_hash"],
            "authorized_operation": AUTHORIZED_OPERATION,
            "record_kind": "exclusive_single_use_access_claim",
            "claim_precedes_target_validation": True,
            "accessed_at_utc": _normalize_utc_timestamp(accessed_at),
            "identities": dict(identities),
            "sealed_target_manifest_path": (
                authorization["sealed_target_manifest_path"]
            ),
            "sealed_target_manifest_hash": target_manifest["manifest_hash"],
            "split_identity_hash": target_manifest["split_identity_hash"],
            "sealed_target_path": target_manifest["sealed_target_path"],
            "sealed_target_byte_size": target_manifest["sealed_target_byte_size"],
            "sealed_target_sha256": target_manifest["sealed_target_sha256"],
            "test_logical_example_count": (
                target_manifest["test_logical_example_count"]
            ),
            "target_commitment_digest_sha256": (
                target_manifest["target_commitment_digest_sha256"]
            ),
        },
    )


def _access_record_path(
    repository_root: Path,
    policy: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> Path:
    record_directory = repository_root / str(policy["access_record_directory"])
    _reject_symlink_components(repository_root, record_directory)
    record_directory.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(repository_root, record_directory)
    filename = "test_access_{0}.json".format(authorization["authorization_id"])
    return record_directory / filename


def _parse_sealed_target_bytes(payload: bytes) -> Tuple[SealedTestTarget, ...]:
    _validate_deterministic_gzip_header(payload)
    try:
        decompressed = gzip.decompress(payload)
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise SealedTestAccessError("Sealed target is not valid gzip data.") from error
    try:
        text = decompressed.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SealedTestAccessError("Sealed target is not UTF-8 TSV.") from error
    if not text.endswith("\n") or "\r" in text:
        raise SealedTestAccessError(
            "Sealed target must use final-newline LF-only TSV."
        )
    lines = text.split("\n")
    for line in lines[:-1]:
        if not line:
            raise SealedTestAccessError("Sealed target contains a blank TSV row.")

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(reader.fieldnames or ()) != SEALED_TARGET_FIELDS:
        raise SealedTestAccessError("Sealed-target TSV fields do not match v1.")
    targets = []
    previous_id = None
    observed_ids = set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise SealedTestAccessError("Malformed sealed-target TSV row.")
        target = _parse_sealed_target_row(row)
        if target.logical_example_id in observed_ids:
            raise SealedTestAccessError("Duplicate logical-example ID in targets.")
        if previous_id is not None and target.logical_example_id <= previous_id:
            raise SealedTestAccessError(
                "Sealed-target rows must be strictly ID-sorted."
            )
        observed_ids.add(target.logical_example_id)
        previous_id = target.logical_example_id
        targets.append(target)
    if not targets:
        raise SealedTestAccessError("Sealed target must contain at least one row.")
    return tuple(targets)


def _validate_deterministic_gzip_header(payload: bytes) -> None:
    if len(payload) < 18 or payload[0:3] != b"\x1f\x8b\x08":
        raise SealedTestAccessError("Sealed target is not gzip data.")
    flags = payload[3]
    if flags & 0xE0:
        raise SealedTestAccessError("Sealed target has invalid gzip flags.")
    if flags & 0x08:
        raise SealedTestAccessError(
            "Deterministic gzip must not embed a filename."
        )
    if payload[4:8] != b"\x00\x00\x00\x00":
        raise SealedTestAccessError("Deterministic gzip must use mtime=0.")


def _parse_sealed_target_row(row: Mapping[str, str]) -> SealedTestTarget:
    logical_example_id = row["logical_example_id"]
    if not logical_example_id:
        raise SealedTestAccessError("Logical-example ID must not be empty.")
    try:
        parsed_value = float(row["target_value_float32"])
    except ValueError as error:
        raise SealedTestAccessError("Invalid float32 target value.") from error
    if not math.isfinite(parsed_value):
        raise SealedTestAccessError("Target value must be finite.")
    if parsed_value < 0.0 or parsed_value > 1.0:
        raise SealedTestAccessError("Target value must lie in [0, 1].")
    try:
        packed_value = struct.pack(">f", parsed_value)
    except (OverflowError, struct.error) as error:
        raise SealedTestAccessError("Target value is outside float32 range.") from error
    target_bits = row["target_bits_big_endian_hex"]
    _validate_float32_bits(target_bits)
    if packed_value.hex() != target_bits:
        raise SealedTestAccessError(
            "Target decimal value does not match its float32 bits."
        )
    float32_value = struct.unpack(">f", packed_value)[0]
    canonical_value_text = format(float32_value, ".9g")
    if row["target_value_float32"] != canonical_value_text:
        raise SealedTestAccessError(
            "Target value must use canonical float32 text."
        )
    expected_commitment = target_commitment_sha256(
        logical_example_id,
        target_bits,
    )
    target_commitment = row["target_commitment_sha256"]
    _validate_sha256(target_commitment, "Target commitment")
    if target_commitment != expected_commitment:
        raise SealedTestAccessError("Target commitment mismatch.")
    return SealedTestTarget(
        logical_example_id=logical_example_id,
        target_value_float32=float32_value,
        target_bits_big_endian_hex=target_bits,
        target_commitment_sha256=target_commitment,
    )
