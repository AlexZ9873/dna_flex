"""Synthetic tests for fail-closed Exd-Hox sealed-test access."""

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from src.downstream_fingerprints import (
    build_hashed_manifest,
    validate_hashed_manifest,
    write_tsv_gzip_exclusive,
)
from src.exd_hox_splits import (
    SEALED_TARGET_FIELDS as SPLIT_SEALED_TARGET_FIELDS,
    SEALED_TARGET_MANIFEST_FILENAME as SPLIT_TARGET_MANIFEST_FILENAME,
    SEALED_TARGET_MANIFEST_SCHEMA_VERSION as SPLIT_TARGET_MANIFEST_SCHEMA,
    target_commitment as split_target_commitment,
)
from src.sealed_test_access import (
    ACCESS_RECORD_DIRECTORY,
    IDENTITY_FIELDS,
    PRIMARY_SPLIT_POLICY_IDENTIFIER,
    SEALED_TARGET_DIRECTORY,
    SEALED_TARGET_FIELDS,
    SEALED_TARGET_MANIFEST_FILENAME,
    SEALED_TARGET_MANIFEST_SCHEMA_VERSION,
    SealedTestAccessError,
    build_test_access_authorization,
    build_test_access_policy,
    load_authorized_sealed_test_targets,
    load_test_access_policy,
    target_commitment_digest_sha256,
    target_commitment_sha256,
)


class SealedTestAccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.now = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
        self.policy = build_test_access_policy()
        self.policy_path = self.root / "configs" / "policy_v1.yaml"
        self.policy_path.parent.mkdir(parents=True)
        self.policy_path.write_text(
            yaml.safe_dump(self.policy, sort_keys=False),
            encoding="utf-8",
        )
        self.identities = {}
        for field in IDENTITY_FIELDS:
            self.identities[field] = _sha256(field)

        self.target_rows = _valid_target_rows()
        self.target_path = (
            self.root
            / SEALED_TARGET_DIRECTORY
            / "exd_hox_primary_test_targets_v1.tsv.gz"
        )
        self.target_path.parent.mkdir(parents=True)
        write_tsv_gzip_exclusive(
            self.target_path,
            SEALED_TARGET_FIELDS,
            self.target_rows,
        )
        self.target_bytes = self.target_path.read_bytes()

        self.target_manifest_path = (
            self.root
            / "data"
            / "processed"
            / "exd_hox_primary_split_v1"
            / SEALED_TARGET_MANIFEST_FILENAME
        )
        self.target_manifest_path.parent.mkdir(parents=True)
        self.target_manifest = self._build_target_manifest()
        _write_json(self.target_manifest_path, self.target_manifest)

        self.authorization_path = (
            self.root / "authorizations" / "test_access_v1.json"
        )
        self.authorization_path.parent.mkdir(parents=True)
        self.authorization = self._build_authorization()
        _write_json(self.authorization_path, self.authorization)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _build_target_manifest(self, **updates):
        content = {
            "split_identity_hash": self.identities["split_identity_hash"],
            "split_policy_identifier": PRIMARY_SPLIT_POLICY_IDENTIFIER,
            "sealed_target_path": self.target_path.relative_to(
                self.root
            ).as_posix(),
            "sealed_target_byte_size": len(self.target_bytes),
            "sealed_target_sha256": hashlib.sha256(
                self.target_bytes
            ).hexdigest(),
            "test_logical_example_count": len(self.target_rows),
            "target_commitment_digest_sha256": (
                target_commitment_digest_sha256(self.target_rows)
            ),
        }
        content.update(updates)
        return build_hashed_manifest(
            SEALED_TARGET_MANIFEST_SCHEMA_VERSION,
            content,
        )

    def _build_authorization(self, **updates):
        arguments = {
            "policy": self.policy,
            "authorization_id": _sha256("authorization"),
            "issued_at_utc": "2026-09-03T11:00:00Z",
            "expires_at_utc": "2026-09-03T13:00:00Z",
            "identities": self.identities,
            "sealed_target_manifest_path": self.target_manifest_path.relative_to(
                self.root
            ).as_posix(),
            "sealed_target_manifest_hash": self.target_manifest["manifest_hash"],
        }
        arguments.update(updates)
        return build_test_access_authorization(**arguments)

    def _load(self, **updates):
        arguments = {
            "repository_root": self.root,
            "policy_path": self.policy_path,
            "authorization_path": self.authorization_path,
            "expected_identities": self.identities,
        }
        arguments.update(updates)
        with mock.patch(
            "src.sealed_test_access._current_utc_time",
            return_value=self.now,
        ):
            return load_authorized_sealed_test_targets(**arguments)

    def test_authorized_access_loads_targets_and_records_single_use(self) -> None:
        result = self._load()

        self.assertEqual(len(result.targets), 2)
        self.assertEqual(result.targets[0].logical_example_id, "example-001")
        self.assertEqual(result.targets[0].target_value_float32, 0.125)
        self.assertEqual(result.targets[1].target_value_float32, 0.75)
        self.assertTrue(result.access_record_path.is_file())
        validate_hashed_manifest(result.access_record)
        self.assertEqual(
            result.access_record["authorization_manifest_hash"],
            self.authorization["manifest_hash"],
        )
        self.assertEqual(
            result.access_record["sealed_target_manifest_hash"],
            self.target_manifest["manifest_hash"],
        )
        self.assertEqual(
            result.access_record["record_kind"],
            "exclusive_single_use_access_claim",
        )
        self.assertTrue(result.access_record["claim_precedes_target_validation"])
        record_text = result.access_record_path.read_text(encoding="utf-8")
        self.assertNotIn("target_value_float32", record_text)
        self.assertNotIn("0.125", record_text)
        self.assertTrue(
            result.access_record_path.is_relative_to(
                (self.root / ACCESS_RECORD_DIRECTORY).resolve()
            )
        )

        with self.assertRaisesRegex(
            SealedTestAccessError,
            "already been used",
        ):
            self._load()

        reissued_same_id = self._build_authorization(
            expires_at_utc="2026-09-03T14:00:00Z",
        )
        _write_json(self.authorization_path, reissued_same_id)
        with self.assertRaisesRegex(
            SealedTestAccessError,
            "already been used",
        ):
            self._load()

    def test_policy_is_hashed_and_fixed_to_narrow_directories(self) -> None:
        loaded = load_test_access_policy(self.policy_path)
        self.assertEqual(loaded, build_test_access_policy())
        repository_root = Path(__file__).resolve().parents[1]
        checked_in_policy = load_test_access_policy(
            repository_root / "configs" / "exd_hox_test_access_policy_v1.yaml"
        )
        self.assertEqual(checked_in_policy, build_test_access_policy())

        tampered = dict(self.policy)
        tampered["sealed_target_directory"] = "data/sealed/other_v1"
        _write_yaml(self.policy_path, tampered)
        with self.assertRaisesRegex(SealedTestAccessError, "hash"):
            self._load()

        changed_content = dict(self.policy)
        changed_content.pop("manifest_hash")
        changed_content["sealed_target_directory"] = "data/sealed/other_v1"
        rehashed = build_hashed_manifest(
            str(changed_content.pop("schema_version")),
            changed_content,
        )
        _write_yaml(self.policy_path, rehashed)
        with self.assertRaisesRegex(SealedTestAccessError, "fixed versioned"):
            self._load()

    def test_target_commitments_match_the_split_builder(self) -> None:
        self.assertEqual(SEALED_TARGET_FIELDS, SPLIT_SEALED_TARGET_FIELDS)
        self.assertEqual(
            SEALED_TARGET_MANIFEST_FILENAME,
            SPLIT_TARGET_MANIFEST_FILENAME,
        )
        self.assertEqual(
            SEALED_TARGET_MANIFEST_SCHEMA_VERSION,
            SPLIT_TARGET_MANIFEST_SCHEMA,
        )
        self.assertEqual(
            target_commitment_sha256("example-001", "3e000000"),
            split_target_commitment("example-001", "3e000000"),
        )

    def test_missing_tampered_and_stale_authorizations_fail(self) -> None:
        missing_path = self.root / "authorizations" / "missing.json"
        with self.assertRaisesRegex(SealedTestAccessError, "missing"):
            self._load(authorization_path=missing_path)

        tampered = json.loads(self.authorization_path.read_text(encoding="utf-8"))
        tampered["identities"]["model_hash"] = _sha256("tampered model")
        _write_json(self.authorization_path, tampered)
        with self.assertRaisesRegex(SealedTestAccessError, "hash"):
            self._load()

        stale = self._build_authorization(
            authorization_id=_sha256("stale authorization"),
            issued_at_utc="2026-09-02T11:00:00Z",
            expires_at_utc="2026-09-02T13:00:00Z",
        )
        _write_json(self.authorization_path, stale)
        with self.assertRaisesRegex(SealedTestAccessError, "stale or expired"):
            self._load()

    def test_each_current_run_identity_must_match_authorization(self) -> None:
        for field in IDENTITY_FIELDS:
            with self.subTest(field=field):
                mismatched = dict(self.identities)
                mismatched[field] = _sha256("mismatch " + field)
                with self.assertRaisesRegex(
                    SealedTestAccessError,
                    field + " identity mismatch",
                ):
                    self._load(expected_identities=mismatched)

    def test_public_descriptor_is_hashed_and_bound_by_authorization(self) -> None:
        tampered = json.loads(
            self.target_manifest_path.read_text(encoding="utf-8")
        )
        tampered["test_logical_example_count"] = 3
        _write_json(self.target_manifest_path, tampered)
        with self.assertRaisesRegex(SealedTestAccessError, "manifest hash"):
            self._load()

        changed_manifest = self._build_target_manifest(
            split_identity_hash=_sha256("changed split identity"),
        )
        _write_json(self.target_manifest_path, changed_manifest)
        with self.assertRaisesRegex(
            SealedTestAccessError,
            "manifest identity mismatch",
        ):
            self._load()

        self._install_manifest_and_authorization(
            changed_manifest,
            "changed descriptor split identity",
        )
        with self.assertRaisesRegex(
            SealedTestAccessError,
            "descriptor split identity mismatch",
        ):
            self._load()

    def test_target_path_and_exact_bytes_are_descriptor_constrained(self) -> None:
        outside_path = self.root / "outside.tsv.gz"
        outside_path.write_bytes(self.target_bytes)
        outside_manifest = self._build_target_manifest(
            sealed_target_path="outside.tsv.gz",
        )
        self._install_manifest_and_authorization(
            outside_manifest,
            "outside authorization",
        )
        with self.assertRaisesRegex(
            SealedTestAccessError,
            "outside the versioned sealed directory",
        ):
            self._load()

        valid_manifest = self._build_target_manifest()
        self._install_manifest_and_authorization(
            valid_manifest,
            "tampered target authorization",
        )
        self.target_path.write_bytes(self.target_bytes + b"tamper")
        with self.assertRaisesRegex(
            SealedTestAccessError,
            "byte-size mismatch",
        ):
            self._load()
        claim_path = (
            self.root
            / ACCESS_RECORD_DIRECTORY
            / "test_access_{0}.json".format(
                self.authorization["authorization_id"]
            )
        )
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        self.assertEqual(claim["record_kind"], "exclusive_single_use_access_claim")
        self.assertTrue(claim["claim_precedes_target_validation"])
        with self.assertRaisesRegex(SealedTestAccessError, "already been used"):
            self._load()

    def test_gzip_mtime_and_embedded_filename_are_rejected(self) -> None:
        nonzero_mtime = _gzip_target_rows(self.target_rows, mtime=1)
        self._install_target_variant(nonzero_mtime, "nonzero mtime")
        with self.assertRaisesRegex(SealedTestAccessError, "mtime=0"):
            self._load()

        embedded_filename = _gzip_target_rows(
            self.target_rows,
            embedded_filename="targets.tsv",
        )
        self._install_target_variant(embedded_filename, "embedded filename")
        with self.assertRaisesRegex(SealedTestAccessError, "embed a filename"):
            self._load()

    def test_target_schema_bits_commitments_count_and_digest_are_checked(self) -> None:
        wrong_bits_rows = [dict(row) for row in self.target_rows]
        wrong_bits_rows[0]["target_bits_big_endian_hex"] = "3f000000"
        invalid_bits = _gzip_target_rows(wrong_bits_rows)
        self._install_target_variant(invalid_bits, "invalid bits")
        with self.assertRaisesRegex(SealedTestAccessError, "does not match"):
            self._load()

        noncanonical_rows = [dict(row) for row in self.target_rows]
        noncanonical_rows[0]["target_value_float32"] = "0.1250"
        noncanonical_target = _gzip_target_rows(noncanonical_rows)
        self._install_target_variant(noncanonical_target, "noncanonical value")
        with self.assertRaisesRegex(SealedTestAccessError, "canonical float32"):
            self._load()

        self.target_bytes = _gzip_target_rows(self.target_rows)
        self.target_path.write_bytes(self.target_bytes)
        wrong_count_manifest = self._build_target_manifest(
            test_logical_example_count=3,
        )
        self._install_manifest_and_authorization(
            wrong_count_manifest,
            "wrong count",
        )
        with self.assertRaisesRegex(SealedTestAccessError, "count mismatch"):
            self._load()

        wrong_digest_manifest = self._build_target_manifest(
            target_commitment_digest_sha256=_sha256("wrong digest"),
        )
        self._install_manifest_and_authorization(
            wrong_digest_manifest,
            "wrong digest",
        )
        with self.assertRaisesRegex(SealedTestAccessError, "digest mismatch"):
            self._load()

        out_of_range_rows = [dict(row) for row in self.target_rows]
        out_of_range_rows[0]["target_value_float32"] = "1.25"
        out_of_range_rows[0]["target_bits_big_endian_hex"] = "3fa00000"
        out_of_range_rows[0]["target_commitment_sha256"] = (
            target_commitment_sha256("example-001", "3fa00000")
        )
        out_of_range_target = _gzip_target_rows(out_of_range_rows)
        self._install_target_variant(out_of_range_target, "out of range")
        with self.assertRaisesRegex(SealedTestAccessError, r"\[0, 1\]"):
            self._load()

    def _install_manifest_and_authorization(
        self,
        manifest,
        authorization_label: str,
    ) -> None:
        self.target_manifest = manifest
        _write_json(self.target_manifest_path, manifest)
        self.authorization = self._build_authorization(
            authorization_id=_sha256(authorization_label),
        )
        _write_json(self.authorization_path, self.authorization)

    def _install_target_variant(self, target_bytes: bytes, label: str) -> None:
        self.target_bytes = target_bytes
        self.target_path.write_bytes(target_bytes)
        manifest = self._build_target_manifest()
        self._install_manifest_and_authorization(manifest, label)


def _valid_target_rows():
    rows = []
    for logical_example_id, value_text, target_bits in (
        ("example-001", "0.125", "3e000000"),
        ("example-002", "0.75", "3f400000"),
    ):
        rows.append(
            {
                "logical_example_id": logical_example_id,
                "target_value_float32": value_text,
                "target_bits_big_endian_hex": target_bits,
                "target_commitment_sha256": target_commitment_sha256(
                    logical_example_id,
                    target_bits,
                ),
            }
        )
    return rows


def _gzip_target_rows(
    rows,
    *,
    mtime: int = 0,
    embedded_filename: str = "",
) -> bytes:
    lines = ["\t".join(SEALED_TARGET_FIELDS)]
    for row in rows:
        values = []
        for field in SEALED_TARGET_FIELDS:
            values.append(str(row[field]))
        lines.append("\t".join(values))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    output = io.BytesIO()
    with gzip.GzipFile(
        filename=embedded_filename,
        mode="wb",
        fileobj=output,
        mtime=mtime,
    ) as gzip_file:
        gzip_file.write(payload)
    return output.getvalue()


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload) -> None:
    serialized = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")


def _write_yaml(path: Path, payload) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
