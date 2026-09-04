"""Deterministic fingerprints and exclusive writers for downstream artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize logical content deterministically for hashing."""

    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def hash_logical_content(payload: Any) -> str:
    """Return the SHA-256 digest of canonical JSON logical content."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def hash_file_bytes(path: Path | str) -> str:
    """Return the SHA-256 digest of exact file bytes."""

    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_repository_relative_path(path: str) -> str:
    """Validate and normalize a repository-relative POSIX path."""

    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("Artifact paths must be repository-relative.")
    if ".." in candidate.parts:
        raise ValueError("Artifact paths must not escape the repository root.")
    normalized = candidate.as_posix()
    if normalized in ("", "."):
        raise ValueError("Artifact path must name a file or directory.")
    return normalized


def repository_relative_path(path: Path | str, repository_root: Path | str) -> str:
    """Return a stable repository-relative POSIX path."""

    resolved_path = Path(path).resolve()
    resolved_root = Path(repository_root).resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        message = "Path is outside the repository root: {0}"
        raise ValueError(message.format(resolved_path)) from error
    return validate_repository_relative_path(relative_path.as_posix())


@dataclass(frozen=True)
class FileFingerprint:
    """Byte identity for one repository artifact."""

    path: str
    byte_size: int
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        """Return deterministic serialized content."""

        return {
            "path": self.path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


def fingerprint_file(
    physical_path: Path | str,
    logical_path: str,
) -> FileFingerprint:
    """Fingerprint bytes while recording only a supplied logical path."""

    path = Path(physical_path)
    normalized_logical_path = validate_repository_relative_path(logical_path)
    return FileFingerprint(
        path=normalized_logical_path,
        byte_size=path.stat().st_size,
        sha256=hash_file_bytes(path),
    )


def build_hashed_manifest(
    schema_version: str,
    content: Mapping[str, Any],
) -> Dict[str, Any]:
    """Add a schema and deterministic manifest hash to logical content."""

    if not schema_version:
        raise ValueError("Manifest schema version must not be empty.")
    payload = {"schema_version": schema_version}
    payload.update(dict(content))
    if "manifest_hash" in payload:
        raise ValueError("Manifest content must not define manifest_hash.")
    payload["manifest_hash"] = hash_logical_content(payload)
    return payload


def validate_hashed_manifest(payload: Mapping[str, Any]) -> None:
    """Fail when stored manifest content does not match its hash."""

    content = dict(payload)
    stored_hash = str(content.pop("manifest_hash"))
    expected_hash = hash_logical_content(content)
    if stored_hash != expected_hash:
        raise ValueError("Manifest hash mismatch.")


def write_json_exclusive(path: Path | str, payload: Any) -> None:
    """Write deterministic JSON and refuse to overwrite an artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    with open(output_path, "x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(serialized)
        output_file.write("\n")


def write_tsv_exclusive(
    path: Path | str,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write a deterministic TSV and refuse to overwrite an artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_tsv_gzip_exclusive(
    path: Path | str,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write deterministic gzip-compressed TSV and refuse overwrite."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "xb") as compressed_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=compressed_file,
            mtime=0,
        ) as gzip_file:
            with io.TextIOWrapper(
                gzip_file,
                encoding="utf-8",
                newline="",
            ) as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=fieldnames,
                    delimiter="\t",
                    lineterminator="\n",
                    extrasaction="raise",
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
