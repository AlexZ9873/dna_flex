"""Deterministic sequence-source fingerprints and split-overlap audits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.coordinates import normalize_sequence, reverse_complement


SOURCE_FINGERPRINT_SCHEMA_VERSION = "sequence_source_fingerprint.v2"
SPLIT_MANIFEST_SCHEMA_VERSION = "pretraining_split_manifest.v2"
OVERLAP_AUDIT_SCHEMA_VERSION = "sequence_overlap_audit.v1"
VERSIONED_ARTIFACT_NAME_PATTERN = re.compile(
    r"(^|[_-])v[0-9]+($|[_-])"
)


class SplitOverlapError(ValueError):
    """Raised when strict split auditing detects sequence leakage."""


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize logical content deterministically for hashing."""

    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def hash_logical_content(payload: Mapping[str, Any]) -> str:
    """Return a SHA-256 hash of canonical JSON logical content."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def hash_file_bytes(path: str) -> str:
    """Return the SHA-256 hash of the exact source-file bytes."""

    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reverse_complement_canonical_sequence(sequence: str) -> str:
    """Return a strand-independent key without changing the input sequence."""

    normalized = normalize_sequence(sequence)
    reverse = reverse_complement(normalized)
    if normalized <= reverse:
        return normalized
    return reverse


def repository_relative_source_path(path: str, repository_root: str) -> str:
    """Return a stable POSIX path relative to the repository root."""

    resolved_path = Path(path).resolve()
    resolved_root = Path(repository_root).resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        message = "Sequence source is outside repository root: {0}"
        raise ValueError(message.format(resolved_path)) from error
    return relative_path.as_posix()


@dataclass(frozen=True)
class SourceFingerprint:
    """Deterministic metadata for one sequence source file."""

    schema_version: str
    source_path: str
    source_file_sha256: str
    logical_sequence_sha256: str
    total_rows: int
    unique_sequences: int
    minimum_sequence_length: Optional[int]
    maximum_sequence_length: Optional[int]
    observed_sequence_lengths: Tuple[int, ...]
    sequences_containing_n: int
    exact_sequence_duplicate_count: int
    reverse_complement_canonical_duplicate_count: int
    fingerprint_hash: str

    def content_dict(self) -> Dict[str, Any]:
        """Return hashable content excluding the stored fingerprint hash."""

        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "source_file_sha256": self.source_file_sha256,
            "logical_sequence_sha256": self.logical_sequence_sha256,
            "total_rows": self.total_rows,
            "unique_sequences": self.unique_sequences,
            "minimum_sequence_length": self.minimum_sequence_length,
            "maximum_sequence_length": self.maximum_sequence_length,
            "observed_sequence_lengths": list(self.observed_sequence_lengths),
            "sequences_containing_n": self.sequences_containing_n,
            "exact_sequence_duplicate_count": (
                self.exact_sequence_duplicate_count
            ),
            "reverse_complement_canonical_duplicate_count": (
                self.reverse_complement_canonical_duplicate_count
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return complete serialized content."""

        payload = self.content_dict()
        payload["fingerprint_hash"] = self.fingerprint_hash
        return payload

    def validate_hash(self) -> None:
        """Fail if serialized metadata does not match its stored hash."""

        expected_hash = hash_logical_content(self.content_dict())
        if self.fingerprint_hash != expected_hash:
            raise ValueError("Sequence-source fingerprint hash mismatch.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceFingerprint":
        """Reconstruct and validate a source fingerprint."""

        schema_version = str(payload["schema_version"])
        if schema_version != SOURCE_FINGERPRINT_SCHEMA_VERSION:
            message = "Unsupported source-fingerprint schema version: {0}"
            raise ValueError(message.format(schema_version))
        fingerprint = cls(
            schema_version=schema_version,
            source_path=str(payload["source_path"]),
            source_file_sha256=str(payload["source_file_sha256"]),
            logical_sequence_sha256=str(payload["logical_sequence_sha256"]),
            total_rows=int(payload["total_rows"]),
            unique_sequences=int(payload["unique_sequences"]),
            minimum_sequence_length=_optional_int(
                payload["minimum_sequence_length"]
            ),
            maximum_sequence_length=_optional_int(
                payload["maximum_sequence_length"]
            ),
            observed_sequence_lengths=tuple(
                int(value) for value in payload["observed_sequence_lengths"]
            ),
            sequences_containing_n=int(payload["sequences_containing_n"]),
            exact_sequence_duplicate_count=int(
                payload["exact_sequence_duplicate_count"]
            ),
            reverse_complement_canonical_duplicate_count=int(
                payload["reverse_complement_canonical_duplicate_count"]
            ),
            fingerprint_hash=str(payload["fingerprint_hash"]),
        )
        fingerprint.validate_hash()
        return fingerprint


@dataclass(frozen=True)
class OverlapExample:
    """One deterministically selected cross-split equivalence group."""

    group_key: str
    training_sequences: Tuple[str, ...]
    validation_sequences: Tuple[str, ...]
    training_row_count: int
    validation_row_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Return serialized overlap-example content."""

        return {
            "group_key": self.group_key,
            "training_sequences": list(self.training_sequences),
            "validation_sequences": list(self.validation_sequences),
            "training_row_count": self.training_row_count,
            "validation_row_count": self.validation_row_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OverlapExample":
        """Reconstruct an overlap example."""

        return cls(
            group_key=str(payload["group_key"]),
            training_sequences=tuple(
                str(value) for value in payload["training_sequences"]
            ),
            validation_sequences=tuple(
                str(value) for value in payload["validation_sequences"]
            ),
            training_row_count=int(payload["training_row_count"]),
            validation_row_count=int(payload["validation_row_count"]),
        )


@dataclass(frozen=True)
class OverlapSummary:
    """Counts and representative examples for one overlap definition."""

    group_count: int
    training_row_count: int
    validation_row_count: int
    representative_examples: Tuple[OverlapExample, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return serialized overlap-summary content."""

        examples = []
        for example in self.representative_examples:
            examples.append(example.to_dict())
        return {
            "group_count": self.group_count,
            "training_row_count": self.training_row_count,
            "validation_row_count": self.validation_row_count,
            "representative_examples": examples,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OverlapSummary":
        """Reconstruct an overlap summary."""

        examples = []
        for example_payload in payload["representative_examples"]:
            examples.append(OverlapExample.from_dict(example_payload))
        return cls(
            group_count=int(payload["group_count"]),
            training_row_count=int(payload["training_row_count"]),
            validation_row_count=int(payload["validation_row_count"]),
            representative_examples=tuple(examples),
        )


@dataclass(frozen=True)
class CrossSplitOverlapAudit:
    """Exact and reverse-complement-canonical overlap results."""

    schema_version: str
    exact_sequence_overlap: OverlapSummary
    reverse_complement_equivalent_overlap: OverlapSummary
    reverse_complement_only_group_count: int
    reverse_complement_overlap_includes_exact_matches: bool

    @property
    def has_overlap(self) -> bool:
        """Return whether either overlap definition found leakage."""

        return (
            self.exact_sequence_overlap.group_count > 0
            or self.reverse_complement_equivalent_overlap.group_count > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return serialized overlap-audit content."""

        return {
            "schema_version": self.schema_version,
            "exact_sequence_overlap": self.exact_sequence_overlap.to_dict(),
            "reverse_complement_equivalent_overlap": (
                self.reverse_complement_equivalent_overlap.to_dict()
            ),
            "reverse_complement_only_group_count": (
                self.reverse_complement_only_group_count
            ),
            "reverse_complement_overlap_includes_exact_matches": (
                self.reverse_complement_overlap_includes_exact_matches
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrossSplitOverlapAudit":
        """Reconstruct and validate an overlap audit."""

        audit = cls(
            schema_version=str(payload["schema_version"]),
            exact_sequence_overlap=OverlapSummary.from_dict(
                payload["exact_sequence_overlap"]
            ),
            reverse_complement_equivalent_overlap=OverlapSummary.from_dict(
                payload["reverse_complement_equivalent_overlap"]
            ),
            reverse_complement_only_group_count=int(
                payload["reverse_complement_only_group_count"]
            ),
            reverse_complement_overlap_includes_exact_matches=bool(
                payload[
                    "reverse_complement_overlap_includes_exact_matches"
                ]
            ),
        )
        if audit.schema_version != OVERLAP_AUDIT_SCHEMA_VERSION:
            message = "Unsupported overlap-audit schema version: {0}"
            raise ValueError(message.format(audit.schema_version))
        return audit


@dataclass(frozen=True)
class PretrainingSplitManifest:
    """Fingerprint-bound training and validation split representation."""

    schema_version: str
    training_source: SourceFingerprint
    validation_source: SourceFingerprint
    overlap_audit: CrossSplitOverlapAudit
    manifest_hash: str

    def content_dict(self) -> Dict[str, Any]:
        """Return hashable manifest content without its stored hash."""

        return {
            "schema_version": self.schema_version,
            "training_source": self.training_source.to_dict(),
            "validation_source": self.validation_source.to_dict(),
            "overlap_audit": self.overlap_audit.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return complete serialized manifest content."""

        payload = self.content_dict()
        payload["manifest_hash"] = self.manifest_hash
        return payload

    def validate_hashes(self) -> None:
        """Validate nested source fingerprints and the manifest hash."""

        self.training_source.validate_hash()
        self.validation_source.validate_hash()
        expected_hash = hash_logical_content(self.content_dict())
        if self.manifest_hash != expected_hash:
            raise ValueError("Pretraining split-manifest hash mismatch.")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PretrainingSplitManifest":
        """Reconstruct and validate a pretraining split manifest."""

        schema_version = str(payload["schema_version"])
        if schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
            message = "Unsupported split-manifest schema version: {0}"
            raise ValueError(message.format(schema_version))
        manifest = cls(
            schema_version=schema_version,
            training_source=SourceFingerprint.from_dict(
                payload["training_source"]
            ),
            validation_source=SourceFingerprint.from_dict(
                payload["validation_source"]
            ),
            overlap_audit=CrossSplitOverlapAudit.from_dict(
                payload["overlap_audit"]
            ),
            manifest_hash=str(payload["manifest_hash"]),
        )
        manifest.validate_hashes()
        return manifest


@dataclass(frozen=True)
class _SequenceFileScan:
    fingerprint: SourceFingerprint
    exact_counts: Mapping[str, int]
    canonical_members: Mapping[str, Mapping[str, int]]


def fingerprint_sequence_file(
    path: str,
    repository_root: str,
) -> SourceFingerprint:
    """Fingerprint a sequence file without modifying its content."""

    return _scan_sequence_file(path, repository_root).fingerprint


def build_pretraining_split_manifest(
    training_path: str,
    validation_path: str,
    repository_root: str,
    mode: str = "report",
    maximum_examples: int = 10,
) -> PretrainingSplitManifest:
    """Fingerprint and audit training/validation files deterministically."""

    if mode not in ("report", "strict"):
        raise ValueError("Split audit mode must be 'report' or 'strict'.")
    if maximum_examples < 0:
        raise ValueError("Maximum overlap examples must be non-negative.")

    training_scan = _scan_sequence_file(training_path, repository_root)
    validation_scan = _scan_sequence_file(validation_path, repository_root)
    overlap_audit = _audit_scans(
        training_scan,
        validation_scan,
        maximum_examples,
    )
    manifest_content = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "training_source": training_scan.fingerprint.to_dict(),
        "validation_source": validation_scan.fingerprint.to_dict(),
        "overlap_audit": overlap_audit.to_dict(),
    }
    manifest = PretrainingSplitManifest(
        schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        training_source=training_scan.fingerprint,
        validation_source=validation_scan.fingerprint,
        overlap_audit=overlap_audit,
        manifest_hash=hash_logical_content(manifest_content),
    )
    if mode == "strict" and overlap_audit.has_overlap:
        exact_count = overlap_audit.exact_sequence_overlap.group_count
        reverse_count = (
            overlap_audit.reverse_complement_equivalent_overlap.group_count
        )
        message = (
            "Strict split audit failed: {0} exact overlap groups and "
            "{1} reverse-complement-equivalent overlap groups."
        )
        raise SplitOverlapError(message.format(exact_count, reverse_count))
    return manifest


def save_split_manifest(
    manifest: PretrainingSplitManifest,
    path: str,
) -> None:
    """Save deterministic JSON without overwriting an existing file."""

    manifest.validate_hashes()
    _write_json_exclusive(path, manifest.to_dict())


def load_split_manifest(path: str) -> PretrainingSplitManifest:
    """Load a split manifest and verify all stored hashes."""

    with open(path, "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    return PretrainingSplitManifest.from_dict(payload)


def validate_new_artifact_output_path(
    path: str,
    repository_root: str,
    allowed_relative_directories: Sequence[str],
    input_paths: Sequence[str] = (),
) -> str:
    """Validate a versioned, repository-local output path without creating it."""

    output_path = Path(path).resolve()
    resolved_root = Path(repository_root).resolve()
    try:
        output_path.relative_to(resolved_root)
    except ValueError as error:
        message = "Artifact output is outside repository root: {0}"
        raise ValueError(message.format(output_path)) from error

    if output_path.suffix.lower() != ".json":
        raise ValueError("Artifact output must use a .json suffix.")
    if VERSIONED_ARTIFACT_NAME_PATTERN.search(output_path.stem) is None:
        raise ValueError(
            "Artifact output filename must contain a version such as '_v1'."
        )
    for input_path in input_paths:
        if output_path == Path(input_path).resolve():
            raise ValueError("Artifact output path must differ from every input.")
    if output_path.exists():
        message = "Artifact output already exists: {0}"
        raise FileExistsError(message.format(output_path))

    is_in_allowed_directory = False
    for relative_directory in allowed_relative_directories:
        allowed_directory = (resolved_root / relative_directory).resolve()
        try:
            output_path.relative_to(allowed_directory)
            is_in_allowed_directory = True
        except ValueError:
            pass
    if not is_in_allowed_directory:
        allowed_text = ", ".join(allowed_relative_directories)
        message = "Artifact output must be under one of: {0}"
        raise ValueError(message.format(allowed_text))
    return str(output_path)


def _scan_sequence_file(path: str, repository_root: str) -> _SequenceFileScan:
    source_path = repository_relative_source_path(path, repository_root)
    source_file_digest = hashlib.sha256()
    exact_counts: Dict[str, int] = {}
    canonical_members: Dict[str, Dict[str, int]] = {}
    observed_lengths = set()
    total_rows = 0
    sequences_containing_n = 0
    logical_sequence_digest = hashlib.sha256()

    with open(path, "rb") as sequence_file:
        for line_number, raw_line in enumerate(sequence_file, start=1):
            source_file_digest.update(raw_line)
            line = raw_line.decode("utf-8")
            stripped = line.strip()
            if not stripped:
                message = "Blank sequence row at {0}:{1}"
                raise ValueError(message.format(source_path, line_number))
            sequence = normalize_sequence(stripped)
            sequence_bytes = sequence.encode("ascii")
            logical_sequence_digest.update(
                len(sequence_bytes).to_bytes(8, byteorder="big")
            )
            logical_sequence_digest.update(sequence_bytes)
            total_rows += 1
            observed_lengths.add(len(sequence))
            if "N" in sequence:
                sequences_containing_n += 1
            exact_counts[sequence] = exact_counts.get(sequence, 0) + 1

            canonical = reverse_complement_canonical_sequence(sequence)
            if canonical not in canonical_members:
                canonical_members[canonical] = {}
            member_counts = canonical_members[canonical]
            member_counts[sequence] = member_counts.get(sequence, 0) + 1

    source_file_hash = source_file_digest.hexdigest()
    sorted_lengths = tuple(sorted(observed_lengths))
    minimum_length = None
    maximum_length = None
    if sorted_lengths:
        minimum_length = sorted_lengths[0]
        maximum_length = sorted_lengths[-1]
    fingerprint_content = {
        "schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "source_path": source_path,
        "source_file_sha256": source_file_hash,
        "logical_sequence_sha256": logical_sequence_digest.hexdigest(),
        "total_rows": total_rows,
        "unique_sequences": len(exact_counts),
        "minimum_sequence_length": minimum_length,
        "maximum_sequence_length": maximum_length,
        "observed_sequence_lengths": list(sorted_lengths),
        "sequences_containing_n": sequences_containing_n,
        "exact_sequence_duplicate_count": total_rows - len(exact_counts),
        "reverse_complement_canonical_duplicate_count": (
            total_rows - len(canonical_members)
        ),
    }
    fingerprint = SourceFingerprint(
        schema_version=SOURCE_FINGERPRINT_SCHEMA_VERSION,
        source_path=source_path,
        source_file_sha256=source_file_hash,
        logical_sequence_sha256=logical_sequence_digest.hexdigest(),
        total_rows=total_rows,
        unique_sequences=len(exact_counts),
        minimum_sequence_length=minimum_length,
        maximum_sequence_length=maximum_length,
        observed_sequence_lengths=sorted_lengths,
        sequences_containing_n=sequences_containing_n,
        exact_sequence_duplicate_count=total_rows - len(exact_counts),
        reverse_complement_canonical_duplicate_count=(
            total_rows - len(canonical_members)
        ),
        fingerprint_hash=hash_logical_content(fingerprint_content),
    )
    return _SequenceFileScan(
        fingerprint=fingerprint,
        exact_counts=exact_counts,
        canonical_members=canonical_members,
    )


def _audit_scans(
    training_scan: _SequenceFileScan,
    validation_scan: _SequenceFileScan,
    maximum_examples: int,
) -> CrossSplitOverlapAudit:
    exact_keys = sorted(
        set(training_scan.exact_counts).intersection(
            validation_scan.exact_counts
        )
    )
    canonical_keys = sorted(
        set(training_scan.canonical_members).intersection(
            validation_scan.canonical_members
        )
    )
    exact_summary = _summarize_exact_overlap(
        exact_keys,
        training_scan.exact_counts,
        validation_scan.exact_counts,
        maximum_examples,
    )
    canonical_summary = _summarize_canonical_overlap(
        canonical_keys,
        training_scan.canonical_members,
        validation_scan.canonical_members,
        maximum_examples,
    )
    reverse_only_count = 0
    for canonical_key in canonical_keys:
        training_sequences = set(
            training_scan.canonical_members[canonical_key]
        )
        validation_sequences = set(
            validation_scan.canonical_members[canonical_key]
        )
        if not training_sequences.intersection(validation_sequences):
            reverse_only_count += 1

    return CrossSplitOverlapAudit(
        schema_version=OVERLAP_AUDIT_SCHEMA_VERSION,
        exact_sequence_overlap=exact_summary,
        reverse_complement_equivalent_overlap=canonical_summary,
        reverse_complement_only_group_count=reverse_only_count,
        reverse_complement_overlap_includes_exact_matches=True,
    )


def _summarize_exact_overlap(
    keys: Sequence[str],
    training_counts: Mapping[str, int],
    validation_counts: Mapping[str, int],
    maximum_examples: int,
) -> OverlapSummary:
    examples = []
    training_row_count = 0
    validation_row_count = 0
    for key in keys:
        training_count = training_counts[key]
        validation_count = validation_counts[key]
        training_row_count += training_count
        validation_row_count += validation_count
        if len(examples) < maximum_examples:
            examples.append(
                OverlapExample(
                    group_key=key,
                    training_sequences=(key,),
                    validation_sequences=(key,),
                    training_row_count=training_count,
                    validation_row_count=validation_count,
                )
            )
    return OverlapSummary(
        group_count=len(keys),
        training_row_count=training_row_count,
        validation_row_count=validation_row_count,
        representative_examples=tuple(examples),
    )


def _summarize_canonical_overlap(
    keys: Sequence[str],
    training_members: Mapping[str, Mapping[str, int]],
    validation_members: Mapping[str, Mapping[str, int]],
    maximum_examples: int,
) -> OverlapSummary:
    examples = []
    training_row_count = 0
    validation_row_count = 0
    for key in keys:
        training_group = training_members[key]
        validation_group = validation_members[key]
        training_count = sum(training_group.values())
        validation_count = sum(validation_group.values())
        training_row_count += training_count
        validation_row_count += validation_count
        if len(examples) < maximum_examples:
            examples.append(
                OverlapExample(
                    group_key=key,
                    training_sequences=tuple(sorted(training_group)),
                    validation_sequences=tuple(sorted(validation_group)),
                    training_row_count=training_count,
                    validation_row_count=validation_count,
                )
            )
    return OverlapSummary(
        group_count=len(keys),
        training_row_count=training_row_count,
        validation_row_count=validation_row_count,
        representative_examples=tuple(examples),
    )


def _write_json_exclusive(path: str, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    with open(output_path, "x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(text)
        output_file.write("\n")


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)
