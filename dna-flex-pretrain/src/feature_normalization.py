"""Training-only native-coordinate biophysical normalization artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from src.coordinates import normalize_sequence
from src.data_fingerprints import (
    PretrainingSplitManifest,
    SourceFingerprint,
    fingerprint_sequence_file,
    hash_logical_content,
)
from src.feature_schema import (
    BiophysicalFeatureProvider,
    FeatureBatch,
    FeatureCoordinateType,
    FeatureSchema,
    FeatureTrack,
)


NORMALIZATION_ARTIFACT_SCHEMA_VERSION = "biophysical_normalization.v2"
FEATURE_SCHEMA_HASH_VERSION = "feature_schema_hash.v1"
STANDARD_DEVIATION_CONVENTION = "population_standard_deviation_ddof_0"
FIT_VALUE_HANDLING = (
    "use_only_provider-valid finite values; exclude NaN, missing lookup, "
    "ambiguous-base, and boundary-invalid positions"
)
TRANSFORM_VALUE_HANDLING = (
    "preserve invalid positions as NaN with a false validity mask; "
    "never zero-impute"
)


class NormalizationCompatibilityError(ValueError):
    """Raised when an artifact does not match the expected experiment."""


class NormalizationFitError(ValueError):
    """Raised when a feature cannot produce valid normalization statistics."""


@dataclass(frozen=True)
class FeatureNormalizationStatistics:
    """Training-only statistics for one ordered physical feature."""

    name: str
    display_name: str
    feature_order: int
    coordinate_type: str
    unit: str
    valid_count: int
    mean: float
    standard_deviation: float

    def to_dict(self) -> Dict[str, Any]:
        """Return deterministic serialized statistics."""

        return {
            "name": self.name,
            "display_name": self.display_name,
            "feature_order": self.feature_order,
            "coordinate_type": self.coordinate_type,
            "unit": self.unit,
            "valid_count": self.valid_count,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "FeatureNormalizationStatistics":
        """Reconstruct and validate one feature's statistics."""

        statistics = cls(
            name=str(payload["name"]),
            display_name=str(payload["display_name"]),
            feature_order=int(payload["feature_order"]),
            coordinate_type=str(payload["coordinate_type"]),
            unit=str(payload["unit"]),
            valid_count=int(payload["valid_count"]),
            mean=float(payload["mean"]),
            standard_deviation=float(payload["standard_deviation"]),
        )
        _validate_finished_statistics(statistics)
        return statistics


@dataclass(frozen=True)
class BiophysicalNormalizationArtifact:
    """Versioned, fingerprint-bound normalization artifact."""

    artifact_schema_version: str
    creation_entry_point: str
    training_source_fingerprint: SourceFingerprint
    experiment_split_manifest_hash: str
    feature_schema_hash: str
    feature_provider_identifier: str
    feature_provider_source_version: str
    feature_provider_fingerprint: str
    feature_names: Tuple[str, ...]
    feature_order: Tuple[int, ...]
    display_names: Tuple[str, ...]
    coordinate_types: Tuple[str, ...]
    units: Tuple[str, ...]
    statistics: Tuple[FeatureNormalizationStatistics, ...]
    standard_deviation_convention: str
    fit_value_handling: str
    transform_value_handling: str
    artifact_hash: str

    def content_dict(self) -> Dict[str, Any]:
        """Return artifact content without the stored artifact hash."""

        serialized_statistics = []
        for feature_statistics in self.statistics:
            serialized_statistics.append(feature_statistics.to_dict())
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "creation_entry_point": self.creation_entry_point,
            "training_source_fingerprint": (
                self.training_source_fingerprint.to_dict()
            ),
            "experiment_split_manifest_hash": (
                self.experiment_split_manifest_hash
            ),
            "feature_schema_hash": self.feature_schema_hash,
            "feature_provider_identifier": self.feature_provider_identifier,
            "feature_provider_source_version": (
                self.feature_provider_source_version
            ),
            "feature_provider_fingerprint": (
                self.feature_provider_fingerprint
            ),
            "feature_names": list(self.feature_names),
            "feature_order": list(self.feature_order),
            "display_names": list(self.display_names),
            "coordinate_types": list(self.coordinate_types),
            "units": list(self.units),
            "statistics": serialized_statistics,
            "standard_deviation_convention": (
                self.standard_deviation_convention
            ),
            "fit_value_handling": self.fit_value_handling,
            "transform_value_handling": self.transform_value_handling,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return complete artifact content."""

        payload = self.content_dict()
        payload["artifact_hash"] = self.artifact_hash
        return payload

    def validate_internal_consistency(self) -> None:
        """Validate artifact schema, arrays, statistics, and hashes."""

        if self.artifact_schema_version != NORMALIZATION_ARTIFACT_SCHEMA_VERSION:
            message = "Unsupported normalization artifact schema version: {0}"
            raise ValueError(message.format(self.artifact_schema_version))
        if not self.creation_entry_point:
            raise ValueError(
                "Normalization artifact creation entry point is empty."
            )
        if not self.feature_provider_identifier:
            raise ValueError(
                "Normalization artifact provider identifier is empty."
            )
        if not self.feature_provider_source_version:
            raise ValueError(
                "Normalization artifact provider source version is empty."
            )
        if not self.feature_provider_fingerprint:
            raise ValueError(
                "Normalization artifact provider fingerprint is empty."
            )
        if not self.experiment_split_manifest_hash:
            raise ValueError(
                "Normalization artifact experiment manifest hash is empty."
            )
        if (
            self.standard_deviation_convention
            != STANDARD_DEVIATION_CONVENTION
        ):
            raise ValueError(
                "Unsupported normalization standard-deviation convention."
            )
        self.training_source_fingerprint.validate_hash()

        expected_length = len(self.statistics)
        aligned_fields = (
            self.feature_names,
            self.feature_order,
            self.display_names,
            self.coordinate_types,
            self.units,
        )
        for field_values in aligned_fields:
            if len(field_values) != expected_length:
                raise ValueError(
                    "Normalization artifact feature metadata length mismatch."
                )
        if len(set(self.feature_names)) != expected_length:
            raise ValueError(
                "Normalization artifact feature names must be unique."
            )
        if self.feature_order != tuple(range(expected_length)):
            raise ValueError(
                "Normalization artifact feature order must be contiguous."
            )
        valid_coordinate_types = set(
            coordinate_type.value
            for coordinate_type in FeatureCoordinateType
        )
        for coordinate_type in self.coordinate_types:
            if coordinate_type not in valid_coordinate_types:
                raise ValueError(
                    "Normalization artifact has an unknown coordinate type."
                )

        for index, feature_statistics in enumerate(self.statistics):
            _validate_finished_statistics(feature_statistics)
            if feature_statistics.name != self.feature_names[index]:
                raise ValueError("Normalization feature-name alignment mismatch.")
            if feature_statistics.feature_order != self.feature_order[index]:
                raise ValueError("Normalization feature-order alignment mismatch.")
            if (
                feature_statistics.display_name
                != self.display_names[index]
            ):
                raise ValueError(
                    "Normalization display-name alignment mismatch."
                )
            if (
                feature_statistics.coordinate_type
                != self.coordinate_types[index]
            ):
                raise ValueError(
                    "Normalization coordinate-type alignment mismatch."
                )
            if feature_statistics.unit != self.units[index]:
                raise ValueError("Normalization unit alignment mismatch.")

        expected_hash = hash_logical_content(self.content_dict())
        if self.artifact_hash != expected_hash:
            raise ValueError("Normalization artifact hash mismatch.")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "BiophysicalNormalizationArtifact":
        """Reconstruct and validate a normalization artifact."""

        artifact_schema_version = str(payload["artifact_schema_version"])
        if artifact_schema_version != NORMALIZATION_ARTIFACT_SCHEMA_VERSION:
            message = "Unsupported normalization artifact schema version: {0}"
            raise ValueError(message.format(artifact_schema_version))
        statistics = []
        for statistics_payload in payload["statistics"]:
            statistics.append(
                FeatureNormalizationStatistics.from_dict(statistics_payload)
            )
        artifact = cls(
            artifact_schema_version=artifact_schema_version,
            creation_entry_point=str(payload["creation_entry_point"]),
            training_source_fingerprint=SourceFingerprint.from_dict(
                payload["training_source_fingerprint"]
            ),
            experiment_split_manifest_hash=str(
                payload["experiment_split_manifest_hash"]
            ),
            feature_schema_hash=str(payload["feature_schema_hash"]),
            feature_provider_identifier=str(
                payload["feature_provider_identifier"]
            ),
            feature_provider_source_version=str(
                payload["feature_provider_source_version"]
            ),
            feature_provider_fingerprint=str(
                payload["feature_provider_fingerprint"]
            ),
            feature_names=tuple(
                str(value) for value in payload["feature_names"]
            ),
            feature_order=tuple(
                int(value) for value in payload["feature_order"]
            ),
            display_names=tuple(
                str(value) for value in payload["display_names"]
            ),
            coordinate_types=tuple(
                str(value) for value in payload["coordinate_types"]
            ),
            units=tuple(str(value) for value in payload["units"]),
            statistics=tuple(statistics),
            standard_deviation_convention=str(
                payload["standard_deviation_convention"]
            ),
            fit_value_handling=str(payload["fit_value_handling"]),
            transform_value_handling=str(
                payload["transform_value_handling"]
            ),
            artifact_hash=str(payload["artifact_hash"]),
        )
        artifact.validate_internal_consistency()
        return artifact


@dataclass
class _RunningFeatureStatistics:
    count: int = 0
    mean: float = 0.0
    sum_squared_deviations: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        """Combine a one-dimensional finite value batch using Welford moments."""

        if values.numel() == 0:
            return
        values_64 = values.to(dtype=torch.float64)
        batch_count = int(values_64.numel())
        batch_mean = float(values_64.mean().item())
        centered = values_64 - batch_mean
        batch_sum_squared_deviations = float(
            torch.sum(centered * centered).item()
        )

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.sum_squared_deviations = batch_sum_squared_deviations
            return

        previous_count = self.count
        combined_count = previous_count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / combined_count)
        correction = (
            delta
            * delta
            * previous_count
            * batch_count
            / combined_count
        )
        self.sum_squared_deviations = (
            self.sum_squared_deviations
            + batch_sum_squared_deviations
            + correction
        )
        self.count = combined_count


class FeatureNormalizer:
    """Strict transform/inverse-transform operations for one artifact."""

    def __init__(self, artifact: BiophysicalNormalizationArtifact) -> None:
        artifact.validate_internal_consistency()
        self.artifact = artifact
        self._statistics_by_order = {}
        for statistics in artifact.statistics:
            self._statistics_by_order[statistics.feature_order] = statistics

    def transform(self, feature_batch: FeatureBatch) -> FeatureBatch:
        """Standardize valid values and preserve invalid positions as masked NaN."""

        self._validate_feature_batch(feature_batch)
        transformed_tracks = []
        for track in feature_batch.tracks:
            transformed_tracks.append(self._transform_track(track, inverse=False))
        return FeatureBatch(
            sequence=feature_batch.sequence,
            coordinates=feature_batch.coordinates,
            tracks=tuple(transformed_tracks),
            canonical_orientation=feature_batch.canonical_orientation,
            is_reverse_complement_palindrome=(
                feature_batch.is_reverse_complement_palindrome
            ),
        )

    def inverse_transform(self, feature_batch: FeatureBatch) -> FeatureBatch:
        """Restore valid standardized values to their original physical scale."""

        self._validate_feature_batch(feature_batch)
        restored_tracks = []
        for track in feature_batch.tracks:
            restored_tracks.append(self._transform_track(track, inverse=True))
        return FeatureBatch(
            sequence=feature_batch.sequence,
            coordinates=feature_batch.coordinates,
            tracks=tuple(restored_tracks),
            canonical_orientation=feature_batch.canonical_orientation,
            is_reverse_complement_palindrome=(
                feature_batch.is_reverse_complement_palindrome
            ),
        )

    def _validate_feature_batch(self, feature_batch: FeatureBatch) -> None:
        batch_schemas = []
        for track in feature_batch.tracks:
            for schema in track.schemas:
                batch_schemas.append(schema)
        ordered_schemas = _ordered_feature_schemas(batch_schemas)
        if (
            feature_schema_hash(ordered_schemas)
            != self.artifact.feature_schema_hash
        ):
            raise NormalizationCompatibilityError(
                "Feature batch schema hash does not match artifact."
            )
        names = tuple(schema.name for schema in ordered_schemas)
        orders = tuple(schema.feature_order for schema in ordered_schemas)
        coordinate_types = tuple(
            schema.coordinate_type.value for schema in ordered_schemas
        )
        if names != self.artifact.feature_names:
            raise NormalizationCompatibilityError(
                "Feature batch names do not match normalization artifact."
            )
        if orders != self.artifact.feature_order:
            raise NormalizationCompatibilityError(
                "Feature batch order does not match normalization artifact."
            )
        if coordinate_types != self.artifact.coordinate_types:
            raise NormalizationCompatibilityError(
                "Feature batch coordinate types do not match artifact."
            )

    def _transform_track(
        self,
        track: FeatureTrack,
        inverse: bool,
    ) -> FeatureTrack:
        output_values = torch.full_like(track.values, float("nan"))
        output_mask = track.valid_mask & torch.isfinite(track.values)

        for feature_index, schema in enumerate(track.schemas):
            statistics = self._statistics_by_order[schema.feature_order]
            feature_mask = output_mask[:, feature_index]
            input_values = track.values[:, feature_index]
            if inverse:
                output_values[feature_mask, feature_index] = (
                    input_values[feature_mask]
                    * statistics.standard_deviation
                    + statistics.mean
                )
            else:
                output_values[feature_mask, feature_index] = (
                    input_values[feature_mask] - statistics.mean
                ) / statistics.standard_deviation

        return FeatureTrack(
            coordinate_type=track.coordinate_type,
            coordinate_positions=track.coordinate_positions.clone(),
            schemas=track.schemas,
            values=output_values,
            valid_mask=output_mask.clone(),
            support_spans=track.support_spans.clone(),
        )


def feature_schema_hash(schemas: Sequence[FeatureSchema]) -> str:
    """Hash exact ordered schema metadata deterministically."""

    ordered_schemas = _ordered_feature_schemas(schemas)
    serialized_schemas = []
    for schema in ordered_schemas:
        serialized_schemas.append(_feature_schema_dict(schema))
    payload = {
        "hash_version": FEATURE_SCHEMA_HASH_VERSION,
        "features": serialized_schemas,
    }
    return hash_logical_content(payload)


def fit_feature_normalization(
    training_sequence_path: str,
    repository_root: str,
    split_manifest: PretrainingSplitManifest,
    provider: BiophysicalFeatureProvider,
    creation_entry_point: str,
) -> BiophysicalNormalizationArtifact:
    """Fit valid-only statistics from the fingerprinted training source."""

    split_manifest.validate_hashes()
    current_fingerprint = fingerprint_sequence_file(
        training_sequence_path,
        repository_root,
    )
    if (
        current_fingerprint.to_dict()
        != split_manifest.training_source.to_dict()
    ):
        raise NormalizationCompatibilityError(
            "Training source does not match the split-manifest fingerprint."
        )

    ordered_schemas = _ordered_feature_schemas(provider.schemas)
    running_statistics = {}
    for schema in ordered_schemas:
        running_statistics[schema.feature_order] = _RunningFeatureStatistics()

    with open(training_sequence_path, "r", encoding="utf-8") as sequence_file:
        for line_number, line in enumerate(sequence_file, start=1):
            stripped = line.strip()
            if not stripped:
                message = "Blank training sequence row at line {0}."
                raise ValueError(message.format(line_number))
            sequence = normalize_sequence(stripped)
            feature_batch = provider.compute(sequence)
            _validate_computed_batch_schema(
                feature_batch,
                ordered_schemas,
            )
            _update_running_statistics(
                feature_batch,
                running_statistics,
            )

    final_fingerprint = fingerprint_sequence_file(
        training_sequence_path,
        repository_root,
    )
    if final_fingerprint.to_dict() != current_fingerprint.to_dict():
        raise NormalizationCompatibilityError(
            "Training source changed while normalization was being fit."
        )

    finished_statistics = []
    for schema in ordered_schemas:
        accumulator = running_statistics[schema.feature_order]
        finished_statistics.append(
            _finish_feature_statistics(schema, accumulator)
        )

    artifact_content = _normalization_artifact_content(
        creation_entry_point=creation_entry_point,
        training_source_fingerprint=split_manifest.training_source,
        experiment_split_manifest_hash=split_manifest.manifest_hash,
        schema_hash=feature_schema_hash(ordered_schemas),
        provider=provider,
        ordered_schemas=ordered_schemas,
        statistics=finished_statistics,
    )
    artifact = BiophysicalNormalizationArtifact(
        artifact_schema_version=NORMALIZATION_ARTIFACT_SCHEMA_VERSION,
        creation_entry_point=creation_entry_point,
        training_source_fingerprint=split_manifest.training_source,
        experiment_split_manifest_hash=split_manifest.manifest_hash,
        feature_schema_hash=feature_schema_hash(ordered_schemas),
        feature_provider_identifier=provider.provider_identifier,
        feature_provider_source_version=provider.source_version,
        feature_provider_fingerprint=provider.provider_fingerprint,
        feature_names=tuple(schema.name for schema in ordered_schemas),
        feature_order=tuple(
            schema.feature_order for schema in ordered_schemas
        ),
        display_names=tuple(
            schema.display_name for schema in ordered_schemas
        ),
        coordinate_types=tuple(
            schema.coordinate_type.value for schema in ordered_schemas
        ),
        units=tuple(schema.unit for schema in ordered_schemas),
        statistics=tuple(finished_statistics),
        standard_deviation_convention=STANDARD_DEVIATION_CONVENTION,
        fit_value_handling=FIT_VALUE_HANDLING,
        transform_value_handling=TRANSFORM_VALUE_HANDLING,
        artifact_hash=hash_logical_content(artifact_content),
    )
    artifact.validate_internal_consistency()
    return artifact


def validate_normalization_compatibility(
    artifact: BiophysicalNormalizationArtifact,
    provider: BiophysicalFeatureProvider,
    split_manifest: PretrainingSplitManifest,
) -> None:
    """Fail closed on any provider, schema, order, or split mismatch."""

    artifact.validate_internal_consistency()
    split_manifest.validate_hashes()
    ordered_schemas = _ordered_feature_schemas(provider.schemas)
    expected_names = tuple(schema.name for schema in ordered_schemas)
    expected_order = tuple(schema.feature_order for schema in ordered_schemas)
    expected_coordinate_types = tuple(
        schema.coordinate_type.value for schema in ordered_schemas
    )
    expected_display_names = tuple(
        schema.display_name for schema in ordered_schemas
    )
    expected_units = tuple(schema.unit for schema in ordered_schemas)

    comparisons = (
        (
            artifact.feature_schema_hash,
            feature_schema_hash(ordered_schemas),
            "feature-schema hash",
        ),
        (
            artifact.feature_names,
            expected_names,
            "feature names",
        ),
        (
            artifact.feature_order,
            expected_order,
            "feature order",
        ),
        (
            artifact.coordinate_types,
            expected_coordinate_types,
            "coordinate types",
        ),
        (
            artifact.display_names,
            expected_display_names,
            "display names",
        ),
        (
            artifact.units,
            expected_units,
            "feature units",
        ),
        (
            artifact.feature_provider_identifier,
            provider.provider_identifier,
            "feature-provider identifier",
        ),
        (
            artifact.feature_provider_source_version,
            provider.source_version,
            "feature-provider source version",
        ),
        (
            artifact.feature_provider_fingerprint,
            provider.provider_fingerprint,
            "feature-provider fingerprint",
        ),
        (
            artifact.experiment_split_manifest_hash,
            split_manifest.manifest_hash,
            "experiment split-manifest hash",
        ),
        (
            artifact.training_source_fingerprint.to_dict(),
            split_manifest.training_source.to_dict(),
            "training source fingerprint",
        ),
    )
    for actual, expected, label in comparisons:
        if actual != expected:
            message = "Normalization artifact {0} mismatch."
            raise NormalizationCompatibilityError(message.format(label))


def save_normalization_artifact(
    artifact: BiophysicalNormalizationArtifact,
    path: str,
) -> None:
    """Save deterministic JSON without overwriting an existing artifact."""

    artifact.validate_internal_consistency()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        artifact.to_dict(),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    with open(output_path, "x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(serialized)
        output_file.write("\n")


def load_normalization_artifact(
    path: str,
) -> BiophysicalNormalizationArtifact:
    """Load an artifact and validate its internal schema and hash."""

    with open(path, "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    return BiophysicalNormalizationArtifact.from_dict(payload)


def load_validated_normalizer(
    path: str,
    provider: BiophysicalFeatureProvider,
    split_manifest: PretrainingSplitManifest,
) -> FeatureNormalizer:
    """Load an artifact and strictly validate the expected experiment."""

    artifact = load_normalization_artifact(path)
    validate_normalization_compatibility(
        artifact,
        provider,
        split_manifest,
    )
    return FeatureNormalizer(artifact)


def _ordered_feature_schemas(
    schemas: Sequence[FeatureSchema],
) -> Tuple[FeatureSchema, ...]:
    ordered = tuple(sorted(schemas, key=lambda schema: schema.feature_order))
    expected_order = tuple(range(len(ordered)))
    observed_order = tuple(schema.feature_order for schema in ordered)
    if observed_order != expected_order:
        raise ValueError(
            "Feature schema order must be unique and contiguous from zero."
        )
    names = tuple(schema.name for schema in ordered)
    if len(set(names)) != len(names):
        raise ValueError("Feature schema names must be unique.")
    return ordered


def _feature_schema_dict(schema: FeatureSchema) -> Dict[str, Any]:
    return {
        "name": schema.name,
        "display_name": schema.display_name,
        "source": schema.source,
        "source_version": schema.source_version,
        "citation": schema.citation,
        "unit": schema.unit,
        "feature_family": schema.feature_family,
        "coordinate_type": schema.coordinate_type.value,
        "sequence_granularity": schema.sequence_granularity.value,
        "required_context_length": schema.required_context_length,
        "alignment_rule": schema.alignment_rule.value,
        "reverse_complement_rule": schema.reverse_complement_rule.value,
        "ambiguous_base_rule": schema.ambiguous_base_rule.value,
        "missing_value_rule": schema.missing_value_rule.value,
        "normalization_rule": schema.normalization_rule.value,
        "feature_order": schema.feature_order,
        "schema_version": schema.schema_version,
    }


def _update_running_statistics(
    feature_batch: FeatureBatch,
    running_statistics: Mapping[int, _RunningFeatureStatistics],
) -> None:
    for track in feature_batch.tracks:
        for feature_index, schema in enumerate(track.schemas):
            values = track.values[:, feature_index]
            effective_mask = track.valid_mask[:, feature_index]
            effective_mask = effective_mask & torch.isfinite(values)
            valid_values = values[effective_mask]
            running_statistics[schema.feature_order].update(valid_values)


def _validate_computed_batch_schema(
    feature_batch: FeatureBatch,
    expected_schemas: Sequence[FeatureSchema],
) -> None:
    batch_schemas = []
    for track in feature_batch.tracks:
        for schema in track.schemas:
            batch_schemas.append(schema)
    if feature_schema_hash(batch_schemas) != feature_schema_hash(
        expected_schemas
    ):
        raise NormalizationCompatibilityError(
            "Computed feature batch schema does not match provider schemas."
        )


def _finish_feature_statistics(
    schema: FeatureSchema,
    accumulator: _RunningFeatureStatistics,
) -> FeatureNormalizationStatistics:
    if accumulator.count < 2:
        message = (
            "Feature '{0}' has {1} valid values; at least two are required."
        )
        raise NormalizationFitError(
            message.format(schema.name, accumulator.count)
        )
    variance = accumulator.sum_squared_deviations / accumulator.count
    standard_deviation = math.sqrt(variance)
    statistics = FeatureNormalizationStatistics(
        name=schema.name,
        display_name=schema.display_name,
        feature_order=schema.feature_order,
        coordinate_type=schema.coordinate_type.value,
        unit=schema.unit,
        valid_count=accumulator.count,
        mean=accumulator.mean,
        standard_deviation=standard_deviation,
    )
    _validate_finished_statistics(statistics)
    return statistics


def _validate_finished_statistics(
    statistics: FeatureNormalizationStatistics,
) -> None:
    if statistics.valid_count < 2:
        message = "Feature '{0}' has insufficient valid data."
        raise NormalizationFitError(message.format(statistics.name))
    if not math.isfinite(statistics.mean):
        message = "Feature '{0}' has a non-finite mean."
        raise NormalizationFitError(message.format(statistics.name))
    if not math.isfinite(statistics.standard_deviation):
        message = "Feature '{0}' has a non-finite standard deviation."
        raise NormalizationFitError(message.format(statistics.name))
    if statistics.standard_deviation <= 0.0:
        message = "Feature '{0}' has zero variance."
        raise NormalizationFitError(message.format(statistics.name))


def _normalization_artifact_content(
    creation_entry_point: str,
    training_source_fingerprint: SourceFingerprint,
    experiment_split_manifest_hash: str,
    schema_hash: str,
    provider: BiophysicalFeatureProvider,
    ordered_schemas: Sequence[FeatureSchema],
    statistics: Sequence[FeatureNormalizationStatistics],
) -> Dict[str, Any]:
    return {
        "artifact_schema_version": NORMALIZATION_ARTIFACT_SCHEMA_VERSION,
        "creation_entry_point": creation_entry_point,
        "training_source_fingerprint": training_source_fingerprint.to_dict(),
        "experiment_split_manifest_hash": experiment_split_manifest_hash,
        "feature_schema_hash": schema_hash,
        "feature_provider_identifier": provider.provider_identifier,
        "feature_provider_source_version": provider.source_version,
        "feature_provider_fingerprint": provider.provider_fingerprint,
        "feature_names": [schema.name for schema in ordered_schemas],
        "feature_order": [
            schema.feature_order for schema in ordered_schemas
        ],
        "display_names": [
            schema.display_name for schema in ordered_schemas
        ],
        "coordinate_types": [
            schema.coordinate_type.value for schema in ordered_schemas
        ],
        "units": [schema.unit for schema in ordered_schemas],
        "statistics": [
            feature_statistics.to_dict()
            for feature_statistics in statistics
        ],
        "standard_deviation_convention": STANDARD_DEVIATION_CONVENTION,
        "fit_value_handling": FIT_VALUE_HANDLING,
        "transform_value_handling": TRANSFORM_VALUE_HANDLING,
    }
