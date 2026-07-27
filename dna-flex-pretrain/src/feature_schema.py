"""Schemas and provider contracts for coordinate-aligned physical features."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch

from src.coordinates import CanonicalOrientation, SequenceCoordinates


class FeatureCoordinateType(str, Enum):
    """Canonical coordinate axis used by a physical feature."""

    BASE_CENTERED = "base_centered"
    BASE_STEP_CENTERED = "base_step_centered"
    KMER_CENTERED = "kmer_centered"
    SEQUENCE_LEVEL = "sequence_level"


class FeatureGranularity(str, Enum):
    """Sequence context named by the feature source."""

    NUCLEOTIDE = "nucleotide"
    DINUCLEOTIDE = "dinucleotide"
    TRINUCLEOTIDE = "trinucleotide"
    TETRAMER = "tetramer"
    HEXAMER = "hexamer"
    SEQUENCE = "sequence"


class FeatureAlignmentRule(str, Enum):
    """Rule anchoring a context window to canonical coordinates."""

    DINUCLEOTIDE_TO_LEFT_BASE_STEP = "dinucleotide_to_left_base_step"
    CENTERED_TRINUCLEOTIDE_TO_MIDDLE_BASE = (
        "centered_trinucleotide_to_middle_base"
    )
    PROVIDER_DEFINED = "provider_defined"


class ReverseComplementRule(str, Enum):
    """How values transform when the sequence is reverse complemented."""

    REVERSE_POSITIONS_VALUE_INVARIANT = "reverse_positions_value_invariant"
    REVERSE_POSITIONS_SIGN_FLIP = "reverse_positions_sign_flip"
    ORIENTATION_SPECIFIC_LOOKUP = "orientation_specific_lookup"
    SEQUENCE_INVARIANT = "sequence_invariant"


class MissingValueRule(str, Enum):
    """How unavailable feature values are represented."""

    MASK_AS_INVALID = "mask_as_invalid"
    ERROR = "error"


class AmbiguousBaseRule(str, Enum):
    """How ambiguous bases affect a feature context."""

    MASK_OVERLAPPING_CONTEXT = "mask_overlapping_context"
    ERROR = "error"


class NormalizationRule(str, Enum):
    """How a feature is normalized before model supervision."""

    TRAINING_SPLIT_ZSCORE = "training_split_zscore"
    NONE = "none"


@dataclass(frozen=True)
class FeatureSchema:
    """Stable metadata required to interpret one physical feature."""

    name: str
    display_name: str
    source: str
    source_version: str
    citation: str
    unit: str
    feature_family: str
    coordinate_type: FeatureCoordinateType
    sequence_granularity: FeatureGranularity
    required_context_length: int
    alignment_rule: FeatureAlignmentRule
    reverse_complement_rule: ReverseComplementRule
    ambiguous_base_rule: AmbiguousBaseRule
    missing_value_rule: MissingValueRule
    normalization_rule: NormalizationRule
    feature_order: int
    schema_version: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature name must not be empty.")
        if self.required_context_length < 1:
            raise ValueError("Required context length must be positive.")
        if self.feature_order < 0:
            raise ValueError("Feature order must be non-negative.")
        if not self.schema_version:
            raise ValueError("Schema version must not be empty.")


@dataclass(frozen=True)
class FeatureTrack:
    """Values for one coordinate axis, retaining masks and support spans."""

    coordinate_type: FeatureCoordinateType
    coordinate_positions: torch.Tensor
    schemas: Tuple[FeatureSchema, ...]
    values: torch.Tensor
    valid_mask: torch.Tensor
    support_spans: torch.Tensor

    def __post_init__(self) -> None:
        coordinate_count = self.coordinate_positions.shape[0]
        feature_count = len(self.schemas)
        expected_matrix_shape = (coordinate_count, feature_count)
        expected_span_shape = (coordinate_count, feature_count, 2)

        if tuple(self.values.shape) != expected_matrix_shape:
            raise ValueError("Feature values do not match coordinates and schemas.")
        if tuple(self.valid_mask.shape) != expected_matrix_shape:
            raise ValueError("Feature validity mask has the wrong shape.")
        if tuple(self.support_spans.shape) != expected_span_shape:
            raise ValueError("Feature support spans have the wrong shape.")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("Feature validity mask must have boolean dtype.")
        for schema in self.schemas:
            if schema.coordinate_type != self.coordinate_type:
                raise ValueError(
                    "Feature schema coordinate type does not match its track."
                )


@dataclass(frozen=True)
class FeatureBatch:
    """All native-coordinate physical features for one DNA sequence."""

    sequence: str
    coordinates: SequenceCoordinates
    tracks: Tuple[FeatureTrack, ...]
    canonical_orientation: CanonicalOrientation
    is_reverse_complement_palindrome: bool

    def __post_init__(self) -> None:
        coordinate_types = set()
        for track in self.tracks:
            if track.coordinate_type in coordinate_types:
                message = "Duplicate feature track for coordinate type: {0}"
                raise ValueError(message.format(track.coordinate_type.value))
            coordinate_types.add(track.coordinate_type)

    def get_track(
        self,
        coordinate_type: FeatureCoordinateType,
    ) -> FeatureTrack:
        """Return the track stored on one native coordinate type."""

        for track in self.tracks:
            if track.coordinate_type == coordinate_type:
                return track
        message = "No feature track for coordinate type: {0}"
        raise KeyError(message.format(coordinate_type.value))

    @property
    def base_features(self) -> FeatureTrack:
        """Return the base-centered track when it is available."""

        return self.get_track(FeatureCoordinateType.BASE_CENTERED)

    @property
    def base_step_features(self) -> FeatureTrack:
        """Return the base-step-centered track when it is available."""

        return self.get_track(FeatureCoordinateType.BASE_STEP_CENTERED)


class BiophysicalFeatureProvider(ABC):
    """Interface for sequence-derived physical-feature providers."""

    @property
    @abstractmethod
    def provider_identifier(self) -> str:
        """Return a stable identifier for the provider implementation."""

    @property
    @abstractmethod
    def source_version(self) -> str:
        """Return the physical-feature source version."""

    @property
    @abstractmethod
    def provider_fingerprint(self) -> str:
        """Return a deterministic fingerprint of effective provider content."""

    @property
    @abstractmethod
    def schemas(self) -> Tuple[FeatureSchema, ...]:
        """Return all feature schemas in stable output order."""

    @abstractmethod
    def compute(self, sequence: str) -> FeatureBatch:
        """Compute native-coordinate features without tokenizer projection."""


class DeepDNAshapeFeatureProvider(BiophysicalFeatureProvider, ABC):
    """Future provider contract; no DeepDNAshape integration is implemented."""


class ProcessedHexABCFeatureProvider(BiophysicalFeatureProvider, ABC):
    """Future provider contract for processed offline hexABC features."""
