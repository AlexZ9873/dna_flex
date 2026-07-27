"""Coordinate-aligned implementations of biophysical feature providers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from src.coordinates import (
    VALID_BASES,
    SequenceCoordinates,
    canonical_orientation,
    normalize_sequence,
    reverse_complement,
)
from src.data_fingerprints import (
    hash_logical_content,
    repository_relative_source_path,
)
from src.feature_schema import (
    AmbiguousBaseRule,
    BiophysicalFeatureProvider,
    FeatureBatch,
    FeatureAlignmentRule,
    FeatureCoordinateType,
    FeatureGranularity,
    FeatureSchema,
    FeatureTrack,
    MissingValueRule,
    NormalizationRule,
    ReverseComplementRule,
)
from src.flex_features import load_lookup_yaml


LookupTables = Dict[str, Dict[str, float]]
LOOKUP_TABLE_PROVIDER_IDENTIFIER = "lookup_table_native_coordinates.v1"


class LookupTableFeatureProvider(BiophysicalFeatureProvider):
    """Expose existing dinucleotide and trinucleotide tables on native axes."""

    def __init__(
        self,
        lookup_data: Mapping[str, Mapping[str, Mapping[str, float]]],
        dinucleotide_feature_names: Optional[Sequence[str]] = None,
        trinucleotide_feature_names: Optional[Sequence[str]] = None,
        source: str = "current_lookup_table",
        source_version: str = "unversioned",
        citation: str = "not_documented_in_repository_lookup_table",
        schema_version: str = "1.0",
    ) -> None:
        self._source_version = source_version
        dinucleotide_tables = lookup_data.get("dinucleotide", {})
        trinucleotide_tables = lookup_data.get("trinucleotide", {})

        self._dinucleotide_names = self._resolve_feature_names(
            dinucleotide_tables,
            dinucleotide_feature_names,
            "dinucleotide",
        )
        self._trinucleotide_names = self._resolve_feature_names(
            trinucleotide_tables,
            trinucleotide_feature_names,
            "trinucleotide",
        )
        self._dinucleotide_tables = self._prepare_tables(
            dinucleotide_tables,
            self._dinucleotide_names,
            2,
        )
        self._trinucleotide_tables = self._prepare_tables(
            trinucleotide_tables,
            self._trinucleotide_names,
            3,
        )

        self._dinucleotide_schemas = self._build_schemas(
            self._dinucleotide_names,
            source,
            source_version,
            citation,
            "dinucleotide_lookup",
            FeatureCoordinateType.BASE_STEP_CENTERED,
            FeatureGranularity.DINUCLEOTIDE,
            2,
            FeatureAlignmentRule.DINUCLEOTIDE_TO_LEFT_BASE_STEP,
            0,
            schema_version,
        )
        self._trinucleotide_schemas = self._build_schemas(
            self._trinucleotide_names,
            source,
            source_version,
            citation,
            "trinucleotide_lookup",
            FeatureCoordinateType.BASE_CENTERED,
            FeatureGranularity.TRINUCLEOTIDE,
            3,
            FeatureAlignmentRule.CENTERED_TRINUCLEOTIDE_TO_MIDDLE_BASE,
            len(self._dinucleotide_names),
            schema_version,
        )
        self._schemas = (
            self._dinucleotide_schemas + self._trinucleotide_schemas
        )
        provider_content = {
            "fingerprint_version": "lookup_table_provider_fingerprint.v1",
            "provider_identifier": LOOKUP_TABLE_PROVIDER_IDENTIFIER,
            "source": source,
            "source_version": source_version,
            "dinucleotide_feature_names": list(self._dinucleotide_names),
            "trinucleotide_feature_names": list(self._trinucleotide_names),
            "dinucleotide_tables": self._dinucleotide_tables,
            "trinucleotide_tables": self._trinucleotide_tables,
        }
        self._provider_fingerprint = hash_logical_content(provider_content)

    @classmethod
    def from_yaml(
        cls,
        lookup_path: str,
        dinucleotide_feature_names: Optional[Sequence[str]] = None,
        trinucleotide_feature_names: Optional[Sequence[str]] = None,
        source_version: str = "unversioned",
        citation: str = "not_documented_in_repository_lookup_table",
        schema_version: str = "1.0",
        repository_root: Optional[str] = None,
    ) -> "LookupTableFeatureProvider":
        """Load the existing YAML format without changing legacy consumers."""

        lookup_data = load_lookup_yaml(lookup_path)
        if repository_root is None:
            source = Path(lookup_path).name
        else:
            source = repository_relative_source_path(
                lookup_path,
                repository_root,
            )
        return cls(
            lookup_data=lookup_data,
            dinucleotide_feature_names=dinucleotide_feature_names,
            trinucleotide_feature_names=trinucleotide_feature_names,
            source=source,
            source_version=source_version,
            citation=citation,
            schema_version=schema_version,
        )

    @property
    def provider_identifier(self) -> str:
        """Return the stable provider implementation identifier."""

        return LOOKUP_TABLE_PROVIDER_IDENTIFIER

    @property
    def source_version(self) -> str:
        """Return the lookup-table source version."""

        return self._source_version

    @property
    def provider_fingerprint(self) -> str:
        """Return a fingerprint of effective lookup values and configuration."""

        return self._provider_fingerprint

    @property
    def schemas(self) -> Tuple[FeatureSchema, ...]:
        """Return dinucleotide schemas followed by trinucleotide schemas."""

        return self._schemas

    @property
    def dinucleotide_schemas(self) -> Tuple[FeatureSchema, ...]:
        """Return schemas stored on the base-step coordinate axis."""

        return self._dinucleotide_schemas

    @property
    def trinucleotide_schemas(self) -> Tuple[FeatureSchema, ...]:
        """Return schemas stored on the base coordinate axis."""

        return self._trinucleotide_schemas

    def compute(self, sequence: str) -> FeatureBatch:
        """Compute feature tensors without averaging within tokenizer tokens."""

        normalized = normalize_sequence(sequence)
        coordinates = SequenceCoordinates.from_length(len(normalized))
        base_features = self._compute_trinucleotide_track(
            normalized,
            coordinates,
        )
        base_step_features = self._compute_dinucleotide_track(
            normalized,
            coordinates,
        )
        orientation = canonical_orientation(normalized)
        return FeatureBatch(
            sequence=normalized,
            coordinates=coordinates,
            tracks=(base_features, base_step_features),
            canonical_orientation=orientation,
            is_reverse_complement_palindrome=(
                normalized == reverse_complement(normalized)
            ),
        )

    def compute_reverse_complement(self, sequence: str) -> FeatureBatch:
        """Compute features after an explicit reverse-complement transform."""

        return self.compute(reverse_complement(sequence))

    @staticmethod
    def _resolve_feature_names(
        tables: Mapping[str, Mapping[str, float]],
        requested_names: Optional[Sequence[str]],
        table_family: str,
    ) -> Tuple[str, ...]:
        if requested_names is None:
            names = tuple(tables.keys())
        else:
            names = tuple(requested_names)

        seen_names = set()
        for name in names:
            if name in seen_names:
                message = "Duplicate {0} feature name: {1}"
                raise ValueError(message.format(table_family, name))
            if name not in tables:
                message = "Unknown {0} feature name: {1}"
                raise KeyError(message.format(table_family, name))
            seen_names.add(name)
        return names

    @classmethod
    def _prepare_tables(
        cls,
        tables: Mapping[str, Mapping[str, float]],
        names: Sequence[str],
        context_length: int,
    ) -> LookupTables:
        prepared = {}
        for name in names:
            table = {}
            for word, value in tables[name].items():
                normalized_word = word.upper()
                cls._validate_lookup_word(normalized_word, context_length)
                table[normalized_word] = float(value)
            cls._validate_reverse_complement_values(name, table)
            prepared[name] = cls._retain_complete_reverse_complement_pairs(
                table
            )
        return prepared

    @staticmethod
    def _validate_lookup_word(word: str, context_length: int) -> None:
        if len(word) != context_length:
            message = "Lookup word '{0}' has length {1}, expected {2}."
            raise ValueError(message.format(word, len(word), context_length))
        for base in word:
            if base not in VALID_BASES:
                message = "Lookup word '{0}' contains a non-ACGT base."
                raise ValueError(message.format(word))

    @staticmethod
    def _validate_reverse_complement_values(
        feature_name: str,
        table: Mapping[str, float],
    ) -> None:
        for word, value in table.items():
            reverse_word = reverse_complement(word)
            if reverse_word in table:
                reverse_value = table[reverse_word]
                if abs(value - reverse_value) > 1e-8:
                    message = (
                        "Feature '{0}' is not reverse-complement invariant for "
                        "'{1}' and '{2}'."
                    )
                    raise ValueError(
                        message.format(feature_name, word, reverse_word)
                    )

    @staticmethod
    def _retain_complete_reverse_complement_pairs(
        table: Mapping[str, float],
    ) -> Dict[str, float]:
        """Exclude incomplete RC pairs so both orientations remain masked."""

        complete_table = {}
        for word, value in table.items():
            reverse_word = reverse_complement(word)
            if reverse_word in table:
                complete_table[word] = value
        return complete_table

    @staticmethod
    def _build_schemas(
        names: Sequence[str],
        source: str,
        source_version: str,
        citation: str,
        feature_family: str,
        coordinate_type: FeatureCoordinateType,
        sequence_granularity: FeatureGranularity,
        context_length: int,
        alignment_rule: FeatureAlignmentRule,
        feature_order_offset: int,
        schema_version: str,
    ) -> Tuple[FeatureSchema, ...]:
        schemas = []
        for local_feature_order, name in enumerate(names):
            schemas.append(
                FeatureSchema(
                    name=name,
                    display_name=name,
                    source=source,
                    source_version=source_version,
                    citation=citation,
                    unit="unspecified_in_source_table",
                    feature_family=feature_family,
                    coordinate_type=coordinate_type,
                    sequence_granularity=sequence_granularity,
                    required_context_length=context_length,
                    alignment_rule=alignment_rule,
                    reverse_complement_rule=(
                        ReverseComplementRule.REVERSE_POSITIONS_VALUE_INVARIANT
                    ),
                    ambiguous_base_rule=(
                        AmbiguousBaseRule.MASK_OVERLAPPING_CONTEXT
                    ),
                    missing_value_rule=MissingValueRule.MASK_AS_INVALID,
                    normalization_rule=(
                        NormalizationRule.TRAINING_SPLIT_ZSCORE
                    ),
                    feature_order=feature_order_offset + local_feature_order,
                    schema_version=schema_version,
                )
            )
        return tuple(schemas)

    def _compute_dinucleotide_track(
        self,
        sequence: str,
        coordinates: SequenceCoordinates,
    ) -> FeatureTrack:
        coordinate_count = len(coordinates.base_step_positions)
        feature_count = len(self._dinucleotide_schemas)
        values = torch.full(
            (coordinate_count, feature_count),
            float("nan"),
            dtype=torch.float32,
        )
        valid_mask = torch.zeros(
            (coordinate_count, feature_count),
            dtype=torch.bool,
        )
        support_spans = torch.full(
            (coordinate_count, feature_count, 2),
            -1,
            dtype=torch.long,
        )

        for step_index in coordinates.base_step_positions:
            word = sequence[step_index : step_index + 2]
            word_is_valid = self._word_is_valid(word)
            for feature_index, feature_name in enumerate(
                self._dinucleotide_names
            ):
                support_spans[step_index, feature_index, 0] = step_index
                support_spans[step_index, feature_index, 1] = step_index + 2
                if word_is_valid:
                    table = self._dinucleotide_tables[feature_name]
                    if word in table:
                        values[step_index, feature_index] = table[word]
                        valid_mask[step_index, feature_index] = True

        coordinate_positions = torch.arange(
            coordinate_count,
            dtype=torch.long,
        )
        return FeatureTrack(
            coordinate_type=FeatureCoordinateType.BASE_STEP_CENTERED,
            coordinate_positions=coordinate_positions,
            schemas=self._dinucleotide_schemas,
            values=values,
            valid_mask=valid_mask,
            support_spans=support_spans,
        )

    def _compute_trinucleotide_track(
        self,
        sequence: str,
        coordinates: SequenceCoordinates,
    ) -> FeatureTrack:
        coordinate_count = len(coordinates.base_positions)
        feature_count = len(self._trinucleotide_schemas)
        values = torch.full(
            (coordinate_count, feature_count),
            float("nan"),
            dtype=torch.float32,
        )
        valid_mask = torch.zeros(
            (coordinate_count, feature_count),
            dtype=torch.bool,
        )
        support_spans = torch.full(
            (coordinate_count, feature_count, 2),
            -1,
            dtype=torch.long,
        )

        for base_index in coordinates.base_positions:
            context_start = base_index - 1
            context_end = base_index + 2
            has_complete_context = (
                context_start >= 0 and context_end <= len(sequence)
            )
            if has_complete_context:
                word = sequence[context_start:context_end]
                word_is_valid = self._word_is_valid(word)
                for feature_index, feature_name in enumerate(
                    self._trinucleotide_names
                ):
                    support_spans[base_index, feature_index, 0] = context_start
                    support_spans[base_index, feature_index, 1] = context_end
                    if word_is_valid:
                        table = self._trinucleotide_tables[feature_name]
                        if word in table:
                            values[base_index, feature_index] = table[word]
                            valid_mask[base_index, feature_index] = True

        coordinate_positions = torch.arange(
            coordinate_count,
            dtype=torch.long,
        )
        return FeatureTrack(
            coordinate_type=FeatureCoordinateType.BASE_CENTERED,
            coordinate_positions=coordinate_positions,
            schemas=self._trinucleotide_schemas,
            values=values,
            valid_mask=valid_mask,
            support_spans=support_spans,
        )

    @staticmethod
    def _word_is_valid(word: str) -> bool:
        for base in word:
            if base not in VALID_BASES:
                return False
        return True
