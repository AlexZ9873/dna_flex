"""Canonical sequence coordinates and tokenizer-independent token metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


SUPPORTED_TOKEN_SIZES = (1, 3, 6)
VALID_BASES = frozenset(("A", "C", "G", "T"))
VALID_SEQUENCE_SYMBOLS = frozenset(("A", "C", "G", "T", "N"))
_COMPLEMENT_TRANSLATION = str.maketrans(
    {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "N": "N",
    }
)


class TokenCenterType(str, Enum):
    """The canonical coordinate system used for a token center."""

    BASE = "base"
    BASE_STEP = "base_step"


class CanonicalOrientation(str, Enum):
    """A sequence's orientation relative to its canonical strand."""

    FORWARD = "forward"
    REVERSE_COMPLEMENT = "reverse_complement"
    PALINDROME = "reverse_complement_palindrome"


@dataclass(frozen=True)
class BaseSpan:
    """A half-open span over original base coordinates."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("Base-span start must be non-negative.")
        if self.end < self.start:
            raise ValueError("Base-span end must not precede its start.")

    @property
    def length(self) -> int:
        """Return the number of bases in the span."""

        return self.end - self.start

    def reverse_complement(self, sequence_length: int) -> "BaseSpan":
        """Map this span to coordinates in the reverse-complement sequence."""

        if self.end > sequence_length:
            raise ValueError("Base span extends beyond the sequence length.")
        return BaseSpan(
            start=sequence_length - self.end,
            end=sequence_length - self.start,
        )


@dataclass(frozen=True)
class SequenceCoordinates:
    """Canonical base and base-step coordinate axes for one sequence."""

    sequence_length: int
    base_positions: Tuple[int, ...]
    base_step_positions: Tuple[int, ...]

    @classmethod
    def from_length(cls, sequence_length: int) -> "SequenceCoordinates":
        """Build base positions 0..L-1 and base steps 0..L-2."""

        if sequence_length < 0:
            raise ValueError("Sequence length must be non-negative.")
        base_positions = tuple(range(sequence_length))
        base_step_count = max(sequence_length - 1, 0)
        base_step_positions = tuple(range(base_step_count))
        return cls(
            sequence_length=sequence_length,
            base_positions=base_positions,
            base_step_positions=base_step_positions,
        )


@dataclass(frozen=True)
class TokenRecord:
    """One overlapping token with complete coordinate and strand metadata."""

    sequence: str
    span: BaseSpan
    center_type: TokenCenterType
    center_index: int
    base_valid_mask: Tuple[bool, ...]
    valid: bool
    reverse_complement_sequence: str
    reverse_complement_span: BaseSpan
    reverse_complement_center_index: int
    canonical_sequence: str
    canonical_orientation: CanonicalOrientation
    is_reverse_complement_palindrome: bool


@dataclass(frozen=True)
class TokenizedSequence:
    """A sequence and its stride-one tokens on canonical coordinates."""

    sequence: str
    token_size: int
    coordinates: SequenceCoordinates
    tokens: Tuple[TokenRecord, ...]


def normalize_sequence(sequence: str) -> str:
    """Uppercase a DNA sequence and reject undocumented alphabet symbols."""

    normalized = sequence.upper()
    for symbol in normalized:
        if symbol not in VALID_SEQUENCE_SYMBOLS:
            message = "Unsupported DNA symbol '{0}'; expected A, C, G, T, or N."
            raise ValueError(message.format(symbol))
    return normalized


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement while preserving ambiguous N bases."""

    normalized = normalize_sequence(sequence)
    return normalized.translate(_COMPLEMENT_TRANSLATION)[::-1]


def canonical_orientation(sequence: str) -> CanonicalOrientation:
    """Return the sequence orientation relative to lexicographic canonical DNA."""

    normalized = normalize_sequence(sequence)
    reverse = reverse_complement(normalized)
    if normalized == reverse:
        return CanonicalOrientation.PALINDROME
    if normalized < reverse:
        return CanonicalOrientation.FORWARD
    return CanonicalOrientation.REVERSE_COMPLEMENT


def _center_type_for_token_size(token_size: int) -> TokenCenterType:
    if token_size % 2 == 1:
        return TokenCenterType.BASE
    return TokenCenterType.BASE_STEP


def _center_index(span: BaseSpan, center_type: TokenCenterType) -> int:
    if center_type == TokenCenterType.BASE:
        return span.start + (span.length // 2)
    return span.start + (span.length // 2) - 1


def _canonical_token_metadata(
    token_sequence: str,
) -> Tuple[str, CanonicalOrientation, bool]:
    reverse = reverse_complement(token_sequence)
    if token_sequence == reverse:
        return token_sequence, CanonicalOrientation.PALINDROME, True
    if token_sequence < reverse:
        return token_sequence, CanonicalOrientation.FORWARD, False
    return reverse, CanonicalOrientation.REVERSE_COMPLEMENT, False


def tokenize_with_coordinates(sequence: str, token_size: int) -> TokenizedSequence:
    """Create complete stride-one tokens for supported tokenizer sizes."""

    if token_size not in SUPPORTED_TOKEN_SIZES:
        message = "Token size must be one of {0}; received {1}."
        raise ValueError(message.format(SUPPORTED_TOKEN_SIZES, token_size))

    normalized = normalize_sequence(sequence)
    coordinates = SequenceCoordinates.from_length(len(normalized))
    token_count = max(len(normalized) - token_size + 1, 0)
    center_type = _center_type_for_token_size(token_size)
    records = []

    for start in range(token_count):
        end = start + token_size
        span = BaseSpan(start=start, end=end)
        token_sequence = normalized[start:end]
        base_valid_mask = tuple(base in VALID_BASES for base in token_sequence)
        valid = all(base_valid_mask)

        reverse_sequence = reverse_complement(token_sequence)
        reverse_span = span.reverse_complement(len(normalized))
        reverse_center = _center_index(reverse_span, center_type)
        canonical_sequence, orientation, is_palindrome = _canonical_token_metadata(
            token_sequence
        )

        records.append(
            TokenRecord(
                sequence=token_sequence,
                span=span,
                center_type=center_type,
                center_index=_center_index(span, center_type),
                base_valid_mask=base_valid_mask,
                valid=valid,
                reverse_complement_sequence=reverse_sequence,
                reverse_complement_span=reverse_span,
                reverse_complement_center_index=reverse_center,
                canonical_sequence=canonical_sequence,
                canonical_orientation=orientation,
                is_reverse_complement_palindrome=is_palindrome,
            )
        )

    return TokenizedSequence(
        sequence=normalized,
        token_size=token_size,
        coordinates=coordinates,
        tokens=tuple(records),
    )
