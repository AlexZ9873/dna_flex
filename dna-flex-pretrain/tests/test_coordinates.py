"""Tests for canonical coordinates and stride-one token metadata."""

import unittest

from src.coordinates import (
    BaseSpan,
    CanonicalOrientation,
    SequenceCoordinates,
    TokenCenterType,
    reverse_complement,
    tokenize_with_coordinates,
)
from src.tokenization import tokenize_kmers, tokenize_kmers_with_coordinates


class SequenceCoordinateTests(unittest.TestCase):
    def test_canonical_base_and_base_step_positions(self) -> None:
        coordinates = SequenceCoordinates.from_length(5)

        self.assertEqual(coordinates.base_positions, (0, 1, 2, 3, 4))
        self.assertEqual(coordinates.base_step_positions, (0, 1, 2, 3))

    def test_token_counts_spans_and_centers_for_supported_sizes(self) -> None:
        sequence = "ACGTACGT"

        one_mer = tokenize_with_coordinates(sequence, 1)
        self.assertEqual(len(one_mer.tokens), 8)
        self.assertEqual(one_mer.tokens[3].span, BaseSpan(3, 4))
        self.assertEqual(one_mer.tokens[3].center_type, TokenCenterType.BASE)
        self.assertEqual(one_mer.tokens[3].center_index, 3)

        three_mer = tokenize_with_coordinates(sequence, 3)
        self.assertEqual(len(three_mer.tokens), 6)
        self.assertEqual(three_mer.tokens[2].sequence, "GTA")
        self.assertEqual(three_mer.tokens[2].span, BaseSpan(2, 5))
        self.assertEqual(
            three_mer.tokens[2].center_type,
            TokenCenterType.BASE,
        )
        self.assertEqual(three_mer.tokens[2].center_index, 3)

        six_mer = tokenize_with_coordinates(sequence, 6)
        self.assertEqual(len(six_mer.tokens), 3)
        self.assertEqual(six_mer.tokens[1].sequence, "CGTACG")
        self.assertEqual(six_mer.tokens[1].span, BaseSpan(1, 7))
        self.assertEqual(
            six_mer.tokens[1].center_type,
            TokenCenterType.BASE_STEP,
        )
        self.assertEqual(six_mer.tokens[1].center_index, 3)

    def test_coordinate_tokenizer_preserves_legacy_token_strings(self) -> None:
        sequence = "acgtacgt"
        legacy_tokens = tokenize_kmers(sequence, 6)
        tokenized = tokenize_kmers_with_coordinates(sequence, 6)
        coordinate_tokens = []

        for token in tokenized.tokens:
            coordinate_tokens.append(token.sequence)

        self.assertEqual(coordinate_tokens, legacy_tokens)

    def test_reverse_complement_span_and_center_mapping(self) -> None:
        tokenized = tokenize_with_coordinates("AACCGGTT", 3)
        token = tokenized.tokens[1]

        self.assertEqual(token.span, BaseSpan(1, 4))
        self.assertEqual(token.reverse_complement_span, BaseSpan(4, 7))
        self.assertEqual(token.center_index, 2)
        self.assertEqual(token.reverse_complement_center_index, 5)
        self.assertEqual(
            token.reverse_complement_sequence,
            reverse_complement(token.sequence),
        )

    def test_reverse_complement_orientation_and_palindrome_metadata(self) -> None:
        forward = tokenize_with_coordinates("AAAAAA", 6).tokens[0]
        reverse = tokenize_with_coordinates("TTTTTT", 6).tokens[0]
        palindrome = tokenize_with_coordinates("ATGCAT", 6).tokens[0]

        self.assertEqual(
            forward.canonical_orientation,
            CanonicalOrientation.FORWARD,
        )
        self.assertEqual(
            reverse.canonical_orientation,
            CanonicalOrientation.REVERSE_COMPLEMENT,
        )
        self.assertEqual(forward.canonical_sequence, "AAAAAA")
        self.assertEqual(reverse.canonical_sequence, "AAAAAA")
        self.assertEqual(
            palindrome.canonical_orientation,
            CanonicalOrientation.PALINDROME,
        )
        self.assertTrue(palindrome.is_reverse_complement_palindrome)

    def test_ambiguous_n_sets_per_base_and_token_validity_masks(self) -> None:
        tokenized = tokenize_with_coordinates("ACNTA", 3)

        self.assertEqual(
            tokenized.tokens[0].base_valid_mask,
            (True, True, False),
        )
        self.assertFalse(tokenized.tokens[0].valid)
        self.assertEqual(
            tokenized.tokens[1].base_valid_mask,
            (True, False, True),
        )
        self.assertFalse(tokenized.tokens[1].valid)
        self.assertEqual(
            tokenized.tokens[2].base_valid_mask,
            (False, True, True),
        )
        self.assertFalse(tokenized.tokens[2].valid)

    def test_short_sequence_has_coordinates_but_no_incomplete_tokens(self) -> None:
        tokenized = tokenize_with_coordinates("AC", 3)

        self.assertEqual(tokenized.coordinates.base_positions, (0, 1))
        self.assertEqual(tokenized.coordinates.base_step_positions, (0,))
        self.assertEqual(tokenized.tokens, ())

    def test_tokenization_is_deterministic(self) -> None:
        first = tokenize_with_coordinates("ACNTACGT", 3)
        second = tokenize_with_coordinates("acntacgt", 3)

        self.assertEqual(first, second)

    def test_unsupported_token_size_and_alphabet_raise(self) -> None:
        with self.assertRaises(ValueError):
            tokenize_with_coordinates("ACGT", 2)
        with self.assertRaises(ValueError):
            tokenize_with_coordinates("ACGR", 1)


if __name__ == "__main__":
    unittest.main()
