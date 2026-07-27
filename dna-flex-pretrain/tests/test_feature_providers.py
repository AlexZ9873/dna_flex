"""Tests for native-coordinate lookup-table physical features."""

import inspect
import os
from pathlib import Path
import unittest

import torch

from src.coordinates import CanonicalOrientation, SequenceCoordinates
from src.feature_providers import LookupTableFeatureProvider
from src.feature_schema import (
    AmbiguousBaseRule,
    BiophysicalFeatureProvider,
    DeepDNAshapeFeatureProvider,
    FeatureBatch,
    FeatureAlignmentRule,
    FeatureCoordinateType,
    FeatureGranularity,
    FeatureTrack,
    MissingValueRule,
    NormalizationRule,
    ProcessedHexABCFeatureProvider,
    ReverseComplementRule,
)
from src.flex_features import (
    load_lookup_yaml,
    sequence_to_multi_dinuc_targets,
    sequence_to_multi_trinuc_targets,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOOKUP_PATH = REPOSITORY_ROOT / "data" / "raw" / "flex_tables" / "lookup.yaml"


class LookupTableFeatureProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lookup_data = load_lookup_yaml(str(LOOKUP_PATH))
        cls.provider = LookupTableFeatureProvider.from_yaml(
            str(LOOKUP_PATH),
            repository_root=str(REPOSITORY_ROOT),
        )

    def test_current_lookup_tables_load_in_stable_legacy_order(self) -> None:
        expected_names = (
            "xDisp",
            "wedge",
            "prop",
            "freeen",
            "gc",
            "twistDisp",
            "stifness",
            "bendingstiffness",
            "NPP",
            "DNaseI",
            "bendabilityDNase",
            "bendabilityConcensus",
        )
        actual_names = []
        for schema in self.provider.schemas:
            actual_names.append(schema.name)

        self.assertEqual(tuple(actual_names), expected_names)

    def test_fallback_source_identity_is_independent_of_path_spelling(
        self,
    ) -> None:
        relative_lookup_path = os.path.relpath(
            LOOKUP_PATH,
            Path.cwd(),
        )
        relative_provider = LookupTableFeatureProvider.from_yaml(
            relative_lookup_path
        )
        absolute_provider = LookupTableFeatureProvider.from_yaml(
            str(LOOKUP_PATH)
        )

        self.assertEqual(
            relative_provider.schemas[0].source,
            absolute_provider.schemas[0].source,
        )
        self.assertEqual(
            relative_provider.provider_fingerprint,
            absolute_provider.provider_fingerprint,
        )

    def test_future_providers_share_an_unimplemented_abstract_interface(
        self,
    ) -> None:
        self.assertTrue(
            issubclass(
                DeepDNAshapeFeatureProvider,
                BiophysicalFeatureProvider,
            )
        )
        self.assertTrue(
            issubclass(
                ProcessedHexABCFeatureProvider,
                BiophysicalFeatureProvider,
            )
        )
        self.assertTrue(inspect.isabstract(DeepDNAshapeFeatureProvider))
        self.assertTrue(inspect.isabstract(ProcessedHexABCFeatureProvider))

    def test_feature_batch_supports_future_native_coordinate_tracks(
        self,
    ) -> None:
        kmer_track = FeatureTrack(
            coordinate_type=FeatureCoordinateType.KMER_CENTERED,
            coordinate_positions=torch.tensor([0, 1], dtype=torch.long),
            schemas=(),
            values=torch.empty((2, 0), dtype=torch.float32),
            valid_mask=torch.empty((2, 0), dtype=torch.bool),
            support_spans=torch.empty((2, 0, 2), dtype=torch.long),
        )
        batch = FeatureBatch(
            sequence="AACCGG",
            coordinates=SequenceCoordinates.from_length(6),
            tracks=(kmer_track,),
            canonical_orientation=CanonicalOrientation.FORWARD,
            is_reverse_complement_palindrome=False,
        )

        self.assertIs(
            batch.get_track(FeatureCoordinateType.KMER_CENTERED),
            kmer_track,
        )
        with self.assertRaises(KeyError):
            _ = batch.base_features

    def test_schema_records_required_scientific_metadata(self) -> None:
        schema = self.provider.dinucleotide_schemas[0]

        self.assertEqual(schema.name, "xDisp")
        self.assertEqual(schema.display_name, "xDisp")
        self.assertEqual(
            schema.source,
            "data/raw/flex_tables/lookup.yaml",
        )
        self.assertEqual(schema.source_version, "unversioned")
        self.assertEqual(
            schema.citation,
            "not_documented_in_repository_lookup_table",
        )
        self.assertEqual(schema.unit, "unspecified_in_source_table")
        self.assertEqual(schema.feature_family, "dinucleotide_lookup")
        self.assertEqual(
            schema.coordinate_type,
            FeatureCoordinateType.BASE_STEP_CENTERED,
        )
        self.assertEqual(
            schema.sequence_granularity,
            FeatureGranularity.DINUCLEOTIDE,
        )
        self.assertEqual(schema.required_context_length, 2)
        self.assertEqual(
            schema.alignment_rule,
            FeatureAlignmentRule.DINUCLEOTIDE_TO_LEFT_BASE_STEP,
        )
        self.assertEqual(
            schema.reverse_complement_rule,
            ReverseComplementRule.REVERSE_POSITIONS_VALUE_INVARIANT,
        )
        self.assertEqual(
            schema.ambiguous_base_rule,
            AmbiguousBaseRule.MASK_OVERLAPPING_CONTEXT,
        )
        self.assertEqual(
            schema.missing_value_rule,
            MissingValueRule.MASK_AS_INVALID,
        )
        self.assertEqual(
            schema.normalization_rule,
            NormalizationRule.TRAINING_SPLIT_ZSCORE,
        )
        self.assertEqual(schema.feature_order, 0)
        self.assertEqual(schema.schema_version, "1.0")

    def test_known_dinucleotide_values_align_to_base_steps(self) -> None:
        provider = LookupTableFeatureProvider(
            self.lookup_data,
            dinucleotide_feature_names=("xDisp",),
            trinucleotide_feature_names=(),
        )
        batch = provider.compute("ACG")
        track = batch.base_step_features

        self.assertEqual(tuple(track.values.shape), (2, 1))
        self.assertEqual(track.coordinate_positions.tolist(), [0, 1])
        torch.testing.assert_close(
            track.values[:, 0],
            torch.tensor([-0.719, 0.869], dtype=torch.float32),
        )
        self.assertEqual(track.valid_mask[:, 0].tolist(), [True, True])
        self.assertEqual(
            track.support_spans[:, 0, :].tolist(),
            [[0, 2], [1, 3]],
        )

    def test_known_trinucleotide_values_align_to_center_bases(self) -> None:
        provider = LookupTableFeatureProvider(
            self.lookup_data,
            dinucleotide_feature_names=(),
            trinucleotide_feature_names=("NPP",),
        )
        batch = provider.compute("AACG")
        track = batch.base_features

        self.assertEqual(tuple(track.values.shape), (4, 1))
        self.assertEqual(track.coordinate_positions.tolist(), [0, 1, 2, 3])
        self.assertEqual(
            track.valid_mask[:, 0].tolist(),
            [False, True, True, False],
        )
        self.assertTrue(torch.isnan(track.values[0, 0]))
        self.assertEqual(track.values[1, 0].item(), 6.0)
        self.assertEqual(track.values[2, 0].item(), 8.0)
        self.assertTrue(torch.isnan(track.values[3, 0]))
        self.assertEqual(
            track.support_spans[:, 0, :].tolist(),
            [[-1, -1], [0, 3], [1, 4], [-1, -1]],
        )

    def test_base_and_base_step_tracks_preserve_native_lengths(self) -> None:
        batch = self.provider.compute("ACGTAC")

        self.assertEqual(batch.base_features.values.shape[0], 6)
        self.assertEqual(batch.base_step_features.values.shape[0], 5)
        self.assertEqual(batch.coordinates.base_positions, tuple(range(6)))
        self.assertEqual(
            batch.coordinates.base_step_positions,
            tuple(range(5)),
        )

    def test_ambiguous_n_masks_only_overlapping_contexts(self) -> None:
        provider = LookupTableFeatureProvider(
            self.lookup_data,
            dinucleotide_feature_names=("xDisp",),
            trinucleotide_feature_names=("NPP",),
        )
        batch = provider.compute("ANCG")

        self.assertEqual(
            batch.base_step_features.valid_mask[:, 0].tolist(),
            [False, False, True],
        )
        self.assertEqual(
            batch.base_features.valid_mask[:, 0].tolist(),
            [False, False, False, False],
        )
        self.assertEqual(
            batch.base_step_features.support_spans[:, 0, :].tolist(),
            [[0, 2], [1, 3], [2, 4]],
        )

    def test_missing_lookup_entry_is_masked_without_imputation(self) -> None:
        lookup_data = {
            "dinucleotide": {
                "toy": {
                    "AA": 1.0,
                    "TT": 1.0,
                }
            },
            "trinucleotide": {},
        }
        provider = LookupTableFeatureProvider(lookup_data)
        batch = provider.compute("AAT")
        track = batch.base_step_features

        self.assertEqual(track.valid_mask[:, 0].tolist(), [True, False])
        self.assertEqual(track.values[0, 0].item(), 1.0)
        self.assertTrue(torch.isnan(track.values[1, 0]))
        self.assertEqual(
            track.support_spans[:, 0, :].tolist(),
            [[0, 2], [1, 3]],
        )

    def test_incomplete_reverse_complement_pair_masks_both_strands(self) -> None:
        lookup_data = {
            "dinucleotide": {
                "toy": {
                    "AA": 1.0,
                }
            },
            "trinucleotide": {},
        }
        provider = LookupTableFeatureProvider(lookup_data)
        forward = provider.compute("AA")
        reverse = provider.compute("TT")

        self.assertFalse(forward.base_step_features.valid_mask[0, 0])
        self.assertTrue(torch.isnan(forward.base_step_features.values[0, 0]))
        self.assertFalse(reverse.base_step_features.valid_mask[0, 0])
        self.assertTrue(torch.isnan(reverse.base_step_features.values[0, 0]))

    def test_reverse_complement_reverses_base_and_step_axes(self) -> None:
        provider = LookupTableFeatureProvider(
            self.lookup_data,
            dinucleotide_feature_names=("xDisp",),
            trinucleotide_feature_names=("NPP",),
        )
        forward = provider.compute("AACG")
        reverse = provider.compute_reverse_complement("AACG")

        self.assertEqual(reverse.sequence, "CGTT")
        self.assertTrue(
            torch.allclose(
                reverse.base_features.values,
                torch.flip(forward.base_features.values, dims=(0,)),
                equal_nan=True,
            )
        )
        self.assertTrue(
            torch.equal(
                reverse.base_features.valid_mask,
                torch.flip(forward.base_features.valid_mask, dims=(0,)),
            )
        )
        self.assertTrue(
            torch.allclose(
                reverse.base_step_features.values,
                torch.flip(forward.base_step_features.values, dims=(0,)),
                equal_nan=True,
            )
        )
        self.assertTrue(
            torch.equal(
                reverse.base_step_features.valid_mask,
                torch.flip(forward.base_step_features.valid_mask, dims=(0,)),
            )
        )

    def test_reverse_complement_palindrome_is_explicit_and_symmetric(self) -> None:
        provider = LookupTableFeatureProvider(
            self.lookup_data,
            dinucleotide_feature_names=("xDisp",),
            trinucleotide_feature_names=("NPP",),
        )
        batch = provider.compute("ATGCAT")

        self.assertEqual(
            batch.canonical_orientation,
            CanonicalOrientation.PALINDROME,
        )
        self.assertTrue(batch.is_reverse_complement_palindrome)
        self.assertTrue(
            torch.allclose(
                batch.base_features.values,
                torch.flip(batch.base_features.values, dims=(0,)),
                equal_nan=True,
            )
        )
        self.assertTrue(
            torch.allclose(
                batch.base_step_features.values,
                torch.flip(batch.base_step_features.values, dims=(0,)),
                equal_nan=True,
            )
        )

    def test_outputs_are_deterministic(self) -> None:
        first = self.provider.compute("ACNTACG")
        second = self.provider.compute("acntacg")

        self.assertTrue(
            torch.allclose(
                first.base_features.values,
                second.base_features.values,
                equal_nan=True,
            )
        )
        self.assertTrue(
            torch.equal(
                first.base_features.valid_mask,
                second.base_features.valid_mask,
            )
        )
        self.assertTrue(
            torch.allclose(
                first.base_step_features.values,
                second.base_step_features.values,
                equal_nan=True,
            )
        )
        self.assertTrue(
            torch.equal(
                first.base_step_features.valid_mask,
                second.base_step_features.valid_mask,
            )
        )

    def test_native_tracks_reproduce_legacy_six_mer_averages(self) -> None:
        sequence = "ACGTACG"
        dinucleotide_names = tuple(
            self.lookup_data["dinucleotide"].keys()
        )
        trinucleotide_names = tuple(
            self.lookup_data["trinucleotide"].keys()
        )
        batch = self.provider.compute(sequence)
        legacy_dinucleotide_tokens, legacy_dinucleotide_targets = (
            sequence_to_multi_dinuc_targets(
                sequence,
                6,
                self.lookup_data,
                list(dinucleotide_names),
            )
        )
        legacy_trinucleotide_tokens, legacy_trinucleotide_targets = (
            sequence_to_multi_trinuc_targets(
                sequence,
                6,
                self.lookup_data,
                list(trinucleotide_names),
            )
        )

        self.assertEqual(
            legacy_dinucleotide_tokens,
            legacy_trinucleotide_tokens,
        )
        for token_index in range(len(legacy_dinucleotide_tokens)):
            native_dinucleotide_average = batch.base_step_features.values[
                token_index : token_index + 5
            ].mean(dim=0)
            legacy_dinucleotide_average = torch.tensor(
                legacy_dinucleotide_targets[token_index],
                dtype=torch.float32,
            )
            torch.testing.assert_close(
                native_dinucleotide_average,
                legacy_dinucleotide_average,
            )

            native_trinucleotide_average = batch.base_features.values[
                token_index + 1 : token_index + 5
            ].mean(dim=0)
            legacy_trinucleotide_average = torch.tensor(
                legacy_trinucleotide_targets[token_index],
                dtype=torch.float32,
            )
            torch.testing.assert_close(
                native_trinucleotide_average,
                legacy_trinucleotide_average,
            )


if __name__ == "__main__":
    unittest.main()
