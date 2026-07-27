"""Tests for fingerprint-bound training-only feature normalization."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.data_prep.compute_flex_norm_stats import (
    LEGACY_OUTPUT_PATH,
    validated_output_path,
)
from src.data_fingerprints import (
    build_pretraining_split_manifest,
    hash_logical_content,
    save_split_manifest,
)
from src.feature_normalization import (
    FeatureNormalizer,
    NormalizationCompatibilityError,
    NormalizationFitError,
    fit_feature_normalization,
    load_normalization_artifact,
    load_validated_normalizer,
    save_normalization_artifact,
    validate_normalization_compatibility,
)
from src.feature_providers import LookupTableFeatureProvider
from src.feature_schema import (
    BiophysicalFeatureProvider,
    FeatureCoordinateType,
)
from src.genome_dataset import SystematicNativeFeatureDataset


class MismatchedBatchProvider(BiophysicalFeatureProvider):
    """Declare one schema while returning another provider's batches."""

    def __init__(self, declared_provider, computed_provider) -> None:
        self.declared_provider = declared_provider
        self.computed_provider = computed_provider

    @property
    def provider_identifier(self) -> str:
        return self.declared_provider.provider_identifier

    @property
    def source_version(self) -> str:
        return self.declared_provider.source_version

    @property
    def provider_fingerprint(self) -> str:
        return self.declared_provider.provider_fingerprint

    @property
    def schemas(self):
        return self.declared_provider.schemas

    def compute(self, sequence: str):
        return self.computed_provider.compute(sequence)


class FeatureNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repository_root = Path(self.temporary_directory.name)
        self.data_directory = self.repository_root / "data"
        self.data_directory.mkdir()
        self.training_path = self._write_sequences(
            "training.txt",
            ("AAACCC", "AAACCC", "AANCCC"),
        )
        self.validation_path = self._write_sequences(
            "validation.txt",
            ("CGCGCG",),
        )
        self.training_before = self.training_path.read_bytes()
        self.validation_before = self.validation_path.read_bytes()
        self.manifest = build_pretraining_split_manifest(
            str(self.training_path),
            str(self.validation_path),
            str(self.repository_root),
            mode="report",
        )
        self.provider = self._build_provider()
        self.artifact = fit_feature_normalization(
            training_sequence_path=str(self.training_path),
            repository_root=str(self.repository_root),
            split_manifest=self.manifest,
            provider=self.provider,
            creation_entry_point="tests/test_feature_normalization.py",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_sequences(self, name: str, sequences) -> Path:
        path = self.data_directory / name
        with open(path, "w", encoding="utf-8", newline="\n") as output_file:
            for sequence in sequences:
                output_file.write(sequence)
                output_file.write("\n")
        return path

    def _build_provider(
        self,
        dinucleotide_feature_names=("step",),
        trinucleotide_feature_names=("base",),
        source_version="test-v1",
        step_ac_value=3.0,
    ) -> LookupTableFeatureProvider:
        lookup_data = {
            "dinucleotide": {
                "step": {
                    "AA": 1.0,
                    "TT": 1.0,
                    "AC": step_ac_value,
                    "GT": step_ac_value,
                    "CC": 7.0,
                    "GG": 7.0,
                }
            },
            "trinucleotide": {
                "base": {
                    "AAA": 2.0,
                    "TTT": 2.0,
                    "AAC": 4.0,
                    "GTT": 4.0,
                    "ACC": 6.0,
                    "GGT": 6.0,
                    "CCC": 8.0,
                    "GGG": 8.0,
                }
            },
        }
        return LookupTableFeatureProvider(
            lookup_data=lookup_data,
            dinucleotide_feature_names=dinucleotide_feature_names,
            trinucleotide_feature_names=trinucleotide_feature_names,
            source="test_lookup",
            source_version=source_version,
        )

    def _write_modified_artifact(self, payload, name: str) -> Path:
        modified_payload = copy.deepcopy(payload)
        content = copy.deepcopy(modified_payload)
        content.pop("artifact_hash")
        modified_payload["artifact_hash"] = hash_logical_content(content)
        path = self.repository_root / name
        path.write_text(
            json.dumps(modified_payload, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def test_training_only_statistics_exclude_invalid_and_nan_values(
        self,
    ) -> None:
        step_values = [
            1.0,
            1.0,
            3.0,
            7.0,
            7.0,
            1.0,
            1.0,
            3.0,
            7.0,
            7.0,
            1.0,
            7.0,
            7.0,
        ]
        base_values = [
            2.0,
            4.0,
            6.0,
            8.0,
            2.0,
            4.0,
            6.0,
            8.0,
            8.0,
        ]
        step_expected = torch.tensor(step_values, dtype=torch.float64)
        base_expected = torch.tensor(base_values, dtype=torch.float64)
        step_statistics = self.artifact.statistics[0]
        base_statistics = self.artifact.statistics[1]

        self.assertEqual(step_statistics.name, "step")
        self.assertEqual(step_statistics.valid_count, len(step_values))
        self.assertAlmostEqual(
            step_statistics.mean,
            step_expected.mean().item(),
        )
        self.assertAlmostEqual(
            step_statistics.standard_deviation,
            step_expected.std(unbiased=False).item(),
        )
        self.assertEqual(base_statistics.name, "base")
        self.assertEqual(base_statistics.valid_count, len(base_values))
        self.assertAlmostEqual(
            base_statistics.mean,
            base_expected.mean().item(),
        )
        self.assertAlmostEqual(
            base_statistics.standard_deviation,
            base_expected.std(unbiased=False).item(),
        )

    def test_base_and_base_step_statistics_remain_separate(self) -> None:
        self.assertEqual(
            self.artifact.coordinate_types,
            (
                FeatureCoordinateType.BASE_STEP_CENTERED.value,
                FeatureCoordinateType.BASE_CENTERED.value,
            ),
        )
        self.assertEqual(self.artifact.feature_order, (0, 1))
        self.assertEqual(self.artifact.feature_names, ("step", "base"))

    def test_artifact_content_is_deterministic(self) -> None:
        second_artifact = fit_feature_normalization(
            training_sequence_path=str(self.training_path),
            repository_root=str(self.repository_root),
            split_manifest=self.manifest,
            provider=self.provider,
            creation_entry_point="tests/test_feature_normalization.py",
        )

        self.assertEqual(self.artifact.to_dict(), second_artifact.to_dict())
        self.assertEqual(
            self.artifact.artifact_hash,
            second_artifact.artifact_hash,
        )
        self.assertEqual(
            self.artifact.artifact_hash,
            hash_logical_content(self.artifact.content_dict()),
        )
        self.assertNotIn("artifact_hash", self.artifact.content_dict())

    def test_artifact_serialization_round_trip_and_no_overwrite(self) -> None:
        first_path = self.repository_root / "normalization_v1_first.json"
        second_path = self.repository_root / "normalization_v1_second.json"
        save_normalization_artifact(self.artifact, str(first_path))
        save_normalization_artifact(self.artifact, str(second_path))

        loaded = load_normalization_artifact(str(first_path))

        self.assertEqual(loaded.to_dict(), self.artifact.to_dict())
        self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
        with self.assertRaises(FileExistsError):
            save_normalization_artifact(self.artifact, str(first_path))

    def test_old_artifact_schema_fails_before_v2_fields_are_read(self) -> None:
        payload = self.artifact.to_dict()
        payload["artifact_schema_version"] = "biophysical_normalization.v1"
        payload.pop("experiment_split_manifest_hash")
        payload.pop("feature_provider_fingerprint")
        path = self.repository_root / "normalization_v1.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported normalization artifact schema version",
        ):
            load_normalization_artifact(str(path))

    def test_legacy_normalization_output_cannot_replace_existing_artifact(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            validated_output_path(str(LEGACY_OUTPUT_PATH))

        versioned_path = self.repository_root / "legacy_stats_v2.yaml"
        self.assertEqual(
            validated_output_path(str(versioned_path)),
            versioned_path.resolve(),
        )
        versioned_path.write_text("mean: []\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            validated_output_path(str(versioned_path))

    def test_artifact_hash_rejects_scientific_metadata_tampering(self) -> None:
        modified_payloads = []

        statistic_payload = copy.deepcopy(self.artifact.to_dict())
        statistic_payload["statistics"][0]["mean"] += 1.0
        modified_payloads.append(("statistic", statistic_payload))

        name_payload = copy.deepcopy(self.artifact.to_dict())
        name_payload["feature_names"][0] = "renamed_step"
        name_payload["statistics"][0]["name"] = "renamed_step"
        modified_payloads.append(("feature_name", name_payload))

        source_payload = copy.deepcopy(self.artifact.to_dict())
        source_fingerprint = source_payload["training_source_fingerprint"]
        source_fingerprint["source_path"] = "data/renamed_training.txt"
        source_content = copy.deepcopy(source_fingerprint)
        source_content.pop("fingerprint_hash")
        source_fingerprint["fingerprint_hash"] = hash_logical_content(
            source_content
        )
        modified_payloads.append(("source_identity", source_payload))

        coordinate_payload = copy.deepcopy(self.artifact.to_dict())
        coordinate_payload["coordinate_types"][0] = (
            FeatureCoordinateType.BASE_CENTERED.value
        )
        coordinate_payload["statistics"][0]["coordinate_type"] = (
            FeatureCoordinateType.BASE_CENTERED.value
        )
        modified_payloads.append(("coordinate_type", coordinate_payload))

        unit_payload = copy.deepcopy(self.artifact.to_dict())
        unit_payload["units"][0] = "changed_unit"
        unit_payload["statistics"][0]["unit"] = "changed_unit"
        modified_payloads.append(("unit", unit_payload))

        split_payload = copy.deepcopy(self.artifact.to_dict())
        split_payload["experiment_split_manifest_hash"] = "0" * 64
        modified_payloads.append(("experiment_split", split_payload))

        provider_payload = copy.deepcopy(self.artifact.to_dict())
        provider_payload["feature_provider_fingerprint"] = "1" * 64
        modified_payloads.append(("provider", provider_payload))

        for name, payload in modified_payloads:
            with self.subTest(name=name):
                path = self.repository_root / ("tampered_" + name + ".json")
                path.write_text(
                    json.dumps(payload, sort_keys=True),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "artifact hash mismatch",
                ):
                    load_normalization_artifact(str(path))

    def test_feature_name_mismatch_is_rejected(self) -> None:
        payload = self.artifact.to_dict()
        payload["feature_names"][0] = "wrong_name"
        payload["statistics"][0]["name"] = "wrong_name"
        artifact_path = self._write_modified_artifact(
            payload,
            "wrong_name.json",
        )

        with self.assertRaisesRegex(
            NormalizationCompatibilityError,
            "feature names",
        ):
            load_validated_normalizer(
                str(artifact_path),
                self.provider,
                self.manifest,
            )

    def test_feature_order_mismatch_is_rejected(self) -> None:
        payload = self.artifact.to_dict()
        payload["feature_order"] = [1, 0]
        payload["statistics"][0]["feature_order"] = 1
        payload["statistics"][1]["feature_order"] = 0
        artifact_path = self._write_modified_artifact(
            payload,
            "wrong_order.json",
        )

        with self.assertRaisesRegex(ValueError, "feature order"):
            load_validated_normalizer(
                str(artifact_path),
                self.provider,
                self.manifest,
            )

    def test_feature_schema_hash_mismatch_is_rejected(self) -> None:
        payload = self.artifact.to_dict()
        payload["feature_schema_hash"] = "0" * 64
        artifact_path = self._write_modified_artifact(
            payload,
            "wrong_schema_hash.json",
        )

        with self.assertRaisesRegex(
            NormalizationCompatibilityError,
            "feature-schema hash",
        ):
            load_validated_normalizer(
                str(artifact_path),
                self.provider,
                self.manifest,
            )

    def test_provider_content_mismatch_is_rejected(self) -> None:
        changed_value_provider = self._build_provider(step_ac_value=9.0)

        self.assertEqual(
            tuple(self.provider.schemas),
            tuple(changed_value_provider.schemas),
        )
        self.assertNotEqual(
            self.provider.provider_fingerprint,
            changed_value_provider.provider_fingerprint,
        )
        with self.assertRaisesRegex(
            NormalizationCompatibilityError,
            "feature-provider fingerprint",
        ):
            validate_normalization_compatibility(
                self.artifact,
                changed_value_provider,
                self.manifest,
            )

    def test_fit_rejects_provider_batch_schema_mismatch(self) -> None:
        differently_named_provider = LookupTableFeatureProvider(
            lookup_data={
                "dinucleotide": {
                    "other_step": {
                        "AA": 1.0,
                        "TT": 1.0,
                        "AC": 3.0,
                        "GT": 3.0,
                        "CC": 7.0,
                        "GG": 7.0,
                    }
                },
                "trinucleotide": {
                    "other_base": {
                        "AAA": 2.0,
                        "TTT": 2.0,
                        "AAC": 4.0,
                        "GTT": 4.0,
                        "ACC": 6.0,
                        "GGT": 6.0,
                        "CCC": 8.0,
                        "GGG": 8.0,
                    }
                },
            },
            source="test_lookup",
            source_version="test-v1",
        )
        mismatched_provider = MismatchedBatchProvider(
            self.provider,
            differently_named_provider,
        )

        with self.assertRaisesRegex(
            NormalizationCompatibilityError,
            "batch schema",
        ):
            fit_feature_normalization(
                training_sequence_path=str(self.training_path),
                repository_root=str(self.repository_root),
                split_manifest=self.manifest,
                provider=mismatched_provider,
                creation_entry_point="batch_schema_mismatch_test",
            )

    def test_validation_change_preserves_fit_but_changes_experiment_identity(
        self,
    ) -> None:
        other_validation_path = self._write_sequences(
            "other_validation.txt",
            ("ATATAT",),
        )
        other_manifest = build_pretraining_split_manifest(
            str(self.training_path),
            str(other_validation_path),
            str(self.repository_root),
            mode="report",
        )
        other_artifact = fit_feature_normalization(
            training_sequence_path=str(self.training_path),
            repository_root=str(self.repository_root),
            split_manifest=other_manifest,
            provider=self.provider,
            creation_entry_point="tests/test_feature_normalization.py",
        )

        self.assertEqual(
            self.artifact.training_source_fingerprint.to_dict(),
            other_artifact.training_source_fingerprint.to_dict(),
        )
        self.assertEqual(
            self.artifact.statistics,
            other_artifact.statistics,
        )
        self.assertNotEqual(
            self.artifact.experiment_split_manifest_hash,
            other_artifact.experiment_split_manifest_hash,
        )

        with self.assertRaisesRegex(
            NormalizationCompatibilityError,
            "experiment split-manifest hash",
        ):
            validate_normalization_compatibility(
                self.artifact,
                self.provider,
                other_manifest,
            )

    def test_zero_variance_feature_is_rejected(self) -> None:
        constant_provider = LookupTableFeatureProvider(
            lookup_data={
                "dinucleotide": {
                    "constant": {
                        "AA": 1.0,
                        "TT": 1.0,
                    }
                },
                "trinucleotide": {},
            },
            source="constant_lookup",
            source_version="test-v1",
        )

        with self.assertRaisesRegex(
            NormalizationFitError,
            "constant.*zero variance",
        ):
            fit_feature_normalization(
                training_sequence_path=str(self.training_path),
                repository_root=str(self.repository_root),
                split_manifest=self.manifest,
                provider=constant_provider,
                creation_entry_point="zero_variance_test",
            )

    def test_insufficient_valid_data_is_rejected(self) -> None:
        one_value_path = self._write_sequences("one_value.txt", ("AA",))
        other_validation_path = self._write_sequences(
            "one_value_validation.txt",
            ("CCCC",),
        )
        one_value_manifest = build_pretraining_split_manifest(
            str(one_value_path),
            str(other_validation_path),
            str(self.repository_root),
        )
        provider = LookupTableFeatureProvider(
            lookup_data={
                "dinucleotide": {
                    "sparse": {
                        "AA": 1.0,
                        "TT": 1.0,
                    }
                },
                "trinucleotide": {},
            },
            source="sparse_lookup",
            source_version="test-v1",
        )

        with self.assertRaisesRegex(
            NormalizationFitError,
            "sparse.*at least two",
        ):
            fit_feature_normalization(
                training_sequence_path=str(one_value_path),
                repository_root=str(self.repository_root),
                split_manifest=one_value_manifest,
                provider=provider,
                creation_entry_point="insufficient_data_test",
            )

    def test_transform_and_inverse_transform_preserve_masks(self) -> None:
        normalizer = FeatureNormalizer(self.artifact)
        raw_batch = self.provider.compute("AAACCC")
        raw_base_values = raw_batch.base_features.values.clone()
        raw_step_values = raw_batch.base_step_features.values.clone()

        transformed = normalizer.transform(raw_batch)
        restored = normalizer.inverse_transform(transformed)

        self.assertTrue(
            torch.equal(
                transformed.base_features.valid_mask,
                raw_batch.base_features.valid_mask,
            )
        )
        self.assertTrue(torch.isnan(transformed.base_features.values[0, 0]))
        self.assertTrue(torch.isnan(transformed.base_features.values[-1, 0]))
        self.assertTrue(
            torch.allclose(
                restored.base_features.values,
                raw_batch.base_features.values,
                equal_nan=True,
            )
        )
        self.assertTrue(
            torch.allclose(
                restored.base_step_features.values,
                raw_batch.base_step_features.values,
                equal_nan=True,
            )
        )
        self.assertTrue(
            torch.allclose(
                raw_batch.base_features.values,
                raw_base_values,
                equal_nan=True,
            )
        )
        self.assertTrue(
            torch.allclose(
                raw_batch.base_step_features.values,
                raw_step_values,
                equal_nan=True,
            )
        )

    def test_systematic_dataset_requires_validated_artifacts(self) -> None:
        manifest_path = self.repository_root / "split_manifest.json"
        artifact_path = self.repository_root / "normalization_v1.json"
        save_split_manifest(self.manifest, str(manifest_path))
        save_normalization_artifact(self.artifact, str(artifact_path))

        dataset = SystematicNativeFeatureDataset(
            window_txt_path=str(self.training_path),
            split_role="training",
            repository_root=str(self.repository_root),
            provider=self.provider,
            split_manifest_path=str(manifest_path),
            normalization_artifact_path=str(artifact_path),
            max_rows=1,
        )
        example = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(example["seq"], "AAACCC")
        self.assertTrue(
            torch.isnan(
                example["feature_batch"].base_features.values[0, 0]
            )
        )
        self.assertTrue(
            torch.equal(
                example["feature_batch"].base_features.valid_mask,
                self.provider.compute("AAACCC").base_features.valid_mask,
            )
        )

        mismatched_provider = self._build_provider(source_version="wrong-v2")
        with self.assertRaises(NormalizationCompatibilityError):
            SystematicNativeFeatureDataset(
                window_txt_path=str(self.training_path),
                split_role="training",
                repository_root=str(self.repository_root),
                provider=mismatched_provider,
                split_manifest_path=str(manifest_path),
                normalization_artifact_path=str(artifact_path),
                max_rows=1,
            )

    def test_systematic_dataset_fails_closed_on_invalid_inputs(self) -> None:
        manifest_path = self.repository_root / "split_manifest.json"
        artifact_path = self.repository_root / "normalization_v1.json"
        save_split_manifest(self.manifest, str(manifest_path))
        save_normalization_artifact(self.artifact, str(artifact_path))

        missing_path = self.repository_root / "missing_normalization.json"
        with self.assertRaises(FileNotFoundError):
            SystematicNativeFeatureDataset(
                window_txt_path=str(self.training_path),
                split_role="training",
                repository_root=str(self.repository_root),
                provider=self.provider,
                split_manifest_path=str(manifest_path),
                normalization_artifact_path=str(missing_path),
            )

        legacy_path = self.repository_root / "legacy_normalization.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "feature_names": ["step", "base"],
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(KeyError):
            SystematicNativeFeatureDataset(
                window_txt_path=str(self.training_path),
                split_role="training",
                repository_root=str(self.repository_root),
                provider=self.provider,
                split_manifest_path=str(manifest_path),
                normalization_artifact_path=str(legacy_path),
            )

        other_training_path = self._write_sequences(
            "other_training.txt",
            ("AAACCC",),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            SystematicNativeFeatureDataset(
                window_txt_path=str(other_training_path),
                split_role="training",
                repository_root=str(self.repository_root),
                provider=self.provider,
                split_manifest_path=str(manifest_path),
                normalization_artifact_path=str(artifact_path),
            )

    def test_fit_and_transform_do_not_modify_source_files(self) -> None:
        normalizer = FeatureNormalizer(self.artifact)
        normalizer.transform(self.provider.compute("AAACCC"))

        self.assertEqual(self.training_path.read_bytes(), self.training_before)
        self.assertEqual(
            self.validation_path.read_bytes(),
            self.validation_before,
        )


if __name__ == "__main__":
    unittest.main()
