"""Fit a versioned normalization artifact.

Run with ``python -m scripts.data_prep.fit_biophysical_normalization``.
"""

import argparse

from src.data_fingerprints import (
    load_split_manifest,
    validate_new_artifact_output_path,
)
from src.feature_normalization import (
    fit_feature_normalization,
    save_normalization_artifact,
)
from src.feature_providers import LookupTableFeatureProvider


CREATION_ENTRY_POINT = (
    "python -m scripts.data_prep.fit_biophysical_normalization"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fit training-only native-coordinate feature statistics."
    )
    parser.add_argument("--training", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--lookup", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider-source-version", required=True)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    split_manifest = load_split_manifest(arguments.split_manifest)
    provider = LookupTableFeatureProvider.from_yaml(
        arguments.lookup,
        source_version=arguments.provider_source_version,
        repository_root=arguments.repository_root,
    )
    output_path = validate_new_artifact_output_path(
        arguments.output,
        arguments.repository_root,
        allowed_relative_directories=("data/processed", "logs"),
        input_paths=(
            arguments.training,
            arguments.split_manifest,
            arguments.lookup,
        ),
    )
    artifact = fit_feature_normalization(
        training_sequence_path=arguments.training,
        repository_root=arguments.repository_root,
        split_manifest=split_manifest,
        provider=provider,
        creation_entry_point=CREATION_ENTRY_POINT,
    )
    save_normalization_artifact(artifact, output_path)
    print("artifact_hash:", artifact.artifact_hash)
    print(
        "experiment_split_manifest_hash:",
        artifact.experiment_split_manifest_hash,
    )
    print("saved:", output_path)


if __name__ == "__main__":
    main()
