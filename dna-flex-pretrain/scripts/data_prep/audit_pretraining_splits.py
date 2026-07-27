"""Create a split audit.

Run with ``python -m scripts.data_prep.audit_pretraining_splits``.
"""

import argparse

from src.data_fingerprints import (
    build_pretraining_split_manifest,
    save_split_manifest,
    validate_new_artifact_output_path,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fingerprint and audit pretraining sequence files."
    )
    parser.add_argument("--training", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument(
        "--mode",
        choices=("report", "strict"),
        default="report",
    )
    parser.add_argument("--maximum-examples", type=int, default=10)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    manifest = build_pretraining_split_manifest(
        training_path=arguments.training,
        validation_path=arguments.validation,
        repository_root=arguments.repository_root,
        mode=arguments.mode,
        maximum_examples=arguments.maximum_examples,
    )
    if arguments.output is not None:
        output_path = validate_new_artifact_output_path(
            arguments.output,
            arguments.repository_root,
            allowed_relative_directories=("logs",),
            input_paths=(arguments.training, arguments.validation),
        )
        save_split_manifest(manifest, output_path)

    audit = manifest.overlap_audit
    print("manifest_hash:", manifest.manifest_hash)
    print(
        "exact_overlap_groups:",
        audit.exact_sequence_overlap.group_count,
    )
    print(
        "reverse_complement_overlap_groups:",
        audit.reverse_complement_equivalent_overlap.group_count,
    )
    print(
        "reverse_complement_only_groups:",
        audit.reverse_complement_only_group_count,
    )
    if arguments.output is not None:
        print("saved:", output_path)


if __name__ == "__main__":
    main()
