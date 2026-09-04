"""Build the immutable exact/RC-safe Exd-Hox primary split artifacts.

Run with ``python -m scripts.data_prep.build_exd_hox_primary_split``.
"""

from __future__ import annotations

import argparse
import json

from src.exd_hox_splits import build_primary_split_artifacts


DEFAULT_CONFIG_PATH = "configs/exd_hox_primary_split_v1.yaml"


def parse_arguments(argv=None):
    """Parse command-line arguments and require paired test overrides."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the immutable exact/RC-safe Exd-Hox primary split."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--output-directory",
        help="Optional temporary physical directory for public artifacts.",
    )
    parser.add_argument(
        "--sealed-target-directory",
        help="Optional temporary physical directory for sealed test targets.",
    )
    arguments = parser.parse_args(argv)
    output_is_overridden = arguments.output_directory is not None
    sealed_is_overridden = arguments.sealed_target_directory is not None
    if output_is_overridden != sealed_is_overridden:
        parser.error(
            "--output-directory and --sealed-target-directory must be "
            "provided together."
        )
    return arguments


def main(argv=None):
    """Build the split once and print its deterministic manifest."""

    arguments = parse_arguments(argv)
    manifest = build_primary_split_artifacts(
        config_path=arguments.config,
        repository_root=arguments.repository_root,
        output_directory=arguments.output_directory,
        sealed_target_directory=arguments.sealed_target_directory,
    )
    print(
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return manifest


if __name__ == "__main__":
    main()
