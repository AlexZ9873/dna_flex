"""Build immutable nested low-data subsets for the Exd-Hox primary split.

Run with ``python -m scripts.data_prep.build_exd_hox_nested_subsets``.
"""

from __future__ import annotations

import argparse
import json

from src.exd_hox_splits import build_nested_subset_artifacts


DEFAULT_CONFIG_PATH = "configs/exd_hox_primary_split_v1.yaml"


def parse_arguments(argv=None):
    """Parse command-line arguments and require paired test overrides."""

    parser = argparse.ArgumentParser(
        description="Build immutable nested Exd-Hox low-data subsets."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--split-directory",
        help="Optional temporary physical directory containing the split.",
    )
    parser.add_argument(
        "--output-directory",
        help="Optional temporary physical directory for subset artifacts.",
    )
    arguments = parser.parse_args(argv)
    split_is_overridden = arguments.split_directory is not None
    output_is_overridden = arguments.output_directory is not None
    if split_is_overridden != output_is_overridden:
        parser.error(
            "--split-directory and --output-directory must be provided "
            "together."
        )
    return arguments


def main(argv=None):
    """Build the subset set once and print its deterministic manifest."""

    arguments = parse_arguments(argv)
    manifest = build_nested_subset_artifacts(
        config_path=arguments.config,
        repository_root=arguments.repository_root,
        split_directory=arguments.split_directory,
        output_directory=arguments.output_directory,
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
