"""Build or dry-run the approved coordinate-preserving hg38 split.

Run with ``python -m scripts.data_prep.build_hg38_pretraining_split``.
"""

import argparse
import json
from pathlib import Path

from src.genomic_splits import (
    WholeChromosomeSplitConfig,
    build_hg38_pretraining_split,
)


def parse_arguments(argv=None):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the versioned whole-chromosome hg38 pretraining split."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report capacity without creating any output.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run a no-write capacity scan or an exclusive production build."""

    arguments = parse_arguments(argv)
    repository_root = str(Path(arguments.repository_root).resolve())
    config = WholeChromosomeSplitConfig.from_yaml(arguments.config)
    result = build_hg38_pretraining_split(
        config=config,
        repository_root=repository_root,
        dry_run=arguments.dry_run,
    )
    print(
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return result


if __name__ == "__main__":
    main()
