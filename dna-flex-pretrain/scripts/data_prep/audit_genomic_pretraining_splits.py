"""Audit generated coordinate-preserving pretraining split records.

Run with ``python -m scripts.data_prep.audit_genomic_pretraining_splits``.
"""

import argparse
import json
from pathlib import Path

from src.genomic_splits import (
    WholeChromosomeSplitConfig,
    validate_generated_artifacts,
)


def parse_arguments(argv=None):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Audit all three coordinate-preserving genomic splits."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any cross-split leakage is detected.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Read, reference-check, and audit all configured coordinate TSVs."""

    arguments = parse_arguments(argv)
    repository_root = str(Path(arguments.repository_root).resolve())
    config = WholeChromosomeSplitConfig.from_yaml(arguments.config)
    mode = "strict" if arguments.strict else "report"
    audit = validate_generated_artifacts(
        config,
        repository_root,
        mode=mode,
    )
    print(
        json.dumps(
            audit,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return audit


if __name__ == "__main__":
    main()
