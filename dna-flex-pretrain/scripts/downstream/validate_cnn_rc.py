"""Verify a checkpoint's already recorded fixed CNN-RC validation event."""

import argparse
import json
from pathlib import Path
import sys

from src.cnn_rc_training import verify_cnn_rc
from src.downstream_checkpoint import PublishedDurabilityError


def argument_parser() -> argparse.ArgumentParser:
    """Derive scientific choices from the checkpoint without overrides."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--expected-stage-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--expected-software-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Verify the immutable event and distinguish uncertain durability."""
    arguments = argument_parser().parse_args(argv)
    try:
        result = verify_cnn_rc(**vars(arguments))
    except PublishedDurabilityError as error:
        print(json.dumps({"status": "published_but_durability_unconfirmed", "error": str(error)},
                         sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)},
                         sort_keys=True, allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
