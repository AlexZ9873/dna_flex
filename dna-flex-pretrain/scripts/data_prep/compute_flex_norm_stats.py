"""Legacy 6-mer-averaged normalization script; not for systematic studies."""

import argparse
from pathlib import Path
import re

import yaml
import torch

from src.utils import load_yaml
from src.flex_features import load_lookup_yaml, sequence_to_multi_dinuc_targets, sequence_to_multi_trinuc_targets


VERSIONED_OUTPUT_PATTERN = re.compile(r"(^|[_-])v[0-9]+($|[_-])")
LEGACY_OUTPUT_PATH = Path("data/processed/flex_norm_stats.yaml").resolve()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute legacy 6-mer-averaged normalization statistics."
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validated_output_path(path: str) -> Path:
    """Require a new versioned YAML path that cannot replace the old artifact."""

    output_path = Path(path).resolve()
    if output_path == LEGACY_OUTPUT_PATH:
        raise ValueError("Refusing to replace the legacy normalization artifact.")
    if output_path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Legacy normalization output must use YAML.")
    if VERSIONED_OUTPUT_PATTERN.search(output_path.stem) is None:
        raise ValueError(
            "Legacy normalization output must contain a version such as '_v1'."
        )
    if output_path.exists():
        message = "Legacy normalization output already exists: {0}"
        raise FileExistsError(message.format(output_path))
    return output_path


def main():
    arguments = parse_arguments()
    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]

    dinuc_feats = cfg["features"]["dinucleotide"]
    trinuc_feats = cfg["features"]["trinucleotide"]
    feature_names = dinuc_feats + trinuc_feats
    F = len(feature_names)

    lookup = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    train_path = "data/raw/hg38_windows_256_train.txt"
    out_path = validated_output_path(arguments.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Welford-style batch combination stats
    n = 0
    mean = torch.zeros(F, dtype=torch.float64)
    M2 = torch.zeros(F, dtype=torch.float64)

    max_windows = 5000  # better stats for 50k-window training set

    with open(train_path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_windows:
                break
            seq = line.strip().upper()
            if not seq:
                continue

            # targets per k-mer token
            L = len(seq) - k + 1

            if len(dinuc_feats) > 0:
                _, di = sequence_to_multi_dinuc_targets(seq, k, lookup, dinuc_feats)
            else:
                di = [[] for _ in range(L)]

            if len(trinuc_feats) > 0:
                _, tri = sequence_to_multi_trinuc_targets(seq, k, lookup, trinuc_feats)
            else:
                tri = [[] for _ in range(L)]

            combined = [row_di + row_tri for row_di, row_tri in zip(di, tri)]
            x = torch.tensor(combined, dtype=torch.float64)  # [L, F]

            k_batch = x.shape[0]
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_M2 = batch_var * k_batch

            if n == 0:
                mean = batch_mean
                M2 = batch_M2
                n = k_batch
            else:
                delta = batch_mean - mean
                new_n = n + k_batch
                mean = mean + delta * (k_batch / new_n)
                M2 = M2 + batch_M2 + (delta * delta) * (n * k_batch / new_n)
                n = new_n

    std = torch.sqrt(M2 / n)
    std[std == 0] = 1.0

    stats = {
        "feature_names": feature_names,
        "mean": mean.float().tolist(),
        "std": std.float().tolist(),
        "n_tokens": int(n),
        "max_windows_used": int(max_windows),
    }

    with open(out_path, "x") as f:
        yaml.safe_dump(stats, f)

    print("saved:", out_path)
    print("features:", feature_names)
    print("n_tokens:", int(n))
    print("mean[:4]:", stats["mean"][:4])
    print("std[:4]:", stats["std"][:4])

if __name__ == "__main__":
    main()
