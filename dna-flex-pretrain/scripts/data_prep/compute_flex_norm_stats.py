import os
import yaml
import torch

from src.utils import load_yaml
from src.flex_features import load_lookup_yaml, sequence_to_multi_dinuc_targets, sequence_to_multi_trinuc_targets

def main():
    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]

    dinuc_feats = cfg["features"]["dinucleotide"]
    trinuc_feats = cfg["features"]["trinucleotide"]
    feature_names = dinuc_feats + trinuc_feats
    F = len(feature_names)

    lookup = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    train_path = "data/raw/hg38_windows_256_train.txt"
    out_path = "data/processed/flex_norm_stats.yaml"

    os.makedirs("data/processed", exist_ok=True)

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

    with open(out_path, "w") as f:
        yaml.safe_dump(stats, f)

    print("saved:", out_path)
    print("features:", feature_names)
    print("n_tokens:", int(n))
    print("mean[:4]:", stats["mean"][:4])
    print("std[:4]:", stats["std"][:4])

if __name__ == "__main__":
    main()
