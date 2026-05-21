import math
import random
from collections import Counter

import numpy as np
import yaml
from sklearn.linear_model import Ridge


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_pbm_tsv(path: str):
    seqs = []
    ys = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            seq = parts[0].upper()
            y = float(parts[1])
            seqs.append(seq)
            ys.append(y)
    return seqs, np.array(ys, dtype=np.float32)


def choose_common_length(seqs):
    lengths = [len(s) for s in seqs]
    c = Counter(lengths)
    L, n = c.most_common(1)[0]
    return L, c


def onehot_1mer_positional(seqs, L):
    """
    Positional 1-mer encoding:
    For each position i in [0..L-1], one-hot(A,C,G,T) -> 4 dims.
    Total features = 4*L.
    """
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    X = np.zeros((len(seqs), 4 * L), dtype=np.float32)
    for r, s in enumerate(seqs):
        for i, ch in enumerate(s[:L]):
            j = base_to_idx.get(ch, None)
            if j is not None:
                X[r, 4 * i + j] = 1.0
    return X


def split_indices(n, seed, train_frac, val_frac):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def pearson(y, yhat):
    y = y.astype(np.float64)
    yhat = yhat.astype(np.float64)
    y0 = y - y.mean()
    p0 = yhat - yhat.mean()
    denom = (y0.std() * p0.std()) + 1e-12
    return float((y0 * p0).mean() / denom)


def r2(y, yhat):
    y = y.astype(np.float64)
    yhat = yhat.astype(np.float64)
    sse = np.sum((y - yhat) ** 2)
    sst = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1.0 - sse / sst)


def rmse(y, yhat):
    y = y.astype(np.float64)
    yhat = yhat.astype(np.float64)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def main():
    cfg = load_yaml("configs/finetune_pbm.yaml")
    data_path = cfg["data_path"]

    seed = int(cfg["train"]["seed"])
    train_frac = float(cfg["train"]["split"]["train"])
    val_frac = float(cfg["train"]["split"]["val"])

    print("PBM baseline: Ridge regression on positional 1-mer features")
    print("data =", data_path)
    print("seed =", seed)
    print("splits =", train_frac, val_frac, 1.0 - train_frac - val_frac)

    seqs, y = read_pbm_tsv(data_path)
    L_common, length_counts = choose_common_length(seqs)

    # filter to common length (PBM usually fixed length, but this makes it robust)
    keep = [i for i, s in enumerate(seqs) if len(s) == L_common]
    if len(keep) != len(seqs):
        print("WARNING: Not all sequences have same length.")
        print("length counts =", dict(length_counts))
        print(f"Filtering to most common length L={L_common}: keeping {len(keep)}/{len(seqs)}")
        seqs = [seqs[i] for i in keep]
        y = y[keep]

    n = len(seqs)
    train_idx, val_idx, test_idx = split_indices(n, seed, train_frac, val_frac)
    print("n_sequences =", n, "splits:", len(train_idx), len(val_idx), len(test_idx))
    print("sequence_length =", L_common, "feature_dim =", 4 * L_common)

    X = onehot_1mer_positional(seqs, L_common)

    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    Xte, yte = X[test_idx], y[test_idx]

    # tune alpha on validation (like you would tune ridge strength)
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    best = None

    for a in alphas:
        m = Ridge(alpha=a, fit_intercept=True)
        m.fit(Xtr, ytr)
        pva = m.predict(Xva)
        score = r2(yva, pva)
        if (best is None) or (score > best["val_r2"]):
            best = {"alpha": a, "val_r2": score, "val_rmse": rmse(yva, pva), "val_pearson": pearson(yva, pva)}

    print()
    print("BEST alpha (picked on VAL)")
    print("alpha =", best["alpha"])
    print("val_r2 =", best["val_r2"])
    print("val_rmse =", best["val_rmse"])
    print("val_pearson =", best["val_pearson"])

    # refit on train+val with best alpha, then evaluate on test
    idx_trva = np.array(train_idx + val_idx, dtype=np.int64)
    Xtrva, ytrva = X[idx_trva], y[idx_trva]

    final = Ridge(alpha=best["alpha"], fit_intercept=True)
    final.fit(Xtrva, ytrva)
    pte = final.predict(Xte)

    print()
    print("TEST (fit on TRAIN+VAL with best alpha)")
    print("test_r2 =", r2(yte, pte))
    print("test_rmse =", rmse(yte, pte))
    print("test_pearson =", pearson(yte, pte))


if __name__ == "__main__":
    main()
