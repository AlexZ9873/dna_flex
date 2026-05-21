import os
import csv
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.model import TinyMultiTaskModelOneHot
from src.utils import load_yaml


# -------------------------
# Basic utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pearsonr_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def split_indices(n, seed=0, train_frac=0.8, val_frac=0.1):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    idx_train = idx[:n_train].tolist()
    idx_val = idx[n_train:n_train+n_val].tolist()
    idx_test = idx[n_train+n_val:].tolist()

    return idx_train, idx_val, idx_test


# -------------------------
# Tokenization
# -------------------------
BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}

def seq_to_kmer_onehot(seq, k=6):
    seq = seq.upper()
    T = len(seq) - k + 1
    if T <= 0:
        raise ValueError(f"Sequence shorter than k={k}: {seq}")

    x = np.zeros((T, k * 4), dtype=np.float32)
    for t in range(T):
        kmer = seq[t:t+k]
        for i, ch in enumerate(kmer):
            j = BASE_TO_ID.get(ch, None)
            if j is not None:
                x[t, i * 4 + j] = 1.0
    return x


def onehot_1mer_positional(seq):
    seq = seq.upper()
    L = len(seq)
    X = np.zeros((L, 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = BASE_TO_ID.get(ch, None)
        if j is not None:
            X[i, j] = 1.0
    return X.reshape(-1)


# -------------------------
# Dataset
# -------------------------
class HTSelexDataset(Dataset):
    def __init__(self, path, k=6):
        self.path = str(path)
        self.k = k
        self.items = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                seq = parts[0].upper()
                try:
                    score = float(parts[1])
                except ValueError:
                    continue

                count = None
                if len(parts) >= 3:
                    try:
                        count = float(parts[2])
                    except ValueError:
                        count = None

                if len(seq) < k:
                    continue

                self.items.append((seq, score, count))

        if len(self.items) == 0:
            raise ValueError(f"No usable rows in {path}")

        lengths = Counter(len(x[0]) for x in self.items)
        self.major_len = lengths.most_common(1)[0][0]

        # For first version, keep only the dominant sequence length per file.
        # This keeps feature dimensions fixed inside each file.
        self.items = [x for x in self.items if len(x[0]) == self.major_len]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, score, count = self.items[idx]
        x = torch.tensor(seq_to_kmer_onehot(seq, k=self.k), dtype=torch.float32)
        am = torch.ones(x.shape[0], dtype=torch.long)
        y = torch.tensor([score], dtype=torch.float32)
        return {
            "seq": seq,
            "x": x,
            "attention_mask": am,
            "y": y,
            "count": count,
        }


def collate_pad(batch):
    B = len(batch)
    maxT = max(item["x"].shape[0] for item in batch)
    D = batch[0]["x"].shape[1]

    x = torch.zeros((B, maxT, D), dtype=torch.float32)
    am = torch.zeros((B, maxT), dtype=torch.long)
    y = torch.zeros((B, 1), dtype=torch.float32)
    seqs = []

    for i, item in enumerate(batch):
        T = item["x"].shape[0]
        x[i, :T] = item["x"]
        am[i, :T] = 1
        y[i] = item["y"]
        seqs.append(item["seq"])

    return {
        "x": x,
        "attention_mask": am,
        "y": y,
        "seq": seqs,
    }


# -------------------------
# Model loading
# -------------------------
def build_pretrained_model(cfg_pre, ckpt_path):
    k = int(cfg_pre["tokenizer"]["k"])
    n_flex = len(cfg_pre["features"]["dinucleotide"]) + len(cfg_pre["features"]["trinucleotide"])

    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=4100,
        d_model=int(cfg_pre["model"]["d_model"]),
        n_heads=int(cfg_pre["model"]["n_heads"]),
        n_layers=int(cfg_pre["model"]["n_layers"]),
        max_len=int(cfg_pre["model"]["max_len"]),
        n_flex=n_flex,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model


# -------------------------
# Feature extraction
# -------------------------
@torch.no_grad()
def extract_hiddenflex_features(model, ds, indices, batch_size=256, device="cpu"):
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pad,
    )

    feats = []
    ys = []

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].squeeze(1).cpu().numpy()

        _, flex_pred, h = model(x, am, return_hidden=True)

        mask_f = am.unsqueeze(-1).float()
        h = h * mask_f
        flex_pred = flex_pred * mask_f

        z = torch.cat([h, flex_pred], dim=-1)  # [B,T,76]
        feat = z.reshape(z.shape[0], -1)

        feats.append(feat.cpu().numpy())
        ys.append(y)

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(ys, axis=0)

    return X, y


def build_1mer_features(ds, indices):
    X_list = []
    y_list = []

    for i in indices:
        item = ds[i]
        X_list.append(onehot_1mer_positional(item["seq"]))
        y_list.append(float(item["y"].item()))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)

    return X, y


def standardize_train_apply(Xtr, Xva, Xte):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


# -------------------------
# Ridge evaluation
# -------------------------
def fit_ridge_eval(Xtr, ytr, Xva, yva, Xte, yte, alphas):
    best_alpha = None
    best_val_r2 = -1e9
    best_model = None

    for a in alphas:
        reg = Ridge(alpha=a, fit_intercept=True, random_state=0)
        reg.fit(Xtr, ytr)
        pva = reg.predict(Xva)
        val_r2 = r2_score(yva, pva)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = a
            best_model = reg

    # Fit on train + val using chosen alpha
    Xtv = np.concatenate([Xtr, Xva], axis=0)
    ytv = np.concatenate([ytr, yva], axis=0)

    final = Ridge(alpha=best_alpha, fit_intercept=True, random_state=0)
    final.fit(Xtv, ytv)
    pte = final.predict(Xte)

    return {
        "alpha": float(best_alpha),
        "val_r2": float(best_val_r2),
        "test_r2": float(r2_score(yte, pte)),
        "test_rmse": float(np.sqrt(np.mean((yte - pte) ** 2))),
        "test_pearson": float(pearsonr_np(yte, pte)),
    }


# -------------------------
# Main benchmark
# -------------------------
def parse_family(filename):
    return filename.split("_")[0]


def short_tf_name(filename):
    parts = filename.replace(".txt", "").split("_")
    if len(parts) >= 2:
        return parts[1]
    return filename.replace(".txt", "")


def run_one_file(path, model, cfg_pre, seed=0, alphas=None, max_rows=None):
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    ds = HTSelexDataset(path, k=int(cfg_pre["tokenizer"]["k"]))

    # Optional cap for pilot/debug speed.
    if max_rows is not None and len(ds) > max_rows:
        rng = np.random.RandomState(seed)
        keep = rng.choice(np.arange(len(ds)), size=max_rows, replace=False).tolist()
        ds.items = [ds.items[i] for i in keep]

    n = len(ds)
    if n < 200:
        raise ValueError(f"Too few rows after filtering: {path}, n={n}")

    idx_train, idx_val, idx_test = split_indices(
        n,
        seed=seed,
        train_frac=0.8,
        val_frac=0.1,
    )

    # 1-mer ridge
    Xtr1, ytr1 = build_1mer_features(ds, idx_train)
    Xva1, yva1 = build_1mer_features(ds, idx_val)
    Xte1, yte1 = build_1mer_features(ds, idx_test)

    res_1mer = fit_ridge_eval(Xtr1, ytr1, Xva1, yva1, Xte1, yte1, alphas)

    # Transformer hidden+flex ridge
    Xtrf, ytrf = extract_hiddenflex_features(model, ds, idx_train)
    Xvaf, yvaf = extract_hiddenflex_features(model, ds, idx_val)
    Xtef, ytef = extract_hiddenflex_features(model, ds, idx_test)

    Xtrf, Xvaf, Xtef = standardize_train_apply(Xtrf, Xvaf, Xtef)

    res_trf = fit_ridge_eval(Xtrf, ytrf, Xvaf, yvaf, Xtef, ytef, alphas)

    return {
        "file": Path(path).name,
        "family": parse_family(Path(path).name),
        "tf": short_tf_name(Path(path).name),
        "n": n,
        "seq_len": ds.major_len,
        "seed": seed,
        "r2_1mer": res_1mer["test_r2"],
        "pearson_1mer": res_1mer["test_pearson"],
        "rmse_1mer": res_1mer["test_rmse"],
        "alpha_1mer": res_1mer["alpha"],
        "r2_transformer_hiddenflex": res_trf["test_r2"],
        "pearson_transformer_hiddenflex": res_trf["test_pearson"],
        "rmse_transformer_hiddenflex": res_trf["test_rmse"],
        "alpha_transformer_hiddenflex": res_trf["alpha"],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/raw/htselex")
    parser.add_argument("--limit_files", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--out_prefix", default="plots/htselex_panelA_pilot")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")
    ckpt_path = cfg_ft["pretrained_ckpt"]

    model = build_pretrained_model(cfg_pre, ckpt_path)

    folder = Path(args.folder)
    files = sorted(folder.glob("*.txt"))

    if args.limit_files is not None:
        files = files[:args.limit_files]

    print(f"Running {len(files)} HT-SELEX files")
    print("seed =", args.seed)
    print()

    rows = []

    for i, path in enumerate(files, 1):
        try:
            row = run_one_file(path, model, cfg_pre, seed=args.seed, max_rows=args.max_rows)
            rows.append(row)
            print(
                f"[{i}/{len(files)}] {path.name} | "
                f"n={row['n']} L={row['seq_len']} | "
                f"1mer={row['r2_1mer']:.3f} | "
                f"trf_hiddenflex={row['r2_transformer_hiddenflex']:.3f}"
            )
        except Exception as e:
            print(f"[{i}/{len(files)}] SKIP {path.name}: {e}")

    if not rows:
        raise SystemExit("No successful files.")

    out_csv = args.out_prefix + ".csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plot
    families = sorted(set(r["family"] for r in rows))
    cmap = plt.get_cmap("tab20")
    color_map = {fam: cmap(i % 20) for i, fam in enumerate(families)}

    xs = [r["r2_1mer"] for r in rows]
    ys = [r["r2_transformer_hiddenflex"] for r in rows]

    lo = min(xs + ys) - 0.05
    hi = max(xs + ys) + 0.05
    lo = max(lo, -0.2)
    hi = min(hi, 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 6.2), dpi=220)

    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1.2)

    for r in rows:
        ax.scatter(
            r["r2_1mer"],
            r["r2_transformer_hiddenflex"],
            color=color_map[r["family"]],
            s=45,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Test $R^2$ (1-mer ridge)", fontsize=13)
    ax.set_ylabel("Test $R^2$ (Transformer hidden+flex + ridge)", fontsize=13)
    ax.set_title("HT-SELEX transfer benchmark: 1-mer vs transformer representation", fontsize=13)
    ax.grid(True, alpha=0.25)

    # keep legend manageable for pilot
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[fam],
               markeredgecolor="black", markersize=7, label=fam)
        for fam in families
    ]
    ax.legend(handles=handles, title="family", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)

    fig.subplots_adjust(right=0.75)

    out_png = args.out_prefix + ".png"
    fig.savefig(out_png, bbox_inches="tight")

    print()
    print("Saved:", out_png)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
