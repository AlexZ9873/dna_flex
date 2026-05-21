import os
import csv
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.model import TinyMultiTaskModelOneHot
from src.utils import load_yaml


# -------------------------
# Utils
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


def standardize_train_apply(Xtr, Xte):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}


# -------------------------
# HT-SELEX dataset
# -------------------------
class HTSelexDataset(Dataset):
    def __init__(self, path, k=6, max_rows=None, seed=0):
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

                if len(seq) < k:
                    continue

                self.items.append((seq, score))

        if len(self.items) == 0:
            raise ValueError(f"No usable rows in {path}")

        # keep dominant sequence length per file
        lengths = Counter(len(s) for s, _ in self.items)
        self.major_len = lengths.most_common(1)[0][0]
        self.items = [(s, y) for s, y in self.items if len(s) == self.major_len]

        if max_rows is not None and len(self.items) > max_rows:
            rng = np.random.RandomState(seed)
            keep = rng.choice(np.arange(len(self.items)), size=max_rows, replace=False)
            self.items = [self.items[i] for i in keep]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, score = self.items[idx]
        x = torch.tensor(seq_to_kmer_onehot(seq, k=self.k), dtype=torch.float32)
        am = torch.ones(x.shape[0], dtype=torch.long)
        y = torch.tensor([score], dtype=torch.float32)
        return {"seq": seq, "x": x, "attention_mask": am, "y": y}


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

    return {"x": x, "attention_mask": am, "y": y, "seq": seqs}


# -------------------------
# Sequence encodings
# -------------------------
def seq_to_kmer_onehot(seq, k=6):
    seq = seq.upper()
    T = len(seq) - k + 1
    x = np.zeros((T, k * 4), dtype=np.float32)

    for t in range(T):
        kmer = seq[t:t+k]
        for i, ch in enumerate(kmer):
            j = BASE_TO_ID.get(ch, None)
            if j is not None:
                x[t, i * 4 + j] = 1.0
    return x


def positional_nmer_onehot(seq, n):
    seq = seq.upper()
    L = len(seq)
    windows = L - n + 1
    vocab_size = 4 ** n
    X = np.zeros((windows, vocab_size), dtype=np.float32)

    for i in range(windows):
        word = seq[i:i+n]
        idx = 0
        ok = True
        for ch in word:
            v = BASE_TO_ID.get(ch, None)
            if v is None:
                ok = False
                break
            idx = idx * 4 + v
        if ok:
            X[i, idx] = 1.0

    return X.reshape(-1)


def build_feature_matrix(ds, kind):
    X_list = []
    y_list = []

    for i in range(len(ds)):
        item = ds[i]
        seq = item["seq"]
        y = float(item["y"].item())

        if kind == "1mer":
            feat = positional_nmer_onehot(seq, 1)
        elif kind == "2mer":
            feat = positional_nmer_onehot(seq, 2)
        elif kind == "3mer":
            feat = positional_nmer_onehot(seq, 3)
        else:
            raise ValueError(f"Unknown feature kind: {kind}")

        X_list.append(feat)
        y_list.append(y)

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.float32)


# -------------------------
# Transformer hidden+flex feature extraction
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


@torch.no_grad()
def extract_transformer_hiddenflex(model, ds, batch_size=256, device="cpu"):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

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

    return np.concatenate(feats, axis=0), np.concatenate(ys, axis=0)


# -------------------------
# Outer CV evaluation
# -------------------------
def outer_cv_ridge(X, y, outer_folds=5, seed=0, standardize=False):
    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=np.float32)

    alphas = np.array([0.1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64)

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X)):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        if standardize:
            Xtr, Xte = standardize_train_apply(Xtr, Xte)

        reg = RidgeCV(alphas=alphas, store_cv_results=False)
        reg.fit(Xtr, ytr)
        oof[te_idx] = reg.predict(Xte)

    return {
        "r2": float(r2_score(y, oof)),
        "pearson": float(pearsonr_np(y, oof)),
        "rmse": float(np.sqrt(np.mean((y - oof) ** 2))),
    }


# -------------------------
# Main per-file benchmark
# -------------------------
def parse_family(filename):
    return filename.split("_")[0]


def short_tf_name(filename):
    parts = filename.replace(".txt", "").split("_")
    return parts[1] if len(parts) >= 2 else filename.replace(".txt", "")


def run_one_file(path, cfg_pre, model, outer_folds=5, seed=0, max_rows=5000):
    ds = HTSelexDataset(path, k=int(cfg_pre["tokenizer"]["k"]), max_rows=max_rows, seed=seed)

    if len(ds) < outer_folds * 20:
        raise ValueError(f"Too few rows after filtering/subsampling: n={len(ds)}")

    X1, y = build_feature_matrix(ds, "1mer")
    X2, y2 = build_feature_matrix(ds, "2mer")
    X3, y3 = build_feature_matrix(ds, "3mer")
    Xtf, ytf = extract_transformer_hiddenflex(model, ds)

    res1 = outer_cv_ridge(X1, y, outer_folds=outer_folds, seed=seed, standardize=False)
    res2 = outer_cv_ridge(X2, y2, outer_folds=outer_folds, seed=seed, standardize=False)
    res3 = outer_cv_ridge(X3, y3, outer_folds=outer_folds, seed=seed, standardize=False)
    restf = outer_cv_ridge(Xtf, ytf, outer_folds=outer_folds, seed=seed, standardize=True)

    return {
        "file": Path(path).name,
        "family": parse_family(Path(path).name),
        "tf": short_tf_name(Path(path).name),
        "n_rows_used": len(ds),
        "seq_len": ds.major_len,
        "cv_r2_1mer": res1["r2"],
        "cv_r2_2mer": res2["r2"],
        "cv_r2_3mer": res3["r2"],
        "cv_r2_transformer_hiddenflex": restf["r2"],
        "pearson_1mer": res1["pearson"],
        "pearson_2mer": res2["pearson"],
        "pearson_3mer": res3["pearson"],
        "pearson_transformer_hiddenflex": restf["pearson"],
        "rmse_1mer": res1["rmse"],
        "rmse_2mer": res2["rmse"],
        "rmse_3mer": res3["rmse"],
        "rmse_transformer_hiddenflex": restf["rmse"],
    }


# -------------------------
# Plotting
# -------------------------
def make_panelA(rows, out_png):
    families = sorted(set(r["family"] for r in rows))
    cmap = plt.get_cmap("tab20")
    color_map = {fam: cmap(i % 20) for i, fam in enumerate(families)}

    comparisons = [
        ("cv_r2_1mer", "1-mer ridge"),
        ("cv_r2_2mer", "2-mer ridge"),
        ("cv_r2_3mer", "3-mer ridge"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=220)

    for ax, (xkey, xlabel) in zip(axes, comparisons):
        xs = [r[xkey] for r in rows]
        ys = [r["cv_r2_transformer_hiddenflex"] for r in rows]

        lo = min(xs + ys) - 0.05
        hi = max(xs + ys) + 0.05
        lo = max(lo, -0.2)
        hi = min(hi, 1.0)

        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1.2)

        for r in rows:
            ax.scatter(
                r[xkey],
                r["cv_r2_transformer_hiddenflex"],
                color=color_map[r["family"]],
                s=40,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.3,
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"CV $R^2$ ({xlabel})", fontsize=11)
        ax.set_ylabel("CV $R^2$ (Transformer hidden+flex + ridge)", fontsize=11)
        ax.set_title(f"Transformer vs {xlabel}", fontsize=12)
        ax.grid(True, alpha=0.25)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color_map[fam], markeredgecolor="black",
               markersize=7, label=fam)
        for fam in families
    ]

    fig.legend(handles=handles, title="family", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle("HT-SELEX transfer benchmark (pilot): paper-style baselines vs transformer representation", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(out_png, bbox_inches="tight")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/raw/htselex")
    parser.add_argument("--limit_files", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=5000)
    parser.add_argument("--out_prefix", default="plots/htselex_option2_pilot")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")
    ckpt_path = cfg_ft["pretrained_ckpt"]
    model = build_pretrained_model(cfg_pre, ckpt_path)

    files = sorted(Path(args.folder).glob("*.txt"))
    files = files[:args.limit_files]

    print(f"Running {len(files)} HT-SELEX files")
    print(f"outer_folds={args.outer_folds}, seed={args.seed}, max_rows={args.max_rows}")
    print()

    rows = []

    for i, path in enumerate(files, 1):
        try:
            row = run_one_file(
                path,
                cfg_pre=cfg_pre,
                model=model,
                outer_folds=args.outer_folds,
                seed=args.seed,
                max_rows=args.max_rows,
            )
            rows.append(row)
            print(
                f"[{i}/{len(files)}] {row['file']} | "
                f"L={row['seq_len']} n={row['n_rows_used']} | "
                f"1mer={row['cv_r2_1mer']:.3f} | "
                f"2mer={row['cv_r2_2mer']:.3f} | "
                f"3mer={row['cv_r2_3mer']:.3f} | "
                f"trf={row['cv_r2_transformer_hiddenflex']:.3f}"
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

    out_png = args.out_prefix + "_panelA.png"
    make_panelA(rows, out_png)

    print()
    print("Saved:", out_csv)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
