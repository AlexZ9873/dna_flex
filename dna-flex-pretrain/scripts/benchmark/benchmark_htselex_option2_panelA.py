import os
import csv
import math
import random
import inspect
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.model import TinyMultiTaskModelOneHot
from src.utils import load_yaml


# -------------------------
# Utilities
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


def zscore_train_apply(Xtr, Xte):
    return standardize_train_apply(Xtr, Xte)


# -------------------------
# HT-SELEX dataset
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

                if len(seq) < k:
                    continue

                self.items.append((seq, score))

        if len(self.items) == 0:
            raise ValueError(f"No usable rows in {path}")

        # Keep dominant length per file so feature dimensions are fixed.
        lengths = Counter(len(s) for s, _ in self.items)
        self.major_len = lengths.most_common(1)[0][0]
        self.items = [(s, y) for s, y in self.items if len(s) == self.major_len]

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
# Sequence feature encoders
# -------------------------
BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}
BASES = "ACGT"


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
    """
    Positional n-mer one-hot:
    n=1 -> 4L
    n=2 -> 16(L-1)
    n=3 -> 64(L-2)
    """
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


def build_feature_matrix(ds, indices, kind, paper_flex_builder=None):
    X_list = []
    y_list = []

    for i in indices:
        item = ds[i]
        seq = item["seq"]
        y = float(item["y"].item())

        if kind == "1mer":
            feat = positional_nmer_onehot(seq, 1)

        elif kind == "2mer":
            feat = positional_nmer_onehot(seq, 2)

        elif kind == "3mer":
            feat = positional_nmer_onehot(seq, 3)

        elif kind == "1mer_flex":
            if paper_flex_builder is None:
                raise ValueError("paper_flex_builder unavailable")
            one = positional_nmer_onehot(seq, 1)
            flex = paper_flex_builder(seq)
            feat = np.concatenate([one, flex], axis=0)

        else:
            raise ValueError(f"Unknown feature kind: {kind}")

        X_list.append(feat)
        y_list.append(y)

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.float32)


# -------------------------
# Paper-style flex lookup extraction
# -------------------------
def is_num(x):
    return isinstance(x, (int, float, np.integer, np.floating))


def valid_word_key(k, length):
    return isinstance(k, str) and len(k) == length and all(c in BASE_TO_ID for c in k.upper())


def collect_numeric_maps_from_dict(d, prefix=""):
    """
    Try to discover lookup maps from src.flex_features.
    Supports:
    - {"AA": 1.2, "AC": ...}
    - {"AA": {"twistDisp": 1.2, "stiffness": ...}, ...}
    - {"twistDisp": {"AA": 1.2, ...}, ...}
    """
    out = []

    if not isinstance(d, dict):
        return out

    keys = list(d.keys())

    # Case A: dict is direct word -> numeric map
    for L in (2, 3):
        word_keys = [k for k in keys if valid_word_key(k, L)]
        if len(word_keys) >= (10 if L == 2 else 40):
            vals = [d[k] for k in word_keys]
            if all(is_num(v) for v in vals):
                out.append((prefix, L, {k.upper(): float(d[k]) for k in word_keys}))
                return out

            if all(isinstance(v, dict) for v in vals):
                feature_names = set()
                for v in vals:
                    feature_names.update(v.keys())
                for feat in feature_names:
                    fmap = {}
                    ok = True
                    for k in word_keys:
                        val = d[k].get(feat, None)
                        if not is_num(val):
                            ok = False
                            break
                        fmap[k.upper()] = float(val)
                    if ok:
                        out.append((str(feat), L, fmap))
                return out

    # Case B: nested feature -> word -> numeric
    for k, v in d.items():
        if isinstance(v, dict):
            sub_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.extend(collect_numeric_maps_from_dict(v, sub_prefix))

    return out


def discover_flex_maps():
    """
    Attempts to find the five paper-style flexibility maps:
    tri: DNaseI, NPP
    di: twistDisp, trx, stiffness

    If exact maps are not found, 1-mer+flex will be skipped.
    """
    try:
        import src.flex_features as ff
    except Exception as e:
        print("Could not import src.flex_features:", e)
        return None

    all_maps = []

    for name, obj in vars(ff).items():
        if name.startswith("_"):
            continue
        if isinstance(obj, dict):
            all_maps.extend(collect_numeric_maps_from_dict(obj, prefix=name))

    # Deduplicate by name/length
    cleaned = []
    seen = set()
    for nm, L, fmap in all_maps:
        key = (nm.lower(), L, tuple(sorted(fmap.keys()))[:5])
        if key not in seen:
            seen.add(key)
            cleaned.append((nm, L, fmap))

    print()
    print("Discovered possible flex lookup maps:")
    for nm, L, fmap in cleaned[:50]:
        print(f"  {nm} | {L}-mer | {len(fmap)} keys")
    if len(cleaned) > 50:
        print("  ...")

    def find_map(length, aliases):
        for alias in aliases:
            alias_low = alias.lower()
            for nm, L, fmap in cleaned:
                if L == length and alias_low in nm.lower():
                    return nm, fmap
        return None, None

    # Paper features
    # tri: DNaseI, NPP
    # di: twist dispersion, trx, stiffness
    dnase_name, dnase = find_map(3, ["dnasei", "dnase", "bendabilitydnase"])
    npp_name, npp = find_map(3, ["npp", "nucleosome"])
    twist_name, twist = find_map(2, ["twistdisp", "twist_disp", "twist-dispersion", "twist"])
    trx_name, trx = find_map(2, ["trx", "twistroll", "twist_roll", "twist-roll"])
    stiff_name, stiff = find_map(2, ["stiffness", "stiff"])

    found = {
        "DNaseI": (dnase_name, dnase),
        "NPP": (npp_name, npp),
        "twistDisp": (twist_name, twist),
        "trx": (trx_name, trx),
        "stiffness": (stiff_name, stiff),
    }

    print()
    print("Paper-style flex feature match:")
    ok = True
    for feat, (nm, fmap) in found.items():
        if fmap is None:
            print(f"  MISSING: {feat}")
            ok = False
        else:
            print(f"  {feat}: using map '{nm}'")

    if not ok:
        print()
        print("WARNING: Exact 1-mer+flex baseline cannot be computed until missing lookup maps are added/found.")
        print("The script will still compute 1-mer, 2-mer, 3-mer, and transformer hidden+flex.")
        return None

    def paper_flex_builder(seq):
        seq = seq.upper()
        L = len(seq)

        tracks = []

        # Tri features: DNaseI, NPP over L-2 windows
        for _, fmap in [found["DNaseI"], found["NPP"]]:
            vals = []
            for i in range(L - 2):
                word = seq[i:i+3]
                vals.append(float(fmap.get(word, 0.0)))
            tracks.extend(vals)

        # Di features: twistDisp, trx, stiffness over L-1 windows
        for _, fmap in [found["twistDisp"], found["trx"], found["stiffness"]]:
            vals = []
            for i in range(L - 1):
                word = seq[i:i+2]
                vals.append(float(fmap.get(word, 0.0)))
            tracks.extend(vals)

        return np.asarray(tracks, dtype=np.float32)

    return paper_flex_builder


# -------------------------
# Transformer feature extraction
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
def extract_transformer_hiddenflex(model, ds, indices, batch_size=256, device="cpu"):
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

    return np.concatenate(feats, axis=0), np.concatenate(ys, axis=0)


# -------------------------
# Nested CV Ridge evaluation
# -------------------------
def choose_alpha_inner_cv(X, y, alphas, inner_folds=3, seed=0, standardize=False):
    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    mean_scores = []

    for alpha in alphas:
        scores = []
        for tr_idx, va_idx in kf.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            if standardize:
                Xtr, Xva = standardize_train_apply(Xtr, Xva)

            reg = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
            reg.fit(Xtr, ytr)
            pred = reg.predict(Xva)
            scores.append(r2_score(yva, pred))

        mean_scores.append(np.mean(scores))

    best_i = int(np.argmax(mean_scores))
    return float(alphas[best_i])


def nested_cv_oof_r2(X, y, alphas, outer_folds=5, inner_folds=3, seed=0, standardize=False):
    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=np.float32)

    chosen_alphas = []

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X)):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        alpha = choose_alpha_inner_cv(
            Xtr, ytr,
            alphas=alphas,
            inner_folds=inner_folds,
            seed=seed + fold_i,
            standardize=standardize,
        )
        chosen_alphas.append(alpha)

        if standardize:
            Xtr, Xte = standardize_train_apply(Xtr, Xte)

        reg = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
        reg.fit(Xtr, ytr)
        oof[te_idx] = reg.predict(Xte)

    return {
        "r2": float(r2_score(y, oof)),
        "pearson": float(pearsonr_np(y, oof)),
        "rmse": float(np.sqrt(np.mean((y - oof) ** 2))),
        "median_alpha": float(np.median(chosen_alphas)),
    }


# -------------------------
# Run one file
# -------------------------
def parse_family(filename):
    return filename.split("_")[0]


def short_tf_name(filename):
    parts = filename.replace(".txt", "").split("_")
    return parts[1] if len(parts) >= 2 else filename.replace(".txt", "")


def run_one_file(path, cfg_pre, model, paper_flex_builder, seed, outer_folds, inner_folds, max_rows=None):
    ds = HTSelexDataset(path, k=int(cfg_pre["tokenizer"]["k"]))

    if max_rows is not None and len(ds) > max_rows:
        rng = np.random.RandomState(seed)
        keep = rng.choice(np.arange(len(ds)), size=max_rows, replace=False).tolist()
        ds.items = [ds.items[i] for i in keep]

    n = len(ds)
    if n < max(outer_folds * 5, 100):
        raise ValueError(f"Too few rows: n={n}")

    idx_all = list(range(n))

    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    # 1-mer
    X1, y = build_feature_matrix(ds, idx_all, "1mer")
    res_1 = nested_cv_oof_r2(X1, y, alphas, outer_folds, inner_folds, seed, standardize=False)

    # 2-mer
    X2, y2 = build_feature_matrix(ds, idx_all, "2mer")
    res_2 = nested_cv_oof_r2(X2, y2, alphas, outer_folds, inner_folds, seed, standardize=False)

    # 3-mer
    X3, y3 = build_feature_matrix(ds, idx_all, "3mer")
    res_3 = nested_cv_oof_r2(X3, y3, alphas, outer_folds, inner_folds, seed, standardize=False)

    # 1-mer + paper-style flex, if available
    if paper_flex_builder is not None:
        X1f, y1f = build_feature_matrix(ds, idx_all, "1mer_flex", paper_flex_builder=paper_flex_builder)

        # Important: 1mer is binary; flex columns are continuous.
        # To approximate the paper's z-scoring of flexibility tracks, we standardize the whole
        # augmented matrix inside each fold. This is close enough for a pilot; we can refine later.
        res_1f = nested_cv_oof_r2(X1f, y1f, alphas, outer_folds, inner_folds, seed, standardize=True)
    else:
        res_1f = {"r2": float("nan"), "pearson": float("nan"), "rmse": float("nan"), "median_alpha": float("nan")}

    # Transformer hidden+flex
    Xtf, ytf = extract_transformer_hiddenflex(model, ds, idx_all)
    res_tf = nested_cv_oof_r2(Xtf, ytf, alphas, outer_folds, inner_folds, seed, standardize=True)

    return {
        "file": Path(path).name,
        "family": parse_family(Path(path).name),
        "tf": short_tf_name(Path(path).name),
        "n": n,
        "seq_len": ds.major_len,
        "seed": seed,
        "r2_1mer": res_1["r2"],
        "r2_1mer_flex": res_1f["r2"],
        "r2_2mer": res_2["r2"],
        "r2_3mer": res_3["r2"],
        "r2_transformer_hiddenflex": res_tf["r2"],
        "pearson_1mer": res_1["pearson"],
        "pearson_1mer_flex": res_1f["pearson"],
        "pearson_2mer": res_2["pearson"],
        "pearson_3mer": res_3["pearson"],
        "pearson_transformer_hiddenflex": res_tf["pearson"],
    }


# -------------------------
# Plotting
# -------------------------
def plot_panelA(rows, out_prefix):
    comparisons = [
        ("r2_1mer", "1-mer ridge"),
        ("r2_1mer_flex", "1-mer + flex ridge"),
        ("r2_2mer", "2-mer ridge"),
        ("r2_3mer", "3-mer ridge"),
    ]

    families = sorted(set(r["family"] for r in rows))
    cmap = plt.get_cmap("tab20")
    color_map = {fam: cmap(i % 20) for i, fam in enumerate(families)}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10), dpi=220)
    axes = axes.flatten()

    for ax, (xkey, xlabel) in zip(axes, comparisons):
        valid = [r for r in rows if not math.isnan(float(r.get(xkey, float("nan"))))]
        if len(valid) == 0:
            ax.set_title(f"No data for {xlabel}")
            ax.axis("off")
            continue

        xs = [float(r[xkey]) for r in valid]
        ys = [float(r["r2_transformer_hiddenflex"]) for r in valid]

        lo = min(xs + ys) - 0.05
        hi = max(xs + ys) + 0.05
        lo = max(lo, -0.2)
        hi = min(hi, 1.0)

        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1.2)

        for r in valid:
            ax.scatter(
                float(r[xkey]),
                float(r["r2_transformer_hiddenflex"]),
                color=color_map[r["family"]],
                s=38,
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

    # family legend outside
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color_map[fam], markeredgecolor="black",
               markersize=7, label=fam)
        for fam in families
    ]

    fig.legend(handles=handles, title="family", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle("HT-SELEX transfer benchmark: paper-style baselines vs transformer representation", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])

    out_png = out_prefix + "_panelA.png"
    fig.savefig(out_png, bbox_inches="tight")
    print("Saved:", out_png)


# -------------------------
# Main
# -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/raw/htselex")
    parser.add_argument("--limit_files", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=3)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--out_prefix", default="plots/htselex_option2_pilot20")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")
    ckpt_path = cfg_ft["pretrained_ckpt"]

    model = build_pretrained_model(cfg_pre, ckpt_path)
    paper_flex_builder = discover_flex_maps()

    files = sorted(Path(args.folder).glob("*.txt"))
    if args.limit_files is not None:
        files = files[:args.limit_files]

    print()
    print(f"Running {len(files)} HT-SELEX files")
    print(f"outer_folds={args.outer_folds}, inner_folds={args.inner_folds}")
    print(f"seed={args.seed}")
    print()

    rows = []

    for i, path in enumerate(files, 1):
        try:
            row = run_one_file(
                path,
                cfg_pre=cfg_pre,
                model=model,
                paper_flex_builder=paper_flex_builder,
                seed=args.seed,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                max_rows=args.max_rows,
            )
            rows.append(row)

            print(
                f"[{i}/{len(files)}] {path.name} | "
                f"L={row['seq_len']} n={row['n']} | "
                f"1mer={row['r2_1mer']:.3f} | "
                f"1mer+flex={row['r2_1mer_flex']:.3f} | "
                f"2mer={row['r2_2mer']:.3f} | "
                f"3mer={row['r2_3mer']:.3f} | "
                f"trf={row['r2_transformer_hiddenflex']:.3f}"
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

    print()
    print("Saved:", out_csv)

    plot_panelA(rows, args.out_prefix)


if __name__ == "__main__":
    main()
