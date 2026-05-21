import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from src.utils import load_yaml
from src.model import TinyMultiTaskModelOneHot
from src.pbm_dataset import PBMDataset, split_dataset
from matplotlib.ticker import NullLocator


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
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


def metrics_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pearson = pearsonr_np(y_true, y_pred)
    return float(r2), pearson, rmse


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# ------------------------------------------------------------
# Sequence baseline features
# ------------------------------------------------------------
BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}


def get_seq_score(ds, idx):
    """
    PBMDataset in your repo stores (seq, score) in ds.items in existing scripts.
    Fallback tries __getitem__ if needed.
    """
    if hasattr(ds, "items"):
        seq, score = ds.items[idx]
        return seq, float(score)

    item = ds[idx]
    seq = item.get("seq", None)
    if seq is None:
        raise RuntimeError("Could not recover raw sequence from PBMDataset")
    y = item["y"]
    score = float(y.item()) if torch.is_tensor(y) else float(y)
    return seq, score


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


def load_lookup_yaml_ordered(lookup_yaml, pretrain_yaml):
    import yaml

    data = yaml.safe_load(Path(lookup_yaml).read_text())
    cfg = yaml.safe_load(Path(pretrain_yaml).read_text())

    di_names = list(cfg["features"]["dinucleotide"])
    tri_names = list(cfg["features"]["trinucleotide"])

    di_maps = {
        name: {k.upper(): float(v) for k, v in data["dinucleotide"][name].items()}
        for name in di_names
    }
    tri_maps = {
        name: {k.upper(): float(v) for k, v in data["trinucleotide"][name].items()}
        for name in tri_names
    }
    return di_names, di_maps, tri_names, tri_maps


def flex12_track_sequence(seq, di_names, di_maps, tri_names, tri_maps):
    """
    Sequence-level 12-flex baseline:
      - 4 tri features over all overlapping 3-mers -> 4*(L-2)
      - 8 di features over all overlapping 2-mers -> 8*(L-1)
    Concatenated in fixed order.
    """
    seq = seq.upper()
    L = len(seq)
    vals = []

    # tri first? For the custom baseline we just need consistency; keep explicit order.
    for feat in tri_names:
        fmap = tri_maps[feat]
        for i in range(L - 2):
            vals.append(float(fmap.get(seq[i:i+3], 0.0)))

    for feat in di_names:
        fmap = di_maps[feat]
        for i in range(L - 1):
            vals.append(float(fmap.get(seq[i:i+2], 0.0)))

    return np.asarray(vals, dtype=np.float32)


def build_baseline_matrix(ds, indices, kind, flex_tables=None):
    X_list = []
    y_list = []

    if kind == "1mer12flex":
        assert flex_tables is not None
        di_names, di_maps, tri_names, tri_maps = flex_tables

    for idx in indices:
        seq, score = get_seq_score(ds, idx)

        if kind == "1mer":
            feat = positional_nmer_onehot(seq, 1)
        elif kind == "2mer":
            feat = positional_nmer_onehot(seq, 2)
        elif kind == "3mer":
            feat = positional_nmer_onehot(seq, 3)
        elif kind == "1mer12flex":
            one = positional_nmer_onehot(seq, 1)
            flex = flex12_track_sequence(seq, di_names, di_maps, tri_names, tri_maps)
            feat = np.concatenate([one, flex], axis=0)
        else:
            raise ValueError(f"Unknown baseline kind: {kind}")

        X_list.append(feat)
        y_list.append(score)

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


# ------------------------------------------------------------
# Standardization helpers
# ------------------------------------------------------------
def standardize_all(Xtr, Xva, Xte):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


def standardize_flex_part_only(Xtr, Xva, Xte, n_seq_dims):
    Xtr = Xtr.copy()
    Xva = Xva.copy()
    Xte = Xte.copy()

    if Xtr.shape[1] <= n_seq_dims:
        return Xtr, Xva, Xte

    Ftr = Xtr[:, n_seq_dims:]
    Fva = Xva[:, n_seq_dims:]
    Fte = Xte[:, n_seq_dims:]

    mu = Ftr.mean(axis=0, keepdims=True)
    sd = Ftr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0

    Xtr[:, n_seq_dims:] = (Ftr - mu) / sd
    Xva[:, n_seq_dims:] = (Fva - mu) / sd
    Xte[:, n_seq_dims:] = (Fte - mu) / sd
    return Xtr, Xva, Xte


# ------------------------------------------------------------
# Ridge fitting with validation-based alpha choice
# IMPORTANT: do NOT refit on train+val, because panel C x-axis
# is "percentage of training data used". Keep train size controlled.
# ------------------------------------------------------------
def ridge_with_val_selection(Xtr, ytr, Xva, yva, Xte, yte, standardize_mode="none", n_seq_dims=None):
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    if standardize_mode == "all":
        Xtr_s, Xva_s, Xte_s = standardize_all(Xtr, Xva, Xte)
    elif standardize_mode == "flex_only":
        Xtr_s, Xva_s, Xte_s = standardize_flex_part_only(Xtr, Xva, Xte, n_seq_dims=n_seq_dims)
    else:
        Xtr_s, Xva_s, Xte_s = Xtr, Xva, Xte

    best_alpha = None
    best_val_r2 = -1e9
    best_model = None

    for a in alphas:
        reg = Ridge(alpha=a, fit_intercept=True, random_state=0)
        reg.fit(Xtr_s, ytr)
        pred_val = reg.predict(Xva_s)
        val_r2 = r2_score(yva, pred_val)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = a
            best_model = reg

    pred_test = best_model.predict(Xte_s)
    test_r2, test_pearson, test_rmse = metrics_from_arrays(yte, pred_test)

    return {
        "alpha": float(best_alpha),
        "val_r2": float(best_val_r2),
        "test_r2": float(test_r2),
        "test_pearson": float(test_pearson),
        "test_rmse": float(test_rmse),
    }


# ------------------------------------------------------------
# Transformer feature extraction
# ------------------------------------------------------------
def build_model(cfg_pre, ckpt_path):
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
    model.eval()
    return model


@torch.no_grad()
def extract_all_transformer_features(model, ds, device):
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    H_list = []
    HF_list = []
    y_list = []

    model = model.to(device)
    model.eval()

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device).squeeze(1)

        _, flex_pred, h = model(x, am, return_hidden=True)

        mask_f = am.unsqueeze(-1).float()
        h = h * mask_f
        flex_pred = flex_pred * mask_f

        feat_h = h.reshape(h.shape[0], -1)
        feat_hf = torch.cat([h, flex_pred], dim=-1).reshape(h.shape[0], -1)

        H_list.append(feat_h.cpu())
        HF_list.append(feat_hf.cpu())
        y_list.append(y.cpu())

    Xh = torch.cat(H_list, dim=0).numpy()
    Xhf = torch.cat(HF_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()
    return Xh, Xhf, y


# ------------------------------------------------------------
# Benchmark one dataset/seed over all percentages
# ------------------------------------------------------------
def benchmark_dataset_seed(
    ds_name,
    ds_path,
    seed,
    percentages,
    cfg_pre,
    cfg_ft,
    flex_tables,
    ckpt_paths,
    device,
    min_train_examples=16,
):
    k = int(cfg_pre["tokenizer"]["k"])

    ds = PBMDataset(ds_path, k=k, seed=seed)

    train_frac = float(cfg_ft["train"]["split"]["train"])
    val_frac = float(cfg_ft["train"]["split"]["val"])

    idx_train_full, idx_val, idx_test = split_dataset(
        ds,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    rng = np.random.RandomState(seed)
    train_perm = np.array(idx_train_full)[rng.permutation(len(idx_train_full))]

    # Baseline features (computed once over whole dataset; slice later)
    all_indices = list(range(len(ds)))
    X1_all, y_all = build_baseline_matrix(ds, all_indices, "1mer")
    X1f_all, _ = build_baseline_matrix(ds, all_indices, "1mer12flex", flex_tables=flex_tables)
    X2_all, _ = build_baseline_matrix(ds, all_indices, "2mer")
    X3_all, _ = build_baseline_matrix(ds, all_indices, "3mer")

    seq_len = len(get_seq_score(ds, 0)[0])
    n_seq_dims_1mer = 4 * seq_len

    # Transformer features for each checkpoint (computed once)
    feat_cache = {}
    for label, ckpt in ckpt_paths.items():
        model = build_model(cfg_pre, ckpt)
        Xh_all, Xhf_all, y_check = extract_all_transformer_features(model, ds, device=device)
        feat_cache[label] = {
            "hidden": Xh_all,
            "hiddenflex": Xhf_all,
            "y": y_check,
        }

    rows = []

    for pct in percentages:
        n_sub = max(min_train_examples, int(np.ceil(len(train_perm) * pct / 100.0)))
        n_sub = min(n_sub, len(train_perm))
        idx_train = train_perm[:n_sub].tolist()

        # common y splits
        ytr = y_all[idx_train]
        yva = y_all[idx_val]
        yte = y_all[idx_test]

        # 1-mer
        res = ridge_with_val_selection(
            X1_all[idx_train], ytr,
            X1_all[idx_val], yva,
            X1_all[idx_test], yte,
            standardize_mode="none"
        )
        rows.append({
            "dataset": ds_name, "seed": seed, "pct_train": pct, "n_train": len(idx_train),
            "model": "1-mer ridge", **res
        })

        # 1-mer + 12-flex
        res = ridge_with_val_selection(
            X1f_all[idx_train], ytr,
            X1f_all[idx_val], yva,
            X1f_all[idx_test], yte,
            standardize_mode="flex_only",
            n_seq_dims=n_seq_dims_1mer
        )
        rows.append({
            "dataset": ds_name, "seed": seed, "pct_train": pct, "n_train": len(idx_train),
            "model": "1-mer + 12-flex", **res
        })

        # 2-mer
        res = ridge_with_val_selection(
            X2_all[idx_train], ytr,
            X2_all[idx_val], yva,
            X2_all[idx_test], yte,
            standardize_mode="none"
        )
        rows.append({
            "dataset": ds_name, "seed": seed, "pct_train": pct, "n_train": len(idx_train),
            "model": "2-mer ridge", **res
        })

        # 3-mer
        res = ridge_with_val_selection(
            X3_all[idx_train], ytr,
            X3_all[idx_val], yva,
            X3_all[idx_test], yte,
            standardize_mode="none"
        )
        rows.append({
            "dataset": ds_name, "seed": seed, "pct_train": pct, "n_train": len(idx_train),
            "model": "3-mer ridge", **res
        })

        # original checkpoint hidden + ridge
        Xh = feat_cache["original"]["hidden"]
        yh = feat_cache["original"]["y"]
        res = ridge_with_val_selection(
            Xh[idx_train], yh[idx_train],
            Xh[idx_val], yh[idx_val],
            Xh[idx_test], yh[idx_test],
            standardize_mode="all"
        )
        rows.append({
            "dataset": ds_name, "seed": seed, "pct_train": pct, "n_train": len(idx_train),
            "model": "Transformer hidden + ridge", **res
        })

        # original checkpoint hidden+flex + ridge
        Xhf = feat_cache["original"]["hiddenflex"]
        yhf = feat_cache["original"]["y"]
        res = ridge_with_val_selection(
            Xhf[idx_train], yhf[idx_train],
            Xhf[idx_val], yhf[idx_val],
            Xhf[idx_test], yhf[idx_test],
            standardize_mode="all"
        )
        rows.append({
            "dataset": ds_name, "seed": seed, "pct_train": pct, "n_train": len(idx_train),
            "model": "Transformer hidden + flex + ridge", **res
        })

        # checkpoint methods: all use hidden+flex + ridge
        for label in ["bend only", "bend+flex", "flex+MLM", "bend+flex+MLM"]:
            X = feat_cache[label]["hiddenflex"]
            y = feat_cache[label]["y"]
            res = ridge_with_val_selection(
                X[idx_train], y[idx_train],
                X[idx_val], y[idx_val],
                X[idx_test], y[idx_test],
                standardize_mode="all"
            )
            rows.append({
                "dataset": ds_name, "seed": seed, "pct_train": pct, "n_train": len(idx_train),
                "model": label, **res
            })

        print(f"[{ds_name}] seed={seed} pct={pct:>5g}% n_train={len(idx_train)} done")

    return rows


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def make_panel_c_plot(rows, out_png):
    import pandas as pd

    df = pd.DataFrame(rows)

    order = [
        "1-mer ridge",
        "1-mer + 12-flex",
        "2-mer ridge",
        "3-mer ridge",
        "Transformer hidden + ridge",
        "Transformer hidden + flex + ridge",
        "bend only",
        "bend+flex",
        "flex+MLM",
        "bend+flex+MLM",
    ]

    display_names = {
        "1-mer ridge": "1-mer",
        "1-mer + 12-flex": "1-mer + 12-flex",
        "2-mer ridge": "2-mer",
        "3-mer ridge": "3-mer",
        "Transformer hidden + ridge": "pre-trained hidden",
        "Transformer hidden + flex + ridge": "pre-trained hidden+flex",
        "bend only": "bend only",
        "bend+flex": "bend+flex",
        "flex+MLM": "flex+MLM",
        "bend+flex+MLM": "bend+flex+MLM",
    }

    agg = (
        df.groupby(["model", "pct_train"])["test_r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    colors = {
        "1-mer ridge": "#4c4c4c",
        "1-mer + 12-flex": "#d99a00",
        "2-mer ridge": "#e41a1c",
        "3-mer ridge": "#ff7f00",
        "Transformer hidden + ridge": "#1565c0",
        "Transformer hidden + flex + ridge": "#00a7a0",
        "bend only": "#6d3b1f",
        "bend+flex": "#2ca02c",
        "flex+MLM": "#6a1b9a",
        "bend+flex+MLM": "#e91e63",
    }

    markers = {
        "1-mer ridge": "o",
        "1-mer + 12-flex": "p",
        "2-mer ridge": "*",
        "3-mer ridge": "^",
        "Transformer hidden + ridge": "<",
        "Transformer hidden + flex + ridge": "s",
        "bend only": "x",
        "bend+flex": "D",
        "flex+MLM": "o",
        "bend+flex+MLM": "v",
    }

    plt.figure(figsize=(12.5, 7.5), dpi=220)
    ax = plt.gca()

    all_y = []

    for model in order:
        sub = agg[agg["model"] == model].sort_values("pct_train")
        x = sub["pct_train"].values.astype(float)
        y = sub["mean"].values.astype(float)
        e = sub["std"].fillna(0.0).values.astype(float)

        all_y.extend(y.tolist())
        all_y.extend((y - e).tolist())
        all_y.extend((y + e).tolist())

        ax.errorbar(
            x,
            y,
            yerr=e,
            marker=markers[model],
            linewidth=1.6,
            markersize=5.5,
            elinewidth=0.9,
            capsize=2,
            color=colors[model],
            label=display_names[model],
        )

    # log x-axis
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10, 30, 100])
    ax.set_xticklabels(["0.3", "1", "3", "10", "30", "100"])

    # remove minor tick/grid clutter on log axis
    ax.xaxis.set_minor_locator(NullLocator())

    # labels and title
    ax.set_xlabel("Training data used (%)", fontsize=13)
    ax.set_ylabel("Test $R^2$", fontsize=13)
    ax.set_title("gcPBM sample-efficiency", fontsize=16)

    # cleaner y-range
    y_min = max(-0.05, min(all_y) - 0.02)
    y_max = min(0.99, max(all_y) + 0.02)
    ax.set_ylim(y_min, y_max)

    # major grid only
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.grid(False, which="minor", axis="both")

    # legend outside right, shorter labels
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print("Saved plot ->", out_png)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv", type=str, default="plots/panelC_gcpbm_sample_efficiency.csv")
    parser.add_argument("--out_png", type=str, default="plots/panelC_gcpbm_sample_efficiency.png")
    parser.add_argument("--pcts", type=float, nargs="+", default=[0.3, 1, 3, 10, 30, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min_train_examples", type=int, default=16)
    parser.add_argument("--lookup_yaml", type=str, default="data/raw/flex_tables/lookup.yaml")
    parser.add_argument("--pretrain_yaml", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--finetune_yaml", type=str, default="configs/finetune_pbm.yaml")
    args = parser.parse_args()

    set_seed(0)
    device = pick_device()
    print("device =", device)

    cfg_pre = load_yaml(args.pretrain_yaml)
    cfg_ft = load_yaml(args.finetune_yaml)

    flex_tables = load_lookup_yaml_ordered(args.lookup_yaml, args.pretrain_yaml)

    # Checkpoints
    ckpt_paths = {
        "original": "checkpoints/hg38_256_chr1-22_200k_di8_tri4_best_by_val_flex.pt",
        "bend only": "checkpoints/bendability_stage1_data1_bendonly.pt",
        "bend+flex": "checkpoints/bendability_stage1_data1_flex0p2.pt",
        "flex+MLM": "checkpoints/bendstage_flexmlm_data1.pt",
        "bend+flex+MLM": "checkpoints/bendstage_bendflexmlm_data1.pt",
    }

    datasets = {
        "Max": "data/raw/pbm/Max.txt",
        "Mad": "data/raw/pbm/Mad.txt",
        "Myc": "data/raw/pbm/Myc.txt",
    }

    all_rows = []

    for ds_name, ds_path in datasets.items():
        for seed in args.seeds:
            rows = benchmark_dataset_seed(
                ds_name=ds_name,
                ds_path=ds_path,
                seed=seed,
                percentages=args.pcts,
                cfg_pre=cfg_pre,
                cfg_ft=cfg_ft,
                flex_tables=flex_tables,
                ckpt_paths=ckpt_paths,
                device=device,
                min_train_examples=args.min_train_examples,
            )
            all_rows.extend(rows)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print("Saved benchmark CSV ->", args.out_csv)

    make_panel_c_plot(all_rows, args.out_png)


if __name__ == "__main__":
    main()