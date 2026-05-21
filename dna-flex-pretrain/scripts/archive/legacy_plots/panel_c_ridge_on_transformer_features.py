#!/usr/bin/env python3
"""
Panel C (paper-style): sample size vs R^2 (mean ± std across seeds)

We compare:
1) Ridge regression on positional 1-mer features (baseline)
2) Ridge regression on *frozen pretrained transformer* hidden features
   - Extract token hidden states h: [L, d_model]
   - Flatten across positions -> feature vector [L*d_model]
   - Fit ridge on those features (no end-to-end finetuning)

This matches your "hidden-flatten-linear head" idea, but uses Ridge (stable & comparable).
"""
import os
import argparse
import random
from typing import List, Tuple, Dict

import numpy as np
import torch

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

from src.pbm_dataset import PBMDataset, split_dataset
from src.model import TinyMultiTaskModelOneHot


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pearsonr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def load_pretrained_model(ckpt_path: str, device: str = "cpu") -> TinyMultiTaskModelOneHot:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Try common keys first
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError("Checkpoint format not recognized (expected dict).")

    # Infer d_model / n_flex from weights (robust)
    w_in = state.get("encoder.input_proj.weight", None)
    if w_in is None:
        w_in = state.get("input_proj.weight", None)
    d_model = 64 if w_in is None else int(w_in.shape[0])

    w_flex = state.get("flex_head.weight", None)
    n_flex = 12 if w_flex is None else int(w_flex.shape[0])

    model = TinyMultiTaskModelOneHot(
        input_dim=24,
        vocab_size=4100,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        max_len=512,
        n_flex=n_flex,
    )

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def make_positional_1mer_features(seq: str) -> np.ndarray:
    """Positional 1-mer features: dim = len(seq) * 4."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    S = len(seq)
    X = np.zeros((S * 4,), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = mapping.get(ch, None)
        if j is not None:
            X[i * 4 + j] = 1.0
    return X


@torch.no_grad()
def extract_transformer_flatten_features(
    model: TinyMultiTaskModelOneHot,
    ds: PBMDataset,
    indices: List[int],
    device: str = "cpu",
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: [N, L*d_model]
      y: [N]
    """
    feats = []
    ys = []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        x = torch.stack([ds[i]["x"] for i in batch_idx], dim=0).to(device)                # [B,L,24]
        m = torch.stack([ds[i]["attention_mask"] for i in batch_idx], dim=0).to(device)   # [B,L]
        y = torch.stack([ds[i]["y"] for i in batch_idx], dim=0).squeeze(-1).cpu().numpy() # [B]

        # IMPORTANT: request hidden states
        _, _, h = model(x, m, return_hidden=True)   # [B,L,d_model]
        h = h.reshape(h.shape[0], -1).cpu().numpy() # [B, L*d_model]

        feats.append(h)
        ys.append(y)

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    return X, y


def fit_ridge_choose_alpha(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    alphas: List[float],
) -> float:
    best_alpha = None
    best_r2 = -1e9
    for a in alphas:
        reg = Ridge(alpha=a, fit_intercept=True, random_state=0)
        reg.fit(Xtr, ytr)
        pred = reg.predict(Xva)
        r2 = r2_score(yva, pred)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = a
    return float(best_alpha)


def eval_ridge(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    alphas: List[float],
) -> Dict[str, float]:
    alpha = fit_ridge_choose_alpha(Xtr, ytr, Xva, yva, alphas)

    # Refit on train+val
    Xtv = np.concatenate([Xtr, Xva], axis=0)
    ytv = np.concatenate([ytr, yva], axis=0)
    reg = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
    reg.fit(Xtv, ytv)

    pred = reg.predict(Xte)
    return {
        "alpha": float(alpha),
        "test_r2": float(r2_score(yte, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "test_pearson": float(pearsonr_np(yte, pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbm_path", default="data/raw/pbm/Max.txt")
    ap.add_argument("--ckpt", default="checkpoints/hg38_256_chr1-22_200k_di8_tri4_best_by_val_flex.pt")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train_fracs", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.125, 0.0625])
    ap.add_argument("--out", default="figures/panel_c_ridge_on_transformer_features.png")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    alpha_grid = [0.1, 1.0, 10.0, 100.0, 1000.0]

    ridge1_results = []
    trf_results = []

    for seed in args.seeds:
        set_seed(seed)
        ds = PBMDataset(args.pbm_path, k=6, seed=seed)
        idx_tr, idx_va, idx_te = split_dataset(ds, train_frac=0.8, val_frac=0.1)

        # --- Baseline: positional 1-mer ridge
        Xtr_all = np.stack([make_positional_1mer_features(ds[i]["seq"]) for i in idx_tr], axis=0)
        ytr_all = np.asarray([float(ds[i]["y"].item()) for i in idx_tr], dtype=np.float32)
        Xva = np.stack([make_positional_1mer_features(ds[i]["seq"]) for i in idx_va], axis=0)
        yva = np.asarray([float(ds[i]["y"].item()) for i in idx_va], dtype=np.float32)
        Xte = np.stack([make_positional_1mer_features(ds[i]["seq"]) for i in idx_te], axis=0)
        yte = np.asarray([float(ds[i]["y"].item()) for i in idx_te], dtype=np.float32)

        # --- Transformer features
        model = load_pretrained_model(args.ckpt, device=args.device)
        Xtrf_all, ytrf_all = extract_transformer_flatten_features(model, ds, idx_tr, device=args.device, batch_size=args.batch_size)
        Xvaf, yvaf = extract_transformer_flatten_features(model, ds, idx_va, device=args.device, batch_size=args.batch_size)
        Xtef, ytef = extract_transformer_flatten_features(model, ds, idx_te, device=args.device, batch_size=args.batch_size)

        n_train_total = len(idx_tr)
        for frac in args.train_fracs:
            n_use = max(50, int(round(n_train_total * frac)))
            n_use = min(n_use, n_train_total)

            # Deterministic subset (ds is already shuffled by seed)
            Xtr = Xtr_all[:n_use]
            ytr = ytr_all[:n_use]
            res1 = eval_ridge(Xtr, ytr, Xva, yva, Xte, yte, alpha_grid)
            res1.update({"seed": seed, "frac": frac, "n_train": n_use})
            ridge1_results.append(res1)

            Xtrf = Xtrf_all[:n_use]
            ytrf = ytrf_all[:n_use]
            res2 = eval_ridge(Xtrf, ytrf, Xvaf, yvaf, Xtef, ytef, alpha_grid)
            res2.update({"seed": seed, "frac": frac, "n_train": n_use})
            trf_results.append(res2)

            print(f"[seed={seed} frac={frac:.4f} n={n_use}] 1mer_r2={res1['test_r2']:.3f}  trf_r2={res2['test_r2']:.3f}")

    # Aggregate mean/std across seeds for each n_train
    def aggregate(results: List[Dict[str, float]]) -> Dict[int, Tuple[float, float]]:
        by_n = {}
        for r in results:
            by_n.setdefault(int(r["n_train"]), []).append(float(r["test_r2"]))
        out = {}
        for n, vals in sorted(by_n.items(), key=lambda kv: kv[0]):
            v = np.asarray(vals, dtype=np.float32)
            out[n] = (float(v.mean()), float(v.std(ddof=0)))
        return out

    agg1 = aggregate(ridge1_results)
    agg2 = aggregate(trf_results)

    xs = np.array(sorted(agg1.keys()), dtype=np.int32)
    m1 = np.array([agg1[int(x)][0] for x in xs], dtype=np.float32)
    s1 = np.array([agg1[int(x)][1] for x in xs], dtype=np.float32)
    m2 = np.array([agg2[int(x)][0] for x in xs], dtype=np.float32)
    s2 = np.array([agg2[int(x)][1] for x in xs], dtype=np.float32)

    plt.figure(figsize=(10, 6))
    plt.errorbar(xs, m1, yerr=s1, fmt='-o', capsize=4, label="1-mer ridge")
    plt.errorbar(xs, m2, yerr=s2, fmt='-o', capsize=4, label="Transformer (frozen) + ridge on hidden-flatten features")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Sample size (train sequences)")
    plt.ylabel("Test R²")
    plt.title("Panel C-style: Sample size vs R² (mean ± std across seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nSaved plot -> {args.out}")


if __name__ == "__main__":
    main()
