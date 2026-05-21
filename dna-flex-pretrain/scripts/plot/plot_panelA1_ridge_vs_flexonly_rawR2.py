import os
import csv
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.pbm_dataset import PBMDataset, split_dataset
from src.model import TinyMultiTaskModelOneHot
from src.utils import load_yaml


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


def onehot_1mer_positional(seq: str):
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    L = len(seq)
    X = np.zeros((L, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        j = base_to_idx.get(b, None)
        if j is not None:
            X[i, j] = 1.0
    return X.reshape(-1)  # [L*4]


def ridge_eval(ds, idx_train, idx_val, idx_test, alphas):
    def build_Xy(idxs):
        X_list, y_list = [], []
        for i in idxs:
            item = ds[i]
            X_list.append(onehot_1mer_positional(item["seq"]))
            y_list.append(float(item["y"].item()))
        return np.stack(X_list, 0), np.array(y_list, dtype=np.float32)

    Xtr, ytr = build_Xy(idx_train)
    Xva, yva = build_Xy(idx_val)
    Xte, yte = build_Xy(idx_test)

    best_alpha, best_val_r2 = None, -1e9
    for a in alphas:
        reg = Ridge(alpha=a, fit_intercept=True, random_state=0)
        reg.fit(Xtr, ytr)
        pva = reg.predict(Xva)
        r2 = r2_score(yva, pva)
        if r2 > best_val_r2:
            best_val_r2 = r2
            best_alpha = a

    Xtv = np.concatenate([Xtr, Xva], 0)
    ytv = np.concatenate([ytr, yva], 0)
    reg = Ridge(alpha=best_alpha, fit_intercept=True, random_state=0)
    reg.fit(Xtv, ytv)
    pte = reg.predict(Xte)

    return {
        "alpha": float(best_alpha),
        "test_r2": float(r2_score(yte, pte)),
        "test_rmse": float(np.sqrt(np.mean((yte - pte) ** 2))),
        "test_pearson": float(pearsonr_np(yte, pte)),
    }


def masked_max_pool(flex_pred: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
    mask = am.unsqueeze(-1).bool()
    neg_inf = torch.tensor(-1e9, device=flex_pred.device, dtype=flex_pred.dtype)
    flex_masked = torch.where(mask, flex_pred, neg_inf)
    return flex_masked.max(dim=1).values


@torch.no_grad()
def eval_flexonly(model, head, loader, device):
    model.eval()
    head.eval()
    ys, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device).squeeze(1)

        out = model(x, am)
        if isinstance(out, (tuple, list)):
            # expected: (mlm_logits, flex_pred) or (mlm_logits, flex_pred, h)
            flex_pred = out[1]
        else:
            raise RuntimeError("Unexpected model output format while evaluating flex-only head.")

        pooled = masked_max_pool(flex_pred, am)  # [B, 12]
        yhat = head(pooled).squeeze(1)

        ys.append(y.detach().cpu().numpy())
        preds.append(yhat.detach().cpu().numpy())

    y_all = np.concatenate(ys, 0)
    p_all = np.concatenate(preds, 0)

    return {
        "test_r2": float(r2_score(y_all, p_all)),
        "test_rmse": float(np.sqrt(np.mean((y_all - p_all) ** 2))),
        "test_pearson": float(pearsonr_np(y_all, p_all)),
    }


def train_flexonly(ds, idx_train, idx_val, idx_test, model, cfg_ft, seed):
    set_seed(seed)
    device = "cpu"

    bs = int(cfg_ft["train"]["batch_size"])
    train_loader = DataLoader(Subset(ds, idx_train), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(Subset(ds, idx_val),   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(Subset(ds, idx_test),  batch_size=bs, shuffle=False)

    # better head (same family as your best flex-only script)
    head = nn.Sequential(
        nn.Linear(12, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    opt = torch.optim.Adam(
        head.parameters(),
        lr=float(cfg_ft["train"]["lr"]),
        weight_decay=1e-4
    )

    epochs = int(cfg_ft["train"]["epochs"])
    patience = int(cfg_ft["train"].get("patience", 5))

    best_val_r2 = -1e9
    best_state = None
    bad = 0

    @torch.no_grad()
    def eval_val_r2(loader):
        model.eval()
        head.eval()
        ys, preds = [], []
        for batch in loader:
            x = batch["x"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["y"].to(device).squeeze(1)

            out = model(x, am)
            flex_pred = out[1]
            pooled = masked_max_pool(flex_pred, am)
            yhat = head(pooled).squeeze(1)

            ys.append(y.detach().cpu().numpy())
            preds.append(yhat.detach().cpu().numpy())

        y_all = np.concatenate(ys, 0)
        p_all = np.concatenate(preds, 0)
        return float(r2_score(y_all, p_all))

    for ep in range(1, epochs + 1):
        head.train()
        model.eval()  # encoder frozen

        for batch in train_loader:
            x = batch["x"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["y"].to(device)

            with torch.no_grad():
                out = model(x, am)
                flex_pred = out[1]
                pooled = masked_max_pool(flex_pred, am)

            yhat = head(pooled)
            loss = F.mse_loss(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_r2 = eval_val_r2(val_loader)

        if val_r2 > best_val_r2 + 1e-6:
            best_val_r2 = val_r2
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    return eval_flexonly(model, head, test_loader, device)


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
        n_flex=n_flex
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model


def main():
    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")
    ckpt_path = cfg_ft["pretrained_ckpt"]

    datasets = ["Max", "Mad", "Myc"]
    seeds = [0, 1, 2]
    alpha_grid = [0.1, 1.0, 10.0, 100.0]

    rows = []

    for ds_name in datasets:
        pbm_path = f"data/raw/pbm/{ds_name}.txt"

        for seed in seeds:
            set_seed(seed)
            ds = PBMDataset(pbm_path, k=int(cfg_pre["tokenizer"]["k"]), seed=seed)
            idx_train, idx_val, idx_test = split_dataset(
                ds,
                train_frac=float(cfg_ft["train"]["split"]["train"]),
                val_frac=float(cfg_ft["train"]["split"]["val"]),
            )

            # 1-mer ridge
            ridge_res = ridge_eval(ds, idx_train, idx_val, idx_test, alpha_grid)

            # flex-only
            model = build_pretrained_model(cfg_pre, ckpt_path)
            flex_res = train_flexonly(ds, idx_train, idx_val, idx_test, model, cfg_ft, seed)

            row = {
                "dataset": ds_name,
                "seed": seed,
                "ridge_test_r2": ridge_res["test_r2"],
                "ridge_test_rmse": ridge_res["test_rmse"],
                "ridge_test_pearson": ridge_res["test_pearson"],
                "flex_test_r2": flex_res["test_r2"],
                "flex_test_rmse": flex_res["test_rmse"],
                "flex_test_pearson": flex_res["test_pearson"],
            }
            rows.append(row)

            print(
                f"{ds_name} | seed={seed} | "
                f"ridge_r2={ridge_res['test_r2']:.3f} | "
                f"flexonly_r2={flex_res['test_r2']:.3f}"
            )

    # save CSV
    out_csv = "plots/panelA1_ridge_vs_flexonly_rawR2.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # plot
    colors = {"Max": "#1f77b4", "Mad": "#ff7f0e", "Myc": "#2ca02c"}
    markers = {0: "o", 1: "s", 2: "^"}

    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=200)

    xs = [r["ridge_test_r2"] for r in rows]
    ys = [r["flex_test_r2"] for r in rows]

    lo = min(xs + ys) - 0.05
    hi = max(xs + ys) + 0.05
    lo = max(lo, -0.1)
    hi = min(hi, 1.05)

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray")

    for r in rows:
        ax.scatter(
            r["ridge_test_r2"],
            r["flex_test_r2"],
            color=colors[r["dataset"]],
            marker=markers[r["seed"]],
            s=70,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
        )
        ax.text(
            r["ridge_test_r2"] + 0.005,
            r["flex_test_r2"] + 0.005,
            f"{r['seed']}",
            fontsize=7
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Test $R^2$ (1-mer ridge)")
    ax.set_ylabel("Test $R^2$ (flex-only: flex_pred + maxpool + better head)")
    ax.set_title("Panel A1: 1-mer ridge vs flex-only (raw test $R^2$)")
    ax.grid(True, alpha=0.25)

    # legend: dataset colors
    dataset_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[d], markeredgecolor='black',
               markersize=8, label=d) for d in datasets
    ]
    # legend: seed markers
    seed_handles = [
        Line2D([0], [0], marker=markers[s], color='black', linestyle='None',
               markersize=8, label=f"seed {s}") for s in seeds
    ]

    leg1 = ax.legend(handles=dataset_handles, title="dataset", loc="lower right")
    ax.add_artist(leg1)
    ax.legend(handles=seed_handles, title="seed", loc="upper left")

    out_png = "plots/panelA1_ridge_vs_flexonly_rawR2.png"
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")

    print("Saved:", out_png)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
