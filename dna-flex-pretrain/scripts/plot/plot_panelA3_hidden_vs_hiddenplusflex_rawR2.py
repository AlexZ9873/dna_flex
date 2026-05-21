import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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


@torch.no_grad()
def eval_hiddenonly(model, head, loader, device):
    model.eval()
    head.eval()
    ys, preds = [], []

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device).squeeze(1)

        _, _, h = model(x, am, return_hidden=True)   # [B,T,d_model]
        feat = h.reshape(h.shape[0], -1)             # flatten positions
        yhat = head(feat).squeeze(1)

        ys.append(y.detach().cpu().numpy())
        preds.append(yhat.detach().cpu().numpy())

    y_all = np.concatenate(ys, 0)
    p_all = np.concatenate(preds, 0)

    return {
        "test_r2": float(r2_score(y_all, p_all)),
        "test_rmse": float(np.sqrt(np.mean((y_all - p_all) ** 2))),
        "test_pearson": float(pearsonr_np(y_all, p_all)),
    }


@torch.no_grad()
def eval_hiddenplusflex(model, head, loader, device):
    model.eval()
    head.eval()
    ys, preds = [], []

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device).squeeze(1)

        _, flex_pred, h = model(x, am, return_hidden=True)   # h:[B,T,64], flex:[B,T,12]
        mask_f = am.unsqueeze(-1).float()
        h = h * mask_f
        flex_pred = flex_pred * mask_f
        z = torch.cat([h, flex_pred], dim=-1)               # [B,T,76]
        feat = z.reshape(z.shape[0], -1)                    # flatten positions
        yhat = head(feat).squeeze(1)

        ys.append(y.detach().cpu().numpy())
        preds.append(yhat.detach().cpu().numpy())

    y_all = np.concatenate(ys, 0)
    p_all = np.concatenate(preds, 0)

    return {
        "test_r2": float(r2_score(y_all, p_all)),
        "test_rmse": float(np.sqrt(np.mean((y_all - p_all) ** 2))),
        "test_pearson": float(pearsonr_np(y_all, p_all)),
    }


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


def train_hiddenonly(ds, idx_train, idx_val, idx_test, model, cfg_pre, cfg_ft, seed):
    set_seed(seed)
    device = "cpu"

    bs = int(cfg_ft["train"]["batch_size"])
    train_loader = DataLoader(Subset(ds, idx_train), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(Subset(ds, idx_val),   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(Subset(ds, idx_test),  batch_size=bs, shuffle=False)

    T = ds[0]["x"].shape[0]
    d_model = int(cfg_pre["model"]["d_model"])
    head = nn.Linear(T * d_model, 1).to(device)

    opt = torch.optim.Adam(
        head.parameters(),
        lr=float(cfg_ft["train"]["lr"]),
        weight_decay=float(cfg_ft["train"].get("weight_decay", 0.0))
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

            _, _, h = model(x, am, return_hidden=True)
            feat = h.reshape(h.shape[0], -1)
            yhat = head(feat).squeeze(1)

            ys.append(y.detach().cpu().numpy())
            preds.append(yhat.detach().cpu().numpy())

        y_all = np.concatenate(ys, 0)
        p_all = np.concatenate(preds, 0)
        return float(r2_score(y_all, p_all))

    for ep in range(1, epochs + 1):
        head.train()
        model.eval()
        for batch in train_loader:
            x = batch["x"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["y"].to(device)

            with torch.no_grad():
                _, _, h = model(x, am, return_hidden=True)
                feat = h.reshape(h.shape[0], -1)

            yhat = head(feat)
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

    return eval_hiddenonly(model, head, test_loader, device)


def train_hiddenplusflex(ds, idx_train, idx_val, idx_test, model, cfg_pre, cfg_ft, seed):
    set_seed(seed)
    device = "cpu"

    bs = int(cfg_ft["train"]["batch_size"])
    train_loader = DataLoader(Subset(ds, idx_train), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(Subset(ds, idx_val),   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(Subset(ds, idx_test),  batch_size=bs, shuffle=False)

    T = ds[0]["x"].shape[0]
    d_model = int(cfg_pre["model"]["d_model"])
    n_flex = len(cfg_pre["features"]["dinucleotide"]) + len(cfg_pre["features"]["trinucleotide"])

    head = nn.Linear(T * (d_model + n_flex), 1).to(device)

    opt = torch.optim.Adam(
        head.parameters(),
        lr=float(cfg_ft["train"]["lr"]),
        weight_decay=float(cfg_ft["train"].get("weight_decay", 0.0))
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

            _, flex_pred, h = model(x, am, return_hidden=True)
            mask_f = am.unsqueeze(-1).float()
            h = h * mask_f
            flex_pred = flex_pred * mask_f
            z = torch.cat([h, flex_pred], dim=-1)
            feat = z.reshape(z.shape[0], -1)
            yhat = head(feat).squeeze(1)

            ys.append(y.detach().cpu().numpy())
            preds.append(yhat.detach().cpu().numpy())

        y_all = np.concatenate(ys, 0)
        p_all = np.concatenate(preds, 0)
        return float(r2_score(y_all, p_all))

    for ep in range(1, epochs + 1):
        head.train()
        model.eval()
        for batch in train_loader:
            x = batch["x"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["y"].to(device)

            with torch.no_grad():
                _, flex_pred, h = model(x, am, return_hidden=True)
                mask_f = am.unsqueeze(-1).float()
                h = h * mask_f
                flex_pred = flex_pred * mask_f
                z = torch.cat([h, flex_pred], dim=-1)
                feat = z.reshape(z.shape[0], -1)

            yhat = head(feat)
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

    return eval_hiddenplusflex(model, head, test_loader, device)


def main():
    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")
    ckpt_path = cfg_ft["pretrained_ckpt"]

    datasets = ["Max", "Mad", "Myc"]
    seeds = [0, 1, 2]

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

            model_hidden = build_pretrained_model(cfg_pre, ckpt_path)
            hidden_res = train_hiddenonly(ds, idx_train, idx_val, idx_test, model_hidden, cfg_pre, cfg_ft, seed)

            model_hf = build_pretrained_model(cfg_pre, ckpt_path)
            hiddenflex_res = train_hiddenplusflex(ds, idx_train, idx_val, idx_test, model_hf, cfg_pre, cfg_ft, seed)

            row = {
                "dataset": ds_name,
                "seed": seed,
                "hidden_test_r2": hidden_res["test_r2"],
                "hidden_test_rmse": hidden_res["test_rmse"],
                "hidden_test_pearson": hidden_res["test_pearson"],
                "hiddenflex_test_r2": hiddenflex_res["test_r2"],
                "hiddenflex_test_rmse": hiddenflex_res["test_rmse"],
                "hiddenflex_test_pearson": hiddenflex_res["test_pearson"],
            }
            rows.append(row)

            print(
                f"{ds_name} | seed={seed} | "
                f"hidden_r2={hidden_res['test_r2']:.3f} | "
                f"hiddenplusflex_r2={hiddenflex_res['test_r2']:.3f}"
            )

    out_csv = "plots/panelA3_hidden_vs_hiddenplusflex_rawR2.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    colors = {"Max": "#1f77b4", "Mad": "#ff7f0e", "Myc": "#2ca02c"}
    markers = {0: "o", 1: "s", 2: "^"}

    fig, ax = plt.subplots(figsize=(6.6, 6.2), dpi=220)

    xs = [r["hidden_test_r2"] for r in rows]
    ys = [r["hiddenflex_test_r2"] for r in rows]

    lo = min(xs + ys) - 0.03
    hi = max(xs + ys) + 0.03
    lo = max(lo, 0.2)
    hi = min(hi, 1.0)

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="gray")

    for r in rows:
        ax.scatter(
            r["hidden_test_r2"],
            r["hiddenflex_test_r2"],
            color=colors[r["dataset"]],
            marker=markers[r["seed"]],
            s=90,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Test $R^2$ (hidden-only)", fontsize=13)
    ax.set_ylabel("Test $R^2$ (hidden + flex)", fontsize=13)
    ax.set_title("Panel A3: hidden-only vs hidden+flex (raw test $R^2$)", fontsize=14)
    ax.grid(True, alpha=0.25)

    dataset_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=colors[d], markeredgecolor='black',
               markersize=9, label=d)
        for d in ["Max", "Mad", "Myc"]
    ]
    seed_handles = [
        Line2D([0], [0], marker=markers[s], color='black',
               linestyle='None', markersize=9, label=f"seed {s}")
        for s in [0, 1, 2]
    ]

    leg1 = ax.legend(
        handles=dataset_handles,
        title="dataset",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=seed_handles,
        title="seed",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        borderaxespad=0.0,
        frameon=True
    )

    fig.subplots_adjust(right=0.74)

    out_png = "plots/panelA3_hidden_vs_hiddenplusflex_rawR2.png"
    fig.savefig(out_png, bbox_inches="tight")

    print("Saved:", out_png)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
