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


def masked_max_pool(flex_pred: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
    """
    flex_pred: [B, T, 12]
    am:        [B, T]
    Pool across token positions T, separately for each flex feature.
    Output: [B, 12]
    """
    mask = am.unsqueeze(-1).bool()
    neg_inf = torch.tensor(-1e9, device=flex_pred.device, dtype=flex_pred.dtype)
    flex_masked = torch.where(mask, flex_pred, neg_inf)
    return flex_masked.max(dim=1).values


def build_model(cfg_pre, ckpt_path=None, seed=0):
    """
    If ckpt_path is provided: load pretrained encoder/flex head.
    If ckpt_path is None: use randomly initialized model.
    """
    set_seed(seed)

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

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model


@torch.no_grad()
def eval_flex_readout(model, head, loader, device):
    model.eval()
    head.eval()

    ys, preds = [], []

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device).squeeze(1)

        out = model(x, am)

        if not isinstance(out, (tuple, list)):
            raise RuntimeError("Unexpected model output. Expected tuple/list containing flex_pred.")

        # expected: (mlm_logits, flex_pred) or (mlm_logits, flex_pred, hidden)
        flex_pred = out[1]

        pooled = masked_max_pool(flex_pred, am)  # [B, 12]
        yhat = head(pooled).squeeze(1)

        ys.append(y.detach().cpu().numpy())
        preds.append(yhat.detach().cpu().numpy())

    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(preds, axis=0)

    return {
        "test_r2": float(r2_score(y_all, p_all)),
        "test_rmse": float(np.sqrt(np.mean((y_all - p_all) ** 2))),
        "test_pearson": float(pearsonr_np(y_all, p_all)),
    }


def train_flex_readout(ds, idx_train, idx_val, idx_test, model, cfg_ft, seed):
    """
    Train only the downstream MLP head on max-pooled flex_pred.
    The model itself is frozen.
    """
    set_seed(seed)

    device = "cpu"

    bs = int(cfg_ft["train"]["batch_size"])
    lr = float(cfg_ft["train"]["lr"])
    epochs = int(cfg_ft["train"]["epochs"])
    patience = int(cfg_ft["train"].get("patience", 5))

    train_loader = DataLoader(Subset(ds, idx_train), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(Subset(ds, idx_val),   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(Subset(ds, idx_test),  batch_size=bs, shuffle=False)

    # Same better-head style used for flex-only experiments
    head = nn.Sequential(
        nn.Linear(12, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)

    best_val_r2 = -1e9
    best_state = None
    bad = 0

    @torch.no_grad()
    def eval_val_r2():
        model.eval()
        head.eval()

        ys, preds = [], []

        for batch in val_loader:
            x = batch["x"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["y"].to(device).squeeze(1)

            out = model(x, am)
            flex_pred = out[1]
            pooled = masked_max_pool(flex_pred, am)
            yhat = head(pooled).squeeze(1)

            ys.append(y.detach().cpu().numpy())
            preds.append(yhat.detach().cpu().numpy())

        y_all = np.concatenate(ys, axis=0)
        p_all = np.concatenate(preds, axis=0)

        return float(r2_score(y_all, p_all))

    for ep in range(1, epochs + 1):
        head.train()
        model.eval()

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

        val_r2 = eval_val_r2()

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

    return eval_flex_readout(model, head, test_loader, device)


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

            ds = PBMDataset(
                pbm_path,
                k=int(cfg_pre["tokenizer"]["k"]),
                seed=seed
            )

            idx_train, idx_val, idx_test = split_dataset(
                ds,
                train_frac=float(cfg_ft["train"]["split"]["train"]),
                val_frac=float(cfg_ft["train"]["split"]["val"]),
            )

            # Random encoder/flex-head control
            random_model = build_model(cfg_pre, ckpt_path=None, seed=seed + 10000)
            random_res = train_flex_readout(
                ds, idx_train, idx_val, idx_test,
                random_model, cfg_ft, seed=seed
            )

            # Pretrained flex-only readout
            pretrained_model = build_model(cfg_pre, ckpt_path=ckpt_path, seed=seed)
            flex_res = train_flex_readout(
                ds, idx_train, idx_val, idx_test,
                pretrained_model, cfg_ft, seed=seed
            )

            row = {
                "dataset": ds_name,
                "seed": seed,
                "random_test_r2": random_res["test_r2"],
                "random_test_rmse": random_res["test_rmse"],
                "random_test_pearson": random_res["test_pearson"],
                "flexonly_test_r2": flex_res["test_r2"],
                "flexonly_test_rmse": flex_res["test_rmse"],
                "flexonly_test_pearson": flex_res["test_pearson"],
            }
            rows.append(row)

            print(
                f"{ds_name} | seed={seed} | "
                f"random_r2={random_res['test_r2']:.3f} | "
                f"flexonly_r2={flex_res['test_r2']:.3f}"
            )

    # Save CSV
    out_csv = "plots/panelA0_random_vs_flexonly_rawR2.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plot
    colors = {"Max": "#1f77b4", "Mad": "#ff7f0e", "Myc": "#2ca02c"}
    markers = {0: "o", 1: "s", 2: "^"}

    xs = [r["random_test_r2"] for r in rows]
    ys = [r["flexonly_test_r2"] for r in rows]

    lo = min(xs + ys) - 0.05
    hi = max(xs + ys) + 0.05

    # make sure negative random R2 values are visible
    lo = min(lo, -0.25)
    hi = max(hi, 0.45)

    fig, ax = plt.subplots(figsize=(6.6, 6.2), dpi=220)

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="gray")

    for r in rows:
        ax.scatter(
            r["random_test_r2"],
            r["flexonly_test_r2"],
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

    ax.set_xlabel("Test $R^2$ (random encoder + flex-style readout)", fontsize=13)
    ax.set_ylabel("Test $R^2$ (pretrained flex-only readout)", fontsize=13)
    ax.set_title("Panel A0: random encoder vs flex-only (raw test $R^2$)", fontsize=14)
    ax.grid(True, alpha=0.25)

    dataset_handles = [
        Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=colors[d], markeredgecolor='black',
            markersize=9, label=d
        )
        for d in datasets
    ]

    seed_handles = [
        Line2D(
            [0], [0], marker=markers[s], color='black',
            linestyle='None', markersize=9, label=f"seed {s}"
        )
        for s in seeds
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

    out_png = "plots/panelA0_random_vs_flexonly_rawR2.png"
    fig.savefig(out_png, bbox_inches="tight")

    print("Saved:", out_png)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
