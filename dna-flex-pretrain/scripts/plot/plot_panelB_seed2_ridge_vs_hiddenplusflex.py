import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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
    return X.reshape(-1)


def ridge_train_predict(ds, idx_train, idx_val, idx_test, alphas):
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

    stats = {
        "alpha": float(best_alpha),
        "r2": float(r2_score(yte, pte)),
        "rmse": float(np.sqrt(np.mean((yte - pte) ** 2))),
        "pearson": float(pearsonr_np(yte, pte)),
    }
    return yte, pte, stats


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
        feat = z.reshape(z.shape[0], -1)                    # flatten
        yhat = head(feat).squeeze(1)

        ys.append(y.detach().cpu().numpy())
        preds.append(yhat.detach().cpu().numpy())

    y_all = np.concatenate(ys, 0)
    p_all = np.concatenate(preds, 0)

    stats = {
        "r2": float(r2_score(y_all, p_all)),
        "rmse": float(np.sqrt(np.mean((y_all - p_all) ** 2))),
        "pearson": float(pearsonr_np(y_all, p_all)),
    }
    return y_all, p_all, stats


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


def panel(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, s=10, alpha=0.55)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.03 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad

    # y = x
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray")

    # fitted line
    a, b = np.polyfit(y_true, y_pred, 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, a * xs + b, linewidth=1.5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)


def main():
    seed = 2
    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")
    ckpt_path = cfg_ft["pretrained_ckpt"]

    datasets = ["Max", "Mad", "Myc"]
    alpha_grid = [0.1, 1.0, 10.0, 100.0]

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 12), dpi=220)

    summary_rows = []

    for row_i, ds_name in enumerate(datasets):
        set_seed(seed)
        pbm_path = f"data/raw/pbm/{ds_name}.txt"
        ds = PBMDataset(pbm_path, k=int(cfg_pre["tokenizer"]["k"]), seed=seed)

        idx_train, idx_val, idx_test = split_dataset(
            ds,
            train_frac=float(cfg_ft["train"]["split"]["train"]),
            val_frac=float(cfg_ft["train"]["split"]["val"]),
        )

        # ridge
        y_ridge, p_ridge, ridge_stats = ridge_train_predict(ds, idx_train, idx_val, idx_test, alpha_grid)

        # hidden+flex
        model = build_pretrained_model(cfg_pre, ckpt_path)
        y_hf, p_hf, hf_stats = train_hiddenplusflex(ds, idx_train, idx_val, idx_test, model, cfg_pre, cfg_ft, seed)

        panel(
            axes[row_i, 0],
            y_ridge,
            p_ridge,
            f"{ds_name} | Ridge 1-mer\nR²={ridge_stats['r2']:.3f}, Pearson={ridge_stats['pearson']:.3f}, RMSE={ridge_stats['rmse']:.3f}"
        )
        panel(
            axes[row_i, 1],
            y_hf,
            p_hf,
            f"{ds_name} | Hidden+Flex positional head\nR²={hf_stats['r2']:.3f}, Pearson={hf_stats['pearson']:.3f}, RMSE={hf_stats['rmse']:.3f}"
        )

        axes[row_i, 0].set_ylabel("Predicted PBM score", fontsize=12)
        axes[row_i, 1].set_ylabel("Predicted PBM score", fontsize=12)
        axes[row_i, 0].set_xlabel("Observed PBM score", fontsize=12)
        axes[row_i, 1].set_xlabel("Observed PBM score", fontsize=12)

        summary_rows.append({
            "dataset": ds_name,
            "seed": seed,
            "ridge_r2": ridge_stats["r2"],
            "ridge_pearson": ridge_stats["pearson"],
            "ridge_rmse": ridge_stats["rmse"],
            "hiddenflex_r2": hf_stats["r2"],
            "hiddenflex_pearson": hf_stats["pearson"],
            "hiddenflex_rmse": hf_stats["rmse"],
        })

        print(
            f"{ds_name} | seed={seed} | "
            f"ridge_r2={ridge_stats['r2']:.3f} | "
            f"hidden+flex_r2={hf_stats['r2']:.3f}"
        )

    fig.suptitle("Panel B: Predicted vs Observed on PBM test sets (seed=2)", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_png = "plots/panelB_seed2_ridge_vs_hiddenplusflex.png"
    fig.savefig(out_png, bbox_inches="tight")

    out_csv = "plots/panelB_seed2_ridge_vs_hiddenplusflex.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("Saved:", out_png)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
