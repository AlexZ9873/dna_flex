import math
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.utils import load_yaml
from src.pbm_dataset import PBMDataset, split_dataset
from src.model import TinyMultiTaskModelOneHot


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def masked_max_pool(x: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
    """
    x:  [B, T, C]
    am: [B, T] with 1 for real tokens, 0 for padding.
    returns: [B, C] max over T, ignoring padded positions.
    """
    mask = am.unsqueeze(-1).bool()  # [B,T,1]
    neg_inf = torch.tensor(-1e9, device=x.device, dtype=x.dtype)
    x_masked = torch.where(mask, x, neg_inf)
    return x_masked.max(dim=1).values


@torch.no_grad()
def collect_preds(model, head, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      y_all:    [N]
      yhat_all: [N]
    """
    model.eval()
    head.eval()

    ys = []
    yhat = []
    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device)  # [B,1]

        _, flex_pred = model(x, am)                # [B,T,n_flex]
        pooled = masked_max_pool(flex_pred, am)    # [B,n_flex]
        pred = head(pooled)                        # [B,1]

        ys.append(y)
        yhat.append(pred)

    y_all = torch.cat(ys, dim=0).squeeze(1)
    p_all = torch.cat(yhat, dim=0).squeeze(1)
    return y_all, p_all


def pearsonr_torch(y: torch.Tensor, yhat: torch.Tensor) -> float:
    y0 = y - y.mean()
    p0 = yhat - yhat.mean()
    denom = (y0.std(unbiased=False) * p0.std(unbiased=False)) + 1e-8
    return float(((y0 * p0).mean() / denom).item())


def r2_score_torch(y: torch.Tensor, yhat: torch.Tensor) -> float:
    sse = torch.sum((y - yhat) ** 2)
    sst = torch.sum((y - y.mean()) ** 2)
    if float(sst.item()) < 1e-12:
        return float("nan")
    return float((1.0 - (sse / sst)).item())


def rmse_torch(y: torch.Tensor, yhat: torch.Tensor) -> float:
    return float(torch.sqrt(F.mse_loss(yhat, y)).item())


def fit_affine_calibration(y: torch.Tensor, yhat: torch.Tensor) -> Tuple[float, float]:
    """
    Fit y ≈ a*yhat + b using least squares (closed form).
    This fixes scale/offset mismatches (R^2 is sensitive to those).
    """
    yhat_mean = yhat.mean()
    y_mean = y.mean()
    var = torch.mean((yhat - yhat_mean) ** 2)

    if float(var.item()) < 1e-12:
        a = 0.0
        b = float(y_mean.item())
        return a, b

    cov = torch.mean((yhat - yhat_mean) * (y - y_mean))
    a = float((cov / var).item())
    b = float((y_mean - a * yhat_mean).item())
    return a, b


def apply_affine(yhat: torch.Tensor, a: float, b: float) -> torch.Tensor:
    return a * yhat + b


def main():
    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")

    seed = int(cfg_ft["train"]["seed"])
    set_seed(seed)

    device = cfg_ft["train"].get("device", "cpu")

    k = int(cfg_pre["tokenizer"]["k"])
    dinuc_feats = cfg_pre["features"]["dinucleotide"]
    trinuc_feats = cfg_pre["features"]["trinucleotide"]
    n_flex = len(dinuc_feats) + len(trinuc_feats)

    ds = PBMDataset(cfg_ft["data_path"], k=k, seed=seed)
    idx_train, idx_val, idx_test = split_dataset(
        ds,
        train_frac=float(cfg_ft["train"]["split"]["train"]),
        val_frac=float(cfg_ft["train"]["split"]["val"]),
    )

    bs = int(cfg_ft["train"]["batch_size"])
    train_loader = DataLoader(Subset(ds, idx_train), batch_size=bs, shuffle=True)
    val_loader = DataLoader(Subset(ds, idx_val), batch_size=bs, shuffle=False)
    test_loader = DataLoader(Subset(ds, idx_test), batch_size=bs, shuffle=False)

    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=4100,
        d_model=int(cfg_pre["model"]["d_model"]),
        n_heads=int(cfg_pre["model"]["n_heads"]),
        n_layers=int(cfg_pre["model"]["n_layers"]),
        max_len=int(cfg_pre["model"]["max_len"]),
        n_flex=n_flex,
    )

    ckpt_path = cfg_ft["pretrained_ckpt"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    head = nn.Sequential(
        nn.Linear(n_flex, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    opt = torch.optim.Adam(
        head.parameters(),
        lr=float(cfg_ft["train"]["lr"]),
        weight_decay=float(cfg_ft["train"].get("weight_decay", 1e-4)),
    )

    max_epochs = int(cfg_ft["train"]["epochs"])
    patience = int(cfg_ft["train"].get("patience", 8))
    min_delta = float(cfg_ft["train"].get("min_delta", 1e-4))

    best_val_r2_cal_train = -float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    print("PBM fine-tune (frozen encoder, FLEX_PRED + MAX pooling, better head) starting")
    print("pretrained_ckpt =", ckpt_path)
    print("n_sequences =", len(ds), "splits:", len(idx_train), len(idx_val), len(idx_test))
    print("n_flex =", n_flex)
    print("NOTE: val_r2_cal_train fits y≈a*yhat+b on TRAIN, applies to VAL (no leakage).")
    print("NOTE: early stopping uses val_r2_cal_train (maximize).")

    for epoch in range(1, max_epochs + 1):
        head.train()
        total = 0.0
        nb = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["y"].to(device)

            with torch.no_grad():
                _, flex_pred = model(x, am)
                pooled = masked_max_pool(flex_pred, am)

            yhat = head(pooled)
            loss = F.mse_loss(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            nb += 1

        train_mse = total / max(nb, 1)

        # raw metrics
        y_tr, p_tr = collect_preds(model, head, train_loader, device)
        y_va, p_va = collect_preds(model, head, val_loader, device)

        val_rmse = rmse_torch(y_va, p_va)
        val_pearson = pearsonr_torch(y_va, p_va)
        val_r2 = r2_score_torch(y_va, p_va)

        # calibrated metrics (fit on TRAIN only)
        a, b = fit_affine_calibration(y_tr, p_tr)
        p_va_cal = apply_affine(p_va, a, b)

        val_rmse_cal_train = rmse_torch(y_va, p_va_cal)
        val_r2_cal_train = r2_score_torch(y_va, p_va_cal)

        improved = (val_r2_cal_train > (best_val_r2_cal_train + min_delta))

        print(
            f"epoch {epoch:02d} | "
            f"train_mse={train_mse:.4f} | "
            f"val_rmse={val_rmse:.4f} | val_rmse_cal_train={val_rmse_cal_train:.4f} | "
            f"val_pearson={val_pearson:.3f} | "
            f"val_r2={val_r2:.3f} | val_r2_cal_train={val_r2_cal_train:.3f} | "
            f"best_val_r2_cal_train={best_val_r2_cal_train if best_epoch>0 else val_r2_cal_train:.3f} "
            f"improved={improved}"
        )

        if improved:
            best_val_r2_cal_train = val_r2_cal_train
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    # final test eval
    y_tr, p_tr = collect_preds(model, head, train_loader, device)
    y_te, p_te = collect_preds(model, head, test_loader, device)

    test_rmse = rmse_torch(y_te, p_te)
    test_pearson = pearsonr_torch(y_te, p_te)
    test_r2 = r2_score_torch(y_te, p_te)

    a, b = fit_affine_calibration(y_tr, p_tr)
    p_te_cal = apply_affine(p_te, a, b)
    test_rmse_cal = rmse_torch(y_te, p_te_cal)
    test_r2_cal = r2_score_torch(y_te, p_te_cal)

    print()
    print("DONE")
    print("best_epoch_by_val_r2_cal_train =", best_epoch)
    print("best_val_r2_cal_train =", best_val_r2_cal_train)
    print("test_rmse =", test_rmse)
    print("test_pearson =", test_pearson)
    print("test_r2 =", test_r2)
    print("test_rmse_calibrated(train-fit) =", test_rmse_cal)
    print("test_r2_calibrated(train-fit) =", test_r2_cal)
    print(f"calibration (fit on train): y ≈ a*yhat + b where a={a:.6f} b={b:.6f}")

    out_path = "checkpoints/pbm_max_best_head_flex_maxpool_betterhead_best_by_val_r2cal.pt"
    torch.save({"head_state": head.state_dict(), "cfg_pre": cfg_pre, "cfg_ft": cfg_ft}, out_path)
    print("saved head ->", out_path)


if __name__ == "__main__":
    main()
