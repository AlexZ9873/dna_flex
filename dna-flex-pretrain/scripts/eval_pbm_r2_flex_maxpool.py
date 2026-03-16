import math
import random
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


def masked_max_pool(flex_pred: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
    # flex_pred: [B,L,n_flex], am: [B,L]
    mask = am.unsqueeze(-1).bool()
    neg_inf = torch.tensor(-1e9, device=flex_pred.device, dtype=flex_pred.dtype)
    flex_masked = torch.where(mask, flex_pred, neg_inf)
    return flex_masked.max(dim=1).values  # [B,n_flex]


def pearson_r(y: torch.Tensor, yhat: torch.Tensor) -> float:
    y0 = y - y.mean()
    p0 = yhat - yhat.mean()
    return float((y0 * p0).mean() / (y0.std(unbiased=False) * p0.std(unbiased=False) + 1e-8))


def r2_score(y: torch.Tensor, yhat: torch.Tensor) -> float:
    # R^2 = 1 - SSE/SST
    sse = torch.sum((y - yhat) ** 2)
    sst = torch.sum((y - y.mean()) ** 2)
    if float(sst.item()) < 1e-12:
        return float("nan")
    return float(1.0 - (sse / sst).item())


def fit_linear_calibration(yhat: torch.Tensor, y: torch.Tensor):
    # Fit y ≈ a*yhat + b (least squares)
    x = yhat
    vx = torch.var(x, unbiased=False)
    if float(vx.item()) < 1e-12:
        a = torch.tensor(1.0)
        b = y.mean() - a * x.mean()
        return float(a.item()), float(b.item())
    cov = torch.mean((x - x.mean()) * (y - y.mean()))
    a = cov / vx
    b = y.mean() - a * x.mean()
    return float(a.item()), float(b.item())


@torch.no_grad()
def predict_scores(model, head, loader, device):
    model.eval()
    head.eval()
    ys, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device)

        _, flex_pred = model(x, am)
        pooled = masked_max_pool(flex_pred, am)   # [B,n_flex]
        yhat = head(pooled)                       # [B,1]

        ys.append(y)
        preds.append(yhat)

    y_all = torch.cat(ys, dim=0).squeeze(1)
    p_all = torch.cat(preds, dim=0).squeeze(1)
    return y_all, p_all


def main():
    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")

    seed = int(cfg_ft["train"]["seed"])
    set_seed(seed)

    device = "cpu"
    k = int(cfg_pre["tokenizer"]["k"])

    dinuc_feats = cfg_pre["features"]["dinucleotide"]
    trinuc_feats = cfg_pre["features"]["trinucleotide"]
    n_flex = len(dinuc_feats) + len(trinuc_feats)

    # PBM dataset + split
    ds = PBMDataset(cfg_ft["data_path"], k=k, seed=seed)
    idx_train, idx_val, idx_test = split_dataset(
        ds,
        train_frac=float(cfg_ft["train"]["split"]["train"]),
        val_frac=float(cfg_ft["train"]["split"]["val"]),
    )

    bs = int(cfg_ft["train"]["batch_size"])
    val_loader  = DataLoader(Subset(ds, idx_val),  batch_size=bs, shuffle=False)
    test_loader = DataLoader(Subset(ds, idx_test), batch_size=bs, shuffle=False)

    # Load pretrained model
    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=4100,
        d_model=int(cfg_pre["model"]["d_model"]),
        n_heads=int(cfg_pre["model"]["n_heads"]),
        n_layers=int(cfg_pre["model"]["n_layers"]),
        max_len=int(cfg_pre["model"]["max_len"]),
        n_flex=n_flex
    )

    ckpt = torch.load(cfg_ft["pretrained_ckpt"], map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    # Build SAME head architecture as your better-head run
    head = nn.Sequential(
        nn.Linear(n_flex, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    head_ckpt_path = "checkpoints/pbm_max_best_head_flex_maxpool_betterhead.pt"
    head_ckpt = torch.load(head_ckpt_path, map_location="cpu")
    head.load_state_dict(head_ckpt["head_state"], strict=True)

    # Predict on val and test
    y_val, p_val = predict_scores(model, head, val_loader, device)
    y_test, p_test = predict_scores(model, head, test_loader, device)

    # Raw metrics
    val_r = pearson_r(y_val, p_val)
    val_r2 = r2_score(y_val, p_val)

    test_r = pearson_r(y_test, p_test)
    test_r2 = r2_score(y_test, p_test)

    # Calibrated R^2: fit a,b on val then apply to test
    a, b = fit_linear_calibration(p_val, y_val)
    p_test_cal = a * p_test + b
    test_r2_cal = r2_score(y_test, p_test_cal)

    # RMSE too (nice to report)
    test_rmse = math.sqrt(float(F.mse_loss(p_test, y_test).item()))
    test_rmse_cal = math.sqrt(float(F.mse_loss(p_test_cal, y_test).item()))

    print("PBM evaluation (flex_pred + max pooling + better head)")
    print("seed =", seed)
    print("val_pearson =", val_r)
    print("val_r2 =", val_r2)
    print("test_pearson =", test_r)
    print("test_r2 =", test_r2)
    print("test_r2_calibrated =", test_r2_cal)
    print("test_rmse =", test_rmse)
    print("test_rmse_calibrated =", test_rmse_cal)
    print("calibration: y ≈ a*yhat + b where a =", a, "b =", b)


if __name__ == "__main__":
    main()
