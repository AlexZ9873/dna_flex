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
    mask = am.unsqueeze(-1).bool()
    neg_inf = torch.tensor(-1e9, device=flex_pred.device, dtype=flex_pred.dtype)
    flex_masked = torch.where(mask, flex_pred, neg_inf)
    return flex_masked.max(dim=1).values  # [B,n_flex]

@torch.no_grad()
def eval_regression(model, head, loader, device):
    model.eval()
    head.eval()

    ys = []
    preds = []

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device)

        _, flex_pred = model(x, am)
        pooled = masked_max_pool(flex_pred, am)
        yhat = head(pooled)

        ys.append(y)
        preds.append(yhat)

    y_all = torch.cat(ys, dim=0).squeeze(1)
    p_all = torch.cat(preds, dim=0).squeeze(1)

    mse = F.mse_loss(p_all, y_all).item()
    rmse = math.sqrt(mse)

    # Pearson r
    y0 = y_all - y_all.mean()
    p0 = p_all - p_all.mean()
    pearson = (y0 * p0).mean() / (y0.std(unbiased=False) * p0.std(unbiased=False) + 1e-8)
    pearson = float(pearson.item())

    # R^2
    sse = torch.sum((y_all - p_all) ** 2)
    sst = torch.sum((y_all - y_all.mean()) ** 2)
    r2 = float(1.0 - (sse / (sst + 1e-12)).item())

    return rmse, pearson, r2

def r2_score(y: torch.Tensor, yhat: torch.Tensor) -> float:
    # R^2 = 1 - SSE/SST
    sse = torch.sum((y - yhat) ** 2)
    sst = torch.sum((y - y.mean()) ** 2)
    if float(sst.item()) < 1e-12:
        return float("nan")
    return float(1.0 - (sse / sst).item())


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

    ds = PBMDataset(cfg_ft["data_path"], k=k, seed=seed)
    idx_train, idx_val, idx_test = split_dataset(
        ds,
        train_frac=float(cfg_ft["train"]["split"]["train"]),
        val_frac=float(cfg_ft["train"]["split"]["val"]),
    )

    train_loader = DataLoader(Subset(ds, idx_train), batch_size=int(cfg_ft["train"]["batch_size"]), shuffle=True)
    val_loader   = DataLoader(Subset(ds, idx_val),   batch_size=int(cfg_ft["train"]["batch_size"]), shuffle=False)
    test_loader  = DataLoader(Subset(ds, idx_test),  batch_size=int(cfg_ft["train"]["batch_size"]), shuffle=False)

    model = TinyMultiTaskModelOneHot(
        input_dim=k*4,
        vocab_size=4100,
        d_model=int(cfg_pre["model"]["d_model"]),
        n_heads=int(cfg_pre["model"]["n_heads"]),
        n_layers=int(cfg_pre["model"]["n_layers"]),
        max_len=int(cfg_pre["model"]["max_len"]),
        n_flex=n_flex
    )

    ckpt_path = cfg_ft["pretrained_ckpt"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    # better head + dropout
    head = nn.Sequential(
        nn.Linear(n_flex, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    opt = torch.optim.Adam(head.parameters(), lr=float(cfg_ft["train"]["lr"]), weight_decay=1e-4)

    best_val_rmse = float("inf")
    best_state = None
    patience = 5
    bad = 0

    print("PBM fine-tune (frozen encoder, FLEX_PRED + MAX pooling, better head) starting")
    print("pretrained_ckpt =", ckpt_path)
    print("n_sequences =", len(ds), "splits:", len(idx_train), len(idx_val), len(idx_test))
    print("n_flex =", n_flex)

    for epoch in range(1, int(cfg_ft["train"]["epochs"]) + 1):
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
        val_rmse, val_corr, val_r2 = eval_regression(model, head, val_loader, device)
        print(f"epoch {epoch:02d} | train_mse={train_mse:.4f} | val_rmse={val_rmse:.4f} | val_pearson={val_corr:.3f} | val_r2={val_r2:.3f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = head.state_dict()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    test_rmse, test_corr, test_r2 = eval_regression(model, head, test_loader, device)
    print()
    print("DONE")
    print("best_val_rmse =", best_val_rmse)
    print("test_rmse =", test_rmse)
    print("test_pearson =", test_corr)
    print("test_r2 =", test_r2)
    torch.save({"head_state": head.state_dict(), "cfg_pre": cfg_pre, "cfg_ft": cfg_ft}, "checkpoints/pbm_max_best_head_flex_maxpool_betterhead.pt")
    print("saved head -> checkpoints/pbm_max_best_head_flex_maxpool_betterhead.pt")

if __name__ == "__main__":
    main()
