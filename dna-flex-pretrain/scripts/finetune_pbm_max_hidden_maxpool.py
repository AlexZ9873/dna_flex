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

def masked_max_pool(hidden: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
    # hidden: [B,L,D]
    mask = am.unsqueeze(-1).bool()
    neg_inf = torch.tensor(-1e9, device=hidden.device, dtype=hidden.dtype)
    hidden_masked = torch.where(mask, hidden, neg_inf)
    return hidden_masked.max(dim=1).values  # [B,D]

@torch.no_grad()
def eval_regression(model, head, loader, device):
    model.eval()
    head.eval()
    ys, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device)

        _, _, hidden = model(x, am, return_hidden=True)  # hidden: [B,L,D]
        pooled = masked_max_pool(hidden, am)             # [B,D]
        yhat = head(pooled)

        ys.append(y)
        preds.append(yhat)

    y_all = torch.cat(ys, dim=0).squeeze(1)
    p_all = torch.cat(preds, dim=0).squeeze(1)

    mse = F.mse_loss(p_all, y_all).item()
    rmse = math.sqrt(mse)

    y0 = y_all - y_all.mean()
    p0 = p_all - p_all.mean()
    corr = (y0 * p0).mean() / (y0.std(unbiased=False) * p0.std(unbiased=False) + 1e-8)
    return rmse, corr.item()

def main():
    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")

    seed = int(cfg_ft["train"]["seed"])
    set_seed(seed)

    device = "cpu"
    k = int(cfg_pre["tokenizer"]["k"])
    d_model = int(cfg_pre["model"]["d_model"])

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
        d_model=cfg_pre["model"]["d_model"],
        n_heads=cfg_pre["model"]["n_heads"],
        n_layers=cfg_pre["model"]["n_layers"],
        max_len=cfg_pre["model"]["max_len"],
        n_flex=n_flex
    )

    ckpt_path = cfg_ft["pretrained_ckpt"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    # Freeze encoder
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    # Head on pooled hidden states
    head = nn.Sequential(
        nn.Linear(d_model, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    ).to(device)

    opt = torch.optim.Adam(head.parameters(), lr=float(cfg_ft["train"]["lr"]), weight_decay=float(cfg_ft["train"]["weight_decay"]))

    best_val_rmse = float("inf")
    save_path = "checkpoints/pbm_max_best_head_hidden_maxpool.pt"

    print("PBM fine-tune (frozen encoder, HIDDEN + MAX pooling) starting")
    print("data =", cfg_ft["data_path"])
    print("pretrained_ckpt =", ckpt_path)
    print("n_sequences =", len(ds))
    print("splits:", len(idx_train), len(idx_val), len(idx_test))
    print("d_model =", d_model)

    for epoch in range(1, int(cfg_ft["train"]["epochs"]) + 1):
        head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["y"].to(device)

            with torch.no_grad():
                _, _, hidden = model(x, am, return_hidden=True)
                pooled = masked_max_pool(hidden, am)

            yhat = head(pooled)
            loss = F.mse_loss(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_mse = total_loss / max(n_batches, 1)
        val_rmse, val_corr = eval_regression(model, head, val_loader, device)
        print(f"epoch {epoch:02d} | train_mse={train_mse:.4f} | val_rmse={val_rmse:.4f} | val_pearson={val_corr:.3f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({"head_state": head.state_dict(), "cfg_pre": cfg_pre, "cfg_ft": cfg_ft}, save_path)

    test_rmse, test_corr = eval_regression(model, head, test_loader, device)
    print()
    print("DONE")
    print("best_val_rmse =", best_val_rmse)
    print("test_rmse =", test_rmse)
    print("test_pearson =", test_corr)
    print("saved head ->", save_path)

if __name__ == "__main__":
    main()
