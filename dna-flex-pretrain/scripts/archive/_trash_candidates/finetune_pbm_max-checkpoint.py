import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.utils import load_yaml
from src.pbm_dataset import PBMDataset, split_dataset
from src.model import TinyMultiTaskModelOneHot

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def eval_regression(model_frozen, head, loader, device, n_flex):
    model_frozen.eval()
    head.eval()

    ys = []
    preds = []

    for batch in loader:
        x = batch["x"].to(device)                 # [B,L,24]
        am = batch["attention_mask"].to(device)   # [B,L]
        y = batch["y"].to(device)                 # [B,1]

        _, flex_pred = model_frozen(x, am)        # flex_pred: [B,L,n_flex]

        # mean-pool across tokens
        mask = am.unsqueeze(-1).bool()
        neg_inf = torch.tensor(-1e9, device=flex_pred.device, dtype=flex_pred.dtype)
        flex_masked = torch.where(mask, flex_pred, neg_inf)
        pooled = flex_masked.max(dim=1).values
            # [B,n_flex]

        yhat = head(pooled)                       # [B,1]

        ys.append(y)
        preds.append(yhat)

    y_all = torch.cat(ys, dim=0).squeeze(1)
    p_all = torch.cat(preds, dim=0).squeeze(1)

    mse = F.mse_loss(p_all, y_all).item()
    rmse = math.sqrt(mse)

    # Pearson correlation (simple)
    y0 = y_all - y_all.mean()
    p0 = p_all - p_all.mean()
    corr = (y0 * p0).mean() / (y0.std(unbiased=False) * p0.std(unbiased=False) + 1e-8)
    corr = corr.item()

    return rmse, corr

def main():
    cfg_pre = load_yaml("configs/pretrain.yaml")         # for k + model dims + features
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")      # for pbm path + ckpt + training params

    seed = int(cfg_ft["train"]["seed"])
    set_seed(seed)

    device = "cpu"
    k = int(cfg_pre["tokenizer"]["k"])

    # feature count must match the pretrained checkpoint
    dinuc_feats = cfg_pre["features"]["dinucleotide"]
    trinuc_feats = cfg_pre["features"]["trinucleotide"]
    n_flex = len(dinuc_feats) + len(trinuc_feats)

    # PBM dataset
    ds = PBMDataset(cfg_ft["data_path"], k=k, seed=seed)
    idx_train, idx_val, idx_test = split_dataset(
        ds,
        train_frac=float(cfg_ft["train"]["split"]["train"]),
        val_frac=float(cfg_ft["train"]["split"]["val"]),
    )

    train_loader = DataLoader(Subset(ds, idx_train), batch_size=int(cfg_ft["train"]["batch_size"]), shuffle=True)
    val_loader   = DataLoader(Subset(ds, idx_val),   batch_size=int(cfg_ft["train"]["batch_size"]), shuffle=False)
    test_loader  = DataLoader(Subset(ds, idx_test),  batch_size=int(cfg_ft["train"]["batch_size"]), shuffle=False)

    # Load pretrained model
    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=4100,  # not used for our head, but required by constructor
        d_model=int(cfg_pre["model"]["d_model"]),
        n_heads=int(cfg_pre["model"]["n_heads"]),
        n_layers=int(cfg_pre["model"]["n_layers"]),
        max_len=int(cfg_pre["model"]["max_len"]),
        n_flex=n_flex
    )

    ckpt_path = cfg_ft["pretrained_ckpt"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    # Freeze pretrained model
    for p in model.parameters():
        p.requires_grad = False

    model.to(device)

    # Small regression head on pooled flex features
    head = nn.Sequential(
        nn.Linear(n_flex, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    opt = torch.optim.Adam(head.parameters(), lr=float(cfg_ft["train"]["lr"]), weight_decay=float(cfg_ft["train"]["weight_decay"]))

    best_val_rmse = float("inf")

    print("PBM fine-tune (frozen encoder, MAX pooling) starting")
    print("data =", cfg_ft["data_path"])
    print("pretrained_ckpt =", ckpt_path)
    print("n_sequences =", len(ds))
    print("splits:", len(idx_train), len(idx_val), len(idx_test))
    print("n_flex =", n_flex)

    for epoch in range(1, int(cfg_ft["train"]["epochs"]) + 1):
        head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)                 # [B,L,24]
            am = batch["attention_mask"].to(device)   # [B,L]
            y = batch["y"].to(device)                 # [B,1]

            with torch.no_grad():
                _, flex_pred = model(x, am)          # [B,L,n_flex]
                mask = am.unsqueeze(-1).bool()
        neg_inf = torch.tensor(-1e9, device=flex_pred.device, dtype=flex_pred.dtype)
        flex_masked = torch.where(mask, flex_pred, neg_inf)
        pooled = flex_masked.max(dim=1).values
  # [B,n_flex]

            yhat = head(pooled)                       # [B,1]
            loss = F.mse_loss(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_mse = total_loss / max(n_batches, 1)
        val_rmse, val_corr = eval_regression(model, head, val_loader, device, n_flex)

        print(f"epoch {epoch:02d} | train_mse={train_mse:.4f} | val_rmse={val_rmse:.4f} | val_pearson={val_corr:.3f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({"head_state": head.state_dict(), "cfg_pre": cfg_pre, "cfg_ft": cfg_ft},
                       "checkpoints/pbm_max_best_head_maxpool.pt")

    test_rmse, test_corr = eval_regression(model, head, test_loader, device, n_flex)
    print()
    print("DONE")
    print("best_val_rmse =", best_val_rmse)
    print("test_rmse =", test_rmse)
    print("test_pearson =", test_corr)
    print("saved head -> checkpoints/pbm_max_best_head_maxpool.pt")

if __name__ == "__main__":
    main()
