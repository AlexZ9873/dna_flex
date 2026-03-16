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
def eval_regression(model, head, loader, device):
    model.eval()
    head.eval()
    ys, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device)

        _, flex_pred = model(x, am)
        denom = am.sum(dim=1, keepdim=True).clamp(min=1).to(flex_pred.dtype)
        pooled = (flex_pred * am.unsqueeze(-1)).sum(dim=1) / denom

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

def unfreeze_last_transformer_layer(model: nn.Module) -> int:
    """
    Find ALL TransformerEncoderLayer modules inside the model and unfreeze the last one.
    Returns the number of parameters unfrozen.
    """
    layers = [m for m in model.modules() if isinstance(m, nn.TransformerEncoderLayer)]
    if len(layers) == 0:
        raise RuntimeError(
            "Could not find any nn.TransformerEncoderLayer inside the model. "
            "Your encoder may use a different module type."
        )
    last = layers[-1]
    n = 0
    for p in last.parameters():
        p.requires_grad = True
        n += p.numel()
    return n

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
        input_dim=k * 4,
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

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last transformer block (robust)
    unfrozen = unfreeze_last_transformer_layer(model)

    model.to(device)

    head = nn.Sequential(
        nn.Linear(n_flex, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    # Two LRs: tiny for encoder, larger for head
    encoder_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(head.parameters())

    opt = torch.optim.Adam(
        [
            {"params": encoder_params, "lr": 1e-5},
            {"params": head_params, "lr": float(cfg_ft["train"]["lr"])},
        ],
        weight_decay=float(cfg_ft["train"]["weight_decay"])
    )

    best_val_rmse = float("inf")

    print("PBM fine-tune (unfreeze last transformer layer) starting")
    print("pretrained_ckpt =", ckpt_path)
    print("n_sequences =", len(ds))
    print("splits:", len(idx_train), len(idx_val), len(idx_test))
    print("n_flex =", n_flex)
    print("unfrozen params =", unfrozen)

    for epoch in range(1, int(cfg_ft["train"]["epochs"]) + 1):
        model.train()
        head.train()

        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["y"].to(device)

            _, flex_pred = model(x, am)
            denom = am.sum(dim=1, keepdim=True).clamp(min=1).to(flex_pred.dtype)
            pooled = (flex_pred * am.unsqueeze(-1)).sum(dim=1) / denom

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
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "head_state": head.state_dict(),
                    "cfg_pre": cfg_pre,
                    "cfg_ft": cfg_ft,
                },
                "checkpoints/pbm_max_unfreeze_last_best.pt"
            )

    test_rmse, test_corr = eval_regression(model, head, test_loader, device)
    print()
    print("DONE")
    print("best_val_rmse =", best_val_rmse)
    print("test_rmse =", test_rmse)
    print("test_pearson =", test_corr)
    print("saved -> checkpoints/pbm_max_unfreeze_last_best.pt")

if __name__ == "__main__":
    main()
