import random
from typing import Dict, List, Tuple

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


def pearsonr(y: torch.Tensor, yhat: torch.Tensor) -> float:
    y = y.view(-1)
    yhat = yhat.view(-1)
    y0 = y - y.mean()
    p0 = yhat - yhat.mean()
    denom = (y0.std(unbiased=False) * p0.std(unbiased=False)) + 1e-8
    return float(((y0 * p0).mean() / denom).item())


def r2_score(y: torch.Tensor, yhat: torch.Tensor) -> float:
    y = y.view(-1)
    yhat = yhat.view(-1)
    sse = torch.sum((y - yhat) ** 2)
    sst = torch.sum((y - y.mean()) ** 2) + 1e-12
    return float((1.0 - (sse / sst)).item())


def rmse(y: torch.Tensor, yhat: torch.Tensor) -> float:
    return float(torch.sqrt(F.mse_loss(yhat.view(-1), y.view(-1))).item())


def pbm_collate_pad(batch: List[Dict], max_tokens: int) -> Dict[str, torch.Tensor]:
    """
    Pads variable-length token sequences to max_tokens.
    x: [B, max_tokens, 24]
    attention_mask: [B, max_tokens]
    y: [B, 1]
    """
    B = len(batch)
    x = torch.zeros((B, max_tokens, 24), dtype=torch.float32)
    am = torch.zeros((B, max_tokens), dtype=torch.long)
    y = torch.zeros((B, 1), dtype=torch.float32)

    for i, item in enumerate(batch):
        xi = item["x"]  # [Li,24]
        Li = min(xi.shape[0], max_tokens)
        x[i, :Li] = xi[:Li]
        am[i, :Li] = 1
        y[i, 0] = item["y"][0]

    return {"x": x, "attention_mask": am, "y": y}


@torch.no_grad()
def eval_hidden_plus_flex_poslinear(model, head, loader, device) -> Tuple[float, float, float]:
    model.eval()
    head.eval()

    ys = []
    preds = []

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device)

        # Get BOTH hidden states and flex predictions per token
        _, flex_pred, h = model(x, am, return_hidden=True)   # h:[B,T,D], flex_pred:[B,T,F]

        # Mask padding positions to 0 so flattening doesn't leak junk
        mask_f = am.unsqueeze(-1).float()
        h = h * mask_f
        flex_pred = flex_pred * mask_f

        # Concatenate per-token: [B,T,D+F] then flatten to [B, T*(D+F)]
        z = torch.cat([h, flex_pred], dim=-1)
        B, T, DF = z.shape
        feat = z.reshape(B, T * DF)

        yhat = head(feat)  # [B,1]

        ys.append(y)
        preds.append(yhat)

    y_all = torch.cat(ys, dim=0)
    p_all = torch.cat(preds, dim=0)
    return rmse(y_all, p_all), pearsonr(y_all, p_all), r2_score(y_all, p_all)


def main():
    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")

    seed = int(cfg_ft["train"]["seed"])
    set_seed(seed)

    device = cfg_ft["train"].get("device", "cpu")

    k = int(cfg_pre["tokenizer"]["k"])
    d_model = int(cfg_pre["model"]["d_model"])
    n_heads = int(cfg_pre["model"]["n_heads"])
    n_layers = int(cfg_pre["model"]["n_layers"])
    max_len = int(cfg_pre["model"]["max_len"])

    dinuc_feats = cfg_pre["features"]["dinucleotide"]
    trinuc_feats = cfg_pre["features"]["trinucleotide"]
    n_flex = len(dinuc_feats) + len(trinuc_feats)

    # Load PBM dataset
    ds = PBMDataset(cfg_ft["data_path"], k=k, seed=seed)

    # Compute max token length (PBM sequences can vary)
    max_tokens = 0
    for seq, _ in ds.items:
        L = len(seq) - k + 1
        if L > max_tokens:
            max_tokens = L
    if max_tokens <= 0:
        raise ValueError("max_tokens computed <= 0; check PBM sequences and k.")

    idx_train, idx_val, idx_test = split_dataset(
        ds,
        train_frac=float(cfg_ft["train"]["split"]["train"]),
        val_frac=float(cfg_ft["train"]["split"]["val"]),
    )

    bs = int(cfg_ft["train"]["batch_size"])
    collate = lambda batch: pbm_collate_pad(batch, max_tokens=max_tokens)

    train_loader = DataLoader(Subset(ds, idx_train), batch_size=bs, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(Subset(ds, idx_val),   batch_size=bs, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(Subset(ds, idx_test),  batch_size=bs, shuffle=False, collate_fn=collate)

    # Load pretrained model
    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=4100,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
        n_flex=n_flex,
    )

    ckpt_path = cfg_ft["pretrained_ckpt"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    # Freeze encoder (and heads) — we only train the PBM head
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    # Position-aware LINEAR head on [hidden||flex] flattened
    in_dim = max_tokens * (d_model + n_flex)
    head = nn.Linear(in_dim, 1).to(device)

    lr = float(cfg_ft["train"]["lr"])
    weight_decay = float(cfg_ft["train"].get("weight_decay", 0.0))
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(cfg_ft["train"]["epochs"])
    patience = int(cfg_ft["train"].get("patience", 10))

    best_val_r2 = -float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    print("PBM fine-tune (FROZEN encoder, HIDDEN + FLEX, POSITION-AWARE LINEAR head) starting")
    print("data =", cfg_ft["data_path"])
    print("pretrained_ckpt =", ckpt_path)
    print("n_sequences =", len(ds), "splits:", len(idx_train), len(idx_val), len(idx_test))
    print("k =", k, "max_tokens =", max_tokens, "d_model =", d_model, "n_flex =", n_flex, "head_in_dim =", in_dim)
    print("lr =", lr, "weight_decay =", weight_decay)
    print("early stopping uses val_r2 (maximize)")

    for ep in range(1, epochs + 1):
        head.train()
        total = 0.0
        nb = 0

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
                B, T, DF = z.shape
                feat = z.reshape(B, T * DF)

            yhat = head(feat)
            loss = F.mse_loss(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            nb += 1

        train_mse = total / max(nb, 1)
        val_rmse, val_p, val_r2 = eval_hidden_plus_flex_poslinear(model, head, val_loader, device)
        print(f"epoch {ep:02d} | train_mse={train_mse:.4f} | val_rmse={val_rmse:.4f} | val_pearson={val_p:.3f} | val_r2={val_r2:.3f}")

        if val_r2 > best_val_r2 + 1e-6:
            best_val_r2 = val_r2
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = ep
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    test_rmse, test_p, test_r2 = eval_hidden_plus_flex_poslinear(model, head, test_loader, device)

    print()
    print("DONE")
    print("best_epoch =", best_epoch)
    print("best_val_r2 =", best_val_r2)
    print("test_rmse =", test_rmse)
    print("test_pearson =", test_p)
    print("test_r2 =", test_r2)

    out_path = "checkpoints/pbm_hidden_plus_flex_poslinear_best.pt"
    torch.save(
        {
            "head_state": head.state_dict(),
            "cfg_pre": cfg_pre,
            "cfg_ft": cfg_ft,
            "max_tokens": max_tokens,
            "d_model": d_model,
            "n_flex": n_flex,
            "best_epoch": best_epoch,
            "best_val_r2": best_val_r2,
        },
        out_path,
    )
    print("saved head ->", out_path)


if __name__ == "__main__":
    main()
