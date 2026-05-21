from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.bendability_dataset import load_bendability_splits, collate_bendability_batch
from src.model import TinyMultiTaskModelOneHot
from src.utils import load_yaml


# -------------------------
# utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pearsonr_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def masked_mean_pool(h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    h: [B, T, D]
    attention_mask: [B, T]
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (h * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def load_flex_norm_stats(stats_path: str, cfg_pre: dict) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Try to load per-feature mean/std for the flex targets so flex loss is compatible with
    the existing pretrained flex head.

    Supports common formats:
    1) nested:
       dinucleotide:
         feat_name:
           mean: ...
           std: ...
       trinucleotide:
         feat_name:
           mean: ...
           std: ...
    2) flat:
       means: [...]
       stds:  [...]
    3) flat:
       mean: [...]
       std:  [...]
    """
    path = Path(stats_path)
    if not path.exists():
        print(f"[WARN] flex norm stats file not found: {path}")
        return None

    stats = yaml.safe_load(path.read_text())

    di_names = list(cfg_pre["features"]["dinucleotide"])
    tri_names = list(cfg_pre["features"]["trinucleotide"])
    feat_order = di_names + tri_names

    means = []
    stds = []

    # format 1: nested by group / feature
    if isinstance(stats, dict) and "dinucleotide" in stats and "trinucleotide" in stats:
        ok = True
        for name in di_names:
            item = stats["dinucleotide"].get(name, None)
            if not isinstance(item, dict) or "mean" not in item or "std" not in item:
                ok = False
                break
            means.append(float(item["mean"]))
            stds.append(float(item["std"]))

        if ok:
            for name in tri_names:
                item = stats["trinucleotide"].get(name, None)
                if not isinstance(item, dict) or "mean" not in item or "std" not in item:
                    ok = False
                    break
                means.append(float(item["mean"]))
                stds.append(float(item["std"]))

        if ok:
            mu = torch.tensor(means, dtype=torch.float32)
            sd = torch.tensor(stds, dtype=torch.float32)
            sd[sd < 1e-8] = 1.0
            return mu, sd

    # format 2/3: flat arrays
    if isinstance(stats, dict):
        if "means" in stats and "stds" in stats:
            means = [float(x) for x in stats["means"]]
            stds = [float(x) for x in stats["stds"]]
        elif "mean" in stats and "std" in stats and isinstance(stats["mean"], list):
            means = [float(x) for x in stats["mean"]]
            stds = [float(x) for x in stats["std"]]

        if len(means) == len(feat_order) and len(stds) == len(feat_order):
            mu = torch.tensor(means, dtype=torch.float32)
            sd = torch.tensor(stds, dtype=torch.float32)
            sd[sd < 1e-8] = 1.0
            return mu, sd

    print(f"[WARN] Could not parse flex norm stats from {path}; using raw flex targets.")
    return None


# -------------------------
# model pieces
# -------------------------
def build_pretrained_model(cfg_pre: dict, ckpt_path: str) -> TinyMultiTaskModelOneHot:
    k = int(cfg_pre["tokenizer"]["k"])
    n_flex = len(cfg_pre["features"]["dinucleotide"]) + len(cfg_pre["features"]["trinucleotide"])

    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=4100,
        d_model=int(cfg_pre["model"]["d_model"]),
        n_heads=int(cfg_pre["model"]["n_heads"]),
        n_layers=int(cfg_pre["model"]["n_layers"]),
        max_len=int(cfg_pre["model"]["max_len"]),
        n_flex=n_flex,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model


class BendabilityHead(nn.Module):
    """
    Sequence-level bendability regression head.
    Uses mean pooling over token hidden states.
    """
    def __init__(self, d_model: int = 64, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = masked_mean_pool(h, attention_mask)   # [B, d_model]
        return self.net(pooled)                        # [B, 1]


# -------------------------
# train / eval
# -------------------------
def compute_flex_loss(
    flex_pred: torch.Tensor,
    flex_targets: torch.Tensor,
    attention_mask: torch.Tensor,
    norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    flex_pred:    [B, T, 12]
    flex_targets: [B, T, 12] raw targets from lookup tables
    """
    if norm_stats is not None:
        mu, sd = norm_stats
        mu = mu.to(flex_targets.device).view(1, 1, -1)
        sd = sd.to(flex_targets.device).view(1, 1, -1)
        flex_targets = (flex_targets - mu) / sd

    mask = attention_mask.unsqueeze(-1).float()
    diff2 = ((flex_pred - flex_targets) ** 2) * mask
    denom = (mask.sum() * flex_pred.shape[-1]).clamp(min=1.0)
    return diff2.sum() / denom


@torch.no_grad()
def evaluate(
    model: nn.Module,
    bend_head: nn.Module,
    loader: DataLoader,
    device: str,
    lambda_flex: float,
    norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
):
    model.eval()
    bend_head.eval()

    total_loss = 0.0
    total_flex = 0.0
    total_bend = 0.0
    n_batches = 0

    ys = []
    preds = []

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        flex_targets = batch["flex_targets"].to(device)
        y = batch["bendability"].to(device)

        mlm_logits, flex_pred, h = model(x, am, return_hidden=True)
        bend_pred = bend_head(h, am)

        loss_bend = F.mse_loss(bend_pred, y)
        loss_flex = compute_flex_loss(flex_pred, flex_targets, am, norm_stats) if lambda_flex > 0 else torch.tensor(0.0, device=device)

        loss = loss_bend + lambda_flex * loss_flex

        total_loss += float(loss.item())
        total_flex += float(loss_flex.item())
        total_bend += float(loss_bend.item())
        n_batches += 1

        ys.append(y.squeeze(1).detach().cpu().numpy())
        preds.append(bend_pred.squeeze(1).detach().cpu().numpy())

    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(preds, axis=0)

    rmse = float(np.sqrt(np.mean((y_all - p_all) ** 2)))
    pearson = float(pearsonr_np(y_all, p_all))
    r2 = float(1.0 - np.sum((y_all - p_all) ** 2) / max(np.sum((y_all - y_all.mean()) ** 2), 1e-12))

    return {
        "loss": total_loss / max(n_batches, 1),
        "bend_loss": total_bend / max(n_batches, 1),
        "flex_loss": total_flex / max(n_batches, 1),
        "rmse": rmse,
        "pearson": pearson,
        "r2": r2,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=str, default="data/raw/bendability/Data1")
    parser.add_argument("--pretrain_yaml", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--finetune_yaml", type=str, default="configs/finetune_pbm.yaml")
    parser.add_argument("--lookup_yaml", type=str, default="data/raw/flex_tables/lookup.yaml")
    parser.add_argument("--flex_stats", type=str, default="data/processed/flex_norm_stats.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="If omitted, use pretrained_ckpt from finetune_pbm.yaml")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_flex", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="checkpoints/bendability_stage1_data1.pt")
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device =", device)

    cfg_pre = load_yaml(args.pretrain_yaml)
    cfg_ft = load_yaml(args.finetune_yaml)

    ckpt_path = args.checkpoint if args.checkpoint is not None else cfg_ft["pretrained_ckpt"]
    print("loading pretrained checkpoint:", ckpt_path)

    train_ds, valid_ds, test_ds = load_bendability_splits(
        split_dir=args.split_dir,
        k=int(cfg_pre["tokenizer"]["k"]),
        lookup_yaml=args.lookup_yaml,
        config_yaml=args.pretrain_yaml,
    )

    print("train / valid / test sizes:")
    print(len(train_ds), len(valid_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_bendability_batch
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bendability_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bendability_batch
    )

    model = build_pretrained_model(cfg_pre, ckpt_path).to(device)
    bend_head = BendabilityHead(
        d_model=int(cfg_pre["model"]["d_model"]),
        hidden_dim=int(cfg_pre["model"]["d_model"]),
        dropout=0.1,
    ).to(device)

    norm_stats = load_flex_norm_stats(args.flex_stats, cfg_pre)
    if norm_stats is not None:
        print("Loaded flex normalization stats.")
    else:
        print("Using raw flex targets for flex loss.")

    # Two learning-rate groups:
    # - model (encoder + existing flex/mlm heads): small LR
    # - new bendability head: larger LR
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.encoder_lr},
            {"params": bend_head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float("inf")
    best_epoch = -1
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        bend_head.train()

        running_loss = 0.0
        running_bend = 0.0
        running_flex = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            am = batch["attention_mask"].to(device)
            flex_targets = batch["flex_targets"].to(device)
            y = batch["bendability"].to(device)

            mlm_logits, flex_pred, h = model(x, am, return_hidden=True)
            bend_pred = bend_head(h, am)

            loss_bend = F.mse_loss(bend_pred, y)
            loss_flex = compute_flex_loss(flex_pred, flex_targets, am, norm_stats) if args.lambda_flex > 0 else torch.tensor(0.0, device=device)

            loss = loss_bend + args.lambda_flex * loss_flex

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(bend_head.parameters()), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            running_bend += float(loss_bend.item())
            running_flex += float(loss_flex.item())
            n_batches += 1

        train_stats = {
            "loss": running_loss / max(n_batches, 1),
            "bend_loss": running_bend / max(n_batches, 1),
            "flex_loss": running_flex / max(n_batches, 1),
        }

        val_stats = evaluate(
            model=model,
            bend_head=bend_head,
            loader=valid_loader,
            device=device,
            lambda_flex=args.lambda_flex,
            norm_stats=norm_stats,
        )

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"train_bend={train_stats['bend_loss']:.4f} | "
            f"train_flex={train_stats['flex_loss']:.4f} | "
            f"val_rmse={val_stats['rmse']:.4f} | "
            f"val_pearson={val_stats['pearson']:.4f} | "
            f"val_r2={val_stats['r2']:.4f} | "
            f"val_bend={val_stats['bend_loss']:.4f} | "
            f"val_flex={val_stats['flex_loss']:.4f}"
        )

        if val_stats["rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = val_stats["rmse"]
            best_epoch = epoch
            bad = 0
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "bend_head_state": {k: v.detach().cpu().clone() for k, v in bend_head.state_dict().items()},
                "cfg_pre": cfg_pre,
                "split_dir": args.split_dir,
                "seed": args.seed,
                "lambda_flex": args.lambda_flex,
                "encoder_lr": args.encoder_lr,
                "head_lr": args.head_lr,
                "best_val_rmse": best_val_rmse,
                "best_epoch": best_epoch,
            }
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("No best checkpoint was recorded.")

    # restore best
    model.load_state_dict(best_state["model_state"])
    bend_head.load_state_dict(best_state["bend_head_state"])

    test_stats = evaluate(
        model=model,
        bend_head=bend_head,
        loader=test_loader,
        device=device,
        lambda_flex=args.lambda_flex,
        norm_stats=norm_stats,
    )

    print()
    print("BEST CHECKPOINT")
    print("best_epoch =", best_epoch)
    print("best_val_rmse =", best_val_rmse)
    print("test_rmse =", test_stats["rmse"])
    print("test_pearson =", test_stats["pearson"])
    print("test_r2 =", test_stats["r2"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save best state + final test stats
    best_state["test_stats"] = test_stats
    torch.save(best_state, out_path)
    print("saved checkpoint ->", out_path)


if __name__ == "__main__":
    main()
