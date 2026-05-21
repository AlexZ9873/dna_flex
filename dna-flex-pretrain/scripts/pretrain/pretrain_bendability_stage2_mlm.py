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
from src.tokenization import build_kmer_vocab, encode_sequence_to_ids
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
        pooled = masked_mean_pool(h, attention_mask)
        return self.net(pooled)


# -------------------------
# MLM masking
# -------------------------
def make_mlm_batch_from_sequences(
    seqs,
    x_unmasked: torch.Tensor,
    k: int,
    stoi: dict,
    mlm_prob: float,
    rng: random.Random,
):
    """
    Create masked input and MLM labels on the fly from raw sequences.

    seqs: list[str]
    x_unmasked: [B, T, 24]
    returns:
      x_masked:   [B, T, 24]
      mlm_labels: [B, T] with -100 at non-masked positions
    """
    B, T, D = x_unmasked.shape
    device = x_unmasked.device

    x_masked = x_unmasked.clone()
    mlm_labels = torch.full((B, T), -100, dtype=torch.long, device=device)

    for b, seq in enumerate(seqs):
        token_ids = encode_sequence_to_ids(seq, k, stoi, add_cls=False)
        L = min(len(token_ids), T)

        mask_positions = []
        for pos in range(L):
            if rng.random() < mlm_prob:
                mask_positions.append(pos)
        if len(mask_positions) == 0:
            mask_positions.append(rng.randrange(L))

        for pos in mask_positions:
            mlm_labels[b, pos] = token_ids[pos]
            x_masked[b, pos, :] = 0.0

    return x_masked, mlm_labels


# -------------------------
# losses
# -------------------------
def compute_flex_loss(
    flex_pred: torch.Tensor,
    flex_targets: torch.Tensor,
    attention_mask: torch.Tensor,
    norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Match the original hg38 pretraining more closely: Huber loss on normalized flex targets.
    """
    if norm_stats is not None:
        mu, sd = norm_stats
        mu = mu.to(flex_targets.device).view(1, 1, -1)
        sd = sd.to(flex_targets.device).view(1, 1, -1)
        flex_targets = (flex_targets - mu) / sd

    valid = attention_mask.bool()  # [B, T]
    return F.huber_loss(flex_pred[valid], flex_targets[valid], delta=1.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    bend_head: Optional[nn.Module],
    loader: DataLoader,
    device: str,
    k: int,
    stoi: dict,
    mlm_prob: float,
    lambda_mlm: float,
    lambda_flex: float,
    lambda_bend: float,
    norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
):
    model.eval()
    if bend_head is not None:
        bend_head.eval()

    total_active = 0.0
    total_mlm = 0.0
    total_flex = 0.0
    total_bend = 0.0
    n_batches = 0

    ys = []
    preds = []

    # deterministic evaluation masking
    rng_eval = random.Random(12345)

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        flex_targets = batch["flex_targets"].to(device)
        y = batch["bendability"].to(device)
        seqs = batch["seq"]

        if lambda_mlm > 0:
            x_in, mlm_labels = make_mlm_batch_from_sequences(
                seqs=seqs,
                x_unmasked=x,
                k=k,
                stoi=stoi,
                mlm_prob=mlm_prob,
                rng=rng_eval,
            )
        else:
            x_in = x
            mlm_labels = torch.full((x.shape[0], x.shape[1]), -100, dtype=torch.long, device=device)

        mlm_logits, flex_pred, h = model(x_in, am, return_hidden=True)

        mlm_loss = (
            F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
            if lambda_mlm > 0 else torch.tensor(0.0, device=device)
        )

        flex_loss = (
            compute_flex_loss(flex_pred, flex_targets, am, norm_stats)
            if lambda_flex > 0 else torch.tensor(0.0, device=device)
        )

        if lambda_bend > 0:
            bend_pred = bend_head(h, am)
            bend_loss = F.mse_loss(bend_pred, y)

            ys.append(y.squeeze(1).detach().cpu().numpy())
            preds.append(bend_pred.squeeze(1).detach().cpu().numpy())
        else:
            bend_loss = torch.tensor(0.0, device=device)

        active_loss = lambda_mlm * mlm_loss + lambda_flex * flex_loss + lambda_bend * bend_loss

        total_active += float(active_loss.item())
        total_mlm += float(mlm_loss.item())
        total_flex += float(flex_loss.item())
        total_bend += float(bend_loss.item())
        n_batches += 1

    out = {
        "active_loss": total_active / max(n_batches, 1),
        "mlm_loss": total_mlm / max(n_batches, 1),
        "flex_loss": total_flex / max(n_batches, 1),
        "bend_loss": total_bend / max(n_batches, 1),
        "rmse": float("nan"),
        "pearson": float("nan"),
        "r2": float("nan"),
    }

    if lambda_bend > 0 and len(ys) > 0:
        y_all = np.concatenate(ys, axis=0)
        p_all = np.concatenate(preds, axis=0)
        rmse = float(np.sqrt(np.mean((y_all - p_all) ** 2)))
        pearson = float(pearsonr_np(y_all, p_all))
        denom = np.sum((y_all - y_all.mean()) ** 2)
        r2 = float(1.0 - np.sum((y_all - p_all) ** 2) / max(denom, 1e-12))
        out["rmse"] = rmse
        out["pearson"] = pearson
        out["r2"] = r2

    return out


def fmt(x):
    if x is None:
        return "NA"
    try:
        if math.isnan(float(x)):
            return "NA"
    except Exception:
        pass
    return f"{float(x):.4f}"


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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_mlm", type=float, default=0.1)
    parser.add_argument("--lambda_flex", type=float, default=1.0)
    parser.add_argument("--lambda_bend", type=float, default=1.0)
    parser.add_argument("--mlm_prob", type=float, default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="checkpoints/bendability_stage2.pt")
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

    if args.mlm_prob is None:
        mlm_prob = float(cfg_pre["train"]["mlm_prob"])
    else:
        mlm_prob = float(args.mlm_prob)

    k = int(cfg_pre["tokenizer"]["k"])
    stoi, itos = build_kmer_vocab(k)

    train_ds, valid_ds, test_ds = load_bendability_splits(
        split_dir=args.split_dir,
        k=k,
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

    bend_head = None
    if args.lambda_bend > 0:
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

    param_groups = [{"params": model.parameters(), "lr": args.encoder_lr}]
    if bend_head is not None:
        param_groups.append({"params": bend_head.parameters(), "lr": args.head_lr})

    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    best_val_active = float("inf")
    best_epoch = -1
    best_state = None
    bad = 0

    rng_train = random.Random(args.seed + 123)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if bend_head is not None:
            bend_head.train()

        run_active = 0.0
        run_mlm = 0.0
        run_flex = 0.0
        run_bend = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            am = batch["attention_mask"].to(device)
            flex_targets = batch["flex_targets"].to(device)
            y = batch["bendability"].to(device)
            seqs = batch["seq"]

            if args.lambda_mlm > 0:
                x_in, mlm_labels = make_mlm_batch_from_sequences(
                    seqs=seqs,
                    x_unmasked=x,
                    k=k,
                    stoi=stoi,
                    mlm_prob=mlm_prob,
                    rng=rng_train,
                )
            else:
                x_in = x
                mlm_labels = torch.full((x.shape[0], x.shape[1]), -100, dtype=torch.long, device=device)

            mlm_logits, flex_pred, h = model(x_in, am, return_hidden=True)

            mlm_loss = (
                F.cross_entropy(
                    mlm_logits.view(-1, mlm_logits.size(-1)),
                    mlm_labels.view(-1),
                    ignore_index=-100,
                )
                if args.lambda_mlm > 0 else torch.tensor(0.0, device=device)
            )

            flex_loss = (
                compute_flex_loss(flex_pred, flex_targets, am, norm_stats)
                if args.lambda_flex > 0 else torch.tensor(0.0, device=device)
            )

            if args.lambda_bend > 0:
                bend_pred = bend_head(h, am)
                bend_loss = F.mse_loss(bend_pred, y)
            else:
                bend_loss = torch.tensor(0.0, device=device)

            active_loss = (
                args.lambda_mlm * mlm_loss
                + args.lambda_flex * flex_loss
                + args.lambda_bend * bend_loss
            )

            optimizer.zero_grad()
            active_loss.backward()

            params = list(model.parameters())
            if bend_head is not None:
                params += list(bend_head.parameters())
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            optimizer.step()

            run_active += float(active_loss.item())
            run_mlm += float(mlm_loss.item())
            run_flex += float(flex_loss.item())
            run_bend += float(bend_loss.item())
            n_batches += 1

        train_active = run_active / max(n_batches, 1)
        train_mlm = run_mlm / max(n_batches, 1)
        train_flex = run_flex / max(n_batches, 1)
        train_bend = run_bend / max(n_batches, 1)

        val_stats = evaluate(
            model=model,
            bend_head=bend_head,
            loader=valid_loader,
            device=device,
            k=k,
            stoi=stoi,
            mlm_prob=mlm_prob,
            lambda_mlm=args.lambda_mlm,
            lambda_flex=args.lambda_flex,
            lambda_bend=args.lambda_bend,
            norm_stats=norm_stats,
        )

        print(
            f"epoch {epoch:02d} | "
            f"train_active={train_active:.4f} | "
            f"train_mlm={train_mlm:.4f} | "
            f"train_flex={train_flex:.4f} | "
            f"train_bend={train_bend:.4f} | "
            f"val_active={val_stats['active_loss']:.4f} | "
            f"val_mlm={val_stats['mlm_loss']:.4f} | "
            f"val_flex={val_stats['flex_loss']:.4f} | "
            f"val_bend={val_stats['bend_loss']:.4f} | "
            f"val_rmse={fmt(val_stats['rmse'])} | "
            f"val_pearson={fmt(val_stats['pearson'])} | "
            f"val_r2={fmt(val_stats['r2'])}"
        )

        if val_stats["active_loss"] < best_val_active - 1e-6:
            best_val_active = val_stats["active_loss"]
            best_epoch = epoch
            bad = 0
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "bend_head_state": (
                    {k: v.detach().cpu().clone() for k, v in bend_head.state_dict().items()}
                    if bend_head is not None else None
                ),
                "cfg_pre": cfg_pre,
                "split_dir": args.split_dir,
                "seed": args.seed,
                "lambda_mlm": args.lambda_mlm,
                "lambda_flex": args.lambda_flex,
                "lambda_bend": args.lambda_bend,
                "mlm_prob": mlm_prob,
                "encoder_lr": args.encoder_lr,
                "head_lr": args.head_lr,
                "best_val_active": best_val_active,
                "best_epoch": best_epoch,
            }
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("No best checkpoint was recorded.")

    # save last checkpoint too
    last_out = Path(args.out)
    last_out.parent.mkdir(parents=True, exist_ok=True)
    last_path = last_out.with_name(last_out.stem + "_last.pt")

    torch.save(
        {
            "model_state": model.state_dict(),
            "bend_head_state": bend_head.state_dict() if bend_head is not None else None,
            "cfg_pre": cfg_pre,
            "split_dir": args.split_dir,
            "seed": args.seed,
            "lambda_mlm": args.lambda_mlm,
            "lambda_flex": args.lambda_flex,
            "lambda_bend": args.lambda_bend,
            "mlm_prob": mlm_prob,
        },
        last_path,
    )
    print("saved last checkpoint ->", last_path)

    # restore best and evaluate test
    model.load_state_dict(best_state["model_state"])
    if bend_head is not None and best_state["bend_head_state"] is not None:
        bend_head.load_state_dict(best_state["bend_head_state"])

    test_stats = evaluate(
        model=model,
        bend_head=bend_head,
        loader=test_loader,
        device=device,
        k=k,
        stoi=stoi,
        mlm_prob=mlm_prob,
        lambda_mlm=args.lambda_mlm,
        lambda_flex=args.lambda_flex,
        lambda_bend=args.lambda_bend,
        norm_stats=norm_stats,
    )

    print()
    print("BEST CHECKPOINT")
    print("best_epoch =", best_epoch)
    print("best_val_active =", best_val_active)
    print("test_active =", test_stats["active_loss"])
    print("test_mlm =", test_stats["mlm_loss"])
    print("test_flex =", test_stats["flex_loss"])
    print("test_bend =", test_stats["bend_loss"])
    print("test_rmse =", fmt(test_stats["rmse"]))
    print("test_pearson =", fmt(test_stats["pearson"]))
    print("test_r2 =", fmt(test_stats["r2"]))

    best_state["test_stats"] = test_stats
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    print("saved best checkpoint ->", out_path)


if __name__ == "__main__":
    main()