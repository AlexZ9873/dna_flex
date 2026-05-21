import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.utils import load_yaml
from src.model import TinyMultiTaskModelOneHot
from src.pbm_dataset import PBMDataset, split_dataset


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


def metrics_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pearson = pearsonr_np(y_true, y_pred)
    return float(r2), pearson, rmse


# -------------------------
# model loading
# -------------------------
def build_model(cfg_pre, ckpt_path):
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

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model


# -------------------------
# feature extraction
# -------------------------
@torch.no_grad()
def extract_hidden_plus_flex_features(model, subset, batch_size, device):
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    feats = []
    ys = []

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].to(device).squeeze(1)

        _, flex_pred, h = model(x, am, return_hidden=True)

        # zero out padded tokens if any
        mask_f = am.unsqueeze(-1).float()
        h = h * mask_f
        flex_pred = flex_pred * mask_f

        z = torch.cat([h, flex_pred], dim=-1)  # [B, T, 76]
        feat = z.reshape(z.shape[0], -1)       # flatten across token positions

        feats.append(feat.detach())
        ys.append(y.detach())

    X = torch.cat(feats, dim=0)
    y = torch.cat(ys, dim=0)
    return X, y


# -------------------------
# train downstream head
# -------------------------
def train_linear_head(Xtr, ytr, Xva, yva, Xte, yte, device, seed, epochs=30, lr=1e-3, wd=1e-4, patience=5):
    set_seed(seed)

    d_in = Xtr.shape[1]
    head = nn.Linear(d_in, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)

    best_val_r2 = -1e9
    best_state = None
    best_epoch = 0
    bad = 0

    n = Xtr.shape[0]
    batch_size = 256

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = head(Xtr[idx]).squeeze(1)
            loss = F.mse_loss(pred, ytr[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            val_pred = head(Xva).squeeze(1)
            val_r2, _, _ = metrics_from_arrays(
                yva.detach().cpu().numpy(),
                val_pred.detach().cpu().numpy(),
            )

        if val_r2 > best_val_r2 + 1e-6:
            best_val_r2 = val_r2
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        test_pred = head(Xte).squeeze(1)

    test_r2, test_pearson, test_rmse = metrics_from_arrays(
        yte.detach().cpu().numpy(),
        test_pred.detach().cpu().numpy(),
    )

    return {
        "best_epoch": best_epoch,
        "best_val_r2": float(best_val_r2),
        "test_r2": float(test_r2),
        "test_pearson": float(test_pearson),
        "test_rmse": float(test_rmse),
    }


# -------------------------
# benchmark one checkpoint
# -------------------------
def benchmark_one_checkpoint(model_label, ckpt_path, cfg_pre, cfg_ft, device):
    k = int(cfg_pre["tokenizer"]["k"])

    model = build_model(cfg_pre, ckpt_path).to(device)

    datasets = {
        "Max": "data/raw/pbm/Max.txt",
        "Mad": "data/raw/pbm/Mad.txt",
        "Myc": "data/raw/pbm/Myc.txt",
    }

    train_frac = 0.8
    val_frac = 0.1
    if "train" in cfg_ft and "split" in cfg_ft["train"]:
        train_frac = float(cfg_ft["train"]["split"]["train"])
        val_frac = float(cfg_ft["train"]["split"]["val"])

    rows = []

    for ds_name, ds_path in datasets.items():
        for seed in [0, 1, 2]:
            set_seed(seed)

            ds = PBMDataset(ds_path, k=k, seed=seed)
            idx_train, idx_val, idx_test = split_dataset(
                ds,
                train_frac=train_frac,
                val_frac=val_frac,
            )

            train_subset = Subset(ds, idx_train)
            val_subset = Subset(ds, idx_val)
            test_subset = Subset(ds, idx_test)

            Xtr, ytr = extract_hidden_plus_flex_features(model, train_subset, batch_size=256, device=device)
            Xva, yva = extract_hidden_plus_flex_features(model, val_subset, batch_size=256, device=device)
            Xte, yte = extract_hidden_plus_flex_features(model, test_subset, batch_size=256, device=device)

            res = train_linear_head(
                Xtr=Xtr,
                ytr=ytr,
                Xva=Xva,
                yva=yva,
                Xte=Xte,
                yte=yte,
                device=device,
                seed=seed,
                epochs=30,
                lr=1e-3,
                wd=1e-4,
                patience=5,
            )

            row = {
                "model": model_label,
                "checkpoint": ckpt_path,
                "dataset": ds_name,
                "seed": seed,
                "n_total": len(ds),
                "n_train": len(idx_train),
                "n_val": len(idx_val),
                "n_test": len(idx_test),
                "best_epoch": res["best_epoch"],
                "best_val_r2": res["best_val_r2"],
                "test_r2": res["test_r2"],
                "test_pearson": res["test_pearson"],
                "test_rmse": res["test_rmse"],
            }
            rows.append(row)

            print(
                f"[{model_label}] {ds_name} seed={seed} | "
                f"val_r2={res['best_val_r2']:.4f} | "
                f"test_r2={res['test_r2']:.4f} | "
                f"test_pearson={res['test_pearson']:.4f} | "
                f"test_rmse={res['test_rmse']:.4f}"
            )

    return rows


# -------------------------
# main
# -------------------------
def main():
    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("device =", device)

    checkpoints = [
        ("bend only",      "checkpoints/bendability_stage1_data1_bendonly.pt",      "plots/gcpbm_bendonly.csv"),
        ("bend+flex",      "checkpoints/bendability_stage1_data1_flex0p2.pt",       "plots/gcpbm_bendflex.csv"),
        ("flex+mlm",       "checkpoints/bendstage_flexmlm_data1.pt",                "plots/gcpbm_flexmlm.csv"),
        ("bend+flex+mlm",  "checkpoints/bendstage_bendflexmlm_data1.pt",            "plots/gcpbm_bendflexmlm.csv"),
    ]

    all_rows = []

    for label, ckpt, out_csv in checkpoints:
        print("\n============================================================")
        print("Benchmarking:", label)
        print("Checkpoint :", ckpt)
        print("Output CSV :", out_csv)
        print("============================================================")

        rows = benchmark_one_checkpoint(
            model_label=label,
            ckpt_path=ckpt,
            cfg_pre=cfg_pre,
            cfg_ft=cfg_ft,
            device=device,
        )

        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print("Saved:", out_csv)
        all_rows.extend(rows)

    merged_csv = "plots/gcpbm_all4_hiddenplusflex.csv"
    with open(merged_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print("\nSaved merged CSV:", merged_csv)

    # quick summary
    print("\n=== gcPBM summary by model ===")
    summary = {}
    for row in all_rows:
        summary.setdefault(row["model"], []).append(row["test_r2"])
    for k, vals in summary.items():
        vals = np.asarray(vals, dtype=float)
        print(
            f"{k:14s} | n={len(vals)} | "
            f"mean_r2={vals.mean():.4f} | "
            f"median_r2={np.median(vals):.4f} | "
            f"min={vals.min():.4f} | "
            f"max={vals.max():.4f}"
        )


if __name__ == "__main__":
    main()
