import os, math, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.pbm_dataset import PBMDataset, split_dataset
from src.model import TinyMultiTaskModelOneHot

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def eval_torch_regression(model, head, loader, device, tokens, d_model):
    model.eval()
    head.eval()
    ys, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)                     # [B,T,24]
        am = batch["attention_mask"].to(device)       # [B,T]
        y = batch["y"].to(device).squeeze(1)          # [B]

        # get hidden states
        out = model(x, am)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            _, _, h = out
        else:
            # fallback: use encoder directly if forward doesn't return hidden
            h = model.encoder(x, am)                  # [B,T,d_model]

        h = h[:, :tokens, :]                          # [B,T,d_model]
        feat = h.reshape(h.shape[0], tokens * d_model)  # [B, T*d_model]
        yhat = head(feat).squeeze(1)                  # [B]

        ys.append(y.detach().cpu().numpy())
        preds.append(yhat.detach().cpu().numpy())

    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(preds, axis=0)

    rmse = float(np.sqrt(np.mean((y_all - p_all) ** 2)))
    pearson = float(np.corrcoef(y_all, p_all)[0, 1])
    r2 = float(r2_score(y_all, p_all))
    return rmse, pearson, r2

def onehot_1mer_positional(seq: str):
    # seq length L -> feature dim L*4
    # order: A,C,G,T
    base_to_idx = {"A":0, "C":1, "G":2, "T":3}
    L = len(seq)
    X = np.zeros((L, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        j = base_to_idx.get(b, None)
        if j is not None:
            X[i, j] = 1.0
    return X.reshape(-1)  # [L*4]

def ridge_baseline_r2(ds, idx_train, idx_val, idx_test, alphas):
    # Build matrices
    def build_Xy(idxs):
        X_list, y_list = [], []
        for i in idxs:
            item = ds[i]
            seq = item["seq"]
            y = float(item["y"].item())
            X_list.append(onehot_1mer_positional(seq))
            y_list.append(y)
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.float32)
        return X, y

    Xtr, ytr = build_Xy(idx_train)
    Xva, yva = build_Xy(idx_val)
    Xte, yte = build_Xy(idx_test)

    best_alpha, best_val_r2 = None, -1e9
    for a in alphas:
        m = Ridge(alpha=a, fit_intercept=True, random_state=0)
        m.fit(Xtr, ytr)
        p = m.predict(Xva)
        r2 = r2_score(yva, p)
        if r2 > best_val_r2:
            best_val_r2 = r2
            best_alpha = a

    # Fit on train+val with best alpha
    Xtv = np.concatenate([Xtr, Xva], axis=0)
    ytv = np.concatenate([ytr, yva], axis=0)
    m = Ridge(alpha=best_alpha, fit_intercept=True, random_state=0)
    m.fit(Xtv, ytv)
    pte = m.predict(Xte)

    test_r2 = float(r2_score(yte, pte))
    test_pearson = float(np.corrcoef(yte, pte)[0, 1])
    test_rmse = float(np.sqrt(np.mean((yte - pte) ** 2)))
    return test_rmse, test_pearson, test_r2, best_alpha, best_val_r2

def load_ckpt_path():
    # prefer configs/finetune_pbm.yaml if present, else env var, else a common default
    import yaml
    if os.path.exists("configs/finetune_pbm.yaml"):
        cfg = yaml.safe_load(open("configs/finetune_pbm.yaml"))
        if "pretrained_ckpt" in cfg:
            return cfg["pretrained_ckpt"]
    return os.environ.get("PRETRAINED_CKPT", "checkpoints/hg38_256_chr1-22_200k_di8_tri4_best_by_val_flex.pt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--pbm_path", type=str, default="data/raw/pbm/Max.txt")
    args = ap.parse_args()

    ckpt_path = load_ckpt_path()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find pretrained checkpoint: {ckpt_path}")

    device = "cpu"

    rows = []

    for seed in args.seeds:
        set_seed(seed)

        # dataset (shuffled by seed inside PBMDataset)
        ds = PBMDataset(args.pbm_path, k=6, seed=seed)
        idx_train, idx_val, idx_test = split_dataset(ds, train_frac=0.8, val_frac=0.1)

        # ----- Ridge 1-mer baseline -----
        ridge_rmse, ridge_r, ridge_r2, best_alpha, best_val_r2 = ridge_baseline_r2(
            ds, idx_train, idx_val, idx_test, args.alphas
        )

        # ----- Transformer frozen encoder + position-aware flatten linear head -----
        # infer token length from first sample (PBM sequences are fixed length)
        T = ds[0]["x"].shape[0]  # tokens = L-k+1, usually 31 for 36bp seq with k=6

        # build model to get hidden states
        # (n_flex can be any value here; we only use encoder output h)
        model = TinyMultiTaskModelOneHot(
            input_dim=24,
            vocab_size=4100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_len=512,
            n_flex=12
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

        for p in model.parameters():
            p.requires_grad = False
        model.to(device)

        # position-aware linear head on flattened hidden states
        d_model = 64
        head = nn.Linear(T * d_model, 1).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=0.0)

        train_loader = DataLoader(Subset(ds, idx_train), batch_size=args.batch, shuffle=True)
        val_loader   = DataLoader(Subset(ds, idx_val),   batch_size=args.batch, shuffle=False)
        test_loader  = DataLoader(Subset(ds, idx_test),  batch_size=args.batch, shuffle=False)

        best_val_r2_head = -1e9
        best_state = None
        bad = 0

        for epoch in range(1, args.epochs + 1):
            head.train()
            for batch in train_loader:
                x = batch["x"].to(device)
                am = batch["attention_mask"].to(device)
                y = batch["y"].to(device)  # [B,1]

                # hidden
                with torch.no_grad():
                    out = model(x, am)
                    if isinstance(out, (tuple, list)) and len(out) == 3:
                        _, _, h = out
                    else:
                        h = model.encoder(x, am)
                    feat = h.reshape(h.shape[0], T * d_model)

                yhat = head(feat)
                loss = F.mse_loss(yhat, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

            # eval on val
            _, _, val_r2 = eval_torch_regression(model, head, val_loader, device, T, d_model)

            if val_r2 > best_val_r2_head:
                best_val_r2_head = val_r2
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break

        if best_state is not None:
            head.load_state_dict(best_state)

        tr_rmse, tr_r, tr_r2 = eval_torch_regression(model, head, test_loader, device, T, d_model)

        rows.append({
            "seed": seed,
            "ridge_test_r2": ridge_r2,
            "ridge_test_pearson": ridge_r,
            "ridge_test_rmse": ridge_rmse,
            "ridge_best_alpha": best_alpha,
            "ridge_best_val_r2": best_val_r2,
            "tx_test_r2": tr_r2,
            "tx_test_pearson": tr_r,
            "tx_test_rmse": tr_rmse,
        })

        print(f"[seed {seed}] ridge_r2={ridge_r2:.3f} | tx_flat_r2={tr_r2:.3f}")

    # ----- Plot (paper Panel A style) -----
    import matplotlib.pyplot as plt

    xs = np.array([r["ridge_test_r2"] for r in rows], dtype=np.float32)
    ys = np.array([r["tx_test_r2"] for r in rows], dtype=np.float32)

    fig = plt.figure(figsize=(6.2, 5.2))
    ax = plt.gca()
    ax.scatter(xs, ys, s=55)

    # annotate seed numbers
    for r in rows:
        ax.text(r["ridge_test_r2"] + 0.005, r["tx_test_r2"] + 0.005, str(r["seed"]), fontsize=10)

    # diagonal y=x
    lo = min(xs.min(), ys.min()) - 0.05
    hi = max(xs.max(), ys.max()) + 0.05
    lo = max(lo, -0.05)
    hi = min(hi, 1.05)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Test $R^2$ (Ridge 1-mer baseline)")
    ax.set_ylabel("Test $R^2$ (Transformer + flatten pos-linear head)")
    ax.set_title("Panel A-style comparison: Transformer vs 1-mer baseline")
    ax.grid(True, alpha=0.25)

    out_png = "plots/pbm_panelA_r2_ridge_vs_transformer.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print("\nSaved plot:", out_png)

    # save table
    import csv
    out_csv = "plots/pbm_panelA_r2_ridge_vs_transformer.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved table:", out_csv)

if __name__ == "__main__":
    main()
