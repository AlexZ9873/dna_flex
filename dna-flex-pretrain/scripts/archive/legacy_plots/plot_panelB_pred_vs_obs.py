import os, argparse, random
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

def onehot_1mer_positional(seq: str):
    base_to_idx = {"A":0, "C":1, "G":2, "T":3}
    L = len(seq)
    X = np.zeros((L, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        j = base_to_idx.get(b, None)
        if j is not None:
            X[i, j] = 1.0
    return X.reshape(-1)

def ridge_train_predict(ds, idx_train, idx_val, idx_test, alphas):
    def build_Xy(idxs):
        X_list, y_list = [], []
        for i in idxs:
            item = ds[i]
            X_list.append(onehot_1mer_positional(item["seq"]))
            y_list.append(float(item["y"].item()))
        return np.stack(X_list, 0), np.array(y_list, dtype=np.float32)

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

    Xtv = np.concatenate([Xtr, Xva], 0)
    ytv = np.concatenate([ytr, yva], 0)
    m = Ridge(alpha=best_alpha, fit_intercept=True, random_state=0)
    m.fit(Xtv, ytv)
    pte = m.predict(Xte)
    return yte, pte, best_alpha

@torch.no_grad()
def get_hidden(model, x, am):
    out = model(x, am)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        _, _, h = out
        return h
    return model.encoder(x, am)

def train_tx_flat_head(ds, idx_train, idx_val, idx_test, ckpt_path, seed, epochs=30, patience=5, batch=128, lr=1e-3):
    device = "cpu"
    set_seed(seed)

    T = ds[0]["x"].shape[0]
    d_model = 64

    model = TinyMultiTaskModelOneHot(
        input_dim=24, vocab_size=4100,
        d_model=64, n_heads=4, n_layers=2, max_len=512,
        n_flex=12
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    for p in model.parameters():
        p.requires_grad = False
    model.to(device).eval()

    head = nn.Linear(T * d_model, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=0.0)

    train_loader = DataLoader(Subset(ds, idx_train), batch_size=batch, shuffle=True)
    val_loader   = DataLoader(Subset(ds, idx_val),   batch_size=batch, shuffle=False)
    test_loader  = DataLoader(Subset(ds, idx_test),  batch_size=batch, shuffle=False)

    def eval_loader(loader):
        ys, ps = [], []
        for b in loader:
            x = b["x"].to(device)
            am = b["attention_mask"].to(device)
            y = b["y"].to(device).squeeze(1)

            h = get_hidden(model, x, am)
            feat = h.reshape(h.shape[0], T*d_model)
            yhat = head(feat).squeeze(1)

            ys.append(y.cpu().numpy())
            ps.append(yhat.detach().cpu().numpy())

        y_all = np.concatenate(ys, 0)
        p_all = np.concatenate(ps, 0)
        rmse = float(np.sqrt(np.mean((y_all - p_all)**2)))
        pearson = float(np.corrcoef(y_all, p_all)[0,1])
        r2 = float(r2_score(y_all, p_all))
        return rmse, pearson, r2, y_all, p_all

    best_val_r2 = -1e9
    best_state = None
    bad = 0

    for ep in range(1, epochs+1):
        head.train()
        for b in train_loader:
            x = b["x"].to(device)
            am = b["attention_mask"].to(device)
            y = b["y"].to(device)

            with torch.no_grad():
                h = get_hidden(model, x, am)
                feat = h.reshape(h.shape[0], T*d_model)

            yhat = head(feat)
            loss = F.mse_loss(yhat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_rmse, val_p, val_r2, *_ = eval_loader(val_loader)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.detach().cpu().clone() for k,v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    test_rmse, test_p, test_r2, yte, pte = eval_loader(test_loader)
    return yte, pte, {"rmse": test_rmse, "pearson": test_p, "r2": test_r2}

def load_ckpt_path():
    import yaml
    if os.path.exists("configs/finetune_pbm.yaml"):
        cfg = yaml.safe_load(open("configs/finetune_pbm.yaml"))
        if "pretrained_ckpt" in cfg:
            return cfg["pretrained_ckpt"]
    return os.environ.get("PRETRAINED_CKPT", "checkpoints/hg38_256_chr1-22_200k_di8_tri4_best_by_val_flex.pt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--pbm_path", type=str, default="data/raw/pbm/Max.txt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    args = ap.parse_args()

    ckpt_path = load_ckpt_path()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find pretrained checkpoint: {ckpt_path}")

    set_seed(args.seed)

    ds = PBMDataset(args.pbm_path, k=6, seed=args.seed)
    idx_train, idx_val, idx_test = split_dataset(ds, train_frac=0.8, val_frac=0.1)

    # Ridge
    y_ridge, p_ridge, best_alpha = ridge_train_predict(ds, idx_train, idx_val, idx_test, args.alphas)
    ridge_r2 = float(r2_score(y_ridge, p_ridge))
    ridge_p = float(np.corrcoef(y_ridge, p_ridge)[0,1])
    ridge_rmse = float(np.sqrt(np.mean((y_ridge - p_ridge)**2)))

    # Transformer + flatten head
    y_tx, p_tx, tx_stats = train_tx_flat_head(
        ds, idx_train, idx_val, idx_test,
        ckpt_path=ckpt_path, seed=args.seed,
        epochs=args.epochs, patience=args.patience,
        batch=args.batch, lr=args.lr
    )

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=200)

    def panel(ax, y, p, title, tag):
        ax.scatter(y, p, s=8, alpha=0.55)
        lo = min(y.min(), p.min())
        hi = max(y.max(), p.max())
        pad = 0.03*(hi-lo + 1e-9)
        lo -= pad; hi += pad

        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)  # y=x
        a, b = np.polyfit(y, p, 1)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, a*xs + b, linewidth=1.5)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Observed PBM score")
        ax.set_ylabel("Predicted PBM score")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.text(0.02, 0.98, tag, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")

    panel(
        axes[0], y_ridge, p_ridge,
        f"Ridge 1-mer (seed={args.seed})\nR²={ridge_r2:.3f}, Pearson={ridge_p:.3f}, RMSE={ridge_rmse:.3f}",
        "(B1)"
    )
    panel(
        axes[1], y_tx, p_tx,
        f"Transformer + flatten linear head (seed={args.seed})\nR²={tx_stats['r2']:.3f}, Pearson={tx_stats['pearson']:.3f}, RMSE={tx_stats['rmse']:.3f}",
        "(B2)"
    )

    fig.suptitle("Panel B-style: Predicted vs Observed (test set)", y=1.03, fontsize=13)
    fig.tight_layout()

    out_png = f"plots/pbm_panelB_pred_vs_obs_seed{args.seed}.png"
    fig.savefig(out_png, bbox_inches="tight")
    print("Saved:", out_png)
    print("Ridge best alpha =", best_alpha)

if __name__ == "__main__":
    main()
