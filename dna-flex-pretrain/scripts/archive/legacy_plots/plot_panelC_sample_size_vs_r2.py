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

def load_ckpt_path():
    import yaml
    if os.path.exists("configs/finetune_pbm.yaml"):
        cfg = yaml.safe_load(open("configs/finetune_pbm.yaml"))
        if "pretrained_ckpt" in cfg:
            return cfg["pretrained_ckpt"]
    return os.environ.get("PRETRAINED_CKPT", "checkpoints/hg38_256_chr1-22_200k_di8_tri4_best_by_val_flex.pt")

def pick_subset(idxs, n, seed):
    if n <= 0 or n >= len(idxs):
        return list(idxs)
    rng = np.random.default_rng(seed)
    return list(rng.choice(idxs, size=n, replace=False))

def ridge_fit_eval(ds, idx_train, idx_val, idx_test, alphas):
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
        pva = m.predict(Xva)
        r2 = r2_score(yva, pva)
        if r2 > best_val_r2:
            best_val_r2 = r2
            best_alpha = a

    # refit on train+val with best alpha
    Xtv = np.concatenate([Xtr, Xva], 0)
    ytv = np.concatenate([ytr, yva], 0)
    m = Ridge(alpha=best_alpha, fit_intercept=True, random_state=0)
    m.fit(Xtv, ytv)
    pte = m.predict(Xte)

    test_r2 = float(r2_score(yte, pte))
    return test_r2

@torch.no_grad()
def get_hidden(model, x, am):
    out = model(x, am)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        _, _, h = out
        return h
    return model.encoder(x, am)

def transformer_flat_head_fit_eval(ds, idx_train, idx_val, idx_test, ckpt_path, seed, epochs=30, patience=5, batch=128, lr=1e-3):
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

            h = get_hidden(model, x, am)           # [B,T,64]
            feat = h.reshape(h.shape[0], T*d_model)
            yhat = head(feat).squeeze(1)

            ys.append(y.cpu().numpy())
            ps.append(yhat.detach().cpu().numpy())
        y_all = np.concatenate(ys, 0)
        p_all = np.concatenate(ps, 0)
        r2 = float(r2_score(y_all, p_all))
        return r2

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

        val_r2 = eval_loader(val_loader)
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

    test_r2 = eval_loader(test_loader)
    return float(test_r2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbm_path", type=str, default="data/raw/pbm/Max.txt")
    ap.add_argument("--sizes", type=str, default="536,1071,2142,4284,8568")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alphas", type=str, default="0.1,1,10,100")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    ckpt_path = load_ckpt_path()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find pretrained checkpoint: {ckpt_path}")

    # Build dataset once per seed? (split depends on seed in your PBMDataset)
    ridge_results = {n: [] for n in sizes}
    tx_results = {n: [] for n in sizes}

    for sd in seeds:
        set_seed(sd)
        ds = PBMDataset(args.pbm_path, k=6, seed=sd)
        idx_train, idx_val, idx_test = split_dataset(ds, train_frac=0.8, val_frac=0.1)

        for n in sizes:
            sub_train = pick_subset(idx_train, n, seed=sd + 1000 + n)
            # Ridge
            r2_ridge = ridge_fit_eval(ds, sub_train, idx_val, idx_test, alphas)
            ridge_results[n].append(r2_ridge)
            # Transformer head
            r2_tx = transformer_flat_head_fit_eval(
                ds, sub_train, idx_val, idx_test,
                ckpt_path=ckpt_path, seed=sd,
                epochs=args.epochs, patience=args.patience,
                batch=args.batch, lr=args.lr
            )
            tx_results[n].append(r2_tx)

            print(f"seed={sd} n={n} | ridge_r2={r2_ridge:.3f} | tx_r2={r2_tx:.3f}")

    # Aggregate
    xs = np.array(sizes, dtype=np.int32)
    ridge_mean = np.array([np.mean(ridge_results[n]) for n in sizes], dtype=np.float32)
    ridge_std  = np.array([np.std(ridge_results[n])  for n in sizes], dtype=np.float32)
    tx_mean    = np.array([np.mean(tx_results[n])    for n in sizes], dtype=np.float32)
    tx_std     = np.array([np.std(tx_results[n])     for n in sizes], dtype=np.float32)

    # Plot (Panel C style)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.6, 4.8), dpi=200)
    plt.errorbar(xs, ridge_mean, yerr=ridge_std, marker="o", linewidth=2, capsize=3, label="1-mer ridge")
    plt.errorbar(xs, tx_mean,    yerr=tx_std,    marker="o", linewidth=2, capsize=3, label="Transformer (frozen) + flatten head")
    plt.xscale("log")
    plt.xlabel("Sample size (train sequences)")
    plt.ylabel("Test R²")
    plt.title("Panel C-style: Sample size vs R² (mean ± std across seeds)")
    plt.grid(True, alpha=0.25)

    plt.ylim(0.0, 1.0)
    plt.legend()

    out_png = "plots/pbm_panelC_sample_size_vs_r2.png"
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print("Saved:", out_png)

    # Save raw results
    out_csv = "plots/pbm_panelC_sample_size_vs_r2.csv"
    with open(out_csv, "w") as f:
        f.write("n,model,seed,test_r2\n")
        for n in sizes:
            for i, sd in enumerate(seeds):
                f.write(f"{n},ridge,{sd},{ridge_results[n][i]}\n")
                f.write(f"{n},transformer_flat,{sd},{tx_results[n][i]}\n")
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
