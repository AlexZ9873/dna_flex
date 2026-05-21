import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from src.pbm_dataset import PBMDataset, split_dataset
from src.model import TinyMultiTaskModelOneHot
from src.utils import load_yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def onehot_1mer_positional(seq: str):
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    L = len(seq)
    X = np.zeros((L, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        j = base_to_idx.get(b, None)
        if j is not None:
            X[i, j] = 1.0
    return X.reshape(-1)


def fit_ridge_with_val(Xtr, ytr, Xva, yva, Xte, yte, alpha_grid):
    best_alpha = None
    best_val_r2 = -1e9
    best_model = None

    for a in alpha_grid:
        reg = Ridge(alpha=a, fit_intercept=True, random_state=0)
        reg.fit(Xtr, ytr)
        pva = reg.predict(Xva)
        val_r2 = r2_score(yva, pva)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = a
            best_model = reg

    pte = best_model.predict(Xte)
    test_r2 = float(r2_score(yte, pte))
    return test_r2, best_alpha


def build_pretrained_model(cfg_pre, ckpt_path):
    k = int(cfg_pre["tokenizer"]["k"])
    n_flex = len(cfg_pre["features"]["dinucleotide"]) + len(cfg_pre["features"]["trinucleotide"])

    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=4100,
        d_model=int(cfg_pre["model"]["d_model"]),
        n_heads=int(cfg_pre["model"]["n_heads"]),
        n_layers=int(cfg_pre["model"]["n_layers"]),
        max_len=int(cfg_pre["model"]["max_len"]),
        n_flex=n_flex
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model


@torch.no_grad()
def extract_hidden_features(model, ds, indices, use_flex=False, device="cpu"):
    loader = DataLoader(Subset(ds, indices), batch_size=128, shuffle=False)
    feats = []
    ys = []

    for batch in loader:
        x = batch["x"].to(device)
        am = batch["attention_mask"].to(device)
        y = batch["y"].squeeze(1).cpu().numpy()

        _, flex_pred, h = model(x, am, return_hidden=True)

        if use_flex:
            mask_f = am.unsqueeze(-1).float()
            h = h * mask_f
            flex_pred = flex_pred * mask_f
            z = torch.cat([h, flex_pred], dim=-1)  # [B,T,76]
            feat = z.reshape(z.shape[0], -1)
        else:
            feat = h.reshape(h.shape[0], -1)       # [B,T*64]

        feats.append(feat.cpu().numpy())
        ys.append(y)

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def standardize_train_apply(Xtr, Xva, Xte):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Max", choices=["Max", "Mad", "Myc"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train_fracs", nargs="+", type=float, default=[0.0625, 0.125, 0.25, 0.5, 1.0])
    args = parser.parse_args()

    cfg_pre = load_yaml("configs/pretrain.yaml")
    cfg_ft = load_yaml("configs/finetune_pbm.yaml")
    ckpt_path = cfg_ft["pretrained_ckpt"]

    pbm_path = f"data/raw/pbm/{args.dataset}.txt"
    alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

    results = []

    for seed in args.seeds:
        set_seed(seed)

        ds = PBMDataset(pbm_path, k=int(cfg_pre["tokenizer"]["k"]), seed=seed)
        idx_train, idx_val, idx_test = split_dataset(
            ds,
            train_frac=float(cfg_ft["train"]["split"]["train"]),
            val_frac=float(cfg_ft["train"]["split"]["val"]),
        )

        # ---- 1-mer ridge features ----
        Xtr_1mer = np.stack([onehot_1mer_positional(ds[i]["seq"]) for i in idx_train], axis=0)
        ytr_1mer = np.array([float(ds[i]["y"].item()) for i in idx_train], dtype=np.float32)
        Xva_1mer = np.stack([onehot_1mer_positional(ds[i]["seq"]) for i in idx_val], axis=0)
        yva_1mer = np.array([float(ds[i]["y"].item()) for i in idx_val], dtype=np.float32)
        Xte_1mer = np.stack([onehot_1mer_positional(ds[i]["seq"]) for i in idx_test], axis=0)
        yte_1mer = np.array([float(ds[i]["y"].item()) for i in idx_test], dtype=np.float32)

        # ---- transformer features ----
        model = build_pretrained_model(cfg_pre, ckpt_path)
        Xtr_h, ytr_h = extract_hidden_features(model, ds, idx_train, use_flex=False)
        Xva_h, yva_h = extract_hidden_features(model, ds, idx_val, use_flex=False)
        Xte_h, yte_h = extract_hidden_features(model, ds, idx_test, use_flex=False)

        Xtr_hf, ytr_hf = extract_hidden_features(model, ds, idx_train, use_flex=True)
        Xva_hf, yva_hf = extract_hidden_features(model, ds, idx_val, use_flex=True)
        Xte_hf, yte_hf = extract_hidden_features(model, ds, idx_test, use_flex=True)

        for frac in args.train_fracs:
            n_use = max(50, int(round(len(idx_train) * frac)))

            # subset train
            Xtr1 = Xtr_1mer[:n_use]
            ytr1 = ytr_1mer[:n_use]

            Xtrh = Xtr_h[:n_use]
            ytrh = ytr_h[:n_use]

            Xtrhf = Xtr_hf[:n_use]
            ytrhf = ytr_hf[:n_use]

            # standardize transformer features (important)
            Xtrh_s, Xvah_s, Xteh_s = standardize_train_apply(Xtrh, Xva_h, Xte_h)
            Xtrhf_s, Xvahf_s, Xtehf_s = standardize_train_apply(Xtrhf, Xva_hf, Xte_hf)

            # ridge baseline
            r2_1mer, alpha_1mer = fit_ridge_with_val(Xtr1, ytr1, Xva_1mer, yva_1mer, Xte_1mer, yte_1mer, alpha_grid)

            # hidden-only + ridge
            r2_hidden, alpha_hidden = fit_ridge_with_val(Xtrh_s, ytrh, Xvah_s, yva_h, Xteh_s, yte_h, alpha_grid)

            # hidden+flex + ridge
            r2_hiddenflex, alpha_hiddenflex = fit_ridge_with_val(Xtrhf_s, ytrhf, Xvahf_s, yva_hf, Xtehf_s, yte_hf, alpha_grid)

            results.append({
                "dataset": args.dataset,
                "seed": seed,
                "train_frac": frac,
                "n_train": n_use,
                "r2_1mer": r2_1mer,
                "r2_hidden": r2_hidden,
                "r2_hiddenflex": r2_hiddenflex,
                "alpha_1mer": alpha_1mer,
                "alpha_hidden": alpha_hidden,
                "alpha_hiddenflex": alpha_hiddenflex,
            })

            print(
                f"{args.dataset} | seed={seed} | n={n_use} | "
                f"1mer={r2_1mer:.3f} | hidden={r2_hidden:.3f} | hidden+flex={r2_hiddenflex:.3f}"
            )

    # save csv
    out_csv = f"plots/panelC_three_lines_{args.dataset}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # aggregate for plotting
    unique_n = sorted({r["n_train"] for r in results})
    mean_1mer, std_1mer = [], []
    mean_hidden, std_hidden = [], []
    mean_hf, std_hf = [], []

    for n in unique_n:
        rows_n = [r for r in results if r["n_train"] == n]
        v1 = np.array([r["r2_1mer"] for r in rows_n])
        v2 = np.array([r["r2_hidden"] for r in rows_n])
        v3 = np.array([r["r2_hiddenflex"] for r in rows_n])

        mean_1mer.append(v1.mean()); std_1mer.append(v1.std())
        mean_hidden.append(v2.mean()); std_hidden.append(v2.std())
        mean_hf.append(v3.mean()); std_hf.append(v3.std())

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=220)

    ax.errorbar(unique_n, mean_1mer, yerr=std_1mer, marker="o", capsize=4, linewidth=2, label="1-mer ridge")
    ax.errorbar(unique_n, mean_hidden, yerr=std_hidden, marker="o", capsize=4, linewidth=2, label="Transformer hidden + ridge")
    ax.errorbar(unique_n, mean_hf, yerr=std_hf, marker="o", capsize=4, linewidth=2, label="Transformer hidden+flex + ridge")

    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Sample size (train sequences)", fontsize=13)
    ax.set_ylabel("Test $R^2$", fontsize=13)
    ax.set_title(f"Panel C: Sample size vs $R^2$ ({args.dataset}, mean ± std across seeds)", fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_png = f"plots/panelC_three_lines_{args.dataset}.png"
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")

    print("Saved:", out_png)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
