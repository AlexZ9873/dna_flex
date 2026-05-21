import csv
import math
import random
from pathlib import Path
from collections import Counter

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def pearsonr_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}


# -------------------------
# Data loading
# -------------------------
def read_htselex_file(path, k=6, max_rows=None, seed=0):
    items = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            seq = parts[0].upper()
            try:
                score = float(parts[1])
            except ValueError:
                continue

            if len(seq) < k:
                continue

            items.append((seq, score))

    if len(items) == 0:
        raise ValueError(f"No usable rows in {path}")

    lengths = Counter(len(s) for s, _ in items)
    major_len = lengths.most_common(1)[0][0]
    items = [(s, y) for s, y in items if len(s) == major_len]

    if max_rows is not None and len(items) > max_rows:
        rng = np.random.RandomState(seed)
        keep = rng.choice(np.arange(len(items)), size=max_rows, replace=False)
        items = [items[i] for i in keep]

    return items, major_len


# -------------------------
# Feature encodings
# -------------------------
def positional_1mer_onehot(seq):
    seq = seq.upper()
    L = len(seq)
    X = np.zeros((L, 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = BASE_TO_ID.get(ch, None)
        if j is not None:
            X[i, j] = 1.0
    return X.reshape(-1)


def load_lookup_yaml(path):
    data = yaml.safe_load(open(path, "r").read())
    tri = data["trinucleotide"]
    di = data["dinucleotide"]

    tri_names = list(tri.keys())   # 4
    di_names = list(di.keys())     # 8

    tri_maps = {name: {k.upper(): float(v) for k, v in fmap.items()} for name, fmap in tri.items()}
    di_maps = {name: {k.upper(): float(v) for k, v in fmap.items()} for name, fmap in di.items()}

    return tri_names, tri_maps, di_names, di_maps


def flex12_track(seq, tri_names, tri_maps, di_names, di_maps):
    """
    Sequence-level 12-flex representation:
      4 tri features over (L-2) positions
      8 di features over (L-1) positions
    """
    seq = seq.upper()
    L = len(seq)
    vals = []

    for feat in tri_names:
        fmap = tri_maps[feat]
        for i in range(L - 2):
            word = seq[i:i+3]
            vals.append(float(fmap.get(word, 0.0)))

    for feat in di_names:
        fmap = di_maps[feat]
        for i in range(L - 1):
            word = seq[i:i+2]
            vals.append(float(fmap.get(word, 0.0)))

    return np.asarray(vals, dtype=np.float32)


def build_1mer12flex_matrix(items, tri_names, tri_maps, di_names, di_maps):
    X_list = []
    y_list = []

    for seq, score in items:
        one = positional_1mer_onehot(seq)  # 4L
        flex = flex12_track(seq, tri_names, tri_maps, di_names, di_maps)
        feat = np.concatenate([one, flex], axis=0)
        X_list.append(feat)
        y_list.append(score)

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


# -------------------------
# Stable preprocessing
# -------------------------
def preprocess_train_test(Xtr, Xte):
    """
    Numerically stable preprocessing for custom 1mer+12flex:
    1) remove zero-variance columns based on train
    2) z-score all remaining columns based on train
    """
    keep = Xtr.std(axis=0) > 1e-8
    Xtr = Xtr[:, keep]
    Xte = Xte[:, keep]

    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0

    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    return Xtr, Xte


# -------------------------
# Nested CV
# -------------------------
def choose_alpha_inner_cv(X, y, alphas, inner_folds=5, seed=0):
    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    mean_scores = []

    for alpha in alphas:
        scores = []
        for tr_idx, va_idx in kf.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            Xtr_p, Xva_p = preprocess_train_test(Xtr, Xva)

            reg = Ridge(
                alpha=alpha,
                fit_intercept=True,
                solver="lsqr",   # avoids SVD path
                random_state=0
            )
            reg.fit(Xtr_p, ytr)
            pred = reg.predict(Xva_p)
            scores.append(r2_score(yva, pred))

        mean_scores.append(np.mean(scores))

    best_i = int(np.argmax(mean_scores))
    return float(alphas[best_i])


def nested_cv_oof_r2_1mer12flex(X, y, outer_folds=10, inner_folds=5, seed=0):
    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=np.float32)

    # slightly stronger grid than before for numerical stability
    alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X)):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        alpha = choose_alpha_inner_cv(
            Xtr, ytr,
            alphas=alphas,
            inner_folds=inner_folds,
            seed=seed + fold_i,
        )

        Xtr_p, Xte_p = preprocess_train_test(Xtr, Xte)

        reg = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="lsqr",
            random_state=0
        )
        reg.fit(Xtr_p, ytr)
        oof[te_idx] = reg.predict(Xte_p)

    return {
        "r2": float(r2_score(y, oof)),
        "pearson": float(pearsonr_np(y, oof)),
        "rmse": float(np.sqrt(np.mean((y - oof) ** 2))),
    }


# -------------------------
# Plotting
# -------------------------
def detect_key(row, candidates):
    for c in candidates:
        if c in row:
            return c
    return None


def make_single_plot(rows, transformer_key, out_png):
    families = sorted(set(r["family"] for r in rows))
    cmap = plt.get_cmap("tab20")
    color_map = {fam: cmap(i % 20) for i, fam in enumerate(families)}

    xs = [float(r["cv_r2_1mer_12flex"]) for r in rows]
    ys = [float(r[transformer_key]) for r in rows]

    lo = min(xs + ys) - 0.05
    hi = max(xs + ys) + 0.05
    lo = max(lo, -0.2)
    hi = min(hi, 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 6.2), dpi=220)
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1.2)

    for r in rows:
        ax.scatter(
            float(r["cv_r2_1mer_12flex"]),
            float(r[transformer_key]),
            color=color_map[r["family"]],
            s=40,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("CV $R^2$ (1-mer + our 12-flex baseline)", fontsize=12)
    ax.set_ylabel("CV $R^2$ (Transformer hidden+flex + ridge)", fontsize=12)
    ax.set_title("Transformer vs 1-mer + our 12-flex features", fontsize=13)
    ax.grid(True, alpha=0.25)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color_map[fam], markeredgecolor="black",
               markersize=7, label=fam)
        for fam in families
    ]
    ax.legend(handles=handles, title="family", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)

    fig.subplots_adjust(right=0.78)
    fig.savefig(out_png, bbox_inches="tight")
    print("Saved:", out_png)


def make_full_2x2_panel(rows, transformer_key, out_png):
    families = sorted(set(r["family"] for r in rows))
    cmap = plt.get_cmap("tab20")
    color_map = {fam: cmap(i % 20) for i, fam in enumerate(families)}

    key_1mer = detect_key(rows[0], ["cv_r2_1mer", "r2_1mer"])
    key_2mer = detect_key(rows[0], ["cv_r2_2mer", "r2_2mer"])
    key_3mer = detect_key(rows[0], ["cv_r2_3mer", "r2_3mer"])

    comparisons = [
        (key_1mer, "1-mer ridge"),
        ("cv_r2_1mer_12flex", "1-mer + our 12-flex"),
        (key_2mer, "2-mer ridge"),
        (key_3mer, "3-mer ridge"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10), dpi=220)
    axes = axes.flatten()

    for ax, (xkey, xlabel) in zip(axes, comparisons):
        xs = [float(r[xkey]) for r in rows]
        ys = [float(r[transformer_key]) for r in rows]

        lo = min(xs + ys) - 0.05
        hi = max(xs + ys) + 0.05
        lo = max(lo, -0.2)
        hi = min(hi, 1.0)

        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1.2)

        for r in rows:
            ax.scatter(
                float(r[xkey]),
                float(r[transformer_key]),
                color=color_map[r["family"]],
                s=38,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.3,
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"CV $R^2$ ({xlabel})", fontsize=11)
        ax.set_ylabel("CV $R^2$ (Transformer hidden+flex + ridge)", fontsize=11)
        ax.set_title(f"Transformer vs {xlabel}", fontsize=12)
        ax.grid(True, alpha=0.25)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color_map[fam], markeredgecolor="black",
               markersize=7, label=fam)
        for fam in families
    ]

    fig.legend(handles=handles, title="family", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle("HT-SELEX transfer benchmark: transformer vs sequence/flex baselines", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(out_png, bbox_inches="tight")
    print("Saved:", out_png)


# -------------------------
# Main
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing_csv", required=True, help="Existing CSV with 1mer/2mer/3mer/transformer results")
    parser.add_argument("--folder", default="data/raw/htselex")
    parser.add_argument("--lookup_yaml", default="data/raw/flex_tables/lookup.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer_folds", type=int, default=10)
    parser.add_argument("--inner_folds", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--out_prefix", default="plots/htselex_option2_all215_seed0_plus12flex")
    args = parser.parse_args()

    set_seed(args.seed)

    tri_names, tri_maps, di_names, di_maps = load_lookup_yaml(args.lookup_yaml)

    # existing benchmark rows
    existing_rows = []
    with open(args.existing_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_rows.append(row)

    if len(existing_rows) == 0:
        raise SystemExit("Existing CSV is empty")

    transformer_key = detect_key(existing_rows[0], ["cv_r2_transformer_hiddenflex", "r2_transformer_hiddenflex"])
    if transformer_key is None:
        raise SystemExit("Could not find transformer R2 column in existing CSV")

    out_csv = args.out_prefix + ".csv"

    # resume support
    completed = {}
    if Path(out_csv).exists():
        with open(out_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "cv_r2_1mer_12flex" in row and row["cv_r2_1mer_12flex"] not in ("", "nan", "NaN"):
                    completed[row["file"]] = row
        print(f"Found existing partial output with {len(completed)} completed files")

    merged_rows = []

    for i, row in enumerate(existing_rows, 1):
        filename = row["file"]

        if filename in completed:
            merged_rows.append(completed[filename])
            print(f"[{i}/{len(existing_rows)}] SKIP completed {filename}")
            continue

        path = Path(args.folder) / filename
        items, major_len = read_htselex_file(path, k=6, max_rows=args.max_rows, seed=args.seed)

        X, y = build_1mer12flex_matrix(items, tri_names, tri_maps, di_names, di_maps)

        res = nested_cv_oof_r2_1mer12flex(
            X, y,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
        )

        new_row = dict(row)
        new_row["cv_r2_1mer_12flex"] = res["r2"]
        new_row["pearson_1mer_12flex"] = res["pearson"]
        new_row["rmse_1mer_12flex"] = res["rmse"]
        merged_rows.append(new_row)

        print(
            f"[{i}/{len(existing_rows)}] {filename} | "
            f"L={major_len} n={len(items)} | "
            f"1mer+12flex={res['r2']:.3f} | "
            f"transformer={float(row[transformer_key]):.3f}"
        )

        # save progress every file
        with open(out_csv, "w", newline="") as f:
            fieldnames = list(merged_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)

    print("Saved:", out_csv)

    make_single_plot(
        merged_rows,
        transformer_key=transformer_key,
        out_png=args.out_prefix + "_1mer12flex_vs_transformer.png",
    )

    make_full_2x2_panel(
        merged_rows,
        transformer_key=transformer_key,
        out_png=args.out_prefix + "_panelA_full.png",
    )


if __name__ == "__main__":
    main()
