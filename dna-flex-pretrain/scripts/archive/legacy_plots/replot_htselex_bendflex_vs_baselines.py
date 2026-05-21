import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def resolve_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            "Could not find any of these columns in CSV:\n"
            + ", ".join(candidates)
            + "\n\nAvailable columns are:\n"
            + ", ".join(df.columns)
        )
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input benchmark CSV")
    parser.add_argument("--out_png", type=str, required=True, help="Output PNG file")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # --- Resolve columns robustly ---
    family_col = resolve_col(df, ["family"], required=False)

    col_1mer = resolve_col(df, [
        "cv_r2_1mer",
        "r2_1mer",
        "r2_1mer_ridge",
        "r2_ridge_1mer",
    ])

    col_1mer12flex = resolve_col(df, [
        "cv_r2_1mer_12flex",
        "cv_r2_1mer12flex",
        "r2_1mer_12flex",
        "r2_1mer12flex",
        "r2_1mer_plus_12flex",
        "r2_1mer_flex12",
    ])

    col_2mer = resolve_col(df, [
        "cv_r2_2mer",
        "r2_2mer",
        "r2_2mer_ridge",
        "r2_ridge_2mer",
    ])

    col_3mer = resolve_col(df, [
        "cv_r2_3mer",
        "r2_3mer",
        "r2_3mer_ridge",
        "r2_ridge_3mer",
    ])

    # y-axis = pre-trained model with bend+flex checkpoint
    col_model = resolve_col(df, [
        "cv_r2_transformer_hiddenflex",
        "r2_transformer_hiddenflex",
        "cv_r2_pretrained_hiddenflex",
        "r2_pretrained_hiddenflex",
        "cv_r2_hiddenflex",
        "r2_hiddenflex",
        "cv_r2_bendflex",
        "r2_bendflex",
    ])

    needed = [col_1mer, col_1mer12flex, col_2mer, col_3mer, col_model]
    if family_col is not None:
        needed.append(family_col)

    plot_df = df.dropna(subset=needed).copy()

    print("[INFO] rows used:", len(plot_df))
    print("[INFO] x columns:")
    print("  1-mer       =", col_1mer)
    print("  1-mer+12flex=", col_1mer12flex)
    print("  2-mer       =", col_2mer)
    print("  3-mer       =", col_3mer)
    print("[INFO] y column =", col_model)

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    comparisons = [
        (col_1mer,       "Pre-trained model vs 1-mer ridge",     r"$R^2$ (1-mer ridge)"),
        (col_1mer12flex, "Pre-trained model vs 1-mer + 12-flex", r"$R^2$ (1-mer + 12-flex)"),
        (col_2mer,       "Pre-trained model vs 2-mer ridge",     r"$R^2$ (2-mer ridge)"),
        (col_3mer,       "Pre-trained model vs 3-mer ridge",     r"$R^2$ (3-mer ridge)"),
    ]

    y_label = r"$R^2$ (pre-trained model + bend + flex)"

    if family_col is not None:
        families = sorted(plot_df[family_col].astype(str).unique())
        cmap = plt.get_cmap("tab20")
        color_map = {fam: cmap(i % 20) for i, fam in enumerate(families)}
        legend_handles = {}
    else:
        families = []
        color_map = {}
        legend_handles = {}

    all_vals = []
    for xcol, _, _ in comparisons:
        all_vals.extend(plot_df[xcol].tolist())
    all_vals.extend(plot_df[col_model].tolist())

    min_val = float(np.nanmin(all_vals))
    max_val = float(np.nanmax(all_vals))
    pad = 0.03
    lo = max(0.0, min_val - pad)
    hi = min(1.0, max_val + pad)

    for ax, (xcol, title, xlab) in zip(axes, comparisons):
        if family_col is not None:
            for fam in families:
                sub = plot_df[plot_df[family_col].astype(str) == fam]
                sc = ax.scatter(
                    sub[xcol],
                    sub[col_model],
                    s=42,
                    alpha=0.8,
                    color=color_map[fam],
                    edgecolors="k",
                    linewidths=0.3,
                    label=fam,
                )
                if fam not in legend_handles:
                    legend_handles[fam] = sc
        else:
            ax.scatter(
                plot_df[xcol],
                plot_df[col_model],
                s=42,
                alpha=0.8,
                edgecolors="k",
                linewidths=0.3,
            )

        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_title(title, fontsize=15)
        ax.set_xlabel(xlab, fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "HT-SELEX transfer benchmark: pre-trained model vs sequence/flex baselines",
        fontsize=18,
        y=0.98,
    )

    if family_col is not None and len(legend_handles) > 0:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            title="family",
            loc="center left",
            bbox_to_anchor=(0.98, 0.5),
            frameon=True,
            fontsize=10,
            title_fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 0.86, 0.95])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print("[INFO] Saved figure to:", out_png)


if __name__ == "__main__":
    main()