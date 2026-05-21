import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def pick_column(df, candidates, label_name):
    """
    Return the first matching column name from a list of possible candidates.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find a column for " + label_name + ". Tried: " + ", ".join(candidates)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Replot HT-SELEX benchmark results with cleaner titles/labels."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the summary CSV produced by your HT-SELEX benchmark script."
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default=None,
        help="Output PNG path. If omitted, saves next to the CSV with suffix _clean.png"
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: file {args.csv} not found.")
        return

    df = pd.read_csv(args.csv)

    # -----------------------------
    # Try to detect important columns
    # -----------------------------
    family_col = pick_column(
        df,
        ["family", "tf_family", "Family", "TF_family"],
        "family"
    )

    tf_col = pick_column(
        df,
        ["tf", "name", "dataset", "file", "tf_name", "protein", "TF"],
        "TF/dataset name"
    )

    col_1mer = pick_column(
        df,
        ["r2_1mer", "ridge_1mer_r2", "one_mer_r2", "r2_ridge_1mer"],
        "1-mer ridge R^2"
    )

    col_1mer12flex = pick_column(
        df,
        [
            "cv_r2_1mer_12flex",
            "r2_1mer12flex",
            "r2_1mer_plus_12flex",
            "mer12flex_r2",
            "ridge_1mer12flex_r2",
            "one_mer_12flex_r2"
        ],
        "1-mer + 12-flex R^2"
    )

    col_2mer = pick_column(
        df,
        ["r2_2mer", "ridge_2mer_r2", "two_mer_r2", "r2_ridge_2mer"],
        "2-mer ridge R^2"
    )

    col_3mer = pick_column(
        df,
        ["r2_3mer", "ridge_3mer_r2", "three_mer_r2", "r2_ridge_3mer"],
        "3-mer ridge R^2"
    )

    col_transformer = pick_column(
        df,
        [
            "r2_transformer_hiddenflex",
            "r2_transformer",
            "transformer_r2",
            "r2_hidden_plus_flex",
            "hidden_plus_flex_r2",
            "best_model_r2"
        ],
        "Transformer R^2"
    )

    # -----------------------------
    # Color map for TF families
    # -----------------------------
    families = sorted(df[family_col].dropna().unique().tolist())
    cmap = plt.get_cmap("tab20")
    color_map = {}
    for i, fam in enumerate(families):
        color_map[fam] = cmap(i % 20)

    # -----------------------------
    # Create figure
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Updated titles and y-labels
    panels = [
        {
            "ax": axes[0, 0],
            "xcol": col_1mer,
            "ycol": col_transformer,
            "title": "pre-train model vs 1-mer baseline",
            "xlabel": r"$R^2$ (1-mer baseline)",
            "ylabel": r"$R^2$ (pre-train model)"
        },
        {
            "ax": axes[0, 1],
            "xcol": col_1mer12flex,
            "ycol": col_transformer,
            "title": "pre-train model vs 1-mer + 12 flexibility features",
            "xlabel": r"$R^2$ (1-mer + 12-flex)",
            "ylabel": r"$R^2$ (pre-train model)"
        },
        {
            "ax": axes[1, 0],
            "xcol": col_2mer,
            "ycol": col_transformer,
            "title": "pre-train model vs 2-mer baseline",
            "xlabel": r"$R^2$ (2-mer baseline)",
            "ylabel": r"$R^2$ (pre-train model)"
        },
        {
            "ax": axes[1, 1],
            "xcol": col_3mer,
            "ycol": col_transformer,
            "title": "pre-train model vs 3-mer baseline",
            "xlabel": r"$R^2$ (3-mer baseline)",
            "ylabel": r"$R^2$ (pre-train model)"
        }
    ]

    for panel in panels:
        ax = panel["ax"]
        xcol = panel["xcol"]
        ycol = panel["ycol"]

        tmp = df[[family_col, tf_col, xcol, ycol]].dropna()

        # Scatter points
        for fam in families:
            sub = tmp[tmp[family_col] == fam]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub[xcol],
                sub[ycol],
                s=45,
                alpha=0.8,
                color=color_map[fam],
                edgecolors="k",
                linewidths=0.3
            )

        # Diagonal y=x line
        if not tmp.empty:
            xmin = min(tmp[xcol].min(), tmp[ycol].min())
            xmax = max(tmp[xcol].max(), tmp[ycol].max())
            padding = 0.03 * (xmax - xmin + 1e-9)
            lo = xmin - padding
            hi = xmax + padding

            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="gray")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        ax.set_title(panel["title"], fontsize=14)
        ax.set_xlabel(panel["xlabel"], fontsize=12)
        ax.set_ylabel(panel["ylabel"], fontsize=12)
        ax.grid(True, alpha=0.3)

    # -----------------------------
    # Family legend
    # -----------------------------
    legend_handles = []
    for fam in families:
        handle = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=fam,
            markerfacecolor=color_map[fam],
            markeredgecolor="k",
            markersize=8
        )
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        labels=families,
        title="TF family",
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        frameon=True
    )

    # -----------------------------
    # Overall title
    # -----------------------------
    fig.suptitle(
        "HT-SELEX transfer benchmark: pre-train model vs sequence / flexibility baselines",
        fontsize=18,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    # -----------------------------
    # Output path
    # -----------------------------
    if args.out_png is None:
        base, _ = os.path.splitext(args.csv)
        out_png = base + "_clean.png"
    else:
        out_png = args.out_png

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print("Saved figure to:", out_png)
    plt.show()


if __name__ == "__main__":
    main()