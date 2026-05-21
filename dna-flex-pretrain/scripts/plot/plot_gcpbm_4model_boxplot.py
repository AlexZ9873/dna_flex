import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_result_csv(path, model_label):
    df = pd.read_csv(path)

    dataset_col = find_first_existing(
        df.columns,
        ["file", "dataset", "tf", "name", "pbm_file"]
    )
    r2_col = find_first_existing(
        df.columns,
        ["test_r2", "r2", "cv_r2", "best_r2", "val_r2"]
    )

    if dataset_col is None:
        raise ValueError(
            f"Could not find dataset column in {path}. "
            f"Tried: file, dataset, tf, name, pbm_file"
        )

    if r2_col is None:
        raise ValueError(
            f"Could not find R2 column in {path}. "
            f"Tried: test_r2, r2, cv_r2, best_r2, val_r2"
        )

    out = df[[dataset_col, r2_col]].copy()
    out.columns = ["dataset", "r2"]
    out["model"] = model_label
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bend_only_csv", required=True)
    parser.add_argument("--bend_flex_csv", required=True)
    parser.add_argument("--flex_mlm_csv", required=True)
    parser.add_argument("--bend_flex_mlm_csv", required=True)
    parser.add_argument("--out_csv", default="plots/gcpbm_4model_merged.csv")
    parser.add_argument("--out_png", default="plots/gcpbm_4model_boxplot.png")
    args = parser.parse_args()

    order = [
        "bend only",
        "bend+flex",
        "flex+mlm",
        "bend+flex+mlm"
    ]

    df1 = load_result_csv(args.bend_only_csv, "bend only")
    df2 = load_result_csv(args.bend_flex_csv, "bend+flex")
    df3 = load_result_csv(args.flex_mlm_csv, "flex+mlm")
    df4 = load_result_csv(args.bend_flex_mlm_csv, "bend+flex+mlm")

    all_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    # Save merged long-format CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.out_csv, index=False)

    # Summary table
    summary = (
        all_df.groupby("model")["r2"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reindex(order)
    )

    print("\n=== gcPBM summary ===")
    print(summary.to_string())

    # Prepare data for plotting
    plot_data = []
    for model in order:
        vals = all_df.loc[all_df["model"] == model, "r2"].dropna().values
        plot_data.append(vals)

    # Box plot
    plt.figure(figsize=(9, 6))
    bp = plt.boxplot(
        plot_data,
        labels=order,
        showfliers=True,
        patch_artist=False
    )

    # Overlay individual points with slight jitter
    rng = np.random.default_rng(0)
    for i, vals in enumerate(plot_data, start=1):
        x = rng.normal(loc=i, scale=0.05, size=len(vals))
        plt.scatter(x, vals, alpha=0.6, s=18)

    # Add median text
    medians = [np.median(v) if len(v) > 0 else np.nan for v in plot_data]
    for i, m in enumerate(medians, start=1):
        if not np.isnan(m):
            plt.text(i, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    plt.ylabel("R²")
    plt.title("gcPBM benchmark: comparison of four pretraining strategies")
    plt.xticks(rotation=15)
    plt.tight_layout()

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=300, bbox_inches="tight")
    print("\nSaved merged CSV ->", args.out_csv)
    print("Saved box plot   ->", args.out_png)


if __name__ == "__main__":
    main()