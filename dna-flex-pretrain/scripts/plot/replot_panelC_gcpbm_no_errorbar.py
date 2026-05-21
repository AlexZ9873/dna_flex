import argparse
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = [
    "1-mer ridge",
    "1-mer + 12-flex",
    "2-mer ridge",
    "3-mer ridge",
    "Transformer hidden + ridge",
    "Transformer hidden + flex + ridge",
    "bend only",
    "bend+flex",
    "flex+MLM",
    "bend+flex+MLM",
]

DISPLAY_NAME = {
    "1-mer ridge": "1-mer",
    "1-mer + 12-flex": "1-mer + 12-flex",
    "2-mer ridge": "2-mer",
    "3-mer ridge": "3-mer",
    "Transformer hidden + ridge": "pre-trained hidden",
    "Transformer hidden + flex + ridge": "pre-trained hidden+flex",
    "bend only": "bend only",
    "bend+flex": "bend+flex",
    "flex+MLM": "flex+MLM",
    "bend+flex+MLM": "bend+flex+MLM",
}


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="plots/panelC_gcpbm_sample_efficiency.csv",
        help="Input CSV from the sample-efficiency benchmark."
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="plots/panelC_gcpbm_sample_efficiency_noerr.png",
        help="Output PNG path."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Try to detect important columns
    model_col = find_column(df, ["model", "method"])
    pct_col = find_column(df, ["pct_train", "pct", "train_pct"])
    mean_col = find_column(df, ["mean_test_r2", "mean_r2", "r2_mean"])
    raw_r2_col = find_column(df, ["test_r2", "r2"])

    if model_col is None:
        raise ValueError("Could not find model column. Expected one of: model, method")
    if pct_col is None:
        raise ValueError("Could not find percentage column. Expected one of: pct_train, pct, train_pct")

    # Case 1: CSV already has aggregated mean values
    if mean_col is not None:
        plot_df = df[[model_col, pct_col, mean_col]].copy()
        plot_df = plot_df.rename(columns={
            model_col: "model",
            pct_col: "pct_train",
            mean_col: "mean_r2"
        })

    # Case 2: CSV has raw runs; compute mean across seeds/datasets
    elif raw_r2_col is not None:
        tmp = df[[model_col, pct_col, raw_r2_col]].copy()
        tmp = tmp.rename(columns={
            model_col: "model",
            pct_col: "pct_train",
            raw_r2_col: "test_r2"
        })
        plot_df = (
            tmp.groupby(["model", "pct_train"], as_index=False)["test_r2"]
            .mean()
            .rename(columns={"test_r2": "mean_r2"})
        )
    else:
        raise ValueError(
            "Could not find either aggregated mean column "
            "(mean_test_r2 / mean_r2 / r2_mean) or raw test R2 column (test_r2 / r2)."
        )

    # Keep only models that appear
    present_models = [m for m in MODEL_ORDER if m in plot_df["model"].unique()]

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    for model in present_models:
        sub = plot_df[plot_df["model"] == model].copy()
        sub = sub.sort_values("pct_train")

        ax.plot(
            sub["pct_train"],
            sub["mean_r2"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=DISPLAY_NAME.get(model, model)
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training data used (%)", fontsize=14)
    ax.set_ylabel("Test R$^2$", fontsize=14)
    ax.set_title("gcPBM sample-efficiency", fontsize=18)

    # keep only these ticks if they exist in your plot
    xticks = [0.3, 1, 3, 10, 30, 100]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0.3", "1", "3", "10", "30", "100"], fontsize=12)

    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300, bbox_inches="tight")
    print("Saved plot ->", args.out_png)


if __name__ == "__main__":
    main()