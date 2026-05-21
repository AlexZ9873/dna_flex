import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 读已有结果表
csv_path = "plots/panelA1_ridge_vs_flexonly_rawR2.csv"

rows = []
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "ridge_test_r2": float(row["ridge_test_r2"]),
            "flex_test_r2": float(row["flex_test_r2"]),
        })

colors = {"Max": "#1f77b4", "Mad": "#ff7f0e", "Myc": "#2ca02c"}
markers = {0: "o", 1: "s", 2: "^"}

xs = [r["ridge_test_r2"] for r in rows]
ys = [r["flex_test_r2"] for r in rows]

lo = min(xs + ys) - 0.03
hi = max(xs + ys) + 0.03
lo = max(lo, 0.2)
hi = min(hi, 0.85)

fig, ax = plt.subplots(figsize=(6.6, 6.2), dpi=220)

# 对角线
ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="gray")

# 画点（不再写 seed 数字）
for r in rows:
    ax.scatter(
        r["ridge_test_r2"],
        r["flex_test_r2"],
        color=colors[r["dataset"]],
        marker=markers[r["seed"]],
        s=90,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("Test $R^2$ (1-mer ridge)", fontsize=13)
ax.set_ylabel("Test $R^2$ (flex-only)", fontsize=13)
ax.set_title("Panel A1: 1-mer ridge vs flex-only (raw test $R^2$)", fontsize=14)
ax.grid(True, alpha=0.25)

# dataset legend（放右上图外）
dataset_handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=colors[d], markeredgecolor='black',
           markersize=9, label=d)
    for d in ["Max", "Mad", "Myc"]
]

# seed legend（放右下图外）
seed_handles = [
    Line2D([0], [0], marker=markers[s], color='black',
           linestyle='None', markersize=9, label=f"seed {s}")
    for s in [0, 1, 2]
]

leg1 = ax.legend(
    handles=dataset_handles,
    title="dataset",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.00),
    borderaxespad=0.0,
    frameon=True
)
ax.add_artist(leg1)

ax.legend(
    handles=seed_handles,
    title="seed",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.55),
    borderaxespad=0.0,
    frameon=True
)

# 给右边 legend 留空间
fig.subplots_adjust(right=0.74)

out_png = "plots/panelA1_ridge_vs_flexonly_rawR2_clean.png"
fig.savefig(out_png, bbox_inches="tight")
print("Saved:", out_png)
