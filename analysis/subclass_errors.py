import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap

# load data
df = pd.read_csv("../pretrained/ccs_predictions_test.csv")

# errors
df["_abs_err"] = (df["CCS_Pred"] - df["CCS_True"]).abs()
df["_rel_err"] = df["_abs_err"] / df["CCS_True"]

# aggregate by subclass
g = df.groupby("Subclass")
stats = g.agg(
    Count=("CCS_True", "size"),
    MAE=("_abs_err", "mean"),
    MDAE=("_abs_err", "median"),
    MRE=("_rel_err", "mean"),
    MDRE=("_rel_err", "median"),
    RMSE=("CCS_Pred", lambda x: np.sqrt(np.mean((x - df.loc[x.index, "CCS_True"]) ** 2))),
)

# top N by count
TOP_N = 10
stats = stats.sort_values("Count", ascending=False).head(TOP_N)

# truncate + legend mapping (keeps table order)
full_names = stats.index.tolist()
short_names = [(s[:28] + "…") if len(s) > 10 else s for s in full_names]

table_df = stats.copy()
table_df.insert(0, "Subclass", short_names)
table_df = table_df.reset_index(drop=True)

# format for display
table_df["Count"] = table_df["Count"].astype(int).astype(str)
for c in ["MAE", "MDAE", "RMSE"]:
    table_df[c] = table_df[c].map(lambda x: f"{x:.2f}")
for c in ["MRE", "MDRE"]:
    table_df[c] = table_df[c].map(lambda x: f"{x:.3f}")

# build figure with reduced height
fig, ax = plt.subplots(figsize=(16, 7.5))  # Reduced from 10 to 7.5
ax.axis("off")

tbl = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center",
    cellLoc="center",
    colLoc="center",
)

# improve table readability
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)  # Slightly increased for better readability in compact space
tbl.scale(1, 1.6)     # Reduced vertical scaling from 1.35 to 1.6 (wait, no: higher number = taller rows; adjust to 1.5 for compactness)

# style header + zebra stripes + borders
nrows = len(table_df) + 1
ncols = len(table_df.columns)

for (r, c), cell in tbl.get_celld().items():
    cell.set_linewidth(0.8)
    if r == 0:  # header
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#333333")
    else:
        cell.set_facecolor("#f2f2f2" if r % 2 == 0 else "white")

# left-align subclass column
for r in range(1, nrows):
    tbl[(r, 0)]._loc = "left"

# title - slightly lower and smaller
fig.suptitle("CCS Prediction Error Metrics by Subclass (Top 10 by Count)", y=0.95, fontsize=13, weight="bold")

# bottom legend
legend_lines = [f"{short} = {full}" for short, full in zip(short_names, full_names)]
legend_text = "  |  ".join(legend_lines)
legend_text = "\n".join(textwrap.wrap(legend_text, width=170))

fig.text(
    0.5,
    0.04,  # Raised slightly
    legend_text,
    ha="center",
    va="bottom",
    fontsize=8.5,
)

# tighter layout with less bottom margin
plt.tight_layout(rect=[0, 0.1, 1, 1.0])  # Reduced bottom from 0.08 to 0.06, top adjusted

# Save the figure
plt.savefig("ccs_error_metrics_table.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)