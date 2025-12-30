import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("../ccs_predictions_test.csv")

# categorical colors by tag
tags = df["Tag"].astype("category")
tag_names = tags.cat.categories

# more vibrant colormap
cmap = plt.cm.tab10  # higher contrast than tab20

plt.figure(figsize=(7, 7))

# plot per tag so legend is correct
for i, tag in enumerate(tag_names):
    m = tags == tag
    plt.scatter(
        df.loc[m, "CCS_True"],
        df.loc[m, "CCS_Pred"],
        color=cmap(i % cmap.N),
        alpha=0.45,
        s=22,
        linewidths=0,
        label=tag,
    )

# y = x reference line
min_val = min(df["CCS_True"].min(), df["CCS_Pred"].min())
max_val = max(df["CCS_True"].max(), df["CCS_Pred"].max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

plt.xlabel("CCS True")
plt.ylabel("CCS Predicted")
plt.title("CCS True vs CCS Predicted (colored by Tag)")

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=4,
    fontsize=8,
    frameon=False,
)

plt.tight_layout()
plt.show()
