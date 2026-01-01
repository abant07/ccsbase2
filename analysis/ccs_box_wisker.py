import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "../CCSMLDatabase.db"
TABLE = "master_clean"

# load
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    f"SELECT tag, ccs FROM {TABLE}",
    conn,
)
conn.close()

# split comma-delimited tags and duplicate rows
df["tag"] = df["tag"].str.split(",")
df = df.explode("tag")
df["tag"] = df["tag"].str.strip()

# print statistics per dataset/tag
print("CCS statistics by dataset:")
for tag, g in df.groupby("tag"):
    print(
        f"{tag}: "
        f"n={len(g)}, "
        f"mean={g['ccs'].mean():.2f}, "
        f"median={g['ccs'].median():.2f}, "
        f"std={g['ccs'].std():.2f}"
    )

print()

# prepare data + counts for plotting
tags = sorted(df["tag"].unique().tolist())
data = []
labels = []

for t in tags:
    vals = df.loc[df["tag"] == t, "ccs"].values
    data.append(vals)
    labels.append(f"{t} ({len(vals)})")

# box + whisker plots
plt.figure(figsize=(max(8, 0.45 * len(tags)), 6))
plt.boxplot(data, labels=labels, showfliers=False)

plt.ylabel("CCS")
plt.xlabel("Dataset")

plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.show()
