import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "../CCSMLDatabase.db"
TABLE = "master_clean"

# load adducts
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    f"SELECT adduct FROM {TABLE} WHERE adduct IS NOT NULL",
    conn,
)
conn.close()

# count adducts
counts = df["adduct"].value_counts()

# print counts
print("Adduct counts:")
print(counts)
print("\nTotal datapoints:", counts.sum())

# normalize to percent of total database
perc = (counts / counts.sum()) * 100

# bar chart
plt.figure(figsize=(max(8, 0.4 * len(perc)), 6))
plt.bar(perc.index, perc.values)

plt.ylabel("Percent of total datapoints (%)")
plt.xlabel("Adduct")

plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.show()
