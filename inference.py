import sqlite3
import joblib
import numpy as np
import pandas as pd

from utils import Utils

# ---- CONFIG ----
DB_PATH = "CCSMLDatabase.db"             # used to fetch adduct list for one-hot encoding
MODEL_PATH = ""                          # trained model file
INPUT_CSV = ""                           # must include: smi,ionmass,z,instrument,adduct
OUTPUT_CSV = "ccs_predictions.csv"
# ----------------


def load_adducts(database_file: str) -> list[str]:
    conn = sqlite3.connect(database_file)
    try:
        q = "SELECT DISTINCT adduct FROM master_clean"
        adducts = sorted(pd.read_sql_query(q, conn).to_numpy().tolist())
        return [a[0] for a in adducts]
    finally:
        conn.close()


# Load model + utils
model = joblib.load(MODEL_PATH)
utils = Utils()

# Adduct list must match training (same DB table used in training code)
adducts = load_adducts(DB_PATH)

# Read inputs
df = pd.read_csv(INPUT_CSV)

# Validate columns
required = ["smi", "ionmass", "z", "instrument", "adduct"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")

# Featurize
X_list = []
valid_idx = []

for i, row in df.iterrows():
    feats = utils.calculate_descriptors(
        smiles=str(row["smi"]),
        ion_mass=float(row["ionmass"]),
        charge=int(row["z"]),
        instrument=str(row["instrument"]),
        adducts=adducts,
        adduct=str(row["adduct"]),
    )
    
    if feats is not None:
        X_list.append(feats)
        valid_idx.append(i)

if not X_list:
    raise RuntimeError("No rows could be featurized. Check SMILES/adduct/instrument values.")

X = np.asarray(X_list, dtype=float)

# Predict
preds = model.predict(X)

# Write output (preserve all rows; failed rows get NaN)
out = df.copy()
out["CCS_Pred"] = np.nan
out.loc[valid_idx, "CCS_Pred"] = preds

out.to_csv(OUTPUT_CSV, index=False)

print(f"Wrote: {OUTPUT_CSV}")
print(f"Predicted rows: {len(valid_idx)} / {len(df)}")
if len(valid_idx) != len(df):
    print("Some rows were skipped due to featurization errors (CCS_Pred = NaN).")
