import argparse
import sqlite3
import joblib
import numpy as np
import pandas as pd

from utils import Utils

# ---- CONFIG ----
DB_PATH = "CCSMLDatabase.db"
MODEL_PATH = "ccsbase2.joblib"
# ----------------

adduct_to_mass_charge = {
    "[M+H]+": (1.007825,1),
    "[M+Na]+": (22.989770,1),
    "[M-H]-": (-1.007825,-1),
    "[M+NH4]+": (18.034374,1),
    "[M+H-H2O]+": (-17.00274,1),
    "[M+K]+": (38.963707,1),
    "[M]+": (0.0,1),
    "[M+CH3COO]-": (59.013305,-1),
    "[M+HCOO]-": (44.997655,-1),
}


def load_adducts(database_file: str) -> list[str]:
    conn = sqlite3.connect(database_file)
    try:
        q = "SELECT DISTINCT adduct FROM master_clean"
        adducts = sorted(pd.read_sql_query(q, conn).to_numpy().tolist())
        return [a[0] for a in adducts]
    finally:
        conn.close()


def main(input_csv: str):
    # Load model + utils
    model = joblib.load(MODEL_PATH)
    utils = Utils()

    # Adduct list must match training
    adducts = load_adducts(DB_PATH)

    # Read inputs
    df = pd.read_csv(input_csv)
    output_csv = input_csv[:-4] + "_predictions.csv"

    # Validate columns
    required = ["smi", "adduct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    # Featurize
    X_list = []
    valid_idx = []

    for i, row in df.iterrows():
        if row["adduct"] not in adduct_to_mass_charge:
            continue

        feats = utils.calculate_descriptors(
            smiles=str(row["smi"]),
            ion_mass=adduct_to_mass_charge[row["adduct"]][0],
            charge=adduct_to_mass_charge[row["adduct"]][1],
            adducts=adducts,
            adduct=str(row["adduct"]),
        )

        if feats is not None:
            X_list.append(feats)
            valid_idx.append(i)

    if not X_list:
        raise RuntimeError(
            "No rows could be featurized. Check SMILES/adduct/instrument values."
        )

    X = np.asarray(X_list, dtype=float)

    # Predict
    preds = model.predict(X)

    # Write output (preserve all rows; failed rows get NaN)
    out = df.copy()
    out["Predicted CCS"] = np.nan
    out.loc[valid_idx, "Predicted CCS"] = preds

    out.to_csv(output_csv, index=False)

    print(f"Wrote: {output_csv}")
    print(f"Predicted rows: {len(valid_idx)} / {len(df)}")
    if len(valid_idx) != len(df):
        print("Some rows were skipped due to featurization errors (Predicted CCS = NaN).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict CCS values from a CSV file"
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV file (must include: smi, ionmass, z, adduct)",
    )

    args = parser.parse_args()
    main(args.input_csv)
