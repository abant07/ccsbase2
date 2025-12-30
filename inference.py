import argparse
import sqlite3
import joblib
import numpy as np
import pandas as pd

from utils import Utils

# ---- CONFIG ----
DB_PATH = "CCSMLDatabase.db"     # used to fetch adduct list for one-hot encoding
MODEL_PATH = "ccsbase2.joblib"
DEFAULT_OUTPUT_CSV = "ccs_predictions.csv"
# ----------------


def load_adducts(database_file: str) -> list[str]:
    conn = sqlite3.connect(database_file)
    try:
        q = "SELECT DISTINCT adduct FROM master_clean"
        adducts = sorted(pd.read_sql_query(q, conn).to_numpy().tolist())
        return [a[0] for a in adducts]
    finally:
        conn.close()


def main(input_csv: str, output_csv: str):
    # Load model + utils
    model = joblib.load(MODEL_PATH)
    utils = Utils()

    # Adduct list must match training
    adducts = load_adducts(DB_PATH)

    # Read inputs
    df = pd.read_csv(input_csv)

    # Validate columns
    required = ["smi", "ionmass", "z", "adduct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    # Featurize
    X_list = []
    valid_idx = []

    for i, row in df.iterrows():
        feats = utils.calculate_descriptors(
            smiles=str(row["smi"]),
            ion_mass=float(row["ionmass"]),
            charge=int(row["z"]),
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
    out["CCS_Pred"] = np.nan
    out.loc[valid_idx, "CCS_Pred"] = preds

    out.to_csv(output_csv, index=False)

    print(f"Wrote: {output_csv}")
    print(f"Predicted rows: {len(valid_idx)} / {len(df)}")
    if len(valid_idx) != len(df):
        print("Some rows were skipped due to featurization errors (CCS_Pred = NaN).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict CCS values from a CSV file"
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV file (must include: smi, ionmass, z, adduct)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT_CSV})",
    )

    args = parser.parse_args()
    main(args.input_csv, args.output)
