import argparse

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from constants import ADDUCT_OFFSETS
from db import Database
from utils import calculate_charge
from training_utils import calculate_base_features, calculate_sparse_fingerprint, vectorize_sparse_fp

DB_PATH = "CCSMLDatabase.db"
MODEL_PATH = "ccsbase2.joblib"
FP_VOCAB_PATH = "ccsbase2_fp_vocab.joblib"


def load_known_adducts(database_file: str) -> list:
    query = "SELECT adduct FROM master_clean GROUP BY adduct HAVING COUNT(*) >= 100 ORDER BY adduct"
    rows = Database(database_file).read(query)
    return sorted(row[0] for row in rows)


def main(input_csv: str):
    model = joblib.load(MODEL_PATH)
    fp_vocab = joblib.load(FP_VOCAB_PATH)
    fp_index = {env_id: column for column, env_id in enumerate(fp_vocab)}

    known_adducts = load_known_adducts(DB_PATH)

    df = pd.read_csv(input_csv)
    output_csv = input_csv[:-4] + "_predictions.csv"

    required_columns = ["smi", "adduct"]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {input_csv}: {missing_columns}")

    feature_rows, valid_idx = [], []
    for i, row in df.iterrows():
        smiles, adduct = str(row["smi"]), str(row["adduct"])

        if adduct not in ADDUCT_OFFSETS:
            print("Skipping", smiles, adduct, "Adduct not supported")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        mol = Chem.AddHs(mol)
        ion_mass = rdMolDescriptors.CalcExactMolWt(mol) + ADDUCT_OFFSETS[adduct]
        charge = calculate_charge(adduct)

        base = calculate_base_features(smiles, ion_mass, charge, known_adducts, adduct)
        fp = calculate_sparse_fingerprint(smiles)
        if base is None or fp is None:
            continue

        feature_rows.append(np.concatenate([base, vectorize_sparse_fp(fp, fp_index)]))
        valid_idx.append(i)

    if not feature_rows:
        raise RuntimeError("No rows could be featurized. Check SMILES/adduct values.")

    X = np.asarray(feature_rows, dtype=float)
    predictions = model.predict(X)

    out = df.copy()
    out["pred_ccs"] = np.nan
    out.loc[valid_idx, "pred_ccs"] = predictions
    out.to_csv(output_csv, index=False)

    print(f"Wrote: {output_csv}")
    print(f"Predicted rows: {len(valid_idx)} / {len(df)}")
    if len(valid_idx) != len(df):
        print("Some rows were skipped because of adducts (pred_ccs = NaN).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CCS values from a CSV file")
    parser.add_argument("input_csv", help="Input CSV file (must include: smi, adduct)")
    args = parser.parse_args()
    main(args.input_csv)
