import argparse
import sqlite3
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from utils import Utils


# ---- CONFIG ----
DB_PATH = "CCSMLDatabase.db"
MODEL_PATH = "ccsbase2.joblib"
# ----------------

ATOMIC_MASSES = {
    'H': 1.007825,
    'Li': 7.016004,
    'C': 12.000000,
    'N': 14.003074,
    'O': 15.994915,
    'F': 18.998403,
    'Na': 22.989770,
    'S': 31.972071,
    'Cl': 34.968853,
    'K': 38.963707,
    'Br': 78.918338,
    'Rb': 84.911789,
    'Cs': 132.905452,
}

H2O = 2 * ATOMIC_MASSES['H'] + ATOMIC_MASSES['O']
NH3 = ATOMIC_MASSES['N'] + 3 * ATOMIC_MASSES['H']
CO2 = ATOMIC_MASSES['C'] + 2 * ATOMIC_MASSES['O']
SO3 = ATOMIC_MASSES['S'] + 3 * ATOMIC_MASSES['O']
HCOO = ATOMIC_MASSES['H'] + ATOMIC_MASSES['C'] + 2 * ATOMIC_MASSES['O']
CH3COO = 2 * ATOMIC_MASSES['C'] + 3 * ATOMIC_MASSES['H'] + 2 * ATOMIC_MASSES['O']
ClO = ATOMIC_MASSES['Cl'] + ATOMIC_MASSES['O']
BrO = ATOMIC_MASSES['Br'] + ATOMIC_MASSES['O']

HF = ATOMIC_MASSES['H'] + ATOMIC_MASSES['F']
CH3COOH = CH3COO + ATOMIC_MASSES['H']  # Acetic Acid
C6H8O6 = 6 * ATOMIC_MASSES['C'] + 8 * ATOMIC_MASSES['H'] + 6 * ATOMIC_MASSES['O'] # Ascorbic Acid

adduct_to_mass_charge = {
    # --- Positive Mode Adducts ---
    '[M+H]+': (ATOMIC_MASSES['H'], +1),
    '[M+Na]+': (ATOMIC_MASSES['Na'], +1),
    '[M+K]+': (ATOMIC_MASSES['K'], +1),
    '[M+Li]+': (ATOMIC_MASSES['Li'], +1),
    '[M+Rb]+': (ATOMIC_MASSES['Rb'], +1),
    '[M+Cs]+': (ATOMIC_MASSES['Cs'], +1),
    '[M+NH4]+': (ATOMIC_MASSES['N'] + 4 * ATOMIC_MASSES['H'], +1),
    '[M]+': (0.0, +1),
    '[M]-': (0.0, -1),
    '[M+dot]+': (0.000549, +1),
    '[M-Br]+': (-ATOMIC_MASSES['Br'], +1),
    '[M-Cl]+': (-ATOMIC_MASSES['Cl'], +1),
    '[M-Na+2H]+': (-ATOMIC_MASSES['Na'] + 2 * ATOMIC_MASSES['H'], +1),

    # Water/Ammonia Losses (Positive)
    '[M+H-H2O]+': (ATOMIC_MASSES['H'] - H2O, +1),
    '[M+Na-H2O]+': (ATOMIC_MASSES['Na'] - H2O, +1),
    '[M+H-2H2O]+': (ATOMIC_MASSES['H'] - 2 * H2O, +1),
    '[M+Na-2H2O]+': (ATOMIC_MASSES['Na'] - 2 * H2O, +1),
    '[M-3H2O+H]+': (-3 * H2O + ATOMIC_MASSES['H'], +1),
    '[M+H-NH3]+': (ATOMIC_MASSES['H'] - NH3, +1),

    # Sodium/Potassium Exchanges (Positive)
    '[M+Na-H]+': (ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'], +1),
    '[M+2Na-H]+': (2 * ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'], +1),
    '[M-2H+3Na]+': (-2 * ATOMIC_MASSES['H'] + 3 * ATOMIC_MASSES['Na'], +1),
    '[M-H+2K]+': (-ATOMIC_MASSES['H'] + 2 * ATOMIC_MASSES['K'], +1),

    # SO3 / Sulfonate Adducts (Positive)
    '[M-SO3-H2O+H]+': (-SO3 - H2O + ATOMIC_MASSES['H'], +1),
    '[M-SO3-2H2O+H]+': (-SO3 - 2 * H2O + ATOMIC_MASSES['H'], +1),
    '[M-SO3-3H2O+H]+': (-SO3 - 3 * H2O + ATOMIC_MASSES['H'], +1),
    '[M-2SO3-2H2O+H]+': (-2 * SO3 - 2 * H2O + ATOMIC_MASSES['H'], +1),
    '[M-SO3+H]+': (-SO3 + ATOMIC_MASSES['H'], +1),

    # Complex Losses/Gains (Positive)
    '[M-HF-H2O+H]+': (-HF - H2O + ATOMIC_MASSES['H'], +1),
    '[M-HF+H]+': (-HF + ATOMIC_MASSES['H'], +1),
    '[M-CH3COOH-H2O+H]+': (-CH3COOH - H2O + ATOMIC_MASSES['H'], +1),
    '[M-CH3COOH+H]+': (-CH3COOH + ATOMIC_MASSES['H'], +1),
    '[M-C6H8O6-2H2O+H]+': (-C6H8O6 - 2 * H2O + ATOMIC_MASSES['H'], +1),
    '[M-C6H8O6-H2O+H]+': (-C6H8O6 - H2O + ATOMIC_MASSES['H'], +1),

    # --- Negative Mode Adducts ---
    '[M-H]-': (-ATOMIC_MASSES['H'], -1),
    '[M-3H]3-': (-3 * ATOMIC_MASSES['H'], -3),

    # Water/CO2 Losses (Negative)
    '[M-H-H2O]-': (-ATOMIC_MASSES['H'] - H2O, -1),
    '[M+H2O-H]-': (H2O - ATOMIC_MASSES['H'], -1),
    '[M-H-CO2]-': (-ATOMIC_MASSES['H'] - CO2, -1),

    # Salt Adducts (Negative)
    '[M+Na-2H]-': (ATOMIC_MASSES['Na'] - 2 * ATOMIC_MASSES['H'], -1),
    '[M+K-2H]-': (ATOMIC_MASSES['K'] - 2 * ATOMIC_MASSES['H'], -1),
    '[M+2Na-3H]-': (2 * ATOMIC_MASSES['Na'] - 3 * ATOMIC_MASSES['H'], -1),
    '[M+Cl]-': (ATOMIC_MASSES['Cl'], -1),
    '[M+K-H+Cl]-': (ATOMIC_MASSES['K'] - ATOMIC_MASSES['H'] + ATOMIC_MASSES['Cl'], -1),
    '[M+Na-H+Cl]-': (ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'] + ATOMIC_MASSES['Cl'], -1),

    # Formate/Acetate/Organic Adducts (Negative)
    '[M+CH3COO]-': (CH3COO, -1),
    '[M+HCOO]-': (HCOO, -1),
    '[M+K-H+HCOO]-': (ATOMIC_MASSES['K'] - ATOMIC_MASSES['H'] + HCOO, -1),
    '[M+Na-H+HCOO]-': (ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'] + HCOO, -1),
    '[M-H2O+HCOO]-': (-H2O + HCOO, -1),
    '[M+CH3COONa-H]-': (CH3COO + ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'], -1),
    '[M-H+HCOOH]-': (-ATOMIC_MASSES['H'] + (HCOO + ATOMIC_MASSES['H']), -1),

    # SO3 / Sulfonate Adducts (Negative)
    '[M-SO3-H]-': (-SO3 - ATOMIC_MASSES['H'], -1),
    '[M-SO3-H2O-H]-': (-SO3 - H2O - ATOMIC_MASSES['H'], -1),
    '[M-SO3-H2O+HCOO]-': (-SO3 - H2O + HCOO, -1),
    '[M-SO3-H2O+Cl]-': (-SO3 - H2O + ATOMIC_MASSES['Cl'], -1),
    '[M-SO3+Cl]-': (-SO3 + ATOMIC_MASSES['Cl'], -1),

    # Halogen/Oxide Specifics
    '[M-BrO]+': (-BrO, +1),
    '[M-ClO]+': (-ClO, +1),
    '[M-Br+O]-': (-ATOMIC_MASSES['Br'] + ATOMIC_MASSES['O'], -1),
    '[M-Cl+O]-': (-ATOMIC_MASSES['Cl'] + ATOMIC_MASSES['O'], -1),

    # --- Multi-charged Ions ---
    '[M+2H]2+': (2 * ATOMIC_MASSES['H'], +2),
    '[M+3H]3+': (3 * ATOMIC_MASSES['H'], +3),
    '[M+4H]4+': (4 * ATOMIC_MASSES['H'], +4),
    '[M-2H]2-': (-2 * ATOMIC_MASSES['H'], -2),
    '[M+2K]2+': (2 * ATOMIC_MASSES['K'], +2),
    '[M-2Cl]2+': (-2 * ATOMIC_MASSES['Cl'], +2),
}


def load_adducts(database_file: str) -> list[str]:
    conn = sqlite3.connect(database_file)
    try:
        q = "SELECT adduct FROM master_clean GROUP BY adduct HAVING COUNT(*) >= 100 ORDER BY adduct"
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

        adduct_mass = adduct_to_mass_charge[row["adduct"]][0]
        mol = Chem.MolFromSmiles(row["smi"])
        if mol is None:
            continue

        mol = Chem.AddHs(mol)
        mass = rdMolDescriptors.CalcExactMolWt(mol) + adduct_mass

        feats = utils.calculate_descriptors(
            smiles=str(row["smi"]),
            ion_mass=mass,
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
    out["pred_ccs"] = np.nan
    out.loc[valid_idx, "pred_ccs"] = preds

    out.to_csv(output_csv, index=False)

    print(f"Wrote: {output_csv}")
    print(f"Predicted rows: {len(valid_idx)} / {len(df)}")
    if len(valid_idx) != len(df):
        print("Some rows were skipped because of adducts (pred_ccs = NaN).")


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
