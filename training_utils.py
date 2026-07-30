import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator

from db import Database


# ============ Train / Test Splitting ============

def train_test_split_custom(
    database_file,
    train_csv_path,
    test_csv_path,
    test_size=0.2,
    random_state=26,
    use_metlin=True,
    subclass_frequency_threshold=None,
):
    db = Database(database_file)
    query = "SELECT smi, mass, z, ccs, name, subclass, adduct, tag FROM master_clean WHERE ABS(z) = 1 AND subclass != 'NONE (predicted)'"
    if not use_metlin:
        query += " AND tag != 'METLIN'"

    df = db.read_df(query)

    # predicted-subclass rows carry a "(predicted)" suffix that real labels don't
    df["subclass"] = df["subclass"].str.replace(r" \(predicted\)$", "", regex=True)

    df["count"] = df.groupby(["subclass", "adduct"])["subclass"].transform("count")
    df_split = df
    if subclass_frequency_threshold:
        df_split = df[df["count"] >= subclass_frequency_threshold]

    train_parts = []
    test_parts = []

    for (_, _), group_df in df_split.groupby(["subclass", "adduct"]):
        group_df = group_df.copy()
        if len(group_df) < 2:
            train_parts.append(group_df)
            continue

        y = group_df["ccs"].values
        y_arr = y.astype(float)
        stratify_labels = (y_arr // 10).astype(int)
        vc = pd.Series(stratify_labels).value_counts()
        if vc.min() < 2:
            stratify_labels = None
        else:
            n_samples = len(y_arr)
            n_test = int(np.ceil(test_size * n_samples))
            n_classes = vc.size
            if n_test < n_classes:
                stratify_labels = None

        group_train, group_test = sk_train_test_split(
            group_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )

        train_parts.append(group_train)
        test_parts.append(group_test)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)

    train_df = train_df.drop(columns=["count"], errors="ignore")
    test_df = test_df.drop(columns=["count"], errors="ignore")

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(len(train_df), "train rows")
    print(len(test_df), "test rows")


# ============ Hyperparameter Ranges ============

def normalize_hyperparam_range(value):
    return value if isinstance(value, tuple) else (value, value)


# ============ Molecular Feature Calculation ============

def calculate_base_features(smiles: str, ion_mass: float, charge: int, adducts: list, adduct: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    molecular_weight = rdMolDescriptors.CalcExactMolWt(mol)
    adduct_mass = ion_mass - molecular_weight
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)

    adduct_one_hot = [0] * (len(adducts) + 1)
    adduct_index = adducts.index(adduct) if adduct in adducts else len(adducts)
    adduct_one_hot[adduct_index] = 1

    return np.array([molecular_weight, adduct_mass, charge, labute_asa] + adduct_one_hot, dtype=float)


def calculate_sparse_fingerprint(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # no fpSize -- atom environments keep their raw id instead of colliding into a folded bit
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True)
    sparse_fp = morgan_generator.GetSparseCountFingerprint(mol)
    return dict(sparse_fp.GetNonzeroElements())


def calculate_sparse_fingerprints(smiles_series) -> list:
    return [calculate_sparse_fingerprint(smi) for smi in smiles_series]


# ============ Fingerprint Vocabulary + Vectorization ============

def build_fingerprint_vocabulary(fp_dicts, min_molecule_count):
    doc_freq = {}
    for fp in fp_dicts:
        for env_id in fp:
            doc_freq[env_id] = doc_freq.get(env_id, 0) + 1

    return sorted(env_id for env_id, count in doc_freq.items() if count >= min_molecule_count)


def build_fingerprint_index(fp_vocab):
    return {env_id: column for column, env_id in enumerate(fp_vocab)}


def vectorize_sparse_fp(fp_dict: dict, fp_index: dict) -> np.ndarray:
    vector = np.zeros(len(fp_index), dtype=np.float32)
    for env_id, count in fp_dict.items():
        column = fp_index.get(env_id)
        if column is not None:
            vector[column] = count
    return vector


def vectorize_fingerprints(fp_dicts, fp_index) -> np.ndarray:
    return np.array([vectorize_sparse_fp(fp, fp_index) for fp in fp_dicts])


def build_feature_matrix(base_features, fp_dicts, fp_index):
    return np.hstack([
        np.array(base_features, dtype=float),
        vectorize_fingerprints(fp_dicts, fp_index),
    ])


# ============ CCS Regression Dataset Featurization ============

def featurize_ccs_dataset(df, adducts):
    base_features, fp_dicts, ccs_values, metadata = [], [], [], []
    for _, row in df.iterrows():
        base = calculate_base_features(row["smi"], row["mass"], row["z"], adducts, row["adduct"])
        fp = calculate_sparse_fingerprint(row["smi"])
        if base is None or fp is None:
            continue

        base_features.append(base)
        fp_dicts.append(fp)
        ccs_values.append(row["ccs"])
        metadata.append((row["tag"], row["subclass"], row["adduct"], row["name"], row["smi"], row["z"]))

    return base_features, fp_dicts, np.array(ccs_values, dtype=float), metadata
