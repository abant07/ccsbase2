import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator

from db import Database
from constants import ADDUCT_OFFSETS


# ============ Train / Test Splitting ============

def _split_subclass_by_adduct_distribution(subclass_df, test_size, rng):
    count_matrix = subclass_df.groupby(["smi", "adduct"]).size().unstack(fill_value=0)

    smi_order = count_matrix.index.to_numpy().copy()
    rng.shuffle(smi_order)
    count_matrix = count_matrix.loc[smi_order]

    adducts = count_matrix.columns.to_numpy()
    group_adduct_counts = count_matrix.to_numpy(dtype=float)
    group_sizes = group_adduct_counts.sum(axis=1)

    target_props = subclass_df["adduct"].value_counts(normalize=True).reindex(adducts, fill_value=0).to_numpy()
    target_test_rows = round(test_size * len(subclass_df))

    n_groups = group_adduct_counts.shape[0]
    remaining_mask = np.ones(n_groups, dtype=bool)
    test_selected = np.zeros(n_groups, dtype=bool)
    test_adduct_counts = np.zeros(len(adducts))
    test_row_count = 0

    while remaining_mask.sum() > 1 and test_row_count < target_test_rows:
        remaining_idx = np.nonzero(remaining_mask)[0]
        candidate_combined = test_adduct_counts[None, :] + group_adduct_counts[remaining_idx]
        candidate_props = candidate_combined / candidate_combined.sum(axis=1, keepdims=True)
        distances = np.abs(candidate_props - target_props[None, :]).sum(axis=1)

        chosen_idx = remaining_idx[np.argmin(distances)]
        test_adduct_counts += group_adduct_counts[chosen_idx]
        test_row_count += group_sizes[chosen_idx]
        test_selected[chosen_idx] = True
        remaining_mask[chosen_idx] = False

    groups_by_smi = dict(tuple(subclass_df.groupby("smi")))
    train_groups = [groups_by_smi[smi_order[i]] for i in range(n_groups) if not test_selected[i]]
    test_groups = [groups_by_smi[smi_order[i]] for i in range(n_groups) if test_selected[i]]

    return train_groups, test_groups


def _split_group_by_column(group_df, column, test_size, rng):
    train_parts, test_parts = [], []
    for _, column_df in group_df.groupby(column):
        train_groups, test_groups = _split_subclass_by_adduct_distribution(column_df, test_size, rng)
        train_parts.extend(train_groups)
        test_parts.extend(test_groups)
    return train_parts, test_parts


def train_test_split_custom(
    database_file,
    test_size=0.2,
    random_state=26,
    use_metlin=True,
):
    db = Database(database_file)
    query = "SELECT smi, mass, z, ccs, name, superclass, class, subclass, adduct, tag FROM master_clean"
    if not use_metlin:
        query += " WHERE tag != 'METLIN'"

    df = db.read_df(query)

    for column in ("subclass", "class", "superclass"):
        df[column] = df[column].str.replace(r" \(predicted\)$", "", regex=True)

    rng = np.random.default_rng(random_state)

    has_subclass = df["subclass"].notna()
    has_class = df["class"].notna()
    has_superclass = df["superclass"].notna()

    subclass_tier = df[has_subclass]
    class_tier = df[~has_subclass & has_class]
    superclass_tier = df[~has_subclass & ~has_class & has_superclass]
    unclassified_tier = df[~has_subclass & ~has_class & ~has_superclass]

    train_parts, test_parts = [], []
    for tier_df, column in ((subclass_tier, "subclass"), (class_tier, "class"), (superclass_tier, "superclass")):
        tier_train, tier_test = _split_group_by_column(tier_df, column, test_size, rng)
        train_parts.extend(tier_train)
        test_parts.extend(tier_test)
        print(f"{len(tier_df)} rows split by {column}")

    train_parts.append(unclassified_tier)
    print(f"{len(unclassified_tier)} rows with no superclass/class/subclass -- added to training only")

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)

    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)

    print(len(train_df), "train rows")
    print(len(test_df), "test rows")


# ============ Hyperparameter Ranges ============

def normalize_hyperparam_range(value):
    return value if isinstance(value, tuple) else (value, value)


# ============ Molecular Feature Calculation ============

def calculate_base_features(smiles: str, ion_mass:float, adducts: list, adduct: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    molecular_weight = rdMolDescriptors.CalcExactMolWt(mol)
    adduct_mass = ion_mass - molecular_weight

    adduct_one_hot = [0] * (len(adducts) + 1)
    adduct_index = adducts.index(adduct) if adduct in adducts else len(adducts)
    adduct_one_hot[adduct_index] = 1

    return np.array([molecular_weight, adduct_mass] + adduct_one_hot, dtype=float)


def calculate_sparse_fingerprint(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # no fpSize -- atom environments keep their raw id instead of colliding into a folded bit
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True)
    sparse_fp = morgan_generator.GetSparseCountFingerprint(mol)
    return dict(sparse_fp.GetNonzeroElements())


def calculate_sparse_fingerprints(smiles_series) -> list:
    return [calculate_sparse_fingerprint(smi) for smi in smiles_series]


def to_rdkit_fingerprint(fp_dict: dict):
    fp = DataStructs.ULongSparseIntVect(2 ** 64 - 1)
    for env_id, count in fp_dict.items():
        fp[env_id] = count
    return fp


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
        base = calculate_base_features(row["smi"], row["mass"], adducts, row["adduct"])
        fp = calculate_sparse_fingerprint(row["smi"])
        if base is None or fp is None:
            continue

        base_features.append(base)
        fp_dicts.append(fp)
        ccs_values.append(row["ccs"])
        metadata.append((row["tag"], row["subclass"], row["adduct"], row["name"], row["smi"], row["z"]))

    return base_features, fp_dicts, np.array(ccs_values, dtype=float), metadata
