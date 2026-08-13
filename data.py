import pandas as pd
import numpy as np

from constants import ADDUCT_OFFSETS, ADDUCT_STANDARDIZATION
from db import Database
from apis import (
    get_pubchem_smiles_by_cid,
    get_pubchem_smiles_by_name,
    get_classyfire_classification,
    get_cid_inchikey_from_smiles,
)
from utils import calculate_charge, desalt_and_mass
from constants import ADDUCT_STANDARDIZATION

import sqlite3
import h5py

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.SaltRemover import SaltRemover

INSERT_MASTER_SQL = """INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
INSERT_MASTER_NODIMER_SQL = """INSERT INTO master_nodimer(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
NEAR_DUPLICATE_THRESHOLD = 0.70
STANDARD_COLS = ["SMILES", "CCS", "Adduct"]

class CCSDataIntegration:
    def __init__(self, db_filename: str):
        self.db_filename = db_filename
        self.db = Database(db_filename)

        self.db.write(
            "CREATE TABLE IF NOT EXISTS master("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "tag TEXT, name TEXT, pubchemId INTEGER, "
            "adduct TEXT, mass REAL, z INTEGER, "
            "ccs REAL, smi TEXT, inchikey TEXT, "
            "superclass TEXT, class TEXT, subclass TEXT)"
        )

        self.db.write(
            "CREATE TABLE IF NOT EXISTS master_nodimer("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "tag TEXT, name TEXT, pubchemId INTEGER, "
            "adduct TEXT, mass REAL, z INTEGER, "
            "ccs REAL, smi TEXT, inchikey TEXT, "
            "superclass TEXT, class TEXT, subclass TEXT)"
        )

    def add_ccsbase(self):
        print("ADDING CCSBASE")
        c3s = Database("./datasets/CCSbase.db")
        records = c3s.read("SELECT * FROM master WHERE smi IS NOT NULL")

        all_rows, nodimer_rows = [], []
        for record in records:
            name = record[1]
            adduct = ADDUCT_STANDARDIZATION.get(record[2], record[2])
            mass = record[3]
            ccs = record[6]
            smiles = record[7]
            src_tag = record[8]
            superclass = record[11]
            class_ = record[12]
            subclass = record[13]

            z = calculate_charge(adduct)
            is_dimer = "[2M" in adduct

            if is_dimer:
                row = ("CCSBASE", name, None, adduct, mass, z, ccs, None, None, superclass, class_, subclass)
            else:
                desalted_smiles, isotopic_mass = desalt_and_mass(smiles)
                weight_diff = abs(float(mass) - (isotopic_mass + ADDUCT_OFFSETS[adduct]))
                tolerance = max(1, 0.01 * isotopic_mass)
                is_reliable = src_tag != "nguyen25" and weight_diff <= tolerance
                row = (
                    "CCSBASE", name, None, adduct, mass, z, ccs,
                    desalted_smiles if is_reliable else None, None,
                    superclass, class_, subclass,
                )
                nodimer_rows.append(row)

            all_rows.append(row)

        self.db.write_many(INSERT_MASTER_SQL, all_rows)
        self.db.write_many(INSERT_MASTER_NODIMER_SQL, nodimer_rows)

    def add_allccs(self):
        print("ADDING ALLCCS")
        allccs = pd.read_csv("./datasets/allccs.csv")

        all_rows, nodimer_rows = [], []
        for _, row in allccs.iterrows():
            has_required_fields = (
                pd.notna(row["m/z"]) and pd.notna(row["Adduct"])
                and pd.notna(row["CCS"]) and pd.notna(row["Name"])
                and row["Confidence level"] == "1"
            )
            if not has_required_fields:
                continue

            adduct = ADDUCT_STANDARDIZATION.get(row["Adduct"], row["Adduct"])
            z = calculate_charge(adduct)
            record = ("ALLCCS", row["Name"], None, adduct, row["m/z"], z, row["CCS"], None, None, None, None, None)

            all_rows.append(record)
            if "[2M" not in adduct:
                nodimer_rows.append(record)

        self.db.write_many(INSERT_MASTER_SQL, all_rows)
        self.db.write_many(INSERT_MASTER_NODIMER_SQL, nodimer_rows)

    def add_pnnl(self):
        print("ADDING PNNL")
        pnnl = pd.read_csv("./datasets/pnnl.tsv", sep="\t")

        mass_col_ccs_col_adduct = [
            ("mPlusH", "mPlusHCCS", "[M+H]+"),
            ("mPlusNa", "mPlusNaCCS", "[M+Na]+"),
            ("mMinusH", "mMinusHCCS", "[M-H]-"),
            ("mPlusDot", "mPlusDotCCS", "[M+dot]+"),
            ("mPlus", "mPlusCCS", "[M]+"),
            ("mPlusC2H3O2", "mPlusC2H3O2CCS", "[M+CH3COO]-"),
            ("mMinusClO", "mMinusClOCCS", "[M-ClO]+"),
            ("mMinusBrO", "mMinusBrOCCS", "[M-BrO]+"),
        ]

        # none of PNNL's adducts represent dimers, so master and master_nodimer match exactly
        all_rows = []
        for _, row in pnnl.iterrows():
            if not (pd.notna(row["PubChem CID"]) and pd.notna(row["InChi"])):
                continue

            cid = int(row["PubChem CID"])
            name = row["Neutral Name"]
            inchikey = row["InChi"]

            for mass_col, ccs_col, adduct in mass_col_ccs_col_adduct:
                if pd.notna(row[ccs_col]):
                    z = calculate_charge(adduct)
                    all_rows.append(("PNNL", name, cid, adduct, row[mass_col], z, row[ccs_col], None, inchikey, None, None, None))

        self.db.write_many(INSERT_MASTER_SQL, all_rows)
        self.db.write_many(INSERT_MASTER_NODIMER_SQL, all_rows)

    def add_acs(self):
        print("ADDING ACS")
        sheets = pd.read_excel("./datasets/acs.xlsx", sheet_name=["M+H", "M+Na", "M-H", "Others"])

        # ACS's source sheets don't encode dimers, so master and master_nodimer match exactly
        all_rows = []
        for _, df in sheets.items():
            for _, row in df.iterrows():
                cid = row["PubChem CID"]
                mass = None if pd.isna(row["m/z"]) else row["m/z"]
                adduct = None if pd.isna(row["adduct"]) else row["adduct"]
                if not (pd.notna(cid) and mass and adduct):
                    continue

                cid = int(cid)
                adduct = ADDUCT_STANDARDIZATION.get(adduct, adduct)
                name = None if pd.isna(row["name"]) else row["name"]
                ccs = None if pd.isna(row["TWCCSN2"]) else row["TWCCSN2"]
                superclass = None if pd.isna(row["Super class"]) else row["Super class"]
                class_ = None if pd.isna(row["Class"]) else row["Class"]
                subclass = None if pd.isna(row["Subclass"]) else row["Subclass"]
                inchikey = None if pd.isna(row["InChIKey"]) else row["InChIKey"]

                if not (ccs and adduct and mass and name):
                    continue

                z = calculate_charge(adduct)
                all_rows.append(("ACS", name, cid, adduct, mass, z, ccs, None, inchikey, superclass, class_, subclass))

        self.db.write_many(INSERT_MASTER_SQL, all_rows)
        self.db.write_many(INSERT_MASTER_NODIMER_SQL, all_rows)

    def add_metlin(self):
        print("ADDING METLIN")
        metlin = pd.read_csv("./datasets/metlin.csv")

        all_rows, nodimer_rows = [], []
        for _, row in metlin.iterrows():
            cid = row["pubChem"]
            mass = row["m/z"]
            adduct = row["Adduct"]
            is_valid_row = (
                pd.notna(cid) and str(cid).isnumeric()
                and mass and adduct and row["% CV"] <= 1
            )
            if not is_valid_row:
                continue

            adduct = ADDUCT_STANDARDIZATION.get(adduct, adduct)
            z = calculate_charge(adduct)
            record = ("METLIN", row["Molecule Name"], cid, adduct, mass, z, row["CCS_AVG"], None, row["InChIKEY"], None, None, None)

            all_rows.append(record)
            if row["Dimer.1"] == "Monomer":
                nodimer_rows.append(record)

        self.db.write_many(INSERT_MASTER_SQL, all_rows)
        self.db.write_many(INSERT_MASTER_NODIMER_SQL, nodimer_rows)

    def find_smiles(self):
        records = self.db.read("SELECT id, name, pubchemId, mass, adduct FROM master_nodimer WHERE smi IS NULL")

        for id_, name, cid, mass, adduct in records:
            if cid:
                smiles = get_pubchem_smiles_by_cid(cid, mass, adduct, ADDUCT_OFFSETS)
            else:
                smiles = get_pubchem_smiles_by_name(name, mass, adduct, ADDUCT_OFFSETS)

            if smiles:
                self.db.write(
                    "UPDATE master_nodimer SET smi = ?, inchikey = ? WHERE id = ?",
                    (smiles.get("smiles"), smiles.get("inchikey"), id_),
                )

    def find_classes(self):
        records = self.db.read("SELECT id, inchikey FROM master_nodimer WHERE superclass IS NULL AND inchikey NOT NULL")

        cache = {}
        for id_, inchikey in records:
            if inchikey in cache:
                superclass, class_, subclass = cache[inchikey]
            else:
                superclass, class_, subclass = get_classyfire_classification(inchikey)
                if not (superclass or class_ or subclass):
                    continue
                cache[inchikey] = (superclass, class_, subclass)

            self.db.write(
                "UPDATE master_nodimer SET superclass = ?, class = ?, subclass = ? WHERE id = ?",
                (superclass, class_, subclass, id_),
            )

    def find_inchikey(self):
        records = self.db.read("SELECT id, smi FROM master_nodimer WHERE smi IS NOT NULL and inchikey IS NULL")

        for id_, smi in records:
            result = get_cid_inchikey_from_smiles(smi)
            if result:
                self.db.write(
                    "UPDATE master_nodimer set pubchemId = ?, inchikey = ? where id = ?",
                    (result["cid"], result["inchikey"], id_),
                )

    def clean(self):
        print("STARTING DATA CLEANING...")
        df = self.db.read_df("SELECT * FROM master_nodimer WHERE smi IS NOT NULL")

        if df.empty:
            print("No data found to clean.")
            return

        ccs_outlier_threshold_pct = 1.0
        group_cols = ["smi", "adduct"]
        grouped_ccs = df.groupby(group_cols)["ccs"]

        group_size = grouped_ccs.transform("size")
        group_median_ccs = grouped_ccs.transform("median")
        deviation_pct = (df["ccs"] - group_median_ccs).abs() / group_median_ccs * 100

        within_threshold = deviation_pct <= ccs_outlier_threshold_pct
        group_has_survivor = within_threshold.groupby([df["smi"], df["adduct"]]).transform("any")
        is_closest_to_median = deviation_pct == deviation_pct.groupby([df["smi"], df["adduct"]]).transform("min")
        per_point_keep = within_threshold | (~group_has_survivor & is_closest_to_median)

        group_max_min_ratio = grouped_ccs.transform(lambda x: x.max() / x.min())
        whole_group_keep = group_max_min_ratio <= 1 + ccs_outlier_threshold_pct / 100

        keep_mask = whole_group_keep.where(group_size < 3, per_point_keep)
        df_valid = df[keep_mask].copy()

        print(f"Original rows: {len(df)}. Rows after CCS outlier filtering: {len(df_valid)}")

        # (smi, adduct, ccs) matches within 4 decimal places are indistinguishable from
        # dataset-overlap artifacts rather than genuine independent replicate measurements
        ccs_rounded = df_valid["ccs"].round(4)
        df_clean = (
            df_valid.assign(ccs_rounded=ccs_rounded)
            .drop_duplicates(subset=["smi", "adduct", "ccs_rounded"], keep="first")
            .drop(columns=["ccs_rounded"])
        )

        print(f"Rows after exact-duplicate removal: {len(df_clean)}")

        self.db.write("DROP TABLE IF EXISTS master_clean")
        self.db.write(
            "CREATE TABLE master_clean("
            "id INTEGER PRIMARY KEY, "
            "tag TEXT, name TEXT, pubchemId TEXT, "
            "adduct TEXT, mass REAL, z INTEGER, "
            "ccs REAL, smi TEXT, inchikey TEXT, "
            "superclass TEXT, class TEXT, subclass TEXT)"
        )

        cols_order = ["id", "tag", "name", "pubchemId", "adduct", "mass", "z", "ccs", "smi", "inchikey", "superclass", "class", "subclass"]
        df_clean = df_clean[cols_order]

        self.db.write_df(df_clean, "master_clean", if_exists="append")

        print(f"CLEANING COMPLETE")

    def build_ood_dataset(self, output_csv: str):
        df1 = pd.read_csv("all_training.csv", usecols=["SMILES", "Label", "Adduct"])
        df2 = pd.read_csv("Combined dataset.csv")
        df3 = pd.read_csv("external test set1.csv", usecols=["SMILES", "ccs", "Adduct"])
        df4 = pd.read_csv("external test set2.csv", usecols=["SMILES", "ccs", "Adduct"])
        df5 = pd.read_csv("ExternalTestData.csv", usecols=["SMILES", "True CCS", "Adduct"])
        df6 = pd.read_csv("FD_in-house_test.csv", usecols=["smiles", "CCS", "Adduct"])
        df7 = pd.read_csv("filtering.csv", usecols=["SMILES", "Average CCS", "Adduct type"])
        df8 = pd.read_excel("M-H_Testing_Input_SMILES.xlsx", usecols=["Input", "CCS", "Ion Species"])
        df9 = pd.read_excel("M-H_Training_Input_SMILES.xlsx", usecols=["Input", "CCS", "Ion Species"])
        df10 = pd.read_csv("TestData.csv", usecols=["SMILES", "True CCS", "Adduct"])
        df11 = pd.read_csv("TrainData.csv", usecols=["SMILES", "True CCS", "Adduct"])
        df12 = pd.read_csv("TrainingSet.csv", usecols=["SMILES", "True CCS", "Adduct"])

        def _to_py(x):
            if isinstance(x, (bytes, np.bytes_)):
                return x.decode("utf-8", errors="replace")
            if isinstance(x, np.ndarray) and x.dtype == object:
                return np.array([_to_py(v) for v in x], dtype=object)
            return x


        def load_group_df(h5_path, group_name):
            with h5py.File(h5_path, "r") as f:
                g = f[group_name]
                return pd.DataFrame({
                    "SMILES": _to_py(g["SMILES"][()]),
                    "Adduct": _to_py(g["Adducts"][()]),
                    "CCS": g["CCS"][()],
                })


        with h5py.File("DATASETS.h5", "r") as f:
            group_names = list(f.keys())
        datasets = {name: load_group_df("DATASETS.h5", name) for name in group_names}

        dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12] + list(datasets.values())

        rename_maps = {
            1: {"Label": "CCS"}, 7: {"Average CCS": "CCS", "Adduct type": "Adduct"},
            8: {"Input": "SMILES", "Ion Species": "Adduct"}, 9: {"Input": "SMILES", "Ion Species": "Adduct"},
            6: {"smiles": "SMILES"},
        }
        generic_renames = {"ccs": "CCS", "True CCS": "CCS"}

        standardized_dfs = []
        for idx, df in enumerate(dfs, start=1):
            curr_map = rename_maps.get(idx, {}).copy()
            curr_map.update({k: v for k, v in generic_renames.items() if k in df.columns})
            df = df.rename(columns=curr_map)
            standardized_dfs.append(df[STANDARD_COLS])

        merged_df = pd.concat(standardized_dfs, ignore_index=True)

        merged_df = merged_df.dropna(subset=STANDARD_COLS).copy()
        merged_df["CCS"] = pd.to_numeric(merged_df["CCS"], errors="coerce")
        merged_df = merged_df.dropna(subset=["CCS"]).copy()

        merged_df["SMILES"] = merged_df["SMILES"].astype(str).str.strip()
        merged_df["Adduct"] = merged_df["Adduct"].astype(str).str.strip()
        merged_df["Adduct"] = merged_df["Adduct"].map(ADDUCT_STANDARDIZATION).fillna(merged_df["Adduct"])

        def within_1pct(group):
            ccs = group["CCS"].astype(float)
            mean = ccs.mean()
            if mean == 0:
                return True
            return (ccs.max() - ccs.min()) / mean <= 0.01


        good_groups = merged_df.groupby(["SMILES", "Adduct"], sort=False).filter(within_1pct)
        deduped_df = good_groups.groupby(["SMILES", "Adduct"], as_index=False, sort=False).agg(CCS=("CCS", "mean"))

        remover = SaltRemover()


        def remove_salts(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            mol2 = remover.StripMol(mol)
            if mol2 is None or mol2.GetNumAtoms() == 0:
                return None
            return Chem.MolToSmiles(mol2, isomericSmiles=True)


        deduped_df["SMILES"] = deduped_df["SMILES"].apply(remove_salts)
        deduped_df = deduped_df.dropna(subset=["SMILES"]).copy()
        deduped_df = deduped_df[deduped_df["SMILES"].str.len() > 0].copy()

        candidate_df = deduped_df.groupby(["SMILES", "Adduct"], as_index=False, sort=False).agg(CCS=("CCS", "mean"))
        candidate_df = candidate_df.rename(columns={"SMILES": "smi", "Adduct": "adduct", "CCS": "ccs"})

        print(f"Stage 1 -- aggregated candidate pool: {len(candidate_df)} rows, {candidate_df['smi'].nunique()} unique molecules")

        # ============ Stage 2: remove exact smi overlap with current in-distribution data ============

        conn = sqlite3.connect(self.db_filename)
        in_distribution_smi = set(pd.read_sql_query("SELECT DISTINCT smi FROM master_clean WHERE smi IS NOT NULL", conn)["smi"])
        conn.close()

        candidate_df = candidate_df[~candidate_df["smi"].isin(in_distribution_smi)].reset_index(drop=True)
        print(f"Stage 2 -- after removing exact smi overlap: {len(candidate_df)} rows, {candidate_df['smi'].nunique()} unique molecules")

        # ============ Stage 3: remove near-duplicates by nearest-neighbor Tanimoto similarity ============

        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True)


        def compute_fingerprints(smiles_series):
            fps = []
            for smi in smiles_series:
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)
                fps.append(morgan_generator.GetSparseCountFingerprint(mol))
            return fps


        candidate_smi = candidate_df["smi"].drop_duplicates().reset_index(drop=True)
        candidate_fps = compute_fingerprints(candidate_smi)
        in_distribution_fps = compute_fingerprints(pd.Series(sorted(in_distribution_smi)))

        nn_similarity = np.array([
            max(DataStructs.BulkTanimotoSimilarity(fp, in_distribution_fps)) for fp in candidate_fps
        ])

        novel_smi = set(candidate_smi[nn_similarity < NEAR_DUPLICATE_THRESHOLD])
        candidate_df = candidate_df[candidate_df["smi"].isin(novel_smi)].reset_index(drop=True)
        print(
            f"Stage 3 -- after removing nearest-neighbor Tanimoto similarity >= {NEAR_DUPLICATE_THRESHOLD}: "
            f"{len(candidate_df)} rows, {candidate_df['smi'].nunique()} unique molecules"
        )

        # ============ Output ============

        candidate_df.to_csv(output_csv, index=False)
        print(f"Wrote {len(candidate_df)} rows to {output_csv}")


