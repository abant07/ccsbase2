import sqlite3
import pandas as pd
import requests
from urllib.parse import quote

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem.SaltRemover import SaltRemover

class CCSDataIntegration:
    def __init__(self, db_filename: str):
        self.db_filename = db_filename

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

        self.ADDUCT_OFFSETS = {
            # --- Positive Mode Adducts ---
            '[M+H]+': ATOMIC_MASSES['H'],
            '[M+Na]+': ATOMIC_MASSES['Na'],
            '[M+K]+': ATOMIC_MASSES['K'],
            '[M+Li]+': ATOMIC_MASSES['Li'],
            '[M+Rb]+': ATOMIC_MASSES['Rb'],
            '[M+Cs]+': ATOMIC_MASSES['Cs'],
            '[M+NH4]+': ATOMIC_MASSES['N'] + 4 * ATOMIC_MASSES['H'],
            '[M]+': 0.0,
            '[M]-': 0.0,
            '[M+dot]+': 0.000549,
            '[M-Br]+': -ATOMIC_MASSES['Br'],
            '[M-Cl]+': -ATOMIC_MASSES['Cl'],
            '[M-Na+2H]+': -ATOMIC_MASSES['Na'] + 2 * ATOMIC_MASSES['H'],

            # Water/Ammonia Losses (Positive)
            '[M+H-H2O]+': ATOMIC_MASSES['H'] - H2O,
            '[M+Na-H2O]+': ATOMIC_MASSES['Na'] - H2O,
            '[M+H-2H2O]+': ATOMIC_MASSES['H'] - 2 * H2O,
            '[M+Na-2H2O]+': ATOMIC_MASSES['Na'] - 2 * H2O,
            '[M-3H2O+H]+': -3 * H2O + ATOMIC_MASSES['H'],
            '[M+H-NH3]+': ATOMIC_MASSES['H'] - NH3,

            # Sodium/Potassium Exchanges (Positive)
            '[M+Na-H]+': ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'],
            '[M+2Na-H]+': 2 * ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'],
            '[M-2H+3Na]+': -2 * ATOMIC_MASSES['H'] + 3 * ATOMIC_MASSES['Na'],
            '[M-H+2K]+': -ATOMIC_MASSES['H'] + 2 * ATOMIC_MASSES['K'],

            # SO3 / Sulfonate Adducts (Positive)
            '[M-SO3-H2O+H]+': -SO3 - H2O + ATOMIC_MASSES['H'],
            '[M-SO3-2H2O+H]+': -SO3 - 2 * H2O + ATOMIC_MASSES['H'],
            '[M-SO3-3H2O+H]+': -SO3 - 3 * H2O + ATOMIC_MASSES['H'],
            '[M-2SO3-2H2O+H]+': -2 * SO3 - 2 * H2O + ATOMIC_MASSES['H'],
            '[M-SO3+H]+': -SO3 + ATOMIC_MASSES['H'],

            # Complex Losses/Gains (Positive)
            '[M-HF-H2O+H]+': -HF - H2O + ATOMIC_MASSES['H'],
            '[M-HF+H]+': -HF + ATOMIC_MASSES['H'],
            '[M-CH3COOH-H2O+H]+': -CH3COOH - H2O + ATOMIC_MASSES['H'],
            '[M-CH3COOH+H]+': -CH3COOH + ATOMIC_MASSES['H'],
            '[M-C6H8O6-2H2O+H]+': -C6H8O6 - 2 * H2O + ATOMIC_MASSES['H'],
            '[M-C6H8O6-H2O+H]+': -C6H8O6 - H2O + ATOMIC_MASSES['H'],

            # --- Negative Mode Adducts ---
            '[M-H]-': -ATOMIC_MASSES['H'],
            '[M-3H]3-': -3 * ATOMIC_MASSES['H'],

            # Water/CO2 Losses (Negative)
            '[M-H-H2O]-': -ATOMIC_MASSES['H'] - H2O,
            '[M+H2O-H]-': H2O - ATOMIC_MASSES['H'],
            '[M-H-CO2]-': -ATOMIC_MASSES['H'] - CO2,

            # Salt Adducts (Negative)
            '[M+Na-2H]-': ATOMIC_MASSES['Na'] - 2 * ATOMIC_MASSES['H'],
            '[M+K-2H]-': ATOMIC_MASSES['K'] - 2 * ATOMIC_MASSES['H'],
            '[M+2Na-3H]-': 2 * ATOMIC_MASSES['Na'] - 3 * ATOMIC_MASSES['H'],
            '[M+Cl]-': ATOMIC_MASSES['Cl'],
            '[M+K-H+Cl]-': ATOMIC_MASSES['K'] - ATOMIC_MASSES['H'] + ATOMIC_MASSES['Cl'],
            '[M+Na-H+Cl]-': ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'] + ATOMIC_MASSES['Cl'],

            # Formate/Acetate/Organic Adducts (Negative)
            '[M+CH3COO]-': CH3COO,
            '[M+HCOO]-': HCOO,
            '[M+K-H+HCOO]-': ATOMIC_MASSES['K'] - ATOMIC_MASSES['H'] + HCOO,
            '[M+Na-H+HCOO]-': ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'] + HCOO,
            '[M-H2O+HCOO]-': -H2O + HCOO,
            '[M+CH3COONa-H]-': CH3COO + ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'],
            '[M-H+HCOOH]-': -ATOMIC_MASSES['H'] + (HCOO + ATOMIC_MASSES['H']),

            # SO3 / Sulfonate Adducts (Negative)
            '[M-SO3-H]-': -SO3 - ATOMIC_MASSES['H'],
            '[M-SO3-H2O-H]-': -SO3 - H2O - ATOMIC_MASSES['H'],
            '[M-SO3-H2O+HCOO]-': -SO3 - H2O + HCOO,
            '[M-SO3-H2O+Cl]-': -SO3 - H2O + ATOMIC_MASSES['Cl'],
            '[M-SO3+Cl]-': -SO3 + ATOMIC_MASSES['Cl'],

            # Halogen/Oxide Specifics (Negative)
            '[M-BrO]+': -BrO,
            '[M-ClO]+': -ClO,
            '[M-Br+O]-': -ATOMIC_MASSES['Br'] + ATOMIC_MASSES['O'],
            '[M-Cl+O]-': -ATOMIC_MASSES['Cl'] + ATOMIC_MASSES['O'],

            # --- Multi-charged Ions ---
            '[M+2H]2+': 2 * ATOMIC_MASSES['H'],
            '[M+3H]3+': 3 * ATOMIC_MASSES['H'],
            '[M+4H]4+': 4 * ATOMIC_MASSES['H'],
            '[M-2H]2-': -2 * ATOMIC_MASSES['H'],
            '[M+2K]2+': 2 * ATOMIC_MASSES['K'],
            '[M-2Cl]2+': -2 * ATOMIC_MASSES['Cl']
        }

        self.ADDUCT_STANDARDIZATION = {
            "M+": "[M]+",
            "[M+H]": "[M+H]+",
            "[M-H]": "[M-H]-",
            "[M+Na]": "[M+Na]+",
            "+HCOO": "[M+HCOO]-",
            "+NH4": "[M+NH4]+",
            "-Br": "[M-Br]+",
            "-2Cl": "[M-2Cl]2+",
            "-Cl": "[M-Cl]+",
            "-Na+2H": "[M-Na+2H]+",
            "[M-H2O+H]+": "[M+H-H2O]+",
            "[M-H+2Na]+": "[M+2Na-H]+",
            "[M-3H]-": "[M-3H]3-",
            "[M-H2O-H]-": "[M-H-H2O]-",
            "[M-CO2-H]-": "[M-H-CO2]-",
            "[M+H3C2O2]-": "[M+CH3COO]-",
            "[M+FA-H]-": "[M+HCOO]-"
        }

        conn = sqlite3.connect(self.db_filename)
        cur = conn.cursor()

        cur.execute("CREATE TABLE IF NOT EXISTS master(" \
        "id INTEGER PRIMARY KEY AUTOINCREMENT, " \
        "tag TEXT, name TEXT, pubchemId INTEGER, " \
        "adduct TEXT, mass REAL, z INTEGER, " \
        "ccs REAL, smi TEXT, inchikey TEXT, " \
        "superclass TEXT, class TEXT, subclass TEXT)")
        conn.commit()

        cur.close()
        conn.close()


    def _get_pubchem_smiles_cid(self, pubchem_cid: int, experimental_mass: float, adduct: str):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_cid}/property/IsomericSMILES,InChIKey/JSON"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                response.raise_for_status()
                return None

            data = response.json()

            props = data["PropertyTable"]["Properties"][0]

            m = Chem.MolFromSmiles(props.get("SMILES"))
            remover = SaltRemover()
            stripped_mol = remover.StripMol(m)
            desalted_smiles = Chem.MolToSmiles(stripped_mol)
            desalted_smiles_mol = Chem.MolFromSmiles(desalted_smiles)
            isotopic_mass = rdmd.CalcExactMolWt(desalted_smiles_mol)

            weight_diff = abs(float(experimental_mass) - (isotopic_mass+self.ADDUCT_OFFSETS[adduct]))
            tolerance = max(1, 0.01 * isotopic_mass)

            if weight_diff <= tolerance:
                return {
                    "smiles": desalted_smiles,
                    "inchikey": props.get("InChIKey"),
                }
            return None
        except Exception as e:
            print("Error while calling Pubchem API with CID", e)
            return None

    def _get_lipidmaps_smiles_name(self, chemical_name: str, experimental_mass: float, adduct: str):
        url = f"https://www.lipidmaps.org/rest/compound/abbrev/{chemical_name}/pubchem_cid"
        response = requests.get(url)

        data = None
        if response.status_code == 200:
            data = response.json()

        if not data:
            url = f"https://www.lipidmaps.org/rest/compound/abbrev_chains/{chemical_name}/pubchem_cid"
            response = requests.get(url)
            if response.status_code != 200:
                return None
            data = response.json()

            smiles_candidates = set([])
            # Case 1: Multi-result object with "Row1", "Row2", etc.
            if "Row1" in data:
                for result in list(data.values()):
                    cid = int(result.get("pubchem_cid"))
                    smiles = self._get_pubchem_smiles_cid(cid, experimental_mass, adduct)
                    if smiles:
                        smiles_candidates.add((smiles.get("smiles"), smiles.get("inchikey")))
            # Case 2: Single result object with "smiles" key at the top level
            elif "pubchem_cid" in data:
                cid = int(data.get("pubchem_cid"))
                return self._get_pubchem_smiles_cid(cid, experimental_mass, adduct)

            if len(smiles_candidates) == 1:
                return {
                    "smiles": smiles_candidates.pop()[0],
                    "inchikey": smiles_candidates.pop()[1],
                }
            return None
        else:
            smiles_candidates = set([])
            # Case 1: Multi-result object with "Row1", "Row2", etc.
            if "Row1" in data:
                for result in list(data.values()):
                    cid = int(result.get("pubchem_cid"))
                    smiles = self._get_pubchem_smiles_cid(cid, experimental_mass, adduct)
                    if smiles:
                        smiles_candidates.add((smiles.get("smiles"), smiles.get("inchikey")))
            # Case 2: Single result object with "smiles" key at the top level
            elif "pubchem_cid" in data:
                cid = int(data.get("pubchem_cid"))
                return self._get_pubchem_smiles_cid(cid, experimental_mass, adduct)

            if len(smiles_candidates) == 1:
                return {
                    "smiles": smiles_candidates.pop()[0],
                    "inchikey": smiles_candidates.pop()[1],
                }
            return None

    def _get_pubchem_smiles_name(self, chemical_name: str, experimental_mass: float, adduct: str):
        safe_name = quote(chemical_name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_name}/property/IsomericSMILES,InChIKey/JSON"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return self._get_lipidmaps_smiles_name(safe_name, experimental_mass, adduct)

            data = response.json()

            smiles_candidates = set([])
            props = data["PropertyTable"]["Properties"]

            for prop in props:
                m = Chem.MolFromSmiles(prop.get("SMILES"))
                remover = SaltRemover()
                stripped_mol = remover.StripMol(m)
                desalted_smiles = Chem.MolToSmiles(stripped_mol)
                desalted_smiles_mol = Chem.MolFromSmiles(desalted_smiles)
                isotopic_mass = rdmd.CalcExactMolWt(desalted_smiles_mol)

                weight_diff = abs(float(experimental_mass) - (isotopic_mass+self.ADDUCT_OFFSETS[adduct]))
                tolerance = max(1, 0.01 * isotopic_mass)
                if weight_diff <= tolerance:
                    smiles_candidates.add((desalted_smiles, prop.get("InChIKey")))

            if len(smiles_candidates) == 1:
                candidate = smiles_candidates.pop()
                return {
                    "smiles": candidate[0],
                    "inchikey": candidate[1],
                }
            return None
        except Exception as e:
            print("Error while calling Pubchem API with Name", e)
            return None

    def _get_classyfire(self, inchi: str):
        url = f"https://cfb.fiehnlab.ucdavis.edu/entities/{inchi}.json"
        superclass = None
        class_ = None
        subclass = None
        try:
            response = requests.get(url)
            if response.status_code == 200:
                body = response.json()
                if body.get("superclass", None):
                    superclass = body["superclass"]["name"]
                if body.get("class", None):
                    class_ = body["class"]["name"]
                if body.get("subclass", None):
                    subclass = body["subclass"]["name"]

        except requests.exceptions.ConnectTimeout as e:
            print(f"Connection timeout for {inchi}: {e}")

        return superclass, class_, subclass

    def _get_cid_inchikey_from_smiles(self, smiles: str):
        safe_smiles = quote(smiles)
        
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{safe_smiles}/property/InChIKey/JSON"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                print(f"SMILES not found in PubChem: {smiles}")
                return None
            
            data = response.json()

            # The structure is usually: {'PropertyTable': {'Properties': [{'CID': 123, 'InChIKey': '...'}]}}
            if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                prop = data["PropertyTable"]["Properties"][0]
                return {
                    "cid": prop.get("CID"),
                    "inchikey": prop.get("InChIKey")
                }
            return None
        except Exception as e:
            print(f"Error looking up SMILES: {e}")
            return None

    def _calculate_charge(self, adduct: str):
        polarity = -1 if adduct[-1] == "-" else 1
        charge = ""

        for c in adduct[-2::-1]:
            if c == "]":
                break
            charge = c + charge
        
        if charge == "":
            charge = 1

        return int(charge) * polarity

    def add_ccsbase(self):
        print("ADDING CCSBASE")
        merged_dataset = sqlite3.connect(self.db_filename)
        merged_dataset_cur = merged_dataset.cursor()

        c3s = sqlite3.connect("./datasets/C3S.db")
        c3s_cur = c3s.cursor()
        records = c3s_cur.execute("SELECT * FROM master WHERE smi IS NOT NULL").fetchall()
        c3s_cur.close()
        c3s.close()

        for record in records:
            name = record[1]
            adduct = self.ADDUCT_STANDARDIZATION.get(record[2], record[2])
            mass = record[3]
            z = self._calculate_charge(adduct)
            ccs = record[6]
            smiles = record[7]
            src_tag = record[8]
            superclass = record[11]
            class_ = record[12]
            subclass = record[13]

            if "[2M" not in adduct:
                m = Chem.MolFromSmiles(smiles)
                remover = SaltRemover()
                stripped_mol = remover.StripMol(m)
                desalted_smiles = Chem.MolToSmiles(stripped_mol)
                desalted_smiles_mol = Chem.MolFromSmiles(desalted_smiles)
                isotopic_mass = rdmd.CalcExactMolWt(desalted_smiles_mol)

                weight_diff = abs(float(mass) - (isotopic_mass+self.ADDUCT_OFFSETS[adduct]))
                tolerance = max(1, 0.01 * isotopic_mass)
                if src_tag != "nguyen25" and weight_diff <= tolerance:
                    merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", ("CCSBASE", name, None, adduct, mass, z, ccs, desalted_smiles, None, superclass, class_, subclass))
                else:
                    merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", ("CCSBASE", name, None, adduct, mass, z, ccs, None, None, superclass, class_, subclass))
        merged_dataset.commit()
        merged_dataset.close()

    def add_allccs(self):
        print("ADDING ALLCCS")
        merged_dataset = sqlite3.connect(self.db_filename)
        merged_dataset_cur = merged_dataset.cursor()
        allccs = pd.read_csv("./datasets/allccs.csv")

        for _, row in allccs.iterrows():
            if pd.notna(row["m/z"]) and pd.notna(row["Adduct"]) and pd.notna(row["CCS"]) and pd.notna(row["Name"]) and row["Confidence level"] == "1":
                name = row["Name"]
                mz = row["m/z"]
                adduct = self.ADDUCT_STANDARDIZATION.get(row["Adduct"], row["Adduct"])
                if "[2M" not in adduct:
                  ccs = row["CCS"]

                  z = self._calculate_charge(adduct)

                  merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                              VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                              ("ALLCCS", name, None, adduct, mz, z, ccs, None, None, None, None, None))

        merged_dataset.commit()
        merged_dataset.close()

    def add_pnnl(self):
        print("ADDING PNNL")
        merged_dataset = sqlite3.connect(self.db_filename)
        merged_dataset_cur = merged_dataset.cursor()

        pnnl = pd.read_csv("./datasets/pnnl.tsv", sep="\t")

        for _, row in pnnl.iterrows():
            if pd.notna(row["PubChem CID"]) and pd.notna(row["InChi"]):
                cid = int(row["PubChem CID"])
                name = row["Neutral Name"]

                if pd.notna(row["mPlusHCCS"]):
                    mass = row["mPlusH"]
                    ccs = row["mPlusHCCS"]
                    adduct = "[M+H]+"
                    z = self._calculate_charge(adduct)
                    inchikey = row["InChi"]
                    merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        ("PNNL", name, cid, adduct, mass, z, ccs, None, inchikey, None, None, None))

                if pd.notna(row["mPlusNaCCS"]):
                    mass = row["mPlusNa"]
                    ccs = row["mPlusNaCCS"]
                    adduct = "[M+Na]+"
                    z = self._calculate_charge(adduct)
                    inchikey = row["InChi"]
                    merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        ("PNNL", name, cid, adduct, mass, z, ccs, None, inchikey, None, None, None))

                if pd.notna(row["mMinusHCCS"]):
                    mass = row["mMinusH"]
                    ccs = row["mMinusHCCS"]
                    adduct = "[M-H]-"
                    z = self._calculate_charge(adduct)
                    inchikey = row["InChi"]
                    merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        ("PNNL", name, cid, adduct, mass, z, ccs, None, inchikey, None, None, None))

                if pd.notna(row["mPlusDotCCS"]):
                    mass = row["mPlusDot"]
                    ccs = row["mPlusDotCCS"]
                    adduct = "[M+dot]+"
                    z = self._calculate_charge(adduct)
                    inchikey = row["InChi"]
                    merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        ("PNNL", name, cid, adduct, mass, z, ccs, None, inchikey, None, None, None))

                if pd.notna(row["mPlusCCS"]):
                    mass = row["mPlus"]
                    ccs = row["mPlusCCS"]
                    adduct = "[M]+"
                    z = self._calculate_charge(adduct)
                    inchikey = row["InChi"]
                    merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        ("PNNL", name, cid, adduct, mass, z, ccs, None, inchikey, None, None, None))

                if pd.notna(row["mPlusC2H3O2CCS"]):
                    mass = row["mPlusC2H3O2"]
                    ccs = row["mPlusC2H3O2CCS"]
                    adduct = "[M+CH3COO]-"
                    z = self._calculate_charge(adduct)
                    inchikey = row["InChi"]
                    merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        ("PNNL", name, cid, adduct, mass, z, ccs, None, inchikey, None, None, None))

                if pd.notna(row["mMinusClOCCS"]):
                    mass = row["mMinusClO"]
                    ccs = row["mMinusClOCCS"]
                    adduct = "[M-ClO]+"
                    z = self._calculate_charge(adduct)
                    inchikey = row["InChi"]
                    merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        ("PNNL", name, cid, adduct, mass, z, ccs, None, inchikey, None, None, None))

                if pd.notna(row["mMinusBrOCCS"]):
                    mass = row["mMinusBrO"]
                    ccs = row["mMinusBrOCCS"]
                    adduct = "[M-BrO]+"
                    z = self._calculate_charge(adduct)
                    inchikey = row["InChi"]
                    merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        ("PNNL", name, cid, adduct, mass, z, ccs, None, inchikey, None, None, None))

        merged_dataset.commit()
        merged_dataset.close()

    def add_acs(self):
        print("ADDING ACS")
        merged_dataset = sqlite3.connect(self.db_filename)
        merged_dataset_cur = merged_dataset.cursor()
        sheets = pd.read_excel("./datasets/acs.xlsx", sheet_name=["M+H", "M+Na", "M-H", "Others"])

        for _, df in sheets.items():
            for _, row in df.iterrows():
                cid = row["PubChem CID"]
                mass = None if pd.isna(row["m/z"]) else row["m/z"]
                adduct = None if pd.isna(row["adduct"]) else row["adduct"]
                if pd.notna(cid) and mass and adduct:
                    cid = int(cid)
                    adduct = self.ADDUCT_STANDARDIZATION.get(adduct, adduct)
                    name =  None if pd.isna(row["name"]) else row["name"]
                    ccs = None if pd.isna(row["TWCCSN2"]) else row["TWCCSN2"]
                    superclass = None if pd.isna(row["Super class"]) else row["Super class"]
                    class_ = None if pd.isna(row["Class"]) else row["Class"]
                    subclass = None if pd.isna(row["Subclass"]) else row["Subclass"]
                    inchikey = None if pd.isna(row["InChIKey"]) else row["InChIKey"]

                    if ccs and adduct and mass and name:
                        z = self._calculate_charge(adduct)

                        merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemID, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                                    ("ACS", name, cid, adduct, mass, z, ccs, None, inchikey, superclass, class_, subclass))
                                                    

        merged_dataset.commit()
        merged_dataset.close()

    def add_metlin(self):
        print("ADDING METLIN")
        merged_dataset = sqlite3.connect(self.db_filename)
        merged_dataset_cur = merged_dataset.cursor()

        metlin = pd.read_csv("./datasets/metlin.csv")

        for _, row in metlin.iterrows():
            cid = row["pubChem"]
            mass = row["m/z"]
            adduct = row["Adduct"]
            if pd.notna(cid) and str(cid).isnumeric() and row["Dimer.1"] == "Monomer" and mass and adduct and row["% CV"] <= 1:
                adduct = self.ADDUCT_STANDARDIZATION.get(adduct, adduct)
                name = row["Molecule Name"]
                ccs = row["CCS_AVG"]
                inchikey = row["InChIKEY"]
                z = self._calculate_charge(adduct)

                merged_dataset_cur.execute("""INSERT INTO master(tag, name, pubchemId, adduct, mass, z, ccs, smi, inchikey, superclass, class, subclass)
                                                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                                ("METLIN", name, cid, adduct, mass, z, ccs, None, inchikey, None, None, None))

        merged_dataset.commit()
        merged_dataset.close()

    def find_smiles(self):
        merged_dataset = sqlite3.connect(self.db_filename)
        merged_dataset_cur = merged_dataset.cursor()

        records = merged_dataset_cur.execute("SELECT id, name, pubchemId, mass, adduct FROM master WHERE smi IS NULL").fetchall()

        count = 1
        for record in records:
            id = record[0]
            name = record[1]
            cid = record[2]
            mass = record[3]
            adduct = record[4]

            if cid:
                smiles = self._get_pubchem_smiles_cid(cid, mass, adduct)
                if smiles:
                    merged_dataset_cur.execute("UPDATE master SET smi = ?, inchikey = ? WHERE id = ?", (smiles.get("smiles"), smiles.get("inchikey"), id))
            else:
                smiles = self._get_pubchem_smiles_name(name, mass, adduct)
                if smiles:
                    merged_dataset_cur.execute("UPDATE master SET smi = ?, inchikey = ? WHERE id = ?", (smiles.get("smiles"), smiles.get("inchikey"), id))

            if count % 100 == 0:
                merged_dataset.commit()

            count += 1

        merged_dataset.commit()
        merged_dataset.close()


    def find_classes(self):
        merged_dataset = sqlite3.connect(self.db_filename)
        merged_dataset_cur = merged_dataset.cursor()

        records = merged_dataset_cur.execute("SELECT id, inchikey FROM master WHERE superclass IS NULL AND inchikey NOT NULL").fetchall()

        count = 1
        cache = {}
        for record in records:
            id = record[0]
            inchikey = record[1]

            if inchikey in cache:
                superclass, class_, subclass = cache[inchikey]
                merged_dataset_cur.execute("UPDATE master SET superclass = ?, class = ?, subclass = ? WHERE id = ?", (superclass, class_, subclass, id))
            else:
                superclass, class_, subclass = self._get_classyfire(inchikey)
                if superclass or class_ or subclass:
                    merged_dataset_cur.execute("UPDATE master SET superclass = ?, class = ?, subclass = ? WHERE id = ?", (superclass, class_, subclass, id))
                    cache[inchikey] = [superclass, class_, subclass]

            if count % 100 == 0:
                merged_dataset.commit()

            count += 1

        merged_dataset.commit()
        merged_dataset.close()
    
    def find_inchikey(self):
        merged_dataset = sqlite3.connect(self.db_filename)
        merged_dataset_cur = merged_dataset.cursor()

        records = merged_dataset_cur.execute("SELECT id, smi FROM master WHERE smi IS NOT NULL and inchikey IS NULL").fetchall()

        count = 1
        for record in records:
            result = self._get_cid_inchikey_from_smiles(record[1])
            if result:  
                merged_dataset_cur.execute("UPDATE master set pubchemId = ?, inchikey = ? where id = ?", (result['cid'], result['inchikey'], record[0]))

            if count % 100 == 0:
                merged_dataset.commit()

        merged_dataset.commit()
        merged_dataset.close()
    
    def clean(self):
        print("STARTING DATA CLEANING...")
        conn = sqlite3.connect(self.db_filename)

        # 1. Load data
        query = "SELECT * FROM master WHERE smi IS NOT NULL"
        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("No data found to clean.")
            conn.close()
            return

        # 2. Filter Groups based on CCS relative deviation ≤ 1%
        ccs_ratio = df.groupby(['smi', 'adduct',])['ccs'] \
                    .transform(lambda x: x.max() / x.min())
        
        # Keep only rows where values are within 1% of each other
        df_valid = df[ccs_ratio <= 1.01].copy()

        print(f"Original rows: {len(df)}. Rows after CCS conflict filtering: {len(df_valid)}")

        # 3. Define Aggregation Logic
        
        def _join_unique_sorted(series):
            cleaned_vals = []
            for s in series:
                if pd.notna(s) and s != '':
                    # Formatting fix: If ID loaded as float (e.g. 123.0) due to NaNs, convert to int string
                    if isinstance(s, float) and s.is_integer():
                        cleaned_vals.append(str(int(s)))
                    else:
                        cleaned_vals.append(str(s))
            
            unique_vals = set(cleaned_vals)
            
            if not unique_vals:
                return None
            # Sort lexicographically and join
            return ",".join(sorted(unique_vals))

        agg_rules = {
            'id': 'min',                 
            'tag': _join_unique_sorted,  
            'name': _join_unique_sorted, 
            'pubchemId': _join_unique_sorted,
            'mass': 'mean',              
            'z': 'first',                
            'ccs': 'mean',               
            'inchikey': _join_unique_sorted,
            'superclass': _join_unique_sorted,
            'class': _join_unique_sorted,
            'subclass': _join_unique_sorted
        }

        # 4. Perform the Merge
        df_clean = df_valid.groupby(['smi', 'adduct'], as_index=False).agg(agg_rules)

        # 5. Write to master_clean table
        cur = conn.cursor()
        
        cur.execute("DROP TABLE IF EXISTS master_clean")
        
        cur.execute("CREATE TABLE master_clean(" \
            "id INTEGER PRIMARY KEY, " \
            "tag TEXT, name TEXT, pubchemId TEXT, " \
            "adduct TEXT, mass REAL, z INTEGER, " \
            "ccs REAL, smi TEXT, inchikey TEXT, " \
            "superclass TEXT, class TEXT, subclass TEXT)")
        
        cols_order = ['id', 'tag', 'name', 'pubchemId', 'adduct', 'mass', 'z', 'ccs', 'smi', 'inchikey', 'superclass', 'class', 'subclass']
        df_clean = df_clean[cols_order]

        df_clean.to_sql('master_clean', conn, if_exists='append', index=False)
        
        conn.commit()
        conn.close()
        print(f"CLEANING COMPLETE. Merged into {len(df_clean)} unique records.")
