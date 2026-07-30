from urllib.parse import quote

import requests

from utils import desalt_and_mass


def _within_mass_tolerance(experimental_mass, isotopic_mass, adduct_offset):
    weight_diff = abs(float(experimental_mass) - (isotopic_mass + adduct_offset))
    tolerance = max(1, 0.01 * isotopic_mass)
    return weight_diff <= tolerance


def get_pubchem_smiles_by_cid(pubchem_cid, experimental_mass, adduct, adduct_offsets):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_cid}/property/IsomericSMILES,InChIKey/JSON"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            response.raise_for_status()
            return None

        props = response.json()["PropertyTable"]["Properties"][0]
        desalted_smiles, isotopic_mass = desalt_and_mass(props.get("SMILES"))

        if _within_mass_tolerance(experimental_mass, isotopic_mass, adduct_offsets[adduct]):
            return {"smiles": desalted_smiles, "inchikey": props.get("InChIKey")}
        return None
    except Exception as e:
        print("Error while calling Pubchem API with CID", e)
        return None


def _fetch_lipidmaps_json(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()


def get_lipidmaps_smiles_by_name(chemical_name, experimental_mass, adduct, adduct_offsets):
    data = _fetch_lipidmaps_json(f"https://www.lipidmaps.org/rest/compound/abbrev/{chemical_name}/pubchem_cid")
    if not data:
        data = _fetch_lipidmaps_json(f"https://www.lipidmaps.org/rest/compound/abbrev_chains/{chemical_name}/pubchem_cid")
    if not data:
        return None

    smiles_candidates = set()
    if "Row1" in data:
        for result in data.values():
            cid = int(result.get("pubchem_cid"))
            smiles = get_pubchem_smiles_by_cid(cid, experimental_mass, adduct, adduct_offsets)
            if smiles:
                smiles_candidates.add((smiles.get("smiles"), smiles.get("inchikey")))
    elif "pubchem_cid" in data:
        cid = int(data.get("pubchem_cid"))
        return get_pubchem_smiles_by_cid(cid, experimental_mass, adduct, adduct_offsets)

    if len(smiles_candidates) == 1:
        smiles, inchikey = smiles_candidates.pop()
        return {"smiles": smiles, "inchikey": inchikey}
    return None


def get_pubchem_smiles_by_name(chemical_name, experimental_mass, adduct, adduct_offsets):
    safe_name = quote(chemical_name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_name}/property/IsomericSMILES,InChIKey/JSON"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return get_lipidmaps_smiles_by_name(safe_name, experimental_mass, adduct, adduct_offsets)

        props = response.json()["PropertyTable"]["Properties"]

        smiles_candidates = set()
        for prop in props:
            desalted_smiles, isotopic_mass = desalt_and_mass(prop.get("SMILES"))
            if _within_mass_tolerance(experimental_mass, isotopic_mass, adduct_offsets[adduct]):
                smiles_candidates.add((desalted_smiles, prop.get("InChIKey")))

        if len(smiles_candidates) == 1:
            smiles, inchikey = smiles_candidates.pop()
            return {"smiles": smiles, "inchikey": inchikey}
        return None
    except Exception as e:
        print("Error while calling Pubchem API with Name", e)
        return None


def get_classyfire_classification(inchi):
    url = f"https://cfb.fiehnlab.ucdavis.edu/entities/{inchi}.json"
    superclass = class_ = subclass = None

    try:
        response = requests.get(url)
        if response.status_code == 200:
            body = response.json()
            if body.get("superclass"):
                superclass = body["superclass"]["name"]
            if body.get("class"):
                class_ = body["class"]["name"]
            if body.get("subclass"):
                subclass = body["subclass"]["name"]
    except requests.exceptions.ConnectTimeout as e:
        print(f"Connection timeout for {inchi}: {e}")

    return superclass, class_, subclass


def get_cid_inchikey_from_smiles(smiles):
    safe_smiles = quote(smiles)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{safe_smiles}/property/InChIKey/JSON"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"SMILES not found in PubChem: {smiles}")
            return None

        data = response.json()
        if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
            prop = data["PropertyTable"]["Properties"][0]
            return {"cid": prop.get("CID"), "inchikey": prop.get("InChIKey")}
        return None
    except Exception as e:
        print(f"Error looking up SMILES: {e}")
        return None
