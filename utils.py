from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover


def calculate_charge(adduct: str) -> int:
    polarity = -1 if adduct[-1] == "-" else 1
    charge = ""

    for c in adduct[-2::-1]:
        if c == "]":
            break
        charge = c + charge

    if charge == "":
        charge = 1

    return int(charge) * polarity


def desalt_and_mass(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    stripped_mol = SaltRemover().StripMol(mol)
    desalted_smiles = Chem.MolToSmiles(stripped_mol)
    desalted_mol = Chem.MolFromSmiles(desalted_smiles)
    return desalted_smiles, rdMolDescriptors.CalcExactMolWt(desalted_mol)
