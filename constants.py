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
CH3COOH = CH3COO + ATOMIC_MASSES['H']
C6H8O6 = 6 * ATOMIC_MASSES['C'] + 8 * ATOMIC_MASSES['H'] + 6 * ATOMIC_MASSES['O']

ADDUCT_OFFSETS = {
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

    '[M+H-H2O]+': ATOMIC_MASSES['H'] - H2O,
    '[M+Na-H2O]+': ATOMIC_MASSES['Na'] - H2O,
    '[M+H-2H2O]+': ATOMIC_MASSES['H'] - 2 * H2O,
    '[M+Na-2H2O]+': ATOMIC_MASSES['Na'] - 2 * H2O,
    '[M-3H2O+H]+': -3 * H2O + ATOMIC_MASSES['H'],
    '[M+H-NH3]+': ATOMIC_MASSES['H'] - NH3,

    '[M+Na-H]+': ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'],
    '[M+2Na-H]+': 2 * ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'],
    '[M-2H+3Na]+': -2 * ATOMIC_MASSES['H'] + 3 * ATOMIC_MASSES['Na'],
    '[M-H+2K]+': -ATOMIC_MASSES['H'] + 2 * ATOMIC_MASSES['K'],

    '[M-SO3-H2O+H]+': -SO3 - H2O + ATOMIC_MASSES['H'],
    '[M-SO3-2H2O+H]+': -SO3 - 2 * H2O + ATOMIC_MASSES['H'],
    '[M-SO3-3H2O+H]+': -SO3 - 3 * H2O + ATOMIC_MASSES['H'],
    '[M-2SO3-2H2O+H]+': -2 * SO3 - 2 * H2O + ATOMIC_MASSES['H'],
    '[M-SO3+H]+': -SO3 + ATOMIC_MASSES['H'],

    '[M-HF-H2O+H]+': -HF - H2O + ATOMIC_MASSES['H'],
    '[M-HF+H]+': -HF + ATOMIC_MASSES['H'],
    '[M-CH3COOH-H2O+H]+': -CH3COOH - H2O + ATOMIC_MASSES['H'],
    '[M-CH3COOH+H]+': -CH3COOH + ATOMIC_MASSES['H'],
    '[M-C6H8O6-2H2O+H]+': -C6H8O6 - 2 * H2O + ATOMIC_MASSES['H'],
    '[M-C6H8O6-H2O+H]+': -C6H8O6 - H2O + ATOMIC_MASSES['H'],

    '[M-H]-': -ATOMIC_MASSES['H'],
    '[M-3H]3-': -3 * ATOMIC_MASSES['H'],

    '[M-H-H2O]-': -ATOMIC_MASSES['H'] - H2O,
    '[M+H2O-H]-': H2O - ATOMIC_MASSES['H'],
    '[M-H-CO2]-': -ATOMIC_MASSES['H'] - CO2,

    '[M+Na-2H]-': ATOMIC_MASSES['Na'] - 2 * ATOMIC_MASSES['H'],
    '[M+K-2H]-': ATOMIC_MASSES['K'] - 2 * ATOMIC_MASSES['H'],
    '[M+2Na-3H]-': 2 * ATOMIC_MASSES['Na'] - 3 * ATOMIC_MASSES['H'],
    '[M+Cl]-': ATOMIC_MASSES['Cl'],
    '[M+K-H+Cl]-': ATOMIC_MASSES['K'] - ATOMIC_MASSES['H'] + ATOMIC_MASSES['Cl'],
    '[M+Na-H+Cl]-': ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'] + ATOMIC_MASSES['Cl'],

    '[M+CH3COO]-': CH3COO,
    '[M+HCOO]-': HCOO,
    '[M+K-H+HCOO]-': ATOMIC_MASSES['K'] - ATOMIC_MASSES['H'] + HCOO,
    '[M+Na-H+HCOO]-': ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'] + HCOO,
    '[M-H2O+HCOO]-': -H2O + HCOO,
    '[M+CH3COONa-H]-': CH3COO + ATOMIC_MASSES['Na'] - ATOMIC_MASSES['H'],
    '[M-H+HCOOH]-': -ATOMIC_MASSES['H'] + (HCOO + ATOMIC_MASSES['H']),

    '[M-SO3-H]-': -SO3 - ATOMIC_MASSES['H'],
    '[M-SO3-H2O-H]-': -SO3 - H2O - ATOMIC_MASSES['H'],
    '[M-SO3-H2O+HCOO]-': -SO3 - H2O + HCOO,
    '[M-SO3-H2O+Cl]-': -SO3 - H2O + ATOMIC_MASSES['Cl'],
    '[M-SO3+Cl]-': -SO3 + ATOMIC_MASSES['Cl'],

    '[M-BrO]+': -BrO,
    '[M-ClO]+': -ClO,
    '[M-Br+O]-': -ATOMIC_MASSES['Br'] + ATOMIC_MASSES['O'],
    '[M-Cl+O]-': -ATOMIC_MASSES['Cl'] + ATOMIC_MASSES['O'],

    '[M+2H]2+': 2 * ATOMIC_MASSES['H'],
    '[M+3H]3+': 3 * ATOMIC_MASSES['H'],
    '[M+4H]4+': 4 * ATOMIC_MASSES['H'],
    '[M-2H]2-': -2 * ATOMIC_MASSES['H'],
    '[M+2K]2+': 2 * ATOMIC_MASSES['K'],
    '[M-2Cl]2+': -2 * ATOMIC_MASSES['Cl'],
}

ADDUCT_STANDARDIZATION = {
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
    "[M+FA-H]-": "[M+HCOO]-",
}

XGB_INTEGER_HYPERPARAMS = {"n_estimators", "max_depth"}
