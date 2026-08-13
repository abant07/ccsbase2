import sys
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import shap
import joblib
import xgboost as xgb
from PIL import Image

st.set_page_config(layout="wide", page_title="CCS Fingerprint Explorer", page_icon="🧪")

# ── Config ─────────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from utils import (  # noqa: E402
    calculate_base_features,
    calculate_sparse_fingerprint,
    load_or_build_fingerprint_vocabulary,
    build_fingerprint_index,
    build_feature_matrix_sparse,
    train_test_split_custom,
)


def _resolve_existing_file(description: str, candidates: tuple[Path, ...]) -> str:
    """Pick first existing path (Streamlit Cloud cwd varies; assets may live in several layouts)."""
    for path in candidates:
        if path.is_file():
            return str(path)
    lines = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"{description} not found. Checked:\n  {lines}\n"
        f"Add {description} to the deployed repo (README download links). "
        "If it is listed in .gitignore, remove that entry or place a copy Streamlit can clone."
    )


DB_PATH = _resolve_existing_file(
    "CCSMLDatabase.db",
    (
        _REPO_ROOT / "ccsbase2" / "CCSMLDatabase.db",
        _REPO_ROOT / "datasets" / "CCSMLDatabase.db",
        _REPO_ROOT / "CCSMLDatabase.db",
        _APP_DIR / "CCSMLDatabase.db",
    ),
)
MODEL_PATH = _resolve_existing_file(
    "ccsbase2.joblib",
    (
        _REPO_ROOT / "ccsbase2" / "ccsbase2.joblib",
        _REPO_ROOT / "ccsbase2.joblib",
        _APP_DIR / "ccsbase2.joblib",
    ),
)
FP_VOCAB_PATH = _resolve_existing_file(
    "ccsbase2_fp_vocab.joblib",
    (
        _REPO_ROOT / "ccsbase2" / "ccsbase2_fp_vocab.joblib",
        _REPO_ROOT / "ccsbase2_fp_vocab.joblib",
        _APP_DIR / "ccsbase2_fp_vocab.joblib",
    ),
)
ADDUCTS_PATH = _resolve_existing_file(
    "ccsbase2_adduct_list.joblib",
    (
        _REPO_ROOT / "ccsbase2" / "ccsbase2_adduct_list.joblib",
        _REPO_ROOT / "ccsbase2_adduct_list.joblib",
        _APP_DIR / "ccsbase2_adduct_list.joblib",
    ),
)
IMG_W, IMG_H = 400, 350
WATERFALL_W  = 500
PAGE_SIZE    = 5
SHAP_MAX_DISPLAY = 12  # how many features the waterfall plot shows -- also drives the bit-picker candidates

BIT_COLORS = [
    (0.95, 0.25, 0.25),
    (0.20, 0.55, 0.95),
    (0.15, 0.75, 0.30),
    (0.95, 0.55, 0.05),
    (0.70, 0.15, 0.85),
]
BIT_COLOR_HEX = ["#F24040", "#3399F2", "#27BF4D", "#F28C0D", "#B326D9"]


class _BoosterWrapper:
    """Minimal stand-in for pickled sklearn XGBRegressor (only get_booster() is used)."""

    __slots__ = ("_booster",)

    def __init__(self, booster: xgb.Booster):
        self._booster = booster

    def get_booster(self):
        return self._booster


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Prefer ccsbase2.json beside the joblib if present (avoids XGBoost pickle warnings)."""
    joblib_path = Path(MODEL_PATH)
    native_path = joblib_path.with_suffix(".json")
    if native_path.is_file():
        booster = xgb.Booster()
        booster.load_model(str(native_path))
        return _BoosterWrapper(booster)
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_fp_vocab():
    return load_or_build_fingerprint_vocabulary(DB_PATH, FP_VOCAB_PATH)

@st.cache_resource
def load_adducts():
    return joblib.load(ADDUCTS_PATH)

@st.cache_data
def load_test_set():
    """Same split train.py uses (same seed) so the explorer only ever shows held-out molecules."""
    train_test_split_custom(
        database_file=DB_PATH,
        test_size=0.2,
        random_state=26,
        use_metlin=True,
    )
    return pd.read_csv("test_data.csv")

# ── Fingerprint generator (bit-info only; actual model features come from utils.py) ────────────
morgan_bitinfo_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True)

def sanitize(name):
    return (name.replace("[", "").replace("]", "").replace("<", "")
                .replace("+", "plus").replace("-", "minus"))

def featurise_all(df_sub, adducts, fp_index):
    """Featurise every molecule using the exact same feature functions as train.py. Returns X (sparse) + parallel lists."""
    base_feats, fp_dicts, mols, bit_infos, valid_rows = [], [], [], [], []
    for _, row in df_sub.iterrows():
        mol = Chem.MolFromSmiles(row["smi"])
        if mol is None:
            continue
        mol = Chem.AddHs(mol)

        base_feats.append(calculate_base_features(row["smi"], row["mass"], adducts, row["adduct"]))
        fp_dicts.append(calculate_sparse_fingerprint(row["smi"]))

        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        morgan_bitinfo_fpgen.GetSparseCountFingerprint(mol, additionalOutput=ao)

        mols.append(mol)
        bit_infos.append(ao.GetBitInfoMap())
        valid_rows.append(row)

    X_full = build_feature_matrix_sparse(base_feats, fp_dicts, fp_index)
    return X_full, mols, bit_infos, pd.DataFrame(valid_rows).reset_index(drop=True)

def compute_shap_batch(booster, X_batch, feature_names):
    dm = xgb.DMatrix(X_batch, feature_names=feature_names)
    shap_matrix = booster.predict(dm, pred_contribs=True)
    return shap_matrix[:, :-1], float(shap_matrix[0, -1])

def ensure_shap(start, end):
    """Compute SHAP only for indices in [start, end) not yet cached."""
    missing = [i for i in range(start, end) if i not in st.session_state.shap_cache]
    if not missing:
        return
    X_batch = st.session_state.X_full[missing]
    shap_vals, _ = compute_shap_batch(booster, X_batch, feature_names)
    for local_i, global_i in enumerate(missing):
        st.session_state.shap_cache[global_i] = shap_vals[local_i]

def top_fp_bits_for_shap(shap_row, fp_vocab, fp_start_idx, max_display):
    """Env-ids of the fingerprint features among this molecule's top |SHAP| contributors (same set the waterfall shows)."""
    top_idx = np.argsort(-np.abs(shap_row))[:max_display]
    return [fp_vocab[i - fp_start_idx] for i in top_idx if i >= fp_start_idx]

def draw_molecule(mol, active_bits, bit_info, size=(IMG_W, IMG_H)):
    highlight_atoms, highlight_bonds = {}, {}
    for bit_idx, bit in enumerate(active_bits):
        if bit not in bit_info:
            continue
        color = BIT_COLORS[bit_idx % len(BIT_COLORS)]
        for atom_idx, radius in bit_info[bit]:
            env  = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            amap = {}
            Chem.PathToSubmol(mol, env, atomMap=amap)
            for a in amap.keys():
                if a not in highlight_atoms:
                    highlight_atoms[a] = color
            for b in env:
                if b not in highlight_bonds:
                    highlight_bonds[b] = color

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareMolForDrawing(mol)
    if highlight_atoms:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(highlight_atoms.keys()),
            highlightBonds=list(highlight_bonds.keys()),
            highlightAtomColors=highlight_atoms,
            highlightBondColors=highlight_bonds,
        )
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")

def make_waterfall(shap_row, global_base, subclass_base, feature_names,
                   true_ccs, pred_ccs, size=(WATERFALL_W, 420)):
    shift    = global_base - subclass_base
    adjusted = shap_row.copy()
    adjusted[0] += shift
    exp = shap.Explanation(
        values=adjusted,
        base_values=subclass_base,
        feature_names=feature_names,
    )
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100))
    shap.plots.waterfall(exp, max_display=SHAP_MAX_DISPLAY, show=False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Session state init ─────────────────────────────────────────────────────────
defaults = {
    "computed_subclass": None,
    "X_full":            None,
    "mols":              None,
    "bit_infos":         None,
    "df_pool":           None,
    "global_base":       None,
    "subclass_base":     None,
    "shap_cache":        {},
    "pred_cache":        None,
    "n_shown":           PAGE_SIZE,
    "active_bits":       [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Static data ────────────────────────────────────────────────────────────────
model      = load_model()
booster    = model.get_booster()
adducts    = load_adducts()
fp_vocab   = load_fp_vocab()
fp_index   = build_fingerprint_index(fp_vocab)
test_df    = load_test_set()

df_counts = (
    test_df["subclass"].value_counts()
    .rename_axis("subclass")
    .reset_index(name="cnt")
)

base_feature_names = (
    ["MolecularWeight", "AdductMass"]
    + [f"Adduct_{sanitize(a)}" for a in adducts]
    + ["Adduct_other"]
)
feature_names = base_feature_names + [f"FP_{env_id}" for env_id in fp_vocab]
fp_start_idx  = len(base_feature_names)

# ── Left sidebar ───────────────────────────────────────────────────────────────
# Cap the subclass radio list's own height so it scrolls internally, keeping the
# title/caption/Apply button above it always on screen instead of scrolling away.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] [data-testid="stRadio"] {
        max-height: calc(100vh - 260px);
        overflow-y: auto;
        padding-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🔬 Subclass")
    st.caption("Select a subclass below, then click Apply above. Molecules are drawn from the held-out test set.")
    subclass_labels = [
        f"{row['subclass']}  ({int(row['cnt'])})"
        for _, row in df_counts.iterrows()
    ]
    apply_subclass = st.button("Apply Subclass", use_container_width=True)
    selected_label = st.radio("Subclass", subclass_labels, label_visibility="collapsed")

selected_subclass = selected_label.rsplit("  (", 1)[0]

# ── Layout ─────────────────────────────────────────────────────────────────────
body_col, right_col = st.columns([5, 1], gap="large")

# ── Apply Subclass ─────────────────────────────────────────────────────────────
if apply_subclass and selected_subclass != st.session_state.computed_subclass:
    with body_col:
        with st.spinner(f"Featurising all molecules in **{selected_subclass}**…"):
            df_sub = test_df[test_df["subclass"] == selected_subclass].reset_index(drop=True)
            X_full, mols, bit_infos, df_pool = featurise_all(df_sub, adducts, fp_index)

        with st.spinner("Computing predictions for subclass baseline…"):
            # Fast regular predict over all mols to get subclass baseline
            dm_all   = xgb.DMatrix(X_full, feature_names=feature_names)
            preds    = booster.predict(dm_all)
            sub_base = float(preds.mean())

            # Get global base value from a single pred_contribs call
            dm_one      = xgb.DMatrix(X_full[:1], feature_names=feature_names)
            shap_one    = booster.predict(dm_one, pred_contribs=True)
            global_base = float(shap_one[0, -1])

        st.session_state.computed_subclass = selected_subclass
        st.session_state.X_full            = X_full
        st.session_state.mols              = mols
        st.session_state.bit_infos         = bit_infos
        st.session_state.df_pool           = df_pool
        st.session_state.global_base       = global_base
        st.session_state.subclass_base     = sub_base
        st.session_state.pred_cache        = preds
        st.session_state.shap_cache        = {}       # clear old SHAP cache
        st.session_state.n_shown           = PAGE_SIZE

# ── SHAP for the currently visible window (needed by both the bits panel and the body) ─────────
n_shown = 0
if st.session_state.computed_subclass is not None:
    n_shown = min(st.session_state.n_shown, len(st.session_state.mols))
    with body_col:
        with st.spinner(f"Computing SHAP for molecules 1–{n_shown}…"):
            ensure_shap(0, n_shown)

# ── Right sidebar: Fingerprint Bits ─────────────────────────────────────────────
with right_col:
    st.markdown("### 🧩 Fingerprint Bits")
    st.caption(
        "Candidates are the top SHAP-contributing substructures among the molecules "
        "currently loaded below. Select up to 5 to highlight."
    )

    candidate_bits, seen = [], set()
    for i in range(n_shown):
        shap_row = st.session_state.shap_cache.get(i)
        if shap_row is None:
            continue
        for bit in top_fp_bits_for_shap(shap_row, fp_vocab, fp_start_idx, SHAP_MAX_DISPLAY):
            if bit not in seen:
                seen.add(bit)
                candidate_bits.append(bit)

    if not candidate_bits:
        st.caption("Apply a subclass to see candidate bits.")
        raw_selected = []
    else:
        default_bits = [b for b in st.session_state.active_bits if b in candidate_bits]
        raw_selected = st.multiselect(
            "Bits", options=candidate_bits,
            default=default_bits,
            format_func=lambda x: f"FP_{x}",
            label_visibility="collapsed",
        )
    if len(raw_selected) > 5:
        st.warning("Max 5 — only first 5 used.")
    if st.button("Apply Bits", use_container_width=True):
        st.session_state.active_bits = raw_selected[:5]

    if st.session_state.active_bits:
        st.markdown("**Legend:**")
        for i, bit in enumerate(st.session_state.active_bits):
            st.markdown(
                f'<span style="color:{BIT_COLOR_HEX[i]};font-size:18px">■</span> FP_{bit}',
                unsafe_allow_html=True,
            )

# ── Body ───────────────────────────────────────────────────────────────────────
with body_col:
    st.markdown("# 🧪 CCS Fingerprint Explorer")

    if st.session_state.computed_subclass is None:
        st.info("👈  Select a subclass and click **Apply Subclass** to begin.")
    else:
        n        = len(st.session_state.mols)
        sub_base = st.session_state.subclass_base

        st.markdown(f"### {st.session_state.computed_subclass}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total molecules (test set)", n)
        c2.metric("Showing",               n_shown)
        c3.metric("Subclass baseline CCS", f"{sub_base:.2f} Å²")
        c4.metric("Bits highlighted",      len(st.session_state.active_bits))
        st.divider()

        active_bits = st.session_state.active_bits
        df_pool     = st.session_state.df_pool
        preds       = st.session_state.pred_cache

        for i in range(n_shown):
            mol      = st.session_state.mols[i]
            bit_info = st.session_state.bit_infos[i]
            shap_row = st.session_state.shap_cache[i]
            true_ccs = float(df_pool.loc[i, "ccs"])
            pred_ccs = float(preds[i])
            err      = pred_ccs - true_ccs
            smi      = df_pool.loc[i, "smi"]

            label = (f"Molecule {i+1}  |  True: {true_ccs:.2f}  |  "
                     f"Pred: {pred_ccs:.2f}  |  Err: {err:+.2f}")

            with st.expander(label, expanded=True):
                mol_col, wf_col = st.columns([1, 1], gap="medium")

                with mol_col:
                    mol_img = draw_molecule(mol, active_bits, bit_info, size=(IMG_W, IMG_H))
                    st.image(mol_img, use_container_width=True)
                    st.caption(smi[:80] + ("…" if len(smi) > 80 else ""))

                    for bit_idx, bit in enumerate(active_bits):
                        if bit in bit_info:
                            continue
                        st.markdown(
                            f'<span style="color:{BIT_COLOR_HEX[bit_idx % len(BIT_COLOR_HEX)]};'
                            f'font-size:14px">■</span> '
                            f'<span style="font-size:13px">FP_{bit} not found in this compound — '
                            f"the model may be using its absence as a signal.</span>",
                            unsafe_allow_html=True,
                        )

                with wf_col:
                    wf_buf = make_waterfall(
                        shap_row,
                        st.session_state.global_base,
                        sub_base,
                        feature_names,
                        true_ccs, pred_ccs,
                        size=(WATERFALL_W, 420),
                    )
                    st.image(wf_buf, use_container_width=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Subclass baseline", f"{sub_base:.2f}")
                    m2.metric("True CCS",          f"{true_ccs:.2f}")
                    m3.metric("Predicted CCS",     f"{pred_ccs:.2f}",
                              delta=f"{err:+.2f}", delta_color="inverse")

        st.divider()

        if n_shown < n:
            remaining = n - n_shown
            if st.button(
                f"⬇  Load {min(PAGE_SIZE, remaining)} more  ({n_shown} / {n} shown)",
                use_container_width=True,
            ):
                st.session_state.n_shown += PAGE_SIZE
                st.rerun()
        else:
            st.success(f"✅  All {n} molecules shown.")
