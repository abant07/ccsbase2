# Compiled Dataset Analysis Notebooks

This folder contains exploratory data analysis notebooks for the **raw/uncleaned CCS dataset** before any filtering or preprocessing. These figures characterize the initial dataset composition, distributions, and quality.

---

## Notebook Overview

### 1. `dataset_summary.ipynb`
**Purpose:** High-level composition breakdown of the dataset

**Figures:**
| Figure | Type | Description |
|--------|------|-------------|
| Adduct Composition | Bar chart | Percentage breakdown of adduct types (e.g., [M+H]+, [M-H]-). Categories <1% grouped as "Others". X-axis: adduct names, Y-axis: percent of total datapoints. |
| Superclass Composition | Bar chart | Percentage breakdown by chemical superclass (e.g., Lipids, Organoheterocyclic). Categories <3% grouped as "Other". |
| Subclass Composition | Bar chart | Percentage breakdown by chemical subclass. Categories <3% grouped as "Other". |
| Database Entry Breakdown By Lab | Bar chart | Number of entries contributed by each data source/lab. Shows both percentage and raw count (n) per source. |

---

### 2. `mz_ccs_breakdown.ipynb`
**Purpose:** Distribution analysis of core molecular properties (m/z and CCS)

**Figures:**
| Figure | Type | Description |
|--------|------|-------------|
| m/z Histogram | Histogram | Distribution of mass-to-charge ratios across the dataset. Includes vertical lines for mean (red dashed) and median (orange solid) with legend values. |
| CCS Histogram | Histogram | Distribution of collision cross-section values. Includes mean and median markers with legend values. |
| Mass-CCS Distribution by (Subclass) | Scatter plot | All datapoints plotted with m/z on X-axis, CCS on Y-axis. Points colored by chemical superclass (tab20 colormap). Legend shows superclass categories. |
| Mass-CCS Distribution by Tag | Scatter plot | Same scatter but points colored by data source/laboratory origin. Shows which labs contributed data across different m/z and CCS ranges. |

---

### 3. `data_source_breakdown.ipynb`
**Purpose:** Statistical comparison across different data sources/laboratories

**Figures:**
| Figure | Type | Description |
|--------|------|-------------|
| CCS Distribution by Source | Box plot | Box and whisker plots comparing CCS distributions across different data sources. No outliers shown. X-axis: dataset/lab names, Y-axis: CCS values. Useful for identifying systematic differences between labs. |

---

### 4. `charge_state_analysis.ipynb`
**Purpose:** Analysis of positive vs negative ion mode data

**Figures:**
| Figure | Type | Description |
|--------|------|-------------|
| CCS Distribution by Polarity | KDE overlay plot | Kernel density estimation showing CCS distribution for positive ions (blue) overlaid with negative ions (red). Includes sample sizes in legend. Shows whether polarity affects CCS range. |

---

### 5. `adduct_ccs_breakdown.ipynb`
**Purpose:** Adduct-specific CCS analysis

**Figures:**
| Figure | Type | Description |
|--------|------|-------------|
| CCS Distribution by Adduct Type | Violin plot | CCS distributions for the 10 most common adducts. Ordered by median CCS. X-axis labels include adduct name and count (n). Shows distribution shape, useful for identifying adduct-specific CCS ranges. |
| Mass-CCS Distribution by Adduct Type | Scatter plot | m/z vs CCS with top 6 adducts colored distinctly (tab10 colormap), remaining adducts in gray background. Legend shows adduct names and counts. |

---

### 6. `features_pca_umap_breakdown.ipynb`
**Purpose:** Dimensionality reduction visualization of chemical space

**Figures:**
| Figure | Type | Description |
|--------|------|-------------|
| UMAP Projection | 2D scatter plot | UMAP dimensionality reduction of numerical features (mass, z, ccs). Points colored by superclass (tab20). Shows clustering of chemical space and how different superclasses occupy different regions. Axes labeled UMAP 1 and UMAP 2. |
| 3D Density Surface | 3D surface plot | Kernel density estimation of the UMAP embedding rendered as a 3D surface. Height represents density of datapoints. Viridis colormap. Shows "peaks" where chemical space is most densely populated. |

---

### 7. `comprehensive_dataset_analysis.ipynb`
**Purpose:** Complete publication-ready analysis combining all major visualizations

**Figures:**
| Figure | Type | Description |
|--------|------|-------------|
| Dataset Summary Table | Table | Key metrics: total entries, unique compounds, adduct types, superclasses, subclasses, data sources, m/z range, CCS range. |
| m/z and CCS Distributions | Side-by-side histograms | Two-panel figure with m/z distribution (left) and CCS distribution (right). Median markers included. |
| m/z vs CCS Correlation | Scatter + regression | Scatter plot with linear regression line. Shows equation (y = slope*x + intercept), R² value, and Pearson correlation. |
| CCS Box Plots by Source | Box plot | CCS distributions per data source, styled with consistent colors. Labels include sample sizes. |
| Charge State Analysis | Two-panel figure | Left: bar chart of charge states with count labels. Right: donut/pie chart showing positive vs negative ion percentages. |
| CCS by Polarity | Histogram overlay | Overlapping histograms comparing CCS distributions between positive and negative ions. |
| CCS Violin by Adduct | Violin plot | Top 10 adducts ordered by median CCS. Shows distribution shape and width. |
| m/z vs CCS by Adduct | Scatter plot | Top 6 adducts colored, others in gray. Includes legend with sample sizes. |
| Train/Test Split Validation | 4-panel figure | Compares train vs test distributions: m/z density, CCS density, top 15 subclasses (grouped bar), top 10 adducts (grouped bar). |
| Tanimoto Similarity Distribution | Histogram | Pairwise Tanimoto similarities from Morgan fingerprints. Shows chemical diversity—lower values indicate more diverse dataset. Mean and median marked. |
| Missing Data Heatmap | Heatmap | Visual pattern of missing values across columns. Green = present, Red = missing. Percentages annotated. |
| Data Completeness | Horizontal bar chart | Completeness percentage per column. Green = 100%, yellow = >90%, red = <90%. |

---

