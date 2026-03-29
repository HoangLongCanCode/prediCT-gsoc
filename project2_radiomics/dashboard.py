"""
dashboard.py
Interactive clinical dashboard for coronary calcium phenotype analysis.

Shows for each patient:
  - CT slice viewer with calcium highlighted
  - Density color map (HU values colored by risk)
  - Per-lesion breakdown (count, sizes, distribution)
  - Phenotype card (cluster assignment + confidence via GMM)
  - Auto-generated clinical narrative

Usage
-----
    streamlit run project2_radiomics/dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Calcium Phenotype Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────

IMAGES_ROOT    = Path(r"D:\Du_hoc\gsoc\processed\images")
RESULTS_DIR    = Path(r"D:\Du_hoc\gsoc\project2_radiomics\results")
SCAN_INDEX_CSV = Path(r"D:\Du_hoc\gsoc\processed\tables\scan_index.csv")

DENSITY_BINS = {
    "Low (130-200 HU)":      (130, 200),
    "Mild (200-300 HU)":     (200, 300),
    "Moderate (300-400 HU)": (300, 400),
    "High (400+ HU)":        (400, 3000),
}

DENSITY_COLORS_HEX = {
    "Low (130-200 HU)":      "#F44336",
    "Mild (200-300 HU)":     "#FF9800",
    "Moderate (300-400 HU)": "#2196F3",
    "High (400+ HU)":        "#4CAF50",
}

AGATSTON_LABELS = {
    0: "None (0)",
    1: "Mild (1-99)",
    2: "Moderate (100-399)",
    3: "Severe (≥400)",
}

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_scan_index():
    df = pd.read_csv(SCAN_INDEX_CSV)
    df["patient_id"] = df["patient_id"].astype(str)
    df = df[df["voxels"] > 0].copy()
    def cat(v):
        if v <= 500:  return 1
        if v <= 2000: return 2
        return 3
    df["agatston_category"] = df["voxels"].apply(cat)
    df["agatston_label"]    = df["agatston_category"].map(AGATSTON_LABELS)
    return df


@st.cache_data
def load_density_features():
    path = RESULTS_DIR / "density_features.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_per_lesion_features():
    path = RESULTS_DIR / "per_lesion_features.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["patient_id"] = df["patient_id"].astype(str)
        return df
    return None


@st.cache_data
def fit_gmm(n_components=3):
    """Fit GMM on density features for phenotype assignment."""
    df = load_density_features()
    if df is None or len(df) < n_components:
        return None, None, None

    feat_cols = [
        "low_density_pct", "mild_density_pct",
        "moderate_density_pct", "high_density_pct",
        "mean_hu", "density_risk_index",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
    gmm.fit(X_scaled)

    return gmm, scaler, feat_cols


def get_phenotype_name(cluster_id: int, df: pd.DataFrame) -> tuple[str, str, str]:
    """Assign clinical phenotype name based on cluster characteristics."""
    sub = df[df["gmm_cluster"] == cluster_id]
    if len(sub) == 0:
        return "Unknown", "grey", "No data"

    low_pct  = sub["low_density_pct"].mean()
    high_pct = sub["high_density_pct"].mean()
    mean_hu  = sub["mean_hu"].mean()

    all_low  = df["low_density_pct"].mean()
    all_high = df["high_density_pct"].mean()

    if low_pct > all_low * 1.2:
        return "Vulnerable Spotty", "#F44336", "Predominantly low-density calcium — associated with plaque vulnerability"
    elif high_pct > all_high * 1.3:
        return "Dense Stable", "#4CAF50", "Predominantly high-density calcium — associated with stable, calcified plaques"
    else:
        return "Mixed Pattern", "#FF9800", "Mixed density profile — intermediate risk characteristics"


@st.cache_data
def load_nifti(scan_id: str):
    img_path = IMAGES_ROOT / scan_id / f"{scan_id}_img.nii.gz"
    seg_path = IMAGES_ROOT / scan_id / f"{scan_id}_seg.nii.gz"
    if not img_path.exists():
        return None, None
    img = sitk.ReadImage(str(img_path))
    seg = sitk.ReadImage(str(seg_path))
    img_arr = sitk.GetArrayFromImage(img).astype(np.float32)
    seg_arr = sitk.GetArrayFromImage(seg).astype(np.uint8)
    return img_arr, seg_arr


# ── CT Viewer ─────────────────────────────────────────────────────────────────

def get_best_calcium_slice(seg_arr):
    """Return the slice index with the most calcium voxels."""
    calcium_per_slice = seg_arr.sum(axis=(1, 2))
    if calcium_per_slice.max() == 0:
        return seg_arr.shape[0] // 2
    return int(np.argmax(calcium_per_slice))


def plot_ct_slice(img_arr, seg_arr, slice_idx, view="axial", show_density=False, show_overlay=True):
    """
    Render a CT slice.
    show_overlay  : if False → raw CT only (no red, no arrows, no labels)
                    if True + show_density=False → red calcium overlay + arrows
                    if True + show_density=True  → density color map + arrows
    Zoom inset is ALWAYS shown when calcium is present.
    view: axial (Z), coronal (Y), sagittal (X)
    """
    if view == "axial":
        img_slice = img_arr[slice_idx]
        seg_slice = seg_arr[slice_idx]
    elif view == "coronal":
        img_slice = img_arr[:, slice_idx, :]
        seg_slice = seg_arr[:, slice_idx, :]
    else:  # sagittal
        img_slice = img_arr[:, :, slice_idx]
        seg_slice = seg_arr[:, :, slice_idx]

    has_calcium = np.any(seg_slice == 1)
    density_overlay = None

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    ax.set_facecolor("black")

    windowed = np.clip(img_slice, -200, 1000)
    windowed = (windowed + 200) / 1200
    ax.imshow(windowed, cmap="gray", aspect="equal")

    if show_overlay and has_calcium:
        if show_density:
            density_overlay = np.zeros((*img_slice.shape, 4))
            calcium_mask = seg_slice == 1
            colors_rgba = {
                (130, 200): [0.96, 0.26, 0.21, 0.85],
                (200, 300): [1.0,  0.60, 0.0,  0.85],
                (300, 400): [0.13, 0.59, 0.95, 0.85],
                (400, 3000):[0.30, 0.69, 0.31, 0.85],
            }
            for (hu_min, hu_max), rgba in colors_rgba.items():
                mask = calcium_mask & (img_slice >= hu_min) & (img_slice < hu_max)
                density_overlay[mask] = rgba
            ax.imshow(density_overlay, aspect="equal")
            legend_elements = [
                Patch(facecolor="#F44336", label="Low (130-200 HU) — Risky"),
                Patch(facecolor="#FF9800", label="Mild (200-300 HU)"),
                Patch(facecolor="#2196F3", label="Moderate (300-400 HU)"),
                Patch(facecolor="#4CAF50", label="High (400+ HU) — Protective"),
            ]
            ax.legend(handles=legend_elements, loc="lower left",
                      fontsize=7, framealpha=0.7,
                      facecolor="black", labelcolor="white")
        else:
            calcium_overlay = np.zeros((*img_slice.shape, 4))
            calcium_overlay[seg_slice == 1] = [1, 0, 0, 0.75]
            ax.imshow(calcium_overlay, aspect="equal")

        # Arrows + Ca2+ labels only when overlay is on
        from sklearn.cluster import DBSCAN
        import matplotlib.patches as mpatches
        rows, cols = np.where(seg_slice == 1)
        try:
            coords = np.column_stack([rows, cols])
            db = DBSCAN(eps=8, min_samples=1).fit(coords)
            db_labels = db.labels_
            for lbl in set(db_labels):
                if lbl == -1:
                    continue
                cluster_coords = coords[db_labels == lbl]
                cy, cx = cluster_coords.mean(axis=0)
                ax.annotate(
                    "Ca²⁺",
                    xy=(cx, cy), xytext=(cx + 20, cy - 20),
                    color="yellow", fontsize=7, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2),
                )
        except Exception:
            pass

    # ── Zoom inset ALWAYS shown when calcium present ──────────────────────────
    if has_calcium:
        import matplotlib.patches as mpatches
        rows, cols = np.where(seg_slice == 1)
        cy_mean, cx_mean = rows.mean(), cols.mean()
        zoom_size = 60
        y1 = max(0, int(cy_mean - zoom_size))
        y2 = min(img_slice.shape[0], int(cy_mean + zoom_size))
        x1 = max(0, int(cx_mean - zoom_size))
        x2 = min(img_slice.shape[1], int(cx_mean + zoom_size))

        rect = mpatches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1.5, edgecolor="yellow", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)

        axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
        axins.imshow(windowed[y1:y2, x1:x2], cmap="gray", aspect="equal")

        # Zoom overlay mirrors main image overlay
        if show_overlay and show_density and density_overlay is not None:
            axins.imshow(density_overlay[y1:y2, x1:x2], aspect="equal")
        elif show_overlay and not show_density:
            zoom_ov = np.zeros((y2-y1, x2-x1, 4))
            zoom_ov[seg_slice[y1:y2, x1:x2] == 1] = [1, 0, 0, 0.75]
            axins.imshow(zoom_ov, aspect="equal")
        # if show_overlay=False → zoom shows raw CT only (no overlay)

        axins.set_title("Zoom", color="yellow", fontsize=7, pad=2)
        axins.axis("off")
        for spine in axins.spines.values():
            spine.set_edgecolor("yellow")
            spine.set_linewidth(1.5)

    ax.set_title(f"Axial slice {slice_idx}", color="white", fontsize=10)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── Clinical narrative ────────────────────────────────────────────────────────

def generate_narrative(
    patient_id:       str,
    agatston_label:   str,
    density_row:      object,
    lesion_row:       object,
    phenotype_name:   str,
    phenotype_conf:   float,
    density_df:       object,
) -> str:
    """Generate a plain-English clinical summary for this patient."""

    lines = []
    lines.append(f"**Patient {patient_id} — Calcium Phenotype Report**\n")
    lines.append(f"**Agatston Category:** {agatston_label}")
    lines.append(f"**Assigned Phenotype:** {phenotype_name} (confidence: {phenotype_conf:.0%})\n")

    # Density section
    if density_row is not None:
        low_pct  = density_row.get("low_density_pct", 0)
        high_pct = density_row.get("high_density_pct", 0)
        mean_hu  = density_row.get("mean_hu", 0)
        dri      = density_row.get("density_risk_index", 0)

        # Percentile of DRI in cohort
        if density_df is not None and "density_risk_index" in density_df.columns:
            pct_rank = scipy_stats.percentileofscore(
                density_df["density_risk_index"], dri
            )
        else:
            pct_rank = 50.0

        lines.append("**Density Profile:**")
        lines.append(
            f"- Low-density calcium (130–200 HU): **{low_pct:.1f}%** of total calcium"
        )
        lines.append(
            f"- High-density calcium (>400 HU): **{high_pct:.1f}%** of total calcium"
        )
        lines.append(f"- Mean calcium HU: **{mean_hu:.1f}**")
        lines.append(
            f"- Density Risk Index: **{dri:.1f}%** "
            f"(top {100-pct_rank:.0f}% of cohort)\n"
        )

        # Clinical interpretation
        if low_pct > 60:
            lines.append(
                "> ⚠️ **High proportion of low-density calcium.** "
                "Prior literature (Criqui et al., JAMA 2014) associates "
                "low-density spotty calcium with elevated plaque vulnerability. "
                "Consider closer monitoring."
            )
        elif high_pct > 25:
            lines.append(
                "> ✅ **Predominantly dense calcium.** "
                "High-density calcium (>400 HU) is associated with stable, "
                "fibrocalcific plaques and paradoxically lower event rates "
                "(Criqui et al., JAMA 2014; Circulation 2023)."
            )
        else:
            lines.append(
                "> ℹ️ **Mixed density profile.** "
                "Intermediate calcium density distribution. "
                "Full clinical context recommended."
            )

    # Lesion section
    if lesion_row is not None:
        n_lesions   = int(lesion_row.get("lesion_count", 0))
        mean_size   = lesion_row.get("mean_lesion_size", 0)
        size_cv     = lesion_row.get("lesion_size_cv", 0)

        lines.append("\n**Lesion Distribution:**")
        lines.append(f"- Number of discrete calcium lesions: **{n_lesions}**")
        lines.append(f"- Average lesion size: **{mean_size:.0f} voxels**")
        lines.append(
            f"- Lesion size variability (CV): **{size_cv:.2f}** "
            f"({'high' if size_cv > 1.5 else 'moderate' if size_cv > 0.8 else 'low'})"
        )

        if n_lesions > 15:
            lines.append(
                "\n> ⚠️ **High lesion count.** Multiple discrete lesions suggest "
                "diffuse multi-vessel disease. The Nature calcium-omics study "
                "(Hoori et al. 2024) found lesion count to be among the strongest "
                "MACE predictors beyond Agatston score."
            )
        elif n_lesions <= 3:
            lines.append(
                "\n> ℹ️ **Focal calcium pattern.** Few discrete lesions suggest "
                "localized rather than diffuse disease."
            )

    lines.append(
        "\n---\n*This report is AI-generated and descriptive only. "
        "It does not constitute a clinical diagnosis. "
        "All findings should be interpreted by a qualified cardiologist.*"
    )

    return "\n".join(lines)


# ── Main dashboard ────────────────────────────────────────────────────────────

def main():
    st.title("🫀 Calcium Phenotype Dashboard")
    st.caption(
        "Coronary Calcium CT Analysis — Beyond the Agatston Score | PrediCT GSoC"
    )

    # Load data
    scan_df      = load_scan_index()
    density_df   = load_density_features()
    lesion_df    = load_per_lesion_features()
    gmm, scaler, feat_cols = fit_gmm(n_components=3)

    if density_df is not None and gmm is not None:
        # Assign GMM clusters to density features
        X = density_df[feat_cols].fillna(0).values
        X_scaled = scaler.transform(X)
        probs    = gmm.predict_proba(X_scaled)
        density_df["gmm_cluster"] = gmm.predict(X_scaled)
        density_df["gmm_confidence"] = probs.max(axis=1)
        density_df["patient_id"] = density_df["patient_id"].astype(str)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Patient Selection")

    # Filter by Agatston category
    cat_filter = st.sidebar.selectbox(
        "Filter by Agatston Category",
        ["All"] + list(AGATSTON_LABELS.values())
    )

    display_df = scan_df.copy()
    if cat_filter != "All":
        display_df = display_df[display_df["agatston_label"] == cat_filter]

    patient_ids = sorted(display_df["patient_id"].tolist(),
                         key=lambda x: int(x) if x.isdigit() else 0)

    selected_pid = st.sidebar.selectbox(
        "Select Patient ID",
        patient_ids,
        format_func=lambda x: f"Patient {x}"
    )

    # Get patient info
    patient_row  = scan_df[scan_df["patient_id"] == selected_pid].iloc[0]
    scan_id      = patient_row["scan_id"]
    agatston_cat = int(patient_row["agatston_category"])
    agatston_lbl = patient_row["agatston_label"]

    density_row = None
    if density_df is not None:
        density_df["patient_id"] = density_df["patient_id"].astype(str)
        matches = density_df[density_df["patient_id"] == selected_pid]
        if len(matches) > 0:
            density_row = matches.iloc[0]

    lesion_row = None
    if lesion_df is not None:
        matches = lesion_df[lesion_df["patient_id"] == selected_pid]
        if len(matches) > 0:
            lesion_row = matches.iloc[0]

    # Phenotype
    phenotype_name = "N/A"
    phenotype_conf = 0.0
    phenotype_desc = ""
    phenotype_color = "grey"

    if density_row is not None and gmm is not None:
        cluster_id     = int(density_row.get("gmm_cluster", 0))
        phenotype_conf = float(density_row.get("gmm_confidence", 0))
        phenotype_name, phenotype_color, phenotype_desc = get_phenotype_name(
            cluster_id, density_df
        )

    # ── Header metrics ────────────────────────────────────────────────────────
    st.subheader(f"Patient {selected_pid}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Agatston Category", agatston_lbl)
    with col2:
        st.metric("Total Calcium Voxels", f"{patient_row['voxels']:,}")
    with col3:
        if density_row is not None:
            st.metric("Density Risk Index",
                      f"{density_row.get('density_risk_index', 0):.1f}%")
        else:
            st.metric("Density Risk Index", "N/A")
    with col4:
        if lesion_row is not None:
            st.metric("Lesion Count",
                      f"{int(lesion_row.get('lesion_count', 0))}")
        else:
            st.metric("Lesion Count", "N/A")

    st.divider()

    # ── Main content ──────────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2])

    with left_col:
        # CT Viewer
        st.subheader("CT Viewer")
        img_arr, seg_arr = load_nifti(scan_id)

        if img_arr is not None:
            # Overlay mode
            overlay_mode = st.radio(
                "Overlay",
                ["No overlay (raw CT)", "Calcium highlighted (red)", "Density color map"],
                horizontal=True,
                index=1,
            )
            show_density_map = (overlay_mode == "Density color map")
            show_overlay     = (overlay_mode != "No overlay (raw CT)")

            # Find best calcium slice and show axial view
            calcium_slices = np.where(seg_arr.sum(axis=(1,2)) > 0)[0]
            best_slice = get_best_calcium_slice(seg_arr)
            n = img_arr.shape[0]
            slice_idx = st.slider("Slice", int(n*0.1), int(n*0.9), best_slice)

            fig = plot_ct_slice(img_arr, seg_arr, slice_idx,
                                view="axial",
                                show_density=show_density_map,
                                show_overlay=show_overlay)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            if len(calcium_slices) > 0:
                caption = f"Calcium found in {len(calcium_slices)} slices. Auto-jumped to slice {best_slice}. "
                if overlay_mode == "No overlay (raw CT)": caption += "Raw CT — no overlay."
                elif overlay_mode == "Calcium highlighted (red)": caption += "Red = calcium deposits."
                else: caption += "Colors = HU density bins (red=risky, green=protective)."
                st.caption(caption)
        else:
            st.warning("CT image not found for this patient.")

    with right_col:
        # Phenotype Card
        st.subheader("Phenotype Classification")
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {phenotype_color}22, {phenotype_color}11);
                border-left: 4px solid {phenotype_color};
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 16px;
            ">
                <h3 style="color: {phenotype_color}; margin: 0 0 8px 0;">
                    {phenotype_name}
                </h3>
                <p style="margin: 0 0 8px 0; font-size: 0.9em; color: #666;">
                    {phenotype_desc}
                </p>
                <p style="margin: 0; font-weight: bold;">
                    Confidence: {phenotype_conf:.0%}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Density Breakdown
        if density_row is not None:
            st.subheader("Density Profile")
            bin_names = list(DENSITY_BINS.keys())
            pct_cols  = ["low_density_pct", "mild_density_pct",
                         "moderate_density_pct", "high_density_pct"]
            pct_vals  = [density_row.get(c, 0) for c in pct_cols]
            colors    = list(DENSITY_COLORS_HEX.values())

            fig2, ax2 = plt.subplots(figsize=(5, 3))
            bars = ax2.barh(bin_names, pct_vals, color=colors, edgecolor="white")
            ax2.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
            ax2.set_xlabel("% of Calcium Voxels")
            ax2.set_xlim(0, max(pct_vals) * 1.2 + 5)
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # Lesion Breakdown
        if lesion_row is not None:
            st.subheader("Lesion Analysis")
            n_lesions  = int(lesion_row.get("lesion_count", 0))
            mean_size  = lesion_row.get("mean_lesion_size", 0)
            max_size   = lesion_row.get("max_lesion_size", 0)
            size_cv    = lesion_row.get("lesion_size_cv", 0)

            lcol1, lcol2 = st.columns(2)
            with lcol1:
                st.metric("Lesion Count", n_lesions)
                st.metric("Avg Size", f"{mean_size:.0f} vox")
            with lcol2:
                st.metric("Max Size", f"{max_size:.0f} vox")
                st.metric("Size CV", f"{size_cv:.2f}")

    st.divider()

    # ── Clinical Narrative ────────────────────────────────────────────────────
    st.subheader("📋 Clinical Narrative")
    narrative = generate_narrative(
        patient_id      = selected_pid,
        agatston_label  = agatston_lbl,
        density_row     = density_row,
        lesion_row      = lesion_row,
        phenotype_name  = phenotype_name,
        phenotype_conf  = phenotype_conf,
        density_df      = density_df,
    )
    st.markdown(narrative)

    # ── Cohort context ────────────────────────────────────────────────────────
    if density_df is not None:
        st.divider()
        st.subheader("📊 Cohort Context")
        st.caption("Where does this patient sit relative to the full cohort?")

        ctx_col1, ctx_col2 = st.columns(2)
        with ctx_col1:
            if density_row is not None:
                fig3, ax3 = plt.subplots(figsize=(5, 3))
                ax3.hist(density_df["density_risk_index"], bins=30,
                         color="#90CAF9", edgecolor="white", alpha=0.8,
                         label="All patients")
                ax3.axvline(density_row.get("density_risk_index", 0),
                            color="#F44336", linewidth=2,
                            label=f"Patient {selected_pid}")
                ax3.set_xlabel("Density Risk Index (%)")
                ax3.set_ylabel("Count")
                ax3.set_title("Density Risk Index — Cohort Distribution")
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close()

        with ctx_col2:
            if lesion_row is not None and lesion_df is not None:
                fig4, ax4 = plt.subplots(figsize=(5, 3))
                ax4.hist(lesion_df["lesion_count"], bins=20,
                         color="#A5D6A7", edgecolor="white", alpha=0.8,
                         label="All patients")
                ax4.axvline(int(lesion_row.get("lesion_count", 0)),
                            color="#F44336", linewidth=2,
                            label=f"Patient {selected_pid}")
                ax4.set_xlabel("Lesion Count")
                ax4.set_ylabel("Count")
                ax4.set_title("Lesion Count — Cohort Distribution")
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig4, use_container_width=True)
                plt.close()


if __name__ == "__main__":
    main()
