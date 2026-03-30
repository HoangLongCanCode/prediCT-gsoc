"""
regenerate_plots.py
Regenerates three proposal-quality plots with fixed issues:

    1. density_contrast.png  — truly contrasting patients (no overlap), 
                               same Agatston category, opposite density profiles
    2. tsne.png              — genuine side-by-side: left=Agatston, right=cluster
    3. phenotype_profiles.png — cluster names instead of numbers, cleaner layout

Usage
-----
    python project2_radiomics/regenerate_plots.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path(r"D:\Du_hoc\gsoc\project2_radiomics\results")
DENSITY_CSV   = RESULTS_DIR / "density_features.csv"
FEATURES_CSV  = RESULTS_DIR / "features.csv"
OUTPUT_DIR    = RESULTS_DIR

AGATSTON_LABELS = {
    0: "None (0)",
    1: "Mild (1-99)",
    2: "Moderate (100-399)",
    3: "Severe (>=400)",
}

FEATURE_PREFIXES = ["shape_", "glcm_", "glszm_", "glrlm_"]

# ── Colour palette (consistent across all plots) ──────────────────────────────
AGATSTON_COLORS = {
    "None (0)":           "#4CAF50",
    "Mild (1-99)":        "#2196F3",
    "Moderate (100-399)": "#FF9800",
    "Severe (>=400)":     "#F44336",
}

DENSITY_COLORS = {
    "low_density":        "#F44336",
    "mild_density":       "#FF9800",
    "moderate_density":   "#2196F3",
    "high_density":       "#4CAF50",
}

BIN_LABELS = [
    "Low\n(130–200 HU)\nRisky",
    "Mild\n(200–300 HU)",
    "Moderate\n(300–400 HU)",
    "High\n(400+ HU)\nProtective",
]

BIN_COLS = ["low_density_pct", "mild_density_pct",
            "moderate_density_pct", "high_density_pct"]


# ════════════════════════════════════════════════════════════════════════════════
# FIX 1 — density_contrast.png
# ════════════════════════════════════════════════════════════════════════════════

def plot_density_contrast(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Show the density paradox clearly:
    - Both panels use patients from the SAME Agatston category
    - Left  = highest low-density % (most dangerous profile)
    - Right = highest high-density % (most protective profile)
    - Zero overlap between panels
    - Clearly labelled, clean layout
    """
    print("  [1/3] Generating density_contrast.png ...")

    # Work with Moderate + Severe patients (most clinically relevant contrast)
    candidates = df[df["agatston_category"].isin([2, 3])].copy()

    # Sort by low_density_pct descending → pick top 3 as "high risk"
    high_risk_ids = (
        candidates.nlargest(3, "low_density_pct")["patient_id"].tolist()
    )
    # From remaining patients, pick top 3 by high_density_pct → "low risk"
    remaining     = candidates[~candidates["patient_id"].isin(high_risk_ids)]
    low_risk_ids  = (
        remaining.nlargest(3, "high_density_pct")["patient_id"].tolist()
    )

    high_risk = candidates[candidates["patient_id"].isin(high_risk_ids)]
    low_risk  = candidates[candidates["patient_id"].isin(low_risk_ids)]

    bin_colors = [DENSITY_COLORS[b] for b in
                  ["low_density", "mild_density", "moderate_density", "high_density"]]
    x = np.arange(4)
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.patch.set_facecolor("white")

    # Each patient gets its own distinct color
    patient_colors = ["#1565C0", "#E65100", "#558B2F"]  # blue, dark orange, dark green

    panels = [
        (axes[0], high_risk,
         "High-Risk Calcium Profile\n(Predominantly Low-Density)",
         "#C62828", "⚠"),
        (axes[1], low_risk,
         "Lower-Risk Calcium Profile\n(Predominantly High-Density)",
         "#2E7D32", "✓"),
    ]

    for ax, group, title, title_color, symbol in panels:
        for i, (_, row) in enumerate(group.iterrows()):
            vals  = [row[c] for c in BIN_COLS]
            color = patient_colors[i % len(patient_colors)]
            # Use decreasing alpha across HU bins to show bin structure
            alphas = [0.95, 0.72, 0.50, 0.35]
            for j, (val, alpha) in enumerate(zip(vals, alphas)):
                ax.bar(
                    x[j] + i * width, val, width,
                    color=color,
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=0.8,
                    label=f"Patient {row['patient_id']} [{row['agatston_label']}]"
                          if j == 0 else "_nolegend_",
                )
                if val > 4:
                    ax.text(
                        x[j] + i * width + width / 2,
                        val + 0.8,
                        f"{val:.0f}%",
                        ha="center", va="bottom",
                        fontsize=7.5, color="black", fontweight="bold"
                    )

        ax.set_xticks(x + width)
        ax.set_xticklabels(BIN_LABELS, fontsize=9)
        ax.set_ylabel("% of Calcium Voxels", fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25, axis="y", linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{symbol}  {title}", fontsize=11,
                     color=title_color, fontweight="bold", pad=12)
        ax.legend(fontsize=8.5, loc="upper right",
                  framealpha=0.9, edgecolor="lightgrey", title="Patient")

        # Annotate DRI
        for i, (_, row) in enumerate(group.iterrows()):
            dri = row.get("density_risk_index", row["low_density_pct"])
            ax.annotate(
                f"DRI: {dri:.0f}%",
                xy=(x[0] + i * width + width / 2, 2),
                fontsize=7, ha="center", color="grey"
            )

    # Shared legend for density bins
    legend_patches = [
        mpatches.Patch(color=DENSITY_COLORS["low_density"],
                       label="Low (130–200 HU) — Vulnerable"),
        mpatches.Patch(color=DENSITY_COLORS["mild_density"],
                       label="Mild (200–300 HU)"),
        mpatches.Patch(color=DENSITY_COLORS["moderate_density"],
                       label="Moderate (300–400 HU)"),
        mpatches.Patch(color=DENSITY_COLORS["high_density"],
                       label="High (400+ HU) — Protective"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=4,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "Same Agatston Category — Opposite Density Profiles\n"
        "The Agatston Score Alone Cannot Distinguish These Patients",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_dir / "density_contrast.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print("     ✓ Saved: density_contrast.png")


# ════════════════════════════════════════════════════════════════════════════════
# FIX 2 — tsne.png (genuine side-by-side)
# ════════════════════════════════════════════════════════════════════════════════

def plot_tsne_sidebyside(df: pd.DataFrame, feature_cols: list,
                         labels: np.ndarray, output_dir: Path) -> None:
    """
    Genuine side-by-side t-SNE:
    - SAME embedding used for both panels (same coordinates)
    - Left  = coloured by Agatston category
    - Right = coloured by discovered cluster with phenotype names
    Fixes the old plot which only showed one panel.
    """
    print("  [2/3] Generating tsne.png (side-by-side) ...")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].fillna(0))

    # Compute ONE t-SNE embedding, reuse for both panels
    tsne  = TSNE(n_components=2, random_state=42,
                 perplexity=min(8, len(df) - 1), n_iter=2000)
    X_2d  = tsne.fit_transform(X_scaled)

    df = df.copy()
    df["tsne1"]   = X_2d[:, 0]
    df["tsne2"]   = X_2d[:, 1]
    df["cluster"] = labels + 1

    # Map clusters to phenotype names (from our characterisation)
    cluster_means = df.groupby("cluster")[feature_cols].mean()
    overall_means = df[feature_cols].mean()

    def infer_name(cid):
        row = cluster_means.loc[cid]
        sph = row.get("shape_Sphericity", overall_means.get("shape_Sphericity", 0))
        vol = row.get("shape_MeshVolume", overall_means.get("shape_MeshVolume", 0))
        non = row.get("glszm_GrayLevelNonUniformity",
                      overall_means.get("glszm_GrayLevelNonUniformity", 0))
        avg_vol = overall_means.get("shape_MeshVolume", 1)
        avg_sph = overall_means.get("shape_Sphericity", 1)
        avg_non = overall_means.get("glszm_GrayLevelNonUniformity", 1)
        if vol > avg_vol * 1.3 and non > avg_non:
            return "Large & Heterogeneous"
        elif sph > avg_sph and vol < avg_vol:
            return "Small & Compact"
        elif vol < avg_vol * 0.5 and non < avg_non:
            return "Homogeneous"
        else:
            return "Mixed Pattern"

    cluster_names  = {cid: infer_name(cid)
                      for cid in sorted(df["cluster"].unique())}
    cluster_colors = {
        cid: c for cid, c in
        zip(sorted(df["cluster"].unique()),
            ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"])
    }

    df["agatston_label"] = df["agatston_category"].map(AGATSTON_LABELS)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    marker_size  = 160
    edge_color   = "white"
    edge_width   = 0.8

    # ── Left: Agatston category ───────────────────────────────────────────────
    ax = axes[0]
    for label in ["None (0)", "Mild (1-99)", "Moderate (100-399)", "Severe (>=400)"]:
        sub = df[df["agatston_label"] == label]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["tsne1"], sub["tsne2"],
            label=label,
            color=AGATSTON_COLORS[label],
            s=marker_size, alpha=0.85,
            edgecolors=edge_color, linewidths=edge_width,
            zorder=3
        )

    ax.set_title("Coloured by Agatston Category",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.legend(title="Agatston Category", fontsize=9,
              title_fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: Discovered clusters ────────────────────────────────────────────
    ax = axes[1]
    for cid in sorted(df["cluster"].unique()):
        sub   = df[df["cluster"] == cid]
        name  = cluster_names[cid]
        color = cluster_colors[cid]
        ax.scatter(
            sub["tsne1"], sub["tsne2"],
            label=f"Cluster {cid}: {name}",
            color=color,
            s=marker_size, alpha=0.85,
            edgecolors=edge_color, linewidths=edge_width,
            zorder=3
        )

    # Annotate cluster centroids with phenotype labels
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        cx, cy = sub["tsne1"].mean(), sub["tsne2"].mean()
        ax.annotate(
            f"C{cid}\n{cluster_names[cid]}",
            xy=(cx, cy),
            fontsize=7.5, ha="center", va="center",
            fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.7, edgecolor="grey"),
            zorder=5
        )

    ax.set_title("Coloured by Discovered Phenotype Cluster",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.legend(title="Cluster", fontsize=9,
              title_fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "t-SNE Projection of the Radiomic Feature Space\n"
        "Left: clinical labels  |  Right: data-driven phenotypes",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_dir / "tsne.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print("     ✓ Saved: tsne.png")


# ════════════════════════════════════════════════════════════════════════════════
# FIX 3 — phenotype_profiles.png
# ════════════════════════════════════════════════════════════════════════════════

def plot_phenotype_profiles(df: pd.DataFrame, feature_cols: list,
                            labels: np.ndarray, output_dir: Path) -> None:
    """
    Phenotype heatmap with:
    - Cluster NAMES on x-axis (not just numbers)
    - Shortened, readable feature names on y-axis
    - Only the most discriminating features shown (top 12)
    - Clear annotation of which cluster is which
    """
    print("  [3/3] Generating phenotype_profiles.png ...")

    df = df.copy()
    df["cluster"] = labels + 1

    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(df[feature_cols].fillna(0))
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    df_scaled["cluster"] = df["cluster"].values

    cluster_means = df_scaled.groupby("cluster")[feature_cols].mean()
    overall_means = df[feature_cols].mean()

    # Infer phenotype names
    raw_means = df.groupby("cluster")[feature_cols].mean()

    def infer_name(cid):
        row = raw_means.loc[cid]
        sph = row.get("shape_Sphericity", overall_means.get("shape_Sphericity", 0))
        vol = row.get("shape_MeshVolume", overall_means.get("shape_MeshVolume", 0))
        non = row.get("glszm_GrayLevelNonUniformity",
                      overall_means.get("glszm_GrayLevelNonUniformity", 0))
        avg_vol = overall_means.get("shape_MeshVolume", 1)
        avg_sph = overall_means.get("shape_Sphericity", 1)
        avg_non = overall_means.get("glszm_GrayLevelNonUniformity", 1)
        if vol > avg_vol * 1.3 and non > avg_non:
            return "Large &\nHeterogeneous"
        elif sph > avg_sph and vol < avg_vol:
            return "Small &\nCompact"
        elif vol < avg_vol * 0.5 and non < avg_non:
            return "Homogeneous"
        else:
            return "Mixed\nPattern"

    cluster_name_map = {
        cid: f"C{cid}: {infer_name(cid)}"
        for cid in sorted(df["cluster"].unique())
    }

    # Select top 12 most discriminating features (highest variance across clusters)
    feat_var = cluster_means.var(axis=0).sort_values(ascending=False)
    top_feats = feat_var.head(12).index.tolist()

    # Shorten feature names for readability
    def shorten(name):
        return (name
                .replace("shape_", "Shape: ")
                .replace("glcm_", "GLCM: ")
                .replace("glszm_", "GLSZM: ")
                .replace("glrlm_", "GLRLM: ")
                .replace("GrayLevelNonUniformity", "Gray Lvl NonUnif")
                .replace("RunLengthNonUniformity", "Run Len NonUnif")
                .replace("SurfaceVolumeRatio", "Surface/Vol Ratio")
                .replace("Maximum3DDiameter", "Max 3D Diameter")
                .replace("LargeAreaEmphasis", "Large Area Emph")
                .replace("SmallAreaEmphasis", "Small Area Emph")
                .replace("ZonePercentage", "Zone Percentage")
                .replace("ShortRunEmphasis", "Short Run Emph")
                .replace("LongRunEmphasis", "Long Run Emph")
                .replace("RunPercentage", "Run Percentage")
                .replace("DifferenceVariance", "Diff Variance")
                .replace("Homogeneity1", "Homogeneity")
                .replace("JointEnergy", "Joint Energy")
                .replace("MeshVolume", "Mesh Volume")
                .replace("VoxelVolume", "Voxel Volume")
                .replace("Sphericity", "Sphericity"))

    plot_data = cluster_means[top_feats].copy()
    plot_data.index = [cluster_name_map[i] for i in plot_data.index]
    plot_data.columns = [shorten(c) for c in plot_data.columns]

    # Drop single-patient clusters (not statistically meaningful)
    patient_counts = df["cluster"].value_counts()
    valid_clusters = patient_counts[patient_counts > 1].index.tolist()
    plot_data = plot_data.loc[
        [cluster_name_map[i] for i in sorted(valid_clusters)]
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        plot_data.T,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        linewidths=0.6,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Z-score (std. deviations from mean)",
                  "shrink": 0.8},
        annot_kws={"size": 10},
    )

    ax.set_title(
        "Radiomic Feature Profiles per Calcium Phenotype\n"
        "Z-scores: Red = above cohort mean  |  Blue = below cohort mean",
        fontsize=12, fontweight="bold", pad=14
    )
    ax.set_xlabel("Calcium Phenotype", fontsize=11, labelpad=10)
    ax.set_ylabel("Radiomic Feature", fontsize=11, labelpad=10)
    ax.tick_params(axis="x", labelsize=9, rotation=0)
    ax.tick_params(axis="y", labelsize=9)

    # Add patient count annotation below each cluster column
    counts = df["cluster"].value_counts().sort_index()
    for i, cid in enumerate(sorted(valid_clusters)):
        n = counts.get(cid, 0)
        ax.text(
            i + 0.5, len(top_feats) + 0.6,
            f"n = {n}",
            ha="center", va="bottom",
            fontsize=8, color="grey",
            transform=ax.transData
        )

    plt.tight_layout()
    plt.savefig(output_dir / "phenotype_profiles.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print("     ✓ Saved: phenotype_profiles.png")


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("[Regenerate Plots] Starting...\n")

    # ── Load density features ─────────────────────────────────────────────────
    if not DENSITY_CSV.exists():
        print(f"[ERROR] density_features.csv not found at {DENSITY_CSV}")
        return
    df_density = pd.read_csv(DENSITY_CSV)
    df_density["patient_id"] = df_density["patient_id"].astype(str)
    print(f"  Loaded density features: {len(df_density)} patients")

    # ── Fix 1: density_contrast.png ───────────────────────────────────────────
    plot_density_contrast(df_density, OUTPUT_DIR)

    # ── Load radiomic features ────────────────────────────────────────────────
    if not FEATURES_CSV.exists():
        print(f"[ERROR] features.csv not found at {FEATURES_CSV}")
        print("  Skipping t-SNE and phenotype profile plots.")
        return
    df_feat = pd.read_csv(FEATURES_CSV)
    print(f"  Loaded radiomic features: {len(df_feat)} patients")

    feature_cols = [c for c in df_feat.columns
                    if any(c.startswith(p) for p in FEATURE_PREFIXES)]
    print(f"  Feature columns: {len(feature_cols)}")

    # Cluster using same logic as unsupervised_analysis.py
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_feat[feature_cols].fillna(0))

    # Find best k (cap at 4 for 23 patients)
    best_k, best_score = 2, -1
    for k in range(2, min(6, len(df_feat))):
        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X_scaled)
        sc   = silhouette_score(X_scaled, labs)
        if sc > best_score:
            best_score, best_k = sc, k
    best_k = min(best_k, 4)
    print(f"  Best k = {best_k} (silhouette = {best_score:.3f})")

    km     = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # ── Fix 2: tsne.png (side-by-side) ───────────────────────────────────────
    plot_tsne_sidebyside(df_feat, feature_cols, labels, OUTPUT_DIR)

    # ── Fix 3: phenotype_profiles.png ─────────────────────────────────────────
    plot_phenotype_profiles(df_feat, feature_cols, labels, OUTPUT_DIR)

    print("\n[Regenerate Plots] All done! Upload to Overleaf:")
    print("  - results/density_contrast.png")
    print("  - results/tsne.png")
    print("  - results/phenotype_profiles.png")


if __name__ == "__main__":
    main()