"""
unsupervised_analysis.py
Unsupervised clustering and phenotype characterization of radiomic features.

Goes beyond Agatston score correlation to discover natural calcium phenotypes
in the feature space — patients with the same score may have very different
calcium patterns (dense vs diffuse, compact vs irregular).

Steps:
    1. Standardize radiomic features
    2. K-Means clustering (k=3, optimal found via elbow + silhouette)
    3. UMAP visualization coloured by cluster AND Agatston category
    4. Phenotype characterization — what makes each cluster unique?
    5. Cluster vs Agatston category cross-tabulation

Usage
-----
    python unsupervised_analysis.py
    python unsupervised_analysis.py --features_csv "D:/Du_hoc/gsoc/project2_radiomics/results/features.csv"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[WARN] umap-learn not installed. UMAP plot will be skipped.")


# ── Feature columns ───────────────────────────────────────────────────────────

FEATURE_PREFIXES = ["shape_", "glcm_", "glszm_", "glrlm_"]

AGATSTON_LABELS = {
    0: "None (0)",
    1: "Mild (1-99)",
    2: "Moderate (100-399)",
    3: "Severe (>=400)",
}

CLUSTER_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0", "#00BCD4"]


def get_feature_cols(df):
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]


# ── Optimal k selection ───────────────────────────────────────────────────────

def find_optimal_k(X_scaled, k_range=range(2, 7), output_dir=None):
    """
    Elbow method + silhouette score to find optimal number of clusters.
    """
    inertias    = []
    silhouettes = []

    for k in k_range:
        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labs))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, "ro-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score (higher = better)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Optimal Number of Clusters", fontsize=13)
    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / "cluster_selection.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: cluster_selection.png")

    # Best k = highest silhouette
    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n  Silhouette scores: { {k: round(s, 3) for k, s in zip(k_range, silhouettes)} }")
    print(f"  Best k by silhouette: {best_k}")
    return best_k


# ── Clustering ────────────────────────────────────────────────────────────────

def run_clustering(X_scaled, k):
    """Run K-Means with optimal k."""
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score  = silhouette_score(X_scaled, labels)
    print(f"\n[Clustering] K-Means k={k}, silhouette={score:.3f}")
    return labels, km


# ── UMAP visualization ────────────────────────────────────────────────────────

def plot_umap(X_scaled, df, labels, output_dir):
    """
    Two UMAP plots side by side:
    Left  — coloured by cluster
    Right — coloured by Agatston category
    """
    if not UMAP_AVAILABLE:
        print("  [SKIP] UMAP not available.")
        return

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.1)
    X_2d    = reducer.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left — clusters
    for c in sorted(np.unique(labels)):
        mask = labels == c
        axes[0].scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label      = f"Cluster {c+1}",
            color      = CLUSTER_COLORS[c],
            alpha      = 0.85,
            s          = 100,
            edgecolors = "white",
            linewidths = 0.5,
        )
    axes[0].set_title("UMAP — Coloured by Cluster", fontsize=12)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].legend(title="Cluster")
    axes[0].grid(True, alpha=0.2)

    # Right — Agatston category
    agatston_colors = {
        "None (0)":          "#4CAF50",
        "Mild (1-99)":       "#2196F3",
        "Moderate (100-399)":"#FF9800",
        "Severe (>=400)":    "#F44336",
    }
    df["agatston_label"] = df["agatston_category"].map(AGATSTON_LABELS)
    for label, group_df in df.groupby("agatston_label"):
        idx  = group_df.index
        mask = np.isin(np.arange(len(df)), df.index.get_indexer(idx))
        axes[1].scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label      = label,
            color      = agatston_colors.get(label, "grey"),
            alpha      = 0.85,
            s          = 100,
            edgecolors = "white",
            linewidths = 0.5,
        )
    axes[1].set_title("UMAP — Coloured by Agatston Category", fontsize=12)
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].legend(title="Agatston Category")
    axes[1].grid(True, alpha=0.2)

    plt.suptitle(
        "UMAP of Radiomic Feature Space\n"
        "Clusters vs Agatston Categories — Do they align?",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(output_dir / "umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: umap.png")


# ── Phenotype characterization ────────────────────────────────────────────────

def characterize_phenotypes(df, feature_cols, labels, output_dir):
    """
    For each cluster, show mean feature values as a radar/heatmap
    and assign a clinical phenotype name.
    """
    df = df.copy()
    df["cluster"] = labels + 1   # 1-indexed for readability

    # Mean feature values per cluster (standardized)
    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(df[feature_cols])
    df_scaled  = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    df_scaled["cluster"] = df["cluster"].values

    cluster_means = df_scaled.groupby("cluster")[feature_cols].mean()

    # ── Heatmap of cluster feature profiles ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        cluster_means.T,
        annot    = True,
        fmt      = ".2f",
        cmap     = "RdBu_r",
        center   = 0,
        linewidths = 0.5,
        ax       = ax,
        cbar_kws = {"label": "Z-score (standardized)"},
    )
    ax.set_title(
        "Radiomic Feature Profiles per Cluster\n"
        "(Z-scores: red=high, blue=low relative to mean)",
        fontsize=12
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "phenotype_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: phenotype_profiles.png")

    # ── Agatston distribution per cluster ─────────────────────────────────────
    df["agatston_label"] = df["agatston_category"].map(AGATSTON_LABELS)
    crosstab = pd.crosstab(df["cluster"], df["agatston_label"])

    fig, ax = plt.subplots(figsize=(9, 5))
    crosstab.plot(
        kind    = "bar",
        ax      = ax,
        color   = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"],
        edgecolor = "white",
        width   = 0.6,
    )
    ax.set_title(
        "Agatston Category Distribution per Cluster\n"
        "Do clusters align with clinical severity?",
        fontsize=12
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Patients")
    ax.set_xticklabels([f"Cluster {i}" for i in crosstab.index], rotation=0)
    ax.legend(title="Agatston Category", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_agatston_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: cluster_agatston_distribution.png")

    # ── Print phenotype summary ───────────────────────────────────────────────
    print("\n── Phenotype Characterization ────────────────────────────")
    raw_means = df.groupby("cluster")[feature_cols].mean()

    for cluster_id in sorted(df["cluster"].unique()):
        cdf     = df[df["cluster"] == cluster_id]
        cmeans  = raw_means.loc[cluster_id]

        # Infer phenotype from key features
        sphericity = cmeans.get("shape_Sphericity", 0)
        volume     = cmeans.get("shape_MeshVolume", 0)
        nonunif    = cmeans.get("glszm_GrayLevelNonUniformity", 0)

        all_vol    = df["shape_MeshVolume"].mean() if "shape_MeshVolume" in df else 1
        all_sph    = df["shape_Sphericity"].mean() if "shape_Sphericity" in df else 1
        all_non    = df["glszm_GrayLevelNonUniformity"].mean() \
                     if "glszm_GrayLevelNonUniformity" in df else 1

        # Phenotype naming logic
        if volume > all_vol * 1.3 and nonunif > all_non:
            phenotype = "Large & Heterogeneous — dense, complex deposits"
        elif sphericity > all_sph and volume < all_vol:
            phenotype = "Small & Compact — focal round nodules"
        elif nonunif < all_non and volume < all_vol:
            phenotype = "Small & Homogeneous — fine uniform deposits"
        else:
            phenotype = "Mixed pattern"

        agatston_dist = cdf["agatston_label"].value_counts().to_dict()

        print(f"\n  Cluster {cluster_id} ({len(cdf)} patients)")
        print(f"  Phenotype : {phenotype}")
        print(f"  Avg Agatston score : {cdf['agatston_score'].mean():.1f}")
        print(f"  Agatston categories: {agatston_dist}")
        print(f"  Key features:")
        print(f"    Sphericity         : {sphericity:.3f}")
        print(f"    MeshVolume         : {volume:.1f}")
        print(f"    GrayLevelNonUnif   : {nonunif:.2f}")

    print("\n──────────────────────────────────────────────────────────\n")

    # Save cluster assignments
    df[["patient_id", "agatston_score", "agatston_label", "cluster"]].to_csv(
        output_dir / "cluster_assignments.csv", index=False
    )
    print("  Saved: cluster_assignments.csv")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_unsupervised(features_csv, output_dir):
    features_csv = Path(features_csv)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df           = pd.read_csv(features_csv)
    feature_cols = get_feature_cols(df)

    print(f"[Unsupervised] {len(df)} patients, {len(feature_cols)} features")

    # Standardize
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].fillna(0))

    # Find optimal k
    print("\n── Finding Optimal Number of Clusters ────────────────")
    best_k = min(find_optimal_k(X_scaled, output_dir=output_dir), 4)

    # Cluster
    print(f"\n── Running K-Means (k={best_k}) ──────────────────────")
    labels, km = run_clustering(X_scaled, best_k)

    # UMAP
    print("\n── Generating UMAP ───────────────────────────────────")
    plot_umap(X_scaled, df.copy(), labels, output_dir)

    # Phenotype characterization
    print("\n── Characterizing Phenotypes ─────────────────────────")
    df = characterize_phenotypes(df, feature_cols, labels, output_dir)

    print(f"\n[Unsupervised] All outputs saved to: {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_csv",
        default=r"D:\Du_hoc\gsoc\project2_radiomics\results\features.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=r"D:\Du_hoc\gsoc\project2_radiomics\results",
    )
    args = parser.parse_args()

    run_unsupervised(
        features_csv = args.features_csv,
        output_dir   = args.output_dir,
    )