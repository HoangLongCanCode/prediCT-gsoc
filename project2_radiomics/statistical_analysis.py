"""
statistical_analysis.py
Performs statistical analysis on extracted radiomic features:
  - Spearman correlation with Agatston score
  - Kruskal-Wallis test across Agatston categories
  - Saves correlation matrix and significant features

Usage
-----
    python statistical_analysis.py
    python statistical_analysis.py --features_csv "D:/Du_hoc/gsoc/results/features.csv"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


# ── Feature columns ───────────────────────────────────────────────────────────

# These are the radiomic feature column prefixes we care about
FEATURE_PREFIXES = [
    "shape_", "glcm_", "glszm_", "glrlm_"
]

AGATSTON_LABELS = {
    0: "None (0)",
    1: "Mild (1-99)",
    2: "Moderate (100-399)",
    3: "Severe (≥400)",
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all radiomic feature column names."""
    return [
        c for c in df.columns
        if any(c.startswith(p) for p in FEATURE_PREFIXES)
    ]


# ── Statistical tests ─────────────────────────────────────────────────────────

def spearman_analysis(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Spearman correlation between each feature and Agatston score.
    Returns DataFrame sorted by absolute correlation.
    """
    results = []
    for feat in feature_cols:
        valid = df[[feat, "agatston_score"]].dropna()
        if len(valid) < 5:
            continue
        rho, pval = stats.spearmanr(valid[feat], valid["agatston_score"])
        results.append({
            "feature":     feat,
            "spearman_r":  round(rho,  4),
            "p_value":     round(pval, 4),
            "significant": pval < 0.05,
            "abs_r":       abs(rho),
        })

    df_out = pd.DataFrame(results).sort_values("abs_r", ascending=False)
    return df_out.drop(columns="abs_r")


def kruskal_wallis_analysis(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Kruskal-Wallis H-test: is each feature significantly different
    across Agatston categories?
    Returns DataFrame sorted by p-value.
    """
    results = []
    categories = sorted(df["agatston_category"].unique())

    for feat in feature_cols:
        groups = [
            df[df["agatston_category"] == c][feat].dropna().values
            for c in categories
        ]
        # Need at least 2 non-empty groups
        non_empty = [g for g in groups if len(g) > 0]
        if len(non_empty) < 2:
            continue

        h_stat, pval = stats.kruskal(*non_empty)
        results.append({
            "feature":     feat,
            "H_statistic": round(h_stat, 4),
            "p_value":     round(pval,   4),
            "significant": pval < 0.05,
        })

    return pd.DataFrame(results).sort_values("p_value")


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_correlation_matrix(
    df:           pd.DataFrame,
    feature_cols: list[str],
    output_dir:   Path,
) -> None:
    """Heatmap of Spearman correlations between all features + Agatston score."""
    cols  = feature_cols + ["agatston_score"]
    corr  = df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot    = False,
        cmap     = "RdBu_r",
        center   = 0,
        vmin     = -1, vmax = 1,
        square   = True,
        linewidths = 0.3,
        ax       = ax,
    )
    ax.set_title("Spearman Correlation Matrix — Radiomic Features", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: correlation_matrix.png")


def plot_significant_features(
    df:           pd.DataFrame,
    spearman_df:  pd.DataFrame,
    output_dir:   Path,
    top_n:        int = 6,
) -> None:
    """Box plots of top N significant features across Agatston categories."""
    sig = spearman_df[spearman_df["significant"]].head(top_n)
    if sig.empty:
        print("  No significant features to plot.")
        return

    n_feats = len(sig)
    ncols   = 3
    nrows   = (n_feats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten() if n_feats > 1 else [axes]

    df["agatston_label"] = df["agatston_category"].map(AGATSTON_LABELS)
    order = list(AGATSTON_LABELS.values())
    order = [o for o in order if o in df["agatston_label"].unique()]

    for i, (_, row) in enumerate(sig.iterrows()):
        feat = row["feature"]
        ax   = axes[i]
        sns.boxplot(
            data   = df,
            x      = "agatston_label",
            y      = feat,
            order  = order,
            palette= "Set2",
            ax     = ax,
        )
        ax.set_title(
            f"{feat}\nρ={row['spearman_r']:.3f}, p={row['p_value']:.3f}",
            fontsize=9
        )
        ax.set_xlabel("Agatston Category")
        ax.set_ylabel("Feature Value")
        ax.tick_params(axis="x", rotation=15)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Top Significant Radiomic Features by Agatston Category",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "significant_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: significant_features.png")


def plot_agatston_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of Agatston category distribution in the extracted sample."""
    df["agatston_label"] = df["agatston_category"].map(AGATSTON_LABELS)
    counts = df["agatston_label"].value_counts().reindex(AGATSTON_LABELS.values()).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=["#4CAF50","#2196F3","#FF9800","#F44336"])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_title("Agatston Score Distribution in Extracted Sample", fontsize=13)
    ax.set_xlabel("Agatston Category")
    ax.set_ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(output_dir / "agatston_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: agatston_distribution.png")


def plot_tsne(
    df:           pd.DataFrame,
    feature_cols: list[str],
    output_dir:   Path,
) -> None:
    """t-SNE visualization of radiomic feature space coloured by Agatston category."""
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        X = df[feature_cols].dropna(axis=1).values
        if X.shape[0] < 5:
            print("  [SKIP] t-SNE needs at least 5 samples.")
            return

        X_scaled = StandardScaler().fit_transform(X)
        perplexity = min(30, X.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = tsne.fit_transform(X_scaled)

        df["agatston_label"] = df["agatston_category"].map(AGATSTON_LABELS)
        colors = {"None (0)": "#4CAF50", "Mild (1-99)": "#2196F3",
                  "Moderate (100-399)": "#FF9800", "Severe (≥400)": "#F44336"}

        fig, ax = plt.subplots(figsize=(9, 7))
        for label, group in df.groupby("agatston_label"):
            idx = df[df["agatston_label"] == label].index
            ax.scatter(
                X_2d[df.index.get_indexer(idx), 0],
                X_2d[df.index.get_indexer(idx), 1],
                label   = label,
                color   = colors.get(label, "grey"),
                alpha   = 0.8,
                s       = 80,
                edgecolors = "white",
                linewidths = 0.5,
            )
        ax.set_title("t-SNE of Radiomic Features (coloured by Agatston Category)",
                     fontsize=13)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(title="Agatston Category")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: tsne.png")

    except Exception as e:
        print(f"  [SKIP] t-SNE failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis(features_csv: str, output_dir: str) -> None:
    features_csv = Path(features_csv)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    print(f"[Analysis] Loaded {len(df)} patients, {df.shape[1]} columns")

    feature_cols = get_feature_cols(df)
    print(f"[Analysis] Found {len(feature_cols)} radiomic features")

    # ── Statistical tests ─────────────────────────────────────────────────────
    print("\n── Spearman Correlation with Agatston Score ──────────")
    spearman_df = spearman_analysis(df, feature_cols)
    spearman_df.to_csv(output_dir / "spearman_results.csv", index=False)
    print(spearman_df.head(10).to_string(index=False))

    print(f"\n  Significant features (p<0.05): {spearman_df['significant'].sum()}")

    print("\n── Kruskal-Wallis Test across Agatston Categories ────")
    kruskal_df = kruskal_wallis_analysis(df, feature_cols)
    kruskal_df.to_csv(output_dir / "kruskal_results.csv", index=False)
    print(kruskal_df.head(10).to_string(index=False))

    print(f"\n  Significant features (p<0.05): {kruskal_df['significant'].sum()}")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n── Generating plots ──────────────────────────────────")
    plot_agatston_distribution(df, output_dir)
    plot_correlation_matrix(df, feature_cols, output_dir)
    plot_significant_features(df, spearman_df, output_dir)
    plot_tsne(df, feature_cols, output_dir)

    print(f"\n[Analysis] All outputs saved to: {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv",
        default=r"D:\Du_hoc\gsoc\results\features.csv")
    parser.add_argument("--output_dir",
        default=r"D:\Du_hoc\gsoc\results")
    args = parser.parse_args()

    run_analysis(
        features_csv = args.features_csv,
        output_dir   = args.output_dir,
    )
