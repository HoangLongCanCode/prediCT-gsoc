"""
density_fingerprint.py
Calcium density fingerprinting for coronary calcium CT scans.

Key insight from literature (Criqui et al., JAMA 2014; Circulation 2023):
    - Low-density calcium (130-200 HU) = spotty, potentially unstable plaques
    - High-density calcium (>400 HU)   = dense, paradoxically PROTECTIVE
    - Two patients with identical Agatston scores can have opposite risk profiles
      based on their density distribution

This script:
    1. Reads original HU values from calcium voxels (mask=1)
    2. Computes density fingerprint (HU histogram across 4 clinical bins)
    3. Derives a Density Risk Index (DRI) — proportion of low-density calcium
    4. Compares patients with same Agatston category but different density profiles
    5. Saves density_features.csv and fingerprint plots

Usage
-----
    python density_fingerprint.py
    python density_fingerprint.py --images_root "D:/Du_hoc/gsoc/processed/images"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

# ── Clinical HU density bins ──────────────────────────────────────────────────
# Based on Agatston density weighting factors + Criqui et al. JAMA 2014
DENSITY_BINS = {
    "low_density":    (130, 200),   # spotty, vulnerable — associated with plaque rupture
    "mild_density":   (200, 300),   # intermediate
    "moderate_density":(300, 400),  # moderate calcification
    "high_density":   (400, 3000),  # dense, stable — paradoxically protective
}

DENSITY_COLORS = {
    "low_density":     "#F44336",   # red — dangerous
    "mild_density":    "#FF9800",   # orange
    "moderate_density":"#2196F3",   # blue
    "high_density":    "#4CAF50",   # green — protective
}

AGATSTON_LABELS = {
    0: "None (0)",
    1: "Mild (1-99)",
    2: "Moderate (100-399)",
    3: "Severe (>=400)",
}


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_density_fingerprint(img_path: Path, seg_path: Path) -> dict:
    """
    Extract calcium density fingerprint from original HU image + mask.

    Returns dict with:
        - voxel counts per density bin
        - percentage per density bin
        - mean/max/std HU
        - Density Risk Index (DRI) = % low-density calcium
        - Density Stability Index (DSI) = % high-density calcium
    """
    img = sitk.ReadImage(str(img_path))
    seg = sitk.ReadImage(str(seg_path))

    img_arr = sitk.GetArrayFromImage(img).astype(np.float32)
    seg_arr = sitk.GetArrayFromImage(seg).astype(np.uint8)

    # Only calcium voxels above threshold
    calcium_mask = (seg_arr == 1) & (img_arr >= 130)
    total_voxels = int(np.sum(calcium_mask))

    if total_voxels == 0:
        return _empty_fingerprint()

    calcium_hu = img_arr[calcium_mask]

    result = {
        "total_calcium_voxels": total_voxels,
        "mean_hu":   round(float(calcium_hu.mean()), 2),
        "max_hu":    round(float(calcium_hu.max()),  2),
        "std_hu":    round(float(calcium_hu.std()),  2),
        "median_hu": round(float(np.median(calcium_hu)), 2),
    }

    # Per-bin counts and percentages
    for bin_name, (hu_min, hu_max) in DENSITY_BINS.items():
        count = int(np.sum((calcium_hu >= hu_min) & (calcium_hu < hu_max)))
        pct   = round(100 * count / total_voxels, 2) if total_voxels > 0 else 0.0
        result[f"{bin_name}_voxels"] = count
        result[f"{bin_name}_pct"]    = pct

    # Derived indices
    result["density_risk_index"]      = result["low_density_pct"]       # high = risky
    result["density_stability_index"] = result["high_density_pct"]      # high = stable
    result["density_ratio"] = round(
        result["low_density_pct"] / max(result["high_density_pct"], 0.1), 2
    )   # > 1 means more low-density than high — concerning

    return result


def _empty_fingerprint() -> dict:
    result = {
        "total_calcium_voxels": 0,
        "mean_hu": 0, "max_hu": 0, "std_hu": 0, "median_hu": 0,
        "density_risk_index": 0,
        "density_stability_index": 0,
        "density_ratio": 0,
    }
    for bin_name in DENSITY_BINS:
        result[f"{bin_name}_voxels"] = 0
        result[f"{bin_name}_pct"]    = 0.0
    return result


# ── Agatston category helper ──────────────────────────────────────────────────

def voxels_to_agatston_category(voxels: int) -> int:
    if voxels == 0:   return 0
    if voxels <= 500: return 1
    if voxels <= 2000: return 2
    return 3


# ── Batch processing ──────────────────────────────────────────────────────────

def process_all_patients(
    scan_index_csv: Path,
    images_root:    Path,
    output_dir:     Path,
    max_patients:   int = None,
) -> pd.DataFrame:
    """
    Run density fingerprinting on all patients with calcium.
    """
    df = pd.read_csv(scan_index_csv)
    df = df[df["voxels"] > 0].copy()

    if max_patients:
        df = df.head(max_patients)

    rows = []
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Density fingerprinting"):
        scan_id    = row["scan_id"]
        patient_id = str(row["patient_id"])

        img_path = images_root / scan_id / f"{scan_id}_img.nii.gz"
        seg_path = images_root / scan_id / f"{scan_id}_seg.nii.gz"

        if not img_path.exists() or not seg_path.exists():
            failed.append(patient_id)
            continue

        try:
            fingerprint = extract_density_fingerprint(img_path, seg_path)
            cat = voxels_to_agatston_category(int(row["voxels"]))
            rows.append({
                "patient_id":       patient_id,
                "scan_id":          scan_id,
                "voxels":           int(row["voxels"]),
                "agatston_category": cat,
                "agatston_label":   AGATSTON_LABELS[cat],
                **fingerprint,
            })
        except Exception as e:
            tqdm.write(f"  [ERROR] Patient {patient_id}: {e}")
            failed.append(patient_id)

    df_out = pd.DataFrame(rows)
    out_path = output_dir / "density_features.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\n[Density] Processed: {len(rows)}, Failed: {len(failed)}")
    print(f"[Density] Saved → {out_path}")
    return df_out


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_density_fingerprints(df: pd.DataFrame, output_dir: Path) -> None:
    """
    1. Average density fingerprint per Agatston category
    2. Density Risk Index distribution per category
    3. Scatter: low-density % vs high-density % colored by category
    4. Case study: same Agatston score, different density profile
    """

    # ── 1. Average fingerprint per category ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    categories = sorted(df["agatston_category"].unique())
    bin_names  = list(DENSITY_BINS.keys())
    bin_labels = ["Low\n(130-200)", "Mild\n(200-300)",
                  "Moderate\n(300-400)", "High\n(400+)"]
    colors     = [DENSITY_COLORS[b] for b in bin_names]

    ax = axes[0, 0]
    width = 0.2
    x = np.arange(len(bin_names))
    for i, cat in enumerate(categories):
        cat_df  = df[df["agatston_category"] == cat]
        means   = [cat_df[f"{b}_pct"].mean() for b in bin_names]
        label   = AGATSTON_LABELS[cat]
        ax.bar(x + i * width, means, width, label=label, alpha=0.8)
    ax.set_xticks(x + width * len(categories) / 2)
    ax.set_xticklabels(bin_labels)
    ax.set_title("Average Density Profile per Agatston Category", fontsize=11)
    ax.set_ylabel("% of Calcium Voxels")
    ax.legend(title="Agatston", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 2. Density Risk Index (DRI) per category ──────────────────────────────
    ax = axes[0, 1]
    df["agatston_label"] = df["agatston_category"].map(AGATSTON_LABELS)
    order = list(AGATSTON_LABELS.values())
    order = [o for o in order if o in df["agatston_label"].unique()]
    sns.boxplot(
        data=df, x="agatston_label", y="density_risk_index",
        order=order, palette=["#4CAF50", "#2196F3", "#FF9800", "#F44336"],
        ax=ax
    )
    ax.set_title("Density Risk Index by Agatston Category\n"
                 "(% Low-Density Calcium — higher = more vulnerable)", fontsize=10)
    ax.set_xlabel("Agatston Category")
    ax.set_ylabel("Low-Density Calcium (%)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 3. Low vs High density scatter ────────────────────────────────────────
    ax = axes[1, 0]
    cat_colors = {0: "#4CAF50", 1: "#2196F3", 2: "#FF9800", 3: "#F44336"}
    for cat in categories:
        sub = df[df["agatston_category"] == cat]
        ax.scatter(
            sub["low_density_pct"], sub["high_density_pct"],
            label=AGATSTON_LABELS[cat],
            color=cat_colors[cat],
            alpha=0.7, s=60, edgecolors="white", linewidths=0.5
        )
    ax.axline((0, 0), slope=1, color="grey", linestyle="--",
              alpha=0.5, label="Equal balance line")
    ax.set_xlabel("Low-Density Calcium % (Risky)")
    ax.set_ylabel("High-Density Calcium % (Protective)")
    ax.set_title("Low vs High Density Calcium\n"
                 "Points above diagonal = more protective calcium", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 4. Density ratio distribution ─────────────────────────────────────────
    ax = axes[1, 1]
    for cat in categories:
        sub = df[df["agatston_category"] == cat]
        sub["density_ratio"].clip(0, 10).plot.kde(
            ax=ax, label=AGATSTON_LABELS[cat], color=cat_colors[cat]
        )
    ax.axvline(x=1, color="black", linestyle="--", alpha=0.6,
               label="Equal low/high density")
    ax.set_xlabel("Density Ratio (Low/High) — >1 means more risky calcium")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Density Ratio\n"
                 "(Ratio >1 = predominantly low-density/vulnerable calcium)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Calcium Density Fingerprinting\n"
        "Same Agatston Score ≠ Same Risk Profile",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_dir / "density_fingerprints.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: density_fingerprints.png")


def plot_same_score_different_density(df: pd.DataFrame, output_dir: Path) -> None:
    """
    The key clinical insight: find pairs of patients with the same Agatston
    category but very different density profiles and visualize them side by side.
    """
    # Find patients in Moderate/Severe with highest and lowest DRI
    target_cats = df[df["agatston_category"].isin([2, 3])].copy()
    if len(target_cats) < 2:
        return

    high_risk = target_cats.nlargest(3, "density_risk_index")
    low_risk   = target_cats.nsmallest(3, "density_stability_index")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bin_names  = list(DENSITY_BINS.keys())
    bin_labels = ["Low\n(130-200 HU)", "Mild\n(200-300 HU)",
                  "Moderate\n(300-400 HU)", "High\n(400+ HU)"]
    colors     = [DENSITY_COLORS[b] for b in bin_names]

    for ax, group, title_prefix, color in [
        (axes[0], high_risk, "⚠️ High Risk Profile\n(Predominantly Low-Density)", "#F44336"),
        (axes[1], low_risk,  "✅ Lower Risk Profile\n(Predominantly High-Density)", "#4CAF50"),
    ]:
        x = np.arange(len(bin_names))
        width = 0.25
        for i, (_, row) in enumerate(group.iterrows()):
            vals = [row[f"{b}_pct"] for b in bin_names]
            ax.bar(x + i * width, vals, width,
                   label=f"Patient {row['patient_id']} (Cat:{row['agatston_label']})",
                   alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(bin_labels, fontsize=9)
        ax.set_ylabel("% of Calcium Voxels")
        ax.set_title(title_prefix, fontsize=11, color=color)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Same Agatston Category — Very Different Density Profiles\n"
        "The Agatston Score Alone Misses This Distinction",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(output_dir / "density_contrast.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: density_contrast.png")


def print_summary(df: pd.DataFrame) -> None:
    print("\n── Density Fingerprint Summary ───────────────────────────")
    for cat in sorted(df["agatston_category"].unique()):
        sub = df[df["agatston_category"] == cat]
        print(f"\n  {AGATSTON_LABELS[cat]} ({len(sub)} patients):")
        print(f"    Low-density  (risky)    : {sub['low_density_pct'].mean():.1f}% ± {sub['low_density_pct'].std():.1f}%")
        print(f"    High-density (protective): {sub['high_density_pct'].mean():.1f}% ± {sub['high_density_pct'].std():.1f}%")
        print(f"    Mean HU                  : {sub['mean_hu'].mean():.1f}")
        print(f"    Density Risk Index (avg) : {sub['density_risk_index'].mean():.1f}%")
    print("\n──────────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(scan_index_csv, images_root, output_dir, max_patients=None):
    scan_index_csv = Path(scan_index_csv)
    images_root    = Path(images_root)
    output_dir     = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Density Fingerprinting] Starting...")

    df = process_all_patients(
        scan_index_csv, images_root, output_dir, max_patients
    )

    if df.empty:
        print("[ERROR] No patients processed.")
        return

    print_summary(df)

    print("\n── Generating plots ──────────────────────────────────────")
    plot_density_fingerprints(df, output_dir)
    plot_same_score_different_density(df, output_dir)

    print(f"\n[Density Fingerprinting] Done! Results in: {output_dir}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_index",
        default=r"D:\Du_hoc\gsoc\processed\tables\scan_index.csv")
    parser.add_argument("--images_root",
        default=r"D:\Du_hoc\gsoc\processed\images")
    parser.add_argument("--output_dir",
        default=r"D:\Du_hoc\gsoc\project2_radiomics\results")
    parser.add_argument("--max_patients", type=int, default=None,
        help="Limit patients for testing (default: all)")
    args = parser.parse_args()

    run(
        scan_index_csv = args.scan_index,
        images_root    = args.images_root,
        output_dir     = args.output_dir,
        max_patients   = args.max_patients,
    )
