"""
per_lesion_features.py
Per-lesion radiomic feature extraction for coronary calcium CT scans.

Instead of one feature vector per patient, this extracts features for each
individual calcium lesion separately, then aggregates using 6 statistics
(mean, max, min, std, skewness, kurtosis) per feature.

This mirrors the methodology of the Framingham Heart Study radiomics paper
which achieved significant improvement in MACE prediction using per-lesion
features vs patient-level features.

Result: ~18 features × 6 statistics = 108 features per patient
Plus lesion-level metadata: count, size distribution, spatial spread

Usage
-----
    python per_lesion_features.py --n_patients 30
    python per_lesion_features.py  # runs on all patients with calcium
"""

import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings("ignore")

logging.getLogger("radiomics").setLevel(logging.ERROR)
from radiomics import featureextractor


# ── PyRadiomics settings ──────────────────────────────────────────────────────

RADIOMICS_SETTINGS = {
    "binWidth":       25,
    "minimumROISize": 3,    # lower threshold for small lesions
    "verbose":        False,
}

AGATSTON_LABELS = {
    0: "None (0)",
    1: "Mild (1-99)",
    2: "Moderate (100-399)",
    3: "Severe (>=400)",
}


# ── Connected component lesion detection ──────────────────────────────────────

def detect_lesions(seg_arr: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Label individual connected calcium lesions using SimpleITK.

    Returns
    -------
    labeled_arr : ndarray where each lesion has a unique integer label (1, 2, 3...)
    n_lesions   : total number of lesions found
    """
    seg_sitk  = sitk.GetImageFromArray(seg_arr.astype(np.uint8))
    connected = sitk.ConnectedComponent(seg_sitk)
    labeled   = sitk.GetArrayFromImage(connected)
    n_lesions = int(labeled.max())
    return labeled, n_lesions


# ── Per-lesion feature extraction ─────────────────────────────────────────────

def extract_lesion_features(
    img_path: Path,
    seg_path: Path,
    extractor,
) -> list[dict]:
    """
    Extract radiomic features for each individual calcium lesion.

    Returns list of dicts, one per lesion, with:
        - lesion_id, lesion_voxels, lesion_mean_hu
        - all radiomic features
    """
    img = sitk.ReadImage(str(img_path))
    seg = sitk.ReadImage(str(seg_path))

    img_arr = sitk.GetArrayFromImage(img).astype(np.float32)
    seg_arr = sitk.GetArrayFromImage(seg).astype(np.uint8)

    # Label individual lesions
    labeled_arr, n_lesions = detect_lesions(seg_arr)

    if n_lesions == 0:
        return []

    lesion_features = []

    for lesion_id in range(1, n_lesions + 1):
        # Create mask for this lesion only
        lesion_mask = (labeled_arr == lesion_id).astype(np.uint8)
        n_voxels    = int(np.sum(lesion_mask))

        if n_voxels < 3:   # skip tiny noise voxels
            continue

        # HU stats for this lesion
        lesion_hu    = img_arr[lesion_mask == 1]
        lesion_mean_hu = float(lesion_hu.mean()) if len(lesion_hu) > 0 else 0.0

        try:
            # Create SimpleITK objects for this lesion
            lesion_sitk = sitk.GetImageFromArray(lesion_mask)
            lesion_sitk.CopyInformation(seg)

            result = extractor.execute(img, lesion_sitk)

            features = {"lesion_id": lesion_id, "lesion_voxels": n_voxels,
                        "lesion_mean_hu": round(lesion_mean_hu, 2)}

            for k, v in result.items():
                if k.startswith("original_") and "diagnostics" not in k:
                    try:
                        features[k.replace("original_", "")] = float(v)
                    except (TypeError, ValueError):
                        pass

            lesion_features.append(features)

        except Exception:
            # If radiomics fails for this lesion, still record basic stats
            lesion_features.append({
                "lesion_id":      lesion_id,
                "lesion_voxels":  n_voxels,
                "lesion_mean_hu": round(lesion_mean_hu, 2),
            })

    return lesion_features


# ── Aggregation statistics ────────────────────────────────────────────────────

def aggregate_lesion_features(lesion_list: list[dict]) -> dict:
    """
    Aggregate per-lesion features into patient-level statistics.
    Uses 6 summary statistics per feature: mean, max, min, std, skewness, kurtosis

    Also computes lesion-level metadata:
        - lesion_count
        - total_lesion_voxels
        - lesion_size_cv (coefficient of variation of lesion sizes)
        - spatial_spread (std of lesion centroids — diffusivity proxy)
    """
    if not lesion_list:
        return {"lesion_count": 0}

    df = pd.DataFrame(lesion_list)

    # Feature columns (exclude metadata)
    meta_cols = {"lesion_id", "lesion_voxels", "lesion_mean_hu"}
    feat_cols = [c for c in df.columns if c not in meta_cols]

    result = {}

    # Lesion metadata
    result["lesion_count"]       = len(df)
    result["total_lesion_voxels"] = int(df["lesion_voxels"].sum())
    result["mean_lesion_size"]   = float(df["lesion_voxels"].mean())
    result["max_lesion_size"]    = float(df["lesion_voxels"].max())
    result["lesion_size_std"]    = float(df["lesion_voxels"].std()) if len(df) > 1 else 0.0
    result["lesion_size_cv"]     = (
        result["lesion_size_std"] / max(result["mean_lesion_size"], 1)
    )   # coefficient of variation — high = unequal lesion sizes
    result["mean_lesion_hu"]     = float(df["lesion_mean_hu"].mean())

    # 6 statistics per radiomic feature
    for col in feat_cols:
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue
        result[f"{col}_mean"] = float(np.mean(vals))
        result[f"{col}_max"]  = float(np.max(vals))
        result[f"{col}_min"]  = float(np.min(vals))
        result[f"{col}_std"]  = float(np.std(vals))  if len(vals) > 1 else 0.0
        result[f"{col}_skew"] = float(scipy_stats.skew(vals)) if len(vals) > 2 else 0.0
        result[f"{col}_kurt"] = float(scipy_stats.kurtosis(vals)) if len(vals) > 2 else 0.0

    return result


# ── Agatston helper ───────────────────────────────────────────────────────────

def voxels_to_category(voxels: int) -> int:
    if voxels == 0:    return 0
    if voxels <= 500:  return 1
    if voxels <= 2000: return 2
    return 3


# ── Main processing ───────────────────────────────────────────────────────────

def run_per_lesion_extraction(
    scan_index_csv: str,
    images_root:    str,
    output_dir:     str,
    n_patients:     int = None,
):
    scan_index_csv = Path(scan_index_csv)
    images_root    = Path(images_root)
    output_dir     = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scan index — only patients with calcium
    df = pd.read_csv(scan_index_csv)
    df = df[df["voxels"] > 0].copy()

    if n_patients:
        # Sample balanced across categories
        df["cat"] = df["voxels"].apply(voxels_to_category)
        per_cat   = max(1, n_patients // 3)
        selected  = pd.concat([
            df[df["cat"] == c].sample(min(per_cat, len(df[df["cat"] == c])),
                                      random_state=42)
            for c in [1, 2, 3]
        ]).head(n_patients).reset_index(drop=True)
    else:
        selected = df.reset_index(drop=True)

    print(f"[Per-Lesion] Processing {len(selected)} patients...")

    # Build extractor once
    extractor = featureextractor.RadiomicsFeatureExtractor(**RADIOMICS_SETTINGS)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("glrlm")

    rows   = []
    failed = []

    for _, row in tqdm(selected.iterrows(), total=len(selected),
                       desc="Per-lesion extraction"):
        scan_id    = row["scan_id"]
        patient_id = str(row["patient_id"])

        img_path = images_root / scan_id / f"{scan_id}_img.nii.gz"
        seg_path = images_root / scan_id / f"{scan_id}_seg.nii.gz"

        if not img_path.exists() or not seg_path.exists():
            failed.append(patient_id)
            continue

        try:
            lesion_list = extract_lesion_features(img_path, seg_path, extractor)

            if not lesion_list:
                failed.append(patient_id)
                continue

            aggregated = aggregate_lesion_features(lesion_list)
            cat        = voxels_to_category(int(row["voxels"]))

            rows.append({
                "patient_id":        patient_id,
                "scan_id":           scan_id,
                "voxels":            int(row["voxels"]),
                "agatston_category": cat,
                "agatston_label":    AGATSTON_LABELS[cat],
                **aggregated,
            })

        except Exception as e:
            tqdm.write(f"  [ERROR] Patient {patient_id}: {e}")
            failed.append(patient_id)

    df_out   = pd.DataFrame(rows)
    out_path = output_dir / "per_lesion_features.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\n[Per-Lesion] Done!")
    print(f"  Successful : {len(rows)}")
    print(f"  Failed     : {len(failed)}")
    print(f"  Features   : {len(df_out.columns)} columns")
    print(f"  Output     : {out_path}")

    if not df_out.empty:
        _print_lesion_summary(df_out)
        _plot_lesion_analysis(df_out, output_dir)

    return df_out


def _print_lesion_summary(df: pd.DataFrame) -> None:
    print("\n── Per-Lesion Summary ────────────────────────────────────")
    for cat in sorted(df["agatston_category"].unique()):
        sub = df[df["agatston_category"] == cat]
        print(f"\n  {AGATSTON_LABELS[cat]} ({len(sub)} patients):")
        print(f"    Avg lesion count : {sub['lesion_count'].mean():.1f}")
        print(f"    Max lesion count : {sub['lesion_count'].max()}")
        print(f"    Avg lesion size  : {sub['mean_lesion_size'].mean():.1f} voxels")
        print(f"    Size variability : {sub['lesion_size_cv'].mean():.2f} (CV)")
    print("──────────────────────────────────────────────────────────\n")


def _plot_lesion_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cat_colors = {
        "Mild (1-99)":        "#2196F3",
        "Moderate (100-399)": "#FF9800",
        "Severe (>=400)":     "#F44336",
    }
    order = ["Mild (1-99)", "Moderate (100-399)", "Severe (>=400)"]
    order = [o for o in order if o in df["agatston_label"].unique()]

    # 1. Lesion count per category
    ax = axes[0, 0]
    sns.boxplot(data=df, x="agatston_label", y="lesion_count",
                order=order, ax=ax,
                hue="agatston_label", palette=cat_colors, legend=False)
    ax.set_title("Number of Calcium Lesions per Patient", fontsize=11)
    ax.set_xlabel("Agatston Category")
    ax.set_ylabel("Lesion Count")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Mean lesion size per category
    ax = axes[0, 1]
    sns.boxplot(data=df, x="agatston_label", y="mean_lesion_size",
                order=order, ax=ax,
                hue="agatston_label", palette=cat_colors, legend=False)
    ax.set_title("Average Lesion Size per Patient", fontsize=11)
    ax.set_xlabel("Agatston Category")
    ax.set_ylabel("Mean Lesion Size (voxels)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Lesion size variability (CV)
    ax = axes[1, 0]
    sns.boxplot(data=df, x="agatston_label", y="lesion_size_cv",
                order=order, ax=ax,
                hue="agatston_label", palette=cat_colors, legend=False)
    ax.set_title("Lesion Size Variability (CV)\n"
                 "Higher = more unequal lesion sizes", fontsize=10)
    ax.set_xlabel("Agatston Category")
    ax.set_ylabel("Coefficient of Variation")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Lesion count vs total voxels scatter
    ax = axes[1, 1]
    for label in order:
        sub = df[df["agatston_label"] == label]
        ax.scatter(sub["lesion_count"], sub["total_lesion_voxels"],
                   label=label, color=cat_colors[label],
                   alpha=0.6, s=50, edgecolors="white", linewidths=0.3)
    ax.set_xlabel("Number of Lesions")
    ax.set_ylabel("Total Calcium Voxels")
    ax.set_title("Lesion Count vs Total Calcium Volume\n"
                 "Many small vs few large lesions?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Lesion Analysis — Beyond the Agatston Score",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "per_lesion_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: per_lesion_analysis.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_index",
        default=r"D:\Du_hoc\gsoc\processed\tables\scan_index.csv")
    parser.add_argument("--images_root",
        default=r"D:\Du_hoc\gsoc\processed\images")
    parser.add_argument("--output_dir",
        default=r"D:\Du_hoc\gsoc\project2_radiomics\results")
    parser.add_argument("--n_patients", type=int, default=None,
        help="Number of patients to process (default: all with calcium)")
    args = parser.parse_args()

    run_per_lesion_extraction(
        scan_index_csv = args.scan_index,
        images_root    = args.images_root,
        output_dir     = args.output_dir,
        n_patients     = args.n_patients,
    )
