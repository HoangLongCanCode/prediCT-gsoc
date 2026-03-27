"""
extract_features.py
Extracts PyRadiomics features and calculates Agatston scores
from 20-30 COCA Gated scans.

Uses:
- Original spacing images (processed/images/) for Agatston score
- Resampled images (processed/data_resampled/) for radiomic features

Output:
- features.csv : one row per patient, columns = features + agatston score

Usage
-----
    python extract_features.py
    python extract_features.py --n_scans 25 --output_dir "D:/Du_hoc/gsoc/results"
"""

import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import logging

logging.getLogger("radiomics").setLevel(logging.ERROR)
from radiomics import featureextractor


# ── Agatston score calculation ────────────────────────────────────────────────

def calculate_agatston(img_path, seg_path):
    """
    Calculate Agatston score from ORIGINAL spacing image + mask.

    Formula per slice:
        score += pixel_area_mm2 * n_calcium_pixels * density_factor
    density_factor:
        130-199 HU -> 1
        200-299 HU -> 2
        300-399 HU -> 3
        >=400   HU -> 4
    """
    img = sitk.ReadImage(str(img_path))
    seg = sitk.ReadImage(str(seg_path))

    img_arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (Z, Y, X)
    seg_arr = sitk.GetArrayFromImage(seg).astype(np.uint8)     # (Z, Y, X)

    spacing        = img.GetSpacing()                           # (x, y, z) mm
    pixel_area_mm2 = spacing[0] * spacing[1]

    # Only voxels in mask AND above calcium threshold
    calcium_mask = (seg_arr == 1) & (img_arr >= 130)

    if not np.any(calcium_mask):
        return {
            "agatston_score":    0.0,
            "agatston_category": 0,
            "agatston_label":    "None(0)",
            "max_hu":            0.0,
            "mean_hu":           0.0,
            "total_volume_mm3":  0.0,
        }

    agatston = 0.0
    for z in range(img_arr.shape[0]):
        slice_mask = calcium_mask[z]
        if not np.any(slice_mask):
            continue
        max_hu = float(img_arr[z][slice_mask].max())
        area   = float(np.sum(slice_mask)) * pixel_area_mm2
        if max_hu >= 400:
            density = 4
        elif max_hu >= 300:
            density = 3
        elif max_hu >= 200:
            density = 2
        else:
            density = 1
        agatston += area * density

    if agatston == 0:
        cat = 0
    elif agatston < 100:
        cat = 1
    elif agatston < 400:
        cat = 2
    else:
        cat = 3

    labels = {0: "None(0)", 1: "Mild(1-99)", 2: "Moderate(100-399)", 3: "Severe(>=400)"}
    calcium_hu = img_arr[calcium_mask]

    return {
        "agatston_score":    round(agatston, 2),
        "agatston_category": cat,
        "agatston_label":    labels[cat],
        "max_hu":            round(float(calcium_hu.max()), 2),
        "mean_hu":           round(float(calcium_hu.mean()), 2),
        "total_volume_mm3":  round(
            float(np.sum(calcium_mask)) * spacing[0] * spacing[1] * spacing[2], 2
        ),
    }


# ── PyRadiomics feature extraction ───────────────────────────────────────────

def extract_radiomics(img_path, seg_path):
    """
    Extract radiomic features from RESAMPLED image + mask.
    Returns dict of feature_name -> value, or None on failure.

    Feature names use PyRadiomics 3.x naming:
      - Idm  = InverseDifferenceMoment (GLCM)
      - Homogeneity1 = Homogeneity (GLCM)
    """
    try:
        settings = {
            "binWidth":       25,
            "minimumROISize": 5,
            "verbose":        False,
        }
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        extractor.enableFeaturesByName(
            shape = ["Sphericity", "SurfaceVolumeRatio", "Maximum3DDiameter", "MeshVolume", "VoxelVolume"],
            glcm  = ["Contrast", "Correlation", "Idm", "Homogeneity1", "JointEnergy", "DifferenceVariance"],
            glszm = ["SmallAreaEmphasis", "LargeAreaEmphasis", "ZonePercentage", "GrayLevelNonUniformity"],
            glrlm = ["ShortRunEmphasis", "LongRunEmphasis", "RunPercentage", "RunLengthNonUniformity"],
        )

        result = extractor.execute(str(img_path), str(seg_path))

        features = {}
        for k, v in result.items():
            if k.startswith("original_") and "diagnostics" not in k:
                try:
                    features[k.replace("original_", "")] = float(v)
                except (TypeError, ValueError):
                    pass

        return features if features else None

    except Exception as e:
        print(f"    [RADIOMICS ERROR] {e}")
        return None


# ── Patient selection ─────────────────────────────────────────────────────────

def select_patients(scan_index_csv, n_scans=25, seed=42):
    """
    Select n_scans patients with non-zero calcium, balanced across categories.
    """
    df = pd.read_csv(scan_index_csv)
    df = df[df["voxels"] > 0].copy()

    def cat(v):
        if v <= 500:  return 1
        if v <= 2000: return 2
        return 3

    df["cat"] = df["voxels"].apply(cat)

    selected = []
    per_cat  = max(1, n_scans // 3)
    for c in [1, 2, 3]:
        subset = df[df["cat"] == c]
        k      = min(per_cat, len(subset))
        selected.append(subset.sample(k, random_state=seed))

    result = pd.concat(selected).head(n_scans).reset_index(drop=True)
    print(f"[Selection] Selected {len(result)} patients with calcium:")
    print(result["cat"].value_counts().sort_index().to_string())
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run_extraction(
    scan_index_csv,
    original_root,
    resampled_root,
    output_dir,
    n_scans=25,
):
    scan_index_csv = Path(scan_index_csv)
    original_root  = Path(original_root)
    resampled_root = Path(resampled_root)
    output_dir     = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_patients(scan_index_csv, n_scans=n_scans)

    rows   = []
    failed = []

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Extracting"):
        scan_id    = row["scan_id"]
        patient_id = str(row["patient_id"])

        orig_img = original_root  / scan_id / f"{scan_id}_img.nii.gz"
        orig_seg = original_root  / scan_id / f"{scan_id}_seg.nii.gz"
        res_img  = resampled_root / scan_id / f"{scan_id}_img.nii.gz"
        res_seg  = resampled_root / scan_id / f"{scan_id}_seg.nii.gz"

        if not all(p.exists() for p in [orig_img, orig_seg, res_img, res_seg]):
            tqdm.write(f"  [SKIP] Patient {patient_id}: missing files")
            failed.append(patient_id)
            continue

        agatston = calculate_agatston(orig_img, orig_seg)
        features = extract_radiomics(res_img, res_seg)

        if features is None:
            tqdm.write(f"  [SKIP] Patient {patient_id}: radiomics failed")
            failed.append(patient_id)
            continue

        rows.append({
            "patient_id": patient_id,
            "scan_id":    scan_id,
            "voxels":     int(row["voxels"]),
            **agatston,
            **features,
        })

    df_out   = pd.DataFrame(rows)
    out_path = output_dir / "features.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\n[Extraction] Done!")
    print(f"  Successful : {len(rows)}")
    print(f"  Failed     : {len(failed)}")
    print(f"  Output     : {out_path}")

    if not df_out.empty:
        print(f"\nAgatston distribution:")
        print(df_out["agatston_label"].value_counts().to_string())
        print(f"\nFeatures extracted: {len([c for c in df_out.columns if '_' in c and c not in ['scan_id','patient_id','agatston_label']])}")

    return df_out


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_index",
        default=r"D:\Du_hoc\gsoc\processed\tables\scan_index.csv")
    parser.add_argument("--original_root",
        default=r"D:\Du_hoc\gsoc\processed\images")
    parser.add_argument("--resampled_root",
        default=r"D:\Du_hoc\gsoc\processed\data_resampled")
    parser.add_argument("--output_dir",
        default=r"D:\Du_hoc\gsoc\results")
    parser.add_argument("--n_scans", type=int, default=25)
    args = parser.parse_args()

    run_extraction(
        scan_index_csv = args.scan_index,
        original_root  = args.original_root,
        resampled_root = args.resampled_root,
        output_dir     = args.output_dir,
        n_scans        = args.n_scans,
    )