"""
dataset_statistics.py
Generates dataset statistics from scan_index.csv and saves them as:
  - stats_summary.csv     : per-split category counts
  - stats_voxels.csv      : voxel distribution per category
  - dataset_statistics.md : human-readable report

Usage
-----
    python dataset_statistics.py
    python dataset_statistics.py --scan_index "D:/Du_hoc/gsoc/processed/tables/scan_index.csv"
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from splits import make_splits, CATEGORY_LABELS, voxels_to_category


def generate_statistics(
    scan_index_csv: str,
    output_dir:     str,
) -> None:
    scan_index_csv = Path(scan_index_csv)
    output_dir     = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and categorise ───────────────────────────────────────────────────
    df = pd.read_csv(scan_index_csv)
    df["category"]       = df["voxels"].apply(voxels_to_category)
    df["category_label"] = df["category"].map(CATEGORY_LABELS)

    # ── Splits ────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = make_splits(scan_index_csv)

    # ── Build stats tables ────────────────────────────────────────────────────

    # 1. Overall distribution
    overall = (
        df.groupby(["category", "category_label"])
        .agg(count=("patient_id", "count"))
        .reset_index()
        .sort_values("category")
    )
    overall["percentage"] = (overall["count"] / len(df) * 100).round(1)

    # 2. Per-split category counts
    split_rows = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for cat, label in CATEGORY_LABELS.items():
            n = (split_df["category"] == cat).sum()
            split_rows.append({
                "split":      split_name,
                "category":   cat,
                "label":      label,
                "count":      n,
                "percentage": round(100 * n / len(split_df), 1),
            })
    split_stats = pd.DataFrame(split_rows)

    # 3. Voxel distribution per category
    voxel_stats = (
        df.groupby("category_label")["voxels"]
        .agg(["min", "max", "mean", "median", "std"])
        .round(1)
        .reset_index()
    )

    # 4. Annotation coverage
    n_total        = len(df)
    n_with_xml     = int(df["has_xml"].sum())
    n_nonzero      = int((df["voxels"] > 0).sum())
    n_zero         = int((df["voxels"] == 0).sum())
    n_xml_no_voxel = int((df["has_xml"] & (df["voxels"] == 0)).sum())

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    split_stats.to_csv(output_dir / "stats_summary.csv",   index=False)
    voxel_stats.to_csv(output_dir / "stats_voxels.csv",    index=False)
    overall.to_csv(    output_dir / "stats_overall.csv",   index=False)

    # ── Write markdown report ─────────────────────────────────────────────────
    report = _build_report(
        df, overall, split_stats, voxel_stats,
        n_total, n_with_xml, n_nonzero, n_zero, n_xml_no_voxel,
        train_df, val_df, test_df,
    )
    report_path = output_dir / "dataset_statistics.md"
    report_path.write_text(report, encoding="utf-8")

    # ── Print to console ──────────────────────────────────────────────────────
    print(report)
    print(f"\nStatistics saved to: {output_dir}")
    print(f"  stats_overall.csv")
    print(f"  stats_summary.csv")
    print(f"  stats_voxels.csv")
    print(f"  dataset_statistics.md")


def _build_report(
    df, overall, split_stats, voxel_stats,
    n_total, n_with_xml, n_nonzero, n_zero, n_xml_no_voxel,
    train_df, val_df, test_df,
) -> str:

    lines = []
    lines.append("# COCA Gated Dataset — Statistics Report\n")

    # ── Overall ───────────────────────────────────────────────────────────────
    lines.append("## 1. Overall Dataset\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total patients | {n_total} |")
    lines.append(f"| Patients with XML annotation | {n_with_xml} ({100*n_with_xml/n_total:.1f}%) |")
    lines.append(f"| Patients with calcium (voxels > 0) | {n_nonzero} ({100*n_nonzero/n_total:.1f}%) |")
    lines.append(f"| Patients with no calcium (voxels = 0) | {n_zero} ({100*n_zero/n_total:.1f}%) |")
    lines.append(f"| XML exists but 0 voxels drawn | {n_xml_no_voxel} |")
    lines.append("")

    # ── Category distribution ─────────────────────────────────────────────────
    lines.append("## 2. Calcium Burden Categories\n")
    lines.append("Categories derived from calcium voxel count at 0.7×0.7×3.0mm spacing:\n")
    lines.append("| Category | Voxel Range | Count | Percentage |")
    lines.append("|----------|-------------|-------|------------|")
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['category_label']} | "
            f"{'0' if row['category']==0 else '1–500' if row['category']==1 else '501–2000' if row['category']==2 else '>2000'} | "
            f"{row['count']} | {row['percentage']}% |"
        )
    lines.append("")

    # ── Voxel stats ───────────────────────────────────────────────────────────
    lines.append("## 3. Voxel Count Distribution per Category\n")
    lines.append("| Category | Min | Max | Mean | Median | Std |")
    lines.append("|----------|-----|-----|------|--------|-----|")
    for _, row in voxel_stats.iterrows():
        lines.append(
            f"| {row['category_label']} | {row['min']} | {row['max']} | "
            f"{row['mean']} | {row['median']} | {row['std']} |"
        )
    lines.append("")

    # ── Split stats ───────────────────────────────────────────────────────────
    lines.append("## 4. Train / Val / Test Split (Stratified 70/15/15)\n")
    lines.append(f"| Category | Train ({len(train_df)}) | Val ({len(val_df)}) | Test ({len(test_df)}) |")
    lines.append("|----------|--------|-----|------|")
    for cat, label in CATEGORY_LABELS.items():
        tr = split_stats[(split_stats["split"]=="train") & (split_stats["category"]==cat)].iloc[0]
        va = split_stats[(split_stats["split"]=="val")   & (split_stats["category"]==cat)].iloc[0]
        te = split_stats[(split_stats["split"]=="test")  & (split_stats["category"]==cat)].iloc[0]
        lines.append(
            f"| {label} | {tr['count']} ({tr['percentage']}%) | "
            f"{va['count']} ({va['percentage']}%) | "
            f"{te['count']} ({te['percentage']}%) |"
        )
    lines.append("")

    # ── Data loader config ────────────────────────────────────────────────────
    lines.append("## 5. Data Loader Configuration\n")
    lines.append("| Parameter | Value | Rationale |")
    lines.append("|-----------|-------|-----------|")
    configs = [
        ("Voxel spacing",        "0.7 × 0.7 × 3.0 mm",   "PrediCT recommended"),
        ("HU window",            "[−200, 1000]",           "Calcium-relevant range"),
        ("Normalisation",        "Linear → [0, 1]",        "Model/radiomics compatible"),
        ("Pad size (Z×Y×X)",     "80 × 512 × 512",         "Covers >99% of volumes"),
        ("Batch size",           "2",                       "3D volumes are large"),
        ("Sampler",              "WeightedRandomSampler",   "Corrects class imbalance"),
        ("Train augmentation",   "H-flip + intensity ±2%", "Radiomics-safe"),
        ("Val/Test augmentation","None",                    "Deterministic evaluation"),
    ]
    for param, val, reason in configs:
        lines.append(f"| {param} | {val} | {reason} |")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan_index",
        default=r"D:\Du_hoc\gsoc\processed\tables\scan_index.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=r"D:\Du_hoc\gsoc\processed\tables",
    )
    args = parser.parse_args()

    generate_statistics(
        scan_index_csv = args.scan_index,
        output_dir     = args.output_dir,
    )
