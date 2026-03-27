"""
splits.py
Stratified train/val/test split for the COCA Gated dataset.

Since the Gated dataset has no Agatston scores file, we derive calcium
burden categories directly from the voxel counts in scan_index.csv:

    Category 0 → 0 voxels        (no calcium)
    Category 1 → 1–500 voxels    (mild)
    Category 2 → 501–2000 voxels (moderate)
    Category 3 → >2000 voxels    (severe)

These thresholds are approximate mappings from voxel count to clinical
Agatston categories, appropriate for the 0.7×0.7×3.0mm resampled spacing.

Usage
-----
    from splits import make_splits, compute_sample_weights

    train_df, val_df, test_df = make_splits(
        scan_index_csv = r"D:\Du_hoc\gsoc\processed\tables\scan_index.csv"
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit


# ── Calcium burden categories ─────────────────────────────────────────────────

CATEGORY_LABELS = {
    0: "None (0 voxels)",
    1: "Mild (1–500)",
    2: "Moderate (501–2000)",
    3: "Severe (>2000)",
}


def voxels_to_category(voxels: int) -> int:
    if voxels == 0:
        return 0
    elif voxels <= 500:
        return 1
    elif voxels <= 2000:
        return 2
    else:
        return 3


# ── Main split function ───────────────────────────────────────────────────────

def make_splits(
    scan_index_csv: str | Path,
    train_frac:     float = 0.70,
    val_frac:       float = 0.15,
    seed:           int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read scan_index.csv, assign categories, perform stratified split.

    Returns
    -------
    train_df, val_df, test_df
        Each DataFrame has columns:
        patient_id | scan_id | voxels | num_slices | has_xml |
        folder_path | category | category_label | split
    """
    df = pd.read_csv(scan_index_csv)
    df["patient_id"] = df["patient_id"].astype(str)

    # Assign category from voxel count
    df["category"]       = df["voxels"].apply(voxels_to_category)
    df["category_label"] = df["category"].map(CATEGORY_LABELS)

    # ── Split 1: train vs (val + test) ───────────────────────────────────────
    test_frac = 1.0 - train_frac - val_frac
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(val_frac + test_frac), random_state=seed
    )
    train_idx, tmp_idx = next(sss1.split(df, df["category"]))
    train_df = df.iloc[train_idx].copy()
    tmp_df   = df.iloc[tmp_idx].copy()

    # ── Split 2: val vs test ──────────────────────────────────────────────────
    val_ratio = val_frac / (val_frac + test_frac)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - val_ratio), random_state=seed
    )
    val_idx, test_idx = next(sss2.split(tmp_df, tmp_df["category"]))
    val_df  = tmp_df.iloc[val_idx].copy()
    test_df = tmp_df.iloc[test_idx].copy()

    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"

    _print_stats(train_df, val_df, test_df)

    return train_df, val_df, test_df


# ── Class imbalance weights ───────────────────────────────────────────────────

def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Inverse-frequency weights per sample.
    Pass to torch.utils.data.WeightedRandomSampler.
    """
    counts  = Counter(df["category"])
    total   = len(df)
    weights = np.array([
        total / (len(counts) * counts[cat])
        for cat in df["category"]
    ], dtype=np.float32)
    return weights


def get_dataset_statistics(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a summary DataFrame of category counts per split.
    Useful for the written deliverable.
    """
    rows = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for cat, label in CATEGORY_LABELS.items():
            n = (split_df["category"] == cat).sum()
            rows.append({
                "split":    split_name,
                "category": cat,
                "label":    label,
                "count":    n,
                "pct":      f"{100 * n / len(split_df):.1f}%",
            })
    return pd.DataFrame(rows)


# ── Printing ──────────────────────────────────────────────────────────────────

def _print_stats(train_df, val_df, test_df):
    print("\n── Dataset split statistics ──────────────────────────")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n  {name} ({len(df)} patients):")
        for cat, label in CATEGORY_LABELS.items():
            n = (df["category"] == cat).sum()
            bar = "█" * int(20 * n / len(df))
            print(f"    {label:25s}: {n:3d}  {bar}")
    print("\n──────────────────────────────────────────────────────\n")


# ── CLI: run directly to preview splits ──────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan_index",
        default=r"D:\Du_hoc\gsoc\processed\tables\scan_index.csv",
        help="Path to scan_index.csv"
    )
    parser.add_argument(
        "--output_dir",
        default=r"D:\Du_hoc\gsoc\processed\tables",
        help="Where to save the split CSVs"
    )
    args = parser.parse_args()

    train_df, val_df, test_df = make_splits(args.scan_index)

    # Save split manifests
    out = Path(args.output_dir)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(  out / "val.csv",   index=False)
    test_df.to_csv( out / "test.csv",  index=False)
    print(f"Split CSVs saved to {out}")

    # Print full statistics table
    stats = get_dataset_statistics(train_df, val_df, test_df)
    print("\nFull statistics:")
    print(stats.to_string(index=False))
