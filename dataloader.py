"""
dataloader.py
PyTorch Dataset + DataLoader for the resampled COCA Gated dataset.

Reads resampled NIfTI pairs from:
    processed/data_resampled/<scan_id>/<scan_id>_img.nii.gz
    processed/data_resampled/<scan_id>/<scan_id>_seg.nii.gz

Designed for radiomics compatibility (Project 2):
  - Returns both image AND mask so PyRadiomics can use them directly
  - HU windowing applied on load (calcium window: -200 to 1000 HU)
  - Augmentation is optional and mild (preserves texture features)
  - WeightedRandomSampler handles class imbalance

Usage
-----
    from splits import make_splits
    from dataloader import make_dataloaders

    train_df, val_df, test_df = make_splits("processed/tables/scan_index.csv")
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df,
        resampled_root="D:/Du_hoc/gsoc/processed/data_resampled"
    )

    for batch in train_loader:
        images = batch["image"]       # (B, 1, Z, Y, X) float32 tensor
        masks  = batch["mask"]        # (B, 1, Z, Y, X) uint8 tensor
        labels = batch["label"]       # (B,) long tensor  — category 0-3
        pids   = batch["patient_id"]  # list of str
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

from splits import compute_sample_weights


# ── HU windowing ──────────────────────────────────────────────────────────────

HU_MIN = -200.0   # below this = air/background, clip
HU_MAX = 1000.0   # above this = dense bone, clip
# Calcium appears at ~130 HU — well within this window


def apply_hu_window(arr: np.ndarray) -> np.ndarray:
    """Clip to calcium-relevant HU range and normalise to [0, 1]."""
    arr = np.clip(arr, HU_MIN, HU_MAX).astype(np.float32)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
    return arr


def pad_to_size(
    arr: np.ndarray,
    target: tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad a (Z, Y, X) array to target size with pad_value.
    If arr is already larger than target in any dim, it is centre-cropped.
    """
    result = np.full(target, pad_value, dtype=arr.dtype)
    # Compute how much of arr fits
    z = min(arr.shape[0], target[0])
    y = min(arr.shape[1], target[1])
    x = min(arr.shape[2], target[2])
    result[:z, :y, :x] = arr[:z, :y, :x]
    return result


# Target size — covers >99% of COCA gated volumes at 0.7×0.7×3.0mm
# (Z, Y, X) — adjust if you see crops in the logs
TARGET_SHAPE = (80, 512, 512)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """
    Mild augmentation that is safe for radiomics:
      - Random left-right flip  (anatomically valid for chest CTs)
      - Mild intensity shift     (simulates scanner variability)

    NOT applied:
      - Elastic deformation      (changes shape features)
      - Aggressive scaling       (changes texture statistics)
      - Rotations > 10°          (changes orientation-sensitive features)
    """
    # Random flip along X axis
    if rng.random() < 0.5:
        image = np.flip(image, axis=2).copy()
        mask  = np.flip(mask,  axis=2).copy()

    # Mild intensity shift ±2%
    shift  = rng.uniform(-0.02, 0.02)
    image  = np.clip(image + shift, 0.0, 1.0)

    return image, mask


# ── Dataset ───────────────────────────────────────────────────────────────────

class COCADataset(Dataset):
    """
    Parameters
    ----------
    df             : split DataFrame from make_splits()
    resampled_root : folder containing <scan_id>/<scan_id>_img.nii.gz
    is_train       : if True, applies augmentation
    seed           : random seed for augmentation reproducibility
    """

    def __init__(
        self,
        df:             pd.DataFrame,
        resampled_root: str | Path,
        is_train:       bool = True,
        seed:           int  = 42,
    ):
        self.df             = df.reset_index(drop=True)
        self.resampled_root = Path(resampled_root)
        self.is_train       = is_train
        self.rng            = np.random.default_rng(seed)

        self._validate()

    def _validate(self):
        missing = []
        for _, row in self.df.iterrows():
            img_path = self._img_path(row["scan_id"])
            if not img_path.exists():
                missing.append(row["scan_id"])
        if missing:
            print(f"[DataLoader] WARNING: {len(missing)} scans not found in resampled_root.")
            print(f"  First missing: {missing[0]}")
            print(f"  Make sure resampling has completed.")

    def _img_path(self, scan_id: str) -> Path:
        return self.resampled_root / scan_id / f"{scan_id}_img.nii.gz"

    def _seg_path(self, scan_id: str) -> Path:
        return self.resampled_root / scan_id / f"{scan_id}_seg.nii.gz"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row     = self.df.iloc[idx]
        scan_id = row["scan_id"]

        # ── Load image ────────────────────────────────────────────────────────
        img_sitk = sitk.ReadImage(str(self._img_path(scan_id)))
        img_arr  = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # (Z,Y,X)
        img_arr  = apply_hu_window(img_arr)

        # ── Load mask ─────────────────────────────────────────────────────────
        seg_path = self._seg_path(scan_id)
        if seg_path.exists():
            seg_sitk = sitk.ReadImage(str(seg_path))
            seg_arr  = sitk.GetArrayFromImage(seg_sitk).astype(np.uint8)
        else:
            seg_arr = np.zeros_like(img_arr, dtype=np.uint8)

        # ── Augmentation (train only) ─────────────────────────────────────────
        if self.is_train:
            img_arr, seg_arr = augment(img_arr, seg_arr, self.rng)

        # ── Pad to fixed size so batches can be stacked ───────────────────────
        img_arr = pad_to_size(img_arr, TARGET_SHAPE, pad_value=0.0)
        seg_arr = pad_to_size(seg_arr, TARGET_SHAPE, pad_value=0)

        # ── To tensors  (1, Z, Y, X) ─────────────────────────────────────────
        image_tensor = torch.from_numpy(img_arr).unsqueeze(0)
        mask_tensor  = torch.from_numpy(seg_arr.copy()).unsqueeze(0)

        return {
            "image":      image_tensor,                    # (1, Z, Y, X) float32
            "mask":       mask_tensor,                     # (1, Z, Y, X) uint8
            "label":      torch.tensor(row["category"],    dtype=torch.long),
            "voxels":     int(row["voxels"]),
            "patient_id": str(row["patient_id"]),
            "scan_id":    scan_id,
            "spacing":    img_sitk.GetSpacing(),           # (x, y, z) mm
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def make_dataloaders(
    train_df:       pd.DataFrame,
    val_df:         pd.DataFrame,
    test_df:        pd.DataFrame,
    resampled_root: str | Path,
    batch_size:     int  = 2,
    num_workers:    int  = 0,    # 0 = main process (safe on Windows)
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, test DataLoaders.

    Notes
    -----
    - batch_size=2 by default — 3D CT volumes are large
    - num_workers=0 by default — Windows has issues with multiprocessing
      and SimpleITK; increase only if you're on Linux
    - WeightedRandomSampler oversamples minority classes during training
    """
    train_ds = COCADataset(train_df, resampled_root, is_train=True)
    val_ds   = COCADataset(val_df,   resampled_root, is_train=False)
    test_ds  = COCADataset(test_df,  resampled_root, is_train=False)

    # Weighted sampler for class imbalance
    if use_weighted_sampler:
        weights = compute_sample_weights(train_df)
        sampler = WeightedRandomSampler(
            weights     = torch.from_numpy(weights),
            num_samples = len(weights),
            replacement = True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size  = batch_size,
            sampler     = sampler,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
    )

    print(f"[DataLoader] Train batches : {len(train_loader)}")
    print(f"[DataLoader] Val   batches : {len(val_loader)}")
    print(f"[DataLoader] Test  batches : {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from splits import make_splits

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_index",
        default=r"D:\Du_hoc\gsoc\processed\tables\scan_index.csv")
    parser.add_argument("--resampled_root",
        default=r"D:\Du_hoc\gsoc\processed\data_resampled")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    train_df, val_df, test_df = make_splits(args.scan_index)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df,
        resampled_root = args.resampled_root,
        batch_size     = args.batch_size,
    )

    print("\nLoading one batch from train loader...")
    batch = next(iter(train_loader))
    print(f"  image shape  : {batch['image'].shape}")
    print(f"  mask shape   : {batch['mask'].shape}")
    print(f"  labels       : {batch['label']}")
    print(f"  patient IDs  : {batch['patient_id']}")
    print(f"  voxel counts : {batch['voxels']}")
    print("\nDataLoader test passed!")