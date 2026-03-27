"""
COCA_processor.py
Processes the Gated_release_final COCA dataset:
  - Loads each patient's DICOM series (after unnesting)
  - Parses the corresponding XML segmentation mask
  - Saves image + mask as .nii.gz
  - Writes a scan_index.csv for the resampler

Expected structure AFTER running unnester.py:
    Gated_release_final/
    ├── calcium_xml/
    │   ├── 1.xml
    │   ├── 2.xml
    │   └── ...
    └── patient/
        ├── 1/
        │   ├── IM-0001-0001.dcm
        │   ├── IM-0001-0002.dcm
        │   └── ...
        ├── 2/
        └── ...

Usage
-----
    python COCA_processor.py \
        --gated_root "D:\Du_hoc\gsoc\cocacoronarycalciumandchestcts-2\Gated_release_final" \
        --output_dir "D:\Du_hoc\gsoc\processed"
"""

import os
import json
import hashlib
import plistlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from tqdm import tqdm


class COCAProcessor:
    def __init__(self, gated_root: str, output_dir: str):
        """
        Parameters
        ----------
        gated_root : path to Gated_release_final folder
                     (contains calcium_xml/ and patient/)
        output_dir : where to write processed NIfTI files and tables
        """
        self.gated_root = Path(gated_root)
        self.dicom_root = self.gated_root / "patient"
        self.xml_root   = self.gated_root / "calcium_xml"

        self.out_images = Path(output_dir) / "images"
        self.out_tables = Path(output_dir) / "tables"

        self.out_images.mkdir(parents=True, exist_ok=True)
        self.out_tables.mkdir(parents=True, exist_ok=True)

        self._validate_paths()

    # ── validation ────────────────────────────────────────────────────────────

    def _validate_paths(self):
        if not self.dicom_root.exists():
            raise FileNotFoundError(f"Patient DICOM root not found: {self.dicom_root}")
        if not self.xml_root.exists():
            raise FileNotFoundError(f"XML root not found: {self.xml_root}")
        print(f"[Processor] DICOM root : {self.dicom_root}")
        print(f"[Processor] XML root   : {self.xml_root}")
        print(f"[Processor] Output dir : {self.out_images.parent}")

    # ── stable scan ID ────────────────────────────────────────────────────────

    @staticmethod
    def generate_stable_id(*parts: str, n: int = 12) -> str:
        h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
        return h[:n]

    # ── DICOM discovery ───────────────────────────────────────────────────────

    def discover_patients(self) -> list[Path]:
        """
        Return sorted list of patient DICOM dirs that contain ≥5 .dcm files
        directly inside (i.e. after unnesting).
        """
        patients = []
        for p in sorted(self.dicom_root.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
            if not p.is_dir() or not p.name.isdigit():
                continue
            dcm_files = list(p.glob("*.dcm"))
            if len(dcm_files) >= 5:
                patients.append(p)
            else:
                # Maybe unnester hasn't been run yet — check one level deeper
                nested_dcms = list(p.rglob("*.dcm"))
                if nested_dcms:
                    print(f"  [WARN] Patient {p.name}: DICOMs still nested. Run unnester.py first.")
        return patients

    # ── XML mask parsing ──────────────────────────────────────────────────────

    def parse_xml_mask(self, xml_path: Path, image_shape: tuple) -> tuple[np.ndarray, list]:
        """
        Parse a COCA plist XML file into a binary 3-D mask.

        Parameters
        ----------
        xml_path     : path to the .xml file (may not exist for score-0 patients)
        image_shape  : (Z, Y, X) shape of the loaded DICOM volume

        Returns
        -------
        mask          : uint8 ndarray of shape (Z, Y, X)
        seg_slices    : sorted list of Z indices that contain calcium
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        seg_slices = []
        total_z, total_y, total_x = image_shape

        if not xml_path.exists():
            return mask, seg_slices   # patient has score 0, no annotation

        try:
            with open(xml_path, "rb") as f:
                data = plistlib.load(f)

            for img_entry in data.get("Images", []):
                z = int(img_entry.get("ImageIndex", -1))
                if z < 0 or z >= total_z:
                    continue

                for roi in img_entry.get("ROIs", []):
                    points_str = roi.get("Point_px", [])
                    if not points_str:
                        continue

                    poly_points = []
                    for p_str in points_str:
                        cleaned = p_str.replace("(", "").replace(")", "")
                        parts = cleaned.split(",")
                        if len(parts) == 2:
                            try:
                                poly_points.append([float(parts[0]), float(parts[1])])
                            except ValueError:
                                continue

                    if not poly_points:
                        continue

                    pts = np.array(poly_points, dtype=np.int32)
                    temp_slice = np.zeros((total_y, total_x), dtype=np.uint8)

                    if len(pts) > 2:
                        cv2.fillPoly(temp_slice, [pts], 1)
                    else:
                        for pt in pts:
                            x_coord, y_coord = int(pt[0]), int(pt[1])
                            if 0 <= x_coord < total_x and 0 <= y_coord < total_y:
                                temp_slice[y_coord, x_coord] = 1

                    if np.any(temp_slice):
                        mask[z] = np.logical_or(mask[z], temp_slice).astype(np.uint8)
                        seg_slices.append(z)

        except Exception as e:
            print(f"  [XML ERROR] {xml_path.name}: {e}")

        return mask, sorted(set(seg_slices))

    # ── main processing loop ──────────────────────────────────────────────────

    def process_all(self) -> pd.DataFrame:
        """
        Process all discovered patient DICOM series.
        Returns the scan index DataFrame.
        """
        patient_dirs = self.discover_patients()
        print(f"\n[Processor] Found {len(patient_dirs)} patients. Processing...\n")

        rows = []

        for patient_dir in tqdm(patient_dirs, desc="Processing"):
            patient_id = patient_dir.name
            xml_path   = self.xml_root / f"{patient_id}.xml"

            try:
                # ── Load DICOM volume ─────────────────────────────────────
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(patient_dir))

                if not dicom_names:
                    tqdm.write(f"  [SKIP] Patient {patient_id}: no DICOM series found")
                    continue

                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                img_array = sitk.GetArrayFromImage(image)   # (Z, Y, X)

                # ── Parse XML mask ────────────────────────────────────────
                mask_array, seg_slices = self.parse_xml_mask(xml_path, img_array.shape)
                voxel_count = int(np.sum(mask_array))

                if xml_path.exists() and voxel_count == 0:
                    tqdm.write(
                        f"  [WARN] Patient {patient_id}: XML exists but 0 voxels drawn. "
                        "Check slice alignment."
                    )

                # ── Save outputs ──────────────────────────────────────────
                scan_id     = self.generate_stable_id(str(patient_dir.resolve()), patient_id)
                scan_folder = self.out_images / scan_id
                scan_folder.mkdir(parents=True, exist_ok=True)

                # Image NIfTI
                img_out = scan_folder / f"{scan_id}_img.nii.gz"
                sitk.WriteImage(image, str(img_out), useCompression=True)

                # Mask NIfTI (inherits geometry from image)
                mask_sitk = sitk.GetImageFromArray(mask_array)
                mask_sitk.CopyInformation(image)
                seg_out = scan_folder / f"{scan_id}_seg.nii.gz"
                sitk.WriteImage(mask_sitk, str(seg_out), useCompression=True)

                # Metadata JSON
                meta = {
                    "scan_id":            scan_id,
                    "patient_id":         patient_id,
                    "calcium_voxels":     voxel_count,
                    "slices_with_calcium": seg_slices,
                    "image_shape":        list(img_array.shape),
                    "spacing":            list(image.GetSpacing()),
                    "original_path":      str(patient_dir),
                }
                (scan_folder / f"{scan_id}_meta.json").write_text(
                    json.dumps(meta, indent=2)
                )

                rows.append({
                    "patient_id":   patient_id,
                    "scan_id":      scan_id,
                    "voxels":       voxel_count,
                    "num_slices":   len(seg_slices),
                    "has_xml":      xml_path.exists(),
                    "folder_path":  str(scan_folder),
                })

            except Exception as e:
                tqdm.write(f"  [ERROR] Patient {patient_id}: {e}")

        # ── Save index ────────────────────────────────────────────────────────
        df = pd.DataFrame(rows)
        index_path = self.out_tables / "scan_index.csv"
        df.to_csv(index_path, index=False)

        print(f"\n[Processor] Done. {len(df)} scans processed.")
        print(f"[Processor] Scan index → {index_path}")
        print(f"\n── Calcium distribution ──────────────────────────")
        print(f"  With XML mask  : {df['has_xml'].sum()}")
        print(f"  Zero voxels    : {(df['voxels'] == 0).sum()}")
        print(f"  Non-zero voxels: {(df['voxels'] > 0).sum()}")
        print(f"──────────────────────────────────────────────────\n")

        return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCA Gated dataset processor")
    parser.add_argument(
        "--gated_root",
        default=r"D:\Du_hoc\gsoc\cocacoronarycalciumandchestcts-2\Gated_release_final",
        help="Path to Gated_release_final folder"
    )
    parser.add_argument(
        "--output_dir",
        default=r"D:\Du_hoc\gsoc\processed",
        help="Where to save processed NIfTI files"
    )
    args = parser.parse_args()

    processor = COCAProcessor(
        gated_root=args.gated_root,
        output_dir=args.output_dir,
    )
    processor.process_all()
