"""
unnester.py
Flattens the Gated_release_final dataset so each patient folder contains
.dcm files directly, instead of having them buried inside a variable-named
subfolder like:
    patient/1/Pro_Gated_CS_3.0_I30f_3_70%/*.dcm
        →
    patient/1/*.dcm

Edge cases handled:
  - Patients with TWO subfolders (e.g. patient 700, 726): files from
    both subfolders are merged; duplicate filenames get a prefix.
  - Subfolders that are already flat (no sub-subfolders): skipped safely.
  - Non-DICOM files inside subfolders: left in place.

Usage
-----
    python unnester.py --patient_root "D:\Du_hoc\gsoc\cocacoronarycalciumandchestcts-2\Gated_release_final\patient"

Or just run it and it will prompt you.
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def flatten_patient(patient_dir: Path, dry_run: bool = False) -> dict:
    """
    Flatten one patient directory.
    Returns a summary dict with counts of moved files and any issues.
    """
    result = {"patient": patient_dir.name, "moved": 0, "skipped": 0, "issues": []}

    # Find all subfolders directly under the patient dir
    subfolders = [p for p in patient_dir.iterdir() if p.is_dir()]

    if not subfolders:
        # Already flat or empty
        return result

    # Collect all .dcm files from all subfolders
    dcm_files = []
    for subfolder in subfolders:
        found = list(subfolder.rglob("*.dcm"))
        dcm_files.extend(found)

    if not dcm_files:
        result["issues"].append("No .dcm files found in subfolders")
        return result

    # Check for filename collisions when multiple subfolders exist
    seen_names = {}
    for dcm_path in dcm_files:
        name = dcm_path.name
        if name in seen_names:
            # Prefix with parent folder name to disambiguate
            new_name = f"{dcm_path.parent.name}_{name}"
        else:
            new_name = name
        seen_names[name] = seen_names.get(name, 0) + 1

        target = patient_dir / new_name

        if target.exists():
            result["skipped"] += 1
            continue

        if not dry_run:
            shutil.move(str(dcm_path), str(target))
        result["moved"] += 1

    # Remove now-empty subfolders
    if not dry_run:
        for subfolder in subfolders:
            try:
                shutil.rmtree(subfolder)
            except Exception as e:
                result["issues"].append(f"Could not remove {subfolder.name}: {e}")

    return result


def flatten_all(patient_root: str, dry_run: bool = False) -> None:
    root = Path(patient_root)

    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}")
        return

    # Only process numeric patient folders
    patient_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name)
    )

    if not patient_dirs:
        print(f"[ERROR] No numeric patient folders found in {root}")
        return

    print(f"Found {len(patient_dirs)} patient folders.")
    if dry_run:
        print("[DRY RUN] No files will be moved.\n")

    total_moved   = 0
    total_skipped = 0
    problems      = []

    for patient_dir in tqdm(patient_dirs, desc="Flattening"):
        result = flatten_patient(patient_dir, dry_run=dry_run)
        total_moved   += result["moved"]
        total_skipped += result["skipped"]
        if result["issues"]:
            problems.append(result)

    print(f"\n── Summary ───────────────────────────────────────")
    print(f"  Files moved   : {total_moved}")
    print(f"  Files skipped : {total_skipped}  (already in place)")
    if problems:
        print(f"  Patients with issues ({len(problems)}):")
        for p in problems:
            print(f"    Patient {p['patient']}: {'; '.join(p['issues'])}")
    else:
        print("  No issues.")
    print(f"──────────────────────────────────────────────────\n")
    print("Flattening complete. Each patient folder now contains .dcm files directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten Gated COCA DICOM structure")
    parser.add_argument(
        "--patient_root",
        default=None,
        help=r"Path to the patient folder, e.g. D:\Du_hoc\gsoc\cocacoronarycalciumandchestcts-2\Gated_release_final\patient"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview what would be moved without actually moving anything"
    )
    args = parser.parse_args()

    if args.patient_root is None:
        default = r"D:\Du_hoc\gsoc\cocacoronarycalciumandchestcts-2\Gated_release_final\patient"
        user_input = input(f"Patient root [{default}]: ").strip()
        patient_root = user_input or default
    else:
        patient_root = args.patient_root

    flatten_all(patient_root, dry_run=args.dry_run)
