"""
COCA_pipeline.py
Runs the full COCA Gated preprocessing pipeline in order:
    Step 1 → unnester.py    : flatten nested DICOM subfolders
    Step 2 → COCA_processor : DICOM → NIfTI image + mask + scan_index.csv
    Step 3 → COCA_resampler : resample to target voxel spacing

Usage (interactive):
    python COCA_pipeline.py

Usage (CLI):
    python COCA_pipeline.py \
        --gated_root  "D:\Du_hoc\gsoc\cocacoronarycalciumandchestcts-2\Gated_release_final" \
        --output_dir  "D:\Du_hoc\gsoc\processed" \
        --spacing     0.7 0.7 3.0 \
        --steps       all
"""

import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from unnester      import flatten_all
    from COCA_processor import COCAProcessor
    from COCA_resampler import COCAResampler
    print("[Pipeline] Modules loaded successfully.\n")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print(f"Make sure unnester.py, COCA_processor.py and COCA_resampler.py")
    print(f"are in the same folder as this script: {SCRIPT_DIR}")
    sys.exit(1)


DEFAULT_GATED_ROOT = r"D:\Du_hoc\gsoc\cocacoronarycalciumandchestcts-2\Gated_release_final"
DEFAULT_OUTPUT_DIR = r"D:\Du_hoc\gsoc\processed"
DEFAULT_SPACING    = [0.7, 0.7, 3.0]


def prompt(label: str, default: str) -> str:
    val = input(f"{label} [{default}]: ").strip()
    return val if val else default


def run_interactive():
    print("=" * 55)
    print("       COCA GATED PREPROCESSING PIPELINE")
    print("=" * 55)

    gated_root = prompt("Gated_release_final path", DEFAULT_GATED_ROOT)
    output_dir = prompt("Output directory",          DEFAULT_OUTPUT_DIR)

    print("\nWhat would you like to run?")
    print("  1) Full pipeline  (unnest → process → resample)")
    print("  2) Unnest only")
    print("  3) Process only   (assumes already unnested)")
    print("  4) Resample only  (assumes already processed)")
    choice = input("Selection [1]: ").strip() or "1"

    spacing_str = prompt("Target voxel spacing x y z (mm)", "0.7 0.7 3.0")
    try:
        spacing = [float(v) for v in spacing_str.split()]
        assert len(spacing) == 3
    except Exception:
        print("[WARN] Invalid spacing, using default 0.7 0.7 3.0")
        spacing = DEFAULT_SPACING

    run_pipeline(
        gated_root = gated_root,
        output_dir = output_dir,
        spacing    = spacing,
        steps      = {"1": "all", "2": "unnest", "3": "process", "4": "resample"}.get(choice, "all"),
    )


def run_pipeline(gated_root, output_dir, spacing=None, steps="all"):
    spacing      = spacing or DEFAULT_SPACING
    patient_root = str(Path(gated_root) / "patient")

    if steps in ("all", "unnest"):
        print("\n" + "─" * 55)
        print("STEP 1 / 3 — Unnesting DICOM subfolders")
        print("─" * 55)
        flatten_all(patient_root, dry_run=False)

    if steps in ("all", "process"):
        print("\n" + "─" * 55)
        print("STEP 2 / 3 — Processing DICOM → NIfTI")
        print("─" * 55)
        processor = COCAProcessor(gated_root=gated_root, output_dir=output_dir)
        processor.process_all()

    if steps in ("all", "resample"):
        print("\n" + "─" * 55)
        print(f"STEP 3 / 3 — Resampling to {spacing} mm")
        print("─" * 55)
        resampler = COCAResampler(project_root=output_dir, target_spacing=spacing)
        resampler.run()

    print("\n" + "=" * 55)
    print("Pipeline finished.")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCA Gated preprocessing pipeline")
    parser.add_argument("--gated_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--spacing", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--steps",
                        choices=["all", "unnest", "process", "resample"],
                        default=None)
    args = parser.parse_args()

    if args.gated_root is None and args.output_dir is None and args.steps is None:
        run_interactive()
    else:
        run_pipeline(
            gated_root = args.gated_root or DEFAULT_GATED_ROOT,
            output_dir = args.output_dir or DEFAULT_OUTPUT_DIR,
            spacing    = args.spacing    or DEFAULT_SPACING,
            steps      = args.steps      or "all",
        )
