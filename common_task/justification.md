# COCA Dataset Preprocessing Pipeline — Written Justification

## Pipeline Design

The preprocessing pipeline was built around the COCA Gated dataset, which contains
ECG-gated cardiac CT scans from 787 patients alongside plist XML segmentation masks
annotating coronary calcium deposits. The pipeline runs in three sequential stages.
First, an unnesting step flattens the variable-named scanner subfolders
(e.g. `Pro_Gated_CS_3.0_I30f_3_70%`) so that DICOM slices sit directly inside each
patient folder, enabling consistent programmatic access. Second, each patient's DICOM
series is stacked into a single 3D NIfTI volume using SimpleITK's ImageSeriesReader,
and the corresponding XML annotation is parsed into a binary segmentation mask where
voxels equal to 1 indicate calcium presence. Both the image and mask are saved as
compressed `.nii.gz` files. Third, all volumes are resampled to a uniform voxel
spacing of 0.7 × 0.7 × 3.0 mm using B-spline interpolation for images and
nearest-neighbour interpolation for masks, preserving label integrity while
standardising spatial resolution across scanners.

HU windowing clips each volume to the range [−200, 1000] Hounsfield Units and
linearly rescales to [0, 1]. This window was chosen specifically for calcium scoring:
it excludes irrelevant air (−1000 HU) and very dense bone artefacts above 1000 HU,
while retaining the clinically meaningful range in which coronary calcium appears
(≥130 HU). For radiomics compatibility (Project 2), augmentation is deliberately
conservative — only random left-right flips and mild additive intensity shifts (±2%)
are applied during training. Elastic deformations and aggressive intensity transforms
were excluded because PyRadiomics texture features (GLCM, GLSZM, GLRLM) are sensitive
to both spatial structure and voxel intensity distributions; distorting these would
invalidate the extracted features. The stratified train/val/test split (70/15/15)
uses voxel count as a proxy for calcium burden, grouping patients into four categories
(none, mild, moderate, severe) and ensuring each split reflects the same class
distribution. Class imbalance — 43% of patients have zero calcium — is addressed
during training via inverse-frequency weighted sampling (WeightedRandomSampler),
preventing the model from collapsing to a majority-class prediction.

---

## Dataset Statistics

### Overall Dataset (787 patients)

| Category         | Voxel Range   | Count | Percentage |
|------------------|---------------|-------|------------|
| None             | 0             | 340   | 43.2%      |
| Mild             | 1 – 500       | 258   | 32.8%      |
| Moderate         | 501 – 2000    | 114   | 14.5%      |
| Severe           | > 2000        |  75   |  9.5%      |
| **Total**        |               | **787** | **100%** |

### Annotation Coverage

| Metric                        | Value  |
|-------------------------------|--------|
| Patients with XML mask        | 449    |
| Patients with non-zero voxels | 447    |
| Patients with zero voxels     | 340    |
| Patients missing XML          | 338    |

> Note: 2 patients have XML files but 0 drawn voxels (patients 135 and 268),
> likely due to slice index misalignment in the original annotation.

### Train / Val / Test Split (Stratified 70 / 15 / 15)

| Category   | Train (550) | Val (118) | Test (119) |
|------------|-------------|-----------|------------|
| None       | 238 (43.3%) | 51 (43.2%)| 51 (42.9%) |
| Mild       | 180 (32.7%) | 39 (33.1%)| 39 (32.8%) |
| Moderate   |  80 (14.5%) | 17 (14.4%)| 17 (14.3%) |
| Severe     |  52  (9.5%) | 11  (9.3%)| 12 (10.1%) |

The near-identical percentages across all three splits confirm that stratification
was successful — each subset is a representative sample of the full dataset.

### Data Loader Configuration

| Parameter              | Value                        | Rationale                              |
|------------------------|------------------------------|----------------------------------------|
| Target voxel spacing   | 0.7 × 0.7 × 3.0 mm          | PrediCT recommended spacing            |
| HU window              | [−200, 1000]                 | Calcium-relevant range                 |
| Normalisation          | Linear → [0, 1]              | Neural network / radiomics compatible  |
| Pad size               | 80 × 512 × 512 voxels        | Covers >99% of COCA volumes            |
| Batch size             | 2                            | 3D volumes are memory-intensive        |
| Sampler                | WeightedRandomSampler        | Corrects 43% none-class imbalance      |
| Augmentation (train)   | H-flip + intensity ±2%       | Radiomics-safe — preserves texture     |
| Augmentation (val/test)| None                         | Deterministic evaluation               |
| num_workers            | 0                            | Windows-safe (no multiprocessing bug)  |
