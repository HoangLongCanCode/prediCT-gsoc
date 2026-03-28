# PrediCT GSoC — COCA Dataset Pipeline & Radiomics Analysis

This repository contains my evaluation tasks for the **PrediCT Google Summer of Code project**
at Stanford AIMI. It includes a full preprocessing pipeline for the COCA cardiac CT dataset
and a radiomics feature extraction + statistical analysis pipeline for Project 2.

---

## Repository Structure

```
prediCT-gsoc/
├── common_task/              # Common Task: COCA Dataset Preprocessing
│   ├── COCA_pipeline.py      # Main runner — orchestrates all steps
│   ├── COCA_processor.py     # DICOM → NIfTI image + segmentation mask
│   ├── COCA_resampler.py     # Resamples to target voxel spacing
│   ├── unnester.py           # Flattens nested DICOM folder structure
│   ├── splits.py             # Stratified train/val/test split
│   ├── dataloader.py         # PyTorch data loader with augmentation
│   ├── dataset_statistics.py # Generates dataset statistics report
│   └── justification.md      # Written justification + dataset statistics
│
├── project2_radiomics/       # Specific Task: Project 2 (Radiomics & Phenotyping) - Feature Extraction
│   ├── extract_features.py       # PyRadiomics feature extraction + Agatston scores
│   ├── statistical_analysis.py   # Spearman, Kruskal-Wallis, visualizations
│   ├── unsupervised_analysis.py  # K-Means clustering, UMAP, phenotype characterization
│   └── results/
│       ├── features.csv           # Extracted features for 23 patients
│       ├── spearman_results.csv   # Spearman correlation results
│       ├── kruskal_results.csv    # Kruskal-Wallis test results
│       ├── correlation_matrix.png
│       ├── significant_features.png
│       ├── agatston_distribution.png
│       ├── tsne.png
│       ├── umap.png
│       ├── cluster_selection.png
│       ├── phenotype_profiles.png
│       ├── cluster_agatston_distribution.png
│       └── cluster_assignments.csv
│
└── README.md
```

---

## Dataset

**COCA — Coronary Calcium and Chest CTs** (Stanford AIMI)
- 787 patients, ECG-gated cardiac CT scans
- Plist XML segmentation masks annotating coronary calcium deposits
- Downloaded via AzCopy from Stanford AIMI Azure storage

> Dataset requires individual registration at https://stanfordaimi.azurewebsites.net/

---

## Common Task: COCA Dataset Preprocessing

### Goal
Build a preprocessing and data loading pipeline tailored to Project 2 (Radiomics).

### Setup

```bash
# Create conda environment (Python 3.9 recommended)
conda create -n gsoc python=3.9
conda activate gsoc

# Install dependencies
pip install numpy pandas SimpleITK opencv-python tqdm scikit-learn torch openpyxl
```

### Usage

```bash
cd common_task
python COCA_pipeline.py
```

The pipeline runs interactively and walks through 3 steps:

**Step 1 — Unnesting**
Flattens the variable-named scanner subfolders (e.g. `Pro_Gated_CS_3.0_I30f_3_70%`)
so DICOM slices sit directly inside each patient folder.

**Step 2 — Processing (DICOM → NIfTI)**
Loads each patient's DICOM series as a 3D volume using SimpleITK, parses the XML
calcium segmentation mask, and saves both as compressed `.nii.gz` files.

**Step 3 — Resampling**
Resamples all volumes to a uniform spacing of **0.7 × 0.7 × 3.0 mm** using B-spline
interpolation for images and nearest-neighbour for masks.

### Results
```
787 patients processed
40,113 DICOM files → 787 image.nii.gz + 787 seg.nii.gz
449 patients with XML calcium annotations
447 patients with non-zero calcium voxels
```

### Dataset Statistics

| Category | Voxel Range | Count | Percentage |
|----------|-------------|-------|------------|
| None | 0 | 340 | 43.2% |
| Mild | 1–500 | 258 | 32.8% |
| Moderate | 501–2000 | 114 | 14.5% |
| Severe | >2000 | 75 | 9.5% |
| **Total** | | **787** | **100%** |

### Stratified Split (70/15/15)

| Category | Train (550) | Val (118) | Test (119) |
|----------|-------------|-----------|------------|
| None | 238 (43.3%) | 51 (43.2%) | 51 (42.9%) |
| Mild | 180 (32.7%) | 39 (33.1%) | 39 (32.8%) |
| Moderate | 80 (14.5%) | 17 (14.4%) | 17 (14.3%) |
| Severe | 52 (9.5%) | 11 (9.3%) | 12 (10.1%) |

### Data Loader

```python
from splits import make_splits
from dataloader import make_dataloaders

train_df, val_df, test_df = make_splits("processed/tables/scan_index.csv")
train_loader, val_loader, test_loader = make_dataloaders(
    train_df, val_df, test_df,
    resampled_root="processed/data_resampled"
)

# Each batch returns:
# image : torch.Size([2, 1, 80, 512, 512])  — (B, C, Z, Y, X)
# mask  : torch.Size([2, 1, 80, 512, 512])  — calcium segmentation
# label : tensor([2, 1])                    — Agatston category
```

**Design choices for radiomics compatibility:**
- HU windowing: [−200, 1000] — captures calcium range (≥130 HU) while excluding noise
- Augmentation: left-right flip + mild intensity shift (±2%) only — elastic deformation
  excluded as it corrupts texture features (GLCM, GLSZM, GLRLM)
- WeightedRandomSampler corrects 43% none-class imbalance during training

---

## Specific Task: Project 2 (Radiomics & Phenotyping) - Feature Extraction

### Goal
Demonstrate radiomics feature extraction from COCA scans and correlate features
with Agatston scores.

### Setup

```bash
conda activate gsoc
pip install pyradiomics scipy scikit-learn matplotlib seaborn
pip install "numpy<2"   # required for PyRadiomics compatibility
```

### Usage

```bash
cd project2_radiomics

# Step 1: Extract features
python extract_features.py

# Step 2: Statistical analysis + plots
python statistical_analysis.py
```

### Features Extracted

From 23 patients (balanced across Agatston categories):

| Feature Class | Features |
|---------------|----------|
| **Shape** | Sphericity, SurfaceVolumeRatio, Maximum3DDiameter, MeshVolume, VoxelVolume |
| **GLCM** | Contrast, Correlation, InverseDifferenceMoment (Idm), JointEnergy, DifferenceVariance |
| **GLSZM** | SmallAreaEmphasis, LargeAreaEmphasis, ZonePercentage, GrayLevelNonUniformity |
| **GLRLM** | ShortRunEmphasis, LongRunEmphasis, RunPercentage, RunLengthNonUniformity |
| **Optional** | Max HU, Mean HU, Total Volume (mm³) |

### Agatston Score Calculation

Calculated from original spacing images using the standard clinical formula:
```
score += pixel_area_mm² × n_calcium_pixels × density_factor  (per slice)

density_factor:
  130–199 HU → 1
  200–299 HU → 2
  300–399 HU → 3
  ≥400    HU → 4
```

Categories: 0 | 1–99 | 100–399 | ≥400

### Statistical Results

**11 out of 18 features statistically significant (p < 0.05)**

Top features by Spearman correlation with Agatston score:

| Feature | Spearman ρ | p-value |
|---------|-----------|---------|
| glszm_GrayLevelNonUniformity | +0.966 | <0.0001 |
| glrlm_RunLengthNonUniformity | +0.960 | <0.0001 |
| shape_MeshVolume | +0.933 | <0.0001 |
| shape_VoxelVolume | +0.933 | <0.0001 |
| glcm_JointEnergy | −0.911 | <0.0001 |
| shape_Sphericity | −0.884 | <0.0001 |
| shape_SurfaceVolumeRatio | −0.838 | <0.0001 |
| glcm_Contrast | +0.762 | <0.0001 |
| shape_Maximum3DDiameter | +0.733 | <0.0001 |
| glcm_DifferenceVariance | +0.726 | <0.0001 |

**Key findings:**
- Volume features (MeshVolume, VoxelVolume) are the strongest positive predictors —
  larger calcium deposits correlate with higher Agatston scores
- Sphericity is strongly negatively correlated — severe calcium is more irregular,
  not compact round nodules
- Texture features (GrayLevelNonUniformity, RunLengthNonUniformity) reflect increasing
  textural complexity as calcium burden grows
- t-SNE shows mild separation between Mild and Severe categories in feature space

### Visualizations

| Plot | Description |
|------|-------------|
| `correlation_matrix.png` | Spearman correlation heatmap of all features |
| `significant_features.png` | Box plots of top 6 significant features by Agatston category |
| `agatston_distribution.png` | Distribution of Agatston categories in extracted sample |
| `tsne.png` | t-SNE of radiomic feature space coloured by Agatston category |

---

## Requirements

```
numpy<2
pandas
SimpleITK
opencv-python
tqdm
scikit-learn
torch
scipy
matplotlib
seaborn
pyradiomics
openpyxl
```

---

### Unsupervised Analysis & Key Findings

### Motivation

The Agatston score reduces complex 3D calcium morphology to a single number.
Two patients with the same score can have fundamentally different calcium patterns —
one with a single compact nodule, another with diffuse irregular deposits spread across
multiple vessels. These differences may carry different clinical risk but are invisible
to the score alone. Unsupervised clustering on radiomic features reveals these hidden phenotypes.

### Method

K-Means clustering (k=4, selected via silhouette score) was applied to the standardized
radiomic feature vectors of 23 patients. UMAP was used for 2D visualization.
Phenotypes were characterized by inspecting mean feature values per cluster.

### Discovered Phenotypes

| Cluster | Patients | Avg Agatston | Sphericity | Volume | Texture Complexity | Phenotype |
|---------|----------|-------------|------------|--------|-------------------|-----------|
| 1 | 7 | 67.8 | 0.599 | 59.2 | Low | Small & Compact — focal round nodules |
| 2 | 5 | 3005.3 | 0.278 | 2574.2 | Very High | Large & Heterogeneous — dense complex plaques |
| 3 | 1 | 0.0 | 0.370 | 399.5 | Low | Homogeneous — no significant calcium |
| 4 | 10 | 681.1 | 0.354 | 621.5 | Medium | Mixed — moderate volume, irregular texture |

### Key Insight — Cluster 4 vs Cluster 2

Both Cluster 2 and Cluster 4 contain high Agatston score patients ("Severe"),
yet their radiomic profiles are strikingly different:

```
Cluster 2 (avg score 3005): Sphericity=0.278, Volume=2574, Texture=59.15
Cluster 4 (avg score 681):  Sphericity=0.354, Volume=621,  Texture=21.32
```

Cluster 2 patients have calcium that is far more irregular, larger and texturally complex.
Under current clinical practice both groups would receive the same risk classification
and treatment recommendation based on Agatston score alone.

### Hypothesis

Irregular, heterogeneous calcium deposits (Cluster 2) may represent a more vulnerable
plaque phenotype — more prone to rupture — compared to compact, homogeneous deposits
(Cluster 4) with similar scores. Radiomic features capture this morphological distinction
that the Agatston score cannot.

This supports the case for **radiomic phenotyping as a complement to Agatston scoring**
in cardiac risk stratification — which is precisely the goal of the PrediCT Project 2.

### Unsupervised Analysis Outputs

| File | Description |
|------|-------------|
| `cluster_selection.png` | Elbow + silhouette plots justifying k=4 |
| `umap.png` | UMAP coloured by cluster vs Agatston category side by side |
| `phenotype_profiles.png` | Z-score heatmap of feature profiles per cluster |
| `cluster_agatston_distribution.png` | Agatston category distribution within each cluster |
| `cluster_assignments.csv` | Patient-level cluster assignments |

---

---

## Author

**HoangLongCan**
GSoC 2025 applicant — PrediCT (Stanford AIMI)
