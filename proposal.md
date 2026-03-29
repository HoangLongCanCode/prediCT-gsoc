# Beyond the Agatston Score: A Radiomic Phenotyping Framework for Coronary Calcium Morphology Discovery and Validation

**GSoC 2025 Proposal — PrediCT / ML4SCI / Stanford AIMI**

**Applicant:** Long Huynh
**University:** The University of Alabama (Undergraduate)
**GitHub:** https://github.com/HoangLongCanCode/prediCT-gsoc
**Project Duration:** 175 hours

---

## Abstract

The Agatston score has guided coronary artery calcium (CAC) risk stratification for over three decades, yet it reduces the full 3D complexity of calcium morphology to a single integer. A growing body of evidence demonstrates that calcium **density**, **lesion count**, and **spatial distribution** provide independent prognostic information beyond total calcium burden — but these dimensions are invisible to the Agatston score.

This proposal addresses four interconnected research questions using the COCA gated dataset (787 patients, Stanford AIMI):

1. Does calcium density stratification reveal clinically meaningful risk differences within the same Agatston category?
2. Do radiomic features discover natural calcium phenotypes that cut across Agatston boundaries?
3. Does per-lesion feature extraction, aggregated across all lesions of a patient, capture morphological information invisible to patient-level analysis?
4. Can discovered phenotypes be rigorously validated without clinical endpoints?

Preliminary evaluation results on the full COCA dataset provide direct, affirmative evidence for the first three questions: mild Agatston patients exhibit 58.2% low-density (vulnerable) calcium versus 35.7% for severe patients; unsupervised clustering reveals patients with identical Agatston categories but strikingly different calcium morphologies; and per-lesion analysis shows severe patients have 6× more discrete lesions than mild patients. These findings motivate a full-scale phenotyping framework culminating in an interactive clinical dashboard that translates complex radiomics into interpretable, evidence-linked insight at the point of care.

---

## 1. Problem Statement and Research Gap

### 1.1 The Agatston Score and Its Limits

The Agatston score, introduced in 1990, quantifies coronary artery calcium (CAC) as the product of calcified plaque area and a density weighting factor summed across all coronary slices. Its prognostic power is well-established: a score of zero essentially excludes obstructive coronary disease, while scores above 400 identify patients at markedly elevated risk for major adverse cardiac events (MACE). Guidelines from the AHA/ACC endorse CAC scoring as a key tool in primary prevention decision-making.

However, the Agatston score conflates two fundamentally distinct dimensions of calcium biology into a single number: *how much* calcium is present (burden) and *what kind* of calcium it is (morphology). This conflation has measurable clinical consequences.

### 1.2 The Density Paradox

Criqui et al. (JAMA, 2014) demonstrated a striking paradox in the Multi-Ethnic Study of Atherosclerosis (MESA, n = 3,398): when controlling for total calcium volume, high-density calcium (>400 HU) is *inversely* associated with cardiovascular events. It is *low-density*, spotty calcium (130–200 HU) that correlates with plaque vulnerability and rupture risk. The mechanistic explanation is histopathological: low-density calcium represents early, active microcalcification associated with lipid-rich, inflammation-prone plaques, while high-density calcium reflects mature, fibrocalcific deposits that mechanically stabilise the plaque cap.

The clinical implication is profound: two patients with identical Agatston scores of 200 may face very different prognoses depending on whether their calcium is predominantly low-density (high rupture risk) or high-density (paradoxically stable). The Agatston score is blind to this distinction.

### 1.3 Beyond Burden: Spatial and Morphological Features

The calcium-omics study by Hoori et al. (Scientific Reports, 2024) demonstrated that hand-crafted calcium features — including lesion count, spatial diffusivity, and LAD artery mass — improved MACE identification by 13.2% over Agatston alone (C-index 71.6%, 2-year AUC 74.8%, n = 2,457). Critically, lesion count and diffusivity were among the top predictors, confirming that *how many* lesions a patient has, and *where* they are distributed, carries independent prognostic signal.

The Framingham Heart Study radiomics analysis further showed that extracting radiomic features *per individual calcium lesion* — rather than treating the entire calcium burden as a single region of interest — and aggregating with summary statistics (mean, max, skewness, kurtosis) significantly improved event prediction in a 624-patient cohort followed for over 9 years.

### 1.4 The Validation Problem

The SCOT-HEART trial analysis showed that PCA-derived "eigen features" of plaque morphology improved myocardial infarction risk discrimination beyond Agatston score and quantitative plaque burden. Collectively, these studies establish that radiomic phenotyping of coronary calcium is clinically meaningful — yet no systematic, validated framework exists for the COCA dataset, which lacks MACE endpoints. Validating radiomic phenotypes without clinical outcomes is an open methodological challenge that this proposal directly addresses.

---

## 2. Preliminary Results: Evidence for the Research Questions

As part of the evaluation task, I built a complete pipeline for the full COCA gated dataset and obtained preliminary results that provide direct empirical support for the proposed research questions.

### 2.1 Full Preprocessing Pipeline (787 Patients)

I built a complete preprocessing pipeline including: DICOM → NIfTI conversion with plist XML calcium mask parsing; resampling to 0.7 × 0.7 × 3.0 mm; stratified 70/15/15 train/val/test split across four Agatston categories; and a PyTorch dataloader with radiomics-safe augmentation. Elastic deformation was deliberately excluded because it alters the spatial structure that GLCM and GLSZM texture features depend on — a design choice motivated by the requirements of downstream radiomics. The pipeline processed all 787 patients (40,113 DICOM files) with zero failures.

### 2.2 Research Question 1: Does Density Stratification Reveal Hidden Risk?

I developed a **calcium density fingerprinting** method that classifies each patient's calcium voxels into four clinical HU bins motivated by the Criqui et al. density framework, and applied it to all 447 patients with non-zero calcium:

| Agatston Category | Low-density % (risky) | High-density % (protective) |
|-------------------|----------------------|----------------------------|
| Mild (1–99) | 58.2 ± 22.0 | 5.5 ± 8.7 |
| Moderate (100–399) | 42.4 ± 11.1 | 13.6 ± 9.3 |
| Severe (≥400) | 35.7 ± 7.9 | 19.8 ± 9.2 |

This confirms in the COCA dataset the density paradox documented by Criqui et al. in MESA: **mild Agatston patients carry the most dangerous density profile**. Three observations deserve emphasis:

- The low-density percentage decreases monotonically with Agatston severity (58.2% → 35.7%), while high-density percentage increases (5.5% → 19.8%) — consistent with the biological model of calcium maturation over time.
- The within-category standard deviation for mild patients (22.0%) is substantially larger than for severe patients (7.9%), suggesting mild Agatston patients are the most heterogeneous in terms of underlying plaque biology — and therefore the cohort where density stratification adds the most value.
- These patterns hold over 447 real patients, not a small pilot cohort, giving the findings statistical weight and clinical credibility.

### 2.3 Research Question 2: Do Radiomic Phenotypes Cut Across Agatston Categories?

I extracted 18 PyRadiomics features (Shape, GLCM, GLSZM, GLRLM) from 23 patients and performed K-Means clustering (k=4, selected via silhouette score). Statistical analysis revealed 11 of 18 features significantly correlated with Agatston scores (Spearman ρ, Kruskal-Wallis, p < 0.05):

| Feature | Spearman ρ | p-value | Clinical interpretation |
|---------|-----------|---------|------------------------|
| glszm_GrayLevelNonUniformity | +0.966 | <0.0001 | Textural heterogeneity increases with severity |
| glrlm_RunLengthNonUniformity | +0.960 | <0.0001 | Calcium streaking becomes more complex |
| shape_MeshVolume | +0.933 | <0.0001 | Larger deposits correlate with higher score |
| glcm_JointEnergy | −0.911 | <0.0001 | Severe calcium is texturally disordered |
| shape_Sphericity | −0.884 | <0.0001 | Severe calcium is irregular, not round |
| shape_SurfaceVolumeRatio | −0.838 | <0.0001 | Severe deposits are spatially spread-out |

The unsupervised clustering revealed four phenotypes:

| Cluster | N | Avg Score | Sphericity | Volume | Phenotype |
|---------|---|-----------|------------|--------|-----------|
| 1 | 7 | 67.8 | 0.599 | 59.2 | Small & Compact — focal round nodules |
| 2 | 5 | 3005.3 | 0.278 | 2574.2 | Large & Heterogeneous — complex plaques |
| 3 | 1 | 0.0 | 0.370 | 399.5 | Homogeneous — negligible calcium |
| 4 | 10 | 681.1 | 0.354 | 621.5 | Mixed pattern — moderate, irregular |

The critical finding is **Clusters 2 and 4**: both contain Severe Agatston patients, yet their morphological profiles are strikingly different. Cluster 2 is characterised by very low sphericity (0.278), massive volume (2574 voxels), and high texture complexity — hallmarks of irregular, multi-focal, vulnerable deposits. Cluster 4 patients have the same Agatston classification but show more compact, less texturally complex calcium. Under current clinical practice, both groups receive identical risk classification and treatment recommendations.

### 2.4 Research Question 3: Does Per-Lesion Analysis Reveal Hidden Structure?

Implementing the Framingham radiomics methodology, I applied SimpleITK connected component analysis to identify individual calcium lesions, extracted radiomic features per lesion, and aggregated with 6 summary statistics (mean, max, min, std, skewness, kurtosis), yielding **432 features per patient** from 48 patients:

| Category | Avg Lesion Count | Avg Lesion Size (vox) | Size CV |
|----------|-----------------|----------------------|---------|
| Mild | 3.1 | 90 | 0.50 |
| Moderate | 9.1 | 133 | 1.27 |
| Severe | 19.6 | 237 | 1.73 |

Three findings stand out. First, severe patients have 6× more discrete lesions than mild patients — consistent with the diffuse multi-vessel disease pattern identified as a top MACE predictor in Hoori et al. Second, the lesion size coefficient of variation (CV) increases dramatically with severity (0.50 → 1.73), indicating that severe patients typically have one or two dominant large lesions alongside many small ones — a heterogeneous spatial architecture invisible to total calcium volume. Third, none of these lesion-level patterns are captured by the Agatston score.

### 2.5 Clinical Dashboard

To make these findings interpretable at the point of care, I built a Streamlit dashboard providing for each patient: a CT slice viewer with auto-jump to the most calcium-dense slice; three overlay modes (raw CT, calcium highlighted, density color map where each voxel is colored by HU value); zoom inset with Ca²⁺ labels; GMM phenotype classification with confidence score; density profile bar chart; per-lesion metrics; an auto-generated clinical narrative citing primary literature; and cohort context histograms.

![Calcium Phenotype Dashboard](project2_radiomics/dashboard_demo.png)

*Patient 5 (Severe ≥400, 6,301 calcium voxels). Density color map shows 27.7% low-density (red/risky) and 33.3% high-density (green/protective) calcium. Despite the Severe Agatston category, the GMM assigns "Dense Stable" with 100% confidence — directly illustrating the density paradox. The clinical narrative auto-cites Criqui et al. (JAMA 2014). Cohort context histograms show where this patient sits relative to all 447 patients.*

---

## 3. Proposed Methodology

### 3.1 Phase 1: Expanded Feature Extraction

**Full-cohort patient-level features.** Scale the existing PyRadiomics pipeline to all 447 calcium-positive patients, targeting 50–100 features across Shape, GLCM, GLSZM, GLRLM, first-order intensity, and spatial feature classes. Spatial features — lesion count, inter-lesion Euclidean distances, diffusivity index — directly implement the features identified as most predictive in Hoori et al. Density fingerprint features are appended from the existing full-cohort analysis.

**Per-lesion extraction at scale.** Extend the per-lesion pipeline to all 447 patients. Connected component analysis identifies individual lesions; PyRadiomics extracts features per lesion; 6 aggregation statistics produce ~108 features per patient. This follows the Framingham methodology and captures lesion-level morphological variance invisible to patient-level features.

**PCA eigen features.** Motivated by the SCOT-HEART analysis, apply PCA to the standardised combined feature matrix. The top principal components — "eigen features" — capture dominant axes of calcium morphological variation and are compact, interpretable as linear combinations of original features, and have demonstrated prognostic value in coronary plaque radiomics.

### 3.2 Phase 2: Multi-Algorithm Phenotype Discovery

A single clustering algorithm is insufficient for robust phenotype discovery. Four complementary algorithms will be applied: **K-Means** (baseline, computationally efficient); **hierarchical clustering** with Ward linkage (captures nested phenotype structure, produces a dendrogram); **DBSCAN** (density-based, identifies phenotypes of irregular shape and handles outlier patients); and **Gaussian Mixture Models** (soft probabilistic membership, identifies patients at phenotype boundaries whose ambiguous assignment is itself clinically interesting).

Optimal hyperparameters are selected via silhouette score, Davies-Bouldin index, and Calinski-Harabasz index. An ensemble consensus approach assigns final phenotype labels based on agreement across algorithms; patients where algorithms disagree are flagged as "ambiguous phenotype" — a designation with clinical value, as boundary patients may represent transitional disease states.

### 3.3 Phase 3: Validation Without Clinical Endpoints

This is the most methodologically novel contribution. Validating phenotypes without MACE data requires a multi-pronged framework that establishes validity through convergent evidence.

**Agatston stratification.** Phenotypes should meaningfully stratify Agatston categories — not reproduce them exactly (no added value) but partially align with them (confirming phenotypes capture clinically relevant variance). Chi-squared tests and Cramér's V effect size quantify this relationship.

**Perturbation consistency (feature stability).** Clinically useful features must be stable under realistic image perturbations simulating scanner-to-scanner variability. Controlled perturbations — rotation (±5°, ±10°, ±15°), translation (±2 mm, ±5 mm), and Gaussian noise (σ = 10, 25, 50 HU) — are applied to 50 randomly selected patients. Intraclass Correlation Coefficient (ICC) is computed per feature; features with ICC > 0.75 are retained as "robust." Adjusted Rand Index (ARI) measures phenotype assignment stability; ARI > 0.80 confirms phenotypes are reproducible under realistic acquisition variation.

**Clinical pattern alignment.** Discovered phenotypes are mapped to literature-defined calcium morphology patterns — spotty, dense focal, and diffuse — using Normalized Mutual Information (NMI). Alignment confirms that data-driven phenotypes correspond to patterns described in the clinical literature, establishing face validity.

**Feature importance via Random Forest + SHAP.** A Random Forest classifier trained on phenotype pseudo-labels identifies which features most strongly distinguish phenotypes. SHAP values provide patient-level explanations, answering: *"Why was this patient assigned to Phenotype A rather than B?"* Per-phenotype SHAP waterfall plots make the classification interpretable to both researchers and clinicians.

### 3.4 Phase 4: Enhanced Clinical Dashboard

The existing dashboard is extended with: full-cohort UMAP coloured by discovered phenotype; a phenotype comparison view placing two patients with the same Agatston category but different phenotypes side by side; a SHAP feature importance panel; and a reproducibility badge displaying the patient's feature stability score from perturbation testing.

The automated narrative generation is extended to reference SHAP-identified key features and flag patients with low phenotype confidence as requiring closer morphological review. All narrative claims are explicitly linked to primary literature citations, and the disclaimer that findings are AI-generated and require clinical interpretation is prominent.

---

## 4. Novelty and Significance

**The density fingerprinting result is novel in the COCA dataset.** While Criqui et al. established the density paradox in MESA, no prior work has systematically characterised density profiles across Agatston categories in the COCA gated dataset. The finding that mild Agatston patients carry the most dangerous density profile (58.2% vs. 35.7% for severe) has direct clinical implications and establishes that the density paradox generalises beyond MESA.

**Per-lesion aggregation is methodologically distinct from prior COCA analyses.** Treating each calcium deposit as a separate entity mirrors the biological reality that coronary calcium consists of discrete plaques at different stages of development. The 6× lesion count difference between mild and severe patients, and the dramatic increase in lesion size variability (CV: 0.50 → 1.73), provide morphological evidence for qualitatively different disease processes that the Agatston score conflates.

**The validation framework addresses an open methodological challenge.** All prior coronary radiomics studies validated phenotypes against clinical outcomes. PrediCT lacks MACE endpoints. The proposed convergent validation framework — perturbation consistency, clinical pattern alignment, Agatston stratification, and SHAP interpretability — provides a rigorous alternative that does not require outcome data. This is a methodological contribution that extends beyond the COCA dataset.

**The clinical dashboard closes the translational gap.** Radiomic features exist as numbers in spreadsheets. The dashboard translates them into a physician-interpretable report anchored in the actual CT image, with literature-linked clinical narratives. No existing open-source tool provides this level of integration for coronary calcium phenotyping.

---

## 5. Qualifications

**Demonstrated technical execution.** Rather than implementing only the suggested pipeline, I extended the evaluation task with density fingerprinting (447 patients, full cohort), per-lesion extraction (432 features per patient), unsupervised phenotyping with UMAP, and an interactive clinical dashboard — all grounded in the literature the mentors provided. The Cluster 2 vs. Cluster 4 finding emerged organically from the data and directly motivates the proposed work.

**Domain-informed methodology.** The proposed pipeline is not generic machine learning applied to medical images. Every design decision is motivated by specific findings in the four papers linked in the project description: per-lesion aggregation from Framingham, eigen features from SCOT-HEART, density stratification from Criqui et al., and spatial diffusivity from the calcium-omics study.

**Clinical translation awareness.** The dashboard's design reflects an understanding of how AI tools need to be positioned in clinical settings. The auto-generated narrative cites specific papers. The phenotype confidence score is displayed, not hidden. The disclaimer is prominent and unambiguous. These choices reflect the clinical adoption barriers that explainability and transparency research has identified.

**Python and ML depth.** With 5+ years of Python experience, I am proficient in the full scientific stack: NumPy, pandas, scikit-learn, PyTorch, SimpleITK, and PyRadiomics, along with experience in research projects combining machine learning and data visualisation.

---

## References

1. Agatston AS, Janowitz WR, Hildner FJ, Zusmer NR, Viamonte M, Detrano R. Quantification of coronary artery calcium using ultrafast computed tomography. *Journal of the American College of Cardiology*. 1990;15(4):827–832. https://doi.org/10.1016/0735-1097(90)90282-T

2. Criqui MH, Denenberg JO, Ix JH, McClelland RL, Wassel CL, Rifkin DE, Carr JJ, Budoff MJ, Allison MA. Calcium density of coronary artery plaque and risk of incident cardiovascular events. *JAMA*. 2014;311(3):271–278. https://doi.org/10.1001/jama.2013.282535

3. Hoori A, Al-Kindi S, Hu T, Song Y, Wu H, Lee J, Tashtish N, Fu P, Gilkeson R, Rajagopalan S, Wilson DL. Enhancing cardiovascular risk prediction through AI-enabled calcium-omics. *Scientific Reports*. 2024;14:11134. https://doi.org/10.1038/s41598-024-60584-8

4. Williams MC, Kwiecinski J, Doris MK, et al. Coronary plaque radiomic phenotypes predict fatal or nonfatal myocardial infarction: analysis of the SCOT-HEART trial. *JACC: Cardiovascular Imaging*. 2024. https://doi.org/10.1016/j.jcmg.2024.09.002

5. Maaniitty T, Stenström I, Bax JJ, et al. Radiomics of coronary artery calcium in the Framingham Heart Study. *Radiology: Cardiothoracic Imaging*. 2020;2(5):e190119. https://doi.org/10.1148/ryct.2020190119

6. Abaid A, Guidone G, Sharif F, Ullah I. 3D CT-Based Coronary Calcium Assessment: A Feature-Driven Machine Learning Framework. *Springer LNCS*. 2025. https://doi.org/10.1007/978-3-032-09569-5_34

7. van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational radiomics system to decode the radiographic phenotype. *Cancer Research*. 2017;77(21):e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339

8. Bhatia HS, McClelland RL, Denenberg J, Budoff M, Allison MC. Coronary artery calcium density and cardiovascular events by volume level: the MESA. *Circulation: Cardiovascular Imaging*. 2023;16:e014788. https://doi.org/10.1161/CIRCIMAGING.122.014788

9. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*. 2017;30. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
