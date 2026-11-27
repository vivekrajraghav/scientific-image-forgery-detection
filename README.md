# Scientific Image Forgery Detection (Pixel-Level Copy-Move Localization)

A PyTorch deep learning pipeline for detecting and localizing pixel-level copy-move forgeries in scientific publications, built on top of Meta AI's **DINOv2** Vision Transformer with custom intra-image patch self-similarity, partial backbone fine-tuning, and validation-tuned post-processing.

---

## 1. Problem Statement & Introduction

Scientific image forgery is a growing concern in the research community, leading to a large number of paper retractions across academic journals. One of the most common and damaging forms of scientific misconduct is **copy-move forgery**, where regions within a figure (such as Western blot bands, microscopy cells, or gel lanes) are duplicated to fabricate or alter experimental results.

This project addresses the **Recod.ai/LUC – Scientific Image Forgery Detection** challenge ([Kaggle Competition](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)), using a curated dataset derived from ~2,000 retracted scientific papers. The system is designed to solve two coupled tasks simultaneously:
1. **Image-Level Detection:** Flag whether a scientific figure is authentic or tampered.
2. **Pixel-Level Localization:** Predict a binary segmentation mask pinpointing the exact duplicated regions.

---

## 2. Dataset & Exploratory Data Analysis (EDA)

The dataset contains **5,128 scientific images** with a 1-to-1 match between forged images and ground-truth `.npy` masks:

| Category | Count | Share |
| :--- | :---: | :---: |
| Authentic Images (`label = 0`) | 2,377 | 46.4% |
| Forged Images (`label = 1`) | 2,751 | 53.6% |
| Ground-Truth Mask Files (`.npy`) | 2,751 | 100% of forged |

### Key Data Challenges Uncovered in EDA
* **Extreme Pixel-Level Class Imbalance:** Although images are nearly balanced at the class level, forged regions are very small. Across the 2,751 forged images, the **median forged area is only 2.71%** of the image (mean: `5.52%`, 25th percentile: `0.75%`, minimum: `0.04%`, maximum: `52.27%`). Standard pixel accuracy is therefore misleading, as predicting all-zeros still achieves >97% pixel accuracy.
* **Multi-Channel Masks:** Ground-truth masks are stored as 3D arrays of shape `(n_regions, H, W)`. Out of 2,751 forged images, **882 (~32%)** contain 2 to 5 separate forged layers (1 region: 1,869; 2 regions: 701; 3 regions: 174; 4 regions: 5; 5 regions: 2). All masks are collapsed into a single 2D binary mask via `.max(axis=0)` so no duplicated regions are lost.
* **High Resolution Variance:** Image dimensions vary widely across figures—from small `256×256` crops (605 images) to high-resolution `1000×666` (686 images) and `1600×1200` (530 images) plates. Aggressive downsampling blurs out sub-1% forgeries.

---

## 3. Leak-Free Dataset Splitting

Scientific paper datasets frequently contain authentic figures paired with their forged twins or near-identical crops. A standard random split would place one version in Train and its twin in Test, causing severe data leakage.

To prevent leakage:
1. Every image was fingerprinted using an **8×8 grayscale Average Hash (`aHash`)**, which revealed **582 near-duplicate visual groups spanning 1,278 images (~25% of the dataset)**.
2. The dataset was partitioned using **`StratifiedGroupKFold(n_splits=6, random_state=42)`**, grouping by `visual_hash` (locking near-duplicates into the same split) and stratifying jointly by `label` and `forged_ratio` quartiles.
3. Folds were mapped to **Train (4 folds) / Validation (1 fold) / Test (1 fold)** and exported to `dataset_split.csv`:

| Split | Authentic (`0`) | Forged (`1`) | Total |
| :--- | :---: | :---: | :---: |
| **Train** | 1,592 | 1,839 | **3,431** (66.9%) |
| **Validation** | 391 | 453 | **844** (16.5%) |
| **Test (Hold-Out)** | 394 | 459 | **853** (16.6%) |
| **Total** | **2,377** | **2,751** | **5,128** |

---

## 4. Evaluation Metrics

Because predicting all-zero masks inflates overall pixel metrics on authentic images, every model is evaluated using both **combined** and **class-separated** metrics:

* **Overall Dice:** Mean pixel-level Dice score across all test images (authentic images score `1.0` if predicted mask is completely empty, `0.0` if any false-positive pixel appears).
* **Forged Dice & Forged IoU:** Pixel-level Dice and Intersection-over-Union computed strictly on truly forged images (`label = 1`).
* **Auth Clean %:** Percentage of authentic images predicted with zero false-positive pixels.
* **Image-Level F1 & ROC-AUC:** Binary classification performance using the maximum pixel probability per image (`probs.max()`) for ROC-AUC and post-processed mask presence for F1.

---

## 5. Open-Source Model Used & Explicit Code Modifications

### Open-Source Baseline Model
We used Meta AI's pretrained Vision Transformer **DINOv2-Small (`dinov2_vits14`)**, loaded directly from the official PyTorch Hub repository (`facebookresearch/dinov2`).
* **Baseline Configuration (`01_Baseline_DINOv2_224_BCE`):**
  * Frozen `dinov2_vits14` backbone extracting `384`-dim patch tokens (`14×14` patch size).
  * Input resolution: `224×224` (`16×16` patch grid).
  * Lightweight 3-layer `Conv2d` + `ReLU` decoder (`384 → 128 → 64 → 1`) with bilinear upsampling.
  * Standard `BCEWithLogitsLoss` and `AdamW(lr=1e-3)` on decoder weights only.

### Summary of Modifications Made to the Open-Source Baseline

| Component | Open-Source Baseline | Our Modified Pipeline | Engineering Motivation |
| :--- | :--- | :--- | :--- |
| **Input Resolution** | `224×224` (`16×16` patches) | **`336×336` (`24×24` patches)** | 2.25× more spatial patch detail to preserve tiny (<1% area) forgeries. |
| **Architecture (Copy-Move Head)** | Plain `Conv2d` decoder on DINOv2 tokens (`384` ch) | **Intra-Image Patch Self-Similarity Module (`392` ch) + `BatchNorm2d`** | Computes L2-normalized pairwise cosine similarity (`torch.bmm`) across all patches, masks local neighbors (`dist ≤ 2.0`), and appends the **top-8 distant match scores** per patch so the decoder can directly detect remote duplicates. |
| **Loss Function** | Pure `BCEWithLogitsLoss` | **Combined `BCEDiceLoss` (50% BCE + 50% Soft Dice)** | Counteracts extreme pixel imbalance (<3% median forged area), preventing the model from collapsing to all-zero predictions. |
| **Backbone Training** | 100% Frozen DINOv2 | **Partial Fine-Tuning (Last 2 Transformer Blocks + `encoder.norm`)** | Adapts natural-image DINOv2 representations to scientific domains (Western blots, gels, microscopy) using **differential learning rates** (`1e-4` encoder, `1e-3` decoder) + `CosineAnnealingLR` + `weight_decay=1e-4`. |
| **Data Augmentation** | None | **On-the-fly Random Horizontal & Vertical Flips** | Applied synchronously to images and masks (`dims=[3]`, `dims=[2]`) to improve generalization. |
| **Checkpointing** | Best raw Validation Dice | **Lowest Validation Loss** | Fixed the "Epoch 1 Trap" observed in Mod #1, where timid early-epoch predictions artificially inflated raw Validation Dice on authentic images while discarding stronger later epochs. |
| **Post-Processing** | Raw `0.50` probability cutoff | **Validation-Tuned Threshold (`0.75`) + Minimum Area Filter (`500 px`)** | Removes stray false-positive specks on authentic images without touching the Hold-Out Test Set. |

---

## 6. Step-by-Step Pipeline & Experimental Results

All experiments were trained on Kaggle GPU (T4/P100) using the exact same leak-free Train/Val/Test splits (`dataset_split.csv`).

### Hold-Out Test Set Ablation Study (`N = 853`)

| # | Experiment | Overall Dice | Forged Dice | Forged IoU | Auth Clean % | Img F1 | Img AUC |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `01_Baseline_DINOv2_224_BCE` | 0.4844 | 0.1552 | 0.1085 | 86.80% | 0.4769 | 0.6020 |
| 1 | `02_SelfSim_336_BCEDice` *(raw Val Dice ckpt)* | 0.3502 | 0.2108 | 0.1468 | 51.27% | 0.5436 | 0.5772 |
| 2 | `03_FineTuned_SelfSim_336_Aug` | 0.5049 | **0.4372** | **0.3252** | 58.38% | **0.8111** | **0.8713** |
| 3 | **`04_PostProc_th0.75_px500` (Final)** | **0.5884** | 0.3527 | 0.2633 | **86.29%** | 0.7159 | **0.8713** |

### Key Takeaways from the Ablation Study
1. **Why the Baseline Failed:** The frozen baseline scored `0.4844` Overall Dice almost entirely by predicting empty masks (`86.80%` Auth Clean), while locating only **`15.52%`** of actual forged pixels (`Img AUC: 0.6020`).
2. **Mod #1 Diagnostic Finding ("Epoch 1 Trap"):** Adding self-similarity + `BCE+Dice` + `336×336` drove Validation AUC up to `0.7805` by Epoch 6, but checkpointing on un-tuned Validation Dice mistakenly saved Epoch 1 (`0.5948` AUC). Even so, `Forged_Dice` rose +36% (`0.1552 → 0.2108`).
3. **Mod #2 Breakthrough:** Fixing checkpointing (lowest Val Loss), unfreezing the last 2 DINOv2 blocks with differential LRs, and adding flip augmentation nearly **tripled `Forged_Dice` (`0.1552 → 0.4372`)** and boosted **`Img_AUC` from `0.6020 → 0.8713`** (`Img_F1: 0.8111`).
4. **Mod #3 Post-Processing & Zero-Leakage Proof:** Tuning `(threshold=0.75, min_pixels=500)` strictly on the Validation set achieved **`0.5908` Val Overall Dice** and **`0.5884` Test Overall Dice** (a gap of just `0.0024`, confirming zero data leakage). Authentic Clean % recovered from `58.38% → 86.29%`.
5. **Precision–Recall Trade-Off:** Filtering masks below `500 px` (~0.44% of a `336×336` image) removes false-positive noise on authentic images, at the cost of filtering out some genuine ultra-tiny (<0.5% area) forgeries (`Forged_Dice`: `0.4372 → 0.3527`).


## 7. Repository Structure

```text
├── notebooks/
│   └── scientific-image-forgery-detection.ipynb   # End-to-end executed Kaggle notebook (Sections 1–8)
├── results/
│   ├── dataset_split.csv                          # Leak-free aHash StratifiedGroupKFold split table
│   ├── ablation_results.csv                       # Final 4-row Test Set comparison table
│   └── error_analysis.png                         # Qualitative Test Set visual error analysis (Best Catch / Missed / False Alarm)
├── .gitignore                                     # Excludes raw images, .npy masks, and .pth weights
└── README.md

```


## 8. How to Reproduce

1. Open `notebooks/scientific-image-forgery-detection.ipynb` in Kaggle and attach the **Recod.ai/LUC – Scientific Image Forgery Detection** competition dataset.
2. Enable **Accelerator → GPU T4 x2** (or **P100**) and **Internet → On** (required to download pretrained `dinov2_vits14` weights via `torch.hub`).
3. Run Sections 1–8 sequentially. All random seeds (`random_state=42`) and split tables (`dataset_split.csv`) are fixed for reproducibility.
