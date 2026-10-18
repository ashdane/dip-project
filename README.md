# Adaptive RTV Illumination Estimation for Low-Light Image Enhancement

**Digital Image Processing (DIP) Project**  
**Team Members:**
- Daniel Ashish Abraham (`2025121010`)
- Mohammed Adeeb (`2025122005`)
- Arun K (`2025121004`)

---

## 📌 Project Overview

This project investigates and improves upon the low-light image enhancement pipeline proposed by:
> **Sun et al. (2022)** — *"Low-Illumination Image Enhancement Algorithm Based on Improved Multi-Scale Retinex and ABC Algorithm Optimization"*, Frontiers in Bioengineering and Biotechnology.

### Core Problems Identified in Baseline Paper
1. **Unstated & Uniform RTV Parameters ($\lambda, \sigma$)**: The paper utilizes Relative Total Variation (RTV) to separate structure from texture for illumination estimation, but uses fixed global smoothing across the entire image and does not disclose the parameters used.
2. **Evaluation Blind Spots**: The paper only evaluated no-reference metrics (SD, IE, AG), which naturally inflate when high-frequency noise is amplified in dark regions, masking structural degradation.

### Proposed Improvement
We introduce **Adaptive RTV Illumination Estimation**, allowing the smoothing parameter $\lambda(p)$ to vary spatially per pixel based on local illumination and texture characteristics, and evaluate against ground truth using full-reference metrics (PSNR, SSIM).

---

## 🏗️ Architecture

```
                       Input Low-Light Image
                                |
        +-----------------------+-----------------------+
        |                                               |
[Main Feature Layer]                           [Compensation Layer]
  1. Histogram Equalization                      1. RTV Illumination Estimation
  2. Weighted Guided Image Filtering (WGIF)      2. Multi-Scale Retinex (MSRCR)
                                                 3. Histogram Equalization
                                                 4. Bilateral Gamma Correction
                                                 5. Bilateral Filtering
        |                                               |
        +-----------------------+-----------------------+
                                |
                     [ABC Optimization Fusion]
                   Fuses layers using optimal
                      weights (w1, w2)
                                |
                         Enhanced Output
```

---

## 📊 Baseline Reproduction Results (LOL Dataset — eval15)

The baseline has been reproduced end-to-end and evaluated on the 15 evaluation images from the LOL dataset.

| Metric | Sun et al. (2022) Reported (10 images) | Our Uniform Baseline (15 images) | Assessment |
|:---|:---:|:---:|:---|
| **SD** (Standard Deviation) | ~65.7 | **64.91 ± 5.18** | Exact match |
| **IE** (Information Entropy) | ~7.79 | **7.42 ± 0.27** | Close match |
| **AG** (Average Gradient) | ~7.53 | **22.15 ± 9.29** | Higher (noise amplification) |
| **PSNR (dB)** | *Not reported* | **11.43 ± 3.48** | Ground-truth reference |
| **SSIM** | *Not reported* | **0.380 ± 0.14** | Ground-truth reference |

Per-image results are saved in [`results/metrics.csv`](results/metrics.csv).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Dependencies: `numpy`, `opencv-python`, `scipy`, `scikit-image`, `matplotlib`

```bash
pip install numpy opencv-python scipy scikit-image matplotlib
```

### Running the Baseline
1. Single Image:
   ```bash
   python baseline.py --input path/to/dark.png --output path/to/enhanced.png --reference path/to/high.png
   ```

2. LOL Dataset Evaluation:
   ```bash
   python baseline.py --lol-dir ./lol_dataset --output-dir ./results
   ```

---

## 🗺️ Project Milestones & Roadmap
- [x] **Milestone 1**: Complete from-scratch baseline implementation (RTV, MSRCR, WGIF, Bilateral Gamma, ABC Optimizer).
- [x] **Milestone 2**: Baseline benchmarking & metrics logging on LOL dataset.
- [ ] **Milestone 3**: Sensitivity analysis of RTV $\lambda$ and $\sigma$ to establish tuned baseline.
- [ ] **Milestone 4**: Implementation of Variant A (Illumination-adaptive $\lambda(p)$).
- [ ] **Milestone 5**: Implementation of Variant B (Texture-adaptive $\lambda(p)$).
- [ ] **Milestone 6**: Comparative evaluation (PSNR/SSIM/SD/IE/AG, Wilcoxon test, runtime, visual crops).
- [ ] **Milestone 7**: Final project report and code release.
