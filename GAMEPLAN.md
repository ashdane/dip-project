# Project Gameplan & Work Allocation

**Project:** Adaptive RTV Illumination Estimation for Low-Light Image Enhancement  
**Dataset:** LOL Dataset (`eval15`)  
**Repository:** [github.com/ashdane/dip-project](https://github.com/ashdane/dip-project.git)  

---

## 👥 Team Work Allocation Overview

| Team Member | Primary Focus Area | Key Deliverables |
|:---|:---|:---|
| **Ashish** (Daniel Ashish Abraham) | **Variant A: Illumination-Adaptive RTV** + Pipeline Architecture | Formulation of $\lambda_{\text{illum}}(p)$, sparse solver adaptation, module integration |
| **Adeeb** (Mohammed Adeeb) | **Variant B: Texture-Adaptive RTV** + Sensitivity Analysis | $\lambda$-$\sigma$ grid sweep script, RTV windowed variation reweighting $\lambda_{\text{tex}}(p)$, tuned baseline |
| **Arun** (Arun K) | **Benchmarking, Statistical Analysis & Visual Proofs** | Comparative testing script (PSNR, SSIM, SD, IE, AG), Wilcoxon signed-rank tests, zoomed visual crop comparisons, runtime profiling |
| **All Together** | **Final Report & Deliverables** | Joint report writing, results discussion, failure case analysis |

---

## 📋 Action Items by Member

### 👨‍💻 Ashish
- [ ] **Formulate Illumination-Adaptive $\lambda(p)$ (Variant A)**:
  - Extract normalized local luminance $\tilde{Y}(p)$ from the input image.
  - Map luminance to per-pixel smoothing weight:
    $$\lambda_A(p) = \lambda_{\min} + (\lambda_{\max} - \lambda_{\min}) \cdot (1 - \tilde{Y}(p))^\gamma$$
    *(Dark regions get stronger smoothing to suppress noise; bright regions get lower smoothing to preserve structure).*
- [ ] **Sparse Solver Integration**:
  - Update the RTV sparse linear system in `adaptive_rtv.py` to support per-pixel diagonal reweighting:
    $$(I_N + \text{diag}(\boldsymbol{\lambda}) \cdot L) \cdot S = I$$
- [ ] **Pipeline Orchestration**:
  - Integrate selectable RTV modes (`--rtv-mode uniform|illumination|texture`) into the unified runner.
- [ ] **Report Sections**:
  - Write Section 1 (Introduction & Problem Statement), Section 3 (Variant A Formulation), and Section 6 (Discussion & Conclusion).

---

### 👨‍💻 Adeeb
- [ ] **Sensitivity Analysis Script (`sensitivity_analysis.py`)**:
  - Sweep $\lambda \in \{0.001, 0.005, 0.015, 0.03, 0.05, 0.1\}$ and $\sigma \in \{1.0, 2.0, 3.0, 5.0\}$.
  - Run with fixed fusion weights $(w_1, w_2)$ to explore parameter sensitivity efficiently.
  - Find the optimal parameter pair $(\lambda^*, \sigma^*)$ = **Tuned Baseline**.
- [ ] **Formulate Texture-Adaptive $\lambda(p)$ (Variant B)**:
  - Extract RTV windowed variation ratio inside the solver:
    $$r(p) = \max\left(\frac{D_x(p)}{L_x(p) + \epsilon}, \; \frac{D_y(p)}{L_y(p) + \epsilon}\right)$$
  - Formulate texture-dependent attenuation:
    $$\lambda_B(p) = \frac{\lambda_{\text{base}}}{1 + \kappa \cdot r(p)}$$
    *(Smooth textured noise while preserving structural edges to avoid halos).*
- [ ] **Numerical Stability & Convergence**:
  - Verify that the IRLS matrix formulation remains symmetric positive-definite for SciPy sparse solvers.
- [ ] **Report Sections**:
  - Write Section 2 (Related Work & Baseline Details) and co-author Section 3 (Variant B Formulation) & Section 4 (Sensitivity Analysis).

---

### 👨‍💻 Arun
- [ ] **Automated Benchmark Harness**:
  - Run evaluation across all 15 LOL eval images for all 4 configurations:
    1. Paper Baseline ($\lambda = 0.015, \sigma = 3.0$)
    2. Tuned Baseline ($\lambda^*, \sigma^*$)
    3. Variant A (Illumination-Adaptive)
    4. Variant B (Texture-Adaptive)
- [ ] **Metrics Computation**:
  - **Full-reference**: PSNR, SSIM
  - **No-reference**: SD, IE, AG
  - **Efficiency**: Per-stage execution time (RTV vs. ABC)
- [ ] **Statistical Significance Testing**:
  - Perform paired two-tailed $t$-tests and Wilcoxon signed-rank tests ($p$-values) comparing each variant against the baseline.
- [ ] **Qualitative Comparison Figures**:
  - Generate high-resolution zoomed crops highlighting noise suppression in dark shadows and edge preservation (no halos).
- [ ] **Report Sections**:
  - Write Section 4 (Experimental Results & Significance Tables) and Section 5 (Qualitative Visual Analysis & Failure Modes).

---

## 📦 Final Deliverables Checklist
- [x] Reproduce Sun et al. (2022) baseline and log LOL eval15 metrics (`baseline.py`, `results/metrics.csv`).
- [ ] Parameter sensitivity analysis and tuned baseline identification (`sensitivity_analysis.py`).
- [ ] Adaptive RTV implementation with Variant A and Variant B (`adaptive_rtv.py`).
- [ ] Comparative benchmark results table and statistical test outputs (`results/comparison_metrics.csv`).
- [ ] Visual crop figures demonstrating noise and halo handling.
- [ ] Final written report and presentation.
