# Detailed Project Gameplan & Work Allocation

**Project:** Adaptive RTV Illumination Estimation for Low-Light Image Enhancement  
**Dataset:** LOL Dataset (`eval15`)  
**Repository:** [github.com/ashdane/dip-project](https://github.com/ashdane/dip-project.git)  

---

## 👥 Team Work Allocation Overview

| Team Member | Primary Focus Area | Key Deliverables |
|:---|:---|:---|
| **Ashish** (Daniel Ashish Abraham) | **Variant A: Illumination-Adaptive RTV** + Pipeline Integration | Mathematical formulation of $\lambda_{\text{illum}}(p)$, sparse solver integration in `baseline.py` / `adaptive_rtv.py`, overall code architecture |
| **Adeeb** (Mohammed Adeeb) | **Variant B: Texture-Adaptive RTV** + Sensitivity Analysis | RTV windowed variation reweighting for $\lambda_{\text{tex}}(p)$, $\lambda$-$\sigma$ grid sweep script, tuned baseline identification |
| **Arun** (Arun K) | **Benchmarking, Statistical Analysis & Visual Proofs** | Comparative testing script (PSNR, SSIM, SD, IE, AG), Wilcoxon signed-rank tests, zoomed visual crop comparisons, runtime profiling |
| **All Together** | **Final Report & Presentation** | Joint report writing, results discussion, failure case analysis |

---

## 🗓️ Detailed Phase-by-Phase Gameplan

### Phase 1: Establish Tuned Baseline & Sensitivity Analysis (Days 1–3)
**Owner: Adeeb** (Assisted by Arun)

* **Objective:** The paper hid $\lambda$ and $\sigma$. We need to systematically explore the parameter space to establish the optimal fixed-parameter baseline (Deliverable 3).
* **Tasks for Adeeb:**
  1. Create `sensitivity_analysis.py`:
     - Grid sweep over $\lambda \in \{0.001, 0.005, 0.015, 0.03, 0.05, 0.1\}$
     - Grid sweep over $\sigma \in \{1.0, 2.0, 3.0, 5.0\}$
     - Fast evaluation mode: run with fixed weights $(w_1, w_2)$ from baseline to save computation, or subsample LOL images to determine sensitivity trends quickly.
  2. Plot sensitivity curves: $\lambda$ vs. PSNR/SSIM, $\sigma$ vs. AG/SD.
  3. Identify the best fixed pair $(\lambda^*, \sigma^*)$ = **"Tuned Baseline"**.
* **Deliverable:** `results/sensitivity/` heatmaps + `metrics_sensitivity.csv`.

---

### Phase 2: Implement Variant A — Illumination-Adaptive RTV (Days 3–6)
**Owner: Ashish**

* **Objective:** Low-SNR dark regions suffer from severe sensor noise amplification. $\lambda(p)$ should increase in dark regions (stronger smoothing) and decrease in well-lit regions (preserve fine detail).
* **Tasks for Ashish:**
  1. Compute local normalized luminance from the input image $I$:
     $$\tilde{Y}(p) = \frac{Y(p) - Y_{\min}}{Y_{\max} - Y_{\min}}$$
     (Optionally smoothed with a lightweight spatial Gaussian to prevent sharp jumps).
  2. Formulate spatially-varying $\lambda_A(p)$:
     $$\lambda_A(p) = \lambda_{\min} + (\lambda_{\max} - \lambda_{\min}) \cdot (1 - \tilde{Y}(p))^\gamma$$
     where $\gamma \ge 1$ controls transition steepness, $\lambda_{\max}$ suppresses dark noise, and $\lambda_{\min}$ preserves bright structure.
  3. Modify the sparse solver:
     - In standard RTV, the linear system solves $(I_N + \lambda L) \cdot S = I$.
     - With per-pixel $\lambda(p)$, the Laplacian weight matrix scales row $p$ by $\lambda(p)$:
       $$(I_N + \text{diag}(\boldsymbol{\lambda}) \cdot L) \cdot S = I$$
  4. Create `adaptive_rtv.py` supporting `--mode illumination`.
* **Deliverable:** Working Variant A module + verification on sample images (`1.png`, `111.png`, `146.png`).

---

### Phase 3: Implement Variant B — Texture-Adaptive RTV (Days 4–7)
**Owner: Adeeb** (Assisted by Ashish)

* **Objective:** Textures and noise should be smoothed out from the illumination layer, while prominent structural edges must be strictly preserved to avoid halos.
* **Tasks for Adeeb:**
  1. Leverage the windowed variation ratio inside RTV:
     $$r_x(p) = \frac{D_x(p)}{L_x(p) + \epsilon}, \quad r_y(p) = \frac{D_y(p)}{L_y(p) + \epsilon}$$
     where $D(p)$ is total variation and $L(p)$ is inherent variation.
  2. In high-texture / noisy regions, $r(p) \approx 1$. Near strong structural edges, $r(p) \gg 1$.
  3. Formulate texture-adaptive mapping:
     $$\lambda_B(p) = \lambda_0 \cdot \exp\left(-\frac{\|\nabla I(p)\|^2}{2\sigma_e^2}\right) \quad \text{or} \quad \lambda_B(p) = \frac{\lambda_{\text{base}}}{1 + \kappa \cdot \max(r_x(p), r_y(p))}$$
  4. Ensure the IRLS solver maintains matrix symmetry or positive definiteness for fast SciPy `spsolve` convergence.
* **Deliverable:** Working Variant B module in `adaptive_rtv.py` supporting `--mode texture`.

---

### Phase 4: Comparative Benchmarking & Statistical Evaluation (Days 7–9)
**Owner: Arun**

* **Objective:** Rigorously prove whether Variant A and Variant B outperform the uniform baseline (Addresses Issue 2 of proposal).
* **Tasks for Arun:**
  1. Build automated batch evaluation harness:
     - Run 4 configurations on all 15 LOL eval images:
       1. Baseline (paper default: $\lambda=0.015, \sigma=3.0$)
       2. Tuned Baseline ($\lambda^*, \sigma^*$)
       3. Variant A (Illumination-Adaptive)
       4. Variant B (Texture-Adaptive)
  2. Compute Full Evaluation Metrics:
     - **Full-reference:** PSNR, SSIM, and LOE (Lightness Order Error).
     - **No-reference:** SD, IE, AG.
     - **Runtime cost:** Execution time per stage (RTV solver vs. ABC optimizer).
  3. Statistical Significance Testing:
     - Compute paired two-tailed $t$-test and Wilcoxon signed-rank test ($p$-values) comparing each variant against the baseline.
  4. High-Resolution Visual Inspection:
     - Generate zoomed comparison crops (bounding box patches) showcasing:
       - Suppression of dark noise in background shadows.
       - Elimination of halo artifacts along high-contrast boundaries.
* **Deliverable:** Master comparison table (`metrics_comparison.csv`), significance test report, and visual patch figures (`figure_crops.png`).

---

### Phase 5: Final Report & Documentation (Days 9–11)
**Owner: All (Lead: Ashish, Sections by Adeeb & Arun)**

* **Section Breakdown:**
  - **Section 1: Introduction & Problem Statement** — *Ashish*
  - **Section 2: Related Work & Baseline Reproduction Details** — *Adeeb* (documenting unstated parameters)
  - **Section 3: Proposed Adaptive RTV Formulations (Variant A & B)** — *Ashish & Adeeb*
  - **Section 4: Experimental Results & Sensitivity Analysis** — *Adeeb & Arun* (Tables, Boxplots, Wilcoxon tests)
  - **Section 5: Qualitative Visual Analysis & Failure Modes** — *Arun*
  - **Section 6: Discussion, Computational Complexity & Conclusion** — *Ashish*
