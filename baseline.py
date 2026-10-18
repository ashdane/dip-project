"""
Baseline implementation of Sun et al. (2022):
"Low-Illumination Image Enhancement Algorithm Based on Improved
 Multi-Scale Retinex and ABC Algorithm Optimization"

Pipeline overview (see Figure 2 of the paper):
  Input image
      |
      +---> Main Feature Layer:  HE  --> WGIF
      |
      +---> Compensation Layer:  RTV illumination --> MSRCR --> HE
      |                          --> Bilateral Gamma Correction --> Bilateral Filter
      |
      +--> ABC-optimised weighted fusion --> Enhanced output

All core algorithms are implemented from scratch in Python.
Libraries used: NumPy, OpenCV (I/O, colour conversion), SciPy (sparse solver),
scikit-image (PSNR/SSIM), Matplotlib (visualisation).
"""

import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import time
import argparse
import matplotlib.pyplot as plt


# ============================================================================
# 1. Histogram Equalization (HE) — per-channel on BGR image
# ============================================================================
def histogram_equalization(img):
    """Apply histogram equalization independently to each channel.

    Parameters
    ----------
    img : ndarray, uint8, shape (H, W, 3)
        Input BGR image.

    Returns
    -------
    ndarray, uint8, shape (H, W, 3)
    """
    result = np.empty_like(img)
    for c in range(3):
        result[:, :, c] = cv2.equalizeHist(img[:, :, c])
    return result


# ============================================================================
# 2. Weighted Guided Image Filtering (WGIF)
#    Reference: Li et al. (2015) "Weighted Guided Image Filtering"
# ============================================================================
def weighted_guided_image_filter(I, p, radius=8, eps=0.01, gamma_reg=None):
    """Weighted Guided Image Filtering (WGIF).

    Parameters
    ----------
    I : ndarray, float64, shape (H, W)
        Guidance image (single-channel, [0,1]).
    p : ndarray, float64, shape (H, W)
        Input image to be filtered (single-channel, [0,1]).
    radius : int
        Window radius for the box filter.
    eps : float
        Regularisation parameter.
    gamma_reg : float or None
        Regularisation for the weighting factor Gamma_G.
        If None, uses (0.001 * 1.0)^2 = 1e-6 as in the paper (L=1 for [0,1]).

    Returns
    -------
    ndarray, float64, shape (H, W)
    """
    if gamma_reg is None:
        L = 1.0  # dynamic range for [0,1] images
        gamma_reg = (0.001 * L) ** 2

    ksize = 2 * radius + 1
    H, W = I.shape

    # Box-filter means
    mean_I = cv2.boxFilter(I, -1, (ksize, ksize), normalize=True)
    mean_p = cv2.boxFilter(p, -1, (ksize, ksize), normalize=True)
    mean_Ip = cv2.boxFilter(I * p, -1, (ksize, ksize), normalize=True)
    mean_II = cv2.boxFilter(I * I, -1, (ksize, ksize), normalize=True)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # Compute the local variance in a 3x3 window for the weighting factor Gamma_G
    small_ksize = 3
    mean_I_small = cv2.boxFilter(I, -1, (small_ksize, small_ksize), normalize=True)
    mean_II_small = cv2.boxFilter(I * I, -1, (small_ksize, small_ksize), normalize=True)
    var_I_small = mean_II_small - mean_I_small * mean_I_small

    # Gamma_G(i) = (1/N) * sum over all I' { (var_small(I') + gamma) / (var_small(I) + gamma) }
    # In practice: Gamma_G = mean(var_small + gamma) / (var_small + gamma)
    global_mean_var = np.mean(var_I_small + gamma_reg)
    Gamma_G = global_mean_var / (var_I_small + gamma_reg)

    # a_k = cov_Ip / (var_I + eps / Gamma_G)
    a = cov_Ip / (var_I + eps / Gamma_G)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (ksize, ksize), normalize=True)
    mean_b = cv2.boxFilter(b, -1, (ksize, ksize), normalize=True)

    q = mean_a * I + mean_b
    return q


def apply_wgif_color(img_float, radius=8, eps=0.01):
    """Apply WGIF to a colour image using the grayscale as the guide.

    Parameters
    ----------
    img_float : ndarray, float64, shape (H, W, 3), [0,1] BGR
    radius : int
    eps : float

    Returns
    -------
    ndarray, float64, shape (H, W, 3), [0,1]
    """
    guide = cv2.cvtColor((img_float * 255).astype(np.uint8),
                         cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    result = np.empty_like(img_float)
    for c in range(3):
        result[:, :, c] = weighted_guided_image_filter(guide, img_float[:, :, c],
                                                       radius=radius, eps=eps)
    return np.clip(result, 0, 1)


# ============================================================================
# 3. Structure Extraction from Texture via Relative Total Variation (RTV)
#    Reference: Xu et al. (2012) "Structure Extraction from Texture via
#    Relative Total Variation"
# ============================================================================
def rtv_smoothing(img, lambda_rtv=0.015, sigma=3.0, max_iter=4, eps=1e-4):
    """Structure extraction from texture via relative total variation.

    The objective (Eq. 17 in Sun et al.):
        S = argmin_S  sum_p (S_p - I_p)^2
            + lambda * [ D_x(p) / (L_x(p) + eps)  +  D_y(p) / (L_y(p) + eps) ]

    where D_x, D_y are windowed total-variation terms and L_x, L_y are the
    windowed directional sums (inherent variation).

    Parameters
    ----------
    img : ndarray, float64, shape (H, W), [0,1]
        Grayscale input image.
    lambda_rtv : float
        Smoothing weight (controls smoothness).
    sigma : float
        Scale of the Gaussian window for texture suppression.
    max_iter : int
        Number of iterations of the IRLS solver.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    S : ndarray, float64, shape (H, W)
        Smoothed (structure) image.
    """
    H, W = img.shape
    N = H * W
    S = img.copy()

    # Gaussian kernel size (must be odd)
    ksize = int(np.ceil(sigma * 3)) * 2 + 1

    for iteration in range(max_iter):
        # Compute gradients of S
        dx_S = np.diff(S, axis=1, append=S[:, -1:])   # dS/dx  (H, W)
        dy_S = np.diff(S, axis=0, append=S[-1:, :])   # dS/dy  (H, W)

        # Windowed variation: D_x, D_y  (sum of |grad| in window)
        abs_dx = np.abs(dx_S)
        abs_dy = np.abs(dy_S)
        Dx = cv2.GaussianBlur(abs_dx, (ksize, ksize), sigma) + eps
        Dy = cv2.GaussianBlur(abs_dy, (ksize, ksize), sigma) + eps

        # Windowed inherent variation: L_x, L_y  (|sum of grad in window|)
        Lx = np.abs(cv2.GaussianBlur(dx_S, (ksize, ksize), sigma)) + eps
        Ly = np.abs(cv2.GaussianBlur(dy_S, (ksize, ksize), sigma)) + eps

        # Weights for x and y: w = Dx / (Lx + eps)  etc.
        # These are the RTV weights — high where texture, low where structure
        wx = Dx / Lx  # (H, W)
        wy = Dy / Ly  # (H, W)

        # Build the sparse linear system:
        #   (I + lambda * (D_x^T W_x D_x + D_y^T W_y D_y)) S = I
        # where D_x, D_y are forward-difference operators and W_x, W_y are
        # diagonal weight matrices.

        # Flatten for sparse matrix construction
        wx_flat = wx.ravel()
        wy_flat = wy.ravel()

        # Forward difference in x: pixel (i,j) -> (i,j+1)
        # Index mapping: idx = i*W + j
        # The sparse Laplacian contribution from x-direction:
        #   For each pixel p at (i,j), if j < W-1:
        #     contribution: wx[p] * (S[p] - S[p+1])
        # This gives a tridiagonal-like structure.

        # Indices for x-direction (horizontal neighbours)
        idx = np.arange(N).reshape(H, W)
        # Pixels that have a right neighbour
        mask_x = np.ones((H, W), dtype=bool)
        mask_x[:, -1] = False
        idx_p_x = idx[mask_x].ravel()
        idx_q_x = idx_p_x + 1
        w_x_vals = wx_flat[idx_p_x]

        # Pixels that have a bottom neighbour
        mask_y = np.ones((H, W), dtype=bool)
        mask_y[-1, :] = False
        idx_p_y = idx[mask_y].ravel()
        idx_q_y = idx_p_y + W
        w_y_vals = wy_flat[idx_p_y]

        # Build the Laplacian matrix L = D^T W D
        # For each edge (p,q) with weight w:
        #   L[p,p] += w, L[q,q] += w, L[p,q] -= w, L[q,p] -= w
        row = np.concatenate([idx_p_x, idx_q_x, idx_p_x, idx_q_x,
                              idx_p_y, idx_q_y, idx_p_y, idx_q_y])
        col = np.concatenate([idx_p_x, idx_q_x, idx_q_x, idx_p_x,
                              idx_p_y, idx_q_y, idx_q_y, idx_p_y])
        data = np.concatenate([w_x_vals, w_x_vals, -w_x_vals, -w_x_vals,
                               w_y_vals, w_y_vals, -w_y_vals, -w_y_vals])

        L = sparse.csr_matrix((data, (row, col)), shape=(N, N))

        # System: (I + lambda * L) S = I_input
        A = sparse.eye(N, format='csr') + lambda_rtv * L
        b = img.ravel()

        S = spsolve(A, b).reshape(H, W)
        S = np.clip(S, 0, 1)

    return S


def rtv_illumination(img_bgr, lambda_rtv=0.015, sigma=3.0, max_iter=4):
    """Estimate illumination from a BGR image using RTV on the V channel.

    Parameters
    ----------
    img_bgr : ndarray, float64, (H,W,3), [0,1]
    lambda_rtv : float
    sigma : float
    max_iter : int

    Returns
    -------
    illumination : ndarray, float64, (H,W), [0,1]
    """
    hsv = cv2.cvtColor((img_bgr * 255).astype(np.uint8),
                       cv2.COLOR_BGR2HSV).astype(np.float64) / 255.0
    V = hsv[:, :, 2]
    illumination = rtv_smoothing(V, lambda_rtv=lambda_rtv, sigma=sigma,
                                 max_iter=max_iter)
    return illumination


# ============================================================================
# 4. Multi-Scale Retinex with Color Restoration (MSRCR)
#    Reference: Rahman et al. (2004)
# ============================================================================
def single_scale_retinex(img, sigma):
    """Single-scale Retinex: log(image) - log(Gaussian_blur(image)).

    Parameters
    ----------
    img : ndarray, float64, (H, W), values > 0
    sigma : float
        Gaussian surround scale.

    Returns
    -------
    ndarray, float64, (H, W)
    """
    ksize = int(np.ceil(sigma * 3)) * 2 + 1
    blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    blur = np.maximum(blur, 1e-6)
    return np.log(img + 1e-6) - np.log(blur)


def multi_scale_retinex(img, scales=(15, 80, 250)):
    """Multi-scale Retinex (MSR) as an equally weighted sum of SSR at multiple
    scales.

    Parameters
    ----------
    img : ndarray, float64, (H, W), values > 0
    scales : tuple of float
        Gaussian surround scale parameters.

    Returns
    -------
    ndarray, float64, (H, W)
    """
    msr = np.zeros_like(img)
    for s in scales:
        msr += single_scale_retinex(img, s)
    msr /= len(scales)
    return msr


def msrcr(img_bgr, illumination=None, scales=(15, 80, 250), alpha=125.0, beta=46.0,
          gain=1.0, offset=0.0):
    """Multi-Scale Retinex with Color Restoration (MSRCR).

    Uses an RTV-estimated illumination instead of Gaussian surround if provided.

    The color restoration factor for channel c is:
        C_c = beta * log(alpha * I_c / sum_channels(I))

    The MSRCR output is:
        MSRCR_c = C_c * MSR_c

    Finally, the result is scaled to [0, 255].

    Parameters
    ----------
    img_bgr : ndarray, float64, (H,W,3), [0,1]
    illumination : ndarray, float64, (H,W), [0,1] or None
        If provided, used as the surround estimate for Retinex.
    scales : tuple
        Gaussian scales for MSR.
    alpha : float
        Color restoration parameter.
    beta : float
        Color restoration gain.
    gain : float
        Final output gain.
    offset : float
        Final output offset.

    Returns
    -------
    ndarray, uint8, (H,W,3)
    """
    img = img_bgr.copy()
    img = np.maximum(img, 1e-6)

    # Compute MSR for each channel
    msr = np.zeros_like(img)
    for c in range(3):
        if illumination is not None:
            # Use RTV illumination: log(reflectance) = log(I) - log(L)
            illum = np.maximum(illumination, 1e-6)
            msr[:, :, c] = np.log(img[:, :, c]) - np.log(illum)
        else:
            msr[:, :, c] = multi_scale_retinex(img[:, :, c], scales)

    # Color restoration
    channel_sum = np.sum(img, axis=2) + 1e-6
    result = np.zeros_like(img)
    for c in range(3):
        color_restoration = beta * np.log(alpha * img[:, :, c] / channel_sum + 1e-6)
        result[:, :, c] = color_restoration * msr[:, :, c]

    # Normalize to [0, 255]
    result = gain * result + offset
    for c in range(3):
        ch = result[:, :, c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        if hi - lo > 0:
            result[:, :, c] = 255.0 * (ch - lo) / (hi - lo)
        else:
            result[:, :, c] = 0

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# ============================================================================
# 5. Bilateral Gamma Correction (Improved)
#    Reference: Wang et al. (2017, 2021) — improved bilateral gamma function
# ============================================================================
def bilateral_gamma_correction(img_bgr):
    """Improved bilateral gamma correction.

    The bilateral gamma function adaptively corrects luminance by using
    different gamma values for dark and bright regions.

    For a pixel I(x,y) with overall mean brightness mu:
        gamma(x,y) = 1 - I(x,y) / (2*mu)  (if I(x,y) <= 128)
    The adjustment parameter alpha selects the correction branch:
        alpha = 1 if I(x,y) <= 128, else 0

    O(x,y) = I(x,y)^gamma(x,y) * alpha + I(x,y) * (1 - alpha)

    Parameters
    ----------
    img_bgr : ndarray, uint8, (H,W,3)

    Returns
    -------
    ndarray, uint8, (H,W,3)
    """
    img_float = img_bgr.astype(np.float64)
    # Compute luminance (mean across channels)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu = np.mean(gray)
    mu = max(mu, 1e-6)  # avoid division by zero

    result = np.empty_like(img_float)

    for c in range(3):
        channel = img_float[:, :, c]
        normalised = channel / 255.0

        # Gamma for each pixel based on luminance
        gamma = 1.0 - gray / (2.0 * mu)
        gamma = np.clip(gamma, 0.1, 3.0)  # safety clamp

        # alpha = 1 where dark, 0 where bright
        alpha_mask = (gray <= 128).astype(np.float64)

        # Bilateral correction
        corrected = np.power(np.maximum(normalised, 1e-6), gamma)
        out = corrected * alpha_mask + normalised * (1.0 - alpha_mask)

        result[:, :, c] = out * 255.0

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# ============================================================================
# 6. Bilateral Filter
# ============================================================================
def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter for edge-preserving smoothing.

    Parameters
    ----------
    img : ndarray, uint8, (H,W,3)
    d : int
        Diameter of each pixel neighbourhood.
    sigma_color : float
    sigma_space : float

    Returns
    -------
    ndarray, uint8, (H,W,3)
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# ============================================================================
# 7. No-reference quality metrics: SD, IE, AG
#    (Used in the paper's fitness function for ABC)
# ============================================================================
def standard_deviation(img):
    """Standard deviation of pixel values (Eq. 25 in paper).

    Parameters
    ----------
    img : ndarray, uint8, (H,W,3)

    Returns
    -------
    float
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return np.std(gray)


def information_entropy(img):
    """Information entropy of the image (Eq. 27 in paper).

    IE = - sum P(x) * log2(P(x))

    Parameters
    ----------
    img : ndarray, uint8, (H,W,3)

    Returns
    -------
    float
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def average_gradient(img):
    """Average gradient of the image (Eq. 26 in paper).

    AG = sum sqrt( (df/dx)^2 + (df/dy)^2 ) / 2 ) / (W*H)

    Parameters
    ----------
    img : ndarray, uint8, (H,W,3)

    Returns
    -------
    float
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    H, W = gray.shape
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    ag = np.sum(np.sqrt((gx ** 2 + gy ** 2) / 2.0)) / (H * W)
    return ag


def fitness_function(img):
    """Fitness = (SD + IE + AG) / 3   (Eq. 28 in paper).

    Parameters
    ----------
    img : ndarray, uint8, (H,W,3)

    Returns
    -------
    float
    """
    sd = standard_deviation(img)
    ie = information_entropy(img)
    ag = average_gradient(img)
    return (sd + ie + ag) / 3.0


# ============================================================================
# 8. Artificial Bee Colony (ABC) Algorithm
#    Reference: Karaboga & Akay (2009)
# ============================================================================
def fuse_images(main_layer, comp_layer, w1, w2):
    """Fuse two layers with weights.

    Parameters
    ----------
    main_layer : ndarray, float64, (H,W,3), [0,1]
    comp_layer : ndarray, float64, (H,W,3), [0,1]
    w1, w2 : float
        Fusion weights.

    Returns
    -------
    ndarray, uint8, (H,W,3)
    """
    fused = w1 * main_layer + w2 * comp_layer
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)
    return fused


def abc_optimise(main_layer, comp_layer, n_pop=45, max_iter=100,
                 abandon_limit=90, w1_bounds=(0.1, 1.0), w2_bounds=(0.2, 1.0),
                 verbose=False):
    """Artificial Bee Colony algorithm to find optimal fusion weights.

    The ABC algorithm searches for (w1, w2) that maximise the fitness function
    (Eq. 28) of the fused image.

    Parameters
    ----------
    main_layer : ndarray, float64, (H,W,3), [0,1]
    comp_layer : ndarray, float64, (H,W,3), [0,1]
    n_pop : int
        Population size (number of food sources).
    max_iter : int
        Maximum number of iterations.
    abandon_limit : int
        If a food source hasn't improved for this many iterations, it's abandoned.
    w1_bounds, w2_bounds : tuple of (float, float)
        Search bounds for w1 and w2.
    verbose : bool

    Returns
    -------
    best_w1, best_w2 : float
        Optimal fusion weights.
    best_fitness : float
        Best fitness value achieved.
    """
    n_dim = 2
    lower = np.array([w1_bounds[0], w2_bounds[0]])
    upper = np.array([w1_bounds[1], w2_bounds[1]])

    # Initialise food sources
    population = np.random.uniform(lower, upper, size=(n_pop, n_dim))
    fitness_vals = np.zeros(n_pop)
    trial_counts = np.zeros(n_pop, dtype=int)

    # Evaluate initial population
    for i in range(n_pop):
        fused = fuse_images(main_layer, comp_layer,
                            population[i, 0], population[i, 1])
        fitness_vals[i] = fitness_function(fused)

    best_idx = np.argmax(fitness_vals)
    best_solution = population[best_idx].copy()
    best_fitness = fitness_vals[best_idx]

    for iteration in range(max_iter):
        # ---- Employed bee phase ----
        for i in range(n_pop):
            # Pick a random dimension and a random partner
            d = np.random.randint(n_dim)
            j = i
            while j == i:
                j = np.random.randint(n_pop)

            # Generate candidate (Eq. 30: phi * (x_id - x_jd))
            candidate = population[i].copy()
            phi = np.random.uniform(-1, 1)
            candidate[d] = population[i, d] + phi * (population[i, d] - population[j, d])
            candidate = np.clip(candidate, lower, upper)

            fused = fuse_images(main_layer, comp_layer, candidate[0], candidate[1])
            candidate_fitness = fitness_function(fused)

            if candidate_fitness > fitness_vals[i]:
                population[i] = candidate
                fitness_vals[i] = candidate_fitness
                trial_counts[i] = 0
            else:
                trial_counts[i] += 1

        # ---- Onlooker bee phase ----
        # Probabilities proportional to fitness (Eq. 31)
        probs = fitness_vals / (fitness_vals.sum() + 1e-10)

        for _ in range(n_pop):
            i = np.random.choice(n_pop, p=probs)

            d = np.random.randint(n_dim)
            j = i
            while j == i:
                j = np.random.randint(n_pop)

            candidate = population[i].copy()
            phi = np.random.uniform(-1, 1)
            candidate[d] = population[i, d] + phi * (population[i, d] - population[j, d])
            candidate = np.clip(candidate, lower, upper)

            fused = fuse_images(main_layer, comp_layer, candidate[0], candidate[1])
            candidate_fitness = fitness_function(fused)

            if candidate_fitness > fitness_vals[i]:
                population[i] = candidate
                fitness_vals[i] = candidate_fitness
                trial_counts[i] = 0
            else:
                trial_counts[i] += 1

        # ---- Scout bee phase (Eq. 32) ----
        for i in range(n_pop):
            if trial_counts[i] >= abandon_limit:
                population[i] = np.random.uniform(lower, upper)
                fused = fuse_images(main_layer, comp_layer,
                                    population[i, 0], population[i, 1])
                fitness_vals[i] = fitness_function(fused)
                trial_counts[i] = 0

        # Track global best
        current_best_idx = np.argmax(fitness_vals)
        if fitness_vals[current_best_idx] > best_fitness:
            best_fitness = fitness_vals[current_best_idx]
            best_solution = population[current_best_idx].copy()

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  ABC iter {iteration+1}/{max_iter}: "
                  f"best_fitness = {best_fitness:.4f}, "
                  f"w1 = {best_solution[0]:.4f}, w2 = {best_solution[1]:.4f}")

    return best_solution[0], best_solution[1], best_fitness


# ============================================================================
# 9. Main Feature Layer Pipeline
# ============================================================================
def main_feature_layer(img_bgr, wgif_radius=8, wgif_eps=0.01):
    """Process the main feature layer: HE -> WGIF.

    Parameters
    ----------
    img_bgr : ndarray, uint8, (H,W,3)

    Returns
    -------
    result : ndarray, float64, (H,W,3), [0,1]
    intermediates : dict of ndarray (for debugging/visualisation)
    """
    intermediates = {}

    # Step 1: Histogram Equalization
    he_img = histogram_equalization(img_bgr)
    intermediates['HE'] = he_img

    # Step 2: WGIF edge-preserving filtering
    he_float = he_img.astype(np.float64) / 255.0
    wgif_img = apply_wgif_color(he_float, radius=wgif_radius, eps=wgif_eps)
    intermediates['WGIF'] = (wgif_img * 255).astype(np.uint8)

    return wgif_img, intermediates


# ============================================================================
# 10. Compensation Layer Pipeline
# ============================================================================
def compensation_layer(img_bgr, lambda_rtv=0.015, sigma_rtv=3.0, rtv_iter=4,
                       msr_scales=(15, 80, 250), alpha_cr=125.0, beta_cr=46.0,
                       bilateral_d=9, bilateral_sc=75, bilateral_ss=75):
    """Process the compensation layer:
    RTV illumination -> MSRCR -> HE -> Bilateral Gamma Correction -> Bilateral Filter.

    Parameters
    ----------
    img_bgr : ndarray, uint8, (H,W,3)

    Returns
    -------
    result : ndarray, float64, (H,W,3), [0,1]
    intermediates : dict of ndarray
    """
    intermediates = {}
    img_float = img_bgr.astype(np.float64) / 255.0

    # Step 1: RTV illumination estimation
    print("  [Compensation] Computing RTV illumination...")
    illumination = rtv_illumination(img_float, lambda_rtv=lambda_rtv,
                                    sigma=sigma_rtv, max_iter=rtv_iter)
    intermediates['RTV_illumination'] = (illumination * 255).astype(np.uint8)

    # Step 2: MSRCR using RTV illumination
    print("  [Compensation] Computing MSRCR...")
    msrcr_img = msrcr(img_float, illumination=illumination,
                      scales=msr_scales, alpha=alpha_cr, beta=beta_cr)
    intermediates['MSRCR'] = msrcr_img

    # Step 3: Histogram Equalization
    he_img = histogram_equalization(msrcr_img)
    intermediates['MSRCR_HE'] = he_img

    # Step 4: Bilateral Gamma Correction
    print("  [Compensation] Applying bilateral gamma correction...")
    gamma_img = bilateral_gamma_correction(he_img)
    intermediates['Gamma'] = gamma_img

    # Step 5: Bilateral Filter
    bf_img = bilateral_filter(gamma_img, d=bilateral_d,
                              sigma_color=bilateral_sc, sigma_space=bilateral_ss)
    intermediates['BilateralFilter'] = bf_img

    result = bf_img.astype(np.float64) / 255.0
    return result, intermediates


# ============================================================================
# 11. Full Baseline Pipeline
# ============================================================================
def enhance_image(img_bgr, params=None, verbose=True):
    """Run the full Sun et al. (2022) enhancement pipeline.

    Parameters
    ----------
    img_bgr : ndarray, uint8, (H,W,3)
        Input low-light BGR image.
    params : dict or None
        Override default parameters. Keys:
        - 'lambda_rtv', 'sigma_rtv', 'rtv_iter'
        - 'msr_scales', 'alpha_cr', 'beta_cr'
        - 'wgif_radius', 'wgif_eps'
        - 'bilateral_d', 'bilateral_sc', 'bilateral_ss'
        - 'abc_n_pop', 'abc_max_iter', 'abc_abandon_limit'
    verbose : bool

    Returns
    -------
    enhanced : ndarray, uint8, (H,W,3)
    info : dict
        Contains intermediates, weights, fitness, and timing info.
    """
    if params is None:
        params = {}

    # Default parameters (from the paper where stated, otherwise our choices)
    p = {
        # RTV
        'lambda_rtv': params.get('lambda_rtv', 0.015),
        'sigma_rtv': params.get('sigma_rtv', 3.0),
        'rtv_iter': params.get('rtv_iter', 4),
        # MSRCR (from paper: scales 15, 80, 250; alpha=125, beta=46)
        'msr_scales': params.get('msr_scales', (15, 80, 250)),
        'alpha_cr': params.get('alpha_cr', 125.0),
        'beta_cr': params.get('beta_cr', 46.0),
        # WGIF
        'wgif_radius': params.get('wgif_radius', 8),
        'wgif_eps': params.get('wgif_eps', 0.01),
        # Bilateral filter
        'bilateral_d': params.get('bilateral_d', 9),
        'bilateral_sc': params.get('bilateral_sc', 75),
        'bilateral_ss': params.get('bilateral_ss', 75),
        # ABC
        'abc_n_pop': params.get('abc_n_pop', 45),
        'abc_max_iter': params.get('abc_max_iter', 100),
        'abc_abandon_limit': params.get('abc_abandon_limit', 90),
    }

    info = {'params': p, 'intermediates': {}}
    t_start = time.time()

    # ---- Main Feature Layer ----
    if verbose:
        print("[Main Feature Layer] HE -> WGIF")
    t0 = time.time()
    main_layer, main_inters = main_feature_layer(
        img_bgr, wgif_radius=p['wgif_radius'], wgif_eps=p['wgif_eps'])
    info['intermediates']['main'] = main_inters
    info['time_main'] = time.time() - t0
    if verbose:
        print(f"  Main feature layer done in {info['time_main']:.2f}s")

    # ---- Compensation Layer ----
    if verbose:
        print("[Compensation Layer] RTV -> MSRCR -> HE -> Gamma -> BF")
    t0 = time.time()
    comp_layer, comp_inters = compensation_layer(
        img_bgr,
        lambda_rtv=p['lambda_rtv'], sigma_rtv=p['sigma_rtv'], rtv_iter=p['rtv_iter'],
        msr_scales=p['msr_scales'], alpha_cr=p['alpha_cr'], beta_cr=p['beta_cr'],
        bilateral_d=p['bilateral_d'], bilateral_sc=p['bilateral_sc'],
        bilateral_ss=p['bilateral_ss'])
    info['intermediates']['comp'] = comp_inters
    info['time_comp'] = time.time() - t0
    if verbose:
        print(f"  Compensation layer done in {info['time_comp']:.2f}s")

    # ---- ABC Fusion ----
    if verbose:
        print(f"[ABC Fusion] pop={p['abc_n_pop']}, iter={p['abc_max_iter']}, "
              f"limit={p['abc_abandon_limit']}")
    t0 = time.time()
    best_w1, best_w2, best_fit = abc_optimise(
        main_layer, comp_layer,
        n_pop=p['abc_n_pop'], max_iter=p['abc_max_iter'],
        abandon_limit=p['abc_abandon_limit'], verbose=verbose)
    info['time_abc'] = time.time() - t0
    info['w1'] = best_w1
    info['w2'] = best_w2
    info['best_fitness'] = best_fit
    if verbose:
        print(f"  ABC done in {info['time_abc']:.2f}s")
        print(f"  Optimal weights: w1={best_w1:.4f}, w2={best_w2:.4f}, "
              f"fitness={best_fit:.4f}")

    # ---- Final fusion ----
    enhanced = fuse_images(main_layer, comp_layer, best_w1, best_w2)
    info['time_total'] = time.time() - t_start

    if verbose:
        print(f"[Done] Total time: {info['time_total']:.2f}s")

    return enhanced, info


# ============================================================================
# 12. Evaluation Metrics
# ============================================================================
def evaluate(enhanced, reference=None):
    """Compute quality metrics.

    Parameters
    ----------
    enhanced : ndarray, uint8, (H,W,3)
    reference : ndarray, uint8, (H,W,3) or None
        If provided, full-reference metrics (PSNR, SSIM) are also computed.

    Returns
    -------
    metrics : dict
    """
    metrics = {}

    # No-reference metrics (from the paper)
    metrics['SD'] = standard_deviation(enhanced)
    metrics['IE'] = information_entropy(enhanced)
    metrics['AG'] = average_gradient(enhanced)
    metrics['Fitness'] = fitness_function(enhanced)

    # Full-reference metrics
    if reference is not None:
        # Ensure same size
        if enhanced.shape != reference.shape:
            enhanced = cv2.resize(enhanced, (reference.shape[1], reference.shape[0]))
        metrics['PSNR'] = psnr(reference, enhanced)
        metrics['SSIM'] = ssim(reference, enhanced, channel_axis=2)

    return metrics


# ============================================================================
# 13. Visualisation helpers
# ============================================================================
def save_intermediates(info, output_dir):
    """Save intermediate images for debugging.

    Parameters
    ----------
    info : dict
        As returned by enhance_image.
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)

    for layer_name, layer_dict in info.get('intermediates', {}).items():
        for step_name, img in layer_dict.items():
            fname = f"{layer_name}_{step_name}.png"
            cv2.imwrite(os.path.join(output_dir, fname), img)


def create_comparison_figure(input_img, enhanced_img, reference_img=None,
                             metrics=None, save_path=None):
    """Create a side-by-side comparison figure.

    Parameters
    ----------
    input_img : ndarray, uint8, (H,W,3) BGR
    enhanced_img : ndarray, uint8, (H,W,3) BGR
    reference_img : ndarray, uint8, (H,W,3) BGR or None
    metrics : dict or None
    save_path : str or None
    """
    n_cols = 3 if reference_img is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    axes[0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Input (Low-light)')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    title = 'Enhanced'
    if metrics:
        title += f"\nSD={metrics.get('SD',0):.2f}  IE={metrics.get('IE',0):.2f}  AG={metrics.get('AG',0):.2f}"
        if 'PSNR' in metrics:
            title += f"\nPSNR={metrics['PSNR']:.2f}  SSIM={metrics['SSIM']:.4f}"
    axes[1].set_title(title, fontsize=9)
    axes[1].axis('off')

    if reference_img is not None:
        axes[2].imshow(cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Reference (Ground Truth)')
        axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Comparison figure saved to {save_path}")
    plt.close()


# ============================================================================
# 14. LOL Dataset Runner
# ============================================================================
def run_on_lol_dataset(lol_dir, output_dir, max_images=None, params=None):
    """Run the baseline on the LOL dataset and compute metrics.

    Expects the LOL dataset structure:
        lol_dir/
            our485/low/    (training low-light images)
            our485/high/   (training reference images)
            eval15/low/    (evaluation low-light images)
            eval15/high/   (evaluation reference images)

    Parameters
    ----------
    lol_dir : str
        Root directory of the LOL dataset.
    output_dir : str
        Where to save enhanced images and results.
    max_images : int or None
        Limit the number of images processed (for quick testing).
    params : dict or None
        Pipeline parameter overrides.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Try to find the eval set first, then fall back to training
    eval_low = os.path.join(lol_dir, 'eval15', 'low')
    eval_high = os.path.join(lol_dir, 'eval15', 'high')

    if not os.path.isdir(eval_low):
        # Try alternate structure
        eval_low = os.path.join(lol_dir, 'low')
        eval_high = os.path.join(lol_dir, 'high')

    if not os.path.isdir(eval_low):
        print(f"ERROR: Cannot find low-light images directory.")
        print(f"  Tried: {os.path.join(lol_dir, 'eval15', 'low')}")
        print(f"  Tried: {os.path.join(lol_dir, 'low')}")
        return

    low_files = sorted([f for f in os.listdir(eval_low)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    if max_images:
        low_files = low_files[:max_images]

    print(f"Processing {len(low_files)} images from {eval_low}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    all_metrics = []

    for idx, fname in enumerate(low_files):
        print(f"\n[{idx+1}/{len(low_files)}] {fname}")
        print("-" * 40)

        # Load input
        input_path = os.path.join(eval_low, fname)
        input_img = cv2.imread(input_path)
        if input_img is None:
            print(f"  WARNING: Could not read {input_path}, skipping.")
            continue

        # Load reference if available
        ref_path = os.path.join(eval_high, fname)
        ref_img = cv2.imread(ref_path) if os.path.exists(ref_path) else None

        # Enhance
        enhanced, info = enhance_image(input_img, params=params, verbose=True)

        # Evaluate
        metrics = evaluate(enhanced, reference=ref_img)
        metrics['filename'] = fname
        metrics['w1'] = info['w1']
        metrics['w2'] = info['w2']
        metrics['time_total'] = info['time_total']
        all_metrics.append(metrics)

        # Print metrics
        print(f"  Metrics: SD={metrics['SD']:.2f}, IE={metrics['IE']:.2f}, "
              f"AG={metrics['AG']:.2f}, Fitness={metrics['Fitness']:.2f}")
        if 'PSNR' in metrics:
            print(f"  PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}")
        print(f"  Weights: w1={metrics['w1']:.4f}, w2={metrics['w2']:.4f}")
        print(f"  Time: {metrics['time_total']:.2f}s")

        # Save enhanced image
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, enhanced)

        # Save comparison figure
        fig_path = os.path.join(output_dir, f"compare_{os.path.splitext(fname)[0]}.png")
        create_comparison_figure(input_img, enhanced, ref_img, metrics, fig_path)

        # Save intermediates for first image
        if idx == 0:
            save_intermediates(info, os.path.join(output_dir, 'intermediates'))

    # ---- Summary ----
    if all_metrics:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        metric_keys = ['SD', 'IE', 'AG', 'Fitness']
        if 'PSNR' in all_metrics[0]:
            metric_keys += ['PSNR', 'SSIM']
        metric_keys += ['time_total']

        for key in metric_keys:
            vals = [m[key] for m in all_metrics]
            print(f"  {key:>12s}: mean={np.mean(vals):.4f}, "
                  f"std={np.std(vals):.4f}, "
                  f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

        # Save metrics to CSV
        csv_path = os.path.join(output_dir, 'metrics.csv')
        with open(csv_path, 'w') as f:
            header = ['filename'] + metric_keys + ['w1', 'w2']
            f.write(','.join(header) + '\n')
            for m in all_metrics:
                row = [m['filename']] + [f"{m[k]:.6f}" for k in metric_keys] + \
                      [f"{m['w1']:.4f}", f"{m['w2']:.4f}"]
                f.write(','.join(row) + '\n')
        print(f"\n  Metrics saved to {csv_path}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Sun et al. (2022) Low-Illumination Image Enhancement Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enhance a single image:
  python baseline.py --input dark_photo.png --output enhanced.png

  # Run on the LOL dataset (eval15 subset):
  python baseline.py --lol-dir ./LOL --output-dir ./results

  # Quick test with 3 images:
  python baseline.py --lol-dir ./LOL --output-dir ./results --max-images 3
        """)

    parser.add_argument('--input', type=str, help='Path to a single input image')
    parser.add_argument('--output', type=str, default='enhanced.png',
                        help='Output path for single image mode')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference image (for PSNR/SSIM)')
    parser.add_argument('--lol-dir', type=str, default=None,
                        help='Path to LOL dataset root directory')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for LOL dataset mode')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Limit number of images to process')

    # Pipeline parameters
    parser.add_argument('--lambda-rtv', type=float, default=0.015,
                        help='RTV smoothing weight (default: 0.015)')
    parser.add_argument('--sigma-rtv', type=float, default=3.0,
                        help='RTV texture suppression scale (default: 3.0)')
    parser.add_argument('--rtv-iter', type=int, default=4,
                        help='RTV iterations (default: 4)')
    parser.add_argument('--abc-pop', type=int, default=45,
                        help='ABC population size (default: 45)')
    parser.add_argument('--abc-iter', type=int, default=100,
                        help='ABC max iterations (default: 100)')
    parser.add_argument('--abc-limit', type=int, default=90,
                        help='ABC abandon limit (default: 90)')

    args = parser.parse_args()

    params = {
        'lambda_rtv': args.lambda_rtv,
        'sigma_rtv': args.sigma_rtv,
        'rtv_iter': args.rtv_iter,
        'abc_n_pop': args.abc_pop,
        'abc_max_iter': args.abc_iter,
        'abc_abandon_limit': args.abc_limit,
    }

    if args.lol_dir:
        # Dataset mode
        run_on_lol_dataset(args.lol_dir, args.output_dir,
                           max_images=args.max_images, params=params)
    elif args.input:
        # Single image mode
        print(f"Loading {args.input}")
        img = cv2.imread(args.input)
        if img is None:
            print(f"ERROR: Could not read {args.input}")
            return

        ref_img = None
        if args.reference:
            ref_img = cv2.imread(args.reference)

        enhanced, info = enhance_image(img, params=params, verbose=True)

        metrics = evaluate(enhanced, reference=ref_img)
        print(f"\nMetrics: SD={metrics['SD']:.2f}, IE={metrics['IE']:.2f}, "
              f"AG={metrics['AG']:.2f}")
        if 'PSNR' in metrics:
            print(f"PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}")

        cv2.imwrite(args.output, enhanced)
        print(f"Enhanced image saved to {args.output}")

        # Save comparison
        fig_path = os.path.splitext(args.output)[0] + '_comparison.png'
        create_comparison_figure(img, enhanced, ref_img, metrics, fig_path)

        # Save intermediates
        inter_dir = os.path.splitext(args.output)[0] + '_intermediates'
        save_intermediates(info, inter_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
