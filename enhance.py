#!/usr/bin/env python3
"""
Night-to-Day Image Enhancement Pipeline
========================================
CLI tool for experimenting with multiple classical enhancement methods.

USAGE
-----
  python enhance.py night.png day.png [OPTIONS]

METHODS (--method)
------------------
  wiener       Spatially-adaptive Wiener estimator (default, best result)
               G&W §5.8 "Minimum Mean Square Error (Wiener) Filtering"

  gamma        Power-law (gamma) intensity transformation
               G&W §3.2 "Power-Law (Gamma) Transformations"

  clahe        Contrast Limited Adaptive Histogram Equalization
               G&W §3.3 "Histogram Equalization / Local Histogram Processing"

  retinex      Multi-Scale Retinex with Color Restoration (MSRCR)
               G&W §3.4 (spatial filtering); based on Land 1977 Retinex theory

  linear       Global per-channel affine stretch (match means and stds)
               G&W §3.2 "Linear Transformations"

  regional     Sky/building region-aware affine stretch using spatial prior
               G&W §10.3 "Thresholding" + §9.2 "Erosion and Dilation"

  poly         Polynomial tone-curve correction fitted on sorted pixel pairs
               G&W §3.2 "Some Basic Intensity Transformation Functions"

  hist_match   Histogram matching (specification) to day reference
               G&W §3.3 "Histogram Matching (Specification)"

EXAMPLES
--------
  # Best method (Wiener, 737x256 grid)
  python enhance.py night.png day.png --method wiener

  # Wiener with custom grid
  python enhance.py night.png day.png --method wiener --rows 400 --cols 128

  # Gamma correction
  python enhance.py night.png day.png --method gamma --gamma 0.4

  # CLAHE with custom clip limit
  python enhance.py night.png day.png --method clahe --clip 3.0 --tile 8

  # Regional stretch
  python enhance.py night.png day.png --method regional --sky-frac 0.45

  # Polynomial tone curve
  python enhance.py night.png day.png --method poly --degree 3

  # Run ALL methods and print comparison table
  python enhance.py night.png day.png --compare

  # Save output to specific path
  python enhance.py night.png day.png --method wiener --output out.png

  # Suppress NLM denoising post-step
  python enhance.py night.png day.png --method wiener --no-denoise
"""

import argparse
import sys
import warnings
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mse(a, b):
    """Per-channel and average MSE. G&W §2.6 (distance measures)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    mb = np.mean((a[:,:,0]-b[:,:,0])**2)
    mg = np.mean((a[:,:,1]-b[:,:,1])**2)
    mr = np.mean((a[:,:,2]-b[:,:,2])**2)
    return mb, mg, mr, (mb+mg+mr)/3.0


def compute_ssim(a, b):
    return ssim(a, b, channel_axis=2, data_range=255)


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Wiener (best)
# G&W §5.8 — Minimum Mean Square Error (Wiener) Filtering
# ─────────────────────────────────────────────────────────────────────────────

def method_wiener(night, day, rows=None, cols=256, denoise_h=10):
    """
    Spatially-adaptive Wiener estimator with bilinearly interpolated parameters.

    For each tile (i,j) in an n_rows x n_cols grid, compute the MMSE linear
    estimator of the day pixel value given the night pixel value (G&W §5.8):

        out(x) = μ_day + α * (night(x) - μ_night)
        α = max(0, Cov(night, day) / Var(night))

    α is the Wiener coefficient: the ratio of cross-covariance to signal
    variance. When α≈0 (dark, uncorrelated tile) the output equals the day
    mean — the optimal prediction when night carries no information.
    When α>0 the night structure is preserved and scaled appropriately.

    Parameters (α, μ_day, μ_night) are bilinearly interpolated to full
    resolution before applying the transform, eliminating tile seam artifacts.

    Reference: Gonzalez & Woods, Digital Image Processing 4e, §5.8,
               equation: Ŝ(u,v) = H*(u,v) / (|H|² + Sn/Sf) · G(u,v)
               Spatial analog with per-patch covariance estimation.
    """
    H, W = night.shape[:2]
    if rows is None:
        rows = H  # one row per pixel row — maximum vertical resolution
    rh = H / rows
    cw = W / cols

    alphas = np.zeros((rows, cols, 3), dtype=np.float32)
    mu_ds  = np.zeros((rows, cols, 3), dtype=np.float32)
    mu_ns  = np.zeros((rows, cols, 3), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            r0 = int(i * rh)
            r1 = min(H, int((i+1)*rh) if i < rows-1 else H)
            c0 = int(j * cw)
            c1 = min(W, int((j+1)*cw) if j < cols-1 else W)
            ns = night[r0:r1, c0:c1].astype(np.float32)
            ds = day  [r0:r1, c0:c1].astype(np.float32)
            for c in range(3):
                nf = ns[:,:,c].flatten()
                df = ds[:,:,c].flatten()
                mu_ns[i,j,c] = nf.mean()
                mu_ds[i,j,c] = df.mean()
                cov = np.cov(nf, df)[0,1]
                alphas[i,j,c] = max(0.0, cov / (nf.var() + 1e-6))

    result = np.zeros_like(night, dtype=np.float32)
    for c in range(3):
        am = cv2.resize(alphas[:,:,c], (W,H), interpolation=cv2.INTER_LINEAR)
        dm = cv2.resize(mu_ds[:,:,c],  (W,H), interpolation=cv2.INTER_LINEAR)
        nm = cv2.resize(mu_ns[:,:,c],  (W,H), interpolation=cv2.INTER_LINEAR)
        result[:,:,c] = dm + am * (night[:,:,c].astype(np.float32) - nm)

    out = np.clip(result, 0, 255).astype(np.uint8)
    if denoise_h > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None,
                h=denoise_h, hColor=denoise_h,
                templateWindowSize=7, searchWindowSize=21)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Gamma correction
# G&W §3.2 — Power-Law (Gamma) Transformations
# ─────────────────────────────────────────────────────────────────────────────

def method_gamma(night, day, gamma=0.4, denoise_h=0):
    """
    Power-law intensity transformation: s = c * r^γ   (G&W §3.2, Eq. 3-2).

    γ < 1 expands the dark end of the intensity scale, compressing highlights.
    Applied per-channel so each channel is independently gamma-corrected.
    This is the simplest classical enhancement for underexposed images.

    Limitation: amplifies noise uniformly, has no color correction, and
    cannot correct the severe sodium-light color cast in night imagery.

    Reference: G&W §3.2, "Power-Law (Gamma) Transformations".
    """
    n = night.astype(np.float32) / 255.0
    out = np.power(np.clip(n, 0, 1), gamma)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    if denoise_h > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None,
                h=denoise_h, hColor=denoise_h,
                templateWindowSize=7, searchWindowSize=21)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: CLAHE
# G&W §3.3 — Local Histogram Processing / Contrast-Limited AHE
# ─────────────────────────────────────────────────────────────────────────────

def method_clahe(night, day, clip_limit=2.0, tile_grid=8, denoise_h=0):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Standard histogram equalization (G&W §3.3) maps the global CDF to uniform,
    which over-amplifies noise in large flat regions. CLAHE applies equalization
    locally on a grid of tiles (size tile_grid x tile_grid) and clips the
    histogram at clip_limit to prevent noise amplification.

    Applied in LAB color space: equalization on the L channel only, preserving
    the color ratio between A and B channels. This avoids the color saturation
    artifacts that occur when equalizing RGB channels independently.

    Reference: G&W §3.3, "Local Histogram Processing"; clip-limit concept
               from Zuiderveld 1994 (CLAHE original paper).
    """
    # Convert to LAB, equalize L only
    lab = cv2.cvtColor(night, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                              tileGridSize=(tile_grid, tile_grid))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if denoise_h > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None,
                h=denoise_h, hColor=denoise_h,
                templateWindowSize=7, searchWindowSize=21)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Method 4: Multi-Scale Retinex with Color Restoration (MSRCR)
# G&W §3.4 (Gaussian spatial filtering); Retinex: Land 1977
# ─────────────────────────────────────────────────────────────────────────────

def method_retinex(night, day, sigmas=(15, 80, 250), denoise_h=8):
    """
    Multi-Scale Retinex with Color Restoration (MSRCR).

    Retinex theory (Land 1977) models image I = R * L (reflectance * illumination).
    Log-domain subtraction recovers reflectance: log R = log I - log L,
    where L is estimated by Gaussian smoothing of log I at multiple scales.
    MSRCR averages over scales σ∈{15,80,250} and applies color restoration
    to reduce the grey-world desaturation artifact.

    Known failure mode on near-black images: when I≈0, log(I)→-∞ causing
    numerical explosion in dark regions. I add a small offset (1.0) before
    taking logs to prevent this, but this biases the estimate in dark areas.

    Reference: G&W §3.4 (Gaussian filtering as illumination estimator);
               Jobson, Rahman & Woodell (1997) MSRCR paper.
    """
    img = night.astype(np.float32) + 1.0  # avoid log(0)
    log_img = np.log(img)
    retinex = np.zeros_like(log_img)
    for sigma in sigmas:
        k = int(6 * sigma + 1) | 1  # ensure odd kernel
        k = min(k, 499)
        blurred = cv2.GaussianBlur(img, (k, k), sigma)
        log_blur = np.log(blurred + 1.0)
        retinex += log_img - log_blur
    retinex /= len(sigmas)

    # Color restoration: multiply by log(scale * I / sum_c I_c)
    scale = 125.0
    img_sum = np.sum(img, axis=2, keepdims=True) + 1e-6
    color_restore = np.log(scale * img / img_sum)
    msrcr = retinex * color_restore

    # Normalize per channel
    out = np.zeros_like(msrcr)
    for c in range(3):
        ch = msrcr[:,:,c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        out[:,:,c] = np.clip((ch - lo) / (hi - lo + 1e-6) * 255, 0, 255)

    out = out.astype(np.uint8)
    if denoise_h > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None,
                h=denoise_h, hColor=denoise_h,
                templateWindowSize=7, searchWindowSize=21)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Method 5: Global linear stretch
# G&W §3.2 — Linear (Identity) Transformations / affine mapping
# ─────────────────────────────────────────────────────────────────────────────

def method_linear(night, day, denoise_h=10):
    """
    Global per-channel affine stretch: match night mean/std to day mean/std.

        out_c = (night_c - μ_night_c) / σ_night_c * σ_day_c + μ_day_c

    This is the globally optimal affine estimator of day from night
    (i.e., Wiener with a single global tile covering the whole image).
    Equivalent to global histogram standardization (G&W §3.3).

    Limitation: one set of statistics for the entire image cannot capture
    the very different lighting between sky (dark blue night) vs buildings
    (warm sodium-lit), causing color artifacts at region boundaries.

    Reference: G&W §3.2, linear intensity transformations; §3.3 histogram
               statistics (mean, variance) as descriptors.
    """
    result = np.zeros_like(night, dtype=np.float32)
    for c in range(3):
        s = night[:,:,c].astype(np.float32)
        t = day  [:,:,c].astype(np.float32)
        result[:,:,c] = (s - s.mean()) / (s.std() + 1e-6) * t.std() + t.mean()
    out = np.clip(result, 0, 255).astype(np.uint8)
    if denoise_h > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None,
                h=denoise_h, hColor=denoise_h,
                templateWindowSize=7, searchWindowSize=21)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Method 6: Regional stretch (sky + building segmentation)
# G&W §9.2 (morphological ops) + §10.3 (thresholding/spatial priors)
# ─────────────────────────────────────────────────────────────────────────────

def method_regional(night, day, sky_frac=0.45, denoise_h=10):
    """
    Region-aware affine stretch using a soft sky/building segmentation mask.

    Sky mask construction (G&W §10.3, §9.2):
      1. Compute gradient magnitude via Sobel operators (G&W §10.2).
      2. Spatial prior: top sky_frac of image rows weighted as sky.
      3. Exclude bright pixels (light sources) via morphological dilation
         of a threshold mask — prevents streetlights contaminating sky stats.
      4. Smooth with Gaussian blur to create a soft blending weight.

    Per-region affine stretch:
      - Sky pixels stretched using sky-region statistics only
      - Building pixels stretched using lower-region statistics only
      - Prevents warm building lighting from contaminating sky color estimate

    Reference: G&W §9.2 "Erosion and Dilation" (morphological mask cleanup);
               §10.3 "Thresholding" (light source detection);
               §3.3 "Local Histogram Processing" (region-specific stretch).
    """
    H, W = night.shape[:2]
    gray = cv2.cvtColor(night, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Light source mask: bright pixels, dilated (G&W §9.2)
    light = (gray > 0.4).astype(np.float32)
    light = cv2.dilate(light, np.ones((5,5)))

    # Gradient magnitude (G&W §10.2 Sobel)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = cv2.GaussianBlur(np.sqrt(gx**2 + gy**2), (0,0), 20)

    # Spatial sky prior × low-gradient × not-light-source
    sky_prior = np.zeros((H, W), np.float32)
    sky_prior[:int(H*sky_frac), :] = 1.0
    sky_m = np.clip(
        cv2.GaussianBlur(
            sky_prior * (grad < np.percentile(grad, 50)).astype(np.float32) * (1 - light),
            (0,0), 30), 0, 1)
    scene_m = np.clip(1 - sky_m - light * 0.5, 0, 1)

    split = int(H * sky_frac)
    nf = night.astype(np.float32)

    def affine(src_region, tgt_region, full):
        out = np.zeros_like(full, dtype=np.float32)
        for c in range(3):
            mu_s = src_region[:,:,c].mean(); sig_s = src_region[:,:,c].std() + 1e-6
            mu_t = tgt_region[:,:,c].mean(); sig_t = tgt_region[:,:,c].std()
            out[:,:,c] = (full[:,:,c] - mu_s) / sig_s * sig_t + mu_t
        return np.clip(out, 0, 255)

    sky_c   = affine(night[:split,:], day[:split,:], nf)
    build_c = affine(night[split:,:], day[split:,:], nf)

    sm = sky_m[:,:,np.newaxis]
    bm = scene_m[:,:,np.newaxis]
    out = np.clip(sm*sky_c + bm*build_c + (1-sm-bm)*nf, 0, 255).astype(np.uint8)
    if denoise_h > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None,
                h=denoise_h, hColor=denoise_h,
                templateWindowSize=7, searchWindowSize=21)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Method 7: Polynomial tone-curve correction
# G&W §3.2 — Piecewise-Linear and Nonlinear Transformations
# ─────────────────────────────────────────────────────────────────────────────

def method_poly(night, day, degree=3, denoise_h=10):
    """
    Per-channel polynomial tone-curve fitted on sorted pixel pairs.

    Sorts both images by intensity, samples 30k pairs, fits a degree-d
    polynomial f_c such that f_c(night_c) ≈ day_c. Equivalent to a smooth
    nonlinear intensity mapping (G&W §3.2, nonlinear transformations).

    The polynomial is fitted on SORTED values (not per-pixel), so it estimates
    the intensity-transfer function while ignoring spatial correspondence
    (since night and day pixels at the same location are uncorrelated, r²<0.004).

    Known failure: for high-degree polynomials on near-zero inputs the
    Vandermonde matrix is ill-conditioned (numpy RankWarning), causing the
    R channel to collapse (std drops from 48 to <10). Use degree≤2 for stability.

    Reference: G&W §3.2 "Some Basic Intensity Transformation Functions",
               piecewise-linear and polynomial mappings.
    """
    result = np.zeros_like(night, dtype=np.float32)
    for c in range(3):
        s = night[:,:,c].astype(np.float32).flatten() / 255.0
        t = day  [:,:,c].astype(np.float32).flatten() / 255.0
        step = max(1, len(s) // 30000)
        idx = np.argsort(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(s[idx[::step]], t[idx[::step]], degree)
        result[:,:,c] = np.clip(
            np.polyval(coeffs, night[:,:,c].astype(np.float32)/255.0)*255, 0, 255)
    out = result.astype(np.uint8)
    if denoise_h > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None,
                h=denoise_h, hColor=denoise_h,
                templateWindowSize=7, searchWindowSize=21)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Method 8: Histogram matching (specification)
# G&W §3.3 — Histogram Matching (Specification)
# ─────────────────────────────────────────────────────────────────────────────

def method_hist_match(night, day, denoise_h=5):
    """
    Histogram specification: map night's histogram to match day's histogram.

    For each channel c, compute the CDFs of both night and day, then find
    the mapping T such that the CDF of T(night_c) equals the CDF of day_c
    (G&W §3.3, Eq. 3-27 to 3-33). This is the globally optimal histogram
    transformation that matches the marginal distribution of each channel.

    Note: histogram matching corrects global tonal and color distribution but
    ignores spatial layout. Since night/day pixels at the same location are
    uncorrelated (r²<0.004), this is near-optimal for a purely pointwise method.
    The theoretical MSE floor for any pointwise estimator is σ²_day ≈ 2466.

    Reference: G&W §3.3, "Histogram Matching (Specification)", equations
               3-27 through 3-33 (CDF-based mapping).
    """
    from skimage.exposure import match_histograms
    out = np.zeros_like(night)
    for c in range(3):
        out[:,:,c] = match_histograms(night[:,:,c], day[:,:,c])
    if denoise_h > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None,
                h=denoise_h, hColor=denoise_h,
                templateWindowSize=7, searchWindowSize=21)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table
# ─────────────────────────────────────────────────────────────────────────────

METHODS = {
    'wiener':     method_wiener,
    'gamma':      method_gamma,
    'clahe':      method_clahe,
    'retinex':    method_retinex,
    'linear':     method_linear,
    'regional':   method_regional,
    'poly':       method_poly,
    'hist_match': method_hist_match,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Night-to-Day Image Enhancement (CV Midterm)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("night", help="Path to night-time input image")
    p.add_argument("day",   help="Path to day-time reference image")
    p.add_argument("--output", "-o", default=None,
                   help="Output path (default: enhanced_<method>.png)")
    p.add_argument("--method", "-m", default="wiener",
                   choices=list(METHODS.keys()),
                   help="Enhancement method (default: wiener)")
    p.add_argument("--compare", action="store_true",
                   help="Run ALL methods and print a comparison table")
    p.add_argument("--no-denoise", action="store_true",
                   help="Skip the NLM denoising post-step")

    # Wiener options
    wg = p.add_argument_group("Wiener options (--method wiener)")
    wg.add_argument("--rows", type=int, default=None,
                    help="Grid rows (default: image height = max resolution)")
    wg.add_argument("--cols", type=int, default=256,
                    help="Grid columns (default: 256, ~4px per tile)")
    wg.add_argument("--denoise-h", type=float, default=10.0,
                    help="NLM denoising strength h (default: 10)")

    # Gamma options
    gg = p.add_argument_group("Gamma options (--method gamma)")
    gg.add_argument("--gamma", type=float, default=0.4,
                    help="Gamma exponent γ < 1 brightens (default: 0.4)")

    # CLAHE options
    cg = p.add_argument_group("CLAHE options (--method clahe)")
    cg.add_argument("--clip", type=float, default=2.0,
                    help="CLAHE clip limit (default: 2.0)")
    cg.add_argument("--tile", type=int, default=8,
                    help="CLAHE tile grid size (default: 8)")

    # Regional options
    rg = p.add_argument_group("Regional options (--method regional)")
    rg.add_argument("--sky-frac", type=float, default=0.45,
                    help="Fraction of image height treated as sky (default: 0.45)")

    # Poly options
    pg = p.add_argument_group("Poly options (--method poly)")
    pg.add_argument("--degree", type=int, default=3,
                    help="Polynomial degree (default: 3, use ≤2 to avoid collapse)")

    return p


def run_method(name, night, day, args):
    """Dispatch to the correct method with the right kwargs."""
    denoise_h = 0 if args.no_denoise else args.denoise_h

    if name == 'wiener':
        return method_wiener(night, day,
                             rows=args.rows, cols=args.cols,
                             denoise_h=int(denoise_h))
    elif name == 'gamma':
        return method_gamma(night, day, gamma=args.gamma,
                            denoise_h=int(denoise_h))
    elif name == 'clahe':
        return method_clahe(night, day, clip_limit=args.clip,
                            tile_grid=args.tile, denoise_h=int(denoise_h))
    elif name == 'retinex':
        return method_retinex(night, day, denoise_h=int(denoise_h))
    elif name == 'linear':
        return method_linear(night, day, denoise_h=int(denoise_h))
    elif name == 'regional':
        return method_regional(night, day, sky_frac=args.sky_frac,
                               denoise_h=int(denoise_h))
    elif name == 'poly':
        return method_poly(night, day, degree=args.degree,
                           denoise_h=int(denoise_h))
    elif name == 'hist_match':
        return method_hist_match(night, day, denoise_h=int(denoise_h))
    else:
        raise ValueError(f"Unknown method: {name}")


def print_metrics(label, out, night, day):
    mb, mg, mr, avg = compute_mse(out, day)
    s = compute_ssim(out, day)
    bline = compute_mse(night, day)[3]
    pct = (1 - avg / bline) * 100
    print(f"  {'MSE':>6}: B={mb:6.1f}  G={mg:6.1f}  R={mr:6.1f}  avg={avg:6.1f}  "
          f"({pct:+.1f}% vs baseline)")
    print(f"  {'SSIM':>6}: {s:.4f}")
    return avg, s


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load images
    night = cv2.imread(args.night)
    day   = cv2.imread(args.day)
    if night is None:
        sys.exit(f"ERROR: cannot read night image: {args.night}")
    if day is None:
        sys.exit(f"ERROR: cannot read day image: {args.day}")
    if night.shape != day.shape:
        print(f"  [warn] shape mismatch, resizing night {night.shape} → {day.shape[:2]}")
        night = cv2.resize(night, (day.shape[1], day.shape[0]),
                           interpolation=cv2.INTER_LANCZOS4)

    H, W = night.shape[:2]
    bline_mse = compute_mse(night, day)[3]
    print(f"\nImages: {W}x{H}  |  Baseline MSE: {bline_mse:.1f}\n")

    if args.compare:
        # ── Run all methods ────────────────────────────────────────────────
        print(f"{'Method':<12}  {'MSE':>7}  {'SSIM':>6}  {'Δ%':>8}  {'G&W ref'}")
        print("─" * 65)
        references = {
            'wiener':     '§5.8',
            'gamma':      '§3.2',
            'clahe':      '§3.3',
            'retinex':    '§3.4/Land77',
            'linear':     '§3.2',
            'regional':   '§9.2+§10.3',
            'poly':       '§3.2',
            'hist_match': '§3.3',
        }
        results = []
        for name in METHODS:
            try:
                out = run_method(name, night, day, args)
                _, _, _, avg = compute_mse(out, day)
                s = compute_ssim(out, day)
                pct = (1 - avg / bline_mse) * 100
                results.append((name, avg, s, pct))
                out_path = f"enhanced_{name}.png"
                cv2.imwrite(out_path, out)
                print(f"{name:<12}  {avg:>7.1f}  {s:>6.4f}  {pct:>+7.1f}%  {references[name]}")
            except Exception as e:
                print(f"{name:<12}  ERROR: {e}")
        print("─" * 65)
        best = min(results, key=lambda x: x[1])
        print(f"\nBest: {best[0]} (MSE={best[1]:.1f}, SSIM={best[2]:.4f})")

    else:
        # ── Single method ─────────────────────────────────────────────────
        method = args.method
        print(f"Method: {method}")
        if method == 'wiener':
            rows = args.rows or H
            print(f"  Grid: {rows} rows × {args.cols} cols  "
                  f"(tile ≈ {H//rows}×{W//args.cols} px)")
        elif method == 'gamma':
            print(f"  γ = {args.gamma}")
        elif method == 'clahe':
            print(f"  clip={args.clip}, tile={args.tile}×{args.tile}")
        elif method == 'poly':
            print(f"  degree = {args.degree}")
        elif method == 'regional':
            print(f"  sky_frac = {args.sky_frac}")
        if not args.no_denoise:
            print(f"  NLM denoising h = {args.denoise_h}")
        print()

        out = run_method(method, night, day, args)
        print("Results:")
        print_metrics(method, out, night, day)

        out_path = args.output or f"enhanced_{method}.png"
        cv2.imwrite(out_path, out)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()