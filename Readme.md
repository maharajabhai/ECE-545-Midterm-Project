# Night-to-Day Image Enhancement

A classical image processing pipeline for transforming night-time urban imagery into daytime equivalents — no machine learning at inference time.

**Best result:** MSE 9018.6 (no image processing) → 79.8 · SSIM 0.870 · 99.1% improvement

---

## Quick Start

```bash
# Best method (Wiener, default settings)
python enhance.py images/night.jpg images/day.jpg

# Run all methods and print comparison table
python enhance.py images/night.jpg images/day.jpg --compare

# Save output to a specific path
python enhance.py images/night.jpg images/day.jpg --method wiener --output PATH
```

---

## Methods

| Method | MSE | SSIM | Δ% | 
|--------|-----|------|----|
| `wiener` | **79.8** | **0.870** | **+99.1%** |
| `poly` | 2442.2 | 0.588 | +73.0% | 
| `regional` | 2593.7 | 0.541 | +71.3% | 
| `linear` | 3473.9 | 0.503 | +61.5% | 
| `gamma` | 3763.6 | 0.314 | +58.3% | 
| `hist_match` | 4307.9 | 0.118 | +52.3% | 
| `retinex` | 5848.4 | 0.136 | +35.3% | 
| `clahe` | 6888.9 | 0.225 | +23.7% | 
| Baseline | 9018.6 | 0.160 | — | — |

---

## Options

### Wiener (recommended)
```bash
python enhance.py night.png day.png --method wiener \
  --rows 737 \        # grid rows (default: image height)
  --cols 256 \        # grid columns (default: 256, ~4px per tile)
  --denoise-h 10      # NLM denoising strength (default: 10)

# Skip NLM denoising (lower MSE, noisier output)
python enhance.py night.png day.png --method wiener --no-denoise
```

### Gamma
```bash
python enhance.py night.png day.png --method gamma --gamma 0.4
```

### CLAHE
```bash
python enhance.py night.png day.png --method clahe --clip 2.0 --tile 8
```

### Regional
```bash
python enhance.py night.png day.png --method regional --sky-frac 0.45
```


---

## Dependencies

```bash
pip install opencv-python numpy scikit-image
```

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | >= 4.5 | Image I/O, filtering, CLAHE, NLM |
| `numpy` | >= 1.21 | Array ops, covariance estimation |
| `scikit-image` | >= 0.19 | SSIM metric, histogram matching |


