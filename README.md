# Cell Morphometry Analyzer

A clean, professional Python application for quantitative cell and nuclear morphology analysis with a focus on distinguishing **normal vs malignant-like** features.

The tool extracts the six key morphometric parameters requested:
- Cell area
- Nucleus area
- Perimeter
- Circularity
- Eccentricity
- Nucleus-to-cytoplasm (N/C) area ratio

It includes a fully transparent, rule-based classifier that flags abnormal cells using classic cytopathology criteria (high N/C ratio, nuclear pleomorphism, loss of circularity, etc.).

## Project Structure

```
cell-analyzer/
├── analyzer.py          # Image processing backend (OpenCV + scikit-image)
├── app.py               # Streamlit web UI
├── requirements.txt     # All Python dependencies
└── README.md
```

## Features

**Backend (`analyzer.py`)**
- Robust segmentation for both cell bodies and nuclei (works on H&E, brightfield, fluorescence, etc.)
- Automatic image inversion detection
- Per-cell extraction of all requested shape + size metrics
- Lightweight explainable classifier (no ML black box)
- High-quality synthetic image generator for instant demos and testing

**Frontend (`app.py`)**
- Drag-and-drop image upload
- One-click synthetic demo with known mix of healthy + abnormal cells
- Side-by-side original vs segmented overlay (color-coded boundaries)
- Interactive pandas table with all metrics + classification
- Multiple charts: N/C histograms, N/C vs eccentricity scatter, circularity distributions
- Adjustable analysis parameters (sidebar)
- CSV export of results
- Fully cached analysis for fast iteration

If no image is uploaded, the app generates realistic synthetic microscopy-like images on the fly so you can test everything immediately.

## Installation

```bash
cd cell-analyzer
pip install -r requirements.txt
```

Recommended (clean environment):
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Web Interface (recommended)

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually http://localhost:8501).

**Workflow**
1. Either upload a cell image, **or**
2. Click **"Load synthetic demo (healthy + abnormal cells)"**
3. Adjust parameters in the sidebar if desired
4. Explore the metrics table and charts
5. Download the CSV for further analysis

### Programmatic Use (analyzer only)

```python
from analyzer import (
    load_image,
    segment_and_analyze,
    generate_synthetic_cell_image,
    AnalysisParams,
)

# Option A: synthetic image (great for testing)
img = generate_synthetic_cell_image(n_healthy=5, n_abnormal=3, seed=42)

# Option B: load from disk or file-like
# img = load_image("path/to/your/cell_image.png")

result = segment_and_analyze(img)

print("Cells found:", result["summary"]["num_cells"])
print("Abnormal flagged:", result["summary"]["num_abnormal"])

for cell in result["cells"]:
    print(cell["cell_id"], cell["nc_ratio"], cell["classification"])
```

## The Lightweight Classifier

The classifier is deliberately simple and fully explainable:

**Abnormal (malignant-like)** if:
- N/C ratio is very high (≥ 0.72), **or**
- At least two of the following are true:
  - N/C ratio ≥ 0.58
  - Eccentricity ≥ 0.74 (elongated/irregular)
  - Circularity ≤ 0.58 (atypical shape)
  - Nucleus area ≥ 520 px (enlarged nucleus)

**Borderline** — exactly one of the above criteria is met.

**Normal morphology** — none of the criteria triggered.

These thresholds are exposed in the Streamlit sidebar so you can tune sensitivity.

## Synthetic Data Generator

The `generate_synthetic_cell_image()` function creates realistic-looking test images containing:
- Round, low-N/C "healthy" cells
- Larger, more eccentric, high-N/C "abnormal" cells

It adds subtle noise, blur, and vignette to mimic real microscope output. All metrics and classifications shown in demo mode are produced by the exact same analysis pipeline that runs on real uploads.

## Dependencies

Core stack:
- `numpy`, `pandas`
- `opencv-python-headless`
- `scikit-image`
- `matplotlib`
- `Pillow`
- `scipy`
- `streamlit` (UI only)

All versions are pinned conservatively in `requirements.txt`.

## Extending the Tool

- Add a trained scikit-learn / PyTorch classifier as an optional backend (the current rule-based one remains the fallback).
- Support multi-channel fluorescence (separate channels for cytoplasm vs nucleus).
- Export annotated images or full segmentation masks.
- Add batch folder processing.

## License & Attribution

This project was created as a focused demonstration of clean scientific Python architecture, reproducible synthetic data, and explainable analysis for biomedical imaging.

## Quick Start (TL;DR)

```bash
cd cell-analyzer
pip install -r requirements.txt
streamlit run app.py
# Then click "Load synthetic demo"
```

Enjoy exploring cell morphology!
