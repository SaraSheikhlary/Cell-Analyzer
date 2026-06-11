#!/usr/bin/env python3
"""
analyzer.py — Cell Morphometry Analysis Backend
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy import ndimage as ndi
from skimage import (
    color,
    draw,
    exposure,
    feature,
    filters,
    measure,
    morphology,
    segmentation,
    util,
)

# ----------------------------- Configuration ---------------------------------
@dataclass
class AnalysisParams:
    """Tunable parameters for segmentation and classification."""
    min_cell_area: int = 200          # pixels — lowered to catch small platelets
    max_cell_area: int = 40000        # pixels — Discard giant artifacts
    min_nucleus_area: int = 65        # pixels
    nucleus_dark_percentile: float = 26.0
    cell_gaussian_sigma: float = 1.2
    nucleus_gaussian_sigma: float = 0.6
    otsu_multiplier: float = 1.15     # Pushes threshold to catch lighter platelets
    
    # Classification thresholds
    nc_ratio_abnormal: float = 0.58
    nc_ratio_very_high: float = 0.72
    eccentricity_abnormal: float = 0.74
    circularity_abnormal: float = 0.58
    nucleus_area_large: float = 520.0

# ----------------------------- Core Functions --------------------------------
def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img.astype(np.float32) / 255.0
    else:
        gray = color.rgb2gray(img)
    return gray.astype(np.float32)

def _maybe_invert(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    p1, p2 = np.percentile(gray, [2, 98])
    if p2 - p1 < 0.05: return gray, False
    t = filters.threshold_otsu(gray)
    dark_fraction = (gray < t).mean()
    bright_fraction = (gray > t).mean()
    if bright_fraction > 0.35 and dark_fraction < 0.25:
        return 1.0 - gray, True
    return gray, False

def _segment_cells(gray: np.ndarray, params: AnalysisParams) -> np.ndarray:
    blurred = filters.gaussian(gray, sigma=params.cell_gaussian_sigma)
    # Use multiplier to capture lighter cells on the edges
    thresh = min(1.0, filters.threshold_otsu(blurred) * params.otsu_multiplier)
    mask = blurred < thresh

    mask = morphology.remove_small_objects(mask, min_size=params.min_cell_area // 2)
    mask = morphology.remove_small_holes(mask, area_threshold=200)
    mask = morphology.closing(mask, morphology.disk(2))

    distance = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(distance, min_distance=15, labels=mask)
    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = ndi.label(markers)
    
    labeled = segmentation.watershed(-distance, markers, mask=mask)
    boundaries = segmentation.find_boundaries(labeled, mode='inner')
    mask[boundaries] = False
    return morphology.remove_small_objects(mask, min_size=params.min_cell_area)

def segment_and_analyze(image: np.ndarray, params: Optional[AnalysisParams] = None) -> Dict[str, Any]:
    if params is None: params = AnalysisParams()
    gray = _to_grayscale(image)
    gray, was_inverted = _maybe_invert(gray)
    gray = exposure.rescale_intensity(gray, in_range="image", out_range=(0.0, 1.0))

    cell_mask = _segment_cells(gray, params)
    labeled_cells = measure.label(cell_mask)

    cell_regions = measure.regionprops(labeled_cells, intensity_image=gray)
    
    cells = []
    total_vacuole_pixels = 0.0
    total_valid_cell_area = 0.0

    for creg in cell_regions:
        # Filter artifacts using max_cell_area
        if creg.area < params.min_cell_area or creg.area > params.max_cell_area:
            continue

        cell_area = float(creg.area)
        minr, minc, maxr, maxc = creg.bbox
        cell_bbox_mask = labeled_cells[minr:maxr, minc:maxc] == creg.label
        
        # Vacuolization Calculation (Statistical Outlier Method)
        cell_pixels = gray[minr:maxr, minc:maxc][cell_bbox_mask]
        if len(cell_pixels) > 0:
            median_val = np.median(cell_pixels)
            std_val = np.std(cell_pixels)
            diff = np.abs(cell_pixels - median_val)
            vac_px_count = np.sum((diff > 0.20) & (diff > 2.0 * std_val))
            vacuolization_pct = (vac_px_count / cell_area) * 100.0
        else:
            vac_px_count = 0
            vacuolization_pct = 0.0
            
        total_vacuole_pixels += vac_px_count
        total_valid_cell_area += cell_area

        cells.append({
            "cell_id": int(creg.label),
            "cell_area": round(cell_area, 1),
            "vacuolization_pct": round(vacuolization_pct, 1),
            "bbox": (minr, minc, maxr, maxc)
        })

    cells = sorted(cells, key=lambda c: (c["bbox"][0], c["bbox"][1]))
    for idx, c in enumerate(cells, start=1): c["cell_id"] = idx

    summary = {
        "num_cells": len(cells),
        "total_vacuolization_pct": round(100.0 * total_vacuole_pixels / total_valid_cell_area, 1) if total_valid_cell_area > 0 else 0.0,
        "image_inverted": was_inverted
    }

    return {"cells": cells, "summary": summary}
