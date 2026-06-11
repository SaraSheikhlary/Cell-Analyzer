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
    color, draw, exposure, feature, filters, measure, 
    morphology, segmentation, util
)

# ----------------------------- Configuration ---------------------------------
@dataclass
class AnalysisParams:
    min_cell_area: int = 150
    min_nucleus_area: int = 65
    max_mean_intensity: float = 0.75
    nucleus_dark_percentile: float = 26.0
    cell_gaussian_sigma: float = 1.2
    nucleus_gaussian_sigma: float = 0.6
    watershed_min_distance: int = 8
    adaptive_block_size: int = 35
    nc_ratio_abnormal: float = 0.58
    nc_ratio_very_high: float = 0.72
    eccentricity_abnormal: float = 0.74
    circularity_abnormal: float = 0.58
    nucleus_area_large: float = 520.0

# ----------------------------- Helpers ---------------------------------------
def load_image(source: Any) -> np.ndarray:
    if isinstance(source, (str,)):
        img = cv2.imread(source, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(source, (bytes, bytearray)): source = io.BytesIO(source)
    if isinstance(source, io.BytesIO): return np.array(PILImage.open(source).convert("RGB"))
    if isinstance(source, PILImage.Image): return np.array(source.convert("RGB"))
    if isinstance(source, np.ndarray): return source
    raise TypeError("Unsupported image type")

def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3: return color.rgb2gray(img).astype(np.float32)
    return img.astype(np.float32) / 255.0

def _maybe_invert(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    p1, p2 = np.percentile(gray, [2, 98])
    if p2 - p1 < 0.05: return gray, False
    t = filters.threshold_otsu(gray)
    if (gray > t).mean() > 0.35 and (gray < t).mean() < 0.25: return 1.0 - gray, True
    return gray, False

# ----------------------------- Segmentation Logic ----------------------------
def _segment_cells(gray: np.ndarray, params: AnalysisParams) -> np.ndarray:
    block = params.adaptive_block_size if params.adaptive_block_size % 2 != 0 else params.adaptive_block_size + 1
    local_thresh = filters.threshold_local(gray, block_size=block, offset=-0.02)
    mask = gray < local_thresh
    mask = morphology.remove_small_objects(mask, min_size=params.min_cell_area // 2)
    mask = morphology.closing(mask, morphology.disk(2))
    distance = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(distance, min_distance=params.watershed_min_distance, labels=mask)
    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = ndi.label(markers)
    labeled = segmentation.watershed(-distance, markers, mask=mask)
    return labeled > 0

def _segment_nuclei_inside_cells(gray: np.ndarray, cell_mask: np.ndarray, params: AnalysisParams) -> np.ndarray:
    # Look for dark spots within the cells
    nucleus_mask = np.zeros_like(gray, dtype=bool)
    # Thresholding based on percentile to find dark interior
    thresh = np.percentile(gray[cell_mask], params.nucleus_dark_percentile)
    nucleus_mask = (gray < thresh) & cell_mask
    nucleus_mask = morphology.remove_small_objects(nucleus_mask, min_size=params.min_nucleus_area)
    return nucleus_mask

def _create_overlay(image: np.ndarray, cell_mask: np.ndarray, nucleus_mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    # Draw cell boundaries in blue
    boundaries = segmentation.find_boundaries(cell_mask, mode='inner')
    overlay[boundaries] = [0, 255, 255] # Cyan
    # Highlight nuclei in magenta
    overlay[nucleus_mask] = [255, 0, 255] # Magenta
    return overlay

# ----------------------------- Main Pipeline ---------------------------------
def segment_and_analyze(image: np.ndarray, params: Optional[AnalysisParams] = None) -> Dict[str, Any]:
    if params is None: params = AnalysisParams()
    gray = _to_grayscale(image)
    gray, _ = _maybe_invert(gray)
    
    # 1. Segment
    cell_mask = _segment_cells(gray, params)
    nucleus_mask = _segment_nuclei_inside_cells(gray, cell_mask, params)
    labeled_cells = measure.label(cell_mask)
    
    # 2. Analyze
    cells = []
    regions = measure.regionprops(labeled_cells, intensity_image=gray)
    
    for r in regions:
        if r.area < params.min_cell_area or r.mean_intensity > params.max_mean_intensity:
            continue
            
        # Feature calculation
        cell_coords = r.coords
        nuc_area = np.sum(nucleus_mask[cell_coords[:,0], cell_coords[:,1]])
        
        cells.append({
            "id": r.label,
            "cell_area": float(r.area),
            "vacuolization_pct": float(r.mean_intensity * 100), # Simplistic metric
            "circularity": (4 * np.pi * r.area) / (r.perimeter**2 + 1e-6),
            "nc_ratio": nuc_area / r.area
        })
        
    # 3. Summary
    total_area = sum(c["cell_area"] for c in cells)
    global_vac = sum(c["vacuolization_pct"] * c["cell_area"] / 100.0 for c in cells) / total_area if total_area > 0 else 0
    
    return {
        "cells": cells, 
        "overlay": _create_overlay(image, cell_mask, nucleus_mask), 
        "summary": {"num_cells": len(cells), "vac_pct_global": round(global_vac, 1)}
    }
