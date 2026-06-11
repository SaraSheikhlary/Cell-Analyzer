#!/usr/bin/env python3
"""
analyzer.py — Cell Morphometry Analysis Backend

Updated: Includes intensity-based background filtering to ignore noise
and weighted aggregation for summary statistics.
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

# ----------------------------- Type Aliases ----------------------------------
ImageLike = Union[str, bytes, io.BytesIO, np.ndarray, PILImage.Image]


# ----------------------------- Configuration ---------------------------------
@dataclass
class AnalysisParams:
    """Tunable parameters for segmentation and classification."""
    min_cell_area: int = 150            
    min_nucleus_area: int = 65          
    # NEW: Filter out bright background artifacts (0.0 = black, 1.0 = white)
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


# ----------------------------- Image Loading ---------------------------------
def load_image(source: ImageLike) -> np.ndarray:
    if isinstance(source, (str,)):
        img = cv2.imread(source, cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(f"Could not read image at {source}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if isinstance(source, (bytes, bytearray)): source = io.BytesIO(source)
    if isinstance(source, io.BytesIO): return np.array(PILImage.open(source).convert("RGB"))
    if isinstance(source, PILImage.Image): return np.array(source.convert("RGB"))
    
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2: arr = color.gray2rgb(arr)
        elif arr.shape[2] == 4: arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return arr
    raise TypeError(f"Unsupported image source type: {type(source)}")


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2: return img.astype(np.float32) / 255.0
    return color.rgb2gray(img).astype(np.float32)


def _maybe_invert(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    p1, p2 = np.percentile(gray, [2, 98])
    if p2 - p1 < 0.05: return gray, False
    t = filters.threshold_otsu(gray)
    if (gray > t).mean() > 0.35 and (gray < t).mean() < 0.25: return 1.0 - gray, True
    return gray, False


# ----------------------------- Segmentation ----------------------------------
def _segment_cells(gray: np.ndarray, params: AnalysisParams) -> np.ndarray:
    block_size = params.adaptive_block_size if params.adaptive_block_size % 2 != 0 else params.adaptive_block_size + 1
    local_thresh = filters.threshold_local(gray, block_size=block_size, offset=-0.02)
    mask = gray < local_thresh

    mask = morphology.remove_small_objects(mask, min_size=params.min_cell_area // 2)
    mask = morphology.closing(mask, morphology.disk(2))

    distance = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(distance, min_distance=params.watershed_min_distance, labels=mask)
    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = ndi.label(markers)
    
    labeled_platelets = segmentation.watershed(-distance, markers, mask=mask)
    boundaries = segmentation.find_boundaries(labeled_platelets, mode='inner')
    mask[boundaries] = False
    return morphology.remove_small_objects(mask, min_size=params.min_cell_area)


# ----------------------------- Core Pipeline ---------------------------------
def segment_and_analyze(image: np.ndarray, params: Optional[AnalysisParams] = None) -> Dict[str, Any]:
    if params is None: params = AnalysisParams()
    gray = _to_grayscale(image)
    gray, was_inverted = _maybe_invert(gray)
    gray = exposure.rescale_intensity(gray, in_range="image", out_range=(0.0, 1.0))

    cell_mask = _segment_cells(gray, params)
    labeled_cells = measure.label(cell_mask)

    # Nuclei segmentation
    nucleus_mask = np.zeros_like(cell_mask, dtype=bool)
    cell_regions = measure.regionprops(labeled_cells, intensity_image=gray)
    
    # Pre-calculate Nuclei to avoid re-running logic
    # (Simplified for brevity, same logic as before)
    # ... [Keep your existing _segment_nuclei_inside_cells logic here] ...
    # (I have omitted it to keep code block concise, ensure you keep your function)
    
    # FILTERING: Exclude cells that are "too bright" (background/noise)
    cells: List[Dict[str, Any]] = []
    
    # We iterate and validate before adding
    for creg in cell_regions:
        if creg.area < params.min_cell_area: continue
        
        # Check if the mean intensity of this "cell" is too bright (background noise)
        if creg.mean_intensity > params.max_mean_intensity:
            continue
            
        # ... [Calculate features] ...
        # (Ensure you keep your existing calculation logic here)
        
        # New calculation for summary
        cell_area = float(creg.area)
        # ... rest of your loop ...
        
        # Add to list
        # ...

    # UPDATED SUMMARY STATS
    if cells:
        total_vac_pixels = sum(c["vacuolization_pct"] * c["cell_area"] / 100.0 for c in cells)
        total_area = sum(c["cell_area"] for c in cells)
        global_vac_pct = (total_vac_pixels / total_area) * 100.0 if total_area > 0 else 0.0
        
        # ... [Rest of summary construction] ...
        summary = {
            "num_cells": len(cells),
            "vac_pct_global": round(global_vac_pct, 1) # This is your accurate global metric
        }
    
    return {"cells": cells, "overlay": _create_overlay(image, cell_mask, nucleus_mask), "summary": summary}
