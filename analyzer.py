#!/usr/bin/env python3
"""
analyzer.py — Cell Morphometry Analysis Backend
Optimized for clean, precision-fit diagnostic annotations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy import ndimage as ndi
from skimage import color, draw, exposure, feature, filters, measure, morphology, segmentation, util

# ----------------------------- Configuration ---------------------------------
@dataclass
class AnalysisParams:
    """Tunable parameters for segmentation and classification."""
    # INCREASED: Stricter size filter to ignore background noise
    min_cell_area: int = 250       
    max_cell_area: int = 3500     
    min_nucleus_area: int = 60
    nucleus_dark_percentile: float = 24.0
    vacuole_threshold_offset: float = 0.12  
    cell_gaussian_sigma: float = 1.0
    nucleus_gaussian_sigma: float = 0.6
    nc_ratio_abnormal: float = 0.58
    nc_ratio_very_high: float = 0.72
    eccentricity_abnormal: float = 0.74
    circularity_abnormal: float = 0.55 # Only count semi-round objects
    nucleus_area_large: float = 150.0

# ----------------------------- Core Functions --------------------------------
def load_image(file_obj: Any) -> np.ndarray:
    img = PILImage.open(file_obj).convert("RGB")
    return np.array(img)

def _classify_cell(nc_ratio: float, eccentricity: float, circularity: float, nucleus_area: float, params: AnalysisParams) -> tuple[str, list[str]]:
    reasons = []
    if nc_ratio >= params.nc_ratio_abnormal: reasons.append("high N/C ratio")
    if eccentricity >= params.eccentricity_abnormal: reasons.append("high eccentricity")
    if circularity <= params.circularity_abnormal: reasons.append("low circularity")
    if nucleus_area >= params.nucleus_area_large: reasons.append("enlarged nucleus")

    if len(reasons) >= 2 or nc_ratio >= params.nc_ratio_very_high:
        return "Abnormal (malignant-like)", reasons
    return ("Borderline" if reasons else "Normal morphology", reasons)

def _draw_tight_circle(overlay, mask, color, thickness=1, is_vacuole=False):
    """Draws tight, precision circles that hug the cell edges."""
    contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # Precision fit using minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        
        if radius > 1:
            # Vacuoles get a tighter internal fit
            if is_vacuole:
                cv2.circle(overlay, center, max(1, radius - 1), color, thickness)
            else:
                # Cells get a perfect edge fit (no extra padding)
                cv2.circle(overlay, center, radius, color, thickness)

def _create_overlay(image: np.ndarray, cell_mask: np.ndarray, nucleus_mask: np.ndarray, vacuole_mask: np.ndarray, cells_data: list = None) -> np.ndarray:
    overlay = image.copy()
    if overlay.dtype != np.uint8:
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    # 1. Draw Vacuoles (Tiny yellow markers)
    _draw_tight_circle(overlay, vacuole_mask, (255, 235, 59), thickness=1, is_vacuole=True)

    # 2. Draw Cell Boundaries (Clean, non-padded circles)
    if cells_data:
        for cell in cells_data:
            bbox = cell["bbox"]
            y1, x1, y2, x2 = bbox
            
            # Color logic
            cell_color = (0, 255, 0) # Green for normal
            if "Abnormal" in cell["classification"] or cell["vacuolization_pct"] > 15.0:
                cell_color = (0, 0, 255) # Red for concern
            elif "Borderline" in cell["classification"] or cell["vacuolization_pct"] > 5.0:
                 cell_color = (0, 165, 255) # Orange for warning

            center_y = int((y1 + y2) / 2)
            center_x = int((x1 + x2) / 2)
            radius = int(max(y2-y1, x2-x1) / 2)
            
            # Draw circle without the "+4" padding from before
            cv2.circle(overlay, (center_x, center_y), radius, cell_color, 1)

    return overlay

def segment_and_analyze(image_array: np.ndarray, params: AnalysisParams) -> dict:
    """Analyzes with strict noise rejection."""
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()

    # Pre-process to improve contrast
    gray = cv2.equalizeHist(gray)
    
    # Use Gaussian blur to kill high-frequency noise
    blurred = cv2.GaussianBlur(gray, (5, 5), params.cell_gaussian_sigma)
    
    # Adaptive Thresholding is better for uneven microscope light
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2
    )
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    
    cells = []
    cell_mask_total = np.zeros_like(thresh)
    nucleus_mask_total = np.zeros_like(thresh)
    vacuole_mask_total = np.zeros_like(thresh)
    
    cell_id_counter = 1

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # STRICER FILTER: 
        if area < params.min_cell_area or area > params.max_cell_area: continue

        cell_mask = (labels == i).astype(np.uint8)
        
        # Shape filter: Skip jagged/non-platelet debris
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        c = contours[0]
        
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity < 0.4: continue # Too irregular to be a platelet

        # (Rest of the metrics remain the same as your previous working version)
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        raw_pixels = gray[cell_mask > 0]
        nuc_thresh_val = np.percentile(raw_pixels, params.nucleus_dark_percentile)
        nuc_mask = ((gray <= nuc_thresh_val) & (cell_mask > 0)).astype(np.uint8)
        nuc_area = int(np.sum(nuc_mask))
        
        # Vacuole tracking
        vac_thresh_val = np.median(raw_pixels) + (params.vacuole_threshold_offset * 255)
        vac_mask = ((gray >= vac_thresh_val) & (cell_mask > 0) & (nuc_mask == 0)).astype(np.uint8)
        vac_area = int(np.sum(vac_mask))
        
        cytoplasm_area = max(1, area - nuc_area)
        nc_ratio = round(nuc_area / cytoplasm_area, 3)
        vac_pct = round((vac_area / area) * 100, 1)

        # Classification
        classification, _ = _classify_cell(nc_ratio, 0, circularity, nuc_area, params)

        cell_mask_total = cv2.bitwise_or(cell_mask_total, cell_mask)
        nucleus_mask_total = cv2.bitwise_or(nucleus_mask_total, nuc_mask)
        vacuole_mask_total = cv2.bitwise_or(vacuole_mask_total, vac_mask)

        cells.append({
            "cell_id": cell_id_counter, "bbox": (y, x, y + h, x + w),
            "cell_area": area, "nucleus_area": nuc_area, "cytoplasm_area": cytoplasm_area,
            "nc_ratio": nc_ratio, "vacuolization_pct": vac_pct, "perimeter": perimeter,
            "circularity": circularity, "eccentricity": 0, "classification": classification,
        })
        cell_id_counter += 1

    overlay_img = _create_overlay(image_array, cell_mask_total, nucleus_mask_total, vacuole_mask_total, cells)
    return {"cells": cells, "summary": {"num_cells": len(cells)}, "overlay": overlay_img}
