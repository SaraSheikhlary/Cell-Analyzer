#!/usr/bin/env python3
"""
analyzer.py — Cell Morphometry Analysis Backend
Updated with true physical hole-tracking, precision edge segmentation, 
and clinical-style diagnostic circle overlays.
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
    min_cell_area: int = 120  # Lowered default to automatically catch small round cells
    min_nucleus_area: int = 40
    nucleus_dark_percentile: float = 26.0
    vacuole_threshold_offset: float = 0.15  # How much brighter than cell median to be a hole
    cell_gaussian_sigma: float = 1.0
    nucleus_gaussian_sigma: float = 0.6
    nc_ratio_abnormal: float = 0.58
    nc_ratio_very_high: float = 0.72
    eccentricity_abnormal: float = 0.74
    circularity_abnormal: float = 0.60
    nucleus_area_large: float = 150.0

# ----------------------------- Core Functions --------------------------------
def load_image(file_obj: Any) -> np.ndarray:
    """Loads image from file object."""
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

def _draw_diagnostic_circles(overlay: np.ndarray, mask: np.ndarray, color: tuple, thickness: int = 2, is_vacuole: bool = False):
    """Draws smooth, clinical-style circles around detected features."""
    contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # Calculate the minimum enclosing circle for the feature
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        
        if radius > 0:
            # If it's a vacuole, make the circle slightly tighter
            if is_vacuole:
                radius = max(2, radius - 1)
            else:
                # Add a little padding to mimic a human drawing around the cell
                radius += 3
                
            cv2.circle(overlay, center, radius, color, thickness)

def _create_overlay(image: np.ndarray, cell_mask: np.ndarray, nucleus_mask: np.ndarray, vacuole_mask: np.ndarray, cells_data: list = None) -> np.ndarray:
    """Creates an overlay mimicking manual diagnostic annotations."""
    overlay = image.copy()
    if overlay.dtype != np.uint8:
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    # 1. Draw Vacuoles (Yellow circles)
    _draw_diagnostic_circles(overlay, vacuole_mask, (255, 235, 59), thickness=1, is_vacuole=True)

    # 2. Draw Cell Boundaries
    # Color-code the circles based on the classification and vacuolization
    if cells_data:
        for cell in cells_data:
            bbox = cell["bbox"]
            y1, x1, y2, x2 = bbox
            
            # RGB Color Assignments
            cell_color = (0, 255, 0) # Default Green (Normal)
            if "Abnormal" in cell["classification"] or cell["vacuolization_pct"] > 15.0:
                cell_color = (255, 0, 0) # Red for high concern/abnormal
            elif "Borderline" in cell["classification"] or cell["vacuolization_pct"] > 5.0:
                 cell_color = (255, 165, 0) # Orange for borderline

            # Draw a circle roughly around the bounding box center
            center_y = int((y1 + y2) / 2)
            center_x = int((x1 + x2) / 2)
            radius = int(max(y2-y1, x2-x1) / 2) + 4 # Padding
            
            cv2.circle(overlay, (center_x, center_y), radius, cell_color, 2)
    else:
        # Fallback if no specific cell data is passed
        _draw_diagnostic_circles(overlay, cell_mask, (0, 255, 0), thickness=2)

    return overlay

def segment_and_analyze(image_array: np.ndarray, params: AnalysisParams) -> dict:
    """Analyzes cell morphology, extracts specific target metrics, and tracks individual holes."""
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()

    # Determine background style to automate proper segmentation visibility
    mean_val = np.mean(gray)
    image_inverted = False
    if mean_val > 127:
        gray_inverted = cv2.bitwise_not(gray)
        image_inverted = True
    else:
        gray_inverted = gray.copy()

    # Smooth local texture variance slightly while maintaining border lines
    blurred = cv2.GaussianBlur(gray_inverted, (5, 5), params.cell_gaussian_sigma)
    
    # Enhanced segmentation combination: Global Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up gaps or jagged edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    
    cells = []
    cell_mask_total = np.zeros_like(thresh)
    nucleus_mask_total = np.zeros_like(thresh)
    vacuole_mask_total = np.zeros_like(thresh)
    
    cell_id_counter = 1

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Skip micro debris particles based on current params
        if area < params.min_cell_area:
            continue

        cell_mask = (labels == i).astype(np.uint8)
        
        # Geolocation bounding box properties
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bbox = (y, x, y + h, x + w)

        raw_pixels = gray[cell_mask > 0]
        if len(raw_pixels) == 0:
            continue

        # --- Exact Nucleus Thresholding ---
        nuc_thresh_val = np.percentile(raw_pixels, params.nucleus_dark_percentile)
        nuc_mask = ((gray <= nuc_thresh_val) & (cell_mask > 0)).astype(np.uint8)
        nuc_mask = cv2.morphologyEx(nuc_mask, cv2.MORPH_OPEN, kernel)
        nuc_area = int(np.sum(nuc_mask))

        # --- REAL VACUOLE (HOLE) TRACKING ---
        # Locates specific zones inside cytoplasm brighter than the current local baseline standard
        raw_median = np.median(raw_pixels)
        vac_thresh_val = raw_median + (params.vacuole_threshold_offset * 255)
        vac_mask = ((gray >= vac_thresh_val) & (cell_mask > 0) & (nuc_mask == 0)).astype(np.uint8)
        
        vac_area = int(np.sum(vac_mask))
        vacuolization_pct = round((vac_area / area) * 100.0, 1) if area > 0 else 0.0

        cytoplasm_area = max(0, area - nuc_area)
        nc_ratio = round(nuc_area / cytoplasm_area, 3) if cytoplasm_area > 0 else 0.0

        # Mathematical Shape Metrics
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        c = contours[0]
        perimeter = round(float(cv2.arcLength(c, True)), 1)
        circularity = round((4 * np.pi * area) / (perimeter ** 2), 3) if perimeter > 0 else 0.0
        if circularity > 1.0: circularity = 1.0

        # Structural Axis Eccentricity Calculation via Structural Moments
        M = cv2.moments(c)
        if M["mu20"] + M["mu02"] > 0:
            common = np.sqrt((M["mu20"] - M["mu02"])**2 + 4 * M["mu11"]**2)
            major = np.sqrt(2 * (M["mu20"] + M["mu02"] + common))
            minor = np.sqrt(2 * (M["mu20"] + M["mu02"] - common))
            eccentricity = round(np.sqrt(1 - (minor / major)**2), 3) if major > 0 else 0.0
        else:
            eccentricity = 0.0

        classification, _ = _classify_cell(nc_ratio, eccentricity, circularity, nuc_area, params)

        # Merge local tracking matrices into global layers
        cell_mask_total = cv2.bitwise_or(cell_mask_total, cell_mask)
        nucleus_mask_total = cv2.bitwise_or(nucleus_mask_total, nuc_mask)
        vacuole_mask_total = cv2.bitwise_or(vacuole_mask_total, vac_mask)

        cells.append({
            "cell_id": cell_id_counter,
            "bbox": bbox,
            "cell_area": area,
            "nucleus_area": nuc_area,
            "cytoplasm_area": cytoplasm_area,
            "nc_ratio": nc_ratio,
            "vacuolization_pct": vacuolization_pct,
            "perimeter": perimeter,
            "circularity": circularity,
            "eccentricity": eccentricity,
            "classification": classification,
        })
        cell_id_counter += 1

    # Call with cells to enable dynamic color coding
    overlay_img = _create_overlay(image_array, cell_mask_total, nucleus_mask_total, vacuole_mask_total, cells)
    
    num_cells = len(cells)
    num_abnormal = sum(1 for c in cells if "Abnormal" in c["classification"])
    abnormal_pct = round((num_abnormal / num_cells * 100.0), 1) if num_cells > 0 else 0.0
    mean_nc = round(np.mean([c["nc_ratio"] for c in cells]), 3) if cells else 0.0
    max_nc = round(np.max([c["nc_ratio"] for c in cells]), 3) if cells else 0.0

    summary = {
        "num_cells": num_cells,
        "num_abnormal": num_abnormal,
        "abnormal_pct": abnormal_pct,
        "mean_nc_ratio": mean_nc,
        "max_nc_ratio": max_nc,
        "image_inverted": image_inverted
    }

    return {"cells": cells, "summary": summary, "overlay": overlay_img}

def generate_synthetic_cell_image(width=800, height=600, n_healthy=10, n_abnormal=6, seed=42) -> np.ndarray:
    """Generates a high-quality synthetic template image for pipeline fallback validation."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Ground color details
    img[:, :] = [240, 245, 245]

    def draw_cell(cy, cx, ry, rx, angle_deg, cell_color, nucleus_ry, nucleus_rx, nucleus_color):
        # Base cellular body contour mapping
        rr, cc = draw.ellipse(int(cy), int(cx), int(ry), int(rx), rotation=np.deg2rad(angle_deg), shape=img.shape[:2])
        for i in range(3):
            img[rr, cc, i] = np.clip(cell_color[i] + rng.integers(-10, 11, size=len(rr)), 0, 255)
        
        # Inject randomized mock structural clearings (vacuoles)
        for _ in range(rng.integers(1, 4)):
            v_ang = rng.uniform(0, 2 * np.pi)
            v_dist = rng.uniform(0.3, 0.6)
            v_cy = cy + ry * v_dist * np.sin(v_ang)
            v_cx = cx + rx * v_dist * np.cos(v_ang)
            v_r = rng.uniform(2, 5)
            rr_v, cc_v = draw.disk((int(v_cy), int(v_cx)), int(v_r), shape=img.shape[:2])
            img[rr_v, cc_v, :] = [255, 255, 255]  # Clean white bright patches

        # Nuclear interior region rendering
        n_cy = cy + rng.uniform(-ry * 0.08, ry * 0.08)
        n_cx = cx + rng.uniform(-rx * 0.08, rx * 0.08)
        rr_n, cc_n = draw.ellipse(int(n_cy), int(n_cx), int(max(3, nucleus_ry)), int(max(3, nucleus_rx)), 
                              rotation=np.deg2rad(angle_deg), shape=img.shape[:2])
        for i in range(3):
            img[rr_n, cc_n, i] = np.clip(nucleus_color[i] + rng.integers(-8, 9, size=len(rr_n)), 0, 255)

    for _ in range(n_healthy):
        cy, cx = rng.uniform(40, height - 40), rng.uniform(40, width - 40)
        draw_cell(cy, cx, rng.uniform(15, 25), rng.uniform(15, 25), rng.uniform(0, 360), 
                  (180, 190, 220), rng.uniform(5, 9), rng.uniform(5, 9), (80, 70, 130))

    for _ in range(n_abnormal):
        cy, cx = rng.uniform(40, height - 40), rng.uniform(40, width - 40)
        draw_cell(cy, cx, rng.uniform(25, 40), rng.uniform(25, 40), rng.uniform(0, 360), 
                  (160, 170, 200), rng.uniform(14, 22), rng.uniform(14, 22), (50, 40, 90))

    return img
