#!/usr/bin/env python3
"""
analyzer.py — Cell Morphometry Analysis Backend

Image processing pipeline for quantitative analysis of cell and nuclear morphology.
Updated with global vacuole tracking and visualization.
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
    min_cell_area: int = 380  # pixels — discard tiny debris
    min_nucleus_area: int = 65  # pixels
    nucleus_dark_percentile: float = 26.0
    vacuole_threshold_offset: float = 0.15 # Sensitivity for hole detection
    cell_gaussian_sigma: float = 1.2
    nucleus_gaussian_sigma: float = 0.6
    # Classification thresholds
    nc_ratio_abnormal: float = 0.58
    nc_ratio_very_high: float = 0.72
    eccentricity_abnormal: float = 0.74
    circularity_abnormal: float = 0.58
    nucleus_area_large: float = 520.0


# ----------------------------- Image Loading ---------------------------------
def load_image(source: ImageLike) -> np.ndarray:
    if isinstance(source, (str,)):
        img = cv2.imread(source, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {source}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if isinstance(source, io.BytesIO):
        pil = PILImage.open(source)
        return np.array(pil.convert("RGB"))
    if isinstance(source, PILImage.Image):
        return np.array(source.convert("RGB"))
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = color.gray2rgb(arr)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return arr
    raise TypeError(f"Unsupported image source type: {type(source)}")


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img.astype(np.float32) / 255.0
    else:
        gray = color.rgb2gray(img)
    return gray.astype(np.float32)


def _maybe_invert(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    p1, p2 = np.percentile(gray, [2, 98])
    if p2 - p1 < 0.05:
        return gray, False
    t = filters.threshold_otsu(gray)
    dark_fraction = (gray < t).mean()
    bright_fraction = (gray > t).mean()
    if bright_fraction > 0.35 and dark_fraction < 0.25:
        return 1.0 - gray, True
    return gray, False


# ----------------------------- Segmentation ----------------------------------
def _segment_cells(gray: np.ndarray, params: AnalysisParams) -> np.ndarray:
    blurred = filters.gaussian(gray, sigma=params.cell_gaussian_sigma)
    thresh = filters.threshold_otsu(blurred)
    mask = blurred < thresh
    mask = morphology.remove_small_objects(mask, min_size=params.min_cell_area // 2)
    mask = morphology.remove_small_holes(mask, area_threshold=200)
    mask = morphology.closing(mask, morphology.disk(2))
    distance = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(distance, min_distance=15, labels=mask)
    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = ndi.label(markers)
    labeled_platelets = segmentation.watershed(-distance, markers, mask=mask)
    boundaries = segmentation.find_boundaries(labeled_platelets, mode='inner')
    mask[boundaries] = False
    mask = morphology.remove_small_objects(mask, min_size=params.min_cell_area)
    return mask


def _segment_nuclei_inside_cells(
        gray: np.ndarray, cell_mask: np.ndarray, labeled_cells: np.ndarray, params: AnalysisParams
) -> np.ndarray:
    nucleus_mask = np.zeros_like(cell_mask, dtype=bool)
    regions = measure.regionprops(labeled_cells)
    for region in regions:
        if region.area < params.min_cell_area: continue
        minr, minc, maxr, maxc = region.bbox
        cell_sub = gray[minr:maxr, minc:maxc]
        sub_mask = region.image
        intensities = cell_sub[sub_mask]
        if len(intensities) < 30: continue
        t = np.percentile(intensities, params.nucleus_dark_percentile)
        nuc_sub = (cell_sub < t) & sub_mask
        nuc_sub = morphology.remove_small_objects(nuc_sub, min_size=params.min_nucleus_area // 2)
        nucleus_mask[minr:maxr, minc:maxc] |= nuc_sub
    nucleus_mask = morphology.opening(nucleus_mask, morphology.disk(1))
    nucleus_mask = morphology.remove_small_objects(nucleus_mask, min_size=params.min_nucleus_area)
    return nucleus_mask


def segment_and_analyze(
        image: np.ndarray, params: Optional[AnalysisParams] = None
) -> Dict[str, Any]:
    if params is None: params = AnalysisParams()

    gray = _to_grayscale(image)
    gray, was_inverted = _maybe_invert(gray)
    gray = exposure.rescale_intensity(gray, in_range="image", out_range=(0.0, 1.0))

    cell_mask = _segment_cells(gray, params)
    labeled_cells = measure.label(cell_mask)
    nucleus_mask = _segment_nuclei_inside_cells(gray, cell_mask, labeled_cells, params)
    
    # Initialize global vacuole mask
    vacuole_mask = np.zeros_like(cell_mask, dtype=bool)

    cell_regions = measure.regionprops(labeled_cells, intensity_image=gray)
    nucleus_regions = measure.regionprops(measure.label(nucleus_mask))
    nucleus_props_by_label = {nr.label: nr for nr in nucleus_regions}

    cells: List[Dict[str, Any]] = []

    for creg in cell_regions:
        if creg.area < params.min_cell_area: continue
        
        minr, minc, maxr, maxc = creg.bbox
        cell_bbox_mask = labeled_cells[minr:maxr, minc:maxc] == creg.label
        
        # --- Calculate Hole/Vacuole Mask for this specific cell ---
        cell_sub = gray[minr:maxr, minc:maxc]
        nuc_sub_mask = nucleus_mask[minr:maxr, minc:maxc] & cell_bbox_mask
        
        median_val = np.median(cell_sub[cell_bbox_mask])
        local_vac_mask = (cell_sub < (median_val - params.vacuole_threshold_offset)) & cell_bbox_mask & (~nuc_sub_mask)
        
        # Add to global mask
        vacuole_mask[minr:maxr, minc:maxc] |= local_vac_mask
        
        # --- Metrics ---
        vac_area_count = np.sum(local_vac_mask)
        vacuolization_pct = (vac_area_count / creg.area) * 100.0
        
        # Standard features
        cell_area = float(creg.area)
        perimeter = float(creg.perimeter) if creg.perimeter > 0 else 1.0
        circularity = (4.0 * np.pi * cell_area) / (perimeter * perimeter)
        eccentricity = float(creg.eccentricity)

        # Nucleus Area calculation
        nucleus_area = 0.0
        for nl in np.unique(measure.label(nucleus_mask)[minr:maxr, minc:maxc]):
            if nl == 0: continue
            nreg = nucleus_props_by_label.get(nl)
            if nreg is None: continue
            cy, cx = nreg.centroid
            if (minr <= cy < maxr) and (minc <= cx < maxc):
                 nucleus_area += float(nreg.area)

        cytoplasm_area = max(cell_area - nucleus_area, 1.0)
        nc_ratio = nucleus_area / cytoplasm_area

        classification, reasons = _classify_morphology(nc_ratio, circularity, eccentricity, nucleus_area, cell_area, params)

        cells.append({
            "cell_id": int(creg.label),
            "cell_area": round(cell_area, 1),
            "nucleus_area": round(nucleus_area, 1),
            "cytoplasm_area": round(cytoplasm_area, 1),
            "nc_ratio": round(nc_ratio, 3),
            "perimeter": round(perimeter, 1),
            "circularity": round(circularity, 3),
            "eccentricity": round(eccentricity, 3),
            "vacuolization_pct": round(vacuolization_pct, 1),
            "bbox": (minr, minc, maxr, maxc),
            "classification": classification,
            "reasons": reasons,
        })

    # Summary Statistics
    total_cell_area = np.sum(cell_mask)
    total_vac_area = np.sum(vacuole_mask)
    global_vac_pct = (total_vac_area / total_cell_area * 100) if total_cell_area > 0 else 0.0

    cells = sorted(cells, key=lambda c: (c["bbox"][0], c["bbox"][1]))
    for idx, c in enumerate(cells, start=1): c["cell_id"] = idx

    summary = {
        "num_cells": len(cells),
        "abnormal_pct": round(100.0 * sum(1 for c in cells if "Abnormal" in c["classification"]) / len(cells), 1) if cells else 0.0,
        "total_vacuolization_pct": round(global_vac_pct, 2),
        "image_inverted": was_inverted,
    }

    return {
        "cells": cells,
        "cell_mask": cell_mask,
        "nucleus_mask": nucleus_mask,
        "vacuole_mask": vacuole_mask, # Included for visualization
        "overlay": _create_overlay(image, cell_mask, nucleus_mask, vacuole_mask),
        "summary": summary,
        "params_used": params,
    }


# ----------------------------- Classification --------------------------------
def _classify_morphology(nc_ratio, circularity, eccentricity, nucleus_area, cell_area, params) -> Tuple[str, List[str]]:
    reasons = []
    if nc_ratio >= params.nc_ratio_very_high: reasons.append("very high N/C ratio")
    elif nc_ratio >= params.nc_ratio_abnormal: reasons.append("elevated N/C ratio")
    if eccentricity >= params.eccentricity_abnormal: reasons.append("high eccentricity")
    if circularity <= params.circularity_abnormal: reasons.append("low circularity")
    if nucleus_area >= params.nucleus_area_large: reasons.append("enlarged nucleus")
    
    if len(reasons) >= 2 or nc_ratio >= params.nc_ratio_very_high: return "Abnormal (malignant-like)", reasons
    elif reasons: return "Borderline", reasons
    return "Normal morphology", []


# ----------------------------- Visualization ---------------------------------
def _create_overlay(image, cell_mask, nucleus_mask, vacuole_mask) -> np.ndarray:
    overlay = image.copy()
    if overlay.dtype != np.uint8: overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    cell_u8 = (cell_mask.astype(np.uint8) * 255)
    nuc_u8 = (nucleus_mask.astype(np.uint8) * 255)
    vac_u8 = (vacuole_mask.astype(np.uint8) * 255)

    cell_contours, _ = cv2.findContours(cell_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nuc_contours, _ = cv2.findContours(nuc_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vac_contours, _ = cv2.findContours(vac_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(overlay, cell_contours, -1, (0, 188, 212), 2)  # Cyan
    cv2.drawContours(overlay, nuc_contours, -1, (233, 30, 99), 2)   # Magenta
    cv2.drawContours(overlay, vac_contours, -1, (0, 255, 255), 2)   # Yellow for holes

    # Add light semi-transparent fill for vacuoles
    vac_color = np.array([255, 255, 0], dtype=np.uint8)
    overlay[vacuole_mask] = (0.5 * overlay[vacuole_mask] + 0.5 * vac_color).astype(np.uint8)

    return overlay

# (Synthetic data generator remains unchanged...)
