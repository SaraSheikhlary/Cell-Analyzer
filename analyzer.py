#!/usr/bin/env python3
"""
analyzer.py — Cell Morphometry Analysis Backend

Image processing pipeline for quantitative analysis of cell and nuclear morphology.
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
    min_cell_area: int = 200          # pixels — lowered to catch small platelets
    max_cell_area: int = 40000        # pixels — NEW: discard giant artifacts/lines
    min_nucleus_area: int = 65        # pixels
    nucleus_dark_percentile: float = 26.0   # inside each cell, take darkest X%
    cell_gaussian_sigma: float = 1.2
    nucleus_gaussian_sigma: float = 0.6
    otsu_multiplier: float = 1.15     # NEW: push threshold to catch lighter platelets
    
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
    """Segment cell bodies using Watershed."""
    blurred = filters.gaussian(gray, sigma=params.cell_gaussian_sigma)
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
        if region.area < params.min_cell_area or region.area > params.max_cell_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        cell_sub = gray[minr:maxr, minc:maxc]
        sub_mask = region.image

        intensities = cell_sub[sub_mask]
        if len(intensities) < 30:
            continue

        t = np.percentile(intensities, params.nucleus_dark_percentile)
        nuc_sub = (cell_sub < t) & sub_mask

        nuc_sub = morphology.remove_small_objects(nuc_sub, min_size=params.min_nucleus_area // 2)
        nucleus_mask[minr:maxr, minc:maxc] |= nuc_sub

    nucleus_mask = morphology.opening(nucleus_mask, morphology.disk(1))
    nucleus_mask = morphology.remove_small_objects(nucleus_mask, min_size=params.min_nucleus_area)
    return nucleus_mask


def segment_and_analyze(image: np.ndarray, params: Optional[AnalysisParams] = None) -> Dict[str, Any]:
    if params is None:
        params = AnalysisParams()

    gray = _to_grayscale(image)
    gray, was_inverted = _maybe_invert(gray)
    gray = exposure.rescale_intensity(gray, in_range="image", out_range=(0.0, 1.0))

    cell_mask = _segment_cells(gray, params)
    labeled_cells = measure.label(cell_mask)

    nucleus_mask = _segment_nuclei_inside_cells(gray, cell_mask, labeled_cells, params)
    labeled_nuclei = measure.label(nucleus_mask)

    cell_regions = measure.regionprops(labeled_cells, intensity_image=gray)
    nucleus_regions = measure.regionprops(labeled_nuclei)
    nucleus_props_by_label = {nr.label: nr for nr in nucleus_regions}

    cells = []
    total_vacuole_pixels = 0.0
    total_valid_cell_area = 0.0

    for creg in cell_regions:
        # Filter artifacts
        if creg.area < params.min_cell_area or creg.area > params.max_cell_area:
            continue

        cell_area = float(creg.area)
        perimeter = float(creg.perimeter) if creg.perimeter > 0 else 1.0
        circularity = (4.0 * np.pi * cell_area) / (perimeter * perimeter)
        eccentricity = float(creg.eccentricity)

        # Nucleus calc
        minr, minc, maxr, maxc = creg.bbox
        cell_bbox_mask = labeled_cells[minr:maxr, minc:maxc] == creg.label
        nucleus_area = 0.0
        for nl in np.unique(labeled_nuclei[minr:maxr, minc:maxc]):
            if nl == 0: continue
            nreg = nucleus_props_by_label.get(nl)
            if nreg and (minr <= nreg.centroid[0] < maxr) and (minc <= nreg.centroid[1] < maxc):
                nucleus_area += float(nreg.area)

        cytoplasm_area = max(cell_area - nucleus_area, 1.0)
        nc_ratio = nucleus_area / cytoplasm_area

        # Vacuolization Calculation (Statistical Outlier Method)
        cell_pixels = gray[minr:maxr, minc:maxc][cell_bbox_mask]
        if len(cell_pixels) > 0:
            median_val = np.median(cell_pixels)
            std_val = np.std(cell_pixels)
            diff = np.abs(cell_pixels - median_val)
            # Flag pixels that are dark AND statistical outliers
            vac_px_count = np.sum((diff > 0.20) & (diff > 2.0 * std_val))
            vacuolization_pct = (vac_px_count / cell_area) * 100.0
        else:
            vac_px_count = 0
            vacuolization_pct = 0.0
            
        total_vacuole_pixels += vac_px_count
        total_valid_cell_area += cell_area

        classification, reasons = _classify_morphology(
            nc_ratio, circularity, eccentricity, nucleus_area, cell_area, params
        )

        cells.append({
            "cell_id": int(creg.label),
            "cell_area": round(cell_area, 1),
            "vacuolization_pct": round(vacuolization_pct, 1),
            "bbox": (minr, minc, maxr, maxc),
            "classification": classification,
            "reasons": reasons,
            "circularity": round(circularity, 3),
            "nc_ratio": round(nc_ratio, 3),
            "eccentricity": round(eccentricity, 3)
        })

    cells = sorted(cells, key=lambda c: (c["bbox"][0], c["bbox"][1]))
    for idx, c in enumerate(cells, start=1): c["cell_id"] = idx

    summary = {
        "num_cells": len(cells),
        "total_vacuolization_pct": round(100.0 * total_vacuole_pixels / total_valid_cell_area, 1) if total_valid_cell_area > 0 else 0.0,
        "image_inverted": was_inverted
    }

    return {"cells": cells, "cell_mask": cell_mask, "nucleus_mask": nucleus_mask, "overlay": _create_overlay(image, cell_mask, nucleus_mask), "summary": summary}

def _classify_morphology(nc_ratio, circularity, eccentricity, nucleus_area, cell_area, params):
    reasons = []
    if nc_ratio >= params.nc_ratio_very_high: reasons.append("very high N/C ratio")
    elif nc_ratio >= params.nc_ratio_abnormal: reasons.append("elevated N/C ratio")
    if eccentricity >= params.eccentricity_abnormal: reasons.append("high eccentricity")
    if circularity <= params.circularity_abnormal: reasons.append("low circularity")
    if nucleus_area >= params.nucleus_area_large: reasons.append("enlarged nucleus")
    
    if len(reasons) >= 2 or nc_ratio >= params.nc_ratio_very_high: return "Abnormal (malignant-like)", reasons
    elif reasons: return "Borderline", reasons
    return "Normal morphology", []

def _create_overlay(image, cell_mask, nucleus_mask):
    overlay = image.copy()
    if overlay.dtype != np.uint8: overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    cell_contours, _ = cv2.findContours((cell_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nuc_contours, _ = cv2.findContours((nucleus_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cell_contours, -1, (0, 188, 212), 2)
    cv2.drawContours(overlay, nuc_contours, -1, (233, 30, 99), 2)
    return overlay
