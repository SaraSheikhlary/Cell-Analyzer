#!/usr/bin/env python3
"""
analyzer.py — Cell Morphometry Analysis Backend

Image processing pipeline for quantitative analysis of cell and nuclear morphology.
Designed for brightfield, H&E, or fluorescence microscopy images of cells.

Core capabilities:
- Robust cell segmentation (Watershed + morphology, auto-inversion detection)
- Nucleus segmentation inside detected cells (adaptive darkness threshold)
- Extraction of clinically relevant morphometric features:
    * Cell area, nucleus area, cytoplasm area
    * Cell perimeter
    * Circularity (4πA/P²)
    * Eccentricity (from second moments)
    * Nucleus-to-cytoplasm (N/C) area ratio
    * Valorization/Vacuolization percentage (internal structure variance)
- Lightweight, fully explainable rule-based classifier for "normal" vs "abnormal"
  morphology (no machine learning required as fallback).

The module also provides a high-quality synthetic image generator so the
Streamlit app (and tests) can run immediately without real user images.

Dependencies: numpy, opencv-python-headless, scikit-image, scipy, Pillow
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
    min_cell_area: int = 380          # pixels — discard tiny debris
    min_nucleus_area: int = 65        # pixels
    nucleus_dark_percentile: float = 26.0   # inside each cell, take darkest X%
    cell_gaussian_sigma: float = 1.2
    nucleus_gaussian_sigma: float = 0.6
    # Classification thresholds (tuned for typical 20-60x cell images)
    nc_ratio_abnormal: float = 0.58
    nc_ratio_very_high: float = 0.72
    eccentricity_abnormal: float = 0.74
    circularity_abnormal: float = 0.58
    nucleus_area_large: float = 520.0


# ----------------------------- Image Loading ---------------------------------
def load_image(source: ImageLike) -> np.ndarray:
    """
    Load an image from multiple possible sources and return RGB uint8 array.

    Accepts:
    - filesystem path (str)
    - bytes / BytesIO (e.g. Streamlit upload)
    - PIL Image
    - numpy array (RGB, RGBA, or grayscale)
    """
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
        elif arr.shape[2] == 4:  # RGBA
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return arr

    raise TypeError(f"Unsupported image source type: {type(source)}")


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB to float32 grayscale in [0, 1]."""
    if img.ndim == 2:
        gray = img.astype(np.float32) / 255.0
    else:
        gray = color.rgb2gray(img)
    return gray.astype(np.float32)


def _maybe_invert(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Heuristic: if the "dark" objects (putative cells/nuclei) occupy the
    brighter part of the histogram, invert the image. This makes the
    pipeline robust to both fluorescence (bright objects) and brightfield/H&E.
    """
    # Simple bimodal test using Otsu on both versions
    p1, p2 = np.percentile(gray, [2, 98])
    if p2 - p1 < 0.05:
        return gray, False

    t = filters.threshold_otsu(gray)
    dark_fraction = (gray < t).mean()

    # If dark pixels are a minority but very bright, likely need inversion
    bright_fraction = (gray > t).mean()
    if bright_fraction > 0.35 and dark_fraction < 0.25:
        return 1.0 - gray, True
    return gray, False


# ----------------------------- Segmentation ----------------------------------
def _segment_cells(gray: np.ndarray, params: AnalysisParams) -> np.ndarray:
    """Segment cell bodies using Watershed to split touching platelets."""
    blurred = filters.gaussian(gray, sigma=params.cell_gaussian_sigma)
    thresh = filters.threshold_otsu(blurred)
    
    # Initial binary mask
    mask = blurred < thresh

    # Morphological cleanup
    mask = morphology.remove_small_objects(mask, min_size=params.min_cell_area // 2)
    mask = morphology.remove_small_holes(mask, area_threshold=200)
    mask = morphology.closing(mask, morphology.disk(2))

    # --- WATERSHED SEGMENTATION ---
    # 1. Calculate the distance from the edge of the platelet to its center
    distance = ndi.distance_transform_edt(mask)
    
    # 2. Find the peaks (the absolute centers of each platelet)
    # min_distance prevents creating two centers inside one slightly oblong platelet
    coords = feature.peak_local_max(distance, min_distance=15, labels=mask)
    
    # 3. Create markers at those peak locations
    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = ndi.label(markers)
    
    # 4. Run watershed to separate the touching regions
    labeled_platelets = segmentation.watershed(-distance, markers, mask=mask)
    
    # 5. Carve 1-pixel boundaries between touching platelets so the 
    # downstream code recognizes them as separate objects
    boundaries = segmentation.find_boundaries(labeled_platelets, mode='inner')
    mask[boundaries] = False

    # Final cleanup of any dust created by the boundary splitting
    mask = morphology.remove_small_objects(mask, min_size=params.min_cell_area)
    
    return mask


def _segment_nuclei_inside_cells(
    gray: np.ndarray, cell_mask: np.ndarray, labeled_cells: np.ndarray, params: AnalysisParams
) -> np.ndarray:
    """
    For each detected cell, find the darker sub-region as nucleus.
    Returns a boolean nucleus mask aligned with the original image.
    """
    nucleus_mask = np.zeros_like(cell_mask, dtype=bool)
    regions = measure.regionprops(labeled_cells)

    for region in regions:
        if region.area < params.min_cell_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        cell_sub = gray[minr:maxr, minc:maxc]
        sub_mask = region.image  # boolean mask of this cell inside bbox

        # Only consider pixels that actually belong to this cell
        intensities = cell_sub[sub_mask]
        if len(intensities) < 30:
            continue

        # Nucleus = darkest X% of pixels inside this cell
        t = np.percentile(intensities, params.nucleus_dark_percentile)
        nuc_sub = (cell_sub < t) & sub_mask

        # Clean small speckles inside the cell
        nuc_sub = morphology.remove_small_objects(nuc_sub, min_size=params.min_nucleus_area // 2)
        nucleus_mask[minr:maxr, minc:maxc] |= nuc_sub

    # Global morphological cleanup on the assembled nucleus mask
    nucleus_mask = morphology.opening(nucleus_mask, morphology.disk(1))
    nucleus_mask = morphology.remove_small_objects(nucleus_mask, min_size=params.min_nucleus_area)
    return nucleus_mask


def segment_and_analyze(
    image: np.ndarray, params: Optional[AnalysisParams] = None
) -> Dict[str, Any]:
    """
    Main entry point. Runs the full pipeline and returns metrics + visualizations.

    Returns dict with keys:
        cells: list of per-cell feature dictionaries
        cell_mask, nucleus_mask: boolean arrays
        overlay: RGB visualization with colored boundaries
        summary: aggregate statistics
        params_used: the AnalysisParams that were applied
    """
    if params is None:
        params = AnalysisParams()

    # --- Preprocessing ---
    gray = _to_grayscale(image)
    gray, was_inverted = _maybe_invert(gray)

    # Contrast enhancement helps on low-dynamic-range uploads
    gray = exposure.rescale_intensity(gray, in_range="image", out_range=(0.0, 1.0))

    # --- Cell segmentation ---
    cell_mask = _segment_cells(gray, params)
    labeled_cells = measure.label(cell_mask)
    n_cells_raw = labeled_cells.max()

    # --- Nucleus segmentation (constrained inside cells) ---
    nucleus_mask = _segment_nuclei_inside_cells(gray, cell_mask, labeled_cells, params)
    labeled_nuclei = measure.label(nucleus_mask)

    # --- Feature extraction using regionprops ---
    cell_regions = measure.regionprops(labeled_cells, intensity_image=gray)
    nucleus_regions = measure.regionprops(labeled_nuclei)

    # Build a quick nucleus lookup by centroid for assignment
    nucleus_props_by_label = {nr.label: nr for nr in nucleus_regions}

    cells: List[Dict[str, Any]] = []

    for creg in cell_regions:
        if creg.area < params.min_cell_area:
            continue

        # Cell shape features (directly from regionprops where possible)
        cell_area = float(creg.area)
        perimeter = float(creg.perimeter) if creg.perimeter > 0 else 1.0
        circularity = (4.0 * np.pi * cell_area) / (perimeter * perimeter)
        eccentricity = float(creg.eccentricity)

        # Find nuclei whose centroid lies inside this cell
        minr, minc, maxr, maxc = creg.bbox
        cell_bbox_mask = labeled_cells[minr:maxr, minc:maxc] == creg.label

        nucleus_area = 0.0
        for nl in np.unique(labeled_nuclei[minr:maxr, minc:maxc]):
            if nl == 0:
                continue
            nreg = nucleus_props_by_label.get(nl)
            if nreg is None:
                continue
            # Centroid test (in global coords)
            cy, cx = nreg.centroid
            if (minr <= cy < maxr) and (minc <= cx < maxc):
                # Verify the nucleus centroid is actually inside this cell's mask
                local_y = int(cy - minr)
                local_x = int(cx - minc)
                if 0 <= local_y < cell_bbox_mask.shape[0] and 0 <= local_x < cell_bbox_mask.shape[1]:
                    if cell_bbox_mask[local_y, local_x]:
                        nucleus_area += float(nreg.area)

        cytoplasm_area = max(cell_area - nucleus_area, 1.0)
        nc_ratio = nucleus_area / cytoplasm_area

        # --- NEW: Calculate % of Valorization/Vacuolization ---
        # We calculate the variance/spread of pixels inside the cell to quantify internal structures
        cell_pixels = gray[minr:maxr, minc:maxc][cell_bbox_mask]
        if len(cell_pixels) > 0:
            median_val = np.median(cell_pixels)
            vacuolization_pixels = np.sum(np.abs(cell_pixels - median_val) > 0.15) 
            vacuolization_pct = (vacuolization_pixels / cell_area) * 100.0
        else:
            vacuolization_pct = 0.0

        # Classification
        classification, reasons = _classify_morphology(
            nc_ratio=nc_ratio,
            circularity=circularity,
            eccentricity=eccentricity,
            nucleus_area=nucleus_area,
            cell_area=cell_area,
            params=params,
        )

        cells.append(
            {
                "cell_id": int(creg.label),
                "cell_area": round(cell_area, 1),
                "nucleus_area": round(nucleus_area, 1),
                "cytoplasm_area": round(cytoplasm_area, 1),
                "nc_ratio": round(nc_ratio, 3),
                "perimeter": round(perimeter, 1),
                "circularity": round(circularity, 3),
                "eccentricity": round(eccentricity, 3),
                "vacuolization_pct": round(vacuolization_pct, 1), # NEW METRIC
                "bbox": (minr, minc, maxr, maxc), # NEW: Bounding box for zooming
                "classification": classification,
                "reasons": reasons,
            }
        )

    # Optional sort so ID numbers flow logically from top-left to bottom-right
    cells = sorted(cells, key=lambda c: (c["bbox"][0], c["bbox"][1]))
    for idx, c in enumerate(cells, start=1):
        c["cell_id"] = idx

    # --- Visualizations ---
    overlay = _create_overlay(image, cell_mask, nucleus_mask)

    # Summary statistics
    if cells:
        nc_values = np.array([c["nc_ratio"] for c in cells])
        abnormal_count = sum(1 for c in cells if "Abnormal" in c["classification"])
        summary = {
            "num_cells": len(cells),
            "num_abnormal": abnormal_count,
            "abnormal_pct": round(100.0 * abnormal_count / len(cells), 1),
            "mean_nc_ratio": round(float(nc_values.mean()), 3),
            "median_nc_ratio": round(float(np.median(nc_values)), 3),
            "max_nc_ratio": round(float(nc_values.max()), 3),
            "image_inverted": was_inverted,
        }
    else:
        summary = {
            "num_cells": 0,
            "num_abnormal": 0,
            "abnormal_pct": 0.0,
            "mean_nc_ratio": 0.0,
            "median_nc_ratio": 0.0,
            "max_nc_ratio": 0.0,
            "image_inverted": was_inverted,
        }

    return {
        "cells": cells,
        "cell_mask": cell_mask,
        "nucleus_mask": nucleus_mask,
        "overlay": overlay,
        "summary": summary,
        "params_used": params,
    }


# ----------------------------- Classification --------------------------------
def _classify_morphology(
    nc_ratio: float,
    circularity: float,
    eccentricity: float,
    nucleus_area: float,
    cell_area: float,
    params: AnalysisParams,
) -> Tuple[str, List[str]]:
    """
    Lightweight, transparent rule-based classifier.

    Returns (label, list_of_triggered_reasons).
    Designed as a fallback when no trained ML model is available.
    """
    reasons: List[str] = []

    if nc_ratio >= params.nc_ratio_very_high:
        reasons.append("very high N/C ratio")
    elif nc_ratio >= params.nc_ratio_abnormal:
        reasons.append("elevated N/C ratio")

    if eccentricity >= params.eccentricity_abnormal:
        reasons.append("high eccentricity (elongated/irregular)")

    if circularity <= params.circularity_abnormal:
        reasons.append("low circularity (atypical shape)")

    if nucleus_area >= params.nucleus_area_large:
        reasons.append("enlarged nucleus")

    # Decision logic
    if len(reasons) >= 2 or nc_ratio >= params.nc_ratio_very_high:
        return "Abnormal (malignant-like)", reasons
    elif reasons:
        return "Borderline", reasons
    else:
        return "Normal morphology", []


# ----------------------------- Visualization ---------------------------------
def _create_overlay(
    image: np.ndarray, cell_mask: np.ndarray, nucleus_mask: np.ndarray
) -> np.ndarray:
    """Create a nice color overlay with cell (teal) and nucleus (magenta) boundaries."""
    overlay = image.copy()
    if overlay.dtype != np.uint8:
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    # Find contours (OpenCV expects uint8 single-channel)
    cell_u8 = (cell_mask.astype(np.uint8) * 255)
    nuc_u8 = (nucleus_mask.astype(np.uint8) * 255)

    cell_contours, _ = cv2.findContours(cell_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nuc_contours, _ = cv2.findContours(nuc_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Teal-ish for cells, magenta for nuclei
    cv2.drawContours(overlay, cell_contours, -1, (0, 188, 212), 2)      # cell boundaries
    cv2.drawContours(overlay, nuc_contours, -1, (233, 30, 99), 2)      # nucleus boundaries

    # Light transparent fill for nuclei
    nuc_color = np.array([233, 30, 99], dtype=np.uint8)
    overlay[nucleus_mask] = (0.55 * overlay[nucleus_mask] + 0.45 * nuc_color).astype(np.uint8)

    return overlay


# ----------------------------- Synthetic Data Generator ----------------------
def generate_synthetic_cell_image(
    width: int = 640,
    height: int = 480,
    n_healthy: int = 6,
    n_abnormal: int = 4,
    seed: int = 123,
) -> np.ndarray:
    """
    Generate a realistic-looking synthetic RGB image containing a mixture of
    "healthy" and "abnormal/malignant-like" cells.
    """
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 248, dtype=np.uint8)  # very light background

    def draw_cell(
        cy: float,
        cx: float,
        ry: float,
        rx: float,
        angle_deg: float,
        cyto_color: Tuple[int, int, int],
        nucleus_ry: float,
        nucleus_rx: float,
        nucleus_color: Tuple[int, int, int],
    ):
        """Draw one cell (cytoplasm ellipse + darker nucleus ellipse)."""
        # Cytoplasm
        rr, cc = draw.ellipse(int(cy), int(cx), int(ry), int(rx), rotation=np.deg2rad(angle_deg), shape=img.shape[:2])
        for i in range(3):
            img[rr, cc, i] = np.clip(
                cyto_color[i] + rng.integers(-12, 13, size=len(rr)), 0, 255
            )

        # Nucleus (smaller, darker, slightly offset for realism)
        n_cy = cy + rng.uniform(-ry * 0.08, ry * 0.08)
        n_cx = cx + rng.uniform(-rx * 0.08, rx * 0.08)
        n_ry = max(3, nucleus_ry)
        n_rx = max(3, nucleus_rx)
        rr_n, cc_n = draw.ellipse(int(n_cy), int(n_cx), int(n_ry), int(n_rx), rotation=np.deg2rad(angle_deg), shape=img.shape[:2])
        for i in range(3):
            img[rr_n, cc_n, i] = np.clip(
                nucleus_color[i] + rng.integers(-8, 9, size=len(rr_n)), 0, 255
            )

    # Draw cells
    for _ in range(n_healthy):
        cy, cx = rng.uniform(40, height - 40), rng.uniform(40, width - 40)
        ry, rx = rng.uniform(15, 25), rng.uniform(15, 25)
        draw_cell(
            cy, cx, ry, rx, rng.uniform(0, 360), (180, 190, 220),
            ry * rng.uniform(0.3, 0.45), rx * rng.uniform(0.3, 0.45), (80, 70, 130)
        )

    for _ in range(n_abnormal):
        cy, cx = rng.uniform(50, height - 50), rng.uniform(50, width - 50)
        ry, rx = rng.uniform(25, 40), rng.uniform(12, 20)  # elongated
        draw_cell(
            cy, cx, ry, rx, rng.uniform(0, 360), (160, 170, 200),
            ry * rng.uniform(0.7, 0.9), rx * rng.uniform(0.7, 0.9), (60, 40, 100) # large nuclei
        )

    # Add Gaussian blur and noise
    img_float = img.astype(np.float32) / 255.0
    img_float = filters.gaussian(img_float, sigma=0.8, channel_axis=-1)
    img_float = util.random_noise(img_float, mode="gaussian", var=0.001, rng=rng)
    
    return (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
