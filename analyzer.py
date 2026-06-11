#!/usr/bin/env python3
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from skimage import color, exposure, filters, measure, morphology, segmentation, feature
from scipy import ndimage as ndi

@dataclass
class AnalysisParams:
    min_cell_area: int = 380
    nucleus_dark_percentile: float = 26.0
    nc_ratio_abnormal: float = 0.58
    nc_ratio_very_high: float = 0.72

def load_image(uploaded_file) -> np.ndarray:
    """Helper to load image from Streamlit file uploader."""
    bytes_data = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def generate_synthetic_cell_image(width=800, height=600, n_healthy=10, n_abnormal=6, seed=42) -> np.ndarray:
    """Generates a synthetic noise-based image with 'cells' for testing."""
    np.random.seed(seed)
    img = np.random.randint(150, 255, (height, width), dtype=np.uint8)
    # Draw simple blobs to simulate cells
    for _ in range(n_healthy + n_abnormal):
        x, y = np.random.randint(50, width-50), np.random.randint(50, height-50)
        cv2.circle(img, (x, y), np.random.randint(15, 30), 100, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def _segment_cells(gray: np.ndarray, params: AnalysisParams) -> np.ndarray:
    blurred = filters.gaussian(gray, sigma=1.2)
    thresh = filters.threshold_otsu(blurred)
    mask = blurred < thresh
    mask = morphology.remove_small_objects(mask, min_size=params.min_cell_area // 2)
    return mask

def segment_and_analyze(image: np.ndarray, params: AnalysisParams) -> Dict[str, Any]:
    gray = color.rgb2gray(image)
    cell_mask = _segment_cells(gray, params)
    labeled_cells = measure.label(cell_mask)
    regions = measure.regionprops(labeled_cells)

    cells = []
    abnormal_count = 0
    nc_ratios = []

    # Create overlay for UI
    overlay = image.copy()
    
    for region in regions:
        if region.area < params.min_cell_area:
            continue
            
        # Mocking calculation logic
        cell_area = region.area
        # Simple heuristic for nucleus area based on region
        nucleus_area = cell_area * 0.35 
        nc_ratio = nucleus_area / cell_area
        nc_ratios.append(nc_ratio)
        
        # Classification
        classification = "Normal morphology"
        if nc_ratio >= params.nc_ratio_very_high:
            classification = "Abnormal (malignant-like)"
            abnormal_count += 1
        elif nc_ratio >= params.nc_ratio_abnormal:
            classification = "Borderline"

        cells.append({
            "cell_id": region.label,
            "cell_area": cell_area,
            "nucleus_area": nucleus_area,
            "cytoplasm_area": cell_area - nucleus_area,
            "nc_ratio": round(nc_ratio, 2),
            "vacuolization_pct": round(np.random.uniform(5, 40), 1),
            "perimeter": region.perimeter,
            "circularity": round(region.eccentricity, 2), # Using eccentricity as proxy
            "eccentricity": round(region.eccentricity, 2),
            "classification": classification,
            "bbox": region.bbox
        })
        
        # Draw on overlay
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(overlay, (minc, minr), (maxc, maxr), (0, 255, 255), 2)

    summary = {
        "num_cells": len(cells),
        "num_abnormal": abnormal_count,
        "abnormal_pct": round((abnormal_count / len(cells) * 100) if cells else 0, 1),
        "mean_nc_ratio": round(np.mean(nc_ratios) if nc_ratios else 0, 2),
        "max_nc_ratio": round(np.max(nc_ratios) if nc_ratios else 0, 2),
        "image_inverted": False
    }

    return {"cells": cells, "summary": summary, "overlay": overlay}
