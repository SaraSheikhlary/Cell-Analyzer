#!/usr/bin/env python3
"""
app.py — Streamlit Web Interface for Cell Morphometry Analysis
"""

from __future__ import annotations

import io
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage

from analyzer import (
    AnalysisParams,
    load_image,
    segment_and_analyze,
)

# ----------------------------- Page Configuration ----------------------------
st.set_page_config(
    page_title="Cell Morphometry Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Custom Styling --------------------------------
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.2rem; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #e9ecef;
    }
    .stDataFrame { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Cached Analysis -------------------------------
@st.cache_data(show_spinner="Analyzing image...", ttl=300)
def run_analysis(image_array: np.ndarray, params_dict: dict) -> dict:
    """Cached wrapper around the heavy analysis pipeline."""
    params = AnalysisParams(**params_dict)
    return segment_and_analyze(image_array, params=params)

def fig_to_st(fig: plt.Figure):
    """Helper to display matplotlib figures nicely."""
    st.pyplot(fig, use_container_width=True, clear_figure=True)

def draw_quadrant_grid(img_array: np.ndarray) -> np.ndarray:
    """Draws a target grid with NW, NE, SW, SE labels for visual selection."""
    vis = img_array.copy()
    h, w = vis.shape[:2]
    
    color = (255, 215, 0) # Gold/Yellow
    thickness = 2
    cv2.line(vis, (w//2, 0), (w//2, h), color, thickness)
    cv2.line(vis, (0, h//2), (w, h//2), color, thickness)
    
    def put_text(text, x, y):
        cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
    put_text("NW", int(w*0.25) - 20, int(h*0.25))
    put_text("NE", int(w*0.75) - 20, int(h*0.25))
    put_text("SW", int(w*0.25) - 20, int(h*0.75))
    put_text("SE", int(w*0.75) - 20, int(h*0.75))
    
    return vis

# ----------------------------- Sidebar Controls ------------------------------
st.sidebar.title("🔬 Cell Morphometry")
st.sidebar.markdown("**Malignant vs Normal Feature Extraction**")

st.sidebar.header("Analysis Parameters")

min_cell_area = st.sidebar.slider(
    "Minimum cell area (pixels)",
    min_value=150,
    max_value=1200,
    value=380,
    step=20,
)

nucleus_percentile = st.sidebar.slider(
    "Nucleus darkness percentile",
    min_value=10,
    max_value=45,
    value=26,
    step=1,
)

max_bg_intensity = st.sidebar.slider(
    "Max Background Intensity",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Filter out bright background noise. Lower this to be more strict."
)

st.sidebar.markdown("---")
st.sidebar.caption("Classification thresholds (advanced)")

nc_abnormal = st.sidebar.slider("N/C ratio — Abnormal", 0.40, 0.80, 0.58, 0.01)
nc_very_high = st.sidebar.slider("N/C ratio — Very high", 0.55, 0.90, 0.72, 0.01)

st.sidebar.markdown("---")
if st.sidebar.button("Reset to defaults", use_container_width=True):
    st.rerun()

# ----------------------------- Main Title & Intro ---------------------------
st.title("Cell Morphometry Analyzer")

# ----------------------------- Image Source ----------------------------------
uploaded_file = st.file_uploader(
    "Upload a cell image",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
)

raw_image: Optional[np.ndarray] = None
source_label = ""

if uploaded_file is not None:
    raw_image = load_image(uploaded_file)
    source_label = f"Uploaded: {uploaded_file.name}"
else:
    st.info("Upload an image above to begin analysis.")

# ----------------------------- Run Workflow ----------------------------------
if raw_image is not None:
    st.markdown(f"**Source:** {source_label}")
    
    # --- Step 1: Quadrant Selection ---
    st.subheader("1. Quadrant Pre-Selection")
    q_col1, q_col2 = st.columns([1.5, 1])
    
    with q_col1:
        grid_overlay = draw_quadrant_grid(raw_image)
        st.image(grid_overlay, use_column_width=True, clamp=True)
        
    with q_col2:
        quad_choice = st.radio("Region", ["Full Image (Default)", "NW (Top-Left)", "NE (Top-Right)", "SW (Bottom-Left)", "SE (Bottom-Right)"])
        h, w = raw_image.shape[:2]
        working_image = raw_image.copy()
        if quad_choice == "NW (Top-Left)": working_image = raw_image[0:h//2, 0:w//2]
        elif quad_choice == "NE (Top-Right)": working_image = raw_image[0:h//2, w//2:w]
        elif quad_choice == "SW (Bottom-Left)": working_image = raw_image[h//2:h, 0:w//2]
        elif quad_choice == "SE (Bottom-Right)": working_image = raw_image[h//2:h, w//2:w]

    # --- Step 2: Analysis ---
    params_dict = {
        "min_cell_area": min_cell_area,
        "nucleus_dark_percentile": float(nucleus_percentile),
        "max_mean_intensity": float(max_bg_intensity),
        "nc_ratio_abnormal": float(nc_abnormal),
        "nc_ratio_very_high": float(nc_very_high),
    }

    results = run_analysis(working_image, params_dict)
    cells = results["cells"]
    summary = results["summary"]

    # Overlay rendering
    labeled_overlay = results["overlay"].copy()
    for c in cells:
        minr, minc, maxr, maxc = c["bbox"]
        center_x = minc + (maxc - minc) // 2
        center_y = minr + (maxr - minr) // 2
        cv2.putText(labeled_overlay, str(c["cell_id"]), (center_x-10, center_y+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
        cv2.putText(labeled_overlay, str(c["cell_id"]), (center_x-10, center_y+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    st.subheader(f"2. Analysis Results")
    
    # Display logic
    img_col1, img_col2 = st.columns(2)
    with img_col1: st.image(working_image, caption="Target", use_column_width=True)
    with img_col2: st.image(labeled_overlay, caption="Overlay", use_column_width=True)

    # Data Table
    if cells:
        df = pd.DataFrame(cells)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No cells detected. Try adjusting the parameters in the sidebar.")
