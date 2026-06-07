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
    generate_synthetic_cell_image,
    load_image,
    segment_and_analyze,
)

# ----------------------------- Page Configuration ----------------------------
st.set_page_config(page_title="Cell Morphometry Analyzer", layout="wide", initial_sidebar_state="expanded")

# ----------------------------- Custom Styling --------------------------------
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Cached Analysis -------------------------------
@st.cache_data(show_spinner="Analyzing image...", ttl=300)
def run_analysis(image_array: np.ndarray, params_dict: dict) -> dict:
    params = AnalysisParams(**params_dict)
    return segment_and_analyze(image_array, params=params)

def fig_to_st(fig: plt.Figure):
    st.pyplot(fig, use_container_width=True, clear_figure=True)

def draw_quadrant_grid(img_array: np.ndarray) -> np.ndarray:
    vis = img_array.copy()
    h, w = vis.shape[:2]
    cv2.line(vis, (w//2, 0), (w//2, h), (255, 215, 0), 3)
    cv2.line(vis, (0, h//2), (w, h//2), (255, 215, 0), 3)
    return vis

# ----------------------------- Sidebar Controls ------------------------------
st.sidebar.title("🔬 Cell Morphometry")
min_cell_area = st.sidebar.slider("Minimum cell area (pixels)", 150, 1200, 380)
nucleus_percentile = st.sidebar.slider("Nucleus darkness percentile", 10, 45, 26)

# ----------------------------- Main Logic ------------------------------------
st.title("Cell Morphometry Analyzer")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif"])
if st.button("✨ Load synthetic demo"):
    raw_image = generate_synthetic_cell_image()
    st.session_state["raw_image"] = raw_image
elif uploaded_file:
    raw_image = load_image(uploaded_file)
    st.session_state["raw_image"] = raw_image

if "raw_image" in st.session_state:
    raw_image = st.session_state["raw_image"]
    
    # 1. PARAMETERS
    params_dict = {
        "min_cell_area": min_cell_area,
        "nucleus_dark_percentile": float(nucleus_percentile),
    }

    # 2. GLOBAL ANALYSIS (Total Platelets)
    with st.spinner("Calculating global statistics..."):
        global_results = run_analysis(raw_image, params_dict)
    
    st.subheader("1. Global Image Statistics")
    m1, m2 = st.columns(2)
    m1.metric("Total Platelets Detected (Whole Image)", global_results["summary"]["num_cells"])
    m2.metric("Global Abnormal Rate", f"{global_results['summary']['abnormal_pct']}%")
    st.markdown("---")

    # 3. QUADRANT SELECTION
    st.subheader("2. Quadrant Targeting")
    q_col1, q_col2 = st.columns([1.5, 1])
    with q_col1:
        st.image(draw_quadrant_grid(raw_image), use_column_width=True, caption="Full Image Overview")
    with q_col2:
        quad_choice = st.radio("Select Region to Analyze:", ["Full Image", "NW", "NE", "SW", "SE"])
        
        h, w = raw_image.shape[:2]
        if quad_choice == "NW": working_img = raw_image[0:h//2, 0:w//2]
        elif quad_choice == "NE": working_img = raw_image[0:h//2, w//2:w]
        elif quad_choice == "SW": working_img = raw_image[h//2:h, 0:w//2]
        elif quad_choice == "SE": working_img = raw_image[h//2:h, w//2:w]
        else: working_img = raw_image

    # 4. TARGETED ANALYSIS
    if quad_choice != "Full Image":
        with st.spinner(f"Analyzing {quad_choice} quadrant..."):
            results = run_analysis(working_img, params_dict)
            st.success(f"Region {quad_choice} Analysis: {results['summary']['num_cells']} platelets found.")
    else:
        results = global_results

    # 5. DISPLAY RESULTS
    st.image(results["overlay"], use_column_width=True)
    
    # --- Data Table & Zoom (Same as before) ---
    cells = results["cells"]
    df = pd.DataFrame(cells)
    st.dataframe(df, use_container_width=True)

    if cells:
        st.subheader("🔍 Single Platelet Inspection")
        selected_id = st.selectbox("Select Platelet (ID)", options=[c["cell_id"] for c in cells])
        selected_cell = next(c for c in cells if c["cell_id"] == selected_id)
        
        minr, minc, maxr, maxc = selected_cell["bbox"]
        pad = 15
        zoom_img = working_img[max(0, minr-pad):min(working_img.shape[0], maxr+pad), 
                               max(0, minc-pad):min(working_img.shape[1], maxc+pad)]
        
        z1, z2 = st.columns(2)
        z1.image(zoom_img, caption="Zoomed View")
        z2.metric("Valorization %", f"{selected_cell['valorization_pct']}%")
