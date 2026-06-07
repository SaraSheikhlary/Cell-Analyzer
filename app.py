#!/usr/bin/env python3
"""
app.py — Cell Morphometry Analyzer (with Global & Quadrant Analysis)
"""

from __future__ import annotations

import io
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analyzer import (
    AnalysisParams,
    generate_synthetic_cell_image,
    load_image,
    segment_and_analyze,
)

st.set_page_config(page_title="Cell Morphometry Analyzer", layout="wide")

# ----------------------------- Cached Analysis -------------------------------
@st.cache_data(show_spinner="Analyzing...", ttl=300)
def run_analysis(image_array: np.ndarray, params_dict: dict) -> dict:
    return segment_and_analyze(image_array, params=params_dict)

# ----------------------------- Sidebar ---------------------------------------
st.sidebar.header("Analysis Parameters")
min_cell_area = st.sidebar.slider("Minimum cell area", 150, 1200, 380)
nucleus_percentile = st.sidebar.slider("Nucleus darkness percentile", 10, 45, 26)

# ----------------------------- Workflow --------------------------------------
st.title("Cell Morphometry Analyzer")
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file is not None or st.button("Load Synthetic Demo"):
    raw_image = load_image(uploaded_file) if uploaded_file else generate_synthetic_cell_image()
    
    params_dict = {"min_cell_area": min_cell_area, "nucleus_dark_percentile": float(nucleus_percentile)}

    # --- 1. Global Scan (Total Count) ---
    with st.spinner("Performing global scan..."):
        global_results = run_analysis(raw_image, params_dict)
    
    st.subheader("1. Global Image Statistics")
    col1, col2 = st.columns(2)
    col1.metric("Total Platelets Detected", global_results["summary"]["num_cells"])
    col2.metric("Global Abnormal Rate", f"{global_results['summary']['abnormal_pct']}%")
    
    st.markdown("---")

    # --- 2. Quadrant Selection ---
    st.subheader("2. Quadrant Targeting")
    q_col1, q_col2 = st.columns([1, 1])
    
    with q_col1:
        # Create visual grid
        vis = raw_image.copy()
        h, w = vis.shape[:2]
        cv2.line(vis, (w//2, 0), (w//2, h), (255, 215, 0), 3)
        cv2.line(vis, (0, h//2), (w, h//2), (255, 215, 0), 3)
        st.image(vis, caption="Select target quadrant")

    with q_col2:
        quad_choice = st.radio("Choose Region:", ["Full Image", "NW", "NE", "SW", "SE"])
        
        # Determine crop
        if quad_choice == "Full Image":
            working_image = raw_image
        else:
            h, w = raw_image.shape[:2]
            if quad_choice == "NW": working_image = raw_image[0:h//2, 0:w//2]
            elif quad_choice == "NE": working_image = raw_image[0:h//2, w//2:w]
            elif quad_choice == "SW": working_image = raw_image[h//2:h, 0:w//2]
            elif quad_choice == "SE": working_image = raw_image[h//2:h, w//2:w]

    # --- 3. Targeted Analysis ---
    if quad_choice != "Full Image":
        results = run_analysis(working_image, params_dict)
        st.info(f"Targeted Region Analysis: {quad_choice} ({results['summary']['num_cells']} platelets found)")
    else:
        results = global_results

    # --- 4. Display Results (Same as before) ---
    st.image(results["overlay"], use_column_width=True)
    st.dataframe(pd.DataFrame(results["cells"]))
