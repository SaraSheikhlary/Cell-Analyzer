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
    .metric-card { background-color: #f8f9fa; border-radius: 8px; padding: 12px 16px; border: 1px solid #e9ecef; }
    .stDataFrame { font-size: 0.9rem; }
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
    color = (255, 215, 0)
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
min_cell_area = st.sidebar.slider("Minimum cell area (pixels)", 150, 1200, 380, step=20)
nucleus_percentile = st.sidebar.slider("Nucleus darkness percentile", 10, 45, 26, step=1)
nc_abnormal = st.sidebar.slider("N/C ratio — Abnormal threshold", 0.40, 0.80, 0.58, step=0.01)
nc_very_high = st.sidebar.slider("N/C ratio — Very high threshold", 0.55, 0.90, 0.72, step=0.01)

if st.sidebar.button("Reset to defaults", use_container_width=True):
    st.rerun()

# ----------------------------- Main Title & Intro ---------------------------
st.title("Cell Morphometry Analyzer")
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
use_synthetic = st.button("✨ Load synthetic demo", type="primary")

raw_image = None
if uploaded_file:
    raw_image = load_image(uploaded_file)
elif use_synthetic:
    raw_image = generate_synthetic_cell_image(width=800, height=600, n_healthy=10, n_abnormal=6, seed=42)

# ----------------------------- Run Workflow ----------------------------------
if raw_image is not None:
    params_dict = {
        "min_cell_area": min_cell_area,
        "nucleus_dark_percentile": float(nucleus_percentile),
        "nc_ratio_abnormal": float(nc_abnormal),
        "nc_ratio_very_high": float(nc_very_high),
    }

    # 1. GLOBAL ANALYSIS (Run immediately for summary)
    with st.spinner("Calculating global statistics..."):
        global_results = run_analysis(raw_image, params_dict)
    
    st.subheader("1. Global Image Summary")
    g1, g2, g3 = st.columns(3)
    g1.metric("Total Cells Detected", global_results["summary"]["num_cells"])
    g2.metric("Flagged Abnormal", f"{global_results['summary']['num_abnormal']} ({global_results['summary']['abnormal_pct']}%)")
    g3.metric("Mean N/C Ratio", global_results["summary"]["mean_nc_ratio"])
    st.markdown("---")

    # 2. QUADRANT SELECTION
    st.subheader("2. Quadrant Pre-Selection")
    q_col1, q_col2 = st.columns([1.5, 1])
    h, w = raw_image.shape[:2]
    
    with q_col1:
        st.image(draw_quadrant_grid(raw_image), use_column_width=True, caption="Full Image Overview")
        
    with q_col2:
        quad_choice = st.radio("Select Region to Analyze:", ["Full Image (Default)", "NW (Top-Left)", "NE (Top-Right)", "SW (Bottom-Left)", "SE (Bottom-Right)"])
        
        # Determine working image
        if quad_choice == "NW (Top-Left)": working_image = raw_image[0:h//2, 0:w//2]
        elif quad_choice == "NE (Top-Right)": working_image = raw_image[0:h//2, w//2:w]
        elif quad_choice == "SW (Bottom-Left)": working_image = raw_image[h//2:h, 0:w//2]
        elif quad_choice == "SE (Bottom-Right)": working_image = raw_image[h//2:h, w//2:w]
        else: working_image = raw_image

    # 3. ANALYSIS FLOW
    if quad_choice == "Full Image (Default)":
        results = global_results
    else:
        results = run_analysis(working_image, params_dict)

    # Display Results
    st.markdown("---")
    st.subheader(f"3. Analysis Results ({quad_choice})")
    
    # ... [Keep your original Metric/Image/Table display code here] ...
    # (Ensure you use results["cells"] and results["overlay"])

    # NOTE: Fix for Pandas .applymap() Error
    # Ensure your styled dataframe line looks like this:
    # styled = display_df.style.map(color_class, subset=["Classification"])
    # st.dataframe(styled, use_container_width=True, hide_index=True)
