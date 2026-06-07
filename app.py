#!/usr/bin/env python3
"""
app.py — Integrated Cell Morphometry Analyzer
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

st.set_page_config(page_title="Cell Morphometry Analyzer", layout="wide", initial_sidebar_state="expanded")

# --- Styling ---
st.markdown("""<style>.main .block-container { padding-top: 1.2rem; }</style>""", unsafe_allow_html=True)

# --- Helpers ---
@st.cache_data(show_spinner="Analyzing...", ttl=300)
def run_analysis(image_array: np.ndarray, params_dict: dict) -> dict:
    return segment_and_analyze(image_array, params=AnalysisParams(**params_dict))

def draw_quadrant_grid(img_array: np.ndarray) -> np.ndarray:
    vis = img_array.copy()
    h, w = vis.shape[:2]
    cv2.line(vis, (w//2, 0), (w//2, h), (255, 215, 0), 3)
    cv2.line(vis, (0, h//2), (w, h//2), (255, 215, 0), 3)
    return vis

# --- Sidebar ---
st.sidebar.header("Analysis Parameters")
min_cell_area = st.sidebar.slider("Min cell area", 150, 1200, 380)
nucleus_percentile = st.sidebar.slider("Nucleus darkness %", 10, 45, 26)

# --- Main App ---
st.title("Cell Morphometry Analyzer")
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif"])

raw_image = None
if st.button("✨ Load synthetic demo"):
    raw_image = generate_synthetic_cell_image()
elif uploaded_file:
    raw_image = load_image(uploaded_file)

if raw_image is not None:
    params_dict = {"min_cell_area": min_cell_area, "nucleus_dark_percentile": float(nucleus_percentile)}

    # 1. GLOBAL ANALYSIS (Run immediately)
    with st.spinner("Performing global image scan..."):
        global_results = run_analysis(raw_image, params_dict)
    
    st.subheader("1. Global Image Statistics")
    m1, m2 = st.columns(2)
    m1.metric("Total Platelets (Whole Image)", global_results["summary"]["num_cells"])
    m2.metric("Global Abnormal Rate", f"{global_results['summary']['abnormal_pct']}%")
    st.markdown("---")

    # 2. QUADRANT SELECTION
    st.subheader("2. Target Region")
    q_col1, q_col2 = st.columns([1.5, 1])
    with q_col1:
        st.image(draw_quadrant_grid(raw_image), use_column_width=True, caption="Full Image Overview")
    with q_col2:
        quad_choice = st.radio("Select region:", ["Full Image", "NW", "NE", "SW", "SE"])
        
        h, w = raw_image.shape[:2]
        if quad_choice == "NW": working_img = raw_image[0:h//2, 0:w//2]
        elif quad_choice == "NE": working_img = raw_image[0:h//2, w//2:w]
        elif quad_choice == "SW": working_img = raw_image[h//2:h, 0:w//2]
        elif quad_choice == "SE": working_img = raw_image[h//2:h, w//2:w]
        else: working_img = raw_image

    # 3. ANALYSIS
    if quad_choice != "Full Image":
        results = run_analysis(working_img, params_dict)
    else:
        results = global_results
        working_img = raw_image

    # 4. RESULTS DISPLAY
    st.subheader(f"3. Results ({quad_choice})")
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected", results["summary"]["num_cells"])
    c2.metric("Abnormal", f"{results['summary']['num_abnormal']}")
    c3.metric("Mean N/C", results["summary"]["mean_nc_ratio"])
    c4.metric("Max N/C", results["summary"]["max_nc_ratio"])

    # Visuals
    i1, i2 = st.columns(2)
    i1.image(working_img, caption="Target Area", use_column_width=True)
    i2.image(results["overlay"], caption="Segmentation", use_column_width=True)

    # DataFrame (FIXED: using .map() instead of .applymap())
    st.subheader("Metrics Table")
    df = pd.DataFrame(results["cells"])
    def color_class(val):
        return "background-color: #ffcccc" if "Abnormal" in str(val) else ""
    
    # Note: .map() is the modern replacement for applymap
    st.dataframe(df.style.map(color_class, subset=["classification"]), use_container_width=True)

    # Zoom Section
    st.subheader("🔍 Single Platelet Inspection")
    if results["cells"]:
        ids = [c["cell_id"] for c in results["cells"]]
        s_id = st.selectbox("Select ID", ids)
        s_cell = next(c for c in results["cells"] if c["cell_id"] == s_id)
        
        minr, minc, maxr, maxc = s_cell["bbox"]
        pad = 15
        zoom = working_img[max(0, minr-pad):min(working_img.shape[0], maxr+pad), 
                           max(0, minc-pad):min(working_img.shape[1], maxc+pad)]
        st.image(zoom, caption=f"ID: {s_id}")
        st.metric("Valorization %", f"{s_cell['valorization_pct']}%")

    # Charts
    st.subheader("Feature Charts")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(df["nc_ratio"], bins=10)
    ax[0].set_title("N/C Ratio")
    ax[1].scatter(df["eccentricity"], df["nc_ratio"])
    ax[1].set_title("Eccentricity vs N/C")
    st.pyplot(fig)
