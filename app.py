#!/usr/bin/env python3
"""
app.py — Streamlit Web Interface for Cell Morphometry Analysis

A clean, professional tool for quantitative cell and nuclear morphology analysis
with a focus on distinguishing normal vs malignant-like features.

Features
--------
- Upload your own cell images (PNG, JPG, TIFF, etc.)
- Or instantly load a synthetic demo image containing a realistic mix of
  "healthy" and "abnormal" cells (generated on the fly)
- **Quadrant Pre-Selection:** Divide image into NW, NE, SW, SE for targeted analysis.
- Side-by-side view: original image vs color-coded segmentation overlay
  (teal = cell boundaries, magenta = nuclei, with overlay numbers)
- Interactive data table with all extracted metrics per cell
- Explainable rule-based classifier output ("Normal", "Borderline", "Abnormal")
- Single Platelet Zoom inspection to isolate internal structures (Valorization)
- Multiple interactive charts:
    • N/C ratio distribution
    • N/C ratio vs Eccentricity scatter (colored by classification)
    • Circularity comparison
- Adjustable analysis parameters (sidebar)
- One-click CSV download of the full metrics table
- Summary statistics and malignancy-risk indicators
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
    
    # Draw crosshairs
    color = (255, 215, 0) # Gold/Yellow
    thickness = 2
    cv2.line(vis, (w//2, 0), (w//2, h), color, thickness)
    cv2.line(vis, (0, h//2), (w, h//2), color, thickness)
    
    # Add text with dark outline for readability on any background
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
    help="Discard objects smaller than this (removes debris and fragments).",
)

nucleus_percentile = st.sidebar.slider(
    "Nucleus darkness percentile",
    min_value=10,
    max_value=45,
    value=26,
    step=1,
    help="Inside each cell, pixels darker than this percentile are considered nucleus.",
)

st.sidebar.markdown("---")
st.sidebar.caption("Classification thresholds (advanced)")

nc_abnormal = st.sidebar.slider(
    "N/C ratio — Abnormal threshold",
    min_value=0.40,
    max_value=0.80,
    value=0.58,
    step=0.01,
)
nc_very_high = st.sidebar.slider(
    "N/C ratio — Very high threshold",
    min_value=0.55,
    max_value=0.90,
    value=0.72,
    step=0.01,
)

st.sidebar.markdown("---")
if st.sidebar.button("Reset to defaults", use_container_width=True):
    st.rerun()


# ----------------------------- Main Title & Intro ---------------------------
st.title("Cell Morphometry Analyzer")
st.caption("Quantitative extraction of cell & nuclear shape features with targeted region analysis.")

# ----------------------------- Image Source ----------------------------------
uploaded_file = st.file_uploader(
    "Upload a cell image (PNG, JPG, TIFF, etc.)",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    accept_multiple_files=False,
)

use_synthetic = False
raw_image: Optional[np.ndarray] = None
source_label = ""

col_a, col_b = st.columns([1, 1])

with col_a:
    if st.button("✨ Load synthetic demo (healthy + abnormal cells)", type="primary", use_container_width=True):
        use_synthetic = True

with col_b:
    if uploaded_file is None and not use_synthetic:
        st.info("Upload an image above, or load the synthetic demo.", icon="ℹ️")

# Load image
if uploaded_file is not None:
    raw_image = load_image(uploaded_file)
    source_label = f"Uploaded: {uploaded_file.name}"
elif use_synthetic:
    with st.spinner("Generating realistic synthetic cell image..."):
        raw_image = generate_synthetic_cell_image(width=800, height=600, n_healthy=10, n_abnormal=6, seed=42)
    source_label = "Synthetic demo image"

# ----------------------------- Run Workflow ----------------------------------
if raw_image is not None:
    st.markdown(f"**Source:** {source_label}")
    st.markdown("---")
    
    # --- Step 1: Quadrant Selection ---
    st.subheader("1. Quadrant Pre-Selection")
    
    q_col1, q_col2 = st.columns([1.5, 1])
    
    with q_col1:
        grid_overlay = draw_quadrant_grid(raw_image)
        st.image(grid_overlay, use_column_width=True, clamp=True, caption="Full Image Overview")
        
    with q_col2:
        st.markdown("**Select a region to analyze:**")
        quad_choice = st.radio(
            "Region",
            ["Full Image (Default)", "NW (Top-Left)", "NE (Top-Right)", "SW (Bottom-Left)", "SE (Bottom-Right)"],
            label_visibility="collapsed"
        )
        
        # Perform the actual crop based on selection
        h, w = raw_image.shape[:2]
        working_image = raw_image.copy()
        
        if quad_choice == "NW (Top-Left)":
            working_image = raw_image[0:h//2, 0:w//2]
        elif quad_choice == "NE (Top-Right)":
            working_image = raw_image[0:h//2, w//2:w]
        elif quad_choice == "SW (Bottom-Left)":
            working_image = raw_image[h//2:h, 0:w//2]
        elif quad_choice == "SE (Bottom-Right)":
            working_image = raw_image[h//2:h, w//2:w]
            
        st.success(f"Target locked: **{quad_choice}**")
        st.caption(f"Resolution of selected area: {working_image.shape[1]}x{working_image.shape[0]} px")

    st.markdown("---")
    
    # --- Step 2: Analysis ---
    params_dict = {
        "min_cell_area": min_cell_area,
        "nucleus_dark_percentile": float(nucleus_percentile),
        "nc_ratio_abnormal": float(nc_abnormal),
        "nc_ratio_very_high": float(nc_very_high),
    }

    results = run_analysis(working_image, params_dict)

    cells = results["cells"]
    summary = results["summary"]

    # --- Generate Labeled Overlay with High-Contrast Text ---
    labeled_overlay = results["overlay"].copy()
    for c in cells:
        cid = c["cell_id"]
        minr, minc, maxr, maxc = c["bbox"]
        
        # Determine cell center dynamically via bounding box properties
        center_x = minc + (maxc - minc) // 2
        center_y = minr + (maxr - minr) // 2
        text_pos = (center_x - 10, center_y + 5)
        
        # High-contrast render (thicker black backdrop line, followed by sharp white interior)
        cv2.putText(labeled_overlay, str(cid), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(labeled_overlay, str(cid), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    st.subheader(f"2. Analysis Results ({quad_choice})")

    # ====================== SUMMARY METRICS ======================
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cells detected", summary["num_cells"])
    m2.metric("Flagged abnormal", f"{summary['num_abnormal']} ({summary['abnormal_pct']}%)")
    m3.metric("Mean N/C ratio", summary["mean_nc_ratio"])
    m4.metric("Max N/C ratio", summary["max_nc_ratio"])

    if summary["image_inverted"]:
        st.caption("ℹ️ Image was automatically inverted for analysis (bright objects on dark background detected).")

    # ====================== SIDE-BY-SIDE IMAGES ======================
    img_col1, img_col2 = st.columns(2, gap="medium")

    with img_col1:
        st.markdown(f"**Target Area ({quad_choice})**")
        st.image(working_image, use_column_width=True, clamp=True)

    with img_col2:
        st.markdown("**Segmented Overlay**")
        st.image(
            labeled_overlay,
            use_column_width=True,
            clamp=True,
            caption="Teal = cell boundaries | Magenta = nuclei | Numbers = Cell ID Index",
        )

    # ====================== DATA TABLE ======================
    st.subheader("Extracted Metrics per Cell")

    if cells:
        df = pd.DataFrame(cells)

        display_df = df[
            [
                "cell_id",
                "cell_area",
                "nucleus_area",
                "cytoplasm_area",
                "nc_ratio",
                "vacuolization_pct",
                "perimeter",
                "circularity",
                "eccentricity",
                "classification",
            ]
        ].copy()
        
        display_df.columns = [
            "Cell ID",
            "Cell Area",
            "Nucleus Area",
            "Cytoplasm Area",
            "N/C Ratio",
            "vacuolization %",
            "Perimeter",
            "Circularity",
            "Eccentricity",
            "Classification",
        ]

        def color_class(val: str):
            if "Abnormal" in val:
                return "background-color: #ffcccc; font-weight: 600"
            elif "Borderline" in val:
                return "background-color: #fff3cd; font-weight: 500"
            return "background-color: #d4edda"

        styled = display_df.style.map(color_class, subset=["Classification"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=320)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download metrics as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"cell_metrics_{quad_choice.replace(' ', '_')}.csv",
            mime="text/csv",
        )

        # ====================== SINGLE PLATELET ZOOM & ANALYZE ======================
        st.markdown("---")
        st.subheader("🔍 Single Platelet Inspection")

        if cells:
            cell_ids = [c["cell_id"] for c in cells]
            selected_id = st.selectbox("Select Platelet (Cell ID)", options=cell_ids)
            
            selected_cell = next(c for c in cells if c["cell_id"] == selected_id)
            minr, minc, maxr, maxc = selected_cell["bbox"]
            
            pad = 15
            minr_p = max(0, minr - pad)
            minc_p = max(0, minc - pad)
            maxr_p = min(working_image.shape[0], maxr + pad)
            maxc_p = min(working_image.shape[1], maxc + pad)

            zoom_img = working_image[minr_p:maxr_p, minc_p:maxc_p]
            zoom_overlay = labeled_overlay[minr_p:maxr_p, minc_p:maxc_p]

            z_col1, z_col2, z_col3 = st.columns([1.5, 1.5, 1])
            
            with z_col1:
                st.markdown(f"**Zoomed Platelet (ID: {selected_id})**")
                st.image(zoom_img, use_column_width=True, clamp=True)
                
            with z_col2:
                st.markdown("**Segmented Overlay**")
                st.image(zoom_overlay, use_column_width=True, clamp=True)
                
            with z_col3:
                st.markdown("**Specific Metrics**")
                st.metric("vacuolization", f"{selected_cell['vacuolization_pct']}%")
                st.metric("Total Area (pixels)", f"{selected_cell['cell_area']}")
                st.metric("Circularity", f"{selected_cell['circularity']}")

        # ====================== CHARTS ======================
        st.markdown("---")
        st.subheader("Interactive Feature Charts")

        chart_col1, chart_col2 = st.columns(2, gap="large")

        with chart_col1:
            st.markdown("**N/C Ratio Distribution**")
            fig1, ax1 = plt.subplots(figsize=(6, 3.8))
            nc_vals = df["nc_ratio"].values

            ax1.hist(nc_vals, bins=np.linspace(0, max(1.0, nc_vals.max() + 0.05), 18),
                     color="#3498db", edgecolor="white", alpha=0.85)
            ax1.axvline(nc_abnormal, color="#e74c3c", linestyle="--", lw=2, label=f"Abnormal ≥ {nc_abnormal}")
            ax1.axvline(nc_very_high, color="#c0392b", linestyle=":", lw=2, label=f"Very high ≥ {nc_very_high}")
            ax1.set_xlabel("Nucleus-to-Cytoplasm Ratio")
            ax1.set_ylabel("Number of cells")
            ax1.legend(loc="upper right", fontsize=8)
            ax1.grid(True, alpha=0.3)
            fig_to_st(fig1)

        with chart_col2:
            st.markdown("**N/C Ratio vs Eccentricity**")
            fig2, ax2 = plt.subplots(figsize=(6, 3.8))

            colors = {"Normal morphology": "#2ecc71", "Borderline": "#f1c40f", "Abnormal (malignant-like)": "#e74c3c"}

            for cls, grp in df.groupby("classification"):
                ax2.scatter(
                    grp["eccentricity"],
                    grp["nc_ratio"],
                    s=70,
                    c=colors.get(cls, "#7f8c8d"),
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.6,
                    label=cls,
                )

            ax2.axhline(nc_abnormal, color="#e74c3c", linestyle="--", alpha=0.5, lw=1.2)
            ax2.set_xlabel("Eccentricity (0 = round, 1 = elongated)")
            ax2.set_ylabel("N/C Ratio")
            ax2.set_xlim(-0.02, 1.02)
            ax2.legend(loc="upper left", fontsize=8, framealpha=0.95)
            ax2.grid(True, alpha=0.3)
            fig_to_st(fig2)

    else:
        st.warning(f"No cells were detected in the {quad_choice} region. Try adjusting parameters or selecting a different quadrant.")

else:
    st.markdown("---")
    st.markdown("Upload an image to begin quadrant targeting and analysis.")
