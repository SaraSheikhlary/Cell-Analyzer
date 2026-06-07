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
- Side-by-side view: original image vs color-coded segmentation overlay
  (teal = cell boundaries, magenta = nuclei)
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

When no image is provided, the synthetic demo allows immediate exploration
without requiring any external data.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
from typing import Optional

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
st.caption("Quantitative extraction of cell & nuclear shape features with explainable abnormal morphology flagging")

with st.expander("How it works & What the metrics mean", expanded=False):
    st.markdown(
        """
        **Pipeline**
        1. Image is converted to grayscale and (if needed) inverted so that cells/nuclei appear dark.
        2. Global Otsu thresholding + morphological cleaning segments individual **cells**.
        3. Inside each cell, the darkest pixels (user-controlled percentile) are labeled as **nucleus**.
        4. For every cell we compute the requested morphometric features.
        5. A transparent rule-based classifier flags cells as *Normal*, *Borderline*, or *Abnormal (malignant-like)*.

        **Key Metrics**
        - **N/C ratio** — Nucleus area ÷ Cytoplasm area. Elevated values are a classic cytological sign of malignancy.
        - **Valorization %** — Computes the variance of internal granular/vacuole structures inside the cell.
        - **Circularity** — 4π × Area / Perimeter² (1.0 = perfect circle). Lower values indicate irregularity.
        - **Eccentricity** — 0 (circle) to 1 (very elongated). High values suggest atypical nuclear or cell shape.
        - **Perimeter** — Boundary length of the cell (pixels).

        The classifier is deliberately simple and fully explainable (no black-box ML). It triggers on combinations of high N/C, low circularity, high eccentricity, and enlarged nuclei.
        """
    )


# ----------------------------- Image Source ----------------------------------
uploaded_file = st.file_uploader(
    "Upload a cell image (PNG, JPG, TIFF, etc.)",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    accept_multiple_files=False,
    help="For best results use images with clear cell/nuclear contrast (H&E, Wright-Giemsa, fluorescence, etc.).",
)

use_synthetic = False
image: Optional[np.ndarray] = None
source_label = ""

col_a, col_b = st.columns([1, 1])

with col_a:
    if st.button("✨ Load synthetic demo (healthy + abnormal cells)", type="primary", use_container_width=True):
        use_synthetic = True

with col_b:
    if uploaded_file is None and not use_synthetic:
        st.info("Upload an image above, or click the button to load a ready-to-analyze synthetic demo image.", icon="ℹ️")

# Load image (priority: uploaded > synthetic button)
if uploaded_file is not None:
    image = load_image(uploaded_file)
    source_label = f"Uploaded: {uploaded_file.name}"
elif use_synthetic:
    with st.spinner("Generating realistic synthetic cell image..."):
        image = generate_synthetic_cell_image(width=680, height=520, n_healthy=6, n_abnormal=4, seed=42)
    source_label = "Synthetic demo image (generated on-the-fly)"

# ----------------------------- Run Analysis ----------------------------------
if image is not None:
    st.markdown(f"**Source:** {source_label}")

    # Build parameter dict from sidebar
    params_dict = {
        "min_cell_area": min_cell_area,
        "nucleus_dark_percentile": float(nucleus_percentile),
        "nc_ratio_abnormal": float(nc_abnormal),
        "nc_ratio_very_high": float(nc_very_high),
    }

    # Run (cached)
    results = run_analysis(image, params_dict)

    cells = results["cells"]
    summary = results["summary"]

    # ====================== SUMMARY METRICS ======================
    st.subheader("Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cells detected", summary["num_cells"])
    m2.metric("Flagged abnormal", f"{summary['num_abnormal']} ({summary['abnormal_pct']}%)")
    m3.metric("Mean N/C ratio", summary["mean_nc_ratio"])
    m4.metric("Max N/C ratio", summary["max_nc_ratio"])

    if summary["image_inverted"]:
        st.caption("ℹ️ Image was automatically inverted for analysis (bright objects on dark background detected).")

    # ====================== SIDE-BY-SIDE IMAGES ======================
    st.subheader("Visualization")

    img_col1, img_col2 = st.columns(2, gap="medium")

    with img_col1:
        st.markdown("**Original Image**")
        st.image(image, use_column_width=True, clamp=True)

    with img_col2:
        st.markdown("**Segmented Overlay**")
        st.image(
            results["overlay"],
            use_column_width=True,
            clamp=True,
            caption="Teal = cell boundaries | Magenta = nuclei",
        )

    # ====================== DATA TABLE ======================
    st.subheader("Extracted Metrics per Cell")

    if cells:
        df = pd.DataFrame(cells)

        # Reorder & pretty column names for display
        display_df = df[
            [
                "cell_id",
                "cell_area",
                "nucleus_area",
                "cytoplasm_area",
                "nc_ratio",
                "valorization_pct",
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
            "Valorization %",
            "Perimeter",
            "Circularity",
            "Eccentricity",
            "Classification",
        ]

        # Color the classification column
        def color_class(val: str):
            if "Abnormal" in val:
                return "background-color: #ffcccc; font-weight: 600"
            elif "Borderline" in val:
                return "background-color: #fff3cd; font-weight: 500"
            return "background-color: #d4edda"

        # FIXED: applymap is deprecated in Pandas >=2.1.0. Replaced with map.
        styled = display_df.style.map(color_class, subset=["Classification"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=320)

        # CSV download
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download metrics as CSV",
            data=csv_buffer.getvalue(),
            file_name="cell_morphometry_metrics.csv",
            mime="text/csv",
            use_container_width=False,
        )

        # ====================== SINGLE PLATELET ZOOM & ANALYZE ======================
        st.markdown("---")
        st.subheader("🔍 Single Platelet Inspection (Zoom & Valorization)")
        st.markdown("Select a specific platelet from the TEM image to zoom in and calculate internal structural percentages.")

        if cells:
            # Create a dropdown to select a specific cell ID
            cell_ids = [c["cell_id"] for c in cells]
            selected_id = st.selectbox("Select Platelet (Cell ID)", options=cell_ids)
            
            # Find the data for the selected cell
            selected_cell = next(c for c in cells if c["cell_id"] == selected_id)
            minr, minc, maxr, maxc = selected_cell["bbox"]
            
            # Add a 15-pixel padding box around the platelet so it's not cropped too tightly
            pad = 15
            minr_p = max(0, minr - pad)
            minc_p = max(0, minc - pad)
            maxr_p = min(image.shape[0], maxr + pad)
            maxc_p = min(image.shape[1], maxc + pad)

            # Crop the original image and the segmented overlay
            zoom_img = image[minr_p:maxr_p, minc_p:maxc_p]
            zoom_overlay = results["overlay"][minr_p:maxr_p, minc_p:maxc_p]

            # Display the zoomed interface
            z_col1, z_col2, z_col3 = st.columns([1.5, 1.5, 1])
            
            with z_col1:
                st.markdown(f"**Zoomed Platelet (ID: {selected_id})**")
                st.image(zoom_img, use_column_width=True, clamp=True)
                
            with z_col2:
                st.markdown("**Segmented Overlay**")
                st.image(zoom_overlay, use_column_width=True, clamp=True)
                
            with z_col3:
                st.markdown("**Specific Metrics**")
                st.metric("Valorization / Vacuolization", f"{selected_cell['valorization_pct']}%")
                st.metric("Total Area (pixels)", f"{selected_cell['cell_area']}")
                st.metric("Circularity", f"{selected_cell['circularity']}")

        # ====================== CHARTS ======================
        st.markdown("---")
        st.subheader("Interactive Feature Charts")

        chart_col1, chart_col2 = st.columns(2, gap="large")

        # --- Chart 1: N/C Ratio Histogram ---
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

        # --- Chart 2: N/C vs Eccentricity scatter ---
        with chart_col2:
            st.markdown("**N/C Ratio vs Eccentricity (by classification)**")
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

        # --- Chart 3: Circularity by classification (box + swarm) ---
        st.markdown("**Circularity by Classification** (higher = more round)")
        fig3, ax3 = plt.subplots(figsize=(9, 3.6))

        classes_order = ["Normal morphology", "Borderline", "Abnormal (malignant-like)"]
        data_for_box = [df[df["classification"] == c]["circularity"].values for c in classes_order if c in df["classification"].values]

        bp = ax3.boxplot(
            data_for_box,
            positions=[0, 1, 2][: len(data_for_box)],
            widths=0.55,
            patch_artist=True,
            showfliers=True,
        )

        palette = ["#2ecc71", "#f1c40f", "#e74c3c"]
        for patch, color in zip(bp["boxes"], palette[: len(data_for_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)

        # overlay individual points
        for i, c in enumerate(classes_order):
            if c in df["classification"].values:
                vals = df[df["classification"] == c]["circularity"].values
                x = np.random.normal(i, 0.08, size=len(vals))
                ax3.scatter(x, vals, c=palette[i], s=22, alpha=0.7, edgecolors="white", linewidths=0.3)

        ax3.set_xticks(range(len(data_for_box)))
        ax3.set_xticklabels([c for c in classes_order if c in df["classification"].values])
        ax3.set_ylabel("Circularity (4πA/P²)")
        ax3.set_ylim(0, 1.05)
        ax3.axhline(0.6, color="#7f8c8d", linestyle=":", alpha=0.6, lw=1)
        ax3.grid(True, axis="y", alpha=0.3)
        fig_to_st(fig3)

    else:
        st.warning("No cells were detected with the current parameters. Try lowering the minimum cell area or adjusting the nucleus darkness percentile.")

    # ====================== CLASSIFIER EXPLANATION ======================
    with st.expander("About the lightweight classifier (fully explainable)"):
        st.markdown(
            f"""
            The classifier uses simple, transparent rules (no neural nets or black boxes):

            - **Abnormal (malignant-like)** if ≥2 of the following are true **or** N/C ratio ≥ {nc_very_high}:
              - N/C ratio ≥ {nc_abnormal}
              - Eccentricity ≥ 0.74 (elongated / irregular)
              - Circularity ≤ 0.58 (atypical shape)
              - Nucleus area ≥ 520 px (enlarged nucleus)

            - **Borderline** if exactly one of the above criteria is met.
            - Otherwise **Normal morphology**.

            This mirrors classic cytopathology criteria (high nuclear-to-cytoplasmic ratio, nuclear pleomorphism, etc.) and is intended as a fast, reproducible screening aid or fallback when ML models are unavailable.
            """
        )

else:
    # No image loaded yet
    st.markdown("---")
    st.markdown(
        """
        ### Ready to explore?

        1. Click **"Load synthetic demo"** above to see the full pipeline in action immediately.
        2. Or upload your own microscopy image (H&E, Pap stain, fluorescence, etc.).

        The synthetic demo contains a realistic mixture of round, low-N/C "healthy" cells and larger, elongated, high-N/C "abnormal" cells. All metrics and classifications are produced by the exact same algorithm used on real uploads.
        """
    )

# ----------------------------- Footer ----------------------------------------
st.markdown("---")
st.caption(
    "Built with NumPy + OpenCV + scikit-image  •  Rule-based classifier  •  Fully reproducible synthetic data generator"
)
