import streamlit as st
import pandas as pd
import sys
import os


# ============================================================
# BACKEND PATH
# ============================================================

BACKEND_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "backend"
)

sys.path.append(BACKEND_PATH)


from utils import generate_dataset_profile


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InsightAI",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 InsightAI")

st.write(
    "Upload a CSV dataset and InsightAI will automatically "
    "analyze its structure, statistics, and data quality."
)


# ============================================================
# CSV UPLOAD
# ============================================================

st.subheader("📁 Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# ============================================================
# DATASET ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        st.success(
            "Dataset uploaded successfully."
        )

        # ----------------------------------------------------
        # Dataset preview
        # ----------------------------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


        # ----------------------------------------------------
        # Generate profile
        # ----------------------------------------------------

        profile = generate_dataset_profile(
            df
        )

        summary = profile["summary"]
        column_profile = profile["column_profile"]
        outliers = profile["outliers"]


        # ----------------------------------------------------
        # Dataset Summary
        # ----------------------------------------------------

        st.subheader("📋 Dataset Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                summary["rows"]
            )

        with col2:
            st.metric(
                "Columns",
                summary["columns"]
            )

        with col3:
            st.metric(
                "Missing Values",
                summary["missing_values"]
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                summary["duplicate_rows"]
            )


        # ----------------------------------------------------
        # Column Profile
        # ----------------------------------------------------

        st.subheader("📊 Column Profile")

        st.dataframe(
            column_profile,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Outlier Analysis
        # ----------------------------------------------------

        st.subheader("🚨 Outlier Analysis")

        if outliers.empty:

            st.success(
                "No numerical columns available for "
                "outlier analysis."
            )

        else:

            st.dataframe(
                outliers,
                use_container_width=True
            )


    except Exception as error:

        st.error(
            f"Unable to analyze dataset: {error}"
        )