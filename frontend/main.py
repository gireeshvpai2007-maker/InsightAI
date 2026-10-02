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

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models"
)

sys.path.append(BACKEND_PATH)


from utils import generate_dataset_profile
from model_selection import detect_task
from training import (
    train_dataset,
    save_training_result
)

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


        # ----------------------------------------------------
        # Target Column Selection
        # ----------------------------------------------------

        st.subheader("🎯 Target Column")

        target_column = st.selectbox(
            "Select the column you want InsightAI to analyze/predict",
            df.columns
        )

        # ----------------------------------------------------
        # Task Detection
        # ----------------------------------------------------

        if target_column:

            target = df[target_column]

            task = detect_task(target)

            st.subheader(
                "🤖 Detected Machine Learning Task"
            )

            if task == "regression":

                st.info(
                    f"Target: **{target_column}**  \n"
                    "Detected task: **Regression**"
                )

            else:

                st.info(
                    f"Target: **{target_column}**  \n"
                    "Detected task: **Classification**"
                )


        # ----------------------------------------------------
        # Model Training
        # ----------------------------------------------------

        st.subheader("🚀 Model Training")

        if st.button(
            "Train InsightAI Model",
            type="primary"
        ):

            with st.spinner(
                "Training and evaluating models..."
            ):

                result = train_dataset(
                    df,
                    target_column
                )
                
                save_paths = save_training_result(
                      result,
                      MODEL_PATH
               )


            st.success(
                "Model training completed successfully."
            )


            # ------------------------------------------------
            # Training Summary
            # ------------------------------------------------

            st.subheader("🧠 Training Summary")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Task",
                    result["task"].title()
                )

            with col2:

                st.metric(
                    "Selected Model",
                    result["model_name"]
                )

            with col3:

                st.metric(
                    "Training Rows",
                    result["train_size"]
                )


            # ------------------------------------------------
            # Model Comparison
            # ------------------------------------------------

            st.subheader(
                "📊 Model Comparison"
            )

            st.dataframe(
                result["model_comparison"],
                use_container_width=True
            )


            # ------------------------------------------------
            # Evaluation Metrics
            # ------------------------------------------------

            st.subheader(
                "📈 Model Evaluation"
            )

            metrics = result["metrics"]

            metric_columns = st.columns(
                len(metrics)
            )

            for column, (metric, value) in zip(
                metric_columns,
                metrics.items()
            ):

                with column:

                    if value is None:

                        st.metric(
                            metric,
                            "N/A"
                        )

                    else:

                        st.metric(
                            metric,
                            f"{value:.4f}"
                        )


    except Exception as error:

        st.error(
            f"Unable to analyze dataset: {error}"
        )