import streamlit as st
import sys
import os


# ============================================================
# BACKEND PATH
# ============================================================

BACKEND_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "backend"
)

sys.path.append(BACKEND_PATH)


from predict import (
    load_saved_model,
    get_expected_features,
    predict_dataset
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InsightAI - Prediction",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🤖 Prediction")

st.write(
    "Upload a dataset containing the features required by "
    "the trained InsightAI model to generate predictions."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

try:

    pipeline, metadata = load_saved_model(
        MODEL_PATH
    )

    expected_features = get_expected_features(
        pipeline
    )

except Exception as error:

    st.error(
        f"Unable to load the saved model: {error}"
    )

    st.stop()


# ============================================================
# MODEL DETAILS
# ============================================================

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Task",
        metadata["task"].title()
    )

with col2:
    st.metric(
        "Model",
        metadata["model_name"]
    )

with col3:
    st.metric(
        "Target",
        metadata["target_column"]
    )


# ============================================================
# REQUIRED FEATURES
# ============================================================

with st.expander("Required Features"):

    st.write(
        "The uploaded dataset must contain the following "
        "features:"
    )

    for feature in expected_features:
        st.write(f"- {feature}")


# ============================================================
# DATASET UPLOAD
# ============================================================

st.subheader("Upload Prediction Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    import pandas as pd

    prediction_data = pd.read_csv(
        uploaded_file
    )

    st.write("### Uploaded Data")

    st.dataframe(
        prediction_data,
        use_container_width=True
    )

    missing_features = [
        column
        for column in expected_features
        if column not in prediction_data.columns
    ]

    if missing_features:

        st.error(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    else:

        if st.button(
            "Generate Predictions",
            type="primary"
        ):

            try:

                result, _ = predict_dataset(
                    prediction_data,
                    MODEL_PATH
                )

                st.success(
                    "Predictions generated successfully."
                )

                st.write("### Prediction Results")

                st.dataframe(
                    result,
                    use_container_width=True
                )

                csv_data = result.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Predictions",
                    data=csv_data,
                    file_name="insightai_predictions.csv",
                    mime="text/csv"
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )