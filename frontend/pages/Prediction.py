import streamlit as st
import sys
import os

# Add backend directory to Python path
BACKEND_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "backend")
)

sys.path.append(BACKEND_PATH)

from predict import predict_price


st.set_page_config(
    page_title="InsightAI - Prediction",
    page_icon="🔮",
    layout="wide"
)


st.title("🔮 House Price Prediction")

st.write(
    "Enter the house details below to estimate its price "
    "using the trained Machine Learning model."
)

st.divider()


# =========================
# INPUT SECTION
# =========================

st.subheader("🏠 Property Details")

col1, col2 = st.columns(2)

with col1:

    area = st.number_input(
        "Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )


with col2:

    stories = st.number_input(
        "Stories",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=10,
        value=2,
        step=1
    )

    age = st.number_input(
        "Property Age (years)",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


st.divider()


# =========================
# PREDICTION
# =========================

if st.button(
    "🔮 Predict House Price",
    use_container_width=True
):

    input_data = {
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Stories": stories,
        "Parking": parking,
        "Age": age
    }

    try:

        predicted_price = predict_price(input_data)

        st.success("Prediction generated successfully!")

        st.subheader("Estimated Property Price")

        st.metric(
            label="Predicted Price",
            value=f"₹{predicted_price:,.2f}"
        )

        st.info(
            "This prediction is generated using the trained "
            "Linear Regression model."
        )

    except Exception as e:

        st.error(
            f"Unable to generate prediction: {e}"
        )