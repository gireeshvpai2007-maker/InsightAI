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


from predict import predict_price


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
    "Enter the property details below to generate "
    "a house-price prediction using the trained "
    "InsightAI model."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Property Details")


col1, col2, col3 = st.columns(3)


with col1:

    area = st.number_input(
        "Area",
        min_value=1,
        value=1500
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        value=3,
        step=1
    )


with col2:

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        value=2,
        step=1
    )

    stories = st.number_input(
        "Stories",
        min_value=1,
        value=2,
        step=1
    )


with col3:

    parking = st.number_input(
        "Parking",
        min_value=0,
        value=1,
        step=1
    )

    age = st.number_input(
        "Age",
        min_value=0,
        value=5,
        step=1
    )


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict Price",
    type="primary"
):

    prediction = predict_price(
        area=area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        stories=stories,
        parking=parking,
        age=age
    )

    st.success(
        f"Predicted Price: ₹{prediction:,.2f}"
    )