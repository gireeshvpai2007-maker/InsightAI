import streamlit as st


st.set_page_config(
    page_title="InsightAI",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 InsightAI")

st.subheader("AI-Powered Business Analytics Platform")

st.write(
    """
    InsightAI transforms raw business data into meaningful insights
    using data analytics, machine learning, and interactive visualizations.
    """
)

st.divider()


# =========================
# PLATFORM OVERVIEW
# =========================

st.header("🚀 What InsightAI Does")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Data Analytics")
    st.write(
        "Explore datasets, understand distributions, "
        "identify missing values, and analyze relationships."
    )

with col2:
    st.markdown("### 🤖 Machine Learning")
    st.write(
        "Train machine learning models and use them "
        "to generate predictions from business data."
    )

with col3:
    st.markdown("### 💡 Automated Insights")
    st.write(
        "Convert analytical results into clear and "
        "actionable business insights."
    )


st.divider()


# =========================
# CURRENT MODEL
# =========================

st.header("🧠 Current ML Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Linear Regression"
    )

with col2:
    st.metric(
        "Task",
        "House Price Prediction"
    )

with col3:
    st.metric(
        "Features",
        "6"
    )


st.divider()


st.info(
    "Use the sidebar to explore the Dashboard, "
    "Insights, and Prediction modules."
)