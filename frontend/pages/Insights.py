import streamlit as st
import sys
import os
import pandas as pd

BACKEND_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    ),
    "backend"
)

sys.path.append(BACKEND_PATH)

from insights import generate_insights


st.set_page_config(
    page_title="InsightAI - Insights",
    page_icon="💡",
    layout="wide"
)


st.title("💡 Automated Insights")

st.write(
    "Upload a dataset and InsightAI will analyze its "
    "data quality, distributions, outliers, and "
    "relationships with the selected target."
)


# ============================================================
# DATASET UPLOAD
# ============================================================

st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset loaded successfully: "
            f"{df.shape[0]:,} rows × {df.shape[1]} columns"
        )

    except Exception as error:

        st.error(
            f"Unable to read the dataset: {error}"
        )

        st.stop()


    # ========================================================
    # DATA PREVIEW
    # ========================================================

    with st.expander("Preview Dataset"):

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


    # ========================================================
    # TARGET SELECTION
    # ========================================================

    st.subheader("Target Selection")

    target_column = st.selectbox(
        "Select the target column",
        options=["None"] + df.columns.tolist()
    )

    if target_column == "None":

        target_column = None


    # ========================================================
    # GENERATE INSIGHTS
    # ========================================================

    if st.button(
        "Generate Insights",
        type="primary"
    ):

        try:

            insights = generate_insights(
                df,
                target_column=target_column
            )

            if not insights:

                st.info(
                    "No insights were generated for this dataset."
                )

            else:

                st.success(
                    f"{len(insights)} insights generated."
                )


                # ============================================
                # GROUP INSIGHTS BY CATEGORY
                # ============================================

                categories = {}

                for item in insights:

                    category = item["category"]

                    if category not in categories:

                        categories[category] = []

                    categories[category].append(
                        item["insight"]
                    )


                # ============================================
                # DISPLAY INSIGHTS
                # ============================================

                for category, messages in categories.items():

                    st.markdown(
                        f"### {category}"
                    )

                    for message in messages:

                        st.info(message)


        except Exception as error:

            st.error(
                f"Insight generation failed: {error}"
            )