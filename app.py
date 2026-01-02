import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------
# Page configuration
# ----------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation using K-Means Clustering")

# ----------------------------------------------------
# Cluster info
# ----------------------------------------------------
CLUSTER_INFO = {
    0: {
        "label": "Low Value Customers",
        "recommendation": "Provide discounts and awareness campaigns",
        "color": "#FF9999"
    },
    1: {
        "label": "High Value Customers",
        "recommendation": "Offer premium rewards and loyalty benefits",
        "color": "#66B2FF"
    },
    2: {
        "label": "Regular Customers",
        "recommendation": "Upsell and cross-sell relevant products",
        "color": "#99FF99"
    },
    3: {
        "label": "Potential Customers",
        "recommendation": "Target with personalized promotions",
        "color": "#FFCC99"
    }
}

# ----------------------------------------------------
# Load trained model
# ----------------------------------------------------
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------------------------------
# Donut chart helper
# ----------------------------------------------------
def plot_donut(series, title):
    counts = series.value_counts().sort_index()
    colors = [CLUSTER_INFO[i]["color"] for i in counts.index]

    fig, ax = plt.subplots()
    ax.pie(
        counts,
        labels=[f"Cluster {i}" for i in counts.index],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white")
    )
    ax.set_title(title)
    ax.axis("equal")
    st.pyplot(fig)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("🔧 Menu")
menu = st.sidebar.radio(
    "Select Option",
    ["Upload Dataset & Analyze", "About"]
)

# ====================================================
# UPLOAD & ANALYZE DATASET
# ====================================================
if menu == "Upload Dataset & Analyze":

    uploaded_file = st.file_uploader(
        "Upload any customer dataset (CSV or Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        # Load dataset
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📄 Uploaded Dataset Preview")
        st.dataframe(df.head())

        # ------------------------------------------------
        # Automatic numeric feature selection
        # ------------------------------------------------
        numeric_df = df.select_dtypes(include=np.number)

        # Drop ID-like columns
        numeric_df = numeric_df.drop(
            columns=[c for c in numeric_df.columns if "id" in c.lower()],
            errors="ignore"
        )

        if numeric_df.shape[1] < 2:
            st.error("❌ Not enough numeric features for clustering.")
        else:
            # Handle missing values
            numeric_df = numeric_df.fillna(numeric_df.median())

            # Scale
            X_scaled = scaler.transform(numeric_df.iloc[:, :scaler.n_features_in_])

            # Predict clusters
            df["Cluster"] = kmeans.predict(X_scaled)
            df["Cluster_Label"] = df["Cluster"].map(
                lambda x: CLUSTER_INFO[x]["label"]
            )

            st.success("✅ Dataset analyzed and clustered successfully!")

            # ------------------------------------------------
            # Preview analyzed dataset
            # ------------------------------------------------
            st.subheader("🔍 Analyzed Dataset Preview (First 10 Rows)")
            st.dataframe(df.head(10))

            # ------------------------------------------------
            # Donut chart
            # ------------------------------------------------
            st.subheader("📊 Cluster Distribution")
            plot_donut(df["Cluster"], "Customer Cluster Distribution")

            # ------------------------------------------------
            # Recommendations
            # ------------------------------------------------
            st.subheader("📌 Cluster-wise Recommendations")
            for k, info in CLUSTER_INFO.items():
                st.markdown(
                    f"**Cluster {k} – {info['label']}**: {info['recommendation']}"
                )

            # ------------------------------------------------
            # Download
            # ------------------------------------------------
            st.download_button(
                "⬇ Download Analyzed Dataset",
                df.to_csv(index=False),
                "analyzed_clustered_data.csv",
                "text/csv"
            )

# ====================================================
# ABOUT
# ====================================================
elif menu == "About":

    st.subheader("ℹ️ About This App")
    st.write("""
    This application performs **automatic customer segmentation** using
    **K-Means clustering**.

    ✔ No manual column mapping  
    ✔ Works with large datasets  
    ✔ Handles missing values automatically  
    ✔ Provides visual insights & business recommendations  
    ✔ Built for real-world deployment  
    """)

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.write("🚀 Deployed using Streamlit Cloud & GitHub")
st.write("📌 Final Model: K-Means Clustering")
