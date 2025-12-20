import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Generic Customer Segmentation App (K-Means)")

# ----------------------------------------------------
# Cluster meanings (generic business labels)
# ----------------------------------------------------
CLUSTER_MEANINGS = {
    0: "Low-value customers",
    1: "High-value customers",
    2: "Medium-value customers",
    3: "Potential growth customers"
}

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("🔧 Menu")
menu = st.sidebar.radio(
    "Select Option",
    ["Upload Dataset & Cluster", "About App"]
)

# ====================================================
# OPTION 1 — GENERIC DATASET CLUSTERING
# ====================================================
if menu == "Upload Dataset & Cluster":

    uploaded_file = st.file_uploader(
        "Upload any customer dataset (CSV or Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        # Load file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        # ------------------------------------------------
        # Auto-select numeric columns
        # ------------------------------------------------
        numeric_df = df.select_dtypes(include=np.number)

        # Drop ID-like columns automatically
        numeric_df = numeric_df.drop(
            columns=[col for col in numeric_df.columns if "id" in col.lower()],
            errors="ignore"
        )

        if numeric_df.shape[1] < 2:
            st.error("❌ Not enough numeric columns for clustering.")
        else:
            st.subheader("📌 Selected Numeric Features")
            st.write(list(numeric_df.columns))

            # ------------------------------------------------
            # Scaling
            # ------------------------------------------------
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(numeric_df)

            # ------------------------------------------------
            # Choose number of clusters
            # ------------------------------------------------
            k = st.slider("Select number of clusters (K)", 2, 6, 4)

            # ------------------------------------------------
            # Train K-Means on uploaded data
            # ------------------------------------------------
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            df['Cluster'] = clusters

            st.success("✅ Clustering completed successfully!")

            # ------------------------------------------------
            # Show cluster meanings
            # ------------------------------------------------
            st.subheader("📌 Cluster Meanings")
            for c in sorted(df['Cluster'].unique()):
                meaning = CLUSTER_MEANINGS.get(c, "Customer segment")
                st.markdown(f"**Cluster {c}:** {meaning}")

            # ------------------------------------------------
            # Cluster distribution
            # ------------------------------------------------
            st.subheader("📈 Cluster Distribution")
            st.bar_chart(df['Cluster'].value_counts())

            # ------------------------------------------------
            # Cluster profile
            # ------------------------------------------------
            st.subheader("📊 Cluster Profile (Mean Values)")
            st.dataframe(df.groupby('Cluster')[numeric_df.columns].mean().round(2))

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------
            st.subheader("📉 Feature Distribution by Cluster")

            feature_to_plot = st.selectbox(
                "Select a numeric feature",
                numeric_df.columns
            )

            fig, ax = plt.subplots()
            sns.boxplot(x='Cluster', y=feature_to_plot, data=df, ax=ax)
            st.pyplot(fig)

            # ------------------------------------------------
            # Download clustered data
            # ------------------------------------------------
            st.download_button(
                "⬇ Download Clustered Dataset",
                df.to_csv(index=False),
                "clustered_customers.csv",
                "text/csv"
            )

# ====================================================
# OPTION 2 — ABOUT
# ====================================================
elif menu == "About App":

    st.subheader("ℹ️ About This Application")
    st.write("""
    This is a **generic customer segmentation application** built using **K-Means clustering**.

    🔹 Works with **any customer dataset**
    🔹 Automatically detects numeric features
    🔹 No predefined column dependency
    🔹 Suitable for real-world datasets

    **Workflow:**
    1. Upload dataset
    2. App selects numeric features
    3. K-Means clusters customers
    4. Visualize & download results
    """)

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.write("🚀 Built with Streamlit | K-Means Clustering")
