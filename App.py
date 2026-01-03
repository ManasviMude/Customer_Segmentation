import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ----------------------------------------------------
# Page configuration
# ----------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation using K-Means Clustering")

# ----------------------------------------------------
# Cluster info (4 CLUSTERS)
# ----------------------------------------------------
CLUSTER_INFO = {
    0: {
        "label": "Low Value Customers",
        "meaning": "Low income and low engagement customers",
        "recommendation": "Run discounts and awareness campaigns",
        "color": "#FF9999"
    },
    1: {
        "label": "High Value Customers",
        "meaning": "High income and high spending loyal customers",
        "recommendation": "Provide premium rewards and loyalty benefits",
        "color": "#66B2FF"
    },
    2: {
        "label": "Regular Customers",
        "meaning": "Moderate income and regular purchasing behavior",
        "recommendation": "Upsell and cross-sell relevant products",
        "color": "#99FF99"
    },
    3: {
        "label": "Potential Customers",
        "meaning": "High income but low spending customers",
        "recommendation": "Target with personalized promotions",
        "color": "#FFCC99"
    }
}

# ----------------------------------------------------
# Load trained model & scaler (trained with K=4)
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
    [
        "Saved Dataset Analysis",
        "Manual Input & Analysis",
        "Upload Dataset & Analyze"
    ]
)

# ====================================================
# 1️⃣ SAVED DATASET ANALYSIS
# ====================================================
if menu == "Saved Dataset Analysis":

    df = pd.read_csv("final_customer_segmentation_output.csv")

    if "Final_Cluster" not in df.columns and "KMeans_Cluster" in df.columns:
        df["Final_Cluster"] = df["KMeans_Cluster"]

    st.subheader("📄 Saved Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Cluster Distribution (Donut Chart)")
    plot_donut(df["Final_Cluster"], "Saved Dataset Cluster Distribution")

    st.subheader("📌 Cluster Meanings & Recommendations")
    for k, info in CLUSTER_INFO.items():
        st.markdown(
            f"""
            **Cluster {k} – {info['label']}**  
            Meaning: {info['meaning']}  
            Recommendation: *{info['recommendation']}*
            """
        )

# ====================================================
# 2️⃣ MANUAL INPUT & ANALYSIS
# ====================================================
elif menu == "Manual Input & Analysis":

    st.subheader("🧍 Enter Customer Details")

    income = st.number_input("Income", min_value=0.0)
    age = st.number_input("Age", min_value=18)
    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    web = st.number_input("Web Purchases", min_value=0)
    store = st.number_input("Store Purchases", min_value=0)
    catalog = st.number_input("Catalog Purchases", min_value=0)

    if st.button("Analyze Customer"):

        X = np.array([[
            np.log1p(income),
            age,
            recency,
            web,
            store,
            catalog
        ]])

        X_scaled = scaler.transform(X)
        cluster = int(kmeans.predict(X_scaled)[0])
        info = CLUSTER_INFO[cluster]

        st.success(f"🎯 Customer belongs to **Cluster {cluster} – {info['label']}**")

        st.markdown(
            f"""
            **Cluster Meaning:** {info['meaning']}  
            **Business Recommendation:** {info['recommendation']}
            """
        )

# ====================================================
# 3️⃣ UPLOAD DATASET & ANALYZE (GENERIC)
# ====================================================
elif menu == "Upload Dataset & Analyze":

    uploaded_file = st.file_uploader(
        "Upload any customer dataset (CSV or Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📄 Uploaded Dataset Preview")
        st.dataframe(df.head())

        # Select numeric columns
        numeric_df = df.select_dtypes(include=np.number)

        # Remove ID-like columns
        numeric_df = numeric_df.drop(
            columns=[c for c in numeric_df.columns if "id" in c.lower()],
            errors="ignore"
        )

        if numeric_df.shape[1] < scaler.n_features_in_:
            st.error("❌ Not enough numeric features for clustering.")
        else:
            # Handle missing values
            numeric_df = numeric_df.fillna(numeric_df.median())

            # Convert to NumPy (critical fix)
            X = numeric_df.to_numpy()[:, :scaler.n_features_in_]
            X_scaled = scaler.transform(X)

            df["Cluster"] = kmeans.predict(X_scaled)
            df["Cluster_Label"] = df["Cluster"].map(
                lambda x: CLUSTER_INFO[x]["label"]
            )

            st.success("✅ Dataset analyzed successfully!")

            st.subheader("🔍 Analyzed Dataset (First 10 Rows)")
            st.dataframe(df.head(10))

            st.subheader("📊 Cluster Distribution (Donut Chart)")
            plot_donut(df["Cluster"], "Uploaded Dataset Cluster Distribution")

            st.subheader("📌 Cluster-wise Recommendations")
            for k, info in CLUSTER_INFO.items():
                st.markdown(
                    f"**Cluster {k} – {info['label']}**: {info['recommendation']}"
                )

            st.download_button(
                "⬇ Download Clustered Dataset",
                df.to_csv(index=False),
                "clustered_dataset.csv",
                "text/csv"
            )

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.write("🚀 Deployed using Streamlit Cloud & GitHub")
st.write("📌 Final Model: K-Means Clustering (4 Clusters)")
