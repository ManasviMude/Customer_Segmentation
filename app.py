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
# Cluster info (3 CLUSTERS)
# ----------------------------------------------------
CLUSTER_INFO = {
    0: {
        "label": "Low Value Customers",
        "meaning": "Low income and low engagement customers",
        "recommendation": "Offer discounts and awareness campaigns",
        "color": "#FF9999"
    },
    1: {
        "label": "High Value Customers",
        "meaning": "High income and high spending loyal customers",
        "recommendation": "Provide premium offers and loyalty rewards",
        "color": "#66B2FF"
    },
    2: {
        "label": "Regular Customers",
        "meaning": "Moderate income and regular purchasing behavior",
        "recommendation": "Upsell and cross-sell relevant products",
        "color": "#99FF99"
    }
}

# ----------------------------------------------------
# Load trained model & data
# ----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("final_customer_segmentation_output.csv")

df = load_data()
kmeans = joblib.load("kmeans_model.pkl")   # trained with k=3
scaler = joblib.load("scaler.pkl")

# ----------------------------------------------------
# Ensure cluster column
# ----------------------------------------------------
if "Final_Cluster" not in df.columns and "KMeans_Cluster" in df.columns:
    df["Final_Cluster"] = df["KMeans_Cluster"]

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
        "View Clusters",
        "Predict Customer Cluster",
        "Upload Dataset & Analyze"
    ]
)

# ====================================================
# 1️⃣ VIEW CLUSTERS
# ====================================================
if menu == "View Clusters":

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Cluster Distribution")
    plot_donut(df["Final_Cluster"], "Customer Distribution Across Clusters")

    st.subheader("📌 Cluster Details & Recommendations")
    for k, info in CLUSTER_INFO.items():
        st.markdown(
            f"""
            **Cluster {k} – {info['label']}**  
            • Meaning: {info['meaning']}  
            • Recommendation: *{info['recommendation']}*
            """
        )

# ====================================================
# 2️⃣ PREDICT CUSTOMER CLUSTER (SINGLE CUSTOMER)
# ====================================================
elif menu == "Predict Customer Cluster":

    st.subheader("📌 Cluster Definitions")
    for k, info in CLUSTER_INFO.items():
        st.markdown(f"**Cluster {k}:** {info['label']}")

    st.markdown("---")

    st.subheader("🧍 Enter Customer Details")

    income = st.number_input("Income", min_value=0.0)
    age = st.number_input("Age", min_value=18)
    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    web = st.number_input("Web Purchases", min_value=0)
    store = st.number_input("Store Purchases", min_value=0)
    catalog = st.number_input("Catalog Purchases", min_value=0)

    if st.button("Predict Cluster"):

        X_input = np.array([[
            np.log1p(income),
            age,
            recency,
            web,
            store,
            catalog
        ]], dtype=float)

        X_scaled = scaler.transform(X_input)
        cluster = int(kmeans.predict(X_scaled)[0])
        info = CLUSTER_INFO[cluster]

        st.success(f"🎯 Customer belongs to **Cluster {cluster} – {info['label']}**")

        st.markdown(
            f"""
            **Meaning:** {info['meaning']}  
            **Business Recommendation:** {info['recommendation']}
            """
        )

# ====================================================
# 3️⃣ UPLOAD DATASET & ANALYZE (NO COLUMN MAPPING)
# ====================================================
elif menu == "Upload Dataset & Analyze":

    uploaded_file = st.file_uploader(
        "Upload any customer dataset (CSV or Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)

        st.subheader("📄 Uploaded Dataset Preview")
        st.dataframe(new_df.head())

        # ------------------------------------------------
        # Auto numeric feature selection
        # ------------------------------------------------
        numeric_df = new_df.select_dtypes(include=np.number)

        # Drop ID-like columns
        numeric_df = numeric_df.drop(
            columns=[c for c in numeric_df.columns if "id" in c.lower()],
            errors="ignore"
        )

        if numeric_df.shape[1] < scaler.n_features_in_:
            st.error("❌ Dataset does not have enough numeric features.")
        else:
            # Handle missing values
            numeric_df = numeric_df.fillna(numeric_df.median())

            # Use only required features
            X_scaled = scaler.transform(
                numeric_df.iloc[:, :scaler.n_features_in_]
            )

            # Predict clusters
            new_df["Cluster"] = kmeans.predict(X_scaled)
            new_df["Cluster_Label"] = new_df["Cluster"].map(
                lambda x: CLUSTER_INFO[x]["label"]
            )

            st.success("✅ Dataset analyzed successfully!")

            st.subheader("🔍 Analyzed Dataset Preview (First 10 Rows)")
            st.dataframe(new_df.head(10))

            st.subheader("📊 Cluster Distribution")
            plot_donut(new_df["Cluster"], "Cluster Distribution of Uploaded Dataset")

            st.subheader("📌 Cluster-wise Recommendations")
            for k, info in CLUSTER_INFO.items():
                st.markdown(
                    f"**Cluster {k} – {info['label']}**: {info['recommendation']}"
                )

            st.download_button(
                "⬇ Download Analyzed Dataset",
                new_df.to_csv(index=False),
                "analyzed_clustered_data.csv",
                "text/csv"
            )

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.write("🚀 Deployed using Streamlit Cloud & GitHub")
st.write("📌 Final Model: K-Means Clustering (3 Clusters)")
