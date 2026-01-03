import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation using K-Means Clustering")

# ----------------------------------------------------
# Cluster info (3 clusters)
# ----------------------------------------------------
CLUSTER_INFO = {
    0: {
        "label": "Low Value Customers",
        "meaning": "Low income and low engagement",
        "recommendation": "Run discount and awareness campaigns",
        "color": "#FF9999"
    },
    1: {
        "label": "High Value Customers",
        "meaning": "High income and high spending",
        "recommendation": "Provide premium rewards and loyalty benefits",
        "color": "#66B2FF"
    },
    2: {
        "label": "Regular Customers",
        "meaning": "Moderate income and regular purchases",
        "recommendation": "Upsell and cross-sell products",
        "color": "#99FF99"
    }
}

# ----------------------------------------------------
# Load model
# ----------------------------------------------------
kmeans = joblib.load("kmeans_model.pkl")   # k=3
scaler = joblib.load("scaler.pkl")

# ----------------------------------------------------
# Donut chart
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
    ["View Clusters", "Predict Customer Cluster", "Upload Dataset & Analyze"]
)

# ====================================================
# 1️⃣ VIEW CLUSTERS (from trained dataset)
# ====================================================
if menu == "View Clusters":

    df = pd.read_csv("final_customer_segmentation_output.csv")

    if "Final_Cluster" not in df.columns and "KMeans_Cluster" in df.columns:
        df["Final_Cluster"] = df["KMeans_Cluster"]

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Cluster Distribution")
    plot_donut(df["Final_Cluster"], "Customer Distribution")

    st.subheader("📌 Cluster Insights")
    for k, v in CLUSTER_INFO.items():
        st.markdown(
            f"""
            **Cluster {k} – {v['label']}**  
            Meaning: {v['meaning']}  
            Recommendation: *{v['recommendation']}*
            """
        )

# ====================================================
# 2️⃣ SINGLE CUSTOMER PREDICTION
# ====================================================
elif menu == "Predict Customer Cluster":

    st.subheader("🧍 Enter Customer Details")

    income = st.number_input("Income", min_value=0.0)
    age = st.number_input("Age", min_value=18)
    recency = st.number_input("Recency", min_value=0)
    web = st.number_input("Web Purchases", min_value=0)
    store = st.number_input("Store Purchases", min_value=0)
    catalog = st.number_input("Catalog Purchases", min_value=0)

    if st.button("Predict Cluster"):

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

        st.success(f"🎯 Cluster {cluster} – {info['label']}")
        st.markdown(
            f"""
            **Meaning:** {info['meaning']}  
            **Recommendation:** {info['recommendation']}
            """
        )

# ====================================================
# 3️⃣ UPLOAD DATASET & ANALYZE (GENERIC & SAFE)
# ====================================================
elif menu == "Upload Dataset & Analyze":

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📄 Uploaded Dataset Preview")
        st.dataframe(df.head())

        # Select numeric features
        numeric_df = df.select_dtypes(include=np.number)

        # Drop ID-like columns
        numeric_df = numeric_df.drop(
            columns=[c for c in numeric_df.columns if "id" in c.lower()],
            errors="ignore"
        )

        if numeric_df.shape[1] < scaler.n_features_in_:
            st.error("❌ Not enough numeric features for clustering.")
        else:
            # Fill missing values
            numeric_df = numeric_df.fillna(numeric_df.median())

            # ---- CRITICAL FIX ----
            X = numeric_df.to_numpy()[:, :scaler.n_features_in_]
            X_scaled = scaler.transform(X)

            # Predict clusters
            df["Cluster"] = kmeans.predict(X_scaled)
            df["Cluster_Label"] = df["Cluster"].map(
                lambda x: CLUSTER_INFO[x]["label"]
            )

            st.success("✅ Dataset analyzed successfully!")

            st.subheader("🔍 Analyzed Dataset (First 10 Rows)")
            st.dataframe(df.head(10))

            st.subheader("📊 Cluster Distribution")
            plot_donut(df["Cluster"], "Cluster Distribution")

            st.subheader("📌 Recommendations")
            for k, v in CLUSTER_INFO.items():
                st.markdown(
                    f"**Cluster {k} – {v['label']}**: {v['recommendation']}"
                )

            st.download_button(
                "⬇ Download Clustered Dataset",
                df.to_csv(index=False),
                "clustered_output.csv",
                "text/csv"
            )

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.write("🚀 Deployed using Streamlit Cloud & GitHub")
st.write("📌 Final Model: K-Means (3 Clusters)")
