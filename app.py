import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ----------------------------------------------------
# Page configuration
# ----------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation using K-Means Clustering")

# ----------------------------------------------------
# Cluster meanings & recommendations
# ----------------------------------------------------
CLUSTER_INFO = {
    0: {
        "label": "Low Value Customers",
        "meaning": "Low income, low spending, less active customers",
        "recommendation": "Offer discounts and basic promotions to increase engagement",
        "color": "#FF9999"
    },
    1: {
        "label": "High Value Customers",
        "meaning": "High income, high spending, loyal customers",
        "recommendation": "Provide premium offers, loyalty rewards, and personalized services",
        "color": "#66B2FF"
    },
    2: {
        "label": "Regular Customers",
        "meaning": "Medium income, moderate spending, regular customers",
        "recommendation": "Upsell and cross-sell products to increase basket size",
        "color": "#99FF99"
    },
    3: {
        "label": "Potential Customers",
        "meaning": "High income but low spending customers",
        "recommendation": "Target with personalized recommendations and awareness campaigns",
        "color": "#FFCC99"
    }
}

# ----------------------------------------------------
# Load data and models
# ----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("final_customer_segmentation_output.csv")

df = load_data()
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------------------------------
# Prepare visualization columns
# ----------------------------------------------------
if 'Final_Cluster' not in df.columns and 'KMeans_Cluster' in df.columns:
    df['Final_Cluster'] = df['KMeans_Cluster']

# ----------------------------------------------------
# Donut chart helper
# ----------------------------------------------------
def plot_donut(series, title):
    counts = series.value_counts().sort_index()
    labels = [f"Cluster {i}" for i in counts.index]
    colors = [CLUSTER_INFO[i]["color"] for i in counts.index]

    fig, ax = plt.subplots()
    ax.pie(
        counts,
        labels=labels,
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
    ["View Clusters", "Predict Customer Cluster", "Upload CSV/Excel for Prediction"]
)

# ====================================================
# 1️⃣ VIEW CLUSTERS
# ====================================================
if menu == "View Clusters":

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Cluster Distribution ")
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
# 2️⃣ PREDICT CUSTOMER CLUSTER
# ====================================================
elif menu == "Predict Customer Cluster":

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
            **Cluster Meaning:** {info['meaning']}  
            **Business Recommendation:** {info['recommendation']}
            """
        )

# ====================================================
# 3️⃣ CSV / EXCEL UPLOAD
# ====================================================
elif menu == "Upload CSV/Excel for Prediction":

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(new_df.head())

        if st.button("Run Clustering"):

            model_df = new_df[
                [income_col, age_col, recency_col, web_col, store_col, catalog_col]
            ].copy()

            model_df.columns = [
                "Income", "Age", "Recency",
                "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases"
            ]

            model_df = model_df.fillna(model_df.median(numeric_only=True))

            X_new = np.column_stack([
                np.log1p(model_df["Income"]),
                model_df["Age"],
                model_df["Recency"],
                model_df["NumWebPurchases"],
                model_df["NumStorePurchases"],
                model_df["NumCatalogPurchases"]
            ])

            X_scaled = scaler.transform(X_new)
            new_df["Predicted_Cluster"] = kmeans.predict(X_scaled)
            new_df["Cluster_Label"] = new_df["Predicted_Cluster"].map(
                lambda x: CLUSTER_INFO[x]["label"]
            )

            st.success("✅ Clustering Completed")

            st.subheader("📊 Cluster Distribution (Uploaded Data)")
            plot_donut(new_df["Predicted_Cluster"], "Uploaded Dataset Cluster Distribution")

            st.dataframe(new_df.head())

            st.download_button(
                "⬇ Download Clustered File",
                new_df.to_csv(index=False),
                "clustered_customers.csv",
                "text/csv"
            )

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.write("🚀 Deployed using Streamlit Cloud & GitHub")
st.write("📌 Final Model: K-Means Clustering")
