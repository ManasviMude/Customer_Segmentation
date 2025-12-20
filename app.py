import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ----------------------------------------------------
# Page configuration
# ----------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation using K-Means Clustering")

# ----------------------------------------------------
# Cluster meanings
# ----------------------------------------------------
CLUSTER_MEANINGS = {
    0: "Low income, low spending, less active customers",
    1: "High income, high spending, loyal customers",
    2: "Medium income, moderate spending, regular customers",
    3: "High income but low spending, potential customers"
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
# Prepare columns
# ----------------------------------------------------
if 'Final_Cluster' not in df.columns and 'KMeans_Cluster' in df.columns:
    df['Final_Cluster'] = df['KMeans_Cluster']

if 'TotalSpending' not in df.columns:
    spend_cols = [c for c in df.columns if c.startswith('Mnt')]
    if spend_cols:
        df['TotalSpending'] = df[spend_cols].sum(axis=1)

# ----------------------------------------------------
# Helper function: Pie chart
# ----------------------------------------------------
def plot_pie(series, title):
    fig, ax = plt.subplots()
    counts = series.value_counts().sort_index()
    ax.pie(
        counts,
        labels=[f"Cluster {i}" for i in counts.index],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white'}
    )
    ax.set_title(title)
    ax.axis('equal')
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

    st.subheader("📊 Cluster Distribution (Percentage View)")
    plot_pie(df['Final_Cluster'], "Customer Distribution Across Clusters")

    st.subheader("📌 Cluster Meanings")
    for k, v in CLUSTER_MEANINGS.items():
        st.markdown(f"**Cluster {k}:** {v}")

    st.subheader("📊 Cluster Profile (INR)")
    profile_cols = [
        'Income','TotalSpending','Age','Recency',
        'NumWebPurchases','NumStorePurchases','NumCatalogPurchases'
    ]
    profile_cols = [c for c in profile_cols if c in df.columns]

    profile_df = df.groupby('Final_Cluster')[profile_cols].mean().round(2)

    # Rename for INR display
    if 'Income' in profile_df.columns:
        profile_df.rename(columns={'Income': 'Income (₹)'}, inplace=True)
    if 'TotalSpending' in profile_df.columns:
        profile_df.rename(columns={'TotalSpending': 'Total Spending (₹)'}, inplace=True)

    st.dataframe(profile_df)

# ====================================================
# 2️⃣ PREDICT CUSTOMER CLUSTER (NO PIE CHART HERE)
# ====================================================
elif menu == "Predict Customer Cluster":

    st.subheader("📌 Cluster Definitions")
    for k, v in CLUSTER_MEANINGS.items():
        st.markdown(f"**Cluster {k}:** {v}")

    st.markdown("---")

    st.subheader("🧍 Enter Customer Details (INR)")

    income = st.number_input("Income (₹)", min_value=0.0)
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

        st.success(
            f"🎯 **Customer belongs to Cluster {cluster}**\n\n"
            f"📌 **Cluster Meaning:** {CLUSTER_MEANINGS.get(cluster)}"
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

        st.subheader("🔧 Map Columns")
        cols = new_df.columns.tolist()

        income_col = st.selectbox("Income Column (₹)", cols)
        age_col = st.selectbox("Age Column", cols)
        recency_col = st.selectbox("Recency Column", cols)
        web_col = st.selectbox("Web Purchases Column", cols)
        store_col = st.selectbox("Store Purchases Column", cols)
        catalog_col = st.selectbox("Catalog Purchases Column", cols)

        if st.button("Run Clustering"):

            model_df = new_df[[income_col, age_col, recency_col,
                                web_col, store_col, catalog_col]].copy()

            model_df.columns = [
                'Income','Age','Recency',
                'NumWebPurchases','NumStorePurchases','NumCatalogPurchases'
            ]

            model_df = model_df.fillna(model_df.median(numeric_only=True))

            X_new = np.column_stack([
                np.log1p(model_df['Income']),
                model_df['Age'],
                model_df['Recency'],
                model_df['NumWebPurchases'],
                model_df['NumStorePurchases'],
                model_df['NumCatalogPurchases']
            ])

            X_scaled = scaler.transform(X_new)
            new_df['Predicted_Cluster'] = kmeans.predict(X_scaled)
            new_df['Cluster_Meaning'] = new_df['Predicted_Cluster'].map(CLUSTER_MEANINGS)

            st.success("✅ Clustering Completed")

            st.subheader("📊 Cluster Distribution (Uploaded Dataset)")
            plot_pie(new_df['Predicted_Cluster'], "Cluster Distribution of Uploaded Dataset")

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
