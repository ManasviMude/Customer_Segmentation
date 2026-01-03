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
# Prepare visualization columns
# ----------------------------------------------------
if 'Final_Cluster' not in df.columns and 'KMeans_Cluster' in df.columns:
    df['Final_Cluster'] = df['KMeans_Cluster']

if 'TotalSpending' not in df.columns:
    spend_cols = [c for c in df.columns if c.startswith('Mnt')]
    if spend_cols:
        df['TotalSpending'] = df[spend_cols].sum(axis=1)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("🔧 Menu")
menu = st.sidebar.radio(
    "Select Option",
    ["View Clusters", "Predict Customer Cluster", "Upload CSV/Excel for Prediction"]
)

# ====================================================
# VIEW CLUSTERS
# ====================================================
if menu == "View Clusters":

    st.dataframe(df.head())

    st.subheader("📈 Cluster Distribution")
    st.bar_chart(df['Final_Cluster'].value_counts().sort_index())

    st.subheader("📊 Cluster Profile")
    cols = [
        'Income','TotalSpending','Age','Recency',
        'NumWebPurchases','NumStorePurchases','NumCatalogPurchases'
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df.groupby('Final_Cluster')[cols].mean().round(2))

    fig, ax = plt.subplots()
    sns.boxplot(x='Final_Cluster', y='Income', data=df, ax=ax)
    st.pyplot(fig)

# ====================================================
# SINGLE CUSTOMER PREDICTION
# ====================================================
elif menu == "Predict Customer Cluster":

    st.subheader("📌 Cluster Definitions")
    for k, v in CLUSTER_MEANINGS.items():
        st.markdown(f"**Cluster {k}:** {v}")

    st.markdown("---")

    income = st.number_input("Income", min_value=0.0)
    age = st.number_input("Age", min_value=18)
    recency = st.number_input("Recency", min_value=0)
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
            f"🎯 Customer belongs to **Cluster {cluster}**\n\n"
            f"📌 **Meaning:** {CLUSTER_MEANINGS.get(cluster)}"
        )

# ====================================================
# ENHANCED CSV / EXCEL UPLOAD (NEW FEATURES)
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

        st.subheader("🔧 Map Columns (works for any dataset)")

        col_list = new_df.columns.tolist()

        income_col = st.selectbox("Select Income column", col_list)
        age_col = st.selectbox("Select Age column", col_list)
        recency_col = st.selectbox("Select Recency column", col_list)
        web_col = st.selectbox("Select Web Purchases column", col_list)
        store_col = st.selectbox("Select Store Purchases column", col_list)
        catalog_col = st.selectbox("Select Catalog Purchases column", col_list)

        if st.button("Run Clustering"):

            # Select and rename
            model_df = new_df[[income_col, age_col, recency_col,
                                web_col, store_col, catalog_col]].copy()

            model_df.columns = [
                'Income','Age','Recency',
                'NumWebPurchases','NumStorePurchases','NumCatalogPurchases'
            ]

            # Handle missing values
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

            st.success("✅ Clustering completed successfully")
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
