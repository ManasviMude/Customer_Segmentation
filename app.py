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
# Cluster meanings (BUSINESS INTERPRETATION)
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
# Ensure required columns for visualization
# ----------------------------------------------------
if 'Final_Cluster' not in df.columns and 'KMeans_Cluster' in df.columns:
    df['Final_Cluster'] = df['KMeans_Cluster']

if 'TotalSpending' not in df.columns:
    spend_cols = [c for c in df.columns if c.startswith('Mnt')]
    if spend_cols:
        df['TotalSpending'] = df[spend_cols].sum(axis=1)

# ----------------------------------------------------
# Sidebar menu
# ----------------------------------------------------
st.sidebar.title("🔧 Menu")
menu = st.sidebar.radio(
    "Select Option",
    ["View Clusters", "Predict Customer Cluster", "Upload CSV/Excel for Prediction"]
)

# ====================================================
# OPTION 1 — VIEW CLUSTERS
# ====================================================
if menu == "View Clusters":

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Cluster Distribution")
    st.bar_chart(df['Final_Cluster'].value_counts().sort_index())

    st.subheader("📊 Cluster Profile")
    profile_cols = [
        'Income',
        'TotalSpending',
        'Age',
        'Recency',
        'NumWebPurchases',
        'NumStorePurchases',
        'NumCatalogPurchases'
    ]
    profile_cols = [c for c in profile_cols if c in df.columns]
    st.dataframe(df.groupby('Final_Cluster')[profile_cols].mean().round(2))

    st.subheader("💰 Income by Cluster")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x='Final_Cluster', y='Income', data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("🛒 Total Spending by Cluster")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Final_Cluster', y='TotalSpending', data=df, ax=ax2)
    st.pyplot(fig2)

# ====================================================
# OPTION 2 — SINGLE CUSTOMER PREDICTION
# ====================================================
elif menu == "Predict Customer Cluster":

    # ---- Show cluster meanings first ----
    st.subheader("📌 Cluster Definitions")
    for k, v in CLUSTER_MEANINGS.items():
        st.markdown(f"**Cluster {k}:** {v}")

    st.markdown("---")

    st.subheader("🧍 Enter Customer Details")

    income = st.number_input("Income", min_value=0.0)
    age = st.number_input("Age", min_value=18)
    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    web = st.number_input("Web Purchases", min_value=0)
    store = st.number_input("Store Purchases", min_value=0)
    catalog = st.number_input("Catalog Purchases", min_value=0)

    if st.button("Predict Cluster"):

        # EXACT same features used during training (6 features)
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
        cluster_desc = CLUSTER_MEANINGS.get(cluster, "Description not available")

        st.success(
            f"🎯 **Customer belongs to Cluster {cluster}**\n\n"
            f"📌 **Cluster Meaning:** {cluster_desc}"
        )

# ====================================================
# OPTION 3 — CSV / EXCEL UPLOAD FOR BULK PREDICTION
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

        required_cols = [
            'Income',
            'Age',
            'Recency',
            'NumWebPurchases',
            'NumStorePurchases',
            'NumCatalogPurchases'
        ]

        if all(col in new_df.columns for col in required_cols):

            X_new = np.column_stack([
                np.log1p(new_df['Income']),
                new_df['Age'],
                new_df['Recency'],
                new_df['NumWebPurchases'],
                new_df['NumStorePurchases'],
                new_df['NumCatalogPurchases']
            ])

            X_scaled = scaler.transform(X_new)
            new_df['Predicted_Cluster'] = kmeans.predict(X_scaled)
            new_df['Cluster_Meaning'] = new_df['Predicted_Cluster'].map(CLUSTER_MEANINGS)

            st.success("✅ Clusters assigned successfully")
            st.dataframe(new_df.head())

            st.download_button(
                label="⬇ Download Clustered File",
                data=new_df.to_csv(index=False),
                file_name="clustered_customers.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ Uploaded file is missing required columns.")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.write("🚀 Deployed using Streamlit Cloud & GitHub")
st.write("📌 Final Model: K-Means Clustering")
