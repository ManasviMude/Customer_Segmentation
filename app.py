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
# FEATURE ORDER (MUST MATCH TRAINING)
# ----------------------------------------------------
FEATURE_COLUMNS = [
    'Income_log',
    'TotalSpending',
    'Age',
    'Recency',
    'NumWebPurchases',
    'NumStorePurchases',
    'NumCatalogPurchases'
]

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
# Defensive feature engineering
# ----------------------------------------------------
# Create TotalSpending if missing
if 'TotalSpending' not in df.columns:
    spend_cols = [c for c in df.columns if c.startswith('Mnt')]
    if spend_cols:
        df['TotalSpending'] = df[spend_cols].sum(axis=1)

# Ensure Final_Cluster exists
if 'Final_Cluster' not in df.columns and 'KMeans_Cluster' in df.columns:
    df['Final_Cluster'] = df['KMeans_Cluster']

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

    st.subheader("📊 Cluster Profile (Mean Values)")
    profile_cols = [
        'Income', 'TotalSpending', 'Age', 'Recency',
        'NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases'
    ]
    profile_cols = [c for c in profile_cols if c in df.columns]
    st.dataframe(df.groupby('Final_Cluster')[profile_cols].mean().round(2))

    st.subheader("💰 Income Distribution by Cluster")
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

    st.subheader("🧍 Enter Customer Details")

    income = st.number_input("Income", min_value=0)
    total_spending = st.number_input("Total Spending", min_value=0)
    age = st.number_input("Age", min_value=18)
    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    web = st.number_input("Web Purchases", min_value=0)
    store = st.number_input("Store Purchases", min_value=0)
    catalog = st.number_input("Catalog Purchases", min_value=0)

    if st.button("Predict Cluster"):
        input_data = {
            'Income_log': np.log1p(income),
            'TotalSpending': total_spending,
            'Age': age,
            'Recency': recency,
            'NumWebPurchases': web,
            'NumStorePurchases': store,
            'NumCatalogPurchases': catalog
        }

        input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
        input_scaled = scaler.transform(input_df)
        cluster = kmeans.predict(input_scaled)[0]

        st.success(f"🎯 This customer belongs to **Cluster {cluster}**")

# ====================================================
# OPTION 3 — CSV / EXCEL UPLOAD FOR BULK PREDICTION
# ====================================================
elif menu == "Upload CSV/Excel for Prediction":

    st.subheader("📤 Upload CSV or Excel File")

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        # Read file safely
        if uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(new_df.head())

        required_cols = [
            'Income', 'TotalSpending', 'Age', 'Recency',
            'NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases'
        ]

        if all(col in new_df.columns for col in required_cols):

            new_df['Income_log'] = np.log1p(new_df['Income'])

            X_new = new_df[FEATURE_COLUMNS]
            X_scaled = scaler.transform(X_new)

            new_df['Predicted_Cluster'] = kmeans.predict(X_scaled)

            st.success("✅ Clusters assigned successfully")
            st.dataframe(new_df.head())

            csv = new_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇ Download Clustered File",
                data=csv,
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
