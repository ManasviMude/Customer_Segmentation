import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation using K-Means Clustering")

# ----------------------------------------------------
# FEATURE ORDER (EXACT TRAINING ORDER)
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
# Load data & models
# ----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("final_customer_segmentation_output.csv")

df = load_data()
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------------------------------
# Defensive feature creation
# ----------------------------------------------------
if 'TotalSpending' not in df.columns:
    spend_cols = [c for c in df.columns if c.startswith("Mnt")]
    if spend_cols:
        df['TotalSpending'] = df[spend_cols].sum(axis=1)

if 'Final_Cluster' not in df.columns and 'KMeans_Cluster' in df.columns:
    df['Final_Cluster'] = df['KMeans_Cluster']

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
    cols = ['Income','TotalSpending','Age','Recency',
            'NumWebPurchases','NumStorePurchases','NumCatalogPurchases']
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df.groupby('Final_Cluster')[cols].mean().round(2))

    st.subheader("💰 Income by Cluster")
    fig, ax = plt.subplots()
    sns.boxplot(x='Final_Cluster', y='Income', data=df, ax=ax)
    st.pyplot(fig)

# ====================================================
# SINGLE CUSTOMER PREDICTION (FIXED)
# ====================================================
elif menu == "Predict Customer Cluster":

    income = st.number_input("Income", min_value=0.0)
    total_spending = st.number_input("Total Spending", min_value=0.0)
    age = st.number_input("Age", min_value=18)
    recency = st.number_input("Recency", min_value=0)
    web = st.number_input("Web Purchases", min_value=0)
    store = st.number_input("Store Purchases", min_value=0)
    catalog = st.number_input("Catalog Purchases", min_value=0)

    if st.button("Predict Cluster"):

        input_values = [
            np.log1p(income),
            total_spending,
            age,
            recency,
            web,
            store,
            catalog
        ]

        # Convert to numpy (avoids sklearn feature-name crash)
        X_input = np.array(input_values, dtype=float).reshape(1, -1)

        X_scaled = scaler.transform(X_input)
        cluster = int(kmeans.predict(X_scaled)[0])

        st.success(f"🎯 Customer belongs to **Cluster {cluster}**")

# ====================================================
# CSV / EXCEL UPLOAD (FIXED)
# ====================================================
elif menu == "Upload CSV/Excel for Prediction":

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        # Read file safely
        if uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)

        st.dataframe(new_df.head())

        required = [
            'Income','TotalSpending','Age','Recency',
            'NumWebPurchases','NumStorePurchases','NumCatalogPurchases'
        ]

        if all(col in new_df.columns for col in required):

            new_df = new_df.copy()
            new_df['Income_log'] = np.log1p(new_df['Income'])

            X_new = new_df[FEATURE_COLUMNS].astype(float).values
            X_scaled = scaler.transform(X_new)

            new_df['Predicted_Cluster'] = kmeans.predict(X_scaled)

            st.success("✅ Clusters assigned successfully")
            st.dataframe(new_df.head())

            st.download_button(
                "⬇ Download Results",
                new_df.to_csv(index=False),
                "clustered_customers.csv",
                "text/csv"
            )
        else:
            st.error("❌ File missing required columns.")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.write("🚀 Deployed using Streamlit Cloud & GitHub")
st.write("📌 Final Model: K-Means Clustering")
