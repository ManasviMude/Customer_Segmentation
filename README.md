# 📊 Customer Segmentation using K-Means

This project implements an end-to-end **Customer Segmentation system** using **K-Means clustering**, with full deployment on **Streamlit Cloud**.

---

## 🚀 Features

- Exploratory Data Analysis (EDA)
- K-Means clustering for customer segmentation
- Cluster interpretation with business meanings
- Business recommendations for each cluster
- Interactive Streamlit dashboard
- Supports CSV and Excel uploads
- Donut charts with percentage-based visualization
- Download clustered datasets

---

## 🧠 Cluster Overview

| Cluster | Segment Name | Description | Business Recommendation |
|-------|--------------|-------------|-------------------------|
| 0 | Low Value Customers | Low income, low spending | Discounts & basic offers |
| 1 | High Value Customers | High income, high spending | Premium & loyalty programs |
| 2 | Regular Customers | Medium income, regular spenders | Upselling & cross-selling |
| 3 | Potential Customers | High income, low spending | Personalized promotions |

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
