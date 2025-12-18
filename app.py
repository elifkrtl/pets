# -*- coding: utf-8 -*-
"""
@author: Elif Kartal
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==============================
# App configuration
# ==============================
st.set_page_config(
    page_title="Pet Adoption Dataset Explorer",
    layout="wide"
)

st.title("🐾 Pet Adoption Dataset Explorer")
st.write(
    """
    This Streamlit application provides an exploratory view of a **pet adoption dataset**.
    The main goal is to inspect the dataset structure and summary statistics
    using `describe(include="all")`.
    """
)

# ==============================
# Load dataset
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("pet_adoption_dataset.csv")

df = load_data()

# ==============================
# Dataset overview
# ==============================
st.header("1. Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Shape")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")

with col2:
    st.subheader("Column Names")
    st.write(list(df.columns))

# ==============================
# Display raw data
# ==============================
st.header("2. Raw Data Preview")

rows_to_show = st.slider(
    "Select number of rows to display:",
    min_value=5,
    max_value=50,
    value=10
)

st.dataframe(df.head(rows_to_show), use_container_width=True)

# ==============================
# DESCRIBE (include = all)
# ==============================
st.header("3. Dataset Summary Statistics")

st.write(
    """
    The table below shows **summary statistics for all columns**  
    (numeric + categorical) using:
    ```
    df.describe(include="all")
    ```
    """
)

describe_all = df.describe(include="all")
st.dataframe(describe_all, use_container_width=True)

# ==============================
# Data types inspection
# ==============================
st.header("4. Data Types")

dtypes_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(dtypes_df, use_container_width=True)

# ==============================
# Missing values check
# ==============================
st.header("5. Missing Values Check")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum()
})

st.dataframe(missing_df, use_container_width=True)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption(
    "This app is designed for educational purposes (Data Science / ML Exploratory Data Analysis)."
)


# ==============================
# Numeric feature detection
# ==============================
st.header("6. Numeric Features & Correlation Analysis")

numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

st.subheader("Detected Numeric Features")
st.write(numeric_features)

# ==============================
# Correlation matrix
# ==============================
st.subheader("Correlation Matrix (Numeric Features)")

corr_matrix = df[numeric_features].corr()

# ==============================
# Heatmap visualization
# ==============================
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax
)

ax.set_title("Correlation Heatmap of Numeric Attributes", fontsize=14)
st.pyplot(fig)

# ==============================
# Standardization of numeric features
# ==============================
st.header("7. Standardization of Numeric Features")

st.write(
    """
    Numeric features are standardized using **Z-score normalization**:
    This transformation ensures that each numeric feature has:
    - Mean ≈ 0
    - Standard deviation ≈ 1
    """
)

# Reuse detected numeric features
scaler = StandardScaler()

standardized_values = scaler.fit_transform(df[numeric_features])

df_standardized = pd.DataFrame(
    standardized_values,
    columns=numeric_features
)

st.subheader("Standardized Numeric Data (Preview)")

rows_to_show_std = st.slider(
    "Select number of rows to display (standardized data):",
    min_value=5,
    max_value=50,
    value=10,
    key="std_rows"
)

st.dataframe(df_standardized.head(rows_to_show_std), use_container_width=True)


# ==============================
# PCA on standardized data
# ==============================
st.header("8. Principal Component Analysis (PCA)")

st.write(
    """
    Principal Component Analysis (PCA) is applied to the **standardized numeric features**
    to reduce dimensionality while preserving as much variance as possible.
    
    Here, we project the data onto **two principal components (PC1 and PC2)**.
    """
)

# Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_standardized)

pca_df = pd.DataFrame(
    pca_components,
    columns=["PC1", "PC2"]
)

# Explained variance
st.subheader("Explained Variance Ratio")

explained_variance = pca.explained_variance_ratio_

st.write(
    {
        "PC1": round(explained_variance[0], 4),
        "PC2": round(explained_variance[1], 4),
        "Total Variance Explained": round(explained_variance.sum(), 4)
    }
)

# ==============================
# PCA Scatter Plot
# ==============================
st.subheader("PCA Scatter Plot (PC1 vs PC2)")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    alpha=0.6
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Scatter Plot of First Two Principal Components")

st.pyplot(fig)
