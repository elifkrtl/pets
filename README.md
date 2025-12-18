# Pet Adoption Dataset Explorer (Streamlit App)

This project is a **Streamlit-based exploratory data analysis (EDA) application**
designed for educational purposes in **Data Science and Machine Learning** courses.

The app provides a complete walkthrough from raw data inspection to
correlation analysis, standardization, and **Principal Component Analysis (PCA)**.

---

## Features

The application includes the following steps:

1. **Dataset Overview**
   - Dataset shape (rows & columns)
   - Column names

2. **Raw Data Preview**
   - Interactive row selection
   - Tabular visualization

3. **Descriptive Statistics**
   - Full summary using  
     `df.describe(include="all")`
   - Covers **numeric and categorical** attributes

4. **Data Type Inspection**
   - Column-wise data type listing

5. **Missing Value Analysis**
   - Missing value counts per column

6. **Numeric Feature Detection**
   - Automatic detection of numeric attributes

7. **Correlation Analysis**
   - Pearson correlation matrix
   - Heatmap visualization using Seaborn

8. **Standardization**
   - Z-score normalization applied to numeric features
   - Separate standardized dataset preview

9. **Principal Component Analysis (PCA)**
   - PCA applied on standardized numeric features
   - First two principal components (PC1 & PC2)
   - Explained variance ratio
   - 2D scatter plot visualization