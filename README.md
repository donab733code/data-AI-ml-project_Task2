# 📊 Housing Dataset - Exploratory Data Analysis (EDA)

## 📌 Project Overview
This project performs **Exploratory Data Analysis (EDA)** on a preprocessed housing dataset to understand data distribution, relationships between features, and key patterns affecting house prices.

The analysis helps in identifying important features for building machine learning models.

---

## 🎯 Objectives
- Load and explore preprocessed dataset
- Compute summary statistics
- Analyze data distribution using histograms
- Detect outliers using boxplots
- Study relationships using pairplots
- Analyze feature correlations using heatmaps
- Extract meaningful insights from data

---

## 🛠️ Tools & Libraries Used
- Python 🐍
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 📂 Dataset
The dataset used is `preprocessed_data.csv`, which is already cleaned and prepared from the previous preprocessing stage.

---

## 🔍 Exploratory Data Analysis Steps

### 1. Data Overview
- Loaded dataset using Pandas
- Displayed first 5 rows
- Generated summary statistics:
  - Mean
  - Median
  - Variance
  - Skewness
  - Kurtosis

---

### 2. Distribution Analysis
- Plotted histograms for all numerical features
- Observed:
  - Data distribution patterns
  - Skewed features
  - Potential outliers

---

### 3. Outlier Detection
- Used **boxplots** to visualize feature spread
- Identified remaining extreme values and distribution range

---

### 4. Relationship Analysis
- Used **pairplot** on key features:
  - RM
  - LSTAT
  - PTRATIO
  - MEDV
- Observed:
  - Linear relationships
  - Trends between features and target variable

---

### 5. Correlation Analysis
- Created **correlation heatmap**
- Identified:
  - Strong positive correlations
  - Strong negative correlations
  - Multicollinearity between features

---

### 6. Feature Insights
- Features positively correlated with house price tend to increase value
- Features negatively correlated reduce price
- Correlation helps identify important predictors for ML models

---

## 📊 Visualizations Used
- Histograms → Data distribution
- Boxplots → Outliers detection
- Pairplot → Feature relationships
- Heatmap → Correlation analysis

---

## 📁 Output
No new dataset is generated in this step.

This stage provides:
- Insights for feature selection
- Understanding of data behavior
- Input for machine learning modeling

---

## 🚀 How to Run the Project

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn
