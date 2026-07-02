import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# Load Preprocessed Dataset
# ---------------------------------

df = pd.read_csv("preprocessed_data.csv")

print("="*60)
print("First 5 Rows")
print("="*60)
print(df.head())

# ---------------------------------
# 1. Summary Statistics
# ---------------------------------

print("\nSummary Statistics")
print("="*60)
print(df.describe())

print("\nMedian")
print("="*60)
print(df.median(numeric_only=True))

print("\nVariance")
print("="*60)
print(df.var(numeric_only=True))

print("\nSkewness")
print("="*60)
print(df.skew(numeric_only=True))

print("\nKurtosis")
print("="*60)
print(df.kurt(numeric_only=True))

# ---------------------------------
# 2. Histograms
# ---------------------------------

df.hist(figsize=(18,15), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numerical Features", fontsize=18)
plt.tight_layout()
plt.show()

# ---------------------------------
# Boxplots
# ---------------------------------

plt.figure(figsize=(16,6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplots of Numerical Features")
plt.show()

# ---------------------------------
# 3. Pairplot
# ---------------------------------

sns.pairplot(
    df[['RM','LSTAT','PTRATIO','MEDV']],
    diag_kind='hist'
)
plt.suptitle("Pairplot of Important Features", y=1.02)
plt.show()

# ---------------------------------
# Correlation Matrix
# ---------------------------------

plt.figure(figsize=(12,10))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix")
plt.show()

# ---------------------------------
# 4. Identify Strong Correlations
# ---------------------------------

print("\nStrong Positive Correlations (> 0.70)")
print("="*60)

for i in corr.columns:
    for j in corr.columns:
        if i != j and corr.loc[i, j] > 0.70:
            print(f"{i} <--> {j}: {corr.loc[i,j]:.2f}")

print("\nStrong Negative Correlations (< -0.70)")
print("="*60)

for i in corr.columns:
    for j in corr.columns:
        if i != j and corr.loc[i, j] < -0.70:
            print(f"{i} <--> {j}: {corr.loc[i,j]:.2f}")

# ---------------------------------
# 5. Feature-Level Inferences
# ---------------------------------

print("\nBasic Feature-Level Inferences")
print("="*60)

print("""
1. Features with positive correlation to MEDV
   tend to increase house prices.

2. Features with negative correlation to MEDV
   tend to decrease house prices.

3. Histograms help identify:
   - Normal distributions
   - Skewed distributions
   - Possible outliers

4. Boxplots reveal:
   - Remaining outliers
   - Spread of each feature
   - Median values

5. Pairplot helps visualize:
   - Linear relationships
   - Clusters
   - Trends between variables

6. Correlation heatmap identifies:
   - Strongly related variables
   - Multicollinearity
   - Important predictors of house price.
""")

print("\nEDA Completed Successfully!")