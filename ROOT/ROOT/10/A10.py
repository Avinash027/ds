import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load the dataset
df = pd.read_csv(r"C:\Users\ANKUSH\OneDrive\Documents\SEM 6\SLIII\10\Iris.csv")

# Step 1: Data Preprocessing
print("Initial Data:")
print(df.head())
print("\nData Types:")
print(df.dtypes)

# 1.1. Drop the 'Id' column as it is unnecessary
df.drop("Id", axis=1, inplace=True)

# 1.2. Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Since there are no missing values in the Iris dataset, we'll proceed with further preprocessing.

# 1.3. Handle Outliers using Z-scores
z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))  # Only numeric columns
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)  # Consider Z-scores < 3 as non-outliers
df = df[filtered_entries]  # Remove rows with outliers

# 1.4. Scaling and Normalization
scaler = StandardScaler()
df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = scaler.fit_transform(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])

# Step 2: Data Visualization

# 2.1. Visualize distribution of each numeric feature using histograms
numeric_features = df.columns[:-1]  # Exclude 'Species' column
plt.figure(figsize=(10, 8))
for i, feature in enumerate(numeric_features):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=df, x=feature, bins=20, alpha=1.0)
    plt.title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()

# 2.2. Visualize boxplots to identify outliers
plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x=df[feature], data = df)
    plt.title(f"Boxplot of {feature}")
plt.tight_layout()
plt.show()

# 2.3. Identify and visualize outliers in SepalWidthCm column
print("\nOutliers in SepalWidthCm column (greater than 4.0):")
print(df[df["SepalWidthCm"] > 4.0])

print("\nOutliers in SepalWidthCm column (less than or equal to 2.0):")
print(df[df["SepalWidthCm"] <= 2.0])

# 2.4. Plot the relationship between features using pairplot
sns.pairplot(df, hue="Species")
plt.show()

