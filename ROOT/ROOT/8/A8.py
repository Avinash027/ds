import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np  # Importing numpy to avoid NameError

# Load dataset
titanic = sns.load_dataset('titanic')

# === STEP 1: Data Preprocessing ===
# Check for missing values
print("Missing values in each column:\n", titanic.isnull().sum())

# Check data types and basic info
print("\nBasic Info:\n", titanic.info())

# === 1.1 Handle Missing Values ===
# Handle 'age' and 'embarked' columns (common for nulls in Titanic dataset)
titanic['age'] = titanic['age'].fillna(titanic['age'].median())  # Fill missing age with median
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])  # Fill missing embarked with mode

# Drop rows with missing target 'survived' or 'class'
titanic.dropna(subset=['survived', 'class'], inplace=True)

# Check again after handling missing values
print("\nMissing values after handling:\n", titanic.isnull().sum())

# === 1.2 Remove Duplicates ===
titanic.drop_duplicates(inplace=True)

# === 1.3 Outlier Detection and Handling ===
# Using Z-score method to detect outliers
z_scores = stats.zscore(titanic.select_dtypes(include=['float64', 'int64']))  # Only numeric columns
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)  # Consider Z-scores < 3 as non-outliers
titanic = titanic[filtered_entries]  # Remove outliers

# === 1.4 Scaling and Normalization ===
# Standardizing 'age' and 'fare' columns
scaler = StandardScaler()
titanic[['age', 'fare']] = scaler.fit_transform(titanic[['age', 'fare']])

# Check first few rows after preprocessing
print("\nData after Preprocessing:\n", titanic.head())

# === STEP 2: Data Visualization ===
# 2.1 Barplot: Age vs Sex
plt.figure(figsize=(8, 6))
sns.barplot(x="sex", y="age", hue="sex", data=titanic)
plt.title("Barplot: Age vs Sex")
plt.show()

# 2.2 Catplot: Count of survivors vs Sex
sns.catplot(x="sex", hue="survived", kind="count", data=titanic)
plt.title("Catplot: Count of Survivors by Sex")
plt.show()

# 2.3 Histplot: Fare distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=titanic, x="fare")
plt.title("Histplot: Fare Distribution")
plt.show()

# 2.4 Histplot with bins for better granularity
plt.figure(figsize=(8, 6))
sns.histplot(data=titanic, x="fare", bins=30)  # Use bins instead of binwidth
plt.title("Histplot: Fare Distribution (With Bins)")
plt.show()

# Show all plots at once
plt.show()
