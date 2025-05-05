import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# === STEP 1: Data Preprocessing ===

# 1.1 Check for missing values
print("Missing values in each column:\n", titanic.isnull().sum())

# 1.2 Handle Missing Values
# Fill missing 'age' with median
titanic['age'] = titanic['age'].fillna(titanic['age'].median())

# Fill missing 'embarked' with mode
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])

# Drop rows with missing 'survived' or 'class' (critical columns for analysis)
titanic.dropna(subset=['survived', 'class'], inplace=True)

# 1.3 Handle Duplicates
titanic.drop_duplicates(inplace=True)

# 1.4 Outlier Detection using Z-scores
z_scores = stats.zscore(titanic.select_dtypes(include=['float64', 'int64']))  # Only numeric columns
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)  # Consider Z-scores < 3 as non-outliers
titanic = titanic[filtered_entries]  # Remove outliers

# 1.5 Scaling and Normalization
scaler = StandardScaler()
titanic[['age', 'fare']] = scaler.fit_transform(titanic[['age', 'fare']])

# Check for missing values after cleaning
print("\nMissing values after handling:\n", titanic.isnull().sum())

# === STEP 2: Data Visualization ===

# Barplot: average age by sex
plt.figure(figsize=(8, 6))
sns.barplot(x="sex", y="age", hue="sex", data=titanic)
plt.title("Average Age by Sex")
plt.show()

# Boxplot: age distribution by sex
plt.figure(figsize=(8, 6))
sns.boxplot(x="sex", y="age", hue="sex", data=titanic)
plt.title("Age Distribution by Sex")
plt.show()

# Boxplot: age distribution by sex and survival
plt.figure(figsize=(8, 6))
sns.boxplot(x="sex", y="age", hue="survived", data=titanic)
plt.title("Age Distribution by Sex and Survival")
plt.show()

# Show all plots at once (unnecessary as plt.show() is already called above for each)
