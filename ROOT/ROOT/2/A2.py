import pandas as pd
import numpy as np

# Create a sample "Academic performance" dataset
students_data = {
    "age": [18, 21, 19, 22, 20, 17, 24, 21, 23, 19],
    "gender": ["M", "F", "M", "F", "M", "M", "F", "F", "M", "F"],
    "math_score": [80, 75, 90, 85, 88, 70, 95, 78, 84, 92],
    "reading_score": [85, 78, 92, 88, 91, 72, 94, 79, 85, 90],
    "writing_score": [78, 80, 85, 84, 86, 68, 93, 77, 83, 89]
}

# Create a DataFrame
df = pd.DataFrame(students_data)

# Introduce some NaN values randomly
np.random.seed(42)
missing_indices = np.random.choice(df.index, size=3, replace=False)
for idx in missing_indices:
    df['age'][idx] = np.nan

# Show DataFrame with missing values
print("Initial DataFrame with Missing Values:")
print(df)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values in 'age' column with the mean value
df['age'].fillna(df['age'].mean(), inplace=True)

# Check again for missing values after imputation
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Check for inconsistencies (e.g., age should be between 18 and 22)
print("\nInconsistencies in age:")
print(df[(df['age'] < 18) | (df['age'] > 22)])  # Fixed condition with parentheses

# Fix inconsistency by replacing ages outside the range with NaN (just as an example)
df['age'] = df['age'].apply(lambda x: np.nan if x < 18 or x > 22 else x)

# Drop rows with NaN values (for this example, it's just one row)
df.dropna(inplace=True)

# Display cleaned DataFrame
print("\nCleaned DataFrame (After Fixing Inconsistencies and Removing NaN Rows):")
print(df)

# Standardize the column names
df.columns = df.columns.str.replace(' ', '_')
print("\nStandardized Column Names:")
print(df.columns)

# Data Wrangling Task: Boxplot for numeric columns (outlier detection)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(data=df.select_dtypes(["float64", "int64"]))
plt.title('Boxplot of Numeric Columns')
plt.show()

# Data Visualization: Distribution of each score
plt.figure(figsize=(10, 6))
sns.histplot(df['math_score'], kde=True, color='blue', bins=10, label='Math Score')
sns.histplot(df['reading_score'], kde=True, color='green', bins=10, label='Reading Score')
sns.histplot(df['writing_score'], kde=True, color='red', bins=10, label='Writing Score')
plt.legend()
plt.title('Distribution of Scores')
plt.show()
