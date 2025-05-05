# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Boston Housing Dataset
# -----------------------------
boston = fetch_openml(name='boston', version=1, as_frame=True)
data = boston.frame

print("Initial Dataset Info:")
print(data.info())

# -----------------------------
# Handle Missing Values
# -----------------------------
print("\nMissing values:")
print(data.isnull().sum())  # Boston dataset typically has no missing values

# -----------------------------
# Outlier Detection (IQR Method)
# -----------------------------
# Numerical columns only
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Filter to remove outliers
data_no_outliers = data[~((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print(f"\nOriginal shape: {data.shape}, After removing outliers: {data_no_outliers.shape}")

# -----------------------------
# Scaling (Standardization)
# -----------------------------
X = data_no_outliers.drop(columns=["MEDV"])
y = data_no_outliers["MEDV"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, random_state=42)

# -----------------------------
# Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"✅ R^2 Score: {r2:.2f}")

# -----------------------------
# Optional: Residual Plot
# -----------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([0, 50], [0, 50], '--', color='red')
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted Home Prices")
plt.grid(True)
plt.show()
