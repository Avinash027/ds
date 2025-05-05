import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# -----------------------------
# 1. Adult Dataset - Preprocessing
# -----------------------------

# Load dataset
df = pd.read_csv(r"C:\Users\ANKUSH\OneDrive\Documents\SEM 6\SLIII\3\adult.csv")

# Show basic info
print("\n[Adult.csv] Dataset Info:")
print(df.info())

# Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Drop rows with missing values for simplicity
df.dropna(inplace=True)

# Convert relevant numeric columns
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Detect outliers in 'age' using IQR
Q1 = df["age"].quantile(0.25)
Q3 = df["age"].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df = df[(df["age"] >= Q1 - 1.5 * IQR) & (df["age"] <= Q3 + 1.5 * IQR)]

# Normalize age column using MinMaxScaler
scaler = MinMaxScaler()
df["age_scaled"] = scaler.fit_transform(df[["age"]])

# -----------------------------
# Summary Statistics
# -----------------------------

print("\nSummary stats grouped by gender:")
group = df.groupby("gender")["age"]
print("Mean:\n", group.mean())
print("Median:\n", group.median())
print("Min:\n", group.min())
print("Max:\n", group.max())
print("Std:\n", group.std())

# Summary stats by marital-status
print("\nGrouped by marital-status:\n")
print(df.groupby("marital-status")["age"].agg(["mean", "median", "min", "max", "std"]))

# Group by gender and income
print("\nGrouped by gender and income:\n")
print(df.groupby(["gender", "income"])["age"].agg(["mean", "median", "min", "max", "std"]))

# -----------------------------
# 2. Iris Dataset - Preprocessing
# -----------------------------

# Load dataset
df2 = pd.read_csv(r"C:\Users\ANKUSH\OneDrive\Documents\SEM 6\SLIII\3\iris.csv")

# Show basic info
print("\n[Iris.csv] Dataset Info:")
print(df2.info())

# Check and drop nulls
print("\nMissing values in iris:")
print(df2.isnull().sum())
df2.dropna(inplace=True)

# Outlier detection for SepalLengthCm
num_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
for col in num_cols:
    Q1 = df2[col].quantile(0.25)
    Q3 = df2[col].quantile(0.75)
    IQR = Q3 - Q1
    df2 = df2[(df2[col] >= Q1 - 1.5 * IQR) & (df2[col] <= Q3 + 1.5 * IQR)]

# Scaling the numeric features using StandardScaler
scaler = StandardScaler()
df2[num_cols] = scaler.fit_transform(df2[num_cols])

# -----------------------------
# Statistical Description by Species
# -----------------------------

# Without groupby()
print("\nDescriptive stats (WITHOUT groupby):")
for species in df2["Species"].unique():
    print(f"\nSpecies: {species}")
    print(df2[df2["Species"] == species].describe())

# With groupby
print("\nDescriptive stats (WITH groupby):")
print(df2.groupby("Species")[num_cols].agg(["mean", "std", "min", "max", "median", "quantile"]))

# Example for quantile percentiles
print("\nSepalLengthCm - 25th and 75th percentile by Species:")
print(df2.groupby("Species")["SepalLengthCm"].quantile(0.25))
print(df2.groupby("Species")["SepalLengthCm"].quantile(0.75))
