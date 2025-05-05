# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\ANKUSH\OneDrive\Documents\SEM 6\SLIII\5\Social_Network_Ads.csv")  # Or your dataset name

# -------------------------------
# Step 1: Explore data
# -------------------------------
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# -------------------------------
# Step 2: Handle missing values (if any)
# -------------------------------
# (Assume none based on common datasets like Social_Network_Ads.csv)

# -------------------------------
# Step 3: Handle irrelevant or categorical data
# -------------------------------
# Drop User ID if present
if "User ID" in df.columns:
    df.drop("User ID", axis=1, inplace=True)

# Encode Gender: Male=1, Female=0
df["Gender"] = df["Gender"].astype("category").cat.codes

# -------------------------------
# Step 4: Outlier Detection (IQR)
# -------------------------------
numeric_cols = ["Age", "EstimatedSalary"]
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Remove outliers
df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# -------------------------------
# Step 5: Feature scaling
# -------------------------------
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 6: Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 7: Train Logistic Regression
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 8: Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print metrics
metrics_dict = {
    "True Negatives (TN)": tn,
    "False Positives (FP)": fp,
    "False Negatives (FN)": fn,
    "True Positives (TP)": tp,
    "Accuracy": accuracy,
    "Error Rate": error_rate,
    "Precision": precision,
    "Recall": recall
}
print("\nPerformance Metrics:")
for key, value in metrics_dict.items():
    print(f"{key:<25}: {value:.4f}" if isinstance(value, float) else f"{key:<25}: {value}")

# -------------------------------
# Step 9: Confusion Matrix Plot
# -------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Purchased", "Purchased"],
            yticklabels=["Not Purchased", "Purchased"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
