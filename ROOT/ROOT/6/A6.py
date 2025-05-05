# === STEP 1: IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from scipy import stats

# === STEP 2: LOAD DATASET ===
df = pd.read_csv(r"C:\Users\ANKUSH\OneDrive\Documents\SEM 6\SLIII\6\Iris.csv")

# === STEP 3: INITIAL DATA CHECK ===
print("Initial Data Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# Drop unnecessary 'Id' column
df.drop("Id", axis=1, inplace=True)

# === STEP 4: REMOVE OUTLIERS USING Z-SCORE ===
numeric_cols = df.select_dtypes(include=np.number).columns
z_scores = np.abs(stats.zscore(df[numeric_cols]))
df = df[(z_scores < 3).all(axis=1)]

# === STEP 5: DEFINE FEATURES & TARGET ===
X = df.drop("Species", axis=1)
y = df["Species"]

# === STEP 6: SCALING NUMERIC FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a DataFrame for scaled data + target for visualization
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled["Species"] = y.values

# === STEP 7: CLEANED DATA VISUALIZATION ===
# Pairplot
sns.pairplot(df_scaled, hue="Species")
plt.suptitle("Pairplot of Scaled Iris Data", y=1.02)
plt.show()

# Correlation heatmap (drop non-numeric column before .corr())
plt.figure(figsize=(8, 5))
sns.heatmap(df_scaled.drop("Species", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# === STEP 8: TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === STEP 9: NAIVE BAYES CLASSIFIER ===
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === STEP 10: EVALUATION ===
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')

# Print metrics
print(f"\nAccuracy     : {accuracy:.2f}")
print(f"Precision    : {precision:.2f}")
print(f"Recall       : {recall:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === STEP 11: CONFUSION MATRIX HEATMAP ===
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=df["Species"].unique(),
            yticklabels=df["Species"].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
