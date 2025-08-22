# water_potability_model.py

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib   # <-- Added to save model

# 1. Load dataset
df_water = pd.read_csv(r"C:\Users\hp\Desktop\AI\Quiz\Quiz 11\water_potability.csv")
print("First 5 rows of dataset:")
print(df_water.head())

# 2. Data Cleaning
print("\nNaN values in Water dataset:")
print(df_water.isna().sum())

print("\nDataset info before cleaning:")
print(df_water.info())

# Fill missing values with mean
df_clean = df_water.fillna(df_water.mean())
print("\nSum of Null values after cleaning = ", df_clean.isnull().sum().sum())

print("\nDataset info after cleaning:")
print(df_clean.info())

# 3. Feature Selection
plt.figure(figsize=(10, 5))
sns.heatmap(df_clean.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# 4. Split data
X = df_clean.drop("Potability", axis=1)  # Features
y = df_clean["Potability"]               # Target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nNo of records in training data:", x_train.shape[0])
print("No of records in testing data:", x_test.shape[0])

# 5. Train RandomForest
model2 = RandomForestClassifier(random_state=42)
model2.fit(x_train, y_train)

# 6. Predictions & Accuracy
pred2 = model2.predict(x_test)
print(f"\nAccuracy: {round(100 * accuracy_score(y_test, pred2), 2)} %")

# 7. Confusion Matrix
plt.figure(figsize=(5, 3))
sns.heatmap(confusion_matrix(y_test, pred2), annot=True, fmt='2g', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8. Test on a sample record
sample = x_test.iloc[0, :]  # take first test sample
print("\nSample features:\n", sample)
print("Model prediction for this sample:", model2.predict([sample])[0])

# 9. Save the trained model, features for later use in Streamlit
joblib.dump(model2, "water_potability_model.joblib")
joblib.dump(X.columns.tolist(), "feature_names.joblib")

