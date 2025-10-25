# iris_classification.py
# ------------------------------
# Mini Project: Iris Flower Classification
# ------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
data = sns.load_dataset("iris")  # built-in dataset
print("🔹 First 5 rows of dataset:\n", data.head(), "\n")

# Step 2: Check dataset info
print("Dataset Info:\n")
print(data.info(), "\n")

# Step 3: Encode target variable (species)
le = LabelEncoder()
data["species"] = le.fit_transform(data["species"])

# Step 4: Split into features and target
X = data.drop("species", axis=1)
y = data["species"]

# Step 5: Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%\n")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Visualization 1 - Pairplot
sns.pairplot(data, hue="species", palette="husl")
plt.suptitle("Iris Flower Features Visualization", y=1.02)
plt.show()

# Step 10: Visualization 2 - Feature Importance
plt.figure(figsize=(7,4))
importances = model.feature_importances_
sns.barplot(x=importances, y=X.columns, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.show()
