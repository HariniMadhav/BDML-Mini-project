import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

DATA_PATH = "heart.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_PATH)
if "target" not in df.columns:
    raise RuntimeError("CSV must contain 'target' column.")

# 2. Define feature columns (try these common ones)
# If your CSV differs, edit this list to match your file
feature_cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
X = df[feature_cols].values
y = df["target"].values

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("Evaluation on test set:")
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
print("Confusion matrix:\n", cm)

# 7. Save model and scaler
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Saved model to {MODEL_PATH}")
print(f"Saved scaler to {SCALER_PATH}")
