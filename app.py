import os
import pandas as pd
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler not found. Run train.py first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Must match training order
FEATURE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# 🔹 Load training data to determine an adaptive probability threshold
try:
    df_train = pd.read_csv("heart.csv")
    X = df_train[FEATURE_COLS].values
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]
    mean_prob = np.mean(probs)
    threshold = round(mean_prob, 2)  # adaptive threshold
    print(f"Adaptive probability threshold set to: {threshold}")
except Exception as e:
    print("Could not calculate adaptive threshold, using default 0.45")
    threshold = 0.45


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_single():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON provided"}), 400

    features = data.get("features")
    if not isinstance(features, (list, dict)):
        return jsonify({"error": "Invalid input format"}), 400

    try:
        if isinstance(features, dict):
            x = [float(features[c]) for c in FEATURE_COLS]
        else:
            if len(features) != len(FEATURE_COLS):
                return jsonify({"error": f"Feature list must have {len(FEATURE_COLS)} values"}), 400
            x = [float(v) for v in features]
    except Exception as e:
        return jsonify({"error": f"Invalid feature value: {e}"}), 400

    X = np.array(x).reshape(1, -1)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[0]
    disease_prob = float(proba[1])

    # 🧠 Use adaptive or fallback threshold
    pred = 1 if disease_prob > threshold else 0
    label = "Heart Disease" if pred == 1 else "No Heart Disease"
    color = "red" if pred == 1 else "green"

    print(f"Input={x}, Disease Probability={disease_prob:.2f}, Threshold={threshold}, Prediction={label}")

    return jsonify({
        "input": x,
        "prediction": label,
        "probabilities": {"no_disease": float(proba[0]), "disease": disease_prob},
        "color": color
    })


@app.route("/analytics", methods=["GET"])
def analytics():
    # Show summary of training data
    training_csv = "heart.csv"
    if not os.path.exists(training_csv):
        return jsonify({"error": "No data found"}), 400

    df = pd.read_csv(training_csv)
    numeric = df.select_dtypes(include="number")
    summary = numeric.describe().to_dict()
    return jsonify({
        "n_rows": int(df.shape[0]),
        "columns": list(df.columns),
        "summary": summary
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
