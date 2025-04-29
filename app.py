from flask import Flask, render_template, request, url_for
import pandas as pd
import joblib
import os
import util

app = Flask(__name__)

# --- Load Model & Artifacts ---
model_path = "vehicle_price_model.pkl"
encoder_path = "onehot_encoder.pkl"
features_path = "feature_names.pkl"

# Load model and supporting files
if not all(os.path.exists(p) for p in [model_path, encoder_path, features_path]):
    raise FileNotFoundError("❌ Required .pkl file(s) not found.")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
feature_names = joblib.load(features_path)

# Optional: load scaler
try:
    scaler = joblib.load("standard_scaler.pkl")
except:
    scaler = None

# --- Load Dropdown Options ---
df = pd.read_csv("dataset.csv")
makes = sorted(df["make"].dropna().unique())
models_by_make = df.groupby("make")["model"].unique().apply(list).to_dict()
trims_by_model = df.groupby("model")["trim"].unique().apply(list).to_dict()

# --- Routes ---
@app.route('/')
def index():
    return render_template("index.html",
                           makes=makes,
                           models_by_make=models_by_make,
                           trims_by_model=trims_by_model)

@app.route('/predict', methods=["POST"])
def predict():
    form_data = {
        "make": request.form["make"],
        "model": request.form["model"],
        "trim": request.form["trim"],
        "mileage": float(request.form["mileage"]),
        "vehicle_age": float(request.form["vehicle_age"]),
    }

    prediction = util.predict_price(form_data, model, encoder, feature_names, scaler)
    return render_template("result.html", prediction=prediction)

# --- Run App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Deployment platforms set this
    app.run(host="0.0.0.0", port=port)
