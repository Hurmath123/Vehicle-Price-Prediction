from flask import Flask, render_template, request
import pandas as pd
import joblib
import util

app = Flask(__name__)

# Load model and metadata
model = joblib.load("xgb_vehicle_price_model.pkl")
encoder = joblib.load("onehot_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Optional: scaler
try:
    scaler = joblib.load("standard_scaler.pkl")
except:
    scaler = None

# Load unique options for dropdowns
df = pd.read_csv("dataset.csv")
makes = sorted(df["make"].dropna().unique())
models_by_make = df.groupby("make")["model"].unique().apply(list).to_dict()
trims_by_model = df.groupby("model")["trim"].unique().apply(list).to_dict()

@app.route('/')
def index():
    return render_template("index.html", makes=makes,
                           models_by_make=models_by_make,
                           trims_by_model=trims_by_model)

@app.route('/predict', methods=["POST"])
def predict():
    form_data = {
        "make": request.form["make"],
        "model": request.form["model"],
        "trim": request.form["trim"],
        "mileage": float(request.form["mileage"]),
        "vehicle_age": float(request.form["vehicle_age"])
    }

    prediction = util.predict_price(form_data, model, encoder, feature_names, scaler)
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
