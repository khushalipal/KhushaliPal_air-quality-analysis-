"""
app.py  –  Flask REST API for Air Quality Prediction
=====================================================
Endpoints
---------
GET  /api/health            – liveness probe
GET  /api/cities            – list of all cities
GET  /api/city_defaults/<city> – median pollutant values for a city
POST /api/predict           – predict AQI + bucket from pollutant values
GET  /api/city_trend/<city> – historical AQI data for a city
GET  /api/summary           – overall dataset stats
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── load artefacts ─────────────────────────────────────────────────────────────
MODEL_DIR = "models"
DATA_PATH = "city_day.csv"

reg_model   = joblib.load(os.path.join(MODEL_DIR, "aqi_regressor.pkl"))
clf_model   = joblib.load(os.path.join(MODEL_DIR, "aqi_classifier.pkl"))
le          = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
FEATURE_COLS= joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
city_list   = joblib.load(os.path.join(MODEL_DIR, "city_list.pkl"))
city_medians= pd.read_csv(os.path.join(MODEL_DIR, "city_medians.csv"), index_col="City")
df_full     = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# AQI bucket → colour mapping
BUCKET_COLOURS = {
    "Good":         "#00b050",
    "Satisfactory": "#92d050",
    "Moderate":     "#ffff00",
    "Poor":         "#ff7c00",
    "Very Poor":    "#ff0000",
    "Severe":       "#7030a0",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def aqi_to_bucket(aqi: float) -> str:
    """Rule-based fallback bucket from numeric AQI."""
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
    return "Severe"

def health_advice(bucket: str) -> str:
    advice = {
        "Good":         "Air quality is good. Enjoy outdoor activities.",
        "Satisfactory": "Air quality is acceptable. Sensitive people should reduce prolonged outdoor exertion.",
        "Moderate":     "Moderate health concern. Sensitive groups should limit outdoor activity.",
        "Poor":         "Everyone may begin to experience health effects. Reduce outdoor activity.",
        "Very Poor":    "Health alert! Avoid outdoor activities. Wear a mask if going outside.",
        "Severe":       "Emergency conditions! Stay indoors. Avoid all outdoor exposure.",
    }
    return advice.get(bucket, "No advice available.")

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "random_forest_v1"})


@app.route("/api/cities", methods=["GET"])
def cities():
    return jsonify({"cities": city_list})


@app.route("/api/city_defaults/<city>", methods=["GET"])
def city_defaults(city):
    if city not in city_medians.index:
        return jsonify({"error": f"City '{city}' not found"}), 404
    row = city_medians.loc[city].fillna(0).to_dict()
    return jsonify({"city": city, "defaults": row})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        values = [float(data.get(col, 0) or 0) for col in FEATURE_COLS]
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    X = pd.DataFrame([values], columns=FEATURE_COLS)

    aqi_pred   = float(reg_model.predict(X)[0])
    bucket_enc = int(clf_model.predict(X)[0])
    bucket     = le.inverse_transform([bucket_enc])[0]
    colour     = BUCKET_COLOURS.get(bucket, "#888888")
    advice     = health_advice(bucket)

    # confidence: probability of the predicted class
    proba      = clf_model.predict_proba(X)[0]
    confidence = float(proba[bucket_enc]) * 100

    return jsonify({
        "aqi":        round(aqi_pred, 1),
        "bucket":     bucket,
        "colour":     colour,
        "confidence": round(confidence, 1),
        "advice":     advice,
        "features":   dict(zip(FEATURE_COLS, values)),
    })


@app.route("/api/city_trend/<city>", methods=["GET"])
def city_trend(city):
    df_city = df_full[df_full["City"] == city][["Date", "AQI", "AQI_Bucket", "PM2.5", "PM10"]].dropna(subset=["AQI"])
    if df_city.empty:
        return jsonify({"error": f"No data for city '{city}'"}), 404

    df_city = df_city.sort_values("Date")
    return jsonify({
        "city":   city,
        "dates":  df_city["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "aqi":    df_city["AQI"].round(1).tolist(),
        "pm25":   df_city["PM2.5"].fillna(0).round(1).tolist(),
        "pm10":   df_city["PM10"].fillna(0).round(1).tolist(),
        "bucket": df_city["AQI_Bucket"].fillna("Unknown").tolist(),
    })


@app.route("/api/summary", methods=["GET"])
def summary():
    valid = df_full.dropna(subset=["AQI"])
    bucket_counts = valid["AQI_Bucket"].value_counts().to_dict()
    city_avg = (
        valid.groupby("City")["AQI"]
        .mean()
        .round(1)
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )
    return jsonify({
        "total_records":  int(len(df_full)),
        "valid_aqi":      int(len(valid)),
        "cities":         len(city_list),
        "date_range":     {
            "start": df_full["Date"].min().strftime("%Y-%m-%d"),
            "end":   df_full["Date"].max().strftime("%Y-%m-%d"),
        },
        "bucket_distribution": bucket_counts,
        "top10_polluted_cities": city_avg,
    })


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Air Quality Prediction API on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
