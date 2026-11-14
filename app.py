import os, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from scipy.sparse import hstack, csr_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
app = Flask(__name__)

TEXT_COLUMNS = ["city_name", "admin_ward", "dem_source", "land_use",
                "soil_group", "storm_drain_type", "rainfall_source"]
NUMERIC_COLUMNS = ["latitude", "longitude", "elevation_m",
                   "drainage_density_km_per_km2", "storm_drain_proximity_m",
                   "historical_rainfall_intensity_mm_hr", "return_period_years"]

def load_artifacts():
    required = ["tfidf_vectorizer.joblib", "numeric_scaler.joblib",
                "xgboost_models.joblib", "evaluation_report.json"]
    missing = [x for x in required if not os.path.exists(os.path.join(MODEL_DIR, x))]
    if missing:
        raise RuntimeError("Model files are missing. Run: python train_model.py")
    vec = joblib.load(os.path.join(MODEL_DIR, required[0]))
    scaler = joblib.load(os.path.join(MODEL_DIR, required[1]))
    models = joblib.load(os.path.join(MODEL_DIR, required[2]))
    with open(os.path.join(MODEL_DIR, required[3]), encoding="utf-8") as f:
        report = json.load(f)
    return vec, scaler, models, report

def predict_record(values):
    vec, scaler, models, report = load_artifacts()
    row = pd.DataFrame([values])
    text = row[TEXT_COLUMNS].fillna("unknown").astype(str).agg(" ".join, axis=1)
    xt = vec.transform(text)
    xn = scaler.transform(row[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0))
    x = hstack([xt, csr_matrix(xn)]).tocsr()
    found, scores = [], {}
    for tag, model in models.items():
        probability = float(model.predict_proba(x)[0, 1])
        scores[tag] = round(probability * 100, 1)
        if probability >= 0.5:
            found.append(tag)
    return found, scores, report

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    defaults = {
        "city_name": "Bengaluru", "admin_ward": "Ward 1",
        "dem_source": "DEM", "land_use": "Residential",
        "soil_group": "B", "storm_drain_type": "Covered",
        "rainfall_source": "Rain gauge", "latitude": "12.97",
        "longitude": "77.59", "elevation_m": "900",
        "drainage_density_km_per_km2": "1.5",
        "storm_drain_proximity_m": "100",
        "historical_rainfall_intensity_mm_hr": "50",
        "return_period_years": "5"
    }
    if request.method == "POST":
        try:
            values = {k: request.form.get(k, "").strip() for k in TEXT_COLUMNS}
            values.update({k: float(request.form.get(k, "0") or 0) for k in NUMERIC_COLUMNS})
            tags, scores, report = predict_record(values)
            result = {"tags": tags, "scores": scores, "report": report}
        except Exception as exc:
            error = str(exc)
    return render_template("index.html", defaults=defaults, result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
