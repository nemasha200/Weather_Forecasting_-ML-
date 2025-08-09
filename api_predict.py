# api_predict.py
import os, math, time, datetime as dt
from flask import Blueprint, request, jsonify, current_app
import numpy as np
import pandas as pd
import joblib  # or pickle

api = Blueprint("api", __name__)

# Load your model once at startup
# Put the file in your project (e.g., ./models/weather_prediction_model.pkl)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "weather_prediction_model.pkl")
model = joblib.load(MODEL_PATH)  # adjust if you used pickle

def build_feature_frame(lat, lon, start_date, days=7):
    """
    Build the feature DataFrame your model expects.
    ⚠️ IMPORTANT: Replace columns below with your model’s real features!
    """
    rows = []
    for i in range(days):
        day = start_date + dt.timedelta(days=i)
        rows.append({
            "lat": lat,
            "lon": lon,
            "doy": int(day.timetuple().tm_yday),           # example: day-of-year
            "dow": day.weekday(),                           # example: day-of-week
            "is_weekend": 1 if day.weekday() >= 5 else 0,   # example
            # add any other static/external features you used in training
        })
    X = pd.DataFrame(rows)
    return X

@api.route("/api/predict-7d", methods=["GET"])
def predict_7d():
    """
    Returns 7-day predictions as JSON with fields your UI expects.
    Query params: lat, lon, units=metric|imperial, start (YYYY-MM-DD, optional)
    """
    try:
        lat = float(request.args.get("lat", "6.9271"))
        lon = float(request.args.get("lon", "79.8612"))
        units = request.args.get("units", "metric")
        start_str = request.args.get("start", None)
        start = dt.datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else dt.date.today()

        X = build_feature_frame(lat, lon, start, days=7)

        # Example: model outputs max temp, min temp, rain (mm), wind (m/s), humidity (%)
        # Adjust this part to match your model’s predict() shape / named outputs.
        y = model.predict(X)  # suppose shape (7, 5): [tmax, tmin, rain, wind_ms, humidity]
        y = np.array(y)

        # Convert to list of dicts
        out = []
        for i in range(7):
            day_dt = dt.datetime.combine(start + dt.timedelta(days=i), dt.time())
            ts = int(time.mktime(day_dt.timetuple()))
            tmax, tmin, rain, wind_ms, rh = map(float, y[i])

            # units handling
            if units == "imperial":
                # °C -> °F, mm stays mm, m/s -> mph
                tmax = (tmax * 9/5) + 32
                tmin = (tmin * 9/5) + 32
                wind = wind_ms * 2.23694
            else:
                wind = wind_ms * 3.6  # display as km/h on UI

            out.append({
                "dt": ts,
                "tmax": round(tmax, 1),
                "tmin": round(tmin, 1),
                "rain": round(rain, 1),
                "wind": round(wind),
                "humidity": round(rh),
                # Optional: let UI pick an icon from temps/rain
                "icon_hint": "rain" if rain >= 1 else ("hot" if tmax >= 32 else "mild")
            })

        return jsonify({"ok": True, "data": out})
    except Exception as e:
        current_app.logger.exception("Prediction error")
        return jsonify({"ok": False, "error": str(e)}), 500
