from flask import Blueprint, render_template, request, flash
from flask_login import login_required
import numpy as np
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ✅ Define Blueprint
manual_input_bp = Blueprint('manual_input_bp', __name__)

# ✅ Define the WeatherPredictor class here
class WeatherPredictor:
    def __init__(self, model_path=r"C:\Users\Nemasha\Desktop\Random_F_model\weather_prediction_model.pkl"):
        try:
            self.model_package = joblib.load(model_path)
            self.models = self.model_package['models']
            self.scalers = self.model_package['scalers']
            self.feature_names = self.model_package['feature_names']
            self.performance = self.model_package['performance']
            self.target_mapping = self.model_package['target_mapping']
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def predict_weather(self, temp_max, temp_min, humidity_max, humidity_min,
                        pressure_8am, pressure_5pm, windspeed, precipitation,
                        month=None, day_of_year=None):
        
        if month is None:
            month = datetime.now().month
        if day_of_year is None:
            day_of_year = datetime.now().timetuple().tm_yday

        season = (1 if month in [12, 1, 2] else
                  2 if month in [3, 4, 5] else
                  3 if month in [6, 7, 8] else 4)

        base_features = [temp_max, temp_min, humidity_max, humidity_min,
                         pressure_8am, pressure_5pm, windspeed, precipitation,
                         month, day_of_year, season]
        
        ma_features = [temp_max, temp_min, humidity_max, humidity_min,
                       pressure_8am, pressure_5pm, windspeed, precipitation]

        all_features = base_features + ma_features
        X = np.array(all_features).reshape(1, -1)

        predictions = {}
        for target_name, model in self.models.items():
            scaler = self.scalers[target_name]
            X_scaled = scaler.transform(X)
            pred_value = model.predict(X_scaled)[0]
            readable_name = self.target_mapping[target_name]
            predictions[readable_name] = pred_value

        return predictions

# ✅ Manual input route
@manual_input_bp.route('/manual-input', methods=['GET', 'POST'])
@login_required
def manual_input():
    predictions = None
    if request.method == 'POST':
        try:
            inputs = {
                'temp_max': float(request.form['temp_max']),
                'temp_min': float(request.form['temp_min']),
                'humidity_max': float(request.form['humidity_max']),
                'humidity_min': float(request.form['humidity_min']),
                'pressure_8am': float(request.form['pressure_8am']),
                'pressure_5pm': float(request.form['pressure_5pm']),
                'windspeed': float(request.form['windspeed']),
                'precipitation': float(request.form['precipitation']),
            }

            predictor = WeatherPredictor()
            predictions = predictor.predict_weather(**inputs)

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    return render_template('user_input.html', predictions=predictions)
