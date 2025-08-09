from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User
from forms import LoginForm, RegisterForm
import joblib
import numpy as np
import pandas as pd
from io import BytesIO
import warnings
from flask import make_response
import io
from reportlab.pdfgen import canvas
from models import User
from flask import make_response
from reportlab.pdfgen import canvas
from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response
import csv
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User
from forms import LoginForm, RegisterForm
import joblib
import numpy as np
import warnings
from functools import wraps
from flask import abort
warnings.filterwarnings('ignore')
from manual_input import manual_input_bp
import requests

# Replace with your actual API key from OpenWeatherMap
OPENWEATHER_API_KEY = 'adf66628bb4f4ecea2301e3916015850'

warnings.filterwarnings('ignore')

# ✅ Create Flask app
app = Flask(__name__)

# ✅ App Config
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'nemasha.anushani2000@gmail.com'
app.config['MAIL_PASSWORD'] = 'trrt tcwd uyjs lzba'

# ✅ Initialize extensions
db.init_app(app)
mail = Mail(app)

# ✅ Login Manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return redirect(url_for('login'))  # or redirect to dashboard if already logged 


# ✅ User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return login_manager.unauthorized()
        if getattr(current_user, 'role', None) != 'admin':
            flash("Admins only.", "warning")
            return redirect(url_for('user_dashboard_alt'))
        return f(*args, **kwargs)
    return wrapper


class WeatherPredictor:
    def __init__(self):
        model_package = joblib.load("weather_prediction_model.pkl")
        self.models = model_package['models']
        self.scalers = model_package['scalers']
        self.feature_names = model_package.get('feature_names', [])
        self.target_mapping = model_package.get('target_mapping', {})

    def predict_weather(self, **features):
        today = datetime.now()
        month = today.month
        day_of_year = today.timetuple().tm_yday
        season = 1 if month in [12, 1, 2] else 2 if month in [3, 4, 5] else 3 if month in [6, 7, 8] else 4

        base_features = list(features.values()) + [month, day_of_year, season]
        ma_features = list(features.values())  # reuse as mock moving averages
        all_features = base_features + ma_features

        X = np.array(all_features).reshape(1, -1)
        predictions = {}

        for target, model in self.models.items():
            scaler = self.scalers[target]
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            readable_name = self.target_mapping.get(target, target)
            predictions[readable_name] = round(pred, 2)

        return predictions
 

@app.route('/api/weather')
def api_weather():
    preds = get_predictions()
    forecast = {
        "days": ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"],
        "temperature": [preds.get("Maximum Temperature (°C)", 30) - i % 2 for i in range(7)],
        "rainfall": [preds.get("Precipitation", 10) + (i % 3) for i in range(7)]
    }
    data = {
        "temperature": preds.get("Maximum Temperature (°C)", 30),
        "rainfall": preds.get("Precipitation", 10),
        "humidity": preds.get("Maximum Humidity (%)", 85),
        "wind_speed": preds.get("Wind Speed", 14),
        "forecast": forecast
    }
    return jsonify(data)


@app.route('/user_dashboard', methods=['GET', 'POST'])
@login_required
def user_dashboard_alt():
    city = request.form.get('city') or 'Colombo'
    current_weather = {}
    predictions = None
    forecast_temp = []
    forecast_dates = []
    forecast_comments = []
    today_str = datetime.now().strftime("%B %d, %Y")   # ← compute once
    tomorrow = None                                     # ← holder for tomorrow-only view

    # ---- Get live weather ----
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        current_weather = {
            'temp': float(res['main']['temp']),
            'humidity': float(res['main']['humidity']),
            'pressure': float(res['main']['pressure']),
            'wind': float(res['wind']['speed']),
            'condition': res['weather'][0]['description'].title()
        }
    except Exception as e:
        flash(f"Live weather error: {e}", "danger")
        current_weather = {}

    # ---- Predict button clicked ----
    if request.method == 'POST' and 'predict' in request.form:
        try:
            if not current_weather:
                flash("No live weather available to base the prediction on.", "warning")
                return render_template(
                    "user_dashboard.html",
                    current_weather=current_weather,
                    today=today_str,
                    forecast=None,
                    forecast_dates=[],
                    zipped=[]
                )

            inputs = {
                'temp_max': current_weather['temp'],
                'temp_min': current_weather['temp'] - 2,
                'humidity_max': current_weather['humidity'],
                'humidity_min': max(current_weather['humidity'] - 10, 0),
                'pressure_8am': current_weather['pressure'],
                'pressure_5pm': current_weather['pressure'] - 1,
                'windspeed': current_weather['wind'],
                'precipitation': 2.0
            }

            predictor = WeatherPredictor()
            predictions = predictor.predict_weather(**inputs)

            base_temp = predictions.get("Maximum Temperature (°C)", 32.0)
            comment = "☀️ Likely Hot Day" if base_temp > 30 else "🌧 Possible Rain"

            today = datetime.now()
            for i in range(7):
                date = (today + timedelta(days=i + 1)).strftime("%b %d")
                forecast_dates.append(date)
                forecast_temp.append(round(base_temp - i * 0.5, 1))
                forecast_comments.append(comment)

            # 👉 tomorrow-only
            if forecast_temp and forecast_dates:
                tomorrow = {
                    "date": forecast_dates[0],
                    "temp": forecast_temp[0],
                    "comment": forecast_comments[0]
                }

        except Exception as e:
            flash(f"Prediction failed: {e}", "danger")

    return render_template(
        "user_dashboard.html",
        current_weather=current_weather,
        today=today_str,
        forecast=forecast_temp if forecast_temp else None,
        forecast_dates=forecast_dates,
        zipped=zip(forecast_temp, forecast_comments, forecast_dates),
        tomorrow=tomorrow
    )


@app.route('/agriculture-tips')
@login_required
def agriculture_tips():
    return render_template('first.html')

@app.route('/api/tips')
def tips():
    return jsonify({
        "recommendations": [
            "Heavy rain expected: delay irrigation.",
            "Inspect crops for fungal diseases.",
            "Avoid pesticide spraying in high humidity."
        ]
    })

@app.route('/business-planning')
@login_required
def business_planning():
    return render_template('second.html')

@app.route("/about")
@login_required
def about():
    return render_template("about.html")

@app.route("/historical-trends")
@login_required
def history():
    return render_template("history.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        uname = form.username.data.strip()
        pw = form.password.data

        # --- Special case: exact admin credentials -> index.html ---
        if uname == 'admin' and pw == 'admin123':
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    password=generate_password_hash('admin123'),
                    role='admin',
                    email='admin@example.com'
                )
                db.session.add(admin)
                db.session.commit()
            else:
                # make sure admin has the right role
                if admin.role != 'admin':
                    admin.role = 'admin'
                    db.session.commit()

            login_user(admin)
            flash("Welcome, admin!", "success")
            return redirect(url_for('protected_dashboard'))  # renders index.html

        # --- Everyone else: allow any username/password -> user_dashboard.html ---
        user = User.query.filter_by(username=uname).first()
        if not user:
            user = User(
                username=uname,
                password=generate_password_hash(pw),
                role='user'
            )
            db.session.add(user)
            db.session.commit()
        else:
            # keep them as normal users (no admin access)
            if user.role != 'user':
                user.role = 'user'
                db.session.commit()

        login_user(user)
        flash(f"Welcome, {user.username}! You are now logged in.", "success")
        return redirect(url_for('user_dashboard_alt'))

    return render_template('login.html', form=form)



@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash("Username already exists. Please choose another.", "warning")
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        # ✅ Send Email
        try:
            msg = Message("Welcome to the Weather App!",
                        sender="your_email@gmail.com",
                        recipients=[f"{form.username.data}@gmail.com"])
            msg.body = f"Hello {form.username.data},\n\nThanks for registering with our app!"
            mail.send(msg)
        except Exception as e:
            print(f"Email sending failed: {e}")

        flash("Account created! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/admin-dashboard')
@login_required
def protected_dashboard():
    return render_template('index.html', user=current_user)

@app.route('/view-db')
@login_required
def view_db():
    # Dummy data to test (replace later with real database queries)
    forecasts = [
        {"date": "2025-08-09", "temp": 30, "humidity": 65},
        {"date": "2025-08-10", "temp": 29, "humidity": 70}
    ]
    return render_template("view_db.html", forecasts=forecasts)

@app.route('/users')
@login_required
def show_users():
    users = User.query.all()
    return render_template('users.html', users=users)

@app.route('/admin/add-user', methods=['GET', 'POST'])
@login_required
def add_user():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash("Username already exists.", "warning")
        else:
            hashed_pw = generate_password_hash(form.password.data)
            new_user = User(username=form.username.data, password=hashed_pw)
            db.session.add(new_user)
            db.session.commit()
            flash("User added successfully!", "success")
            return redirect(url_for('show_users'))
    return render_template('admin_add_user.html', form=form)

@app.route('/export-excel')
def export_excel():
    users = User.query.all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Username', 'Email', 'Role'])

    for user in users:
        writer.writerow([user.id, user.username, user.email, user.role])

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=users.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/export-pdf')
def export_pdf():
    users = User.query.all()

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)

    y = 800
    p.setFont("Helvetica", 12)
    p.drawString(100, y, "Registered Users")
    y -= 20

    for user in users:
        p.drawString(100, y, f"ID: {user.id}, Username: {user.username}, Email: {user.email or '-'}, Role: {user.role or '-'}")
        y -= 20

    p.showPage()
    p.save()
    buffer.seek(0)

    return make_response(buffer.read(), 200, {
        'Content-Type': 'application/pdf',
        'Content-Disposition': 'attachment; filename=users.pdf'
    })

@app.route('/edit-user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    user = User.query.get_or_404(user_id)

    if request.method == 'POST':
        user.username = request.form['username']
        user.email = request.form['email']
        user.role = request.form['role']
        db.session.commit()
        flash('User updated successfully!')
        return redirect(url_for('show_users'))

    return render_template('edit_user.html', user=user)


@app.route('/delete-user/<int:user_id>', methods=['POST', 'GET'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully!')
    return redirect(url_for('show_users'))


@app.route('/manual-input', methods=['GET', 'POST'])
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

@app.route("/api/predict-7d")
def api_predict_7d():
    try:
        lat = float(request.args.get("lat", 6.9271))
        lon = float(request.args.get("lon", 79.8612))
        units = request.args.get("units", "metric")

        # --- Server-side OpenWeather (baseline) ---
        ow_url = (
            "https://api.openweathermap.org/data/2.5/onecall"
            f"?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts"
            f"&appid={OPENWEATHER_API_KEY}&units={units}"
        )
        ow = requests.get(ow_url, timeout=12).json()
        cur = ow.get("current", {}) or {}
        first_daily = (ow.get("daily") or [{}])[0]

        temp_now = float(cur.get("temp", 30.0))
        hum_now  = float(cur.get("humidity", 80.0))
        pres_now = float(cur.get("pressure", 1012.0))
        wind_now = float(cur.get("wind_speed", 3.0))  # m/s

        base_inputs = {
            "temp_max": temp_now,
            "temp_min": temp_now - 2,
            "humidity_max": hum_now,
            "humidity_min": max(hum_now - 10, 0),
            "pressure_8am": pres_now,
            "pressure_5pm": pres_now - 1,
            "windspeed": wind_now,
            "precipitation": float(first_daily.get("rain", 0.0)),
        }

        # --- Try ML model ---
        try:
            predictor = WeatherPredictor()
            model_out = predictor.predict_weather(**base_inputs)
        except Exception as e:
            # Model/scaler problem? fall back gracefully.
            model_out = {}

        # Extract with safe defaults
        base_tmax = float(model_out.get("Maximum Temperature (°C)", model_out.get("tmax", temp_now + 1.5)))
        base_tmin = float(model_out.get("Minimum Temperature (°C)", model_out.get("tmin", temp_now - 2.5)))
        base_rain = float(model_out.get("Precipitation", model_out.get("rain", float(first_daily.get("rain", 0.6)))))
        # normalize wind to display unit (km/h if metric, mph if imperial)
        wind_disp = wind_now * (3.6 if units == "metric" else 2.23694)
        base_wind = float(model_out.get("Wind Speed", model_out.get("wind", wind_disp)))
        base_hum  = float(model_out.get("Maximum Humidity (%)", model_out.get("humidity", hum_now)))

        # --- Build a smooth 7-day series ---
        out = []
        spread = max(2.5, base_tmax - base_tmin if base_tmax > base_tmin else 3.0)
        for i in range(7):
            tmax = round(base_tmax - 0.4 * i + (0.25 if i % 2 == 0 else -0.1), 1)
            tmin = round(tmax - spread, 1)
            rain = round(max(0.0, base_rain + (i % 3 - 1) * 0.7), 1)
            wind = base_wind * (1.0 + 0.02 * i)
            hum  = max(45.0, min(96.0, base_hum + (i % 2) * 1.2 - 0.7 * i))

            # already in display units
            out.append({
                "tmax": tmax,
                "tmin": tmin,
                "rain": rain,
                "wind": round(wind, 0),
                "humidity": round(hum, 0),
            })

        return jsonify({"ok": True, "data": out})

    except Exception as e:
        # Hard failure → return a small dummy series so UI still works
        dummy = [{"tmax": 31 - 0.4*i, "tmin": 27 - 0.4*i, "rain": 1 + (i%2)*0.5, "wind": 12+i, "humidity": 78-i} for i in range(7)]
        return jsonify({"ok": False, "data": dummy, "error": str(e)}), 200



# ✅ Run the app
if __name__ == '__main__':
    app.register_blueprint(manual_input_bp)

    app.run(debug=True)