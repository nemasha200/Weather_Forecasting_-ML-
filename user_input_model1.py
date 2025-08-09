import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WeatherPredictor:
    def __init__(self, model_path=r"C:\Users\Nemasha\Desktop\Random_F_model\weather_prediction_model.pkl"):
        """Initialize the weather predictor by loading the trained model"""
        try:
            self.model_package = joblib.load(model_path)
            self.models = self.model_package['models']
            self.scalers = self.model_package['scalers']
            self.feature_names = self.model_package['feature_names']
            self.performance = self.model_package['performance']
            self.target_mapping = self.model_package['target_mapping']
            print("Weather prediction model loaded successfully!")
            self._show_model_info()
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found.")
            print("Please run the training script first to create the model.")
            raise
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def _show_model_info(self):
        """Display information about the loaded model"""
        print("\nModel Performance Summary:")
        print("-" * 40)
        for target, metrics in self.performance.items():
            readable_name = self.target_mapping[target]
            print(f"{readable_name}: R² = {metrics['R2']:.3f}")
    
    def predict_weather(self, temp_max, temp_min, humidity_max, humidity_min, 
                       pressure_8am, pressure_5pm, windspeed, precipitation,
                       month=None, day_of_year=None):
        """
        Predict next day weather based on current conditions
        
        Parameters:
        - temp_max: Maximum temperature (°C)
        - temp_min: Minimum temperature (°C) 
        - humidity_max: Maximum humidity (%)
        - humidity_min: Minimum humidity (%)
        - pressure_8am: Atmospheric pressure at 8:30 AM
        - pressure_5pm: Atmospheric pressure at 5:30 PM
        - windspeed: Wind speed (use 0 for calm conditions)
        - precipitation: Precipitation amount
        - month: Current month (1-12, optional - will use current month if not provided)
        - day_of_year: Current day of year (1-365, optional - will use current day if not provided)
        """
        
        # Use current date if month/day not provided
        if month is None:
            month = datetime.now().month
        if day_of_year is None:
            day_of_year = datetime.now().timetuple().tm_yday
        
        # Calculate season
        season = (1 if month in [12, 1, 2] else  # Winter
                 2 if month in [3, 4, 5] else   # Spring  
                 3 if month in [6, 7, 8] else   # Summer
                 4)  # Autumn
        
        # Create base feature array
        base_features = [temp_max, temp_min, humidity_max, humidity_min,
                        pressure_8am, pressure_5pm, windspeed, precipitation,
                        month, day_of_year, season]
        
        # For moving averages, we'll use current values as approximation
        ma_features = [temp_max, temp_min, humidity_max, humidity_min,
                      pressure_8am, pressure_5pm, windspeed, precipitation]
        
        # Combine all features
        all_features = base_features + ma_features
        
        # Convert to numpy array and reshape for prediction
        X = np.array(all_features).reshape(1, -1)
        
        # Make predictions for each weather parameter
        predictions = {}
        
        for target_name, model in self.models.items():
            # Scale the features using the corresponding scaler
            scaler = self.scalers[target_name]
            X_scaled = scaler.transform(X)
            
            # Make prediction
            pred_value = model.predict(X_scaled)[0]
            
            # Store prediction with readable name
            readable_name = self.target_mapping[target_name]
            predictions[readable_name] = pred_value
        
        return predictions
    
    def print_prediction(self, predictions):
        """Print predictions in a formatted way"""
        print("\n" + "="*50)
        print("TOMORROW'S WEATHER FORECAST FOR COLOMBO")
        print("="*50)
        
        print(f"🌡  Temperature:")
        print(f"   Maximum: {predictions['Maximum Temperature (°C)']:.1f}°C")
        print(f"   Minimum: {predictions['Minimum Temperature (°C)']:.1f}°C")
        
        print(f"\n💧 Humidity:")
        print(f"   Maximum: {predictions['Maximum Humidity (%)']:.1f}%")
        print(f"   Minimum: {predictions['Minimum Humidity (%)']:.1f}%")
        
        print(f"\n🌬  Atmospheric Pressure:")
        print(f"   8:30 AM: {predictions['Pressure at 8:30 AM']:.2f}")
        print(f"   5:30 PM: {predictions['Pressure at 5:30 PM']:.2f}")
        
        print(f"\n🌪  Wind Speed: {predictions['Wind Speed']:.1f}")
        
        print(f"\n🌧  Precipitation: {predictions['Precipitation']:.1f}")

def main():
    """Main function to get manual input and predict weather"""
    
    try:
        # Load the predictor
        predictor = WeatherPredictor()
        
        print("\n" + "="*60)
        print("WEATHER PREDICTION SYSTEM")
        print("="*60)
        
        # Get manual input from user
        print("\nEnter today's weather conditions for Colombo:")
        
        def get_float_input(prompt, min_val=None, max_val=None):
            while True:
                try:
                    value = float(input(prompt))
                    if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                        print(f"Value must be between {min_val} and {max_val}. Try again.")
                        continue
                    return value
                except ValueError:
                    print("Please enter a valid number.")
        
        current_conditions = {
            'temp_max': get_float_input("Maximum Temperature (°C): ", 0, 50),
            'temp_min': get_float_input("Minimum Temperature (°C): ", 0, 50),
            'humidity_max': get_float_input("Maximum Humidity (%): ", 0, 100),
            'humidity_min': get_float_input("Minimum Humidity (%): ", 0, 100),
            'pressure_8am': get_float_input("Pressure at 8:30 AM (hPa): ", 900, 1100),
            'pressure_5pm': get_float_input("Pressure at 5:30 PM (hPa): ", 900, 1100),
            'windspeed': get_float_input("Wind Speed (0 for calm): ", 0, 100),
            'precipitation': get_float_input("Precipitation (mm): ", 0, 1000)
        }
        
        # Display entered conditions
        print(f"\nCurrent Conditions Entered:")
        print(f"Max Temp: {current_conditions['temp_max']}°C")
        print(f"Min Temp: {current_conditions['temp_min']}°C")
        print(f"Max Humidity: {current_conditions['humidity_max']}%")
        print(f"Min Humidity: {current_conditions['humidity_min']}%")
        print(f"Pressure (8:30 AM): {current_conditions['pressure_8am']} hPa")
        print(f"Pressure (5:30 PM): {current_conditions['pressure_5pm']} hPa")
        print(f"Wind Speed: {current_conditions['windspeed']}")
        print(f"Precipitation: {current_conditions['precipitation']} mm")
        
        # Make prediction
        predictions = predictor.predict_weather(**current_conditions)
        
        # Display results
        predictor.print_prediction(predictions)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you've run the training script first and the model file exists!")

if __name__ == "__main__":
    main()