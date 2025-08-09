import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Load and preprocess the weather data"""
    print("Loading data...")
    
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Display basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Clean column names (remove extra spaces but keep the format)
    df.columns = df.columns.str.strip()
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle the Pressure(5.30PM) column if it's object type (mixed data)
    if df['Pressure(5.30PM)'].dtype == 'object':
        # Convert to numeric, coercing errors to NaN
        df['Pressure(5.30PM)'] = pd.to_numeric(df['Pressure(5.30PM)'], errors='coerce')
    
    # Handle missing values in Windspeed (replace with 0 for calm)
   # Convert Windspeed to numeric and handle missing values
    df['Windspeed'] = pd.to_numeric(df['Windspeed'], errors='coerce')  # ensure float
    df['Windspeed'] = df['Windspeed'].fillna(0)  # replace NaNs with 0 for calm

    
    # Handle missing values in other columns
    numeric_columns = ['Temp_Max', 'Temp_Min', 'Humidity_Max', 'Humidity_Min', 
                      'Pressure(8.30AM)', 'Pressure(5.30PM)', 'Precepitation']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
    
    # Sort by date to ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"\nData after preprocessing:")
    print(df.info())
    
    return df

def create_features_and_targets(df):
    """Create features for today and targets for next day prediction"""
    
    # Features for today (input features) - using actual column names from your data
    feature_columns = ['Temp_Max', 'Temp_Min', 'Humidity_Max', 'Humidity_Min', 
                      'Pressure(8.30AM)', 'Pressure(5.30PM)', 'Windspeed', 'Precepitation']
    
    # Create next-day targets (what we want to predict)
    target_columns = ['Next_Temp_Max', 'Next_Temp_Min', 'Next_Humidity_Max', 'Next_Humidity_Min', 
                     'Next_Pressure_8AM', 'Next_Pressure_5PM', 'Next_Windspeed', 'Next_Precepitation']
    
    # Shift the data to create next-day targets
    for i, col in enumerate(feature_columns):
        df[target_columns[i]] = df[col].shift(-1)
    
    # Add time-based features
    df['Month'] = df['Date'].dt.month
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    df['Season'] = df['Month'].apply(lambda x: 
        1 if x in [12, 1, 2] else  # Winter
        2 if x in [3, 4, 5] else   # Spring
        3 if x in [6, 7, 8] else   # Summer
        4)  # Autumn
    
    # Add moving averages for trend analysis
    window = 7  # 7-day moving average
    for col in feature_columns:
        df[f'{col}_MA7'] = df[col].rolling(window=window, min_periods=1).mean()
    
    # Remove the last row (no next day data) and first few rows (for moving averages)
    df = df.iloc[window:-1].copy()
    
    return df, feature_columns, target_columns

def train_models(df, feature_columns, target_columns):
    """Train separate models for each weather parameter"""
    
    # Prepare base features
    base_features = feature_columns + ['Month', 'Day_of_Year', 'Season']
    
    # Add moving average features
    ma_features = [f'{col}_MA7' for col in feature_columns]
    all_features = base_features + ma_features
    
    X = df[all_features]
    
    # Handle any remaining missing values
    X = X.fillna(X.mean())
    
    models = {}
    scalers = {}
    performance = {}
    
    print("\nTraining models for each weather parameter...")
    
    for i, target in enumerate(target_columns):
        print(f"\nTraining model for {target}...")
        
        y = df[target].fillna(df[target].mean())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        performance[target] = {
            'MAE': mae,
            'RMSE': np.sqrt(mse),
            'R2': r2
        }
        
        print(f"  MAE: {mae:.3f}, RMSE: {np.sqrt(mse):.3f}, R2: {r2:.3f}")
        
        # Store model and scaler
        models[target] = model
        scalers[target] = scaler
    
    return models, scalers, all_features, performance

def save_models_and_metadata(models, scalers, feature_names, performance):
    """Save trained models and necessary metadata"""
    
    # Create a package with all necessary components
    model_package = {
        'models': models,
        'scalers': scalers,
        'feature_names': feature_names,
        'performance': performance,
        'target_mapping': {
            'Next_Temp_Max': 'Maximum Temperature (°C)',
            'Next_Temp_Min': 'Minimum Temperature (°C)', 
            'Next_Humidity_Max': 'Maximum Humidity (%)',
            'Next_Humidity_Min': 'Minimum Humidity (%)',
            'Next_Pressure_8AM': 'Pressure at 8:30 AM',
            'Next_Pressure_5PM': 'Pressure at 5:30 PM',
            'Next_Windspeed': 'Wind Speed',
            'Next_Precepitation': 'Precipitation'
        }
    }
    
    # Save the complete package
    joblib.dump(model_package, 'weather_prediction_model.pkl')
    print("\nModel saved as 'weather_prediction_model.pkl'")
    
    # Print performance summary
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    for target, metrics in performance.items():
        readable_name = model_package['target_mapping'][target]
        print(f"\n{readable_name}:")
        print(f"  Mean Absolute Error: {metrics['MAE']:.3f}")
        print(f"  Root Mean Square Error: {metrics['RMSE']:.3f}")
        print(f"  R² Score: {metrics['R2']:.3f}")

def main():
    """Main function to run the complete training pipeline"""
    
    # File path
    file_path = "C:\\Users\\Nemasha\\Desktop\\Random_F_model\\Colombo_Weather.xlsx"
   
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(file_path)
        
        # Create features and targets
        df, feature_columns, target_columns = create_features_and_targets(df)
        
        print(f"\nFinal dataset shape for training: {df.shape}")
        print(f"Number of features: {len(feature_columns) + 3 + len(feature_columns)}")  # base + time + moving avg
        
        # Train models
        models, scalers, feature_names, performance = train_models(df, feature_columns, target_columns)
        
        # Save everything
        save_models_and_metadata(models, scalers, feature_names, performance)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nYour weather prediction model is ready to use!")
        print("Use the companion prediction script to make forecasts.")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data format and try again.")

if __name__ == "__main__":
    main()

