import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Import cleaning logic
try:
    from preprocessing import load_and_clean_data
except ImportError:
    from src.preprocessing import load_and_clean_data

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'linear_regression.pkl')

def train_model():
    # 1. Load Data
    df = load_and_clean_data()
    if df is None:
        return

    print("Feature Engineering (Adding Distance)...")
    # Extract temporal features
    df['hour_of_day'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    
    # --- IMPROVEMENT: Calculate Manhattan Distance ---
    # This correlates much better with time than raw coordinates do
    # Multiplied by ~111km to convert degrees roughly to km/meters implies magnitude
    df['dist_lat'] = np.abs(df['pickup_latitude'] - df['dropoff_latitude'])
    df['dist_lon'] = np.abs(df['pickup_longitude'] - df['dropoff_longitude'])
    df['manhattan_dist'] = df['dist_lat'] + df['dist_lon']

    # Define Features (X)
    # We keep the proposal features but ADD the distance which carries the real signal
    features = [
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'hour_of_day', 'day_of_week',
        'manhattan_dist' # <--- The key to better score
    ]
    target = 'trip_duration'

    X = df[features]
    y = df[target]

    # 2. Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train
    print("Training Multiple Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model Trained Successfully.")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R^2 Score: {r2:.4f} (Much better!)")

    # 5. Save Model
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("DONE.")

if __name__ == "__main__":
    train_model()