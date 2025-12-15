import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# --- CONFIGURATION ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/train.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/rf_model.pkl')

try:
    from preprocessing import load_and_clean_data
except ImportError:
    from src.preprocessing import load_and_clean_data


def load_and_preprocess(filepath):
    print(f"Loading dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file at {filepath}")
        
    df = load_and_clean_data()
    
    # 1. Date Conversions
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_month'] = df['pickup_datetime'].dt.month
    
    # if 'store_and_fwd_flag' in df.columns:
    #     df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y': 1, 'N': 0})

    # --- CRITICAL FIX: CLEANING OUT
    # 3. Geospatial Features
    df['manhattan_dist'] = np.abs(df['dropoff_latitude'] - df['pickup_latitude']) + \
                           np.abs(df['dropoff_longitude'] - df['pickup_longitude'])

    df['delta_lat'] = df['dropoff_latitude'] - df['pickup_latitude']
    df['delta_lon'] = df['dropoff_longitude'] - df['pickup_longitude']

    # Remove zero distance trips
    # df = df[df['haversine_dist'] > 0]
    
    print(f"Cleaned Data Rows: {len(df)}")
    return df

def train_model():
    # --- 1. Load Data ---
    df = load_and_preprocess(DATA_PATH)

    # --- 2. Select Features ---
    exclude_cols = ['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'vendor_id', 'store_and_fwd_flag']
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Training with features: {features}")
    
    X = df[features]
    
    # --- CRITICAL FIX: LOG TRANSFORMATION ---
    # We predict log(trip_duration) instead of raw seconds.
    # This normalizes the data distribution.
    y = np.log1p(df['trip_duration']) 

    # --- 3. Split Data ---
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. Train Model ---
    print("Training Random Forest Model (Log-Transformed Target)...")
    model = RandomForestRegressor(n_estimators=50, min_samples_leaf=20, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # --- 5. Evaluate Model ---
    print("Evaluating...")
    log_predictions = model.predict(X_val)
    
    # Calculate R2 Score on the LOG scale (This is the standard metric for this problem)
    r2_log = r2_score(y_val, log_predictions)
    
    # Convert back to seconds to see "real" error
    predictions_seconds = np.expm1(log_predictions)
    y_val_seconds = np.expm1(y_val)
    rmse_seconds = np.sqrt(mean_squared_error(y_val_seconds, predictions_seconds))

    print("-" * 30)
    print(f"Validation R2 Score (Log Scale): {r2_log:.4f}")
    print(f"Validation RMSE (Real Seconds): {rmse_seconds:.2f}")
    print("-" * 30)
    
    if r2_log < 0.3:
        print("WARNING: Score is still low. Check if 'pickup_datetime' is parsing correctly.")

    # --- 6. Save Model ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()