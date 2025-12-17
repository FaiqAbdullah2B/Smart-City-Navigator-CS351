import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# configuring path
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/train.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/rf_model.pkl')

# errorhandling and alternate fetch for loader/cleaner function
try:
    from preprocessing import load_and_clean_data
except ImportError:
    from src.preprocessing import load_and_clean_data

# load using preprocessor and further process 
def load_and_preprocess(filepath):
    print(f"Loading dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file at {filepath}")
    
    # use preprocessor function
    df = load_and_clean_data()
    
    # 1. Date Conversions and creating new features out of it
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_month'] = df['pickup_datetime'].dt.month
    
    # commented cuz we dont care about this feature anyways
    # will not include it in the training data
    # if 'store_and_fwd_flag' in df.columns:
    #     df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y': 1, 'N': 0})

    # Feature Engineering: Very important
    # sky rocketed the model quality
    # 3. Geospatial Features
    df['manhattan_dist'] = np.abs(df['dropoff_latitude'] - df['pickup_latitude']) + \
                           np.abs(df['dropoff_longitude'] - df['pickup_longitude'])

    df['delta_lat'] = df['dropoff_latitude'] - df['pickup_latitude']
    df['delta_lon'] = df['dropoff_longitude'] - df['pickup_longitude']
    
    print(f"Cleaned and Engineered Data Rows: {len(df)}")
    return df

def train_model():
    # Loading
    df = load_and_preprocess(DATA_PATH)

    # Selecting the important features only
    # exclude_cols includes everything to remove
    exclude_cols = ['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'vendor_id', 'store_and_fwd_flag']
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Training with features: {features}")
    
    X = df[features]
    
    # LOG TRANSFORMATION : VERY IMPORTANT
    # stops the random forest regressor to optimize for outliers
    # We predict log(trip_duration) instead of raw seconds.
    # This normalizes the data distribution.
    y = np.log1p(df['trip_duration']) 

    # splitting the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # training model
    print("Training Random Forest Model (Log-Transformed Target)...")
    model = RandomForestRegressor(n_estimators=50, min_samples_leaf=20, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation phase
    print("Evaluating...")
    log_predictions = model.predict(X_val)
    
    # Calculate R2 Score on the LOG scale 
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

    # saving the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()