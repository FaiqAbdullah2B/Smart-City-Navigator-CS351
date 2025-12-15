import pandas as pd
import numpy as np
import os

# --- PATH CONFIGURATION ---
# Get the absolute path of the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level to project root, then into data
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'train.csv')

# --- CONFIGURATION: MANHATTAN BOUNDING BOX ---
MIN_LAT = 40.7000
MAX_LAT = 40.8000
MIN_LON = -74.0200
MAX_LON = -73.9300

GRID_ROWS = 20
GRID_COLS = 20

class GridSystem:
    def __init__(self):
        self.min_lat = MIN_LAT
        self.max_lat = MAX_LAT
        self.min_lon = MIN_LON
        self.max_lon = MAX_LON
        self.rows = GRID_ROWS
        self.cols = GRID_COLS
        
        self.lat_step = (self.max_lat - self.min_lat) / self.rows
        self.lon_step = (self.max_lon - self.min_lon) / self.cols

    def get_grid_id(self, lat, lon):
        if not (self.min_lat <= lat < self.max_lat) or not (self.min_lon <= lon < self.max_lon):
            return None
        
        row_idx = int((lat - self.min_lat) / self.lat_step)
        col_idx = int((lon - self.min_lon) / self.lon_step)
        
        row_idx = min(row_idx, self.rows - 1)
        col_idx = min(col_idx, self.cols - 1)
        
        return row_idx * self.cols + col_idx

    def get_center_coordinates(self, grid_id):
        row_idx = grid_id // self.cols
        col_idx = grid_id % self.cols
        
        center_lat = self.min_lat + (row_idx * self.lat_step) + (self.lat_step / 2)
        center_lon = self.min_lon + (col_idx * self.lon_step) + (self.lon_step / 2)
        
        return center_lat, center_lon

def load_and_clean_data(filepath=DATA_PATH):
    print(f"Looking for dataset at: {filepath}")
    if not os.path.exists(filepath):
        print("ERROR: File still not found.")
        return None

    print("Loading dataset... (This might take 30-60 seconds)")
    df = pd.read_csv(filepath)
    
    # 1. Datetime conversion
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # 2. Filter Duration (1 min to 2 hours)
    df = df[(df['trip_duration'] >= 60) & (df['trip_duration'] <= 7200)]
    
    # 3. Filter Coordinates (Manhattan Box)
    df = df[
        (df['pickup_latitude'] >= MIN_LAT) & (df['pickup_latitude'] < MAX_LAT) &
        (df['pickup_longitude'] >= MIN_LON) & (df['pickup_longitude'] < MAX_LON) &
        (df['dropoff_latitude'] >= MIN_LAT) & (df['dropoff_latitude'] < MAX_LAT) &
        (df['dropoff_longitude'] >= MIN_LON) & (df['dropoff_longitude'] < MAX_LON)
    ]
    
    print(f"Cleaned Data Size: {len(df)} rows")
    return df

if __name__ == "__main__":
    # Test Data Loading
    df = load_and_clean_data()