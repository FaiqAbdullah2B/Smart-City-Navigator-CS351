import pandas as pd
import numpy as np
import os

# path configuration else doesn't work properly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

# Constants to extract only manhattan data from new york dataset
MIN_LAT = 40.7000
MAX_LAT = 40.8000
MIN_LON = -74.0200
MAX_LON = -73.9300

# splitting the data into grids as an abstraction for our applicatoin
GRID_ROWS = 20
GRID_COLS = 20

# creating the grid system
class GridSystem:
    # constructor
    def __init__(self):
        self.min_lat = MIN_LAT
        self.max_lat = MAX_LAT
        self.min_lon = MIN_LON
        self.max_lon = MAX_LON
        self.rows = GRID_ROWS
        self.cols = GRID_COLS
        # range / no of cells 
        self.lat_step = (self.max_lat - self.min_lat) / self.rows
        self.lon_step = (self.max_lon - self.min_lon) / self.cols

    def get_grid_id(self, lat, lon):
        # first check if its in range
        if not (self.min_lat <= lat < self.max_lat) or not (self.min_lon <= lon < self.max_lon):
            return None
        
        # get indexes using difference, and assign 19 if last one
        row_idx = min(int((lat - self.min_lat) / self.lat_step), self.rows - 1)
        col_idx = min(int((lon - self.min_lon) / self.lon_step), self.cols - 1)
        # IDs ordered by numbers going row by row
        return row_idx * self.cols + col_idx

    def get_center_coordinates(self, grid_id):
        # center for a specific grid box
        row_idx = grid_id // self.cols
        col_idx = grid_id % self.cols
        center_lat = self.min_lat + (row_idx * self.lat_step) + (self.lat_step / 2)
        center_lon = self.min_lon + (col_idx * self.lon_step) + (self.lon_step / 2)
        return center_lat, center_lon

def load_and_clean_data(filename='train.csv'):
    # loading and cleaning data

    filepath = os.path.join(DATA_DIR, filename)
    print(f"Looking for dataset at: {filepath}")
    
    # error handling
    if not os.path.exists(filepath):
        print(f"ERROR: File {filename} not found in data folder.")
        return None

    print(f"Loading {filename}...")
    # creating dataframe
    df = pd.read_csv(filepath)
    
    # 1. Convert Timestamps
    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # 2. Filter Duration to remove big outliers
    if 'trip_duration' in df.columns:
        df = df[(df['trip_duration'] >= 60) & (df['trip_duration'] <= 7200)]
    
    # 3. Filter Coordinates (Manhattan Grid Box)
    df = df[
        (df['pickup_latitude'] >= MIN_LAT) & (df['pickup_latitude'] < MAX_LAT) &
        (df['pickup_longitude'] >= MIN_LON) & (df['pickup_longitude'] < MAX_LON) &
        (df['dropoff_latitude'] >= MIN_LAT) & (df['dropoff_latitude'] < MAX_LAT) &
        (df['dropoff_longitude'] >= MIN_LON) & (df['dropoff_longitude'] < MAX_LON)
    ]
    
    print(f"Cleaned {filename} Size: {len(df)} rows")
    return df

if __name__ == "__main__":
    load_and_clean_data('train.csv')