import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

try:
    from preprocessing import GridSystem
    from router import CityRouter
except ImportError:
    from src.preprocessing import GridSystem
    from src.router import CityRouter

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/rf_model.pkl')

@st.cache_resource
def load_resources():
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            pass
    router = CityRouter()
    grid = GridSystem()
    return model, router, grid

model, router, grid = load_resources()

def grid_to_latlon(grid_sys, row, col):
    lat = grid_sys.min_lat + (row * grid_sys.lat_step) + (grid_sys.lat_step / 2)
    lon = grid_sys.min_lon + (col * grid_sys.lon_step) + (grid_sys.lon_step / 2)
    return lat, lon

def predict_duration(model, p_lat, p_lon, d_lat, d_lon, hour, day):
    if model is None: return 0
    manhattan_dist = np.abs(d_lat - p_lat) + np.abs(d_lon - p_lon)
    delta_lat = d_lat - p_lat
    delta_lon = d_lon - p_lon
    
    input_data = pd.DataFrame([{
        'passenger_count': 1,
        'pickup_longitude': p_lon,
        'pickup_latitude': p_lat,
        'dropoff_longitude': d_lon,
        'dropoff_latitude': d_lat,
        'pickup_hour': hour,
        'pickup_day_of_week': day,
        'pickup_month': 6,
        'manhattan_dist': manhattan_dist,
        'delta_lat': delta_lat,
        'delta_lon': delta_lon
    }])
    try:
        if hasattr(model, "feature_names_in_"):
            input_data = input_data[model.feature_names_in_]
        log_pred = model.predict(input_data)[0]
        return np.expm1(log_pred)
    except Exception:
        return 0

# --- UI LAYOUT ---
st.set_page_config(page_title="GridNavigator AI", layout="wide", page_icon="🧩")

col1, col2 = st.columns([1, 2])

with col1:
    st.title("🧩 Smart City")
    st.markdown("Route planning with **Rush Hour** simulation.")
    
    st.header("1. Controls")
    
    # --- TOGGLE FOR RUSH HOUR ---
    is_rush_hour = st.checkbox("🚨 **Enable Rush Hour**", value=True, 
                               help="Simulates 10x traffic in the center")

    col_s1, col_s2 = st.columns(2)
    start_row = col_s1.slider("Start Row", 0, 19, 2)
    start_col = col_s2.slider("Start Col", 0, 19, 2)
    
    col_e1, col_e2 = st.columns(2)
    end_row = col_e1.slider("End Row", 0, 19, 18)
    end_col = col_e2.slider("End Col", 0, 19, 18)

    st.divider()
    hour = st.slider("Time", 0, 23, 14)

    start_lat, start_lon = grid_to_latlon(grid, start_row, start_col)
    end_lat, end_lon = grid_to_latlon(grid, end_row, end_col)

    # 1. Pathfinding (Pass the Toggle State!)
    path_coords, path_indices = router.find_path(
        start_lat, start_lon, end_lat, end_lon, rush_hour=is_rush_hour
    )
    
    # 2. Prediction
    predicted_seconds = predict_duration(model, start_lat, start_lon, end_lat, end_lon, hour, 2)
    st.metric("AI Duration Estimate", f"{round(predicted_seconds/60, 1)} mins")

with col2:
    grid_matrix = np.zeros((20, 20))
    
    # 1. Paint Traffic Zones ONLY if checkbox is active
    if is_rush_hour:
        grid_matrix[7:13, 7:13] = 0.3 # Buffer (Value 0.3)
        grid_matrix[8:12, 8:12] = 0.6 # Core (Value 0.6)

    # 2. Paint Path
    if path_indices:
        for gid in path_indices:
            r, c = gid // 20, gid % 20
            grid_matrix[r][c] = 1 # Path (Value 1.0)

    # 3. Paint Start/End
    grid_matrix[start_row][start_col] = 2 # Start (Value 2.0)
    grid_matrix[end_row][end_col] = 3 # End (Value 3.0)

    # --- FIXED COLORSCALE ---
    # We map specific value ranges to colors.
    # The scale goes from 0.0 to 1.0. 
    # Since zmax=3, a value of 1.0 is represented by 1/3 = 0.33 on the scale.
    fig = go.Figure(data=go.Heatmap(
        z=grid_matrix,
        x=list(range(20)),
        y=list(range(20)),
        zmin=0, zmax=3,
        colorscale=[
            [0.0, "#f0f2f6"],  # 0: Empty (Gray)
            [0.05, "#f0f2f6"],
            
            [0.05, "#fdba74"], # 0.3: Buffer (Light Orange)
            [0.15, "#fdba74"],
            
            [0.15, "#ea580c"], # 0.6: Core (Dark Orange)
            [0.25, "#ea580c"],
            
            [0.25, "#3b82f6"], # 1.0: Path (Blue)
            [0.50, "#3b82f6"],
            
            [0.50, "#22c55e"], # 2.0: Start (Green)
            [0.80, "#22c55e"],
            
            [0.80, "#ef4444"], # 3.0: End (Red)
            [1.0, "#ef4444"]
        ],
        showscale=False,
        xgap=1, ygap=1
    ))

    # Dynamic Labels
    if is_rush_hour:
        fig.add_annotation(x=9.5, y=9.5, text="RUSH<br>HOUR", showarrow=False, font=dict(color="white", size=10))
        
    fig.add_annotation(x=start_col, y=start_row, text="S", showarrow=False, font=dict(color="white"))
    fig.add_annotation(x=end_col, y=end_row, text="E", showarrow=False, font=dict(color="white"))

    fig.update_layout(
        title="Real-Time Grid Visualization",
        yaxis=dict(autorange="reversed"),
        height=600,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)