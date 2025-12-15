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
    st.markdown("Routes are now weighted by **real AI predictions** for every single block.")
    
    st.header("1. Controls")
    
    col_s1, col_s2 = st.columns(2)
    start_row = col_s1.slider("Start Row", 0, 19, 2)
    start_col = col_s2.slider("Start Col", 0, 19, 2)
    
    col_e1, col_e2 = st.columns(2)
    end_row = col_e1.slider("End Row", 0, 19, 18)
    end_col = col_e2.slider("End Col", 0, 19, 18)

    st.divider()
    
    st.subheader("2. Context")
    hour = st.slider("Time of Day (24h)", 0, 23, 14)
    
    day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    day_val = st.select_slider("Day of Week", options=list(day_map.keys()), format_func=lambda x: day_map[x], value=2)

    start_lat, start_lon = grid_to_latlon(grid, start_row, start_col)
    end_lat, end_lon = grid_to_latlon(grid, end_row, end_col)

    # 1. Pathfinding (NOW PASSING MODEL CONTEXT)
    with st.spinner("AI calculating traffic for all 400 blocks..."):
        path_coords, path_indices = router.find_path(
            start_lat, start_lon, end_lat, end_lon, 
            model=model, hour=hour, day=day_val
        )
    
    # 2. Prediction
    predicted_seconds = predict_duration(model, start_lat, start_lon, end_lat, end_lon, hour, day_val)
    st.metric("Total Trip Estimate", f"{round(predicted_seconds/60, 1)} mins")

with col2:
    # Use the ROUTER'S calculated cost grid for visualization if available
    if router.cost_grid is not None:
        grid_matrix = router.cost_grid.copy()
        # Cap outliers for better color contrast
        v_min, v_max = np.percentile(grid_matrix, [5, 95])
        path_val = v_max * 1.5 
    else:
        grid_matrix = np.zeros((20, 20))
        path_val = 1

    # Paint Path
    if path_indices:
        for gid in path_indices:
            r, c = gid // 20, gid % 20
            grid_matrix[r][c] = path_val

    # Paint Start/End
    grid_matrix[start_row][start_col] = path_val * 1.2 # Start
    grid_matrix[end_row][end_col] = path_val * 1.2 # End

    fig = go.Figure(data=go.Heatmap(
        z=grid_matrix,
        x=list(range(20)),
        y=list(range(20)),
        colorscale='Viridis', 
        showscale=True,
        xgap=1, ygap=1,
        colorbar=dict(title="Sec/Block")
    ))

    fig.add_annotation(x=start_col, y=start_row, text="S", showarrow=False, font=dict(color="white"))
    fig.add_annotation(x=end_col, y=end_row, text="E", showarrow=False, font=dict(color="white"))

    fig.update_layout(
        title=f"AI Traffic Heatmap ({day_map[day_val]} @ {hour}:00)",
        yaxis=dict(autorange="reversed"),
        height=600,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- LEGEND EXPLANATION ---
    st.info("💡 **Heatmap Legend:**")
    st.markdown("""
    * **Color Intensity:** Represents the predicted **Time Cost (seconds)** to cross a single block.
    * 🟣 **Purple / Dark:** Low Traffic (Fast travel).
    * 🟡 **Yellow / Bright:** High Traffic (Slow travel).
    * **The Brightest Line:** This is the path A* selected because it minimizes the total "yellowness" (time) encountered.
    """)