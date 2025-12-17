import math
import numpy as np
import pandas as pd # Needed for batch prediction

try:
    from preprocessing import GridSystem
except ImportError:
    from src.preprocessing import GridSystem

class CityRouter:
    def __init__(self):
        self.grid = GridSystem()
        # 20x20 matrix to store "Seconds to cross this cell"
        self.cost_grid = None 
        self.min_cost = 1.0 # For heuristic scaling

    def precompute_grid_costs(self, model, hour, day):
        """
        Runs a BATCH prediction for every cell in the grid to create a traffic map.
        We simulate a short '1-block' trip starting from every cell.
        """
        if model is None:
            self.cost_grid = np.ones((20, 20))
            self.min_cost = 1.0
            return

        rows = []
        # Calculate fixed deltas for a single block movement
        # We assume a diagonal move (1 block lat + 1 block lon) represents general congestion
        d_lat = self.grid.lat_step
        d_lon = self.grid.lon_step
        manhattan = d_lat + d_lon

        # Build input rows for all 400 cells
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                p_lat, p_lon = self.grid.get_center_coordinates(r * self.grid.cols + c)
                
                rows.append({
                    'passenger_count': 1,
                    'pickup_longitude': p_lon,
                    'pickup_latitude': p_lat,
                    'dropoff_longitude': p_lon + d_lon, # Simulate slight movement
                    'dropoff_latitude': p_lat + d_lat,
                    'pickup_hour': hour,
                    'pickup_day_of_week': day,
                    'pickup_month': 6,
                    'manhattan_dist': manhattan,
                    'delta_lat': d_lat,
                    'delta_lon': d_lon
                })

        # Create DataFrame
        df_input = pd.DataFrame(rows)
        
        # Ensure column order matches model
        if hasattr(model, "feature_names_in_"):
            df_input = df_input[model.feature_names_in_]

        # Batch Predict
        try:
            log_preds = model.predict(df_input)
            seconds_preds = np.expm1(log_preds)
            
            # Reshape into 20x20 grid
            self.cost_grid = seconds_preds.reshape((self.grid.rows, self.grid.cols))
            
            # Update heuristic scaler (Admissibility: h(n) <= true cost)
            self.min_cost = np.min(self.cost_grid)
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            self.cost_grid = np.ones((20, 20))

    def heuristic(self, a_id, b_id):
        """
        Estimated cost = Manhattan Distance (blocks) * Minimum Seconds Per Block.
        This keeps units consistent (Seconds + Seconds).
        """
        r1, c1 = a_id // self.grid.cols, a_id % self.grid.cols
        r2, c2 = b_id // self.grid.cols, b_id % self.grid.cols
        dist_blocks = abs(r1 - r2) + abs(c1 - c2)
        return dist_blocks * self.min_cost

    def get_neighbors(self, current_id):
        neighbors = []
        row, col = current_id // self.grid.cols, current_id % self.grid.cols
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in moves:
            new_r, new_c = row + dr, col + dc
            # Check bounds
            if 0 <= new_r < self.grid.rows and 0 <= new_c < self.grid.cols:
                neighbors.append(new_r * self.grid.cols + new_c)
        return neighbors

    import math # Only needed if you use math.inf for initialization

    def find_path(self, start_lat, start_lon, end_lat, end_lon, model=None, hour=12, day=0):
        """
        Updated find_path using the list-based A* implementation.
        """
        # Update the Cost Grid based on current AI Model context
        self.precompute_grid_costs(model, hour, day)

        start_id = self.grid.get_grid_id(start_lat, start_lon)
        end_id = self.grid.get_grid_id(end_lat, end_lon)

        if start_id is None or end_id is None:
            return None, None 

        fringe = [start_id] # keeps the front-most nodes (next to be evaluated)
        came_from = {}      # parents
        g_score = {start_id: 0} # storing the true cost
        f_score = {start_id: self.heuristic(start_id, end_id)}

        while len(fringe) != 0:
            
            # Initialize min_f with the first element's score
            min_f = g_score[fringe[0]] + self.heuristic(fringe[0], end_id)
            index = 0

            # Iterate through all nodes in the fringe list to find the minimum f
            for i in range(len(fringe)):
                node_id = fringe[i]
                current_f = g_score.get(node_id, math.inf) + self.heuristic(node_id, end_id)
                
                if current_f < min_f:
                    index = i
                    min_f = current_f
            
            # Pop the node with the lowest F-score
            current = fringe.pop(index)

            if current == end_id:
                # Reconstruct path and return
                return self.reconstruct_path(came_from, current, start_lat, start_lon, end_lat, end_lon)
            
            # Explore Neighbors
            for neighbor in self.get_neighbors(current):
                
                # calculating the cost
                nr, nc = neighbor // self.grid.cols, neighbor % self.grid.cols
                move_cost = self.cost_grid[nr][nc] # get cost from the cost grid
                
                # Return current + move_cost if found. Else math.inf
                tentative_g_score = g_score.get(current, math.inf) + move_cost
                
                # Update Path if a Shorter One is Found
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    
                    # Record the better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    # Calculate new F-score (G + H)
                    f = tentative_g_score + self.heuristic(neighbor, end_id)
                    f_score[neighbor] = f 
                    
                    # Add neighbor to fringe if it's not already there (or re-open it)
                    if neighbor not in fringe:
                        fringe.append(neighbor)
                        
        return None, None

    def reconstruct_path(self, came_from, current, start_lat, start_lon, end_lat, end_lon):
        path_ids = [current]
        while current in came_from:
            current = came_from[current]
            path_ids.append(current)
        path_ids.reverse()
        
        path_coords = []
        path_coords.append((start_lat, start_lon))
        for grid_id in path_ids:
            lat, lon = self.grid.get_center_coordinates(grid_id)
            path_coords.append((lat, lon))
        path_coords.append((end_lat, end_lon))
        
        return path_coords, path_ids