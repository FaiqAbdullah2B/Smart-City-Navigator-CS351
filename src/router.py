import heapq
import numpy as np
try:
    from preprocessing import GridSystem
except ImportError:
    from src.preprocessing import GridSystem

class CityRouter:
    def __init__(self):
        self.grid = GridSystem()
        self.rush_hour_active = False # State to track the toggle

    def heuristic(self, a_id, b_id):
        r1, c1 = a_id // self.grid.cols, a_id % self.grid.cols
        r2, c2 = b_id // self.grid.cols, b_id % self.grid.cols
        return abs(r1 - r2) + abs(c1 - c2)

    def get_traffic_cost(self, grid_id):
        """
        Returns cost of moving into a cell.
        If Rush Hour is OFF, cost is always 1 (Flat).
        """
        if not self.rush_hour_active:
            return 1
            
        row, col = grid_id // self.grid.cols, grid_id % self.grid.cols
        
        # 4x4 Core: High Cost (10x)
        if 8 <= row <= 11 and 8 <= col <= 11:
            return 10
        
        # 6x6 Buffer: Medium Cost (5x)
        if 7 <= row <= 12 and 7 <= col <= 12:
            return 5
            
        return 1

    def get_neighbors(self, current_id):
        neighbors = []
        row, col = current_id // self.grid.cols, current_id % self.grid.cols
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in moves:
            new_r, new_c = row + dr, col + dc
            if 0 <= new_r < self.grid.rows and 0 <= new_c < self.grid.cols:
                neighbors.append(new_r * self.grid.cols + new_c)
        return neighbors

    def find_path(self, start_lat, start_lon, end_lat, end_lon, rush_hour=False):
        """
        Added `rush_hour` parameter to toggle logic.
        """
        self.rush_hour_active = rush_hour # Set state for this run
        
        start_id = self.grid.get_grid_id(start_lat, start_lon)
        end_id = self.grid.get_grid_id(end_lat, end_lon)

        if start_id is None or end_id is None:
            return None, None 

        open_set = []
        heapq.heappush(open_set, (0, start_id))
        came_from = {}
        g_score = {start_id: 0}
        f_score = {start_id: self.heuristic(start_id, end_id)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == end_id:
                return self.reconstruct_path(came_from, current, start_lat, start_lon, end_lat, end_lon)
            
            for neighbor in self.get_neighbors(current):
                move_cost = self.get_traffic_cost(neighbor)
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + self.heuristic(neighbor, end_id)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
                    
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