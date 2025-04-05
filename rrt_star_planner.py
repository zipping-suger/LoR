import numpy as np
from scipy.spatial import KDTree
from utils import plot_trajectory

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.cost = 0.0  # Cumulative cost from start

class RRTStarPlanner:
    def __init__(self, start, goal, obstacles, 
                 max_iter=5000, step_size=0.05, 
                 goal_threshold=0.1, search_radius=0.5,
                 num_points=20, seed=None):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.search_radius = search_radius
        self.num_points = num_points
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # Local random generator
        self.reset()

    def reset(self):
        """Reset planner state for fresh planning"""
        self.nodes = [Node(self.start)]
        self.kd_tree = KDTree([self.start])
        self.path = None

    def raw_plan(self):
        for _ in range(self.max_iter):
            sample = self.sample()
            nearest_node = self.find_nearest(sample)
            new_node = self.steer(nearest_node, sample)
            
            if self.collision_free(nearest_node.position, new_node.position):
                near_nodes = self.find_near_nodes(new_node)
                best_node = self.choose_parent(new_node, near_nodes)
                
                if best_node:
                    self.add_node(best_node)
                    self.rewire(near_nodes, best_node)
                    
                    if self.reached_goal(best_node):
                        return self.extract_path(best_node)
        
        return self.extract_path(self.find_best_goal_node())
    
    def plan(self):
        """Main planning entry point"""
        raw_path = self.raw_plan()
        return self.interpolate_path(raw_path)
    
    def interpolate_path(self, raw_path):
        """Uniformly interpolate path including explicit goal point"""
        if raw_path is None or len(raw_path) < 1:
            return None
            
        # Ensure goal is explicitly included
        if not np.allclose(raw_path[-1], self.goal):
            raw_path = np.vstack([raw_path, self.goal])

        # Handle single-point paths
        if len(raw_path) == 1:
            return np.array([raw_path[0]] * self.num_points)

        # Calculate cumulative path length
        segments = raw_path[1:] - raw_path[:-1]
        seg_lengths = np.linalg.norm(segments, axis=1)
        total_length = np.sum(seg_lengths)

        # Handle zero-length edge case
        if total_length == 0:
            return np.tile(raw_path[0], (self.num_points, 1))

        # Create parameterized sampling points
        s_values = np.linspace(0, total_length, self.num_points)
        cum_lengths = np.insert(np.cumsum(seg_lengths), 0, 0)

        interpolated = []
        for s in s_values:
            # Find current segment index
            seg_idx = np.clip(
                np.searchsorted(cum_lengths, s) - 1,
                0, len(segments)-1
            )
            
            # Calculate interpolation ratio
            seg_start = cum_lengths[seg_idx]
            available_length = s - seg_start
            ratio = available_length / seg_lengths[seg_idx]
            
            # Linear interpolation between nodes
            p0 = raw_path[seg_idx]
            p1 = raw_path[seg_idx + 1]
            interpolated.append(p0 + ratio * (p1 - p0))

        # Force exact goal position at endpoint
        interpolated[-1] = self.goal
        return np.array(interpolated)

    def sample(self):
        """Random sampling using local RNG"""
        if self.rng.random() < 0.1:  # Goal bias
            return self.goal
        return self.rng.uniform(0, 1, 2)

    def find_nearest(self, sample):
        _, idx = self.kd_tree.query(sample)
        return self.nodes[idx]

    def steer(self, from_node, to_point):
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)
        step = min(distance, self.step_size)
        new_pos = from_node.position + (direction / distance) * step
        return Node(new_pos, from_node)

    def collision_free(self, p1, p2):
        for (ox, oy, r) in self.obstacles:
            if self.segment_obstacle_distance(p1, p2, (ox, oy, r)) < r:
                return False
        return True

    def segment_obstacle_distance(self, p1, p2, obstacle):
        ox, oy, r = obstacle
        v = p2 - p1
        w = np.array([ox, oy]) - p1
        c1 = np.dot(w, v)
        
        if c1 <= 0:
            return np.linalg.norm(p1 - [ox, oy])
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(p2 - [ox, oy])
        b = c1 / c2
        pb = p1 + b * v
        return np.linalg.norm(pb - [ox, oy])

    def find_near_nodes(self, new_node):
        return [node for node in self.nodes 
                if np.linalg.norm(node.position - new_node.position) <= self.search_radius]

    def choose_parent(self, new_node, near_nodes):
        min_cost = np.inf
        best_node = None
        for node in near_nodes:
            if self.collision_free(node.position, new_node.position):
                cost = node.cost + np.linalg.norm(node.position - new_node.position)
                if cost < min_cost:
                    min_cost = cost
                    best_node = node
        if best_node:
            new_node.parent = best_node
            new_node.cost = min_cost
        return new_node

    def add_node(self, node):
        self.nodes.append(node)
        self.kd_tree = KDTree([n.position for n in self.nodes])

    def rewire(self, near_nodes, new_node):
        for node in near_nodes:
            if self.collision_free(new_node.position, node.position):
                new_cost = new_node.cost + np.linalg.norm(new_node.position - node.position)
                if new_cost < node.cost:
                    node.parent = new_node
                    node.cost = new_cost

    def reached_goal(self, node):
        return np.linalg.norm(node.position - self.goal) <= self.goal_threshold

    def find_best_goal_node(self):
        goal_nodes = [node for node in self.nodes 
                     if np.linalg.norm(node.position - self.goal) <= self.goal_threshold]
        return min(goal_nodes, key=lambda n: n.cost, default=None)

    def extract_path(self, goal_node=None):
        path = []
        current = goal_node or self.find_best_goal_node()
        while current:
            path.append(current.position)
            current = current.parent
        return np.array(path[::-1]) if path else None

# Use the same visualization function and test cases
if __name__ == "__main__":
    test_cases = [
        (np.array([0.1, 0.1]), np.array([0.9, 0.9]), [[0.5, 0.5, 0.2]]),
        (np.array([0.2, 0.2]), np.array([0.8, 0.8]), [[0.4, 0.4, 0.1], [0.6, 0.6, 0.1]]),
        (np.array([0.3, 0.3]), np.array([0.7, 0.7]), [[0.5, 0.5, 0.15]]),
        (np.array([0.1, 0.9]), np.array([0.9, 0.1]), [[0.5, 0.5, 0.25]]),
        (np.array([0.1, 0.2]), np.array([0.9, 0.8]), [[0.3, 0.4, 0.1], [0.7, 0.6, 0.1]]),
        (np.array([0.2, 0.1]), np.array([0.8, 0.9]), [[0.4, 0.3, 0.15], [0.6, 0.7, 0.15]]),
        (np.array([0.3, 0.2]), np.array([0.7, 0.8]), [[0.5, 0.4, 0.2], [0.6, 0.5, 0.1]]),
        (np.array([0.4, 0.1]), np.array([0.6, 0.9]), [[0.5, 0.5, 0.25], [0.4, 0.6, 0.1]]),
        (np.array([0.1, 0.4]), np.array([0.9, 0.6]), [[0.3, 0.5, 0.2], [0.7, 0.4, 0.1]]),
        (np.array([0.2, 0.3]), np.array([0.8, 0.7]), [[0.4, 0.6, 0.15], [0.5, 0.5, 0.2]]),
    ]
    
    for i, (s, g, o) in enumerate(test_cases):
        print(f"Testing case {i+1}")
        # planner = RRTStarPlanner(s, g, o)
        # path = planner.plan()
        
        for seed in [42, 123, 567]:
            planner = RRTStarPlanner(s, g, o, seed=seed)
            path = planner.plan()
            if path is not None:
                plot_trajectory(s, g, o, path)
                print("Path waypoints num", len(path))
            else:
                print("No path found for case", i+1)