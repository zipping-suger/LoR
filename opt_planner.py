import numpy as np
import scipy.optimize as opt
from utils import plot_trajectory

class OptPlanner:
    def __init__(self, start, goal, obstacles, num_points=20):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.num_points = num_points

    def generate_initial_trajectory(self):
        """Generate an initial trajectory with slight offset to avoid straight-line collisions."""
        t = np.linspace(0, 1, self.num_points)
        points = self.start + t[:, None] * (self.goal - self.start)
        # Add perpendicular offset to initial trajectory
        offset = 0.05 * np.sin(np.pi * t)[:, None] * np.array([self.goal[1]-self.start[1], self.start[0]-self.goal[0]])
        return points + offset / np.linalg.norm(self.goal - self.start)

    def trajectory_cost(self, intermediates):
        """Cost function combining length, smoothness, and obstacle proximity."""
        trajectory = np.vstack([self.start, intermediates.reshape(-1, 2), self.goal])
        cost = 0
        
        # Length cost (linear)
        length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        cost += 5.0 * length
        
        # Smoothness cost (acceleration minimization)
        accel = np.diff(trajectory, n=2, axis=0)
        cost += 2.0 * np.sum(np.linalg.norm(accel, axis=1))**2
        
        return cost

    def segment_obstacle_distance(self, p1, p2, obstacle):
        """Calculate minimum distance between segment p1-p2 and obstacle."""
        ox, oy, r = obstacle
        v = p2 - p1
        w = np.array([ox, oy]) - p1
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(p1 - np.array([ox, oy])) - r
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(p2 - np.array([ox, oy])) - r
        b = c1 / c2
        pb = p1 + b*v
        return np.linalg.norm(pb - np.array([ox, oy])) - r

    def obstacle_constraints(self, intermediates):
        """Generate constraints for both waypoints and path segments."""
        points = intermediates.reshape(-1, 2)
        constraints = []
        
        # Check each waypoint
        for (x, y) in points:
            for (ox, oy, r) in self.obstacles:
                constraints.append((x - ox)**2 + (y - oy)**2 - (r*0.95)**2)  # Safety margin
        
        # Check each segment against obstacles
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            for obs in self.obstacles:
                d = self.segment_obstacle_distance(p1, p2, obs)
                constraints.append(d)
        
        return np.array(constraints)

    def optimize_trajectory(self):
        """Optimize trajectory with enhanced constraints and settings."""
        initial_guess = self.generate_initial_trajectory()
        initial_intermediates = initial_guess[1:-1].flatten()
        
        cons = [{'type': 'ineq', 'fun': lambda x: self.obstacle_constraints(x)}]
        
        result = opt.minimize(
            self.trajectory_cost,
            initial_intermediates,
            method='SLSQP',
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-4},
            tol=1e-3
        )
        
        optimized = np.vstack([self.start, result.x.reshape(-1, 2), self.goal])
        return optimized

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
        planner = OptPlanner(s, g, o)
        traj = planner.optimize_trajectory()
        plot_trajectory(s, g, o, traj)
        
