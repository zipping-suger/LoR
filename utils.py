import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(start, goal, obstacles, trajectory):
    """Visualization with obstacle safety margins and fixed [0,1] range"""
    plt.figure(figsize=(6, 6))
    
    # Set axis limits first to ensure proper scaling
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Plot safety margins
    for ox, oy, r in obstacles:
        plt.gca().add_patch(plt.Circle((ox, oy), r*0.95, color='red', alpha=0.3, linestyle='--'))
        plt.gca().add_patch(plt.Circle((ox, oy), r, color='red', alpha=0.2))
    
    # Plot trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', lw=1.5, markersize=4)
    
    # Plot start/goal markers
    plt.scatter(*start, c='green', s=150, marker='s', edgecolor='black', zorder=3)
    plt.scatter(*goal, c='blue', s=150, marker='*', edgecolor='black', zorder=3)
    
    # Add velocity vectors
    for i in range(len(trajectory)-1):
        dx = trajectory[i+1,0] - trajectory[i,0]
        dy = trajectory[i+1,1] - trajectory[i,1]
        plt.arrow(trajectory[i,0], trajectory[i,1], dx*0.8, dy*0.8,
                  head_width=0.02, fc='darkblue', ec='darkblue', zorder=2)
    
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def visualize_samples(dataset, num_samples=3):
    """Visualize samples with fixed [0,1] range in subplots"""
    plt.figure(figsize=(15, 5))
    
    for i in range(min(num_samples, len(dataset))):
        ax = plt.subplot(1, num_samples, i+1)
        sample = dataset[i]
        
        # Set axis limits first
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Plot safety margins and obstacles
        for ox, oy, r in sample['obstacles']:
            ax.add_patch(plt.Circle((ox, oy), r*0.95, color='red', alpha=0.3, linestyle='--'))
            ax.add_patch(plt.Circle((ox, oy), r, color='red', alpha=0.2))
        
        # Plot trajectory
        traj = sample['trajectory']
        ax.plot(traj[:, 0], traj[:, 1], 'b-o', lw=1.5, markersize=4)
        
        # Plot start/goal markers
        ax.scatter(*sample['start'], c='green', s=150, marker='s', edgecolor='black', zorder=3)
        ax.scatter(*sample['goal'], c='blue', s=150, marker='*', edgecolor='black', zorder=3)
        
        # Add velocity vectors
        for j in range(len(traj)-1):
            dx = traj[j+1,0] - traj[j,0]
            dy = traj[j+1,1] - traj[j,1]
            ax.arrow(traj[j,0], traj[j,1], dx*0.8, dy*0.8,
                     head_width=0.02, fc='darkblue', ec='darkblue', zorder=2)
        
        ax.set_title(f"Sample {i+1}")
        ax.set_aspect('equal')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
def distance_line_segment_point(a, b, p):
    """Calculate the shortest distance from point p to the line segment between a and b."""
    a = np.array(a)
    b = np.array(b)
    p = np.array(p)
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)  # Avoid division by zero
    t = np.clip(t, 0.0, 1.0)
    nearest = a + t * ab
    return np.linalg.norm(p - nearest)
    
    
def generate_start_goal(min_dist, obstacles=None, check_collision=True):
    """Generate valid start and goal positions."""
    while True:
        start = np.random.uniform(0, 1, 2)
        goal = np.random.uniform(0, 1, 2)
        if np.linalg.norm(start - goal) < min_dist:
            continue  # Ensure minimum distance between start and goal
        
        if check_collision and obstacles is not None:
            collision = False
            for ox, oy, r in obstacles:
                start_dist = np.linalg.norm(start - [ox, oy])
                goal_dist = np.linalg.norm(goal - [ox, oy])
                if start_dist <= r + 0.01 or goal_dist <= r + 0.01:
                    collision = True
                    break
            if collision:
                continue
        
        return start, goal

def is_path_blocked(start, goal, obstacles):
    """Check if the path between start and goal is blocked by any obstacle."""
    for ox, oy, r in obstacles:
        dist = distance_line_segment_point(start, goal, (ox, oy))
        if dist <= r:
            return True
    return False

def generate_random_task(max_obstacles=6, min_dist=0.5, num_tasks=1):
    """Generate a list of valid planning tasks with collision-free start/goal and fixed obstacles."""
    obstacles = [(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0.05, 0.3))
                 for _ in range(np.random.randint(1, max_obstacles + 1))]
    
    tasks = []
    for _ in range(num_tasks):
        start, goal = generate_start_goal(min_dist, obstacles)
        tasks.append((start, goal, obstacles))
    
    return tasks

def generate_adverse_task(max_obstacles=6, min_dist=0.5, num_tasks=1, max_placement_attempts=100):
    """Generate navigation tasks with obstacles blocking the direct path between start and goal."""
    obstacles = [(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0.05, 0.3))
                 for _ in range(np.random.randint(1, max_obstacles + 1))]
    
    tasks = []
    for _ in range(num_tasks):
        task_found = False
        placement_attempts = 0
        
        while not task_found:
            if placement_attempts >= max_placement_attempts:
                return generate_adverse_task(max_obstacles, min_dist, num_tasks, max_placement_attempts)
            
            start, goal = generate_start_goal(min_dist, obstacles)
            if is_path_blocked(start, goal, obstacles):
                tasks.append((start.tolist(), goal.tolist(), obstacles))
                task_found = True
            
            placement_attempts += 1
    
    return tasks

def generate_random_tasks_given_obs(obstacles, min_dist=0.5, num_tasks=1):
    """Generate random tasks given a fixed set of obstacles."""
    tasks = []
    for _ in range(num_tasks):
        start, goal = generate_start_goal(min_dist, obstacles)
        tasks.append((start, goal, obstacles))
    return tasks

def generate_adverse_tasks_given_obs(obstacles, min_dist=0.5, num_tasks=1, max_placement_attempts=100):
    """Generate adverse tasks given a fixed set of obstacles."""
    tasks = []
    for _ in range(num_tasks):
        task_found = False
        placement_attempts = 0
        
        while not task_found:
            if placement_attempts >= max_placement_attempts:
                return tasks
            
            start, goal = generate_start_goal(min_dist, obstacles)
            if is_path_blocked(start, goal, obstacles):
                tasks.append((start.tolist(), goal.tolist(), obstacles))
                task_found = True
            
            placement_attempts += 1
    
    return tasks

    
def construct_pointcloud(obstacles, num_points):
    """
    Generate a point cloud representing obstacle boundaries
    
    Args:
        obstacles: List of obstacles, each represented as (x, y, radius)
        num_points: Total number of points to generate around all obstacles
        
    Returns:
        Numpy array of shape (N, 2) containing obstacle boundary points
    """
    if obstacles is None or num_points == 0:
        return np.empty((0, 2))
    
    points = []
    total_radius = sum(r for _, _, r in obstacles)  # Total radius
    points_per_obstacle = [int(num_points * (r / total_radius)) for _, _, r in obstacles]
    remainder = num_points - sum(points_per_obstacle)
    
    # Distribute remaining points to the largest obstacles
    for i in range(remainder):
        max_index = max(range(len(obstacles)), key=lambda idx: obstacles[idx][2])
        points_per_obstacle[max_index] += 1
    
    for (ox, oy, r), n_pts in zip(obstacles, points_per_obstacle):
        # Generate evenly spaced angles
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        
        # Calculate boundary points
        x = ox + r * np.cos(theta)
        y = oy + r * np.sin(theta)
        
        points.append(np.column_stack([x, y]))
    
    return np.vstack(points) if points else np.empty((0, 2))


# def construct_pointcloud(obstacles, num_points):
#     """
#     Generate a point cloud representing obstacle interiors
    
#     Args:
#         obstacles: List of obstacles, each represented as (x, y, radius)
#         num_points: Total number of points to generate across all obstacles
        
#     Returns:
#         Numpy array of shape (N, 2) containing obstacle interior points
#     """
#     if obstacles is None or num_points == 0:
#         return np.empty((0, 2))
    
#     points = []
#     total_area = sum(r**2 for _, _, r in obstacles)  # Total area proportional to r^2
#     points_per_obstacle = [int(num_points * (r**2 / total_area)) for _, _, r in obstacles]
#     remainder = num_points - sum(points_per_obstacle)
    
#     # Distribute remaining points to the largest obstacles
#     for i in range(remainder):
#         max_index = max(range(len(obstacles)), key=lambda idx: obstacles[idx][2]**2)
#         points_per_obstacle[max_index] += 1
    
#     for (ox, oy, r), n_pts in zip(obstacles, points_per_obstacle):
#         # Generate random points within the disk
#         radii = np.sqrt(np.random.uniform(0, 1, n_pts)) * r  # Uniform distribution in the disk
#         angles = np.random.uniform(0, 2 * np.pi, n_pts)
        
#         x = ox + radii * np.cos(angles)
#         y = oy + radii * np.sin(angles)
        
#         points.append(np.column_stack([x, y]))
    
#     return np.vstack(points) if points else np.empty((0, 2))