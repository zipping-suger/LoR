import numpy as np
from opt_planner import OptPlanner
from rrt_star_planner import RRTStarPlanner
import os
import matplotlib.pyplot as plt
from utils import visualize_samples, generate_random_task, generate_adverse_task, generate_random_tasks_given_obs, generate_adverse_tasks_given_obs


def plan_task(task, planner_type='opt'):
    """Solve task with specified planner and return trajectory"""
    start, goal, obstacles = task
    
    try:
        if planner_type == 'opt':
            planner = OptPlanner(start, goal, obstacles)
            traj = planner.optimize_trajectory()
        elif planner_type == 'rrt':
            planner = RRTStarPlanner(start, goal, obstacles, max_iter=2000)
            traj = planner.plan()
        else:
            raise ValueError("Invalid planner type")
            
        if traj is None or len(traj) < 2:
            return None
            
        # Check if trajectory reaches goal
        final_pos = traj[-1]
        if np.linalg.norm(final_pos - goal) > 0.1:
            return None
            
        return traj
    except:
        return None

def save_data(dataset, filename='planning_data'):
    """Save dataset as compressed numpy file"""
    np.savez_compressed(
        filename,
        starts=np.array([d['start'] for d in dataset]),
        goals=np.array([d['goal'] for d in dataset]),
        obstacles=np.array([d['obstacles'] for d in dataset], dtype=object),
        trajectories=np.array([d['trajectory'] for d in dataset], dtype=object),
    )

def generate_dataset(num_samples=1000, num_tasks=5, planner_type='opt', task_type = 'random', obstacles=None, start_goal = None):
    """Main data generation function"""
    dataset = []
    attempts = 0
    
    while len(dataset) < num_samples:
        
        if obstacles is not None:
            if start_goal is not None:
                start, goal = start_goal
                # jitter start and goal
                start = start + np.random.uniform(-0.05, 0.05, 2)
                goal = goal + np.random.uniform(-0.05, 0.05, 2)
                tasks = [(start, goal, obstacles)]
            else:
                if task_type == 'random':
                    tasks = generate_random_tasks_given_obs(obstacles, num_tasks=num_tasks)
                elif task_type == 'adverse':
                    tasks = generate_adverse_tasks_given_obs(obstacles, num_tasks=num_tasks)
        else:
        
            if task_type == 'random':
                tasks = generate_random_task(num_tasks=num_tasks)
            elif task_type == 'adverse':
                tasks = generate_adverse_task(num_tasks=num_tasks)
            else:
                raise ValueError("Invalid task type")
        
        for task in tasks:
            traj = plan_task(task, planner_type)
            attempts += 1
            
            if traj is not None:
                dataset.append({
                    'start': task[0],
                    'goal': task[1],
                    'obstacles': task[2],
                    'trajectory': traj,
                })
                print(f"Generated {len(dataset)}/{num_samples} samples (success rate: {len(dataset)/attempts:.2f})")
                
            # Check if enough samples have been generated
            if len(dataset) >= num_samples:
                break
    return dataset

if __name__ == "__main__":
    # Configuration
    NUM_SAMPLES = 10_000  # Number of samples to generate
    PLANNER_TYPE = 'rrt'  # 'opt' or 'rrt'
    TASK_TYPE = 'adverse'  # 'random' or 'adverse'
    SAVE_PATH = 'data/pd_10k_dy.npz'
    VISUALIZE = True  # Set to False to disable visualization
    NUM_TASKS = 1 # Ration of trajectory to environment
    NUM_VISUALIZATIONS = 10
    
    # Create dataset
    print(f"Generating {NUM_SAMPLES} samples using {PLANNER_TYPE} planner...")
    dataset = generate_dataset(NUM_SAMPLES, NUM_TASKS, PLANNER_TYPE, TASK_TYPE)
    
    
    # # Create dataset for a single environment
    # obstacles = [
    #     (0.5, 0.5, 0.15),  # Large obstacle in the center
    #     (0.2, 0.2, 0.1),  # Small obstacle top-left
    #     (0.8, 0.2, 0.1),  # Small obstacle top-right
    #     (0.2, 0.8, 0.1),  # Small obstacle bottom-left
    #     (0.8, 0.8, 0.1),  # Small obstacle bottom-right
    # ]
    # dataset = generate_dataset(NUM_SAMPLES, 1, PLANNER_TYPE, TASK_TYPE, obstacles)
    
    # # Ceate dataset for a single environment with fixed start and goal
    # start_goal = (np.array([0.2, 0.2]), np.array([0.8, 0.8]))
    # obstacles = [
    #     (0.5, 0.5, 0.3),  # Large obstacle in the center
    # ]
    # dataset = generate_dataset(NUM_SAMPLES, 1, PLANNER_TYPE, TASK_TYPE, obstacles, start_goal)
    
    # Save data
    save_data(dataset, SAVE_PATH)
    print(f"Dataset saved to {os.path.abspath(SAVE_PATH)}")
    
    # Visualize samples
    if VISUALIZE and len(dataset) > 0:
        print(f"Visualizing first {NUM_VISUALIZATIONS} samples...")
        visualize_samples(dataset, NUM_VISUALIZATIONS)