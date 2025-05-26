import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from utils import construct_pointcloud, generate_adverse_task, generate_adverse_tasks_given_obs

class Simple2DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human", reference=None, rand_sg=False, num_obstacle_points=256):
        super().__init__()

        # Define observation & action space
        
        # Observation space is a dictionary with keys "current", "goal", and "obstacles"
        self.observation_space = gym.spaces.Dict({
            "current": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "goal": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "obstacles": gym.spaces.Box(low=0, high=1, shape=(num_obstacle_points, 2), dtype=np.float32),
        })
        
        # # Flattened observation space: current(2) + goal(2) + obstacles(num_obstacle_points*2)
        # total_dim = 4 + (num_obstacle_points * 2)  # 2 for current, 2 for goal, rest for obstacles
        # self.observation_space = gym.spaces.Box(
        #     low=0, high=1, shape=(total_dim,), dtype=np.float32
        # )

        self.action_space = gym.spaces.Box(low=-0.08, high=0.08, shape=(2,), dtype=np.float32)

        # Environment parameters
        self.state = [0, 0]
        self.start = [0, 0]
        self.goal = [1,1]
        self.obstacles = None
        self.bounds = [[0, 1], [0, 1]]
        self.num_obstacle_points = num_obstacle_points

        # Reference trajectory
        self.reference = reference  # Expert data for RSI
        self.ref_traj = None

        # Rendering mode
        self.render_mode = render_mode
        
        # Random Start and Goal
        self.rand_sg = rand_sg

        # Step counter
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reference State Initialization (RSI)
        if self.reference is not None and not self.rand_sg:
            idx = np.random.randint(len(self.reference['starts']))
            self.start = self.reference['starts'][idx]
            self.state = self.reference['starts'][idx]
            self.goal = self.reference['goals'][idx]
            self.obstacles = self.reference['obstacles'][idx]
            self.ref_traj = self.reference['trajectories'][idx]
        elif self.reference is not None and self.rand_sg:
            idx = np.random.randint(len(self.reference['obstacles']))
            self.obstacles = self.reference['obstacles'][idx]
            tasks = generate_adverse_tasks_given_obs(self.obstacles, min_dist=0.5, num_tasks=1)
            self.start, self.goal, _ = map(np.array, tasks[0])
            self.state = np.array(self.start)
        else:
            # Random State and Obstacle Initialization
            tasks = generate_adverse_task(max_obstacles=6, min_dist=0.5, num_tasks=1)
            self.start, self.goal, self.obstacles = map(np.array, tasks[0])
            self.state = np.array(self.start)

        self.steps = 0
        return self._get_obs(), {}
    
    @staticmethod
    def get_reward_done_info(current_state, action, goal, obstacles):
        """
        Compute reward, done, and info (collision, goal_reach) for the given state and action.
        """
        # Compute next state
        next_state = current_state + action
        
        done = False

        # Collision Penalty
        min_dist = np.min([np.linalg.norm(next_state - np.array([ox, oy])) - r for ox, oy, r in obstacles])
        collision = min_dist < 0
        reward = min_dist * 5 if collision else 0

        # Progressive Goal Reward
        dist_to_goal = np.linalg.norm(next_state - goal)
        prev_dist_to_goal = np.linalg.norm(current_state - goal)
        reward += prev_dist_to_goal - dist_to_goal  # Reward for getting closer

        goal_reach = dist_to_goal < 0.03
        if goal_reach:
            reward += 1  # Extra reward for reaching the goal
            done = True

        info = {'collision': collision, 'goal_reach': goal_reach}
        return reward, done, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        new_state = self.state + action
        self.steps += 1
        
        collision = False
        goal_reach = False
        
        reward, done, info = self.get_reward_done_info(self.state, action, self.goal, self.obstacles)
        
        collision = info['collision']
        goal_reach = info['goal_reach']
        
        
        # Update state before the reurn
        self.state = new_state
        
        # Check boundary constraints
        if not (self.bounds[0][0] <= new_state[0] <= self.bounds[0][1] and
                self.bounds[1][0] <= new_state[1] <= self.bounds[1][1]):
            return self._get_obs(), reward, True, False, {'collision': collision, 'goal_reach':goal_reach}  # Early termination on out-of-bounds
                
        # Check timeout
        if self.steps >= 30:
            return self._get_obs(), reward, True, False, {'collision': collision, 'goal_reach':goal_reach}
        

        return self._get_obs(), reward, done, False, {'collision': collision, 'goal_reach':goal_reach}

    def _get_obs(self):
        """Construct the observation as a dictionary."""
        obstacle_cloud = construct_pointcloud(self.obstacles, self.num_obstacle_points)
        return {
            "current": self.state,
            "goal": self.goal,
            "obstacles": obstacle_cloud,
        }
    
    # def _get_obs(self):
    #     """Construct the observation as a flattened array."""
    #     obstacle_cloud = construct_pointcloud(self.obstacles, self.num_obstacle_points)
    #     # Flatten the observation
    #     flattened_obs = np.concatenate([
    #         self.state,           # current position (2)
    #         self.goal,            # goal position (2)
    #         obstacle_cloud.reshape(-1)  # flattened obstacles (num_obstacle_points*2)
    #     ]).astype(np.float32)
        
    #     return flattened_obs


    def render(self):
        plt.figure(figsize=(5, 5))
        plt.xlim(self.bounds[0])
        plt.ylim(self.bounds[1])

        plt.scatter(*self.start, color='blue', label='Start', s=100)
        plt.scatter(*self.goal, color='green', label='Goal', s=100)

        for ox, oy, r in self.obstacles:
            circle = plt.Circle((ox, oy), r, color='red', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.scatter(*self.state, color='black', label='Agent', s=100)

        # Plot reference trajectory if available
        if self.ref_traj is not None:
            ref_traj = np.array(self.ref_traj)
            plt.plot(ref_traj[:, 0], ref_traj[:, 1], color='orange', label='Reference Trajectory')
            if self.steps < len(ref_traj):
                plt.scatter(*ref_traj[self.steps], color='purple', label='Current Ref Step', s=100)

        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load env from dataset
    dataset_path = 'data/pd_100_single_env.npz'  # Update this path if needed
    reference = np.load(dataset_path, allow_pickle=True)
    env = Simple2DEnv(reference=reference, rand_sg=False)
    
    # # Generate a random task
    # env = Simple2DEnv(render_mode="human", rand_sg=False)

    obs, _ = env.reset()
    done = False
    while not done:
        direction = obs['goal'] - obs['current']  # Goal - Current
        action = direction / np.linalg.norm(direction) * 0.05  
        obs, reward, done, _, _ = env.step(action)
        print("position:", obs['current'])
        print(f"Reward: {reward}, Done: {done}")
        print("-" * 20)
        env.render()
