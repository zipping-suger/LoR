import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from utils import construct_pointcloud, generate_adverse_task, generate_adverse_tasks_given_obs

class Simple2DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human", reference=None, rand_sg=False, num_obstacle_points=256):
        super().__init__()

        # Define observation & action space
        self.observation_space = gym.spaces.Dict({
            "current": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "goal": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "obstacles": gym.spaces.Box(low=0, high=1, shape=(num_obstacle_points, 2), dtype=np.float32),
        })

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

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        new_state = self.state + action
        self.steps += 1
        
        # Update state
        self.state = new_state
        collision = False
        
        reward = 0
        
        # Check boundary constraints
        if not (self.bounds[0][0] <= new_state[0] <= self.bounds[0][1] and
                self.bounds[1][0] <= new_state[1] <= self.bounds[1][1]):
            return self._get_obs(), 0, True, False, {'collision': collision}  # Early termination on out-of-bounds
                
        # Check timeout
        if self.steps >= 25:
            return self._get_obs(), 0, True, False, {'collision': collision}
        
        # Imitation reward (exponential)
        demo_reward = 0
        if self.ref_traj is not None:
            ref_point = self.ref_traj[self.steps] if self.steps < len(self.ref_traj) else self.goal
            distance = np.linalg.norm(new_state - ref_point)
            pos_reward = np.exp(-20 * distance)
            
            demo_reward = pos_reward
        
        # # Goal dense reward, exponential
        # dense_reward = np.exp(-20 * np.linalg.norm(new_state - self.goal))
        
        # Collision Penalty
        for ox, oy, r in self.obstacles:
            if np.linalg.norm(new_state - np.array([ox, oy])) < r:
                reward -= 0.1  # Collision penalty
                collision = True
                 
        # Collision Penalty by early termination
        for ox, oy, r in self.obstacles:
            if np.linalg.norm(new_state - np.array([ox, oy])) < r:
                collision = True
                return self._get_obs(), 0, True, False, {'collision': collision}  # Early termination on collision
        
        # Check goal condition
        done = np.linalg.norm(self.state - self.goal) < 0.03
        if done:
            reward = reward + 1  # Goal reward
        else:
            reward = reward  + demo_reward

        return self._get_obs(), reward, done, False, {'collision': collision}

    def _get_obs(self):
        """Construct the observation as a dictionary."""
        obstacle_cloud = construct_pointcloud(self.obstacles, self.num_obstacle_points)
        return {
            "current": self.state,
            "goal": self.goal,
            "obstacles": obstacle_cloud,
        }


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
    dataset_path = 'data/pd_8k_r1_ad.npz'  # Update this path if needed
    reference = np.load(dataset_path, allow_pickle=True)
    env = Simple2DEnv(reference=reference, rand_sg=False)

    obs, _ = env.reset()
    done = False
    while not done:
        direction = obs['goal'] - obs['current']  # Goal - Current
        action = direction / np.linalg.norm(direction) * 0.02  
        obs, reward, done, _, _ = env.step(action)
        print("position:", obs['current'])
        print(f"Reward: {reward}, Done: {done}")
        print("-" * 20)
        env.render()
