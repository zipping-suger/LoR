import torch
import torch.nn as nn
import numpy as np

from utils import construct_pointcloud
    
from models.custom_policy import CustomActorCriticPolicy
from models.features_extractor import FeaturesExtractor

class PolicyNet(nn.Module):
    def __init__(self, feature_extractor: FeaturesExtractor, custom_policy: CustomActorCriticPolicy):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.policy_net = custom_policy.mlp_extractor.policy_net
        self.action_net = custom_policy.action_net
    
    def forward(self, observations: dict) -> torch.Tensor:
        features = self.feature_extractor(observations)
        latent_action = self.policy_net(features)
        return self.action_net(latent_action)

class NeuralPlanner:
    """Planner that uses DeltaPredictor to generate trajectories autoregressively."""
    def __init__(self, model: PolicyNet, num_obstacle_points=100, max_steps=30):
        self.model = model.eval()
        self.num_obstacle_points = num_obstacle_points
        self.max_steps = max_steps
        self.bounds = ((0, 1), (0, 1))  # Default bounds matching Simple2DEnv

    def plan(self, start, goal, obstacles):
        # obstacle_cloud = construct_pointcloud(obstacles, num_points=100)
        
        # construct observation
        obs = {
            "current": np.array(start, dtype=np.float32),
            "goal": np.array(goal, dtype=np.float32),
            # "obstacles": obstacle_cloud.astype(np.float32),
        }
        
        trajectory = [np.array(start)]
        current_pos = np.array(start, dtype=np.float32)
        
        for _ in range(self.max_steps):
            with torch.no_grad():
                delta = self.model(obs)
            delta = delta.squeeze(0).cpu().numpy()
            new_pos = current_pos + delta
            
            # Check boundary constraints
            if not (self.bounds[0][0] <= new_pos[0] <= self.bounds[0][1] and
                    self.bounds[1][0] <= new_pos[1] <= self.bounds[1][1]):
                break
            
            current_pos = new_pos
            trajectory.append(current_pos.copy())
            
            # Check goal condition
            if np.linalg.norm(current_pos - goal) < 0.05:
                break
        
        return np.array(trajectory)