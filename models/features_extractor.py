
import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models.pointnet import PointNetEncoder

class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64, env_latent_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        # Compute the final input dimension after concatenation
        input_dim = observation_space["current"].shape[0] + observation_space["goal"].shape[0]
        
        self.pointnet = PointNetEncoder(latent_dim=env_latent_dim)
        
        # Simple MLP to process the concatenated input
        self.fc = nn.Sequential(
            nn.Linear(input_dim + env_latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: dict) -> th.Tensor:
        # Extract and concatenate "current" and "goal"
        env_embedding = self.pointnet(observations["obstacles"])
        concatenated = th.cat([observations["current"], observations["goal"], env_embedding], dim=-1)
        return self.fc(concatenated)
