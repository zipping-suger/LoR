import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    """Simplified PointNet encoder"""
    def __init__(self, latent_dim=3, input_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim, 1)
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        x = x.transpose(1, 2)  # (bs, input_dim, num_points)
        x = self.layers(x)
        x = torch.max(x, 2)[0]  # (bs, latent_dim)
        return x
