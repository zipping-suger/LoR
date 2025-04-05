import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    """PointNet encoder without spatial transformers (preserves absolute coordinates)"""
    def __init__(self, latent_dim=3, input_dim=2):
        super().__init__()
        # Shared MLP layers (no feature alignment)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, latent_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(latent_dim)
        
        self.latent_dim = latent_dim

    def forward(self, x):
        # Input shape: (batch_size, num_points, input_dim)
        x = x.transpose(1, 2)  # (bs, input_dim, num_points)
        
        # Feature extraction without spatial transforms
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        
        # Global feature aggregation
        x = torch.max(x, 2)[0]  # (bs, latent_dim)
        return x