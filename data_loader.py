import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import construct_pointcloud

class TrajDataset(Dataset):
    def __init__(self, data_path, num_obstacle_points=256):
        """
        Args:
            data_path: Path to .npz data file
            num_obstacle_points: Fixed number of points to sample for obstacle cloud
        """
        data = np.load(data_path, allow_pickle=True)
        self.starts = data['starts'].astype(np.float32)
        self.goals = data['goals'].astype(np.float32)
        self.obstacles = np.array(data['obstacles'], dtype=np.float32)
        self.trajectories = [traj.astype(np.float32) for traj in data['trajectories']]
        self.obstacle_clouds = [construct_pointcloud(obs, num_obstacle_points) 
                                for obs in self.obstacles]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        goal_pos = self.goals[idx]
        obstacle_cloud = self.obstacle_clouds[idx]
        start_pos = self.starts[idx]
        obs_primitives = self.obstacles[idx]
        return {
            'start': torch.FloatTensor(start_pos),
            'goal': torch.FloatTensor(goal_pos),
            'obstacles': torch.FloatTensor(obstacle_cloud),
            'obstacle_primitives': torch.FloatTensor(obs_primitives),
            'trajectory': torch.FloatTensor(traj)
        }

class PlanningDataset(Dataset):
    def __init__(self, data_path, num_obstacle_points=256):
        """
        Args:
            data_path: Path to .npz data file
            num_obstacle_points: Fixed number of points to sample for obstacle cloud
        """
        data = np.load(data_path, allow_pickle=True)
        
        # Load raw data arrays and ensure correct types
        self.starts = data['starts'].astype(np.float32)
        self.goals = data['goals'].astype(np.float32)
        self.obstacles = data['obstacles']
        self.trajectories = [traj.astype(np.float32) for traj in data['trajectories']]
        
        # Precompute obstacle point clouds for each trajectory
        self.obstacle_clouds = [construct_pointcloud(obs, num_obstacle_points) 
                               for obs in self.obstacles]
        
        # Create index mapping for trajectory steps
        self.index_map = []
        for traj_idx, traj in enumerate(self.trajectories):
            for step in range(len(traj)-1):  # Each step in trajectory becomes a sample
                self.index_map.append((traj_idx, step))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        traj_idx, step = self.index_map[idx]
        
        # Get trajectory data
        traj = self.trajectories[traj_idx]
        current_pos = traj[step]
        next_pos = traj[step + 1]
        
        # Get associated information
        goal_pos = self.goals[traj_idx]
        obstacle_cloud = self.obstacle_clouds[traj_idx]
        
        # Calculate delta
        delta = next_pos - current_pos
        
        # Convert to tensors
        return {
            'current': torch.FloatTensor(current_pos),
            'goal': torch.FloatTensor(goal_pos),
            'obstacles': torch.FloatTensor(obstacle_cloud),
            'delta': torch.FloatTensor(delta)
        }

def create_data_loader(data_path, batch_size=32, shuffle=True, num_workers=4):
    """Create PyTorch DataLoader with default settings"""
    dataset = PlanningDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: {
            'current': torch.stack([x['current'] for x in batch]),
            'goal': torch.stack([x['goal'] for x in batch]),
            'obstacles': torch.stack([x['obstacles'] for x in batch]),
            'delta': torch.stack([x['delta'] for x in batch])
        }
    )

if __name__ == "__main__":
    # Example usage
    loader = create_data_loader('data/pd_8k.npz')
    
    # Show first batch dimensions
    batch = next(iter(loader))
    print("Batch dimensions:")
    print(f"Current positions: {batch['current'].shape}")      # [B, 2]
    print(f"Goal positions: {batch['goal'].shape}")            # [B, 2]
    print(f"Obstacle clouds: {batch['obstacles'].shape}")      # [B, N, 2]
    print(f"Delta targets: {batch['delta'].shape}")            # [B, 2]
    
