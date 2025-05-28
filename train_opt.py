import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models.neural_planner import PolicyNet
from models.features_extractor import FeaturesExtractor
from models.custom_policy import CustomActorCriticPolicy

from simple_2d import Simple2DEnv
from stable_baselines3 import PPO

from data_loader import TrajDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Obstacle Penalty ---
def obstacle_loss(traj, obstacles_primitives, safe_margin=0.03):
    """
    Penalizes trajectory points that are within a 'safe_margin' of any obstacle.
    traj: [H+1, 2] or [B, H+1, 2]
    obstacles_primitives: [B, N, 3] (ox, oy, r)
    """
    # Ensure batch dimension
    if traj.dim() == 2:
        traj = traj.unsqueeze(0)  # [1, H+1, 2]
    B, H1, D = traj.shape
    _, N, _ = obstacles_primitives.shape

    centers = obstacles_primitives[..., :2]  # [B, N, 2]
    radii = obstacles_primitives[..., 2]     # [B, N]

    # Compute distances: [B, H+1, N]
    dists = torch.cdist(traj, centers)  # [B, H+1, N]
    dists_to_surface = dists - radii.unsqueeze(1)  # [B, H+1, N]
    min_dists = torch.min(dists_to_surface, dim=2).values  # [B, H+1]
    penalty = torch.relu(safe_margin - min_dists).sum()
    return penalty

# --- Trajectory Loss: boundary and smoothness ---
def traj_loss(traj): # traj is [H+1, 2] or [B, H+1, 2]
    """
    Penalize trajectory points that are outside the [0, 1] bounds in any dimension.
    Additionally, encourage smooth trajectories by penalizing large step sizes and accelerations.
    """
    lower_bound = 0.0
    upper_bound = 1.0

    # Penalize points outside bounds
    lower_viol = torch.relu(lower_bound - traj)
    upper_viol = torch.relu(traj - upper_bound)
    out_of_bounds_penalty = (lower_viol + upper_viol).sum()

    # Penalize large step sizes
    step_diffs = traj[:, 1:] - traj[:, :-1]  # [B, H, 2]
    step_size_penalty = torch.norm(step_diffs, dim=-1).sum() * 0.1

    # Penalize large accelerations
    accel_diffs = traj[:, 2:] - 2 * traj[:, 1:-1] + traj[:, :-2]  # [B, H-1, 2]
    acceleration_penalty = torch.norm(accel_diffs, dim=-1).sum() * 0.01

    return out_of_bounds_penalty + step_size_penalty + acceleration_penalty

def main():
    # --- Dataset and DataLoader ---
    dataset_path = 'data/pd_4k.npz'
    dataset = TrajDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Create environment
    reference = np.load(dataset_path, allow_pickle=True)
    env = Simple2DEnv(reference=reference, rand_sg=True)  # Randomize start and goal positions
    
    # --- Policy Network ---
    policy_kwargs = dict(
        features_extractor_class=FeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=64),
    )

    # Define PPO model
    ppo_model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1)

    model = PolicyNet(feature_extractor=ppo_model.policy.features_extractor, custom_policy=ppo_model.policy).to(device)
    model.load_state_dict(torch.load("checkpoints/bc_se_4k/best_model.pth", weights_only=True)) # Load the Pre-trained model

    # # Fix the PointNet of feature extractor (freeze its parameters)
    # for param in model.feature_extractor.pointnet.parameters():
    #     param.requires_grad = False
    # model.feature_extractor.pointnet.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    H = 19  # Planning horizon

    # --- Logging and checkpoint setup ---
    import os
    from torch.utils.tensorboard import SummaryWriter
    save_dir = 'checkpoints/opt_se'
    log_dir = 'runs/opt'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    num_epochs = 120  # <-- Specify number of epochs here

    # Training loop
    total_steps = num_epochs * len(dataloader)
    step = 0
    with tqdm(total=total_steps, desc="Training", unit="batch") as pbar:
        for epoch in range(num_epochs):
            for episode, batch in enumerate(dataloader):
                current = batch['start'].to(device)
                goal = batch['goal'].to(device)
                obstacles = batch['obstacles'].to(device)
                obstacles_primitives = batch['obstacle_primitives'].to(device)

                # Create observations dictionary
                observations = {
                    "current": current,
                    "goal": goal,
                    "obstacles": obstacles  # Not used by feature extractor
                }

                traj = [current]
                for t in range(H):
                    action = model(observations)
                    current = current + action
                    observations = {
                        "current": current,
                        "goal": goal,
                        "obstacles": obstacles
                    }
                    traj.append(current) # H+1 length list of shape [B, D]
                
                # Convert the traj to shape [B, H+1, D]
                traj = torch.stack(traj, dim=1)  # [B, H+1, D] where D=2

                # Compare the last position in traj with the goal for each batch
                final_dist_loss = torch.norm(traj[:, -1, :] - goal, dim=-1).sum()
                
                traj_l = traj_loss(traj)
                obs_l = obstacle_loss(traj, obstacles_primitives)

                loss = 100 * final_dist_loss + 5 * obs_l + traj_l

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # TensorBoard logging
                global_step = epoch * len(dataloader) + episode
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/goal', final_dist_loss.item(), global_step)
                writer.add_scalar('Loss/obs', obs_l.item(), global_step)
                writer.add_scalar('Loss/traj', traj_l.item(), global_step)

                # Update tqdm bar
                pbar.set_postfix({
                    "Epoch": epoch,
                    "Loss": f"{loss.item():.4f}",
                    "Goal": f"{final_dist_loss.item():.4f}",
                    "Obs": f"{obs_l.item():.4f}",
                    "Traj": f"{traj_l.item():.4f}"
                })
                pbar.update(1)
                step += 1

    writer.close()
    # --- Save the trained policy ---
    torch.save(model.state_dict(), os.path.join(save_dir, 'opt_policy.pth'))
    print(f"Training complete. Policy saved as '{os.path.join(save_dir, 'opt_policy.pth')}'.")
    
if __name__ == "__main__":
    main()