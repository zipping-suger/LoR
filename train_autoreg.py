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
loss_fn = nn.MSELoss()

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
    # model.load_state_dict(torch.load("checkpoints/bc_se_4k/best_model.pth", weights_only=True)) # Load the Pre-trained model

    # # Fix the PointNet of feature extractor (freeze its parameters)
    # for param in model.feature_extractor.pointnet.parameters():
    #     param.requires_grad = False
    # model.feature_extractor.pointnet.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    H = 19  # Planning horizon

    # --- Logging and checkpoint setup ---
    import os
    from torch.utils.tensorboard import SummaryWriter
    save_dir = 'checkpoints/autoreg_se'
    log_dir = 'runs/autoreg'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    num_epochs = 50  # <-- Specify number of epochs here

    # Training loop
    total_steps = num_epochs * len(dataloader)
    step = 0
    with tqdm(total=total_steps, desc="Training", unit="batch") as pbar:
        for epoch in range(num_epochs):
            for episode, batch in enumerate(dataloader):
                current = batch['start'].to(device)
                goal = batch['goal'].to(device)
                obstacles = batch['obstacles'].to(device)
                exp_traj = batch['trajectory'].to(device)  

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
                
                # Calculate losses
                loss = loss_fn(traj, exp_traj)  # Trajectory loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # TensorBoard logging
                global_step = epoch * len(dataloader) + episode
                writer.add_scalar('Loss/total', loss.item(), global_step)


                # Update tqdm bar
                pbar.set_postfix({
                    "Epoch": epoch,
                    "Loss": f"{loss.item():.4f}",
                })
                pbar.update(1)
                step += 1

    writer.close()
    # --- Save the trained policy ---
    torch.save(model.state_dict(), os.path.join(save_dir, 'autoreg_policy.pth'))
    print(f"Training complete. Policy saved as '{os.path.join(save_dir, 'autoreg_policy.pth')}'.")
    
if __name__ == "__main__":
    main()