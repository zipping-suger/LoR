import argparse
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time

from data_loader import PlanningDataset
from models.neural_planner import PolicyNet
from models.features_extractor import FeaturesExtractor
from models.custom_policy import CustomActorCriticPolicy

from simple_2d import Simple2DEnv
from stable_baselines3 import PPO

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_dir, log_dir):
    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()
        
        # Training loop
        for batch in train_loader:
            current = batch['current'].to(device)
            goal = batch['goal'].to(device)
            obstacles = batch['obstacles'].to(device)
            delta_target = batch['delta'].to(device)
            

            
            # Create observations dictionary
            observations = {
                "current": current,
                "goal": goal,
                "obstacles": obstacles  # Not used by feature extractor
            }
            
            optimizer.zero_grad()
            delta_pred = model(observations)  # Single output
            loss = criterion(delta_pred, delta_target)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                current = batch['current'].to(device)
                goal = batch['goal'].to(device)
                # obstacles = batch['obstacles'].to(device)
                delta_target = batch['delta'].to(device)
                
                observations = {
                    "current": current,
                    "goal": goal,
                    "obstacles": obstacles
                }
                
                delta_pred = model(observations)
                loss = criterion(delta_pred, delta_target)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Time/epoch', epoch_time, epoch)
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved new best model with val loss {avg_val_loss:.4f}")
    
    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train Delta Predictor')
    parser.add_argument('--data_path', type=str, default='data/pd_10k_dy.npz', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints/bc_se', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='runs/bc', help='Directory for TensorBoard logs')
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset_path = args.data_path
    reference = np.load(dataset_path, allow_pickle=True)
    
    # Create environment
    env = Simple2DEnv(reference=reference)
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
    
    model = PolicyNet(feature_extractor= ppo_model.policy.features_extractor, custom_policy= ppo_model.policy).to(device)

    # Load dataset
    dataset = PlanningDataset(args.data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    collate_fn = lambda batch: {
        'current': torch.stack([x['current'] for x in batch]),
        'goal': torch.stack([x['goal'] for x in batch]),
        'obstacles': torch.stack([x['obstacles'] for x in batch]),
        'delta': torch.stack([x['delta'] for x in batch])
    }
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Train model
    train(model, train_loader, val_loader, criterion, optimizer, device, 
          args.epochs, args.save_dir, args.log_dir)

if __name__ == '__main__':
    main()