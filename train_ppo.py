import torch
import os
from torch import nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from simple_2d import Simple2DEnv
from models.custom_policy import CustomActorCriticPolicy
from models.features_extractor import FeaturesExtractor
from models.neural_planner import PolicyNet

# Configuration for the PPO training
config = {
    "device": "cpu",  # "cuda" for GPU or "cpu"
    "dataset_path": "data/above_paths.npz",
    "log_dir": "runs/ppo",
    "checkpoint_dir": "checkpoints/ppo_base",
    "best_model_dir": "./checkpoints/best_model",
    "wandb_project": "ppo_training",
    "total_timesteps": 100_000,
    "batch_size": 64,
    "features_dim": 64,
    "log_std_init": -1.0,  # -4.0 for fine tuning
    "eval_episodes": 20,
    "eval_freq": 10_000,
    "checkpoint_freq": 10_000,
    "load_pretrained_model": False,  # Whether to load a pre-trained model
    "pretrained_model_path": "checkpoints/best_model_nr/best_model.zip",  # Path to the pre-trained model
}

class EnhancedWandbCallback(WandbCallback):
    def _on_step(self) -> bool:
        super()._on_step()
        # Log additional info
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    wandb.log({
                        "custom/episode_reward": info["episode"]["r"],
                        "custom/episode_length": info["episode"]["l"],
                    })
        return True

if __name__ == "__main__":
    # Initialize W&B
    wandb.init(
        project=config["wandb_project"],
        config=config,
    )
    
    # Set device
    device = config["device"]
    
    # Load dataset
    reference = np.load(config["dataset_path"], allow_pickle=True)
    
    # Create environment using the  reference dataset
    env = Simple2DEnv(reference=reference, rand_sg=False)
    
    # # Randomize the environment
    # env = Simple2DEnv()
    
    # Configure logging
    logger = configure(config["log_dir"], ["stdout", "tensorboard"])
    
    policy_kwargs = dict(
        features_extractor_class=FeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=config["features_dim"]),
        log_std_init=config["log_std_init"],
    )
    
    # Load pre-trained model if specified in the config
    if config["load_pretrained_model"] and os.path.exists(config["pretrained_model_path"]):
        print(f"Loading pre-trained model from {config['pretrained_model_path']}")
        model = PPO.load(config["pretrained_model_path"], env=env, device=device)
    else:
        print("No pre-trained model found or loading disabled. Initializing a new model.")
        model = PPO(
            CustomActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=config["log_dir"],
            device=device,
            batch_size=config["batch_size"],
        )
    
    # Load Pre-trained model
    pretrained_model = PolicyNet(feature_extractor=model.policy.features_extractor, custom_policy=model.policy)
    pretrained_model.load_state_dict(torch.load("checkpoints/bc_old/best_model.pth", weights_only=True))
    
    # Load weights into the existing components (DO NOT replace the modules)
    model.policy.features_extractor.load_state_dict(pretrained_model.feature_extractor.state_dict())
    model.policy.mlp_extractor.policy_net.load_state_dict(pretrained_model.policy_net.state_dict())
    model.policy.action_net.load_state_dict(pretrained_model.action_net.state_dict())
    
    # # Check if the feature extractor's parameters have gradients
    # for name, param in model.policy.features_extractor.named_parameters():
    #     if param.requires_grad:
    #         print(f"Parameter '{name}' is trainable.")
    #     else:
    #         print(f"Parameter '{name}' is NOT trainable.")
    
    # Evaluate the model first
    mean_reward, std_reward = evaluate_policy(
        model,
        env=env,
        n_eval_episodes=config["eval_episodes"],
        deterministic=False,
        render=False,
    )

    print(f"Evaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    model.set_logger(logger)
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=config["checkpoint_dir"],
        name_prefix="ppo_model",
    )
    
    # Set up evaluation environment
    eval_env = Simple2DEnv(reference=reference, rand_sg=False)
    
    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config["best_model_dir"],
        eval_freq=config["eval_freq"],
        deterministic=True,
        render=False,
    )
    
    # Add WandbCallback for logging to W&B
    wandb_callback = EnhancedWandbCallback(
        gradient_save_freq=1000,
        model_save_path="wandb_models/",
        verbose=2,
    )
    
    # Train the model with the evaluation and W&B callbacks
    model.learn(total_timesteps=config["total_timesteps"], callback=[checkpoint_callback, eval_callback, wandb_callback])
    
    # Finish W&B run
    wandb.finish()