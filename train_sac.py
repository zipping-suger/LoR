import torch
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from simple_2d import Simple2DEnv

# Configuration for the SAC training
config = {
    "device": "cpu",  # "cuda" for GPU or "cpu"
    "dataset_path": "data/pd_4k.npz",
    "log_dir": "runs/sac",
    "checkpoint_dir": "checkpoints/sac_se",
    "best_model_dir": "./checkpoints/best_model_sac",
    "wandb_project": "sac_training",
    "total_timesteps": 400_000,
    "batch_size": 64,
    "features_dim": 64,
    "log_std_init": -3.0,  # -4.0 for fine tuning
    "eval_episodes": 20,
    "eval_freq": 10_000,
    "checkpoint_freq": 10_000,
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
    
    # Create environment using the reference dataset
    env = Simple2DEnv(reference=reference, rand_sg=True)
    
    # Configure logging
    logger = configure(config["log_dir"], ["stdout", "tensorboard"])
    
    policy_kwargs = dict(
        log_std_init=config["log_std_init"],
    )
    
    # Initialize a new SAC model with MultiInputPolicy
    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=config["log_dir"],
        device=device,
        batch_size=config["batch_size"],
    )
    
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
        name_prefix="sac_model",
    )
    
    # Set up evaluation environment
    eval_env = Simple2DEnv(reference=reference, rand_sg=True)
    
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