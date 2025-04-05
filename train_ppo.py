import torch
from torch import nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from simple_2d import Simple2DEnv
from models.custom_policy import CustomActorCriticPolicy
from models.features_extractor import FeaturesExtractor
from models.neural_planner import PolicyNet

if __name__ == "__main__":
    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu" # intended to run on CPU
    
    # Load dataset
    dataset_path = "data/pd_4k.npz"
    reference = np.load(dataset_path, allow_pickle=True)
    
    # Create environment
    # env = make_vec_env(lambda: Simple2DEnv(reference=reference), n_envs=4)
    env = Simple2DEnv(reference=reference)
    
    # Configure logging
    log_dir = "runs/ppo"
    logger = configure(log_dir, ["stdout", "tensorboard"])
    
    policy_kwargs = dict(
    features_extractor_class=FeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=64),
    log_std_init = -4.0, # Initialize log_std to a small value as the range of actions is [-0.08, 0.08]
)
    
    model = PPO(
    CustomActorCriticPolicy,
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=log_dir,
    device=device,
    batch_size=64,
    )
    
    # # Load Pre-trained model
    # # Load the Pre-trained model
    # pretrained_model = PolicyNet(feature_extractor= model.policy.features_extractor, custom_policy= model.policy)
    # pretrained_model.load_state_dict(torch.load("checkpoints/sl_4k/best_model.pth", weights_only=True))


    # # Load weights into the existing components (DO NOT replace the modules)
    # model.policy.features_extractor.load_state_dict(pretrained_model.feature_extractor.state_dict())
    # model.policy.mlp_extractor.policy_net.load_state_dict(pretrained_model.policy_net.state_dict())
    # model.policy.action_net.load_state_dict(pretrained_model.action_net.state_dict())
    
    
    # Evaluate the model first
    mean_reward, std_reward = evaluate_policy(
        model,
        env=env,
        n_eval_episodes=20,  # Number of episodes to evaluate
        deterministic=False,   # Use deterministic actions
        render=False,         # Set to True if you want to visualize
    )

    print(f"Evaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    model.set_logger(logger)
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path="checkpoints/ppo_sparse", name_prefix="ppo_model")
       
    # Set up evaluation environment
    eval_env = Simple2DEnv(reference=reference, rand_sg=True)

    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best_model",
        eval_freq=10_000,  # Evaluate every 10,000 steps
        deterministic=True,
        render=False,
    )

    # Train the model with the evaluation callback
    model.learn(total_timesteps=400_000, callback=[checkpoint_callback, eval_callback])
    
    
    
