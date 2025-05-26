import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from simple_2d import Simple2DEnv
from matplotlib.cm import get_cmap

from models.custom_policy import CustomActorCriticPolicy
from models.features_extractor import FeaturesExtractor
from models.neural_planner import PolicyNet

# Load reference data
dataset_path = 'data/pd_100_single_env.npz'  # Update this path if needed
reference = np.load(dataset_path, allow_pickle=True)

# Initialize environment
env = Simple2DEnv(reference=reference, rand_sg=True)


# # Load trained PPO policy
model_ppo = PPO.load("checkpoints/best_model/best_model.zip")

# Load Pre-trained model
# Create environment
policy_kwargs = dict(
features_extractor_class=FeaturesExtractor,
features_extractor_kwargs=dict(features_dim=64),
)
model_bc = PPO(
    CustomActorCriticPolicy,
    env,
    policy_kwargs=policy_kwargs,
    verbose=1)

# Load the Pre-trained model
pretrained_model = PolicyNet(feature_extractor= model_bc.policy.features_extractor, custom_policy= model_bc.policy)
pretrained_model.load_state_dict(torch.load("checkpoints/bc_se/best_model.pth", weights_only=True))

# Load weights into the existing components (DO NOT replace the modules)
model_bc.policy.features_extractor.load_state_dict(pretrained_model.feature_extractor.state_dict())
model_bc.policy.mlp_extractor.policy_net.load_state_dict(pretrained_model.policy_net.state_dict())
model_bc.policy.action_net.load_state_dict(pretrained_model.action_net.state_dict())

models = [model_ppo, model_bc]

# Dynamically generate colors for models using a colormap
cmap = get_cmap("tab10")  # Use a colormap (e.g., 'tab10', 'viridis', etc.)
colors = [cmap(i) for i in range(len(models))]

# Create subplots for rollouts
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    obs, _ = env.reset()
    rollouts = {model_idx: [] for model_idx in range(len(models))}
    done = False

    while not done:
        for model_idx, model in enumerate(models):
            rollouts[model_idx].append(obs["current"].copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

    # Convert trajectories to numpy arrays
    rollouts = {model_idx: np.array(rollout) for model_idx, rollout in rollouts.items()}

    # Plot trajectories in the current subplot
    ax = axes[i]
    ax.set_xlim(env.bounds[0])
    ax.set_ylim(env.bounds[1])
    ax.grid(True)

    # Plot start, goal, and obstacles
    ax.scatter(*env.start, color='blue', label='Start', s=100)
    ax.scatter(*env.goal, color='green', label='Goal', s=100)
    for ox, oy, r in env.obstacles:
        circle = plt.Circle((ox, oy), r, color='red', alpha=0.5)
        ax.add_patch(circle)

    # Get reference trajectory
    ref_traj = env.ref_traj if env.ref_traj is not None else []

    # Plot reference trajectory
    if len(ref_traj) > 0:
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'g--', label='Reference Trajectory')
        ax.scatter(ref_traj[:, 0], ref_traj[:, 1], color='green', s=10, label='Reference Points')

    # Plot rollouts for each model
    for model_idx, rollout in rollouts.items():
        ax.plot(rollout[:, 0], rollout[:, 1], '-', color=colors[model_idx], label=f'Model {model_idx + 1} Rollout')
        ax.scatter(rollout[:, 0], rollout[:, 1], color=colors[model_idx], s=10)

    ax.legend()
    ax.set_title(f"Rollout {i + 1}")

# Adjust layout and show the plot
plt.tight_layout()
plt.suptitle("Reference vs. Model Rollouts", y=1.02)
plt.show()

# Evaluate each model
from stable_baselines3.common.evaluation import evaluate_policy

# Create a separate evaluation environment
eval_env = Simple2DEnv(reference=reference, rand_sg=True)

# Evaluate each model
n_eval_episodes = 500  # Number of episodes to evaluate
for model_idx, model in enumerate(models):
    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        warn=True,
        return_episode_rewards=False
    )

    # Custom loop to count collisions and calculate rollout lengths
    total_collisions = 0
    total_rollout_length = 0
    total_goal_reaches = 0
    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        rollout_length = 0
        episode_goal_reached = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)
            total_collisions += info.get('collision', 0)
            rollout_length += 1
            
            # Check if goal was reached on this step
            if info.get('goal_reach', False):
                episode_goal_reached = True
        
        # Count successful episodes
        if episode_goal_reached:
            total_goal_reaches += 1
        total_rollout_length += rollout_length

    # Calculate average collisions and average rollout length
    average_collisions = total_collisions / n_eval_episodes
    average_rollout_length = total_rollout_length / n_eval_episodes
    goal_reach_rate = total_goal_reaches / n_eval_episodes * 100  # as percentage

    print(f"Evaluation Results for Model {model_idx + 1}:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Goal reach rate: {goal_reach_rate:.2f}%")
    print(f"Average collisions per episode: {average_collisions:.2f}")
    print(f"Average rollout length: {average_rollout_length:.2f}")
    print("-" * 40)