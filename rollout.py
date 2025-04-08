import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from simple_2d import Simple2DEnv

from models.custom_policy import CustomActorCriticPolicy
from models.features_extractor import FeaturesExtractor
from models.neural_planner import PolicyNet

# Load reference data
dataset_path = 'data/pd_2.npz'  # Update this path if needed
reference = np.load(dataset_path, allow_pickle=True)

# Initialize environment
# env = Simple2DEnv()
env = Simple2DEnv(reference=reference, rand_sg=False)


# Load trained PPO policy
model = PPO.load("checkpoints/best_model/best_model.zip")
# model = PPO.load("checkpoints/ppo_nr_s2/ppo_model_500000_steps.zip")

# # Load Pre-trained model
# # Create environment
# policy_kwargs = dict(
# features_extractor_class=FeaturesExtractor,
# features_extractor_kwargs=dict(features_dim=64),
# )
# model = PPO(
#     CustomActorCriticPolicy,
#     env,
#     policy_kwargs=policy_kwargs,
#     verbose=1)

# # Load the Pre-trained model
# pretrained_model = PolicyNet(feature_extractor= model.policy.features_extractor, custom_policy= model.policy)
# pretrained_model.load_state_dict(torch.load("checkpoints/bc_old/best_model.pth", weights_only=True))

# # Load weights into the existing components (DO NOT replace the modules)
# model.policy.features_extractor.load_state_dict(pretrained_model.feature_extractor.state_dict())
# model.policy.mlp_extractor.policy_net.load_state_dict(pretrained_model.policy_net.state_dict())
# model.policy.action_net.load_state_dict(pretrained_model.action_net.state_dict())

# Create subplots for 6 rollouts
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    obs, _ = env.reset()
    rollout = []
    actions = []
    done = False
    total_reward = 0  # Initialize total reward
    collision_count = 0  # Initialize collision count

    while not done:
        rollout.append(obs["current"].copy())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        actions.append(action)
        total_reward += reward  # Accumulate reward
        collision_count += info.get('collision', 0)
        
    # Print total reward for the rollout
    print(f"Rollout {i + 1} Total Reward: {total_reward}")
    print(f"Rollout {i + 1} Collision Count: {collision_count}")

    # Convert trajectories to numpy arrays
    rollout = np.array(rollout)

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
    #   ref_traj = env.ref_traj if env.ref_traj is not None and not env.rand_sg else []

    # Plot reference trajectory
    if len(ref_traj) > 0:
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'g--', label='Reference Trajectory')
        ax.scatter(ref_traj[:, 0], ref_traj[:, 1], color='green', s=10, label='Reference Points')

    # Plot model rollout
    ax.plot(rollout[:, 0], rollout[:, 1], 'b-', label='Trained Model Rollout')
    ax.scatter(rollout[:, 0], rollout[:, 1], color='blue', s=10, label='Rollout Points')

    ax.legend()
    ax.set_title(f"Rollout {i + 1}")

# Adjust layout and show the plot
plt.tight_layout()
plt.suptitle("Reference vs. Model Rollouts", y=1.02)
plt.show()


# Evaluate the model
from stable_baselines3.common.evaluation import evaluate_policy

# Initialize variables to track collisions
total_collisions = 0
n_eval_episodes = 200  # Number of episodes to evaluate

# Create a separate evaluation environment
# eval_env = Simple2DEnv()
eval_env = Simple2DEnv(reference=reference, rand_sg=False)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=n_eval_episodes,  # Number of episodes to evaluate
    deterministic=True,   # Use deterministic actions
    render=False,         # Set to True if you want to visualize
    warn=True,            # Show warnings if environment is not wrapped properly
    return_episode_rewards=False  # Ensure we don't return episode rewards
)

# Custom loop to count collisions
for _ in range(n_eval_episodes):
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        total_collisions += info.get('collision', 0)

# Calculate average collisions
average_collisions = total_collisions / n_eval_episodes

print(f"Evaluation Results:")
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"Average collisions per episode: {average_collisions:.2f}")

