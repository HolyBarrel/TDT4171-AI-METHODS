import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np
import os

# Create the environment
env_small_determ = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
env_small_nondeterm = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env_large_nondeterm = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

def run_dqn(env, total_timesteps=10000):
    """
    Executes DQN on the given environment and returns the mean reward.
    Args:
        env: The environment to train on.
        total_timesteps: Total number of timesteps for training.
    Returns:
        mean_reward: The mean reward after training.
    """
    model = DQN("MlpPolicy", env, verbose=0,
                exploration_fraction=0.2,       
                learning_starts=500,            
                learning_rate=0.001,            
                buffer_size=100000,             
                batch_size=64,                  
                target_update_interval=500,    
                policy_kwargs={"net_arch": [64, 64]}) 
    model.learn(total_timesteps=total_timesteps)
    # Evaluates the policy over 100 episodes and returns the mean reward
    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    return mean_reward

if __name__ == "__main__":
    # Executes DQN on the small deterministic environment; target goal: mean reward ~1.0
    mean_reward_1 = run_dqn(env_small_determ, total_timesteps=50000)
    print(f"Mean reward in small deterministic environment: {mean_reward_1}")
    
    # Executes DQN on the small nondeterministic environment; target goal: mean reward ~0.7
    mean_reward_2 = run_dqn(env_small_nondeterm, total_timesteps=100000)
    print(f"Mean reward in small nondeterministic environment: {mean_reward_2}")
    
    # Executes DQN on the large nondeterministic environment; target goal: mean reward ~0.5
    mean_reward_3 = run_dqn(env_large_nondeterm, total_timesteps=150000)
    print(f"Mean reward in large nondeterministic environment: {mean_reward_3}")