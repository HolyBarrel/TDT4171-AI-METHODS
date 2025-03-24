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

def run_dqn(env, total_timesteps=10000, model_name="DQN"):
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
    # plots the policy
    plot_policy(model, env, filename=f"{model_name}.png")
    return mean_reward

def plot_policy(model, env, filename="policy_visualization.png"):
    """
    Visualizes the agent's policy on the FrozenLake grid by overlaying arrows for the chosen actions.
    
    Terminal states (holes 'H' and the goal 'G') are annotated with their respective letters.
    The generated plot is saved as a PNG image in the './plots/' directory.
    
    Args:
        model: The trained DQN model.
        env: The FrozenLake environment to visualize.
        filename: The filename for the saved plot image.
    """
    # Retrieves the grid layout from the environment's descriptor and converts it to unicode strings
    grid = env.unwrapped.desc.astype('U1')
    nrows, ncols = grid.shape
    
    # Creates a plot with grid lines to represent each cell
    fig, ax = plt.subplots(figsize=(ncols, nrows))
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #ax.grid(color='black', linestyle='-', linewidth=1)
    
    # Maps discrete actions to arrow symbols
    action_arrows = {
        0: '←', 
        1: '↓',  
        2: '→',  
        3: '↑'  
    }
    
    # Iterates over all states in the environment
    for state in range(env.observation_space.n):
        # Computes the grid cell coordinates from the state index
        row = state // ncols
        col = state % ncols
        cell = grid[row, col]
        # Checks if the cell is terminal (hole or goal)
        if cell in 'H':
            ax.text(col, row, cell, ha='center', va='center', fontsize=27, color='red')
        elif cell == 'G':
            ax.text(col, row, cell, ha='center', va='center', fontsize=29, color='green')
        else:
            # Prepares the observation by wrapping the state in an array
            obs = np.array([state])
            # Predicts the action for the given state using the trained model
            action, _ = model.predict(obs, deterministic=True)
            # Converts the predicted action from a NumPy array to an integer
            if isinstance(action, np.ndarray):
                action = int(action.item())
            # Retrieves the corresponding arrow symbol for the predicted action
            arrow = action_arrows.get(action, '?')
            ax.text(col, row, arrow, ha='center', va='center', fontsize=25, color='blue')
    
    plt.title("Policy Visualization")
    
    # Creates the './plots' directory if it does not exist
    os.makedirs("./plots", exist_ok=True)
    # Saves the plot as a PNG image in the './plots' directory
    save_path = os.path.join("./plots", filename)
    plt.savefig(save_path)
    # Closes the figure to free memory
    plt.close(fig)


if __name__ == "__main__":
    # Executes DQN on the small deterministic environment; target goal: mean reward ~1.0
    mean_reward_1 = run_dqn(env_small_determ, total_timesteps=50000, model_name="DQN_small_determ")
    print(f"Mean reward in small deterministic environment: {mean_reward_1}")
    
    # Executes DQN on the small nondeterministic environment; target goal: mean reward ~0.7
    mean_reward_2 = run_dqn(env_small_nondeterm, total_timesteps=120000, model_name="DQN_small_nondeterm")
    print(f"Mean reward in small nondeterministic environment: {mean_reward_2}")
    
    # Executes DQN on the large nondeterministic environment; target goal: mean reward ~0.5
    mean_reward_3 = run_dqn(env_large_nondeterm, total_timesteps=150000, model_name="DQN_large_nondeterm")
    print(f"Mean reward in large nondeterministic environment: {mean_reward_3}")