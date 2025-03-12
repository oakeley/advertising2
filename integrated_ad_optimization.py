#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from collections import deque
import time
import copy
from typing import Optional, Dict, Any, List, Tuple

# TorchRL imports
from torchrl.envs import EnvBase
from torchrl.data import OneHot, Bounded, Unbounded, Binary, Composite
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.objectives import DQNLoss
from torchrl.collectors import SyncDataCollector
from torchrl.modules import EGreedyModule, MLP, QValueModule
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential

# Custom SoftUpdate implementation (since it can't be imported from torchrl.objectives.value)
class SoftUpdate:
    """
    A simple implementation of soft update for target networks.
    
    This class performs soft (Polyak averaging) updates from a source network
    to a target network.
    
    Args:
        source_module: The source module to get parameters from
        target_module: The target module to update parameters in (if None, uses the source's target_network)
        eps (float): The update coefficient (1 - eps) * target_params + eps * source_params
    """
    def __init__(self, source_module, target_module=None, eps=0.01):
        self.source_module = source_module
        
        # If target_module is not specified, try to get it from source_module
        if target_module is None:
            if hasattr(source_module, "target_network"):
                self.target_module = source_module.target_network
            else:
                raise ValueError("target_module not provided and source_module has no target_network attribute")
        else:
            self.target_module = target_module
            
        self.eps = eps
            
    def step(self):
        """Perform a soft update on the target network."""
        with torch.no_grad():
            for target_param, source_param in zip(self.target_module.parameters(), 
                                                 self.source_module.parameters()):
                target_param.data.mul_(1 - self.eps)
                target_param.data.add_(self.eps * source_param.data)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed

# Generate Realistic Synthetic Data
def generate_synthetic_data(num_samples=1000):
    """Generate synthetic advertising data with realistic correlations."""
    base_difficulty = np.random.beta(2.5, 3.5, num_samples)
    data = {
        "keyword": [f"Keyword_{i}" for i in range(num_samples)],
        "competitiveness": np.random.beta(2, 3, num_samples),
        "difficulty_score": np.random.uniform(0, 1, num_samples),
        "organic_rank": np.random.randint(1, 11, num_samples),
        "organic_clicks": np.random.randint(50, 5000, num_samples),
        "organic_ctr": np.random.uniform(0.01, 0.3, num_samples),
        "paid_clicks": np.random.randint(10, 3000, num_samples),
        "paid_ctr": np.random.uniform(0.01, 0.25, num_samples),
        "ad_spend": np.random.uniform(10, 10000, num_samples),
        "ad_conversions": np.random.randint(0, 500, num_samples),
        "ad_roas": np.random.uniform(0.5, 5, num_samples),
        "conversion_rate": np.random.uniform(0.01, 0.3, num_samples),
        "cost_per_click": np.random.uniform(0.1, 10, num_samples),
        "cost_per_acquisition": np.random.uniform(5, 500, num_samples),
        "previous_recommendation": np.random.choice([0, 1], size=num_samples),
        "impression_share": np.random.uniform(0.1, 1.0, num_samples),
        "conversion_value": np.random.uniform(0, 10000, num_samples)
    }
    
    # Add realistic correlations
    data["difficulty_score"] = 0.7 * data["competitiveness"] + 0.3 * base_difficulty
    data["organic_rank"] = 1 + np.floor(9 * data["difficulty_score"] + np.random.normal(0, 1, num_samples).clip(-2, 2))
    data["organic_rank"] = data["organic_rank"].clip(1, 10).astype(int)
    
    # CTR follows a realistic distribution and correlates negatively with rank
    base_ctr = np.random.beta(1.5, 10, num_samples)
    rank_effect = (11 - data["organic_rank"]) / 10
    data["organic_ctr"] = (base_ctr * rank_effect * 0.3).clip(0.01, 0.3)
    
    # Organic clicks based on CTR and a base impression count
    base_impressions = np.random.lognormal(8, 1, num_samples).astype(int)
    data["organic_clicks"] = (base_impressions * data["organic_ctr"]).astype(int)
    
    # Paid CTR correlates with organic CTR but with more variance
    data["paid_ctr"] = (data["organic_ctr"] * np.random.normal(1, 0.3, num_samples)).clip(0.01, 0.25)
    
    # Paid clicks
    paid_impressions = np.random.lognormal(7, 1.2, num_samples).astype(int)
    data["paid_clicks"] = (paid_impressions * data["paid_ctr"]).astype(int)
    
    # Cost per click higher for more competitive keywords
    data["cost_per_click"] = (0.5 + 9.5 * data["competitiveness"] * np.random.normal(1, 0.2, num_samples)).clip(0.1, 10)
    
    # Ad spend based on CPC and clicks
    data["ad_spend"] = data["paid_clicks"] * data["cost_per_click"]
    
    # Conversion rate with realistic e-commerce distribution
    data["conversion_rate"] = np.random.beta(1.2, 15, num_samples).clip(0.01, 0.3)
    
    # Ad conversions
    data["ad_conversions"] = (data["paid_clicks"] * data["conversion_rate"]).astype(int)
    
    # Conversion value with variance
    base_value = np.random.lognormal(4, 1, num_samples)
    data["conversion_value"] = data["ad_conversions"] * base_value
    
    # Cost per acquisition
    with np.errstate(divide='ignore', invalid='ignore'):
        data["cost_per_acquisition"] = np.where(
            data["ad_conversions"] > 0, 
            data["ad_spend"] / data["ad_conversions"], 
            500  # Default high CPA for no conversions
        ).clip(5, 500)
    
    # ROAS (Return on Ad Spend)
    with np.errstate(divide='ignore', invalid='ignore'):
        data["ad_roas"] = np.where(
            data["ad_spend"] > 0,
            data["conversion_value"] / data["ad_spend"],
            0
        ).clip(0.5, 5)
    
    # Impression share (competitive keywords have lower share)
    data["impression_share"] = (1 - 0.6 * data["competitiveness"] * np.random.normal(1, 0.2, num_samples)).clip(0.1, 1.0)
    
    return pd.DataFrame(data)

def split_dataset_by_ratio(dataset, train_ratio=0.8):
    """
    Splits the dataset into training and test sets.
    
    Args:
        dataset (pd.DataFrame): The dataset to split.
        train_ratio (float): Ratio of keywords to include in the training set (0.0-1.0).
        
    Returns:
        tuple: (training_dataset, test_dataset)
    """
    # Get all unique keywords
    keywords = dataset['keyword'].unique()
    
    # Calculate number of keywords for training and test sets
    num_keywords = len(keywords)
    num_train_keywords = int(num_keywords * train_ratio)
    
    # Randomly select keywords for training
    train_keywords = np.random.choice(keywords, size=num_train_keywords, replace=False)
    
    # Create training and test datasets
    train_dataset = dataset[dataset['keyword'].isin(train_keywords)].reset_index(drop=True)
    test_dataset = dataset[~dataset['keyword'].isin(train_keywords)].reset_index(drop=True)
    
    print(f"Training dataset: {len(train_dataset)} rows, {len(train_dataset['keyword'].unique())} keywords")
    print(f"Test dataset: {len(test_dataset)} rows, {len(test_dataset['keyword'].unique())} keywords")
    
    return train_dataset, test_dataset

def get_entry_from_dataset(df, index, unique_keywords_cache=None):
    """
    Retrieves a subset of rows from the DataFrame based on unique keywords and index.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the dataset.
        index (int): The index to determine which subset of rows to retrieve.
        unique_keywords_cache (dict, optional): Cache for unique keywords information.
        
    Returns:
        pandas.DataFrame: A subset of the DataFrame for the given index.
    """
    if unique_keywords_cache is None:
        # Count unique keywords
        seen_keywords = set()
        for i, row in df.iterrows():
            keyword = row['keyword']
            if keyword in seen_keywords:
                break
            seen_keywords.add(keyword)
        keywords_amount = len(seen_keywords)
        unique_keywords_cache = {
            'unique_keywords': seen_keywords,
            'keywords_amount': keywords_amount
        }
    else:
        keywords_amount = unique_keywords_cache['keywords_amount']
    
    # Get the subset of rows based on the index
    start_idx = index * keywords_amount
    end_idx = start_idx + keywords_amount
    
    if start_idx >= len(df) or end_idx > len(df):
        # Handle the case where index is out of bounds
        return df.sample(keywords_amount).reset_index(drop=True)
    
    return df.iloc[start_idx:end_idx].reset_index(drop=True)

# Define feature columns to use in the environment
feature_columns = [
    "competitiveness", 
    "difficulty_score", 
    "organic_rank", 
    "organic_ctr", 
    "paid_ctr", 
    "ad_spend", 
    "ad_conversions", 
    "ad_roas", 
    "conversion_rate", 
    "cost_per_click",
    "impression_share",
    "conversion_value"
]

class FlattenInputs(nn.Module):
    """
    A custom PyTorch module to flatten and combine keyword features, cash, and holdings into a single tensor.
    """
    def forward(self, keyword_features, cash, holdings):
        # Check if we have a batch dimension
        has_batch = keyword_features.dim() > 2
        
        if has_batch:
            batch_size = keyword_features.shape[0]
            # Flatten keyword features while preserving batch dimension: 
            # [batch, num_keywords, feature_dim] -> [batch, num_keywords * feature_dim]
            flattened_features = keyword_features.reshape(batch_size, -1)
            
            # Ensure cash has correct dimensions [batch, 1]
            if cash.dim() == 1:  # [batch]
                cash = cash.unsqueeze(-1)  # [batch, 1]
            elif cash.dim() == 0:  # scalar
                cash = cash.unsqueeze(0).expand(batch_size, 1)  # [batch, 1]
            
            # Ensure holdings has correct dimensions [batch, num_keywords]
            if holdings.dim() == 1:  # [num_keywords]
                holdings = holdings.unsqueeze(0).expand(batch_size, -1)  # [batch, num_keywords]
            
            # Convert holdings to float
            holdings = holdings.float()
            
            # Combine all inputs along dimension 1
            combined = torch.cat([flattened_features, cash, holdings], dim=1)
        else:
            # No batch dimension - single sample case
            # Flatten keyword features: [num_keywords, feature_dim] -> [num_keywords * feature_dim]
            flattened_features = keyword_features.reshape(-1)
            
            # Ensure cash has a dimension
            cash = cash.unsqueeze(-1) if cash.dim() == 0 else cash
            
            # Convert holdings to float
            holdings = holdings.float()
            
            # Combine all inputs
            combined = torch.cat([flattened_features, cash, holdings], dim=0)
            
        return combined

class AdOptimizationEnv(EnvBase):
    """
    Environment for digital advertising optimization using reinforcement learning.
    
    This environment models ad spending decisions across multiple keywords,
    with a focus on maximizing return on ad spend (ROAS) while managing a budget.
    """
    def __init__(self, dataset, initial_cash=100000.0, device="cpu"):
        """
        Initialize the AdOptimizationEnv with the given dataset.
        
        Args:
            dataset (pd.DataFrame): Dataset containing keyword metrics.
            initial_cash (float): Initial cash balance for budget.
            device (str): Device to run the environment on ('cpu' or 'cuda').
        """
        super().__init__(device=device)
        self.initial_cash = initial_cash
        self.dataset = dataset
        self.num_features = len(feature_columns)
        self.num_keywords = len(dataset['keyword'].unique())
        
        # Create cache for unique keywords
        seen_keywords = set()
        for i, row in dataset.iterrows():
            keyword = row['keyword']
            if keyword in seen_keywords:
                break
            seen_keywords.add(keyword)
        self.unique_keywords_cache = {
            'unique_keywords': seen_keywords,
            'keywords_amount': len(seen_keywords)
        }
        
        # Define action space: for each keyword, choose to invest (1) or not (0)
        self.action_spec = OneHot(n=self.num_keywords + 1)  # +1 for "buy nothing" action
        
        # Define reward space
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        
        # Define observation space
        self.observation_spec = Composite(
            observation=Composite(
                keyword_features=Unbounded(shape=(self.num_keywords, self.num_features), dtype=torch.float32),
                cash=Unbounded(shape=(1,), dtype=torch.float32),
                holdings=Bounded(low=0, high=1, shape=(self.num_keywords,), dtype=torch.int, domain="discrete")
            ),
            step_count=Unbounded(shape=(1,), dtype=torch.int64)
        )
        
        # Define done state
        self.done_spec = Composite(
            done=Binary(shape=(1,), dtype=torch.bool),
            terminated=Binary(shape=(1,), dtype=torch.bool),
            truncated=Binary(shape=(1,), dtype=torch.bool)
        )
        
        self.reset()

    def _reset(self, tensordict=None):
        """
        Reset the environment to its initial state.
        
        Args:
            tensordict (TensorDict, optional): A TensorDict to be updated with the reset state.
            
        Returns:
            TensorDict: A TensorDict containing the reset state of the environment.
        """
        self.current_step = 0
        self.holdings = torch.zeros(self.num_keywords, dtype=torch.int, device=self.device)
        self.cash = self.initial_cash
        
        # Get the current keyword features
        keyword_features = torch.tensor(
            get_entry_from_dataset(self.dataset, self.current_step, self.unique_keywords_cache)[feature_columns].values, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Create observation
        obs = TensorDict({
            "keyword_features": keyword_features,
            "cash": torch.tensor([self.cash], dtype=torch.float32, device=self.device),
            "holdings": self.holdings.clone()
        }, batch_size=[])
        
        # Create and return TensorDict
        if tensordict is None:
            tensordict = TensorDict({
                "done": torch.tensor(False, dtype=torch.bool, device=self.device),
                "observation": obs,
                "step_count": torch.tensor(self.current_step, dtype=torch.int64, device=self.device),
                "terminated": torch.tensor(False, dtype=torch.bool, device=self.device),
                "truncated": torch.tensor(False, dtype=torch.bool, device=self.device)
            }, batch_size=[])
        else:
            tensordict["done"] = torch.tensor(False, dtype=torch.bool, device=self.device)
            tensordict["observation"] = obs
            tensordict["step_count"] = torch.tensor(self.current_step, dtype=torch.int64, device=self.device)
            tensordict["terminated"] = torch.tensor(False, dtype=torch.bool, device=self.device)
            tensordict["truncated"] = torch.tensor(False, dtype=torch.bool, device=self.device)
        
        self.obs = obs
        return tensordict

    def _step(self, tensordict):
        """
        Take a step in the environment based on the action in tensordict.
        
        Args:
            tensordict (TensorDict): A TensorDict containing the action to take.
            
        Returns:
            TensorDict: A TensorDict containing the next state, reward, and done information.
        """
        # Get the action from tensordict
        action = tensordict["action"]
        true_indices = torch.nonzero(action, as_tuple=True)[0]
        action_idx = true_indices[0].item() if len(true_indices) > 0 else self.action_spec.n - 1
        
        # Get current keyword metrics
        current_pki = get_entry_from_dataset(self.dataset, self.current_step, self.unique_keywords_cache)
        
        # Update holdings based on action (only one keyword is selected)
        new_holdings = torch.zeros_like(self.holdings)
        if action_idx < self.num_keywords:
            new_holdings[action_idx] = 1
        self.holdings = new_holdings
        
        # Calculate reward
        reward = self._compute_reward(action, current_pki, action_idx)
        
        # Move to next step
        self.current_step += 1
        max_steps = len(self.dataset) // self.unique_keywords_cache['keywords_amount'] - 2  # -2 to avoid going over the last index
        terminated = self.current_step >= max_steps
        truncated = False
        
        # Get next keyword features
        next_keyword_features = torch.tensor(
            get_entry_from_dataset(self.dataset, self.current_step, self.unique_keywords_cache)[feature_columns].values, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Create next observation
        next_obs = TensorDict({
            "keyword_features": next_keyword_features,
            "cash": torch.tensor([self.cash], dtype=torch.float32, device=self.device),
            "holdings": self.holdings.clone()
        }, batch_size=[])
        
        # Update current observation
        self.obs = next_obs
        
        # Create result TensorDict
        result = TensorDict({
            "done": torch.tensor(terminated or truncated, dtype=torch.bool, device=self.device),
            "observation": self.obs,
            "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
            "step_count": torch.tensor(self.current_step-1, dtype=torch.int64, device=self.device),
            "terminated": torch.tensor(terminated, dtype=torch.bool, device=self.device),
            "truncated": torch.tensor(truncated, dtype=torch.bool, device=self.device),
            "next": {
                "done": torch.tensor(terminated or truncated, dtype=torch.bool, device=self.device),
                "observation": next_obs,
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "step_count": torch.tensor(self.current_step, dtype=torch.int64, device=self.device),
                "terminated": torch.tensor(terminated, dtype=torch.bool, device=self.device),
                "truncated": torch.tensor(truncated, dtype=torch.bool, device=self.device)
            }
        }, batch_size=tensordict.batch_size)
        
        return result

    def _compute_reward(self, action, current_pki, action_idx):
        """
        Compute the reward for the current action.
        
        This uses an advanced reward function that considers:
        - Return on Ad Spend (ROAS)
        - Click-Through Rate (CTR)
        - Ad Spend levels
        
        Args:
            action (torch.Tensor): The action tensor.
            current_pki (pd.DataFrame): Current keyword performance indicators.
            action_idx (int): Index of the selected action.
            
        Returns:
            float: The computed reward.
        """
        # If the "buy nothing" action was selected
        if action_idx == self.num_keywords:
            return 0.0
        
        # Initialize reward
        reward = 0.0
        
        # Get the selected keyword's metrics
        sample = current_pki.iloc[action_idx]
        cost = sample["ad_spend"]
        ctr = sample["paid_ctr"]
        roas = sample["ad_roas"]
        
        # Calculate reward based on a more sophisticated strategy
        if cost > 5000 and roas > 2.0:
            # High-spend, high-ROAS strategy (most rewarding)
            reward = 2.0
        elif roas > 1.0:
            # Profitable strategy (good return)
            reward = 1.0
        elif ctr > 0.15:
            # High CTR but not profitable (might build visibility)
            reward = 0.5
        else:
            # Poor performance (negative reward)
            reward = -1.0
            
        return reward

    def _set_seed(self, seed: Optional[int]):
        """Set the random seed for the environment."""
        self.rng = torch.manual_seed(seed)

# Create Q-network using TorchRL's MLP module
def create_q_network(input_size, output_size, hidden_layers=[128, 64]):
    """
    Create a Q-network with TorchRL's MLP module.
    
    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        hidden_layers (list): List of hidden layer sizes.
        
    Returns:
        MLP: The Q-network.
    """
    q_network = MLP(
        in_features=input_size,
        out_features=output_size,
        num_cells=hidden_layers,
        activation_class=nn.ReLU
    )
    return q_network

def visualize_training_progress(metrics, output_dir="plots", window_size=20):
    """
    Visualize training metrics including rewards, losses, and exploration rate.
    
    Args:
        metrics (dict): Dictionary containing training metrics.
        output_dir (str): Directory to save the plots.
        window_size (int): Window size for moving average.
        
    Returns:
        str: Path to the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    rewards = metrics["rewards"]
    losses = metrics["losses"]
    epsilons = metrics.get("epsilon_values", [])
    
    # Ensure tensors are converted to CPU NumPy arrays
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.cpu().numpy()
    if isinstance(losses, torch.Tensor):
        losses = losses.cpu().numpy()
    if isinstance(epsilons, torch.Tensor):
        epsilons = epsilons.cpu().numpy()
    
    if len(rewards) == 0:
        print("No rewards to visualize")
        return None
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("RL Training Progress", fontsize=16)
    
    # Plot rewards
    axes[0].plot(rewards, alpha=0.3, color='blue', label="Episode Rewards")
    
    if len(rewards) >= window_size:
        # Add smoothed rewards line
        smoothed_rewards = []
        for i in range(len(rewards) - window_size + 1):
            smoothed_rewards.append(np.mean(rewards[i:i+window_size]))
        axes[0].plot(range(window_size-1, len(rewards)), smoothed_rewards, 
                   color='red', linewidth=2, label=f"Moving Average ({window_size})")
    
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Training Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot losses
    if len(losses) > 0:
        axes[1].plot(losses, color='purple', alpha=0.5, label="Training Loss")
        
        if len(losses) >= window_size:
            # Add smoothed losses line
            smoothed_losses = []
            for i in range(len(losses) - window_size + 1):
                smoothed_losses.append(np.mean(losses[i:i+window_size]))
            axes[1].plot(range(window_size-1, len(losses)), smoothed_losses, 
                       color='darkred', linewidth=2, label=f"Moving Average ({window_size})")
        
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot exploration rate
    if len(epsilons) > 0:
        axes[2].plot(epsilons, color='green', label="Exploration Rate (ε)")
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel("Episodes")
        axes[2].set_ylabel("Epsilon (ε)")
        axes[2].set_title("Exploration Rate")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = f"{output_dir}/training_progress.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return plot_path

def train_agent(env, policy, total_episodes=200, batch_size=64, lr=0.001, gamma=0.99, target_update_freq=10):
    """
    Train a policy using TorchRL components.
    
    Args:
        env: The environment to train on
        policy: The policy module to train
        total_episodes: Total number of episodes to train for
        batch_size: Batch size for training
        lr: Learning rate
        gamma: Discount factor
        target_update_freq: Frequency of target network updates
        
    Returns:
        dict: Training metrics
    """
    # Create target network
    target_policy = copy.deepcopy(policy)
    target_policy.requires_grad_(False)
    
    # Create epsilon-greedy exploration module
    exploration_module = EGreedyModule(
        env.action_spec, 
        annealing_num_steps=total_episodes * 50, 
        eps_init=1.0,
        eps_end=0.05
    )
    exploration_module = exploration_module.to(device)
    
    # Create policy with exploration
    policy_explore = TensorDictSequential(policy, exploration_module).to(device)
    
    # Set up DQN loss module
    loss_module = DQNLoss(
        value_network=policy,
        action_space=env.action_spec
    ).to(device)

    # Set target network if available
    if hasattr(loss_module, "target_network"):
        loss_module.target_network = target_policy
    
    # Set up optimizer
    optimizer = optim.Adam(loss_module.parameters(), lr=lr)
    
    # Set up updater for target network
    updater = SoftUpdate(policy, target_policy, eps=0.01)
    
    # Create replay buffer
    buffer_capacity = 10000
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(buffer_capacity),
        sampler=None,
        batch_size=batch_size
    )
    
    # Create data collector
    frames_per_batch = 1  # Collect one step at a time
    init_random_frames = min(1000, total_episodes * 5)  # Initial random frames for exploration
    collector = SyncDataCollector(
        env,
        policy_explore,
        frames_per_batch=frames_per_batch,
        total_frames=-1,  # Collect data until manually stopped
        init_random_frames=init_random_frames
    )
    
    # Metrics for tracking
    rewards_history = []
    losses = []
    epsilon_values = []
    episodes_completed = 0
    current_episode_reward = 0
    
    # Training loop
    print("Starting training loop...")
    start_time = time.time()
    total_frames = 0
    
    for i, data in enumerate(collector):
        # Update total frames
        n_frames = data.numel()
        total_frames += n_frames
        
        # Move data to correct device
        data_device = data.to(device)
        
        # Track episodes and rewards
        for j in range(data_device.batch_size[0]):
            current_episode_reward += data_device[j]["reward"].item()
            if data_device[j]["done"].item():
                rewards_history.append(current_episode_reward)
                current_episode_reward = 0
                episodes_completed += 1
                epsilon_values.append(exploration_module.eps)
                print(f"Episode {episodes_completed} completed with reward {rewards_history[-1]:.2f}, ε={exploration_module.eps:.3f}")
        
        # Add data to replay buffer
        replay_buffer.extend(data_device)
        
        # Train if enough data
        if len(replay_buffer) >= batch_size:
            # Multiple optimization steps per data collection for efficiency
            for _ in range(4):
                # Sample from replay buffer
                sample = replay_buffer.sample()
                # Move sample to device if needed
                if sample.device != device:
                    sample = sample.to(device)
                
                # Compute loss and update policy
                loss_dict = loss_module(sample)
                loss = loss_dict["loss"]
                losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update exploration factor
                exploration_module.step(1)
                
                # Update target network periodically
                if (len(losses) % target_update_freq) == 0:
                    updater.step()
        
        # Log progress regularly
        if (i + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history) if rewards_history else 0
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses) if losses else 0
            
            print(f"Frame {total_frames}, Episodes {episodes_completed}/{total_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, Epsilon: {exploration_module.eps:.3f}, "
                  f"Avg Loss: {avg_loss:.4f}")
        
        # Stop if we've completed enough episodes
        if episodes_completed >= total_episodes:
            break
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Clean up
    collector.shutdown()
    
    return {
        "rewards": rewards_history,
        "losses": losses,
        "epsilon_values": epsilon_values,
        "episodes": episodes_completed,
        "training_time": total_time
    }


def evaluate_agent(env, policy, num_episodes=10):
    """
    Evaluate a trained policy without exploration.
    
    Args:
        env: The environment to evaluate on
        policy: The policy to evaluate
        num_episodes: Number of episodes to evaluate
        
    Returns:
        dict: Evaluation metrics
    """
    rewards = []
    actions = []
    states = []
    episode_rewards = []
    decisions = []  # Store (action, reward) pairs
    conservative_rewards = []
    aggressive_rewards = []
    
    for _ in range(num_episodes):
        episode_reward = 0
        td = env.reset()
        done = False
        
        while not done:
            # Store current state
            states.append(td["observation"]["keyword_features"].cpu().numpy()[0])
            
            # Select action deterministically without exploration
            with torch.no_grad():
                # Forward through policy
                td_action = policy(td)
                action = td_action["action"]
                action_idx = torch.argmax(action).item()
                
                # Record action
                actions.append(action_idx)
                
                # Step environment
                td_next = env.step(td_action)
                
                # Get reward
                reward = td_next["reward"].item()
                rewards.append(reward)
                episode_reward += reward
                
                # Store decision
                decisions.append((action_idx, reward))
                
                # Store action-specific rewards
                if action_idx < env.num_keywords:  # Not the "buy nothing" action
                    if action_idx % 2 == 0:  # Even indices for conservative actions
                        conservative_rewards.append(reward)
                    else:  # Odd indices for aggressive actions
                        aggressive_rewards.append(reward)
                
                # Check done state
                done = td_next["done"].item()
                
                # Update state
                td = td_next
        
        episode_rewards.append(episode_reward)
    
    # Calculate action distribution
    action_counts = {}
    for a in actions:
        if a == env.num_keywords:  # "Buy nothing" action
            action_type = "no_action"
        else:
            action_type = a % 2  # 0 for conservative, 1 for aggressive
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
    
    total_actions = len(actions)
    action_distribution = {a: count / total_actions for a, count in action_counts.items()}
    
    # Calculate average rewards by action type
    avg_conservative_reward = np.mean(conservative_rewards) if conservative_rewards else 0
    avg_aggressive_reward = np.mean(aggressive_rewards) if aggressive_rewards else 0
    
    # Calculate success rate (positive rewards)
    success_rate = np.mean([r > 0 for r in rewards]) if rewards else 0
    
    return {
        "avg_reward": np.mean(episode_rewards),
        "total_reward": sum(episode_rewards),
        "rewards": rewards,
        "states": np.array(states),
        "action_distribution": action_distribution,
        "avg_conservative_reward": avg_conservative_reward,
        "avg_aggressive_reward": avg_aggressive_reward,
        "decisions": decisions,
        "success_rate": success_rate
    }


def main():
    """Main function to run the training and evaluation pipeline."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"ad_optimization_results_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    plot_dir = f"{run_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Starting digital advertising optimization pipeline...")
    print(f"Results will be saved to: {run_dir}")
    
    # Set random seeds
    set_all_seeds(42)
    
    # Generate dataset
    print("Generating synthetic dataset...")
    dataset = generate_synthetic_data(1000)
    dataset_path = f"{run_dir}/synthetic_ad_data.csv"
    dataset.to_csv(dataset_path, index=False)
    print(f"Synthetic dataset saved to {dataset_path}")
    
    # Print dataset summary
    print("\nDataset summary:")
    print(f"Shape: {dataset.shape}")
    print("\nFeature stats:")
    print(dataset[feature_columns].describe().to_string())
    
    # Split into training and test datasets
    train_dataset, test_dataset = split_dataset_by_ratio(dataset, train_ratio=0.8)
    
    # Create environment
    print("\nCreating environment...")
    env = AdOptimizationEnv(train_dataset, device=device)
    
    # Create policy network
    print("\nCreating policy network...")
    # Calculate input size based on environment dimensions
    # For a policy network that takes flattened inputs including keyword features, cash, and holdings
    input_size = env.num_keywords * env.num_features + 1 + env.num_keywords  # features + cash + holdings
    output_size = env.action_spec.n  # Number of actions (num_keywords + 1)
    
    # Create neural network architecture
    flatten_module = TensorDictModule(
        FlattenInputs(),
        in_keys=[("observation", "keyword_features"), ("observation", "cash"), ("observation", "holdings")],
        out_keys=["flattened_input"]
    )
    
    value_mlp = MLP(in_features=input_size, out_features=output_size, num_cells=[128, 64])
    value_net = TensorDictModule(value_mlp, in_keys=["flattened_input"], out_keys=["action_value"])
    policy = TensorDictSequential(flatten_module, value_net, QValueModule(spec=env.action_spec))
    
    # Move policy to device
    policy = policy.to(device)
    
    # Train agent
    print("\nTraining agent...")
    total_episodes = 300  # Adjust based on your needs
    print(f"Training for {total_episodes} episodes...")
    
    training_metrics = train_agent(
        env=env,
        policy=policy,
        total_episodes=total_episodes,
        batch_size=64,
        lr=0.001,
        gamma=0.99,
        target_update_freq=10
    )
    
    # Generate training visualization
    print("Generating training visualization...")
    training_plot_path = visualize_training_progress(training_metrics, output_dir=plot_dir)
    print(f"Training plot saved to {training_plot_path}")
    
    # Create test environment
    test_env = AdOptimizationEnv(test_dataset, device=device)
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    eval_results = evaluate_agent(test_env, policy, num_episodes=20)
    
    # Display evaluation results
    print("\nEvaluation Results:")
    print(f"Average Reward: {eval_results['avg_reward']:.2f}")
    print(f"Success Rate: {eval_results['success_rate']:.2f}")
    
    # Print action distribution
    action_distribution = eval_results["action_distribution"]
    
    print("\nAction Distribution:")
    for action, freq in action_distribution.items():
        if action == "no_action":
            action_name = "No Action"
        else:
            action_name = "Conservative" if action == 0 else "Aggressive"
        print(f"  {action_name}: {freq:.2f} ({100 * freq:.1f}%)")
    
    # Save evaluation metrics
    eval_metrics_path = f"{run_dir}/evaluation_metrics.txt"
    with open(eval_metrics_path, "w") as f:
        f.write(f"Average Reward: {eval_results['avg_reward']:.4f}\n")
        f.write(f"Success Rate: {eval_results['success_rate']:.4f}\n")
        f.write(f"Total Reward: {eval_results['total_reward']:.4f}\n")
        f.write("\nAction Distribution:\n")
        for action, freq in action_distribution.items():
            if action == "no_action":
                action_name = "No Action"
            else:
                action_name = "Conservative" if action == 0 else "Aggressive"
            f.write(f"  {action_name}: {freq:.2f} ({100 * freq:.1f}%)\n")
        
        f.write(f"\nAverage Conservative Reward: {eval_results['avg_conservative_reward']:.4f}\n")
        f.write(f"Average Aggressive Reward: {eval_results['avg_aggressive_reward']:.4f}\n")
    
    # Visualize evaluation
    print("Generating evaluation visualization...")
    eval_plot_path = visualize_evaluation(eval_results, feature_columns, output_dir=plot_dir)
    print(f"Evaluation plot saved to {eval_plot_path}")
    
    # Save model
    model_path = f"{run_dir}/ad_optimization_model.pt"
    torch.save({
        'model_state_dict': policy.state_dict(),
        'feature_columns': feature_columns,
        'training_metrics': training_metrics
    }, model_path)
    print(f"Model saved to {model_path}")
    
    print(f"\nPipeline completed successfully. All results saved to {run_dir}")
    
    return policy, training_metrics, eval_results


if __name__ == "__main__":
    main()

def visualize_evaluation(eval_results, feature_columns, output_dir="plots"):
    """
    Create visualizations for evaluation metrics.
    
    Args:
        eval_results (dict): Dictionary containing evaluation results.
        feature_columns (list): List of feature column names.
        output_dir (str): Directory to save the plots.
        
    Returns:
        str: Path to the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Ad Optimization RL Agent Evaluation", fontsize=16)
    
    # 1. Action Distribution
    ax1 = fig.add_subplot(2, 3, 1)
    actions = ["Conservative", "Aggressive"]
    action_counts = [eval_results["action_distribution"].get(0, 0), eval_results["action_distribution"].get(1, 0)]
    ax1.bar(actions, action_counts, color=["skyblue", "coral"])
    ax1.set_title("Action Distribution")
    ax1.set_ylabel("Frequency")
    
    # 2. Average Reward by Action Type
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(["Conservative", "Aggressive"], 
            [eval_results["avg_conservative_reward"], eval_results["avg_aggressive_reward"]], 
            color=["skyblue", "coral"])
    ax2.set_title("Average Reward by Action Type")
    ax2.set_ylabel("Average Reward")
    
    # 3. Feature Correlations with Decisions
    ax3 = fig.add_subplot(2, 3, 3)
    states = eval_results.get("states", np.array([]))
    
    if "decisions" in eval_results and eval_results["decisions"]:
        decisions = np.array([a for a, _ in eval_results["decisions"]])
        
        correlations = []
        feature_names = []
        
        if states.size > 0 and decisions.size > 0 and states.shape[1] == len(feature_columns):
            for i, feature in enumerate(feature_columns):
                try:
                    corr = np.corrcoef(states[:, i], decisions)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                        feature_names.append(feature)
                except:
                    pass
        
        if correlations:
            sorted_indices = np.argsort(np.abs(correlations))[::-1][:5]  # Top 5 features
            top_features = [feature_names[i] for i in sorted_indices]
            top_correlations = [correlations[i] for i in sorted_indices]
            
            ax3.barh(top_features, top_correlations, color='teal')
            ax3.set_title("Top Feature Correlations with Actions")
            ax3.set_xlabel("Correlation Coefficient")
        else:
            ax3.text(0.5, 0.5, "Insufficient data for correlation analysis", 
                    ha='center', va='center')
    else:
        ax3.text(0.5, 0.5, "No decision data available", 
                ha='center', va='center')
    
    # 4. Reward Distribution
    ax4 = fig.add_subplot(2, 3, 4)
    if "rewards" in eval_results and eval_results["rewards"]:
        sns.histplot(eval_results["rewards"], kde=True, ax=ax4)
        ax4.set_title("Reward Distribution")
        ax4.set_xlabel("Reward")
        ax4.set_ylabel("Frequency")
    else:
        ax4.text(0.5, 0.5, "No reward data available", ha='center', va='center')
    
    # 5. Decision Quality Matrix
    ax5 = fig.add_subplot(2, 3, 5)
    decision_quality = np.zeros((2, 2))
    
    if "decisions" in eval_results and eval_results["decisions"]:
        for action, reward in eval_results["decisions"]:
            quality = 1 if reward > 0 else 0
            if action < 2:  # Ensure action is either 0 or 1
                decision_quality[action, quality] += 1
        
        row_sums = decision_quality.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        decision_quality_norm = decision_quality / row_sums
        
        sns.heatmap(decision_quality_norm, annot=True, fmt=".2f", cmap="YlGnBu",
                  xticklabels=["Poor", "Good"], 
                  yticklabels=["Conservative", "Aggressive"],
                  ax=ax5)
        ax5.set_title("Decision Quality Matrix")
        ax5.set_ylabel("Action")
        ax5.set_xlabel("Decision Quality")
    else:
        ax5.text(0.5, 0.5, "No decision data available", 
               ha='center', va='center')
    
    # 6. Success Rate or Pie Chart
    ax6 = fig.add_subplot(2, 3, 6)
    if "decisions" in eval_results and eval_results["decisions"]:
        # Calculate success rate
        success_rate = eval_results["success_rate"]
        
        # Create a pie chart of success rate
        ax6.pie([success_rate, 1-success_rate], 
               labels=["Success", "Failure"], 
               colors=["green", "lightgray"],
               autopct='%1.1f%%', 
               startangle=90)
        ax6.set_title("Decision Success Rate")
    else:
        ax6.text(0.5, 0.5, "Success rate data not available", 
               ha='center', va='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = f"{output_dir}/agent_evaluation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return plot_path