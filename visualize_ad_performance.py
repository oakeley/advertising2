#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime
from tensordict import TensorDict

# Import classes from integrated_ad_optimization.py
# Make sure this script is in the same directory as integrated_ad_optimization.py or modify the import path
from integrated_ad_optimization import (
    FlattenInputs, AdOptimizationEnv, feature_columns, TensorDictModule, TensorDictSequential,
    MLP, QValueModule, generate_synthetic_data, set_all_seeds
)

def load_model(model_path, env):
    """
    Load a saved model and recreate the policy network.
    
    Args:
        model_path (str): Path to the saved model.
        env (AdOptimizationEnv): Environment to use for determining model dimensions.
        
    Returns:
        TensorDictSequential: The loaded policy network.
    """
    # Load the saved model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Calculate input size based on environment dimensions
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
    
    # Load the saved weights
    policy.load_state_dict(checkpoint['model_state_dict'])
    
    return policy

def visualize_keyword_decision_map(dataset, policy, env, output_dir="keyword_analysis", num_keywords=10):
    """
    Create a visualization showing how the agent decides on keyword investments
    across different keyword metrics.
    
    Args:
        dataset (pd.DataFrame): Dataset containing keyword metrics.
        policy (TensorDictSequential): Trained policy network.
        env (AdOptimizationEnv): Environment for ad optimization.
        output_dir (str): Directory to save the visualizations.
        num_keywords (int): Number of keywords to analyze.
        
    Returns:
        str: Path to the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the model to evaluation mode
    policy.eval()
    
    # Select a subset of keywords to analyze
    unique_keywords = dataset['keyword'].unique()[:num_keywords]
    keyword_data = dataset[dataset['keyword'].isin(unique_keywords)]
    
    # Data structures to store results
    decisions = []
    metrics = []
    keyword_names = []
    
    # Analyze each keyword
    for keyword in unique_keywords:
        print(f"Analyzing keyword: {keyword}")
        keyword_subset = keyword_data[keyword_data['keyword'] == keyword]
        
        # Initialize the environment with this keyword
        env.reset()
        
        # For each instance of the keyword, predict the action
        for _, row in keyword_subset.iterrows():
            # Create a feature tensor
            feature_tensor = torch.tensor(
                row[feature_columns].values, 
                dtype=torch.float32
            ).unsqueeze(0)  # Add batch dimension
            
            # Create a TensorDict for the observation
            obs = TensorDict({
                "keyword_features": feature_tensor.unsqueeze(0),  # [batch, 1, features]
                "cash": torch.tensor([env.initial_cash], dtype=torch.float32),
                "holdings": torch.zeros(1, dtype=torch.int)
            }, batch_size=[])
            
            td = TensorDict({
                "observation": obs,
                "done": torch.tensor(False, dtype=torch.bool),
                "step_count": torch.tensor(0, dtype=torch.int64)
            }, batch_size=[])
            
            # Get the policy's decision
            with torch.no_grad():
                td_action = policy(td)
                action = td_action["action"]
                action_idx = torch.argmax(action).item()
            
            # Store the decision and metrics
            decision = "Invest" if action_idx < env.num_keywords else "Don't Invest"
            decisions.append(decision)
            metrics.append({
                "ROAS": row["ad_roas"],
                "CTR": row["paid_ctr"],
                "Ad Spend": row["ad_spend"],
                "Competitiveness": row["competitiveness"],
                "Conversion Rate": row["conversion_rate"]
            })
            keyword_names.append(keyword)
    
    # Create a DataFrame for visualization
    result_df = pd.DataFrame({
        "Keyword": keyword_names,
        "Decision": decisions,
        "ROAS": [m["ROAS"] for m in metrics],
        "CTR": [m["CTR"] for m in metrics],
        "Ad Spend": [m["Ad Spend"] for m in metrics],
        "Competitiveness": [m["Competitiveness"] for m in metrics],
        "Conversion Rate": [m["Conversion Rate"] for m in metrics]
    })
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Keyword Investment Decision Analysis", fontsize=16)
    
    # 1. ROAS vs Ad Spend colored by decision
    ax1 = axes[0, 0]
    sns.scatterplot(
        data=result_df, 
        x="Ad Spend", 
        y="ROAS", 
        hue="Decision", 
        style="Decision",
        palette={"Invest": "green", "Don't Invest": "red"},
        alpha=0.7,
        s=100,
        ax=ax1
    )
    ax1.set_title("ROAS vs Ad Spend")
    ax1.grid(True, alpha=0.3)
    
    # 2. CTR vs Competitiveness colored by decision
    ax2 = axes[0, 1]
    sns.scatterplot(
        data=result_df, 
        x="Competitiveness", 
        y="CTR", 
        hue="Decision", 
        style="Decision",
        palette={"Invest": "green", "Don't Invest": "red"},
        alpha=0.7,
        s=100,
        ax=ax2
    )
    ax2.set_title("CTR vs Competitiveness")
    ax2.grid(True, alpha=0.3)
    
    # 3. Conversion Rate vs Ad Spend colored by decision
    ax3 = axes[1, 0]
    sns.scatterplot(
        data=result_df, 
        x="Ad Spend", 
        y="Conversion Rate", 
        hue="Decision", 
        style="Decision",
        palette={"Invest": "green", "Don't Invest": "red"},
        alpha=0.7,
        s=100,
        ax=ax3
    )
    ax3.set_title("Conversion Rate vs Ad Spend")
    ax3.grid(True, alpha=0.3)
    
    # 4. Decision Distribution by Keyword
    ax4 = axes[1, 1]
    keyword_decision_counts = result_df.groupby(["Keyword", "Decision"]).size().unstack(fill_value=0)
    keyword_decision_counts.plot(
        kind="bar", 
        stacked=True, 
        ax=ax4, 
        color=["red", "green"]
    )
    ax4.set_title("Investment Decisions by Keyword")
    ax4.set_xlabel("Keyword")
    ax4.set_ylabel("Count")
    ax4.legend(title="Decision")
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = f"{output_dir}/keyword_decision_map.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Create an additional visualization showing the decision boundaries
    plt.figure(figsize=(12, 10))
    plt.suptitle("Decision Boundary Analysis: ROAS vs Ad Spend", fontsize=16)
    
    # Create a grid of points covering the space
    roas_range = np.linspace(result_df["ROAS"].min(), result_df["ROAS"].max(), 100)
    spend_range = np.linspace(result_df["Ad Spend"].min(), result_df["Ad Spend"].max(), 100)
    
    ROAS, SPEND = np.meshgrid(roas_range, spend_range)
    
    # Create a contour plot of the decision boundary
    # We'll use a simplified model based on our observations
    boundary_x = []
    boundary_y = []
    
    # Plot the actual data points
    plt.scatter(
        result_df[result_df["Decision"] == "Invest"]["Ad Spend"],
        result_df[result_df["Decision"] == "Invest"]["ROAS"],
        color="green",
        label="Invest",
        alpha=0.7,
        s=100,
        edgecolors='w'
    )
    plt.scatter(
        result_df[result_df["Decision"] == "Don't Invest"]["Ad Spend"],
        result_df[result_df["Decision"] == "Don't Invest"]["ROAS"],
        color="red",
        label="Don't Invest",
        alpha=0.7,
        s=100,
        edgecolors='w'
    )
    
    # Add decision regions (estimated)
    # This is a simplification of what the model might be doing
    # For a more accurate representation, we would need to run the model on the grid points
    
    # High ROAS Region (likely "Invest")
    high_roas_region_x = np.linspace(0, result_df["Ad Spend"].max(), 100)
    high_roas_region_y = 2.0 * np.ones_like(high_roas_region_x)  # ROAS threshold of 2.0
    plt.fill_between(
        high_roas_region_x, 
        high_roas_region_y, 
        np.max(result_df["ROAS"]) * np.ones_like(high_roas_region_y),
        alpha=0.1,
        color="green",
        label="Likely Invest Region"
    )
    
    # Mid-range Region (mixed decisions)
    mid_region_x = np.linspace(0, result_df["Ad Spend"].max(), 100)
    mid_region_y_upper = 2.0 * np.ones_like(mid_region_x)
    mid_region_y_lower = 1.0 * np.ones_like(mid_region_x)
    plt.fill_between(
        mid_region_x, 
        mid_region_y_lower, 
        mid_region_y_upper,
        alpha=0.1,
        color="yellow",
        label="Mixed Decision Region"
    )
    
    # Low ROAS Region (likely "Don't Invest")
    low_roas_region_x = np.linspace(0, result_df["Ad Spend"].max(), 100)
    low_roas_region_y = 1.0 * np.ones_like(low_roas_region_x)
    plt.fill_between(
        low_roas_region_x, 
        low_roas_region_y, 
        np.zeros_like(low_roas_region_y),
        alpha=0.1,
        color="red",
        label="Likely Don't Invest Region"
    )
    
    plt.xlabel("Ad Spend")
    plt.ylabel("ROAS (Return on Ad Spend)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    
    boundary_path = f"{output_dir}/decision_boundary.png"
    plt.savefig(boundary_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return plot_path

def visualize_reward_components(dataset, policy, env, output_dir="reward_analysis"):
    """
    Visualize the components that contribute to the reward function and how 
    they influence the agent's decision-making.
    
    Args:
        dataset (pd.DataFrame): Dataset containing keyword metrics.
        policy (TensorDictSequential): Trained policy network.
        env (AdOptimizationEnv): Environment for ad optimization.
        output_dir (str): Directory to save the visualizations.
    
    Returns:
        str: Path to the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the model to evaluation mode
    policy.eval()
    
    # Sample a number of states from the dataset
    sample_size = min(200, len(dataset))
    sampled_data = dataset.sample(sample_size)
    
    # Data structures to store results
    decisions = []
    rewards = []
    metrics = {
        "ROAS": [],
        "CTR": [],
        "Ad Spend": [],
        "Reward Component": []
    }
    
    # Analyze each sampled state
    for _, row in sampled_data.iterrows():
        # Create a feature tensor
        feature_tensor = torch.tensor(
            row[feature_columns].values, 
            dtype=torch.float32
        ).unsqueeze(0)  # Add batch dimension
        
        # Create a TensorDict for the observation
        obs = TensorDict({
            "keyword_features": feature_tensor.unsqueeze(0),  # [batch, 1, features]
            "cash": torch.tensor([env.initial_cash], dtype=torch.float32),
            "holdings": torch.zeros(1, dtype=torch.int)
        }, batch_size=[])
        
        td = TensorDict({
            "observation": obs,
            "done": torch.tensor(False, dtype=torch.bool),
            "step_count": torch.tensor(0, dtype=torch.int64)
        }, batch_size=[])
        
        # Get the policy's decision
        with torch.no_grad():
            td_action = policy(td)
            action = td_action["action"]
            action_idx = torch.argmax(action).item()
        
        # Store decision
        decision = "Invest" if action_idx < env.num_keywords else "Don't Invest"
        decisions.append(decision)
        
        # Extract metrics
        roas = row["ad_roas"]
        ctr = row["paid_ctr"]
        ad_spend = row["ad_spend"]
        
        # Compute reward components
        roas_component = 2.0 if roas > 2.0 else (1.0 if roas > 1.0 else 0.0)
        ctr_component = 0.5 if ctr > 0.15 else 0.0
        
        # Store reward components
        metrics["ROAS"].append(roas)
        metrics["CTR"].append(ctr)
        metrics["Ad Spend"].append(ad_spend)
        
        # Determine the primary reward component
        if roas > 2.0:
            metrics["Reward Component"].append("High ROAS")
        elif roas > 1.0:
            metrics["Reward Component"].append("Profitable")
        elif ctr > 0.15:
            metrics["Reward Component"].append("High CTR")
        else:
            metrics["Reward Component"].append("Low Performance")
    
    # Create a DataFrame for visualization
    result_df = pd.DataFrame({
        "Decision": decisions,
        "ROAS": metrics["ROAS"],
        "CTR": metrics["CTR"],
        "Ad Spend": metrics["Ad Spend"],
        "Reward Component": metrics["Reward Component"]
    })
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Reward Component Analysis", fontsize=16)
    
    # 1. ROAS Distribution by Decision
    ax1 = axes[0, 0]
    sns.violinplot(
        data=result_df, 
        x="Decision", 
        y="ROAS", 
        hue="Decision",
        palette={"Invest": "green", "Don't Invest": "red"},
        split=True,
        inner="quart",
        ax=ax1
    )
    ax1.set_title("ROAS Distribution by Decision")
    ax1.grid(True, alpha=0.3)
    
    # 2. CTR Distribution by Decision
    ax2 = axes[0, 1]
    sns.violinplot(
        data=result_df, 
        x="Decision", 
        y="CTR", 
        hue="Decision",
        palette={"Invest": "green", "Don't Invest": "red"},
        split=True,
        inner="quart",
        ax=ax2
    )
    ax2.set_title("CTR Distribution by Decision")
    ax2.grid(True, alpha=0.3)
    
    # 3. Decision Distribution by Reward Component
    ax3 = axes[1, 0]
    component_counts = result_df.groupby(["Reward Component", "Decision"]).size().unstack(fill_value=0)
    component_counts.plot(
        kind="bar", 
        stacked=True, 
        ax=ax3, 
        color=["red", "green"]
    )
    ax3.set_title("Decisions by Reward Component")
    ax3.set_xlabel("Reward Component")
    ax3.set_ylabel("Count")
    ax3.legend(title="Decision")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    
    # 4. Ad Spend vs ROAS colored by Reward Component
    ax4 = axes[1, 1]
    sns.scatterplot(
        data=result_df, 
        x="Ad Spend", 
        y="ROAS", 
        hue="Reward Component",
        style="Decision",
        palette={"High ROAS": "darkgreen", "Profitable": "limegreen", 
                "High CTR": "orange", "Low Performance": "crimson"},
        alpha=0.7,
        s=100,
        ax=ax4
    )
    ax4.set_title("Ad Spend vs ROAS by Reward Component")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = f"{output_dir}/reward_component_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return plot_path

def main():
    parser = argparse.ArgumentParser(description="Visualize Ad Performance with Trained RL Model")
    parser.add_argument("--model", type=str, default="ad_optimization_model.pt", help="Path to saved model")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset CSV (if None, generates synthetic data)")
    parser.add_argument("--output_dir", type=str, default="visualization_results", help="Output directory for visualizations")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_all_seeds(42)
    
    # Load or generate dataset
    if args.dataset and os.path.exists(args.dataset):
        print(f"Loading dataset from {args.dataset}")
        dataset = pd.read_csv(args.dataset)
    else:
        print("Generating synthetic dataset")
        dataset = generate_synthetic_data(1000)
        dataset_path = f"{args.output_dir}/synthetic_ad_data.csv"
        dataset.to_csv(dataset_path, index=False)
        print(f"Synthetic dataset saved to {dataset_path}")
    
    # Create environment
    env = AdOptimizationEnv(dataset)
    
    # Load model
    print(f"Loading model from {args.model}")
    policy = load_model(args.model, env)
    
    # Generate visualizations
    print("Generating keyword decision map...")
    keyword_map_path = visualize_keyword_decision_map(dataset, policy, env, 
                                                    output_dir=f"{args.output_dir}/keyword_analysis")
    print(f"Keyword decision map saved to {keyword_map_path}")
    
    print("Generating reward component analysis...")
    reward_analysis_path = visualize_reward_components(dataset, policy, env, 
                                                     output_dir=f"{args.output_dir}/reward_analysis")
    print(f"Reward component analysis saved to {reward_analysis_path}")
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()