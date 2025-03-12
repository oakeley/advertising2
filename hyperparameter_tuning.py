#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import optuna
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from copy import deepcopy

# Import functions and classes from the main script
from integrated_ad_optimization import (
    AdOptimizationEnv, feature_columns, FlattenInputs,
    TensorDictModule, TensorDictSequential, MLP, QValueModule,
    generate_synthetic_data, set_all_seeds, train_agent, evaluate_agent
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_policy_network(env, hidden_layers, activation_class=nn.ReLU):
    """
    Create a policy network with the given architecture.
    
    Args:
        env (AdOptimizationEnv): Environment for ad optimization.
        hidden_layers (list): List of hidden layer sizes.
        activation_class (torch.nn.Module): Activation function class.
        
    Returns:
        TensorDictSequential: The policy network.
    """
    # Calculate input size based on environment dimensions
    input_size = env.num_keywords * env.num_features + 1 + env.num_keywords  # features + cash + holdings
    output_size = env.action_spec.n  # Number of actions (num_keywords + 1)
    
    # Create neural network architecture
    flatten_module = TensorDictModule(
        FlattenInputs(),
        in_keys=[("observation", "keyword_features"), ("observation", "cash"), ("observation", "holdings")],
        out_keys=["flattened_input"]
    )
    
    value_mlp = MLP(
        in_features=input_size, 
        out_features=output_size, 
        num_cells=hidden_layers,
        activation_class=activation_class
    )
    value_net = TensorDictModule(value_mlp, in_keys=["flattened_input"], out_keys=["action_value"])
    policy = TensorDictSequential(flatten_module, value_net, QValueModule(spec=env.action_spec))
    
    # Move policy to device
    return policy.to(device)

def objective(trial, env_train, env_eval, max_episodes=100, fixed_params=None):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial (optuna.Trial): Optuna trial object.
        env_train (AdOptimizationEnv): Training environment.
        env_eval (AdOptimizationEnv): Evaluation environment.
        max_episodes (int): Maximum number of episodes for training.
        fixed_params (dict): Fixed parameters to use (overrides trial suggestions).
        
    Returns:
        float: Evaluation metric to optimize (average reward).
    """
    # Parameters to optimize
    params = {
        # Network architecture
        'hidden_layer_1_size': trial.suggest_int('hidden_layer_1_size', 32, 256),
        'hidden_layer_2_size': trial.suggest_int('hidden_layer_2_size', 16, 128),
        
        # Training parameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'target_update_freq': trial.suggest_int('target_update_freq', 5, 50),
        
        # Exploration parameters
        'eps_init': trial.suggest_float('eps_init', 0.5, 1.0),
        'eps_end': trial.suggest_float('eps_end', 0.01, 0.1),
    }
    
    # Override with fixed parameters if provided
    if fixed_params:
        for k, v in fixed_params.items():
            params[k] = v
    
    # Create policy network
    hidden_layers = [params['hidden_layer_1_size'], params['hidden_layer_2_size']]
    policy = create_policy_network(env_train, hidden_layers)
    
    # Train the agent
    training_metrics = train_agent(
        env=env_train,
        policy=policy,
        total_episodes=max_episodes,
        batch_size=params['batch_size'],
        lr=params['learning_rate'],
        gamma=params['gamma'],
        target_update_freq=params['target_update_freq']
    )
    
    # Evaluate the agent
    eval_results = evaluate_agent(env_eval, policy, num_episodes=10)
    
    # Return the metric to optimize
    avg_reward = eval_results['avg_reward']
    
    # Prune trials that perform very poorly
    if avg_reward < -5.0:
        raise optuna.exceptions.TrialPruned()
    
    return avg_reward

def visualize_optimization_results(study, output_dir="optuna_results"):
    """
    Visualize optimization results from Optuna study.
    
    Args:
        study (optuna.Study): Completed Optuna study.
        output_dir (str): Directory to save visualization results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimization_history.png", dpi=300)
    plt.close()
    
    # 2. Plot parameter importances
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/param_importances.png", dpi=300)
    plt.close()
    
    # 3. Plot parallel coordinate plot
    plt.figure(figsize=(15, 10))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parallel_coordinate.png", dpi=300)
    plt.close()
    
    # 4. Plot slice plot
    plt.figure(figsize=(15, 10))
    optuna.visualization.matplotlib.plot_slice(study)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/slice_plot.png", dpi=300)
    plt.close()
    
    # 5. Plot contour plot
    plt.figure(figsize=(12, 10))
    param_names = ['learning_rate', 'gamma']  # Choose parameters to visualize
    optuna.visualization.matplotlib.plot_contour(study, params=param_names)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/contour_plot.png", dpi=300)
    plt.close()
    
    # 6. Create a table of top trials
    top_trials = study.trials_dataframe().sort_values('value', ascending=False).head(10)
    top_trials.to_csv(f"{output_dir}/top_trials.csv", index=False)
    
    # 7. Visualize the best configurations
    best_params = study.best_params
    param_values = list(best_params.values())
    param_names = list(best_params.keys())
    
    plt.figure(figsize=(12, 8))
    plt.bar(param_names, param_values)
    plt.title("Best Parameter Configuration")
    plt.xlabel("Parameter")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/best_params.png", dpi=300)
    plt.close()
    
    print(f"Optimization visualizations saved to {output_dir}")
    
    return top_trials

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Ad Optimization RL")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset CSV (if None, generates synthetic data)")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--output_dir", type=str, default="hyperparameter_tuning_results", help="Output directory for results")
    parser.add_argument("--study_name", type=str, default=None, help="Name for the Optuna study")
    parser.add_argument("--max_episodes", type=int, default=100, help="Maximum episodes per trial")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set study name if not provided
    if args.study_name is None:
        args.study_name = f"ad_optimization_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    
    # Split into training and validation sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    
    train_dataset = dataset.iloc[:train_size].reset_index(drop=True)
    val_dataset = dataset.iloc[train_size:train_size+val_size].reset_index(drop=True)
    test_dataset = dataset.iloc[train_size+val_size:].reset_index(drop=True)
    
    print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create environments
    env_train = AdOptimizationEnv(train_dataset, device=device)
    env_val = AdOptimizationEnv(val_dataset, device=device)
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize
    print(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, env_train, env_val, max_episodes=args.max_episodes),
        n_trials=args.n_trials,
        timeout=60*60*8  # 8-hour timeout
    )
    
    # Print optimization results
    print("Optimization completed!")
    print(f"Best value: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for param_name, param_value in study.best_params.items():
        print(f"  {param_name}: {param_value}")
    
    # Visualize results
    top_trials = visualize_optimization_results(study, output_dir=f"{args.output_dir}/optuna_visualizations")
    
    # Train the final model with the best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    best_params = study.best_params
    
    # Create policy network with best parameters
    hidden_layers = [best_params['hidden_layer_1_size'], best_params['hidden_layer_2_size']]
    best_policy = create_policy_network(env_train, hidden_layers)
    
    # Train the best model
    training_metrics = train_agent(
        env=env_train,
        policy=best_policy,
        total_episodes=args.max_episodes * 2,  # Train for longer
        batch_size=best_params['batch_size'],
        lr=best_params['learning_rate'],
        gamma=best_params['gamma'],
        target_update_freq=best_params['target_update_freq']
    )
    
    # Evaluate on test set
    env_test = AdOptimizationEnv(test_dataset, device=device)
    final_eval_results = evaluate_agent(env_test, best_policy, num_episodes=20)
    
    # Print final evaluation results
    print("\nFinal Evaluation Results (Test Set):")
    print(f"Average Reward: {final_eval_results['avg_reward']:.4f}")
    print(f"Success Rate: {final_eval_results['success_rate']:.4f}")
    
    # Save final model
    model_path = f"{args.output_dir}/optimized_model.pt"
    torch.save({
        'model_state_dict': best_policy.state_dict(),
        'hyperparameters': best_params,
        'feature_columns': feature_columns,
        'training_metrics': training_metrics,
        'evaluation_results': final_eval_results,
        'study_best_value': study.best_value
    }, model_path)
    print(f"Optimized model saved to {model_path}")
    
    # Save hyperparameter importance analysis
    importance = optuna.importance.get_param_importances(study)
    importance_df = pd.DataFrame(
        importance.items(), 
        columns=['Parameter', 'Importance']
    ).sort_values('Importance', ascending=False)
    
    importance_df.to_csv(f"{args.output_dir}/parameter_importance.csv", index=False)
    
    # Create hyperparameter comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Define parameters to compare (focus on the most important ones)
    top_params = importance_df.head(5)['Parameter'].tolist()
    
    # Get data from all trials
    trials_df = study.trials_dataframe()
    
    # Create a scatter plot of the top parameters vs performance
    fig, axes = plt.subplots(len(top_params), 1, figsize=(10, 3*len(top_params)))
    
    for i, param in enumerate(top_params):
        if param in trials_df.columns:
            sns.scatterplot(x=param, y='value', data=trials_df, ax=axes[i])
            axes[i].set_title(f"Effect of {param} on Performance")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Reward")
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/performance_vs_parameters.png", dpi=300)
    plt.close()
    
    print(f"All results saved to {args.output_dir}")
    
    return study, best_policy, final_eval_results

if __name__ == "__main__":
    main()