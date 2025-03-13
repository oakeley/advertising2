# Digital Advertising Optimization with Reinforcement Learning

This repository contains a collection of tools and scripts for optimizing digital advertising strategies using reinforcement learning. The code leverages TorchRL to train agents that learn effective ad spending decisions across multiple keywords based on metrics like ROAS (Return on Ad Spend), CTR (Click-Through Rate), and other performance indicators.

## Project Structure

- `integrated_ad_optimization.py`: Main script containing the RL environment, model architecture, training, and evaluation logic
- `hyperparameter_tuning.py`: Bayesian optimization framework for systematic hyperparameter exploration
- `visualize_ad_performance.py`: Visualization tools for analyzing trained models and their decision patterns
- `ad_optimization_environment.yml`: Conda environment specification with all dependencies

## Installation

### Setting up the conda environment

```bash
# Clone the repository
git clone https://github.com/oakeley/advertising2.git
cd advertising2

# Create and activate the conda environment
conda env create -f ad_optimization_environment.yml
conda activate torch_rl_ad_optimization
# For GPUs you need to install the GPU version of PyTorch separately
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Usage

### Training a model

To train a new model on synthetic data:

```bash
python integrated_ad_optimization.py
```

This will:
1. Generate synthetic advertising data with realistic correlations
2. Train a reinforcement learning agent to optimize ad spending
3. Evaluate the agent on a test set
4. Save the trained model, training metrics, and visualizations

The results will be saved in a timestamped directory `ad_optimization_results_YYYYMMDD_HHMMSS/`.

### Hyperparameter Tuning

The repository includes a dedicated module for Bayesian optimization of model hyperparameters using Optuna. This enables systematic exploration of the parameter space to identify optimal configurations for the reinforcement learning agent.

#### Running Hyperparameter Optimization

To execute hyperparameter tuning:

```bash
python hyperparameter_tuning.py [options]
```

Optional arguments:
- `--dataset path/to/dataset.csv`: Specify a custom dataset (generates synthetic data if omitted)
- `--n_trials 50`: Number of optimization trials to run (default: 50)
- `--output_dir dir_name`: Output directory for results (default: "hyperparameter_tuning_results")
- `--study_name name`: Name for the Optuna study (default: auto-generated timestamp)
- `--max_episodes 100`: Maximum episodes per trial (default: 100)

#### Optimized Parameters

The optimization process explores various hyperparameters including:

- **Network architecture**:
  - Hidden layer sizes
  - Activation functions

- **Training dynamics**:
  - Learning rate
  - Batch size
  - Discount factor (gamma)
  - Target network update frequency

- **Exploration strategy**:
  - Initial exploration rate
  - Final exploration rate
  - Exploration decay schedule

#### Optimization Results

The hyperparameter tuning process generates comprehensive analyses:

1. **Parameter Importance Analysis**:
   - Visualization of parameter contribution to performance
   - Sensitivity analysis for key parameters

2. **Performance Visualization**:
   - Optimization history plots
   - Parallel coordinate plots showing parameter interactions
   - Contour plots of parameter landscapes

3. **Best Configuration**:
   - Saved model with optimal parameters
   - Detailed evaluation on test dataset
   - Parameter importance rankings

The hyperparameter tuning module implements Bayesian optimization with Tree-structured Parzen Estimator (TPE) sampling, enabling efficient exploration of the high-dimensional parameter space while pruning suboptimal trials to focus computational resources on promising configurations.

### Visualizing model decisions

To visualize how the trained model makes decisions:

```bash
python visualize_ad_performance.py --model path/to/ad_optimization_model.pt
```

Optional arguments:
- `--dataset path/to/dataset.csv`: Use a specific dataset (if omitted, generates synthetic data)
- `--output_dir dir_name`: Specify output directory for visualizations (default: "visualization_results")

## Environment Details

The reinforcement learning environment (`AdOptimizationEnv`) models the digital advertising optimization problem with:

- **State**: Keyword metrics including competitiveness, CTR, ROAS, ad spend, etc.
- **Actions**: For each keyword, decide whether to invest or not
- **Reward**: Based on ROAS, CTR, and ad spend levels

## Model Architecture

The agent uses a neural network with:
- Input layer that processes flattened keyword features, cash balance, and current holdings
- Hidden layers (128, 64 neurons) with ReLU activation
- Output layer that produces Q-values for each possible action
- Uses epsilon-greedy exploration during training

## Visualizations

The project generates various visualizations to analyze model performance and decision patterns:

1. **Training Progress**:
   - Episode rewards
   - Loss values
   - Exploration rate (epsilon)

2. **Evaluation Results**:
   - Action distribution
   - Average reward by action type
   - Feature correlations with decisions
   - Reward distribution
   - Decision quality matrix
   - Success rate

3. **Keyword Decision Analysis**:
   - ROAS vs Ad Spend decision patterns
   - CTR vs Competitiveness decision patterns
   - Conversion Rate vs Ad Spend decision patterns
   - Decision distribution by keyword
   - Decision boundary analysis

4. **Reward Component Analysis**:
   - ROAS distribution by decision
   - CTR distribution by decision
   - Decisions by reward component
   - Ad Spend vs ROAS by reward component

5. **Hyperparameter Analysis**:
   - Parameter importance visualizations
   - Optimization history trajectories
   - Parallel coordinate plots of trial configurations
   - Slice plots showing parameter sensitivities
   - Contour plots of parameter interactions

## Implementation Notes

- The environment uses TorchRL's `EnvBase` for compatibility with the TorchRL ecosystem
- The model leverages TensorDict for structured data management
- Custom reward function considers multiple factors for realistic digital advertising scenarios
- The implementation supports both CPU and GPU training
- Hyperparameter optimization utilizes Optuna's Bayesian optimization framework with pruning

## Extending the Project

To use your own advertising data:

1. Prepare a CSV with the following columns:
   - keyword
   - competitiveness
   - difficulty_score
   - organic_rank
   - organic_ctr
   - paid_ctr
   - ad_spend
   - ad_conversions
   - ad_roas
   - conversion_rate
   - cost_per_click
   - impression_share
   - conversion_value

2. Load this CSV when training:
   ```python
   dataset = pd.read_csv('your_data.csv')
   ```

3. Train the model as normal with your custom dataset.

## Technical Implementation Details

### Environment Implementation

The `AdOptimizationEnv` class extends TorchRL's `EnvBase` to create a Markov Decision Process that models digital advertising optimization. The environment implements:

- A vectorized state space representing multiple keywords and their associated metrics
- A discrete action space where the agent selects which keywords to invest in
- A reward function that evaluates decisions based on ROAS thresholds and CTR performance
- Step-wise transitions that mimic the temporal dynamics of advertising performance

### Neural Network Architecture

The agent's policy network consists of:

1. A flattening module that combines heterogeneous inputs (keyword features, cash balance, and current holdings)
2. A multi-layer perceptron with configurable hidden layers and activation functions
3. A Q-value module that outputs action probabilities across the discrete action space

### Training Process

The training procedure implements:

- Experience replay with a TensorDict-based replay buffer
- Epsilon-greedy exploration with annealing
- Double DQN with periodic target network updates
- Customizable hyperparameters for learning rate, batch size, and discount factor

### Hyperparameter Optimization

The `hyperparameter_tuning.py` script implements Bayesian optimization using Optuna to:

1. Define a search space over network architecture and training parameters
2. Efficiently sample configurations using Tree-structured Parzen Estimator
3. Evaluate configurations with early stopping to conserve computational resources
4. Analyze parameter importance and interactions
5. Select optimal configuration based on evaluation metrics

## References

- TorchRL: https://github.com/pytorch/rl
- PyTorch: https://pytorch.org/
- Optuna: https://optuna.org/
- Digital Advertising Metrics: [Google Ads Help](https://support.google.com/google-ads/answer/2472674?hl=en)
