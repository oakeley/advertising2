# Digital Advertising Optimization with Reinforcement Learning

This repository contains a collection of tools and scripts for optimizing digital advertising strategies using reinforcement learning. The code leverages TorchRL to train agents that learn effective ad spending decisions across multiple keywords based on metrics like ROAS (Return on Ad Spend), CTR (Click-Through Rate), and other performance indicators.

## Project Structure

- `integrated_ad_optimization.py`: Main script containing the RL environment, model architecture, training, and evaluation logic
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

## Implementation Notes

- The environment uses TorchRL's `EnvBase` for compatibility with the TorchRL ecosystem
- The model leverages TensorDict for structured data management
- Custom reward function considers multiple factors for realistic digital advertising scenarios
- The implementation supports both CPU and GPU training

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

## References

- TorchRL: https://github.com/pytorch/rl
- PyTorch: https://pytorch.org/
- Digital Advertising Metrics: [Google Ads Help](https://support.google.com/google-ads/answer/2472674?hl=en)
