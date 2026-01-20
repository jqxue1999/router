# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Code Organization

**The codebase has been reorganized into logical packages:**
- All main result generation code is in `main/` package
- Organized into subpackages: `core/`, `baselines/`, `shared/`, `evaluation/`
- Use Python module syntax: `python -m main.core.train_core`
- See `REORGANIZATION_V2_SUMMARY.md` for full details

## Quick Start

**Run all experiments (IID + OOD + UniRouter):**
```bash
bash run.sh  # Complete pipeline - trains and evaluates all methods
```

**For IID evaluation only (recommended first step):**
```bash
bash main/run_experiment.sh  # Run IID pipeline with all methods
```

**For interactive demo:**
```bash
export OPENROUTER_API_KEY="your-key"
cd demo && python app.py  # Launch Gradio web interface
```

**For training new predictors:**
```bash
python -m main.core.train_core --model_type sklearn  # Fast sklearn-based training
# OR
python -m main.core.train_core --model_type torch_mlp  # Better accuracy with PyTorch
```

## Project Overview

This is a research project for **LLM routing** - intelligently selecting which LLM and token limit to use for each query to optimize the tradeoff between performance (accuracy) and cost (model size × tokens used). The system, called **CoRE (Constrained Response Evaluator)**, predicts performance scores for different (LLM, token_limit) combinations and selects the best option based on a risk function that balances accuracy and cost.

### Dataset: CoRD (Constrained Response Dataset)

The evaluation uses **CoRD**, an extension of the SPROUT benchmark containing **30,968 queries** across **20 categories** from 6 diverse benchmarks:
- TIGER-Lab/MMLU-Pro (8,264 queries) - Professional-level questions
- openhermes/teknium (13,670 queries) - General knowledge
- lighteval/MATH/all (5,122 queries) - Math problems
- Idavidrein/gpqa/gpqa_extended (384 queries) - Graduate-level science
- lighteval/mmlu/high_school_computer_science (100 queries)
- 15 additional categories

Each query has responses from multiple LLMs under **16 token budgets**: {10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000, unlimited}

## Core Architecture

### Routing System
The system operates in two phases:
1. **Training Phase**: Train neural network predictors for each LLM to estimate performance scores across different token limits
2. **Routing Phase**: Use predictions to select optimal (LLM, token_limit) combination for each query based on cost-performance tradeoff

### Key Components

The codebase is organized into the `main/` package with logical subpackages:

**main/shared/** - Shared utilities
- **dataset_manager.py**: Centralized train/test split management
  - `DatasetManager`: Creates and manages a single train/test split used by all LLMs
  - Ensures all models see the same queries for fair comparison
  - Loads embeddings once and provides them to all components
  - Uses fixed seed (42) for reproducibility

- **llm_loader.py**: LLM data loading and management
  - `load_llm()`: Loads a single LLM's data including pre-trained predictor checkpoints, ground-truth performance scores, token counts
  - Accepts centralized embeddings and split indices from `DatasetManager`
  - Accepts optional `predictor_class` parameter to choose between PyTorch or sklearn predictors
  - Returns a dictionary with predicted and true scores/counts for both train and test splits

- **router_dataset.py**: Dataset management
  - `RouterDataset`: Manages performance scores and creates train/test subsets
  - Accepts pre-computed train/test indices from `DatasetManager`
  - Handles multiple token limit columns (e.g., '10_score', '20_score', ..., 'unlimited_score')

- **utils.py**: Utility functions
  - `ultimate_json_score_extractor()`: Robust JSON parsing for extracting correctness scores from LLM outputs
  - Plotting functions for confusion matrices and performance visualization
  - Data loading and preprocessing utilities

**main/core/** - CoRE predictor implementations
- **predictor.py**: PyTorch-based neural network predictor
  - `TokenPerformancePredictor`: Trains per-token-limit MLPs to predict performance scores
  - Architecture: Input → [256, 128, 64] → 1 (with ReLU + Dropout 0.5)
  - Also includes a linear regression model to predict token counts
  - `route_scores()`: Routes queries by computing risk = (1-λ)×predicted_score - λ×predicted_cost
  - Uses PyTorch with BCE loss (scores are in [0,1])
  - Training: 100 epochs, lr=1e-4, AdamW optimizer

- **predictor_sklearn.py**: Sklearn-based linear regression predictor
  - `TokenPerformancePredictor`: Uses sklearn's LinearRegression instead of PyTorch neural networks
  - Simpler and faster for baseline comparisons
  - Same interface as predictor.py for easy swapping
  - Saves models as `.joblib` files instead of `.pt` files

- **train_core.py**: Training script for CoRE predictors
  - Unified training script supporting both PyTorch and sklearn predictors
  - Command-line interface for model configuration

**main/baselines/carrot/** - CARROT baseline methods (sklearn-based)
- **baselines_carrot.py**: CARROT implementations
  - `CarrotKNNBaseline`: KNN-based routing (k=256, cosine distance)
  - `CarrotLinearBaseline`: Linear regression-based routing (vLLM-SR baseline)
  - `route_baseline()`: Generic routing function that selects best (model, token_limit) based on predictions
  - Both classes support checkpoint save/load using `.joblib` files

- **train_carrot.py**: Training script for CARROT baselines

**main/baselines/irt/** - IRT baseline methods (PyTorch-based)
- **baselines_irt.py**: IRT implementations
  - `IRTBaseline`: Multidimensional IRT baseline (MIRT-Router) with latent ability modeling
  - `NIRTBaseline`: Neural IRT baseline (NIRT-Router) with relevance vectors and PCA
  - Both classes support checkpoint save/load using PyTorch `.pt` files

- **train_irt.py**: Training script for IRT baselines

**main/evaluation/** - Evaluation scripts
- **compare_methods.py**: Method comparison script
  - Compares CoRE vs CARROT vs IRT baselines
  - Generates comparison plots and metrics

- **results.py**: Main IID evaluation script
  - Initializes `DatasetManager` to create centralized train/test split (80/20)
  - Loads multiple LLMs with shared split
  - Evaluates routing methods across different λ values (cost-performance tradeoff parameter)
  - Computes Oracle curves (theoretical upper bounds with perfect knowledge)
  - Generates performance metrics: AUDC, Peak Accuracy, QNC
  - Saves results to CSV and generates plots

**main/run_experiment.sh** - Main experiment pipeline
- Complete automated pipeline for training and evaluation
- Handles CoRE, CARROT, and IRT training
- Runs comprehensive evaluation and comparison

## Common Commands

### Training Predictor Models

**Using the unified training script (recommended):**
```bash
# PyTorch-based (Neural Networks - better accuracy)
python -m main.core.train_core \
    --model_type torch_mlp \
    --model "Model-Name" "0.85" "data/Model-Name.csv" "checkpoints/Model-Name_1e4"

# Sklearn-based (Linear Regression - faster)
python -m main.core.train_core \
    --model_type sklearn \
    --model "Model-Name" "0.85" "data/Model-Name.csv" "checkpoints/Model-Name_multi"

# Or use the automated pipeline (trains all configured models)
bash main/run_experiment.sh
```

**Key training parameters:**
- Centralized train/test split (seed=42)
- PyTorch: Hidden dims [256, 128, 64], Dropout 0.5, LR 1e-4, 100 epochs
- Sklearn: LinearRegression (instant training, no hyperparameters)
- Saves to `./checkpoints/{model-name}_{type}/`
- Generates confusion matrix plots in `./plots/`

### Training Baseline Models

**CARROT Baselines (KNN and Linear Regression):**
```bash
# Use the automated pipeline (recommended)
bash main/run_experiment.sh

# Or train manually:
python -c "
from main.shared.dataset_manager import DatasetManager
from main.shared.llm_loader import load_llm
from main.core.predictor_sklearn import TokenPerformancePredictor
from main.baselines.carrot.baselines_carrot import CarrotKNNBaseline, CarrotLinearBaseline

# Load LLM data (example with 2 models)
dataset_manager = DatasetManager('data/prompt_embeddings.pkl', train_ratio=0.8, seed=42)
embeddings = dataset_manager.get_embeddings()
train_idx, test_idx = dataset_manager.get_split_indices()

token_limits_score = ['10_score', ..., 'unlimited_score']  # 16 token limits
token_limits_count = ['10_count', ..., 'unlimited_count']

llms = {}
llms['GLM-4.5-Air'] = load_llm('GLM-4.5-Air', 0.85, 'data/GLM-4.5-Air.csv',
                                './checkpoints/GLM-4.5-Air_multi', embeddings,
                                train_idx, test_idx, token_limits_score,
                                token_limits_count, TokenPerformancePredictor)

# Train CARROT-KNN
carrot_knn = CarrotKNNBaseline(llms, n_neighbors_score=256, n_neighbors_count=256)
carrot_knn.fit(save_dir='./checkpoints/carrot_knn')

# Train CARROT-Linear
carrot_linear = CarrotLinearBaseline(llms, fit_intercept=True)
carrot_linear.fit(save_dir='./checkpoints/carrot_linear')
"
# Saves to ./checkpoints/carrot_knn/ and ./checkpoints/carrot_linear/
# Uses .joblib format for sklearn models (KNN and LinearRegression)
```

**IRT Baselines (MIRT and NIRT with PyTorch):**
```bash
# Use the automated pipeline (recommended)
bash main/run_experiment.sh

# Or train manually:
python -c "
from main.baselines.irt.baselines_irt import IRTBaseline, NIRTBaseline

# Define LLM descriptions for embedding
llm_texts = {
    'GLM-4.5-Air': 'GLM-4.5-Air is a lightweight language model',
    'Llama-3.2-3B': 'Llama-3.2-3B is a small instruction-tuned model'
}

# Train MIRT (Multidimensional IRT)
irt = IRTBaseline(llms, llm_texts=llm_texts, latent_dim=32, device='cuda')
irt.fit(lr=3e-3, batch_size=128, epochs=200,
        save_dir='./checkpoints/irt_mirt',
        plot_dir='./plots/irt_mirt')

# Train NIRT (Neural IRT)
nirt = NIRTBaseline(llms, llm_texts=llm_texts, latent_dim=32, device='cuda')
nirt.fit(lr=3e-3, batch_size=128, epochs=200,
         save_dir='./checkpoints/irt_nirt',
         plot_dir='./plots/irt_nirt')
"
# Saves PyTorch models to ./checkpoints/irt_mirt/ and ./checkpoints/irt_nirt/
# Generates confusion matrices in ./plots/
# Training takes ~5-10 minutes per baseline on GPU
```

**Loading Pre-trained Baselines:**
```bash
# CARROT baselines load automatically from directory:
carrot_knn = CarrotKNNBaseline(llms, load_dir='./checkpoints/carrot_knn')
carrot_linear = CarrotLinearBaseline(llms, load_dir='./checkpoints/carrot_linear')

# IRT baselines load automatically from directory:
irt = IRTBaseline(llms, llm_texts=llm_texts, load_dir='./checkpoints/irt_mirt')
nirt = NIRTBaseline(llms, llm_texts=llm_texts, load_dir='./checkpoints/irt_nirt')
```

### Running Evaluations

**IID (In-Distribution) Evaluation:**
```bash
# Run complete pipeline (recommended)
bash main/run_experiment.sh

# Or run evaluation only (requires pre-trained checkpoints)
python -m main.evaluation.compare_methods \
    --lambda-dist "0,0.001,100;0.001,0.01,100;0.01,0.1,100;0.1,1.0,100" \
    --model "Model-Name" "0.85" "data/Model-Name.csv" "checkpoints/Model-Name_multi"

# This will:
# 1. Initialize DatasetManager with centralized train/test split (80/20)
# 2. Load all LLMs from checkpoints with shared split
# 3. Compute routing performance curves across λ values
# 4. Compare against baselines (CARROT-KNN, CARROT-Linear, MIRT, NIRT)
# 5. Generate plots and save results to ./comparison_results/:
#    - core_vs_baselines_curves.csv: Cost-performance curves for all methods
#    - core_vs_baselines_metrics.csv: AUDC, Peak Accuracy, QNC metrics

# To switch between PyTorch and sklearn predictors:
# Edit main/evaluation/results.py or main/evaluation/compare_methods.py
# and change the import:
#   - from ..core.predictor import TokenPerformancePredictor (PyTorch)
#   - from ..core.predictor_sklearn import TokenPerformancePredictor (sklearn)
```

**OOD (Out-of-Distribution) Evaluation:**
```bash
# Run OOD evaluation on MMLU-Pro (default held-out category)
python ood_evaluation/run_ood.py

# Quick demo with 1 model (fast for testing)
python ood_evaluation/run_ood.py --quick

# Test on a different category
python ood_evaluation/run_ood.py --category "lighteval/MATH/all"

# Use 90% of best LLM performance as QNC target (instead of 100%)
python ood_evaluation/run_ood.py --target-accuracy-rate 0.9

# Or use the experiment script with target rate
bash ood_evaluation/run_ood_experiment.sh --category "rungalileo/ragbench/finqa" --target-rate 0.9

# Results saved to ./comparison_results/ood_evaluation/{category}/
# - {category}_metrics.csv: Peak accuracy, AUDC, QNC for each method
# - {category}_curves.csv: Full cost-performance curves
# - {category}_plot.png: Visual comparison
```

**UniRouter Evaluation:**
```bash
# Run UniRouter vs Uni-CoRE comparison (complete pipeline)
bash unirouter/run_unirouter_experiment.sh

# This will:
# 1. Create validation set (500 queries by default)
# 2. Train Original UniRouter (unlimited tokens only)
# 3. Train Uni-CoRE (multiple token budgets: 10-4000 + unlimited)
# 4. Evaluate initial model pool
# 5. Dynamically add new models WITHOUT retraining
# 6. Compare routing performance across methods

# Results saved to ./comparison_results/unirouter/
# - comparison_metrics.csv: Peak accuracy, AUDC, QNC for each method
# - comparison_curves.csv: Full cost-performance curves
# - comparison_plot_actual.png: Plot with actual costs
# - comparison_plot_normalized.png: Plot with normalized costs

# Checkpoints saved to ./checkpoints/unirouter/
# - original_unirouter.pkl: Original UniRouter model
# - unicore_feature_matrix.pkl: Uni-CoRE feature matrix
# - validation_set.pkl: Validation set indices
# - config.txt: Experiment configuration

# Key difference from CoRE:
# - Original UniRouter: Routes to best LLM (always unlimited tokens)
# - Uni-CoRE: Routes to best (LLM, token_budget) pair
# - Both support dynamic model addition without retraining
```

### Data Inspection
```bash
python check_data.py
# Inspects data files and validates CSV structure
```

### Running Tests

**Validation tests (not CI, but useful for development):**
```bash
# Test baseline implementations after refactoring
python tests/test_baselines_split.py

# Test sklearn predictor implementation
python tests/test_sklearn_refactor.py

# Test API refactoring
python tests/test_refactor_api.py
```

**Note**: Tests may create temporary files in `./test_checkpoints/` and `./test_plots/`

### Comparison and Training Scripts

**Method Comparison:**
```bash
# Compare different routing methods (CoRE vs CARROT vs IRT baselines)
python -m main.evaluation.compare_methods \
    --lambda-dist "0,1,100" \
    --model "Model-Name" "0.85" "data/Model-Name.csv" "checkpoints/Model-Name_multi"
# Generates comparison plots and metrics including AUDC, Peak Accuracy, and QNC
```

**Individual Training Scripts:**
```bash
# Train CoRE predictors
python -m main.core.train_core --model_type sklearn --model ...

# Train CARROT baselines
python -m main.baselines.carrot.train_carrot --model ...

# Train IRT baselines
python -m main.baselines.irt.train_irt --model ...
```

## Data Files and Directory Structure

### Input Data (`data/`)
- `prompt_embeddings.pkl`: Shared prompt embeddings for all LLMs (277MB, 30,968 queries × embedding_dim)
- `{model-name}.csv`: Performance scores and token counts for each LLM
  - Available LLMs: GLM-4.5-Air, GLM-4.6, gemma-3-4b-it, Llama-3.1-70B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-Math-1.5B-Instruct, Qwen2.5-Math-7B-Instruct, Qwen3-0.6B, Qwen3-235B-A22B-Instruct-2507, Qwen3-Next-80B-A3B-Instruct

### Checkpoints (`checkpoints/`)

**Predictor Models:**
- PyTorch: `{model-name}_1e4/score_predictor.pt` - Neural network weights
- Sklearn: `{model-name}_multi/` - Three .joblib files:
  - `limited_score_predictors.joblib` - Dict with 15 LinearRegression models for limited budgets
  - `unlimited_score_predictor.joblib` - Single LinearRegression for unlimited quality
  - `unlimited_token_predictor.joblib` - Single LinearRegression for token counts

**Baseline Models:**
- CARROT-KNN: `carrot_knn/` - KNN models (sklearn)
  - `knn_score.joblib` - KNN for quality scores
  - `knn_count.joblib` - KNN for token counts
- CARROT-Linear: `carrot_linear/` - Linear regression models (sklearn)
  - `linear_score.joblib` - Linear regression for quality scores
  - `linear_count.joblib` - Linear regression for token counts
- MIRT: `irt_mirt/` - Multidimensional IRT models (PyTorch)
  - `mirt_model.pt` - Model state dict
  - `mirt_config.pt` - Model configuration
  - `mirt_llm_embeddings.pt` - LLM embeddings
- NIRT: `irt_nirt/` - Neural IRT models (PyTorch)
  - `nirt_model.pt` - Model state dict
  - `nirt_config.pt` - Model configuration
  - `nirt_llm_embeddings.pt` - LLM embeddings
  - `nirt_pca.pkl` - PCA for relevance vectors

**Important Note on Checkpoint Organization:**
The project uses a unified checkpoint structure under `checkpoints/` with subdirectories for different evaluation types:

- `checkpoints/main/`: **IID checkpoints** trained on ALL 30,968 queries (80/20 random split)
  - Used for IID evaluation and interactive demo
  - Contains CoRE predictors, CARROT baselines, and IRT baselines

- `checkpoints/ood_evaluation/{category}/`: **OOD checkpoints** trained on 19 categories only
  - One subdirectory per held-out category (e.g., `checkpoints/ood_evaluation/rungalileo_ragbench_finqa/`)
  - Ensures OOD evaluation tests true generalization without data leakage
  - Each category has its own trained predictors and baselines

- `checkpoints/unirouter/`: **UniRouter checkpoints** for dynamic model addition experiments
  - Contains Original UniRouter and Uni-CoRE feature matrices
  - Validation set indices for consistent comparison

This separation ensures proper evaluation:
- IID: Tests on random 20% holdout
- OOD: Tests on completely unseen query categories
- UniRouter: Tests dynamic model addition without retraining

### Comparison Results (`comparison_results/`)

Unified results directory for all evaluation types:

**`comparison_results/main/`**: IID evaluation results
- `core_vs_baselines_metrics.csv`: Peak accuracy, AUDC, QNC for each method
- `core_vs_baselines_curves.csv`: Full cost-performance curves
- `core_vs_baselines_curves_actual.png`: Plot with actual costs
- `core_vs_baselines_curves_normalized.png`: Plot with normalized costs

**`comparison_results/ood_evaluation/{category}/`**: OOD evaluation results per category
- `{category}_metrics.csv`: Peak accuracy, AUDC, QNC for each method
- `{category}_curves.csv`: Full cost-performance curves
- `{category}_plot.png`: Visual comparison
- `config.txt`: Experiment configuration (LLM pool, hyperparameters, QNC settings)

**`comparison_results/unirouter/`**: UniRouter vs Uni-CoRE comparison results
- `comparison_metrics.csv`: Peak accuracy, AUDC, QNC for each method
- `comparison_curves.csv`: Full cost-performance curves
- `comparison_plot_actual.png`: Plot with actual costs
- `comparison_plot_normalized.png`: Plot with normalized costs
- `config.txt`: Experiment configuration (initial pool, new models, validation settings)

### OOD Evaluation Files (`ood_evaluation/`)
- `run_ood.py`: Main OOD evaluation script
- `run_ood_experiment.sh`: Automated pipeline for OOD evaluation
- `ood_dataset_manager.py`: Manages category-based train/test splits
- `map_and_split_data.py`: Creates category splits (run once for setup)
- `category_splits/`: Pre-computed splits and embeddings per category
  - `ood_splits.pkl`: Train/test indices for each held-out category
  - `embeddings/`: Category-specific embeddings

### UniRouter Files (`unirouter/`)
- `run_unirouter_experiment.sh`: Automated pipeline for UniRouter comparison
- `eval_compare.py`: Evaluation script comparing Original UniRouter vs Uni-CoRE
- `uni_core.py`: Uni-CoRE implementation (feature matrix and routing)
- `unirouter_original.py`: Original UniRouter implementation

### Visualization (`plots/`)
- Confusion matrices from predictor training
- Performance comparison plots

### Logs (`logs/`)
Created by `run.sh` to capture experiment outputs:
- `run_ood_experiment.log`: OOD evaluation logs
- `run_experiment.log`: IID evaluation logs
- `run_unirouter_experiment.log`: UniRouter comparison logs

## Data Format

### CSV Structure
Each LLM CSV contains these columns (example row shown in check_data.py output):
- `prompts_id`: Unique identifier for the query
- `key`: Hash key for the query
- `original_prompt`: The actual query text
- `{limit}_score`: Performance score (0-1) for token limit (e.g., '10_score', '20_score', ..., 'unlimited_score')
- `{limit}_count`: Actual token count used for that limit (e.g., '10_count', '20_count', ..., 'unlimited_count')
- `{limit}_score_most`: Additional variant scores (used for majority voting or alternative evaluations)
- `{limit}_count_most`: Corresponding token counts for _most variants
- One row per query prompt (30,968 rows total)

**Important**: The code primarily uses the `{limit}_score` and `{limit}_count` columns. The `_most` variants are available but not used in the main routing pipeline.

### Embeddings
- `prompt_embeddings.pkl`: Pickled numpy array of shape (30968, 768)
- Generated using sentence-transformers/all-mpnet-base-v2
- Same ordering as CSV rows
- All LLMs use the same prompt embeddings (shared across models)

## Model Architecture Details

### TokenPerformancePredictor (PyTorch)
- **Architecture**: Separate MLP for each token limit (16 total: 15 limited + 1 unlimited)
  - Input: 768-dimensional query embedding
  - Hidden layers: [256, 128, 64] with ReLU activation
  - Dropout: 0.5 between layers
  - Output: 1 (sigmoid-activated score in [0,1])
- **Loss**: Binary Cross-Entropy with logits
- **Optimizer**: AdamW with lr=1e-4, weight_decay=1e-5
- **Training**: 100 epochs, batch_size=64, gradient clipping (max_norm=1.0)
- **Optional**: Weighted sampling to balance extreme buckets (0.0-0.1 and 0.9-1.0)
- **Token count predictor**: Simple linear regression on embeddings (predicts unlimited_count)

### TokenPerformancePredictor (Sklearn)
- **Architecture**: Linear regression per token limit (16 total)
  - Input: 768-dimensional query embedding
  - Output: Predicted score (clipped to [0,1])
- **Training**: Ordinary least squares (instant)
- **Token count predictor**: Same as PyTorch version

### Routing Decision
For each query embedding x:
1. Predict scores s_{m,t} for each (model m, token limit t)
2. Predict token counts c_{m,t}
3. Compute cost: cost_{m,t} = c_{m,t} × size_m
4. Compute risk: risk_{m,t} = (1-λ)s_{m,t} - λ×cost_{m,t}
5. Select argmax risk across all (m,t) options
6. Evaluate using ground-truth score and cost for that option

## Key Research Concepts

### Cost-Performance Tradeoff
- λ ∈ [0, 1e-4] controls tradeoff: λ=0 prioritizes performance, λ→1e-4 prioritizes cost
- Sweep λ with 100 equally-spaced values to generate cost-performance curves
- Cost formula: cost = token_count × model_size
- Model sizes (in billions): GLM-4.5-Air (0.85B), Llama-3.2-3B (0.06B), Qwen2.5-Math-7B (7B), Qwen3-235B (235B)

### Evaluation Metrics
- **AUDC (Area Under Deferral Curve)**: Area under cost-performance curve (higher is better)
  - **AUDC_normalized**: Computed with costs normalized to [0,1] using **global min/max across all methods**
  - **AUDC_actual**: Computed with actual cost values (token_count × model_size)
  - **Normalization Philosophy**: Uses global cost range (min/max across all methods) to ensure fair comparison
- **Peak Accuracy**: Maximum performance achieved [0, 1] (higher is better)
- **QNC (Query-Normalized Cost)**: Relative cost to achieve the same performance as the most accurate single LLM
  - **Definition**: Cost to match the best single LLM's average unlimited-token performance
  - **Target**: Best single LLM (highest average accuracy) × target_accuracy_rate
    - Default: target_accuracy_rate = 1.0 (100% of best LLM)
    - Configurable: use `--target-accuracy-rate 0.9` for 90% of best LLM
  - **Normalization**: Cost normalized using **global min/max across all methods** (same as AUDC_normalized)
  - **Range**: [0, 1] where lower is better (0 = minimum cost, 1 = maximum cost or cannot reach target)
  - **Failure case**: If a method cannot reach the target accuracy, QNC = 1.0 (100%)
  - **Important**: Only ONE QNC column in results (not QNC_normalized/QNC_actual)
  - **Consistency**: Both IID and OOD evaluation use identical global normalization approach

### Oracle Variants
- **Oracle (Unlimited)**: Always picks best model with unlimited tokens (upper bound on accuracy)
- **Oracle (All Limits)**: Picks best (model, token_limit) with perfect knowledge (absolute upper bound)
- **Pool Oracle Point**: For each query, find max performance across all options, then choose min cost among those

### Baselines
- **CARROT-KNN**: K-nearest neighbors (k=256) with cosine distance
- **CARROT-Linear**: Linear regression baseline (vLLM-SR)
- **MIRT**: Multidimensional IRT with 10-dim latent space, 50 epochs
- **NIRT**: Neural IRT with 64-dim hidden layer, 50 epochs

## Out-of-Distribution (OOD) Evaluation

Tests generalization to unseen query domains using leave-one-category-out:

### Strategy
- **Hold out one category** (e.g., MMLU-Pro with 8,264 queries) as OOD test set
- **Train on remaining 19 categories** (e.g., 22,704 queries)
- **Train all methods on-the-fly** using only OOD training split (no data leakage)
- **Evaluate** routing accuracy on held-out category

### Key Differences from IID
- **IID**: Random 80/20 split across all categories
- **OOD**: Train on 19 categories, test on 1 held-out category
- **Expected**: OOD accuracy typically 5-10% lower than IID

### Available Categories
20 categories from SPROUT dataset (see EXPERIMENTAL_SETTINGS.md for full list)

### Commands
```bash
# Default: MMLU-Pro as held-out category
python ood_evaluation/run_ood.py

# Different held-out category
python ood_evaluation/run_ood.py --category "lighteval/MATH/all"

# Quick demo (1 model only)
python ood_evaluation/run_ood.py --quick
```

## Development Notes

### Switching Between Predictors

The codebase supports two predictor implementations that can be easily swapped:

**In results.py:**
```python
# Option 1: PyTorch neural networks (better accuracy)
from predictor import TokenPerformancePredictor as PredictorClass

# Option 2: Sklearn linear regression (faster training)
from predictor_sklearn import TokenPerformancePredictor as PredictorClass
```

**In OOD evaluation:**
OOD evaluation uses sklearn predictors by default for faster training.

The `llm_loader.py` automatically handles both predictor types through the `predictor_class` parameter. No other code changes needed!

### Training New Models

1. Ensure data files are in `data/` directory
2. Create checkpoint directory: `mkdir -p checkpoints/{model-name}_1e4` (or `_multi` for sklearn)
3. Edit `predictor.py` (PyTorch) or `predictor_sklearn.py` (sklearn) main block:
   - Update model name and paths
   - Configure hyperparameters if using PyTorch
4. Run the training script
5. Checkpoints saved automatically

**Key PyTorch hyperparameters to tune:**
- `hidden_dims`: [256, 128, 64] (default)
- `dropout`: 0.5 (default)
- `lr`: 1e-4 (default)
- `epochs`: 100 (default)
- `use_sampler`: False (set True to balance extreme score buckets)

### Adding New LLMs

1. **Prepare data**: Add CSV file with performance data to `data/`
   - Must include all 16 token limits (10, 20, 30, ..., unlimited)
   - Columns: {limit}_score and {limit}_count
2. **Train predictor**:
   - PyTorch: Edit `predictor.py` main block
   - Sklearn: Add to `models_to_train` list in `predictor_sklearn.py`
3. **Add to evaluation**: Edit `results.py` llms dictionary:
   ```python
   "NEW_LLM": load_llm(
       name="New-LLM-Name",
       size=X.XX,  # Model size in billions
       score_df_path="data/New-LLM.csv",
       load_dir="./checkpoints/New-LLM_multi",
       embeddings=embeddings,
       train_idx=train_idx,
       test_idx=test_idx,
       token_limits_score=token_limits_score,
       token_limits_count=token_limits_count,
       predictor_class=PredictorClass
   ),
   ```
4. The predictor type is automatically determined by the `PredictorClass` import at the top of `results.py`

### Model Checkpoints
- **Required for prediction** (not trained on-the-fly in results.py)
- Training generates plots in `./plots/` directory showing confusion matrices
- PyTorch checkpoints: `.pt` files (larger, more accurate)
- Sklearn checkpoints: `.joblib` files (smaller, faster to load)
- Token predictor always uses sklearn LinearRegression

### Alternative Training Scripts
The codebase includes alternative training interfaces:
- `train_core.py`: Alternative to `predictor.py` with different configuration options
- `train_carrot.py`: Standalone script for training CARROT baselines

These scripts provide the same functionality as the main training scripts but may have different hyperparameter defaults or output formats. Use whichever interface you prefer.

### Typical Workflow

**For IID evaluation (recommended):**
```bash
# 1. Configure models in main/run_experiment.sh
#    Edit the LLM_POOL array with your models

# 2. Run complete pipeline (trains + evaluates)
bash main/run_experiment.sh

# 3. Check results
ls comparison_results/  # Metrics and curves
ls plots/              # Visualization outputs
```

**For manual training and evaluation:**
```bash
# 1. Train CoRE predictors for specific models
python -m main.core.train_core \
    --model_type sklearn \
    --model "Model-Name" "0.85" "data/Model.csv" "checkpoints/Model_multi"

# 2. Train baselines (optional)
python -m main.baselines.carrot.train_carrot --model ...
python -m main.baselines.irt.train_irt --model ...

# 3. Run evaluation
python -m main.evaluation.compare_methods \
    --lambda-dist "0,1,100" \
    --model "Model-Name" "0.85" "data/Model.csv" "checkpoints/Model_multi"

# 4. Check results
ls comparison_results/
```

**For OOD evaluation:**
```bash
# 1. Create category splits (one-time setup, already done)
python ood_evaluation/map_and_split_data.py

# 2. Run OOD evaluation (trains predictors on-the-fly)
python ood_evaluation/run_ood.py

# 3. Check results
ls ood_evaluation/results/  # Metrics, curves, plots per category
```

## Interactive Demo

**Location**: `demo/` directory

A Gradio-based web interface for testing CoRE routing in real-time:

### Running the Demo
```bash
# Set API key for OpenRouter (required for LLM inference)
export OPENROUTER_API_KEY="your-key-here"

# Launch demo
cd demo
python app.py

# Browser opens automatically at http://localhost:7860
```

### Demo Architecture
- **app.py**: Main Gradio interface with tabs for routing results and visualizations
- **config.py**: Configuration for LLM pool, API settings, checkpoints, and feature flags
- **router.py**: CoRE routing logic (loads sklearn predictors from checkpoints)
- **baselines.py**: CARROT-KNN and CARROT-Linear baseline routers
- **embedder.py**: Query embedding using vLLM or sentence-transformers
- **llm_client.py**: OpenRouter API client for LLM inference with token budgets
- **judge.py**: Automated quality evaluation using judge LLM (GPT-4o-mini or Qwen)
- **visualizer.py**: Plotly visualizations for cost-quality tradeoffs

### Demo Features
1. **Real-time Routing**: Enter a query and see CoRE select optimal (LLM, token_limit)
2. **Baseline Comparison**: Compare CoRE vs CARROT-KNN vs CARROT-Linear side-by-side
3. **Visualizations**: Interactive Plotly plots showing predicted cost-quality curves
4. **Input Transparency**: View exact prompts sent to each LLM (with token budget instructions)
5. **Automated Evaluation**: Judge LLM scores response quality (0-1 scale)
6. **Metrics Display**: Tokens used, actual cost, quality score, predicted risk

### Configuration Options (config.py)
- **LLM_POOL**: Models available for routing (currently 5 models: GLM-4.5-Air, Llama-3.1-70B, Llama-3.2-3B, Qwen3-235B, Qwen3-0.6B)
- **TOKEN_LIMITS**: 16 budget levels (10, 20, ..., 4000, unlimited)
- **EMBEDDING_MODEL**: "vllm" (default) or "sentence-transformers"
- **JUDGE_MODEL**: LLM for quality evaluation (default: Qwen3-Next-80B)
- **ENABLE_MOCK_MODE**: Set to True to test without GPU/API (uses dummy predictions)

### Requirements
- OpenRouter API key (for LLM inference)
- GPU with vLLM support (for embeddings) or sentence-transformers CPU fallback
- Pre-trained IID checkpoints in `checkpoints/` directory (trained on all 30,968 queries)

## Common Issues and Troubleshooting

### Missing Checkpoints
If you see errors about missing checkpoint files:
1. Check that checkpoints exist in `./checkpoints/{model-name}_multi/` (for sklearn) or `./checkpoints/{model-name}_1e4/` (for PyTorch)
2. Train predictors first using `predictor_sklearn.py` or `predictor.py`
3. For baselines, ensure `./checkpoints/carrot_knn/` and `./checkpoints/irt_*/` exist

### OOD Evaluation Setup
First-time OOD evaluation requires category splits:
```bash
python ood_evaluation/map_and_split_data.py  # Creates category_splits/
```

This creates `ood_evaluation/category_splits/ood_splits.pkl` with train/test indices for each category.

### Demo Configuration
If demo fails to start:
1. Set `ENABLE_MOCK_MODE = True` in `demo/config.py` to test without GPU/API
2. Verify `OPENROUTER_API_KEY` environment variable is set
3. Check that checkpoint paths in `demo/config.py` match your actual checkpoints
4. Ensure sentence-transformers is installed for CPU-based embeddings

### Predictor Type Mismatches
If you see errors loading predictors:
- Sklearn models use `.joblib` files in `{model}_multi/` directories
- PyTorch models use `.pt` files in `{model}_1e4/` directories
- Check the `PredictorClass` import in `results.py` matches your checkpoint type

### Data File Issues
All CSV files should have 30,968 rows. Check with:
```bash
python check_data.py  # Validates data structure
```

## File Organization Summary

**Complete pipeline (start here):**
- `main/run_experiment.sh` → Automated training + evaluation pipeline

**Main result package structure:**
```
main/
├── core/                    # CoRE predictor implementations
│   ├── predictor.py         # PyTorch neural network
│   ├── predictor_sklearn.py # Sklearn linear regression
│   └── train_core.py        # Training script
├── baselines/               # Baseline methods
│   ├── carrot/             # CARROT baselines
│   │   ├── baselines_carrot.py
│   │   └── train_carrot.py
│   └── irt/                # IRT baselines
│       ├── baselines_irt.py
│       └── train_irt.py
├── shared/                  # Shared utilities
│   ├── dataset_manager.py   # Train/test split management
│   ├── llm_loader.py        # LLM data loader
│   ├── router_dataset.py    # Dataset wrapper
│   └── utils.py             # Utility functions
└── evaluation/              # Evaluation scripts
    ├── compare_methods.py   # Method comparison
    └── results.py           # IID evaluation
```

**OOD evaluation:**
- `ood_evaluation/run_ood.py` → Out-of-distribution evaluation
- `ood_evaluation/ood_dataset_manager.py` → Category-based splits

**UniRouter comparison:**
- `unirouter/run_unirouter_experiment.sh` → UniRouter vs Uni-CoRE comparison
- `unirouter/eval_compare.py` → Evaluation script

**Interactive demo:**
- `demo/app.py` → Gradio interface
- `demo/config.py` → Configuration (modify this first!)
- `demo/router.py` → CoRE routing logic
- `demo/baselines.py` → Baseline routers
- `demo/llm_client.py` → OpenRouter API client
- `demo/judge.py` → Quality evaluation
- `demo/embedder.py` → Query embeddings
- `demo/visualizer.py` → Plotly visualizations

**Ablation studies:**
- `ablation/` → Embedding ablation experiments
  - Tests different embedding models (e.g., LightGBM, alternative embedders)
  - Separate from main evaluation pipeline
  - Run with `bash ablation/run_ablation_study.sh`

## Environment and Dependencies

**Python Environment:**
- Activate the virtual environment before running any scripts:
  ```bash
  source .venv/bin/activate
  ```

**Required packages:**
- Python 3.10
- PyTorch 2.0
- scikit-learn 1.3
- sentence-transformers
- pandas, numpy, matplotlib, seaborn
- tqdm, joblib
- gradio (for demo)
- plotly (for demo visualizations)

**Hardware:**
- Dataset collection: 8× NVIDIA B200 GPUs (for LLM inference)
- Router training: Single GPU sufficient (predictors are lightweight)
- Demo: GPU optional (can use sentence-transformers on CPU)
- Inference: vLLM 0.5.0 framework

**Reproducibility:**
- All experiments use fixed seed (42)
- Centralized DatasetManager ensures all models see identical splits
- Checkpoints provided for reproducibility

## Important Development Notes

### Recent Refactoring (See REFACTORING_STATUS.md)

The codebase has undergone API standardization to follow sklearn conventions:

**Baseline API Changes:**
- All baselines now use `fit(X_train, y_train)` and `predict(X_test)`
- Test data is no longer stored in baseline objects
- No evaluation/plotting during training (separation of concerns)
- Applied to: CARROT-KNN, CARROT-Linear, MIRT, NIRT

**Critical Bug Fixes:**
1. **Token Count Prediction**: Fixed unrealistic use of actual token counts in routing
   - Now uses: `min(token_limit, predicted_unlimited_count)` for limited budgets
   - Ensures realistic evaluation (router cannot know actual usage before inference)
   - Fixed in both `ood_evaluation/run_ood.py` and `main/shared/llm_loader.py`

2. **IRT Cost Calculation**: Fixed IRT baselines using true token counts
   - Now uses constant mean token count (cost ∝ model_size only)
   - Aligns with realistic routing scenario

**Impact**: These fixes resolved performance inversions and ensure consistent evaluation across IID and OOD pipelines.
