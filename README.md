# On the Correlation between Individual Fairness and Predictive Accuracy in Probabilistic Models

This repository contains the implementation and experimental code for the paper "On the Correlation between Individual Fairness and Predictive Accuracy in Probabilistic Models" by A. Antonucci*, E. Rossetto*, and I. Duvniak**.

*IDSIA - USI-SUPSI, Lugano, Switzerland
**SUPSI, Lugano, Switzerland

## Abstract

We investigate individual fairness in generative probabilistic classifiers by analysing the robustness of posterior inferences to perturbations in private features. Building on established results in robustness analysis, we hypothesise a correlation between robustness and predictive accuracy—specifically, instances exhibiting greater
robustness are more likely to be classified accurately. We empirically assess this hypothesis using a benchmark of eleven datasets with fairness concerns, employing Bayesian networks as the underlying generative models. To address the computational complexity associated with robustness analysis over multiple private features
with Bayesian networks, we reformulate the problem as a most probable explanation task in an auxiliary Markov random field. Our experiments confirm the hypothesis about the correlation, suggesting novel directions to mitigate the traditional trade-off between fairness and accuracy.

## Repository Structure Tree

```
├── pipeline.py            # Main pipeline for fairness analysis
├── pyproject.toml         # Project configuration and dependencies
├── bayesian/              # Bayesian network implementation
│   ├── inference.py        # Inference engine and posterior computation
│   ├── learn.py            # Bayesian network structure learning
│   └── modifiers.py        # Network modification utilities
├── datasets/              # Dataset handling and preprocessing
│   ├── data.py             # Data loading and feature extraction
│   ├── processing.py       # Data preprocessing utilities
│   └── utils.py            # Dataset utility functions
├── metrics/               # Fairness and performance metrics
│   ├── evaluate.py         # Model performance evaluation
│   └── fairness.py         # Individual and group fairness metrics
├── mrf/                   # Markov Random Field implementation
│   ├── inference/          # MRF inference algorithms
│   └── network/            # MRF network structures
├── visualization/         # Plotting and visualization utilities
├── data/                  # Data directory
│   └── preprocessed_data/  # Preprocessed datasets

```

Some comments on the organization of the code:

- The main pipeline is in `pipeline.py`, which orchestrates the entire fairness analysis process.
- The `bayesian/` directory contains some specific utilities for `pyagrum` Bayesian networks and additionally specific implementations deriving from the paper.
- The `datasets/` directory handles data loading, preprocessing, and feature extraction.
- The `metrics/` directory implements fairness metrics and model evaluation functions.
- The `mrf/` directory contains the implementation of a Variable Elimination algorithm for answering conditional, MAP, and MPE queries on Markov Random Fields, which is used to compute supposedly faster individual fairness metrics. It also contains the utility function that given a certain `pyagrum` Bayesian network, it builds a Markov Random Field using ratios, that can be used to compute individual fairness metrics.
- The `visualization/` directory provides plotting functions for visualizing results and metrics.

- The `data/` directory contains preprocessed datasets used in the experiments. In general, experiments run from the pipeline will save results in the `data/<dir_name>` directory, however this is not mandatory, and the user can specify a different directory to save results in the `pipeline.py` script.

## Installation

The installation is straightforward and can be done using either `uv` (recommended) or `pip`. The project uses `pyproject.toml` for dependency management (however a requirements.txt file is also provided for compatibility with older systems).

### UV

1. Install `uv` if you haven't already -> <https://docs.astral.sh/uv/getting-started/installation/>
2. Clone the repository
3. Navigate to the project directory and run:

    ```bash
    uv sync
    ```

    This will install all dependencies specified in `pyproject.toml` and set up the project environment in a local virtual environment placed in the `.venv` directory.

### Pip

1. Clone the repository
2. Navigate to the project directory and run:

    ```bash
    pip install -r ./requirements.txt
    ```

## Usage

### Quick Start

Run the main fairness analysis pipeline:

```bash
python pipeline.py
```

or if UV is installed:

```bash
uv run pipeline.py
```

### Advanced Usage

The main script supports several parameters:

```bash
python main.py \
    --learning_method tabu \
    --data_path ./data \
    --save_path ./data/<dir_name> \
    --drop_duplicates False
```

Available learning methods:

- `tabu`: Tabu search for structure learning
- `greedy`: Greedy search algorithm
- `miic`: MIIC algorithm
- `k2`: K2 algorithm

### Pipeline Overview

The main pipeline ([`main.py`](main.py)) performs the following steps:

1. **Data Loading**: Loads preprocessed datasets from [`data/preprocessed_data/`](data/preprocessed_data/)
2. **Preprocessing**: Converts continuous variables to categorical using [`datasets.processing.make_columns_categorical`](datasets/processing.py)
3. **Feature Extraction**: Identifies target, sensible, and public features using [`datasets.data.extract_features`](datasets/data.py)
4. **Network Learning**: Learns Bayesian network structure using [`bayesian.learn.learn_bayesian_network`](bayesian/learn.py)
5. **Fairness Analysis**:
   - Group fairness metrics via [`metrics.fairness.compute_group_fairness_metrics`](metrics/fairness.py)
   - Individual fairness metrics via [`metrics.fairness.compute_individual_fairness`](metrics/fairness.py)
   - MRF-based individual fairness via [`metrics.fairness.compute_individual_fairness_MRF`](metrics/fairness.py)
6. **Visualization**: Generates plots using functions from [`visualization.metrics`](visualization/metrics.py)

## Citation

...

## Dependencies

Main dependencies (see [`pyproject.toml`](pyproject.toml) for complete list):

- `loguru`: Logging
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `matplotlib`: Visualization
- `pyAgrum`: Bayesian networks
- `scikit-learn`: Machine learning utilities

## License

This project is licensed under the MIT License.

## Contact

For questions about the implementation or paper:

- Alessandro Antonucci - [alessandro.antonucci@idsia.ch](mailto:)
- Eric Rossetto - [eric.rossetto@idsia.ch](mailto:)
- Ivan Duvniak
