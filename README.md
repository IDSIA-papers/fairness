# On the Correlation between Individual Fairness and Predictive Accuracy in Probabilistic Models

A. Antonucci, E. Rossetto and I. Duvniak

A library for assessing algorithmic fairness in data-driven systems.

## Overview

Duvnii Fairness offers tools to evaluate and enhance the fairness of machine learning models. It includes metrics to detect bias, reporting functions, and utilities for preprocessing data for fair model training.

## Features

- Evaluate fairness on multiple dimensions.
- Integrate with popular machine learning frameworks.
- Generate bias reports and visualization.
- Easy-to-use API for custom evaluations.

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/yourusername/duvnii-fairness.git
    ```

2. Navigate to the project directory:

    ```
    cd duvnii-fairness
    ```

3. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

### Command Line

Run the main script to evaluate a model:

```
python main.py --data path/to/data.csv --model path/to/model.pkl
```

### As a Library

Import the main module in your Python script:

```python
from duvnii_fairness import fairness_evaluator

# Load data and model
data = load_data("path/to/data.csv")
model = load_model("path/to/model.pkl")

# Evaluate fairness
results = fairness_evaluator.evaluate(data, model)

# Print results
print(results)
```

## Configuration

Customize evaluation settings in the `config.yaml` file:

```yaml
evaluation:
  fairness_threshold: 0.05
  metrics:
     - demographic_parity
     - equal_opportunity
```

## Examples

Check the `examples/` folder for sample notebooks and scripts demonstrating how to integrate with your projects.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests. For major changes, open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

Happy Evaluating!
