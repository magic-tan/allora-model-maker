# Allora Model Maker

## Overview

Allora Model Maker is a comprehensive machine learning framework designed for time series forecasting, specifically optimized for financial market data like cryptocurrency prices, stock prices, and more. It supports multiple models, including traditional statistical models like ARIMA and machine learning models like LSTM, XGBoost, and more.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Model Training](#model-training)
  - [Model & Metric Testing](#model-testing)
  - [Model Inference](#model-inference)
  - [Model Forecasting](#model-forecasting)
  - [Metrics Calculation](#metrics-calculation)
- [Supported Models](#supported-models)
- [Configuration](#configuration)
  - [Model Configurations](#model-configurations)
  - [Interval Configuration](#interval-configuration)
- [Directory Structure](#directory-structure)
- [Makefile](#makefile)
- [Packaging Models for Allora Worker](#packaging-models-for-allora-worker)
- [Contributing](#contributing)
- [License](#license)

## Features
- Multiple model support (ARIMA, LSTM, XGBoost, Random Forest, etc.)
- Configurable time intervals (`5M`, `H`, `D`, etc.) for time series data
- Built-in metrics for evaluating model performance (e.g., CAGR, Sortino Ratio, Expected Shortfall)
- Easy model saving and loading mechanism
- Scalable for large datasets

## Installation

1. Clone the repository:
    ```bash
   git clone https://github.com/allora-network/allora-model-maker.git
   cd allora-model-maker
    ```

2. Create a conda environment:
    ```bash
   conda create --name modelmaker python=3.9
   conda activate modelmaker
    ```

3. Install dependencies:
    ```bash
   pip install -r requirements.txt
    ```

## Usage

### Model Training

You can train models by running the `train.py` script. It supports multiple model types and interval resampling.

Example for training:
 ```bash
python train.py
 ```

### Model Testing

You can test models by running the `test.py` script. It supports multiple model types and metrics.

Example for testing:
 ```bash
python test.py
 ```

During runtime, you will be prompted to select if you want to test models, metrics or both. The testing data is currently synthetic.

### Model Inference

To make predictions using a trained model, you can use the `inference()` method on the desired model.

Example:
 ```python
from models.lstm.model import LstmModel
model = LstmModel()
predictions = model.inference(input_data)
print(predictions)
 ```

### Model Forecasting

Forecast future data based on past time series data using the `forecast()` method:

 ```python
forecast_data = model.forecast(steps=10, last_known_data=input_data)
print(forecast_data)
 ```

### Metrics Calculation

Metrics can be calculated using the provided `metrics` module:

 ```python
from metrics.sortino_ratio.metric import SortinoRatioMetric
metric = SortinoRatioMetric()
result = metric.calculate(input_data)
print(result)
 ```

## Supported Models

The following models are supported out-of-the-box:
- **ARIMA**: Auto-Regressive Integrated Moving Average
- **LSTM**: Long Short-Term Memory Networks
- **Random Forest**: Random Forest for time series and regression
- **XGBoost**: Gradient Boosting for time series and regression
- **Prophet**: Facebook's Prophet for time series forecasting
- **Regression**: Basic regression models

## Configuration

### Model Configurations

Each model has its own configuration class located in its corresponding folder. For example, `LstmConfig` can be found in `models/lstm/configs.py`. Configurations include parameters for training, architecture, and data preprocessing.

You can modify configurations as needed:
 ```python
config = LstmConfig()
config.learning_rate = 0.001
 ```

### Interval Configuration

By default, the system uses daily (`D`) intervals for time series resampling. You can modify this in the configuration files for each model by setting the `interval` parameter (e.g., `5M`, `H`, `D`, etc.).

## Directory Structure

 ```
allora-model-maker/
│
├── models/                   # Directory containing different models
│   ├── lstm/                  # LSTM model files
│   ├── arima/                 # ARIMA model files
│   ├── random_forest/         # Random Forest model files
│   └── ...                    # Other model directories
│
├── metrics/                   # Metrics for evaluating model performance
│   ├── cagr/                  # CAGR metric implementation
│   ├── sortino_ratio/         # Sortino Ratio metric implementation
│   └── ...                    # Other metrics
│
├── data/                      # Data loading and preprocessing utilities
│
├── configs/                   # Global configuration files
│
├── train.py                   # Script for training models
├── test.py                    # Script for testing models and metrics
├── package_model_worker.py    # Script for packaging a model for allora worker
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── Makefile                   # Makefile for streamlined operations
 ```

## Makefile

This project includes a Makefile to simplify common tasks such as linting, formatting, testing, and running scripts. Below are the available commands:

#### Lint

 ```
make lint
 ```

This command runs pylint on all Python files in the project to check for coding errors, stylistic errors, and other issues. It will scan through all .py files.

#### Format

 ```
make format
 ```

This command formats all Python files using black, a widely used code formatter. It automatically reformats code to follow the best practices and standards.

#### Test

 ```
make test
 ```

This command runs the unit tests using pytest. By default, it will search for tests under the tests/ directory.

#### Clean

 ```
make clean
 ```

This command removes common build artifacts and directories, such as Python caches, test logs, and generated model files. Specifically, it will remove:

	•	__pycache__
	•	.pytest_cache
	•	.coverage
	•	trained_models/
	•	packaged_models/
	•	logs/
	•	test_results/

#### Run Training

 ```
make train
 ```

This command executes the training script train.py and starts the model training process.

#### Run Inference Tests

 ```
make eval
 ```

This command runs the script test.py, allowing you to test the model prediction and validation.

#### Format

 ```
make format
 ```

This command will format the entire codebase using black. Use this before committing code to ensure consistency and readability.

#### Package

 ```
make package-[model name]
 ```

This command will package your model for use in an allora worker, remember to replace [model name] with your actual model, ex: "arima"


## Packaging Models for Allora Worker

The purpose of the package_model_worker.py script is to export a trained model along with its configurations in a format that can be deployed into the allora-worker repository.

This script packages the model files, dependencies, and configuration into a structure that allows seamless integration with the allora-worker setup.

#### How to Use

Run the following command to package your model for the Allora worker:

``` make package-arima ```

Replace arima with the name of the model you’d like to package (e.g., lstm, arima, etc.).

This will:

	•	Copy the model’s files and dependencies into the packaged_models/package folder.
    •	Run test's for inference and training to validate funtionality in a worker
	•	Generate a configuration file, config.py, that contains the active model information.

#### Next Steps

After running the packaging command:

	1.	Navigate to the packaged_models folder in your allora-model-maker repo.
	2.	Copy the package folder and the config.py file to the root folder of your allora-worker repository.

By following these steps, your packaged model will be ready for deployment in the allora-worker environment.


## Contributing

Contributions are welcome! To ensure a smooth contribution process, please follow these steps:

1. **Fork the repository.**
2. **Create a new branch:**
    ```git checkout -b feature-branch ```
3. **Make your changes.**
4. **Before committing your changes, run the following Makefile commands to ensure code quality and consistency:**

   - **Lint your code:**
      ```make lint ```

   - **Format your code:**
      ```make format ```

   - **Run tests to ensure everything works:**
      ```make test ```

5. **Commit your changes:**
    ```git commit -am 'Add new feature' ```

6. **Push to your branch:**
    ```git push origin feature-branch ```

7. **Create a pull request.**

### General Best Practices

- Ensure your code follows the project's coding standards by using `pylint` for linting and `black` for formatting.
- Test your changes thoroughly before pushing by running the unit tests.
- Use meaningful commit messages that clearly describe your changes.
- Make sure your branch is up-to-date with the latest changes from the main branch.
- Avoid including unnecessary files in your pull request, such as compiled or cache files. The `make clean` command can help with that:

   ```make clean ```

By following these practices, you help maintain the quality and consistency of the project.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
