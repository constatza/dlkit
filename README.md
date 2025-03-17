# DLkit

**DLkit** is a supplementary module for vanilla [PyTorch](https://github.com/pytorch/pytorch) and [Lightning](https://github.com/Lightning-AI/pytorch-lightning) providing custom classes for common architectures and utility functions to streamline deep learning development.
It also includes a hyperparameter optimization module using [optuna](https://github.com/optuna/optuna) that integrates with [MLFlow](https://github.com/mlflow/mlflow) to track experiments, metrics, and models.

## Installation

**DLkit** requires Python 3.12. 
You can install the package using either `uv` or `pip`.

### Using `uv`

1. **Install `uv`**:

   Ensure that `uv` is installed on your system. For official installation instructions tailored to your platform, please refer to the [uv documentation](https://docs.astral.sh/uv).

2. **Clone the Repository**:

   ```bash
   git clone https://github.com/constatza/dlkit.git
   ```

3. **Navigate to the Project Directory**:

   ```bash
   cd dlkit
   ```

4. **Install Dependencies**:

   ```bash
   uv sync
   ```

   This command will install all the dependencies specified in the `pyproject.toml` and `uv.lock` files.

### Using `pip`

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/constatza/dlkit.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd dlkit
   ```

3. **Create a Virtual Environment**:

   It is recommended to create a virtual environment to manage your dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install Dependencies**:

   ```bash
   pip install .
   ```

   This command will install all the dependencies listed in the `pyproject.toml` file.
   Use the `-e` flag to install the package in editable mode.

## Running Scripts

To execute the provided scripts, use:

### MLFlow Server
Start the MLFlow server using the configuration specified in the configuration (required for training and optimization)
```bash
uv run start-server path/to/config.toml
```
### Training
Run the training process using the configuration specified in the configuration
```bash
uv run train path/to/config.toml
```
or with mlflow logging
```bash
uv run train path/to/config.toml
```
Note: MLflow training requires mlflow server to be running
```bash
uv run server path/to/config.toml
```

### Hyperparameter Optimization
Run the hyperparameter optimization process using the configuration specified in the configuration
```bash
uv run hopt path/to/config.toml
```


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

