# DLkit

**DLkit** is a supplementary module for vanilla [PyTorch](https://github.com/pytorch/pytorch) and [Lightning](https://github.com/Lightning-AI/pytorch-lightning) providing custom classes for common architectures and utility functions to streamline deep learning development.
It also includes a hyperparameter optimization module using [optuna](https://github.com/optuna/optuna) that integrates with [MLFlow](https://github.com/mlflow/mlflow) to track experiments, metrics, and models.

## Installation

**DLkit** requires Python 3.12+. 
You can install the package using either `uv` or `pip`.

### Using `uv`

Ensure that `uv` is installed on your system. For official installation instructions tailored to your platform, please refer to the [uv documentation](https://docs.astral.sh/uv).

~~~bash
uv add git+https://github.com/constatza/dlkit
~~~

## Using the API

To use the api, import the package and use the provided functions:
~~~python
from dlkit.io.settings import load_validated_settings
from dlkit.run.training import train

settings_path = "./config.toml"
settings = load_validated_settings(settings_path)
training_state = train(settings, mlflow=True)
~~~
The configuration file should include the following sections:
~~~toml
[model]
kernel_size = 5
hidden_size = 32
num_layers = 1
scalar = 0.01


[paths]
input_dir = "./data"
output_dir = "./results"
features = "./data/features.npy"
targets = "./data/targets.npy"
~~~

## Running Scripts

To execute the provided scripts, use:

### Training
Run the training process using the configuration specified in the configuration
~~~bash
uv run train path/to/config.toml
~~~
or with mlflow logging
```bash
uv run train --mlflow path/to/config.toml
```
This will automatically start the MLFlow server and log the training process.

### Hyperparameter Optimization
For fine-tuning the model using hyperparameter optimization, run
~~~bash
uv run train --optuna path/to/config.toml
~~~
the configuration should include a section for optuna e.g.:
~~~toml
[optuna]
n_trials = 100

[optuna.pruner]
name = "MedianPruner"
n_startup_trials=5
n_warmup_steps=30
interval_steps=10
~~~
and hyperparameter ranges for the model configuration e.g.:
~~~toml
[model]
kernel_size = {low=3, high=7, step=2}
hidden_size = {low=32, high=64}
num_layers = {choices=[1, 2, 3]}
scalar = {low=0.01, high=0.1}
~~~

### MLFlow Server
If you want to see the MLflow GUI, when server is not auto-started from training, run
~~~bash
uv run server path/to/config.toml
~~~
and open `http://{host}:{port}` where `host` and `port` are specified in the configuration:
~~~toml
[mlflow.server]
host = "localhost"
port = 5000
~~~


## Contributing

Contributions are welcome! Any suggestions or bug reports can be raised [here](https://github.com/constatza/dlkit/issues).

