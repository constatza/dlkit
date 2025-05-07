from typing import Literal

from pydantic import Field

from .base_settings import BaseSettings


class PrunerSettings(BaseSettings):
	name: str = Field(
		default='NopPruner',
		description='Pruner algorithm name for hyperparameter optimization.',
	)
	n_warmup_steps: int | None = Field(
		default=None, description='Number of warmup steps before pruning starts.'
	)
	interval_steps: int | None = Field(default=None, description='Interval between pruning steps.')


class SamplerSettings(BaseSettings):
	name: str = Field(
		default='TPESampler',
		description='Sampler algorithm name for hyperparameter optimization.',
	)
	seed: int | None = Field(default=None, description='Optional random seed for reproducibility.')


class OptunaSettings(BaseSettings):
	n_trials: int = Field(
		default=10, description='Number of trials for hyperparameter optimization.'
	)
	direction: Literal['minimize', 'maximize'] = Field(
		default='minimize',
		description='Optimization direction.',
	)
	sampler: SamplerSettings = Field(default=SamplerSettings(), description='Optuna Sampler.')
	pruner: PrunerSettings = Field(default=PrunerSettings(), description='Optuna Pruner.')
