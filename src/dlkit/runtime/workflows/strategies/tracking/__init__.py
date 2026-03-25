"""Composable experiment tracking layer."""

from .artifact_logger import ArtifactLogger

# Internal services for TrackingDecorator (exported for testing)
from .config_accessor import ConfigAccessor
from .interfaces import IExperimentTracker, IRunContext, NullRunContext, NullTracker
from .metric_logger import MetricLogger
from .mlflow_tracker import MLflowTracker
from .naming import determine_experiment_name, determine_study_name
from .result_enricher import ResultEnricher
from .tracking_decorator import TrackingDecorator

__all__ = [
    # Public API
    "IExperimentTracker",
    "IRunContext",
    "NullTracker",
    "NullRunContext",
    "MLflowTracker",
    "TrackingDecorator",
    "determine_experiment_name",
    "determine_study_name",
    # Internal services (exported for testing and dependency injection)
    "ConfigAccessor",
    "MetricLogger",
    "ArtifactLogger",
    "ResultEnricher",
]
