"""Composable experiment tracking layer."""

from .interfaces import IExperimentTracker, IRunContext, NullTracker, NullRunContext
from .mlflow_tracker import MLflowTracker
from .tracking_decorator import TrackingDecorator
from .naming import determine_experiment_name, determine_study_name

# Internal services for TrackingDecorator (exported for testing)
from .config_accessor import ConfigAccessor
from .metric_logger import MetricLogger
from .artifact_logger import ArtifactLogger
from .result_enricher import ResultEnricher

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
