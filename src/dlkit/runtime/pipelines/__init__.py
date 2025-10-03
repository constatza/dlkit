"""Data processing pipeline for model inference and dataflow flow.

This package provides a decoupled architecture for handling dataflow processing
separate from the dataset itself. It implements several design patterns:

- Chain of Responsibility: Processing pipeline steps
- Command Pattern: Model invocation
- Strategy Pattern: Output classification
- Template Method: Processing step template
- Dependency Inversion: Abstract interfaces

The processing pipeline handles the flow from raw batch dataflow through
model inference to classified outputs (latents, predictions) and
loss dataflow aggregation.
"""

from .interfaces import BatchProcessor, DataProvider, ModelInvoker
from .context import ProcessingContext

__all__ = [
    "BatchProcessor",
    "DataProvider",
    "ModelInvoker",
    "ProcessingContext",
]
