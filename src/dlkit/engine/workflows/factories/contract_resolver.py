"""Re-export resolve_contract from domain.nn.contract_resolver.

The implementation lives in ``domain.nn`` so that ``engine.inference`` can also
import it without creating an upward dependency on ``engine.workflows.factories``.
"""

from dlkit.domain.nn.contract_resolver import ContractInferenceError, resolve_contract

__all__ = [
    "ContractInferenceError",
    "resolve_contract",
]
