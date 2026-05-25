"""Build strategy factories and contract resolution."""

from .contract_resolver import ContractInferenceError, resolve_contract

__all__ = ["ContractInferenceError", "resolve_contract"]
