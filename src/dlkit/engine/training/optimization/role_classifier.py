"""Graph-based parameter role classification."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Node, Tracer

from dlkit.domain.nn.parameter_roles import ParameterRole

_MISSING = object()
_SUPPORTED_FUNCTIONS = {
    F.linear,
    F.bilinear,
    F.conv1d,
    F.conv2d,
    F.conv3d,
    F.embedding,
}


def classify_parameter_roles(model: nn.Module) -> dict[str, ParameterRole]:
    """Classify all trainable parameters in a model."""
    return GraphParameterRoleClassifier().classify(model)


@dataclass(frozen=True, slots=True)
class _DirectParameter:
    """Direct parameter owned by a module."""

    local_name: str
    canonical_name: str
    parameter: nn.Parameter


@dataclass(frozen=True, slots=True)
class _Site:
    """Parameterized execution site extracted from the FX graph."""

    site_id: int
    node: Node
    kind: str
    label: str
    parameter_names: tuple[str, ...]
    parameter_ids: tuple[int, ...]
    bias_parameter_names: tuple[str, ...] = ()


class _ModuleStructure:
    """View over module ownership and canonical parameter names."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self.canonical_name_by_id = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        self.direct_parameters_by_module = self._collect_direct_parameters()
        self.module_paths_by_parameter_id = self._collect_module_paths_by_parameter()
        self.modules_with_parameter_descendants = self._collect_modules_with_parameter_descendants()
        self.parameterized_leaf_modules = self._collect_parameterized_leaf_modules()

    def _collect_direct_parameters(self) -> dict[str, tuple[_DirectParameter, ...]]:
        direct_parameters: dict[str, tuple[_DirectParameter, ...]] = {}

        for module_path, module in self._model.named_modules():
            owned_parameters: list[_DirectParameter] = []
            for local_name, parameter in module.named_parameters(
                recurse=False,
                remove_duplicate=False,
            ):
                if not parameter.requires_grad:
                    continue
                canonical_name = self.canonical_name_by_id.get(id(parameter))
                if canonical_name is None:
                    continue
                owned_parameters.append(
                    _DirectParameter(
                        local_name=local_name,
                        canonical_name=canonical_name,
                        parameter=parameter,
                    )
                )
            direct_parameters[module_path] = tuple(owned_parameters)

        return direct_parameters

    def _collect_module_paths_by_parameter(self) -> dict[int, set[str]]:
        module_paths_by_parameter: dict[int, set[str]] = defaultdict(set)

        for module_path, owned_parameters in self.direct_parameters_by_module.items():
            for owned_parameter in owned_parameters:
                module_paths_by_parameter[id(owned_parameter.parameter)].add(module_path)

        return module_paths_by_parameter

    def _collect_modules_with_parameter_descendants(self) -> set[str]:
        modules_with_direct_parameters = {
            module_path
            for module_path, owned_parameters in self.direct_parameters_by_module.items()
            if owned_parameters
        }
        descendant_modules: set[str] = set()

        for module_path in self.direct_parameters_by_module:
            has_descendant = any(
                other_path != module_path and other_path.startswith(_child_prefix(module_path))
                for other_path in modules_with_direct_parameters
            )
            if has_descendant:
                descendant_modules.add(module_path)

        return descendant_modules

    def _collect_parameterized_leaf_modules(self) -> set[str]:
        leaf_modules: set[str] = set()

        for module_path, owned_parameters in self.direct_parameters_by_module.items():
            if not owned_parameters:
                continue
            if module_path in self.modules_with_parameter_descendants:
                continue
            leaf_modules.add(module_path)

        return leaf_modules

    @property
    def shared_parameter_ids(self) -> set[int]:
        return {
            parameter_id
            for parameter_id, module_paths in self.module_paths_by_parameter_id.items()
            if len(module_paths) > 1
        }


class _ParameterizedLeafTracer(Tracer):
    """Tracer that preserves only fundamental parameter-owning execution modules."""

    def __init__(self, structure: _ModuleStructure) -> None:
        super().__init__()
        self._structure = structure

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if module_qualified_name in self._structure.parameterized_leaf_modules:
            return True
        if module_qualified_name in self._structure.modules_with_parameter_descendants:
            return False
        return super().is_leaf_module(m, module_qualified_name)


class GraphParameterRoleClassifier:
    """Classify parameters by structure first and forward-graph position second."""

    def classify(self, model: nn.Module) -> dict[str, ParameterRole]:
        structure = _ModuleStructure(model)
        roles = {name: ParameterRole.UNKNOWN for name, _ in model.named_parameters()}
        self._assign_structural_roles(model, structure, roles)

        traced = self._trace_model(model, structure)
        if traced is None:
            self._apply_shared_parameter_fallback(structure, roles, set())
            return roles

        sites, unsupported_names = self._extract_sites(model, traced, structure)
        self._apply_site_roles(traced, structure, roles, sites, unsupported_names)
        return roles

    def _assign_structural_roles(
        self,
        model: nn.Module,
        structure: _ModuleStructure,
        roles: dict[str, ParameterRole],
    ) -> None:
        for module_path, module in model.named_modules():
            owned_parameters = structure.direct_parameters_by_module[module_path]
            if not owned_parameters:
                continue

            match module:
                case nn.Embedding() | nn.EmbeddingBag():
                    self._assign_role(roles, owned_parameters, ParameterRole.EMBEDDING)
                    continue
                case _ if isinstance(module, _normalization_module_types()):
                    self._assign_role(roles, owned_parameters, ParameterRole.NORMALIZATION)
                    continue
                case _:
                    pass

            bias_parameter = getattr(module, "bias", None)
            if not isinstance(bias_parameter, nn.Parameter):
                continue

            for owned_parameter in owned_parameters:
                if owned_parameter.local_name != "bias":
                    continue
                if owned_parameter.parameter is not bias_parameter:
                    continue
                roles[owned_parameter.canonical_name] = ParameterRole.BIAS

    def _assign_role(
        self,
        roles: dict[str, ParameterRole],
        owned_parameters: tuple[_DirectParameter, ...],
        role: ParameterRole,
    ) -> None:
        for owned_parameter in owned_parameters:
            roles[owned_parameter.canonical_name] = role

    def _trace_model(
        self,
        model: nn.Module,
        structure: _ModuleStructure,
    ) -> GraphModule | None:
        tracer = _ParameterizedLeafTracer(structure)
        try:
            graph = tracer.trace(model)
        except Exception:
            return None
        return GraphModule(model, graph)

    def _extract_sites(
        self,
        model: nn.Module,
        traced: GraphModule,
        structure: _ModuleStructure,
    ) -> tuple[tuple[_Site, ...], set[str]]:
        sites: list[_Site] = []
        unsupported_names: set[str] = set()

        for node in traced.graph.nodes:
            match node.op:
                case "call_module":
                    site = self._site_from_module_node(model, node, structure, len(sites))
                    if site is None:
                        continue
                    sites.append(site)
                case "call_function":
                    function_target = node.target
                    if function_target not in _SUPPORTED_FUNCTIONS:
                        continue
                    site = self._site_from_function_node(model, node, len(sites))
                    if site is None:
                        unsupported_names.update(
                            self._collect_parameter_names_from_args(model, node)
                        )
                        continue
                    sites.append(site)
                case _:
                    continue

        return tuple(sites), unsupported_names

    def _site_from_module_node(
        self,
        model: nn.Module,
        node: Node,
        structure: _ModuleStructure,
        site_id: int,
    ) -> _Site | None:
        module_path = str(node.target)
        owned_parameters = structure.direct_parameters_by_module.get(module_path)
        if not owned_parameters:
            return None

        parameter_names = tuple(
            owned_parameter.canonical_name for owned_parameter in owned_parameters
        )
        parameters = tuple(
            model.get_parameter(parameter_name) for parameter_name in parameter_names
        )
        return _Site(
            site_id=site_id,
            node=node,
            kind="module",
            label=module_path,
            parameter_names=parameter_names,
            parameter_ids=tuple(id(parameter) for parameter in parameters),
            bias_parameter_names=tuple(
                owned_parameter.canonical_name
                for owned_parameter in owned_parameters
                if owned_parameter.local_name == "bias"
            ),
        )

    def _site_from_function_node(
        self,
        model: nn.Module,
        node: Node,
        site_id: int,
    ) -> _Site | None:
        target = node.target
        match target:
            case _ if target is F.linear:
                resolved = self._resolve_linear_parameters(model, node)
            case _ if target is F.bilinear:
                resolved = self._resolve_bilinear_parameters(model, node)
            case _ if target in {F.conv1d, F.conv2d, F.conv3d}:
                resolved = self._resolve_conv_parameters(model, node)
            case _ if target is F.embedding:
                resolved = self._resolve_embedding_parameters(model, node)
            case _:
                return None

        if resolved is None:
            return None

        parameter_names, bias_parameter_names = resolved
        parameters = tuple(
            model.get_parameter(parameter_name) for parameter_name in parameter_names
        )
        return _Site(
            site_id=site_id,
            node=node,
            kind="function",
            label=getattr(target, "__name__", repr(target)),
            parameter_names=parameter_names,
            parameter_ids=tuple(id(parameter) for parameter in parameters),
            bias_parameter_names=bias_parameter_names,
        )

    def _resolve_linear_parameters(
        self,
        model: nn.Module,
        node: Node,
    ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        weight_name = self._required_parameter_name(model, self._arg_or_kw(node, 1, "weight"))
        if weight_name is None:
            return None

        bias_name = self._optional_parameter_name(model, self._arg_or_kw(node, 2, "bias"))
        if bias_name is _MISSING:
            return None
        if bias_name is None:
            return ((weight_name,), ())
        if not isinstance(bias_name, str):
            return None
        return ((weight_name, bias_name), (bias_name,))

    def _resolve_bilinear_parameters(
        self,
        model: nn.Module,
        node: Node,
    ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        weight_name = self._required_parameter_name(model, self._arg_or_kw(node, 2, "weight"))
        if weight_name is None:
            return None

        bias_name = self._optional_parameter_name(model, self._arg_or_kw(node, 3, "bias"))
        if bias_name is _MISSING:
            return None
        if bias_name is None:
            return ((weight_name,), ())
        if not isinstance(bias_name, str):
            return None
        return ((weight_name, bias_name), (bias_name,))

    def _resolve_conv_parameters(
        self,
        model: nn.Module,
        node: Node,
    ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        weight_name = self._required_parameter_name(model, self._arg_or_kw(node, 1, "weight"))
        if weight_name is None:
            return None

        bias_name = self._optional_parameter_name(model, self._arg_or_kw(node, 2, "bias"))
        if bias_name is _MISSING:
            return None
        if bias_name is None:
            return ((weight_name,), ())
        if not isinstance(bias_name, str):
            return None
        return ((weight_name, bias_name), (bias_name,))

    def _resolve_embedding_parameters(
        self,
        model: nn.Module,
        node: Node,
    ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        weight_name = self._required_parameter_name(model, self._arg_or_kw(node, 1, "weight"))
        if weight_name is None:
            return None
        return ((weight_name,), ())

    def _required_parameter_name(self, model: nn.Module, argument: object) -> str | None:
        if not isinstance(argument, Node):
            return None
        if argument.op != "get_attr":
            return None
        return self._parameter_name_from_attr_node(model, argument)

    def _optional_parameter_name(self, model: nn.Module, argument: object) -> str | object | None:
        if argument is _MISSING:
            return None
        if argument is None:
            return None
        if not isinstance(argument, Node):
            return _MISSING
        if argument.op != "get_attr":
            return _MISSING
        return self._parameter_name_from_attr_node(model, argument)

    def _parameter_name_from_attr_node(self, model: nn.Module, node: Node) -> str | None:
        try:
            parameter = model.get_parameter(str(node.target))
        except AttributeError:
            return None
        for name, candidate in model.named_parameters():
            if candidate is parameter:
                return name
        return None

    def _arg_or_kw(self, node: Node, position: int, key: str) -> object:
        if key in node.kwargs:
            return node.kwargs[key]
        if len(node.args) > position:
            return node.args[position]
        return _MISSING

    def _collect_parameter_names_from_args(self, model: nn.Module, node: Node) -> set[str]:
        parameter_names: set[str] = set()
        worklist = deque(self._iter_argument_nodes(node.args))
        worklist.extend(self._iter_argument_nodes(node.kwargs.values()))
        visited: set[Node] = set()

        while worklist:
            current = worklist.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current.op == "get_attr":
                parameter_name = self._parameter_name_from_attr_node(model, current)
                if parameter_name is not None:
                    parameter_names.add(parameter_name)
                continue

            worklist.extend(current.all_input_nodes)

        return parameter_names

    def _iter_argument_nodes(self, values: Iterable[Any]) -> Iterable[Node]:
        for value in values:
            yield from self._nodes_in_value(value)

    def _nodes_in_value(self, value: Any) -> Iterable[Node]:
        if isinstance(value, Node):
            yield value
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                yield from self._nodes_in_value(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from self._nodes_in_value(item)

    def _apply_site_roles(
        self,
        traced: GraphModule,
        structure: _ModuleStructure,
        roles: dict[str, ParameterRole],
        sites: tuple[_Site, ...],
        unsupported_names: set[str],
    ) -> None:
        graph_roles = self._classify_sites(traced, sites)
        shared_parameter_names = {
            parameter_name
            for site in sites
            for parameter_name, parameter_id in zip(
                site.parameter_names,
                site.parameter_ids,
                strict=True,
            )
            if sum(parameter_id in candidate.parameter_ids for candidate in sites) > 1
        }

        for parameter_name in unsupported_names | shared_parameter_names:
            roles[parameter_name] = ParameterRole.UNKNOWN

        for site in sites:
            for parameter_name in site.bias_parameter_names:
                if parameter_name in unsupported_names:
                    continue
                if parameter_name in shared_parameter_names:
                    continue
                roles[parameter_name] = ParameterRole.BIAS

        for site in sites:
            site_role = graph_roles[site.site_id]
            for parameter_name in site.parameter_names:
                if parameter_name in unsupported_names:
                    continue
                if parameter_name in shared_parameter_names:
                    continue
                if parameter_name in site.bias_parameter_names:
                    continue
                if roles[parameter_name] is not ParameterRole.UNKNOWN:
                    continue
                roles[parameter_name] = site_role

        self._apply_shared_parameter_fallback(
            structure,
            roles,
            shared_parameter_names,
        )

    def _classify_sites(
        self,
        traced: GraphModule,
        sites: tuple[_Site, ...],
    ) -> dict[int, ParameterRole]:
        if not sites:
            return {}

        site_ids_by_node = {site.node: site.site_id for site in sites}
        nodes_reachable_from_input = self._nodes_reachable_from_placeholders(traced)
        nodes_reaching_output = self._nodes_reaching_output(traced)
        successors_by_site = {
            site.site_id: self._next_site_ids(site.node, site_ids_by_node) for site in sites
        }
        predecessors_by_site = defaultdict(set)

        for site_id, successors in successors_by_site.items():
            for successor_id in successors:
                predecessors_by_site[successor_id].add(site_id)

        roles: dict[int, ParameterRole] = {}

        for site in sites:
            if site.node not in nodes_reachable_from_input:
                roles[site.site_id] = ParameterRole.UNKNOWN
                continue
            if site.node not in nodes_reaching_output:
                roles[site.site_id] = ParameterRole.UNKNOWN
                continue

            predecessors = predecessors_by_site.get(site.site_id, set())
            successors = successors_by_site[site.site_id]

            if not predecessors and not successors:
                roles[site.site_id] = ParameterRole.OUTPUT
                continue
            if not predecessors:
                roles[site.site_id] = ParameterRole.INPUT
                continue
            if not successors:
                roles[site.site_id] = ParameterRole.OUTPUT
                continue
            roles[site.site_id] = ParameterRole.HIDDEN

        return roles

    def _nodes_reachable_from_placeholders(self, traced: GraphModule) -> set[Node]:
        queue = deque(node for node in traced.graph.nodes if node.op == "placeholder")
        visited: set[Node] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            queue.extend(current.users)

        return visited

    def _nodes_reaching_output(self, traced: GraphModule) -> set[Node]:
        output_node = next(node for node in traced.graph.nodes if node.op == "output")
        queue = deque([output_node])
        visited: set[Node] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            queue.extend(current.all_input_nodes)

        return visited

    def _next_site_ids(
        self,
        start_node: Node,
        site_ids_by_node: dict[Node, int],
    ) -> set[int]:
        queue = deque(start_node.users)
        visited: set[Node] = set()
        successor_ids: set[int] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current in site_ids_by_node:
                successor_ids.add(site_ids_by_node[current])
                continue

            queue.extend(current.users)

        return successor_ids

    def _apply_shared_parameter_fallback(
        self,
        structure: _ModuleStructure,
        roles: dict[str, ParameterRole],
        shared_parameter_names: set[str],
    ) -> None:
        for parameter_id in structure.shared_parameter_ids:
            parameter_name = structure.canonical_name_by_id.get(parameter_id)
            if parameter_name is None:
                continue
            roles[parameter_name] = ParameterRole.UNKNOWN

        for parameter_name in shared_parameter_names:
            roles[parameter_name] = ParameterRole.UNKNOWN


def _child_prefix(module_path: str) -> str:
    """Return the descendant prefix for a module path."""
    if not module_path:
        return ""
    return f"{module_path}."


def _normalization_module_types() -> tuple[type[nn.Module], ...]:
    """Return the supported normalization module types."""
    module_types: list[type[nn.Module]] = [
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LocalResponseNorm,
    ]
    rms_norm = getattr(nn, "RMSNorm", None)
    if rms_norm is not None:
        module_types.append(rms_norm)
    return tuple(module_types)
