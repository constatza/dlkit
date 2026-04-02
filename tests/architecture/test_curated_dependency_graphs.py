from __future__ import annotations

import json
import subprocess
import sys
from collections import deque
from pathlib import Path


def _load_tach_map(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    map_path = tmp_path / "tach-map.json"
    subprocess.run(
        ["tach", "map", "-o", str(map_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    json.loads(map_path.read_text())
    return map_path


def _render_dot(
    map_path: Path,
    output_path: Path,
    *,
    root_module: str,
    context_modules: list[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "scripts/render_tach_dependency_graph.py",
        str(map_path),
        str(output_path),
        root_module,
    ]
    if context_modules:
        command.append(",".join(context_modules))

    subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def _parse_dot(dot_path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()

    for raw_line in dot_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line in {"digraph {", "}"}:
            continue

        if "->" in line and "[style=" not in line:
            source, target = line.rstrip(";").split(" -> ")
            edges.add((source.strip('"'), target.strip('"')))
            continue

        if line.startswith('"') and "[label=" in line:
            node_name = line.split('"', 2)[1]
            nodes.add(node_name)

    return nodes, edges


def _assert_acyclic(nodes: set[str], edges: set[tuple[str, str]], *, graph_name: str) -> None:
    adjacency = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}

    for source, target in edges:
        if source not in nodes or target not in nodes:
            continue
        if target in adjacency[source]:
            continue
        adjacency[source].add(target)
        indegree[target] += 1

    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in sorted(adjacency[node]):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    assert visited == len(nodes), f"Curated dependency view for {graph_name!r} contains a cycle"


def test_curated_dependency_graphs_are_acyclic(tmp_path: Path) -> None:
    map_path = _load_tach_map(tmp_path)

    curated_roots = {
        "dlkit": [],
        "dlkit.shared": [],
        "dlkit.tools": ["dlkit.shared"],
        "dlkit.domain": ["dlkit.shared"],
        "dlkit.runtime": ["dlkit.shared", "dlkit.tools", "dlkit.domain"],
        "dlkit.interfaces": ["dlkit.shared", "dlkit.tools", "dlkit.domain", "dlkit.runtime"],
    }

    for root_module, context_modules in curated_roots.items():
        dot_path = tmp_path / f"{root_module.replace('.', '_')}.dot"
        _render_dot(
            map_path,
            dot_path,
            root_module=root_module,
            context_modules=context_modules,
        )
        nodes, edges = _parse_dot(dot_path)
        _assert_acyclic(nodes, edges, graph_name=root_module)
