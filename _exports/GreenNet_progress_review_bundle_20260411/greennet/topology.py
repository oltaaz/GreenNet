"""Build and load NetworkX graph topologies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - optional dependency
    nx = None
    _nx_import_error = exc
else:
    _nx_import_error = None

# networkx is an optional dependency. We avoid referencing the runtime `nx` variable in
# type annotations because Pylance flags it as an invalid type form.
GraphT = Any
NAMED_TOPOLOGY_DIR = Path(__file__).resolve().parent / "data" / "topologies"


class TopologyValidationError(ValueError):
    """Raised when a topology input file is malformed or unsupported."""


@dataclass(frozen=True)
class NamedTopologyOption:
    """Named packaged topology exposed through the public selection interface."""

    name: str
    file_name: str
    topology_class: str
    description: str
    aliases: tuple[str, ...] = ()


@dataclass
class TopologyConfig:
    """Configuration for generating a topology graph."""

    node_count: int = 10
    edge_prob: float = 0.2
    directed: bool = False
    seed: int | None = None
    topology_name: str | None = None
    topology_path: str | None = None


OFFICIAL_NAMED_TOPOLOGIES: tuple[NamedTopologyOption, ...] = (
    NamedTopologyOption(
        name="small",
        file_name="regional_ring.json",
        topology_class="small",
        description="Six-node regional ring with two higher-capacity cross links.",
        aliases=("regional_ring",),
    ),
    NamedTopologyOption(
        name="medium",
        file_name="metro_hub.json",
        topology_class="medium",
        description="Eight-node metro core with diagonals and access spurs.",
        aliases=("metro_hub",),
    ),
    NamedTopologyOption(
        name="large",
        file_name="backbone_large.json",
        topology_class="large",
        description="Twelve-node backbone with a meshed core and four access spurs.",
        aliases=("backbone_large",),
    ),
)

_NAMED_TOPOLOGY_BY_NAME = {option.name: option for option in OFFICIAL_NAMED_TOPOLOGIES}
_NAMED_TOPOLOGY_BY_ALIAS = {
    alias: option for option in OFFICIAL_NAMED_TOPOLOGIES for alias in option.aliases
}


def build_random_topology(config: TopologyConfig) -> GraphT:
    """Create a random topology based on an Erdos-Renyi model."""
    if nx is None:
        raise ImportError("networkx is required to build a topology") from _nx_import_error

    graph = nx.erdos_renyi_graph(
        n=config.node_count,
        p=config.edge_prob,
        seed=config.seed,
        directed=bool(config.directed),
    )

    # Force plain Graph/DiGraph (keeps things predictable)
    graph = nx.DiGraph(graph) if bool(config.directed) else nx.Graph(graph)
    return graph


def list_named_topologies() -> list[str]:
    """Return accepted packaged topology selectors, including compatibility aliases."""
    names = {option.name for option in OFFICIAL_NAMED_TOPOLOGIES}
    names.update(alias for option in OFFICIAL_NAMED_TOPOLOGIES for alias in option.aliases)
    if NAMED_TOPOLOGY_DIR.exists():
        names.update(path.stem for path in NAMED_TOPOLOGY_DIR.glob("*.json"))
    return sorted(names)


def list_official_topology_classes() -> list[str]:
    """Return the stable named topology classes recommended in docs and configs."""
    return [option.name for option in OFFICIAL_NAMED_TOPOLOGIES]


def describe_named_topologies() -> list[dict[str, str]]:
    """Return metadata for the stable packaged topology options."""
    return [
        {
            "name": option.name,
            "file_name": option.file_name,
            "topology_class": option.topology_class,
            "description": option.description,
            "aliases": ", ".join(option.aliases),
        }
        for option in OFFICIAL_NAMED_TOPOLOGIES
    ]


def load_topology_from_edges(
    edges: Iterable[tuple[int, int]], *, directed: bool = False
) -> GraphT:
    """Load a topology graph from an edge list."""
    if nx is None:
        raise ImportError("networkx is required to load a topology") from _nx_import_error

    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from(edges)
    return graph


def build_topology(config: TopologyConfig) -> GraphT:
    """Build either a random topology or load a packaged/custom topology."""
    if config.topology_path:
        return load_topology_from_file(config.topology_path)
    if config.topology_name:
        return load_named_topology(config.topology_name)
    return build_random_topology(config)


def load_named_topology(name: str) -> GraphT:
    """Load a packaged topology by name."""
    normalized = str(name or "").strip()
    if not normalized:
        raise TopologyValidationError("Topology name must be a non-empty string.")

    option = _NAMED_TOPOLOGY_BY_NAME.get(normalized) or _NAMED_TOPOLOGY_BY_ALIAS.get(normalized)
    if option is not None:
        path = NAMED_TOPOLOGY_DIR / option.file_name
        graph = load_topology_from_file(path)
        graph.graph["topology_name"] = option.name
        graph.graph["topology_requested_name"] = normalized
        graph.graph["topology_class"] = option.topology_class
        graph.graph["topology_description"] = option.description
        graph.graph["topology_aliases"] = list(option.aliases)
        return graph

    path = NAMED_TOPOLOGY_DIR / f"{normalized}.json"
    if not path.exists():
        available = ", ".join(list_named_topologies()) or "<none>"
        official = ", ".join(list_official_topology_classes()) or "<none>"
        raise TopologyValidationError(
            f"Unknown topology '{normalized}'. Official topology classes: {official}. "
            f"All accepted named topologies: {available}."
        )
    graph = load_topology_from_file(path)
    graph.graph["topology_name"] = normalized
    graph.graph["topology_requested_name"] = normalized
    graph.graph["topology_class"] = "custom_named"
    return graph


def load_topology_from_file(path: str | Path) -> GraphT:
    """Load and validate a topology graph from a JSON file."""
    resolved = _resolve_input_path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TopologyValidationError(f"Topology file does not exist: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise TopologyValidationError(f"Topology file is not valid JSON: {resolved}") from exc
    except OSError as exc:
        raise TopologyValidationError(f"Failed to read topology file '{resolved}': {exc}") from exc

    return load_topology_from_dict(payload, source=str(resolved))


def load_topology_from_dict(payload: object, *, source: str = "<memory>") -> GraphT:
    """Load and validate a topology graph from a decoded JSON object."""
    if nx is None:
        raise ImportError("networkx is required to load a topology") from _nx_import_error
    if not isinstance(payload, dict):
        raise TopologyValidationError(f"Topology payload from {source} must be a JSON object.")

    format_version = payload.get("format_version", 1)
    if format_version != 1:
        raise TopologyValidationError(
            f"Unsupported topology format_version={format_version!r} in {source}; expected 1."
        )

    directed = bool(payload.get("directed", False))
    nodes = _parse_nodes(payload.get("nodes"), source=source)
    edge_defaults = _parse_edge_defaults(payload.get("edge_defaults"), source=source)
    edges_raw = payload.get("edges")
    if not isinstance(edges_raw, list) or not edges_raw:
        raise TopologyValidationError(f"Topology '{source}' must define a non-empty 'edges' list.")

    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(nodes)

    seen_edges: set[tuple[int, int]] = set()
    for idx, edge_raw in enumerate(edges_raw):
        edge = _parse_edge(edge_raw, index=idx, source=source, node_ids=set(nodes), defaults=edge_defaults)
        edge_key = (edge[0], edge[1]) if directed else _undirected_edge_key(edge[0], edge[1])
        if edge_key in seen_edges:
            raise TopologyValidationError(
                f"Topology '{source}' contains a duplicate edge for {edge_key} at index {idx}."
            )
        seen_edges.add(edge_key)
        graph.add_edge(edge[0], edge[1], **edge[2])

    _validate_loaded_graph(graph, source=source)
    graph.graph["topology_source"] = source
    return graph


def _resolve_input_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    return resolved


def _parse_nodes(raw_nodes: object, *, source: str) -> list[int]:
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise TopologyValidationError(f"Topology '{source}' must define a non-empty 'nodes' list.")

    nodes: list[int] = []
    for idx, raw in enumerate(raw_nodes):
        node = _coerce_int(raw, field=f"nodes[{idx}]", source=source)
        nodes.append(node)

    if len(set(nodes)) != len(nodes):
        raise TopologyValidationError(f"Topology '{source}' contains duplicate node IDs.")

    expected = list(range(len(nodes)))
    if sorted(nodes) != expected:
        raise TopologyValidationError(
            f"Topology '{source}' must use contiguous integer node IDs 0..{len(nodes) - 1}."
        )
    return nodes


def _parse_edge_defaults(raw_defaults: object, *, source: str) -> dict[str, Any]:
    if raw_defaults is None:
        return {}
    if not isinstance(raw_defaults, dict):
        raise TopologyValidationError(f"Topology '{source}' field 'edge_defaults' must be an object.")
    return _parse_edge_attrs(raw_defaults, field_prefix="edge_defaults", source=source)


def _parse_edge(
    raw_edge: object,
    *,
    index: int,
    source: str,
    node_ids: set[int],
    defaults: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    if not isinstance(raw_edge, dict):
        raise TopologyValidationError(f"Topology '{source}' edge #{index} must be an object.")

    src = _coerce_int(raw_edge.get("source"), field=f"edges[{index}].source", source=source)
    dst = _coerce_int(raw_edge.get("target"), field=f"edges[{index}].target", source=source)
    if src == dst:
        raise TopologyValidationError(f"Topology '{source}' edge #{index} cannot be a self-loop ({src}->{dst}).")
    if src not in node_ids or dst not in node_ids:
        raise TopologyValidationError(
            f"Topology '{source}' edge #{index} references unknown nodes ({src}, {dst})."
        )

    attrs = dict(defaults)
    attrs.update(_parse_edge_attrs(raw_edge, field_prefix=f"edges[{index}]", source=source))
    return src, dst, attrs


def _parse_edge_attrs(raw_attrs: dict[str, Any], *, field_prefix: str, source: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if "capacity" in raw_attrs:
        attrs["capacity"] = _coerce_float(
            raw_attrs.get("capacity"),
            field=f"{field_prefix}.capacity",
            source=source,
            min_value=0.0,
            strictly_positive=True,
        )
    if "latency_ms" in raw_attrs:
        attrs["latency_ms"] = _coerce_float(
            raw_attrs.get("latency_ms"),
            field=f"{field_prefix}.latency_ms",
            source=source,
            min_value=0.0,
            strictly_positive=False,
        )
    if "weight" in raw_attrs:
        attrs["weight"] = _coerce_float(
            raw_attrs.get("weight"),
            field=f"{field_prefix}.weight",
            source=source,
            min_value=0.0,
            strictly_positive=True,
        )
    if "active" in raw_attrs:
        active = raw_attrs.get("active")
        if not isinstance(active, bool):
            raise TopologyValidationError(
                f"Topology '{source}' field '{field_prefix}.active' must be true or false."
            )
        attrs["active"] = active
    return attrs


def _validate_loaded_graph(graph: GraphT, *, source: str) -> None:
    if graph.number_of_nodes() < 2:
        raise TopologyValidationError(f"Topology '{source}' must contain at least 2 nodes.")
    if graph.number_of_edges() < 1:
        raise TopologyValidationError(f"Topology '{source}' must contain at least 1 edge.")

    undirected = graph.to_undirected()
    if not nx.is_connected(undirected):
        raise TopologyValidationError(
            f"Topology '{source}' must be connected when treated as an undirected physical graph."
        )


def _coerce_int(value: object, *, field: str, source: str) -> int:
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TopologyValidationError(
            f"Topology '{source}' field '{field}' must be an integer."
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise TopologyValidationError(f"Topology '{source}' field '{field}' must be an integer.")
    return coerced


def _coerce_float(
    value: object,
    *,
    field: str,
    source: str,
    min_value: float,
    strictly_positive: bool,
) -> float:
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TopologyValidationError(
            f"Topology '{source}' field '{field}' must be numeric."
        ) from exc
    if coerced != coerced or coerced in (float("inf"), float("-inf")):
        raise TopologyValidationError(f"Topology '{source}' field '{field}' must be finite.")
    if strictly_positive and coerced <= min_value:
        raise TopologyValidationError(
            f"Topology '{source}' field '{field}' must be greater than {min_value}."
        )
    if (not strictly_positive) and coerced < min_value:
        raise TopologyValidationError(
            f"Topology '{source}' field '{field}' must be at least {min_value}."
        )
    return coerced


def _undirected_edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)
