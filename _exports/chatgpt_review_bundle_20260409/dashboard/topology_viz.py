from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from greennet.env import EnvConfig, GreenNetEnv


@dataclass
class TopologyPlotData:
    graph: nx.Graph
    pos: Dict[int, Tuple[float, float]]
    toggled_counts: Dict[Tuple[int, int], int]


def _load_env_config(run_dir: Path) -> EnvConfig:
    cfg_path = run_dir / "env_config.json"
    if cfg_path.exists():
        return EnvConfig.from_json(str(cfg_path))
    return EnvConfig()


def _infer_topology_seed(run_dir: Path) -> int:
    cfg_path = run_dir / "env_config.json"
    if cfg_path.exists():
        try:
            import json

            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                topo = data.get("topology_seed")
                if topo is not None:
                    return int(topo)
                seeds = data.get("topology_seeds")
                if isinstance(seeds, list) and seeds:
                    return int(seeds[0])
        except Exception:
            pass
    return 0


def _action_to_edge(env: GreenNetEnv, action: int) -> Optional[Tuple[int, int]]:
    if action == 0:
        return None
    idx = action - 1
    if 0 <= idx < len(env.edge_list):
        u, v = env.edge_list[idx]
        return (int(u), int(v))
    return None


def build_topology_plot_data(run_dir: Path, per_step: pd.DataFrame) -> Optional[TopologyPlotData]:
    if per_step is None or per_step.empty:
        return None

    cfg = _load_env_config(run_dir)
    topo_seed = _infer_topology_seed(run_dir)

    env = GreenNetEnv(cfg)
    env.reset(seed=topo_seed)

    base_graph = None
    for name in ["_base_graph", "base_graph", "graph", "_graph"]:
        if hasattr(env, name):
            cand = getattr(env, name)
            if hasattr(cand, "nodes") and hasattr(cand, "edges"):
                base_graph = cand
                break

    if base_graph is None:
        env.close()
        return None

    G = nx.Graph()
    G.add_nodes_from(list(base_graph.nodes()))
    G.add_edges_from([(int(u), int(v)) for (u, v) in list(base_graph.edges())])

    toggled_counts: Dict[Tuple[int, int], int] = {}
    if "toggle_applied" in per_step.columns and "action" in per_step.columns:
        toggled_rows = per_step[per_step["toggle_applied"].astype(bool)]
        for a in toggled_rows["action"].tolist():
            try:
                edge = _action_to_edge(env, int(a))
            except Exception:
                edge = None
            if edge is None:
                continue
            u, v = edge
            key = (u, v) if u <= v else (v, u)
            toggled_counts[key] = toggled_counts.get(key, 0) + 1

    env.close()

    pos = nx.spring_layout(G, seed=int(topo_seed))
    return TopologyPlotData(graph=G, pos=pos, toggled_counts=toggled_counts)


def plot_topology(
    data: TopologyPlotData,
    title: str = "Topology (toggled edges highlighted)",
) -> plt.Figure:
    G, pos, counts = data.graph, data.pos, data.toggled_counts

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_title(title)

    nx.draw_networkx_nodes(G, pos, node_size=250, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4, ax=ax)

    if counts:
        edges = list(counts.keys())
        widths = [1.0 + 0.8 * float(counts[e]) for e in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, alpha=0.9, ax=ax)

    ax.axis("off")
    fig.tight_layout()
    return fig
