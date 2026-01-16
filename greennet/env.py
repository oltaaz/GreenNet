"""Gymnasium environment wrapper for GreenNet."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from greennet.routing import ShortestPathPolicy
from greennet.simulator import Flow, Simulator
from greennet.topology import TopologyConfig, build_random_topology


@dataclass
class EnvConfig:
    """Configuration for the GreenNet environment."""

    max_steps: int = 1000
    topology_seed: int | None = 0
    topology_seeds: Tuple[int, ...] | None = None
    topology_randomize: bool = False
    node_count: int = 10
    edge_prob: float = 0.5
    directed: bool = False

    base_capacity: float = 5.0
    base_latency_ms: float = 10.0

    flows_per_step: int = 4
    demand_min: float = 1.0
    demand_max: float = 5.0

    drop_penalty_lambda: float = 0.4
    # ENERGY WEIGHT: Increase to prioritize energy savings more; decrease to prioritize performance (drops).
    energy_weight: float = 10.0

    toggle_penalty: float = 0.0005

    revert_penalty_scale: float = 0.5
    top_k_edge_utils: int = 0  # not used yet; placeholder for richer observations
    normalize_drop: bool = True
    saturation_util_threshold: float = 0.9  # for counting near-saturated edges
    toggle_cooldown_steps: int = 5  # minimum steps between toggles of the same edge
    util_block_threshold: float = 0.10  # block turning OFF an edge if its utilization is above this threshold
    global_toggle_cooldown_steps: int = 20


class GreenNetEnv(gym.Env):
    """Minimal Gymnasium-compatible environment for GreenNet.

    Action space:
      Discrete(E + 1), where 0 = no-op, 1..E toggle edge edge_list[i-1].
      The edge_list is fixed for the whole episode (built from the topology seed).

    Observation keys:
      - time: normalized [0, 1]
      - avg_util: average utilization [0, 1]
      - active_ratio: fraction of edges active [0, 1]
      - max_util: maximum edge utilization
      - dropped_prev: packets/flow units dropped on previous step
      - num_active_edges: count of active edges

    Reward:
      -(energy_kwh + lambda * dropped + nu * toggle_cost)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()

        self._step_count: int = 0
        self.simulator: Simulator | None = None
        self._debug_logged: bool = False
        # Build base topology once; action space is derived from toggleable (non-bridge) edges.
        topo_config = self._topology_config(self.config.topology_seed)
        self._base_graph = build_random_topology(topo_config)
        # Ensure the fixed base graph is connected so safety checks don't revert everything.
        # Do this once here (before action_space is created) so action dimensions stay consistent for SB3.
        if self._base_graph.number_of_nodes() > 0:
            und = self._base_graph.to_undirected()
            if not nx.is_connected(und):
                components = list(nx.connected_components(und))
                for left, right in zip(components, components[1:]):
                    u = next(iter(left))
                    v = next(iter(right))
                    self._base_graph.add_edge(u, v)
        self.edge_list = self._compute_toggleable_edges(self._base_graph)
        self.E = len(self.edge_list)
        action_size = self.E + 1
        self.action_space = spaces.Discrete(max(1, action_size))

        self.observation_space = spaces.Dict(
            {
                "time": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "avg_util": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "active_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "max_util": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "min_util": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "p95_util": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "dropped_prev": spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
                "num_active_edges": spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
                "near_saturated_edges": spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
                "edge_active": spaces.Box(low=0.0, high=1.0, shape=(self.E,), dtype=np.float32),
                "edge_util": spaces.Box(low=0.0, high=1.0, shape=(self.E,), dtype=np.float32),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self._step_count = 0
        self._prev_dropped = 0.0
        self._global_toggle_cooldown_remaining = 0

        if self.config.topology_randomize:
            graph = self._build_topology()
            # IMPORTANT: keep self.edge_list FIXED (from __init__) so action indices stay consistent for SB3.
            # Some edges from self.edge_list may not exist in this randomized graph; those actions become effectively no-ops.
        else:
            # Reuse the base topology so edge indices remain valid and match action space.
            graph = self._base_graph.copy()
            # Keep the same self.edge_list mapping used to define action_space in __init__.

        # (Connectivity surgery removed: graph is now connected at __init__ for fixed topology;
        # for randomized topology, accept that action_space is fixed and edge_list may not match.)

        # Set baseline edge attributes.
        for u, v in graph.edges():
            graph.edges[u, v]["capacity"] = float(self.config.base_capacity)
            graph.edges[u, v]["latency_ms"] = float(self.config.base_latency_ms)
            graph.edges[u, v]["weight"] = 1.0
            graph.edges[u, v]["active"] = True
            graph.edges[u, v]["last_toggled"] = -self.config.toggle_cooldown_steps

        if not self._debug_logged:
            self._debug_logged = True
            print(
                f"[env_cfg] drop_lambda={self.config.drop_penalty_lambda} "
                f"energy_w={self.config.energy_weight} toggle_pen={self.config.toggle_penalty} "
                f"normalize_drop={self.config.normalize_drop} cooldown={self.config.toggle_cooldown_steps} "
                f"util_block_thr={self.config.util_block_threshold} global_cd={self.config.global_toggle_cooldown_steps}"
            )

        def _power_model_watts(g) -> float:
            # Simple power model: base + per-active-edge.
            active_edges = sum(1 for (a, b) in g.edges() if g.edges[a, b].get("active", True))
            return 50.0 + 10.0 * active_edges

        self.simulator = Simulator(
            graph,
            routing_policy=ShortestPathPolicy(weight="weight"),
            dt_seconds=1.0,
            default_capacity=self.config.base_capacity,
            default_latency_ms=self.config.base_latency_ms,
            power_model_watts=_power_model_watts,
            carbon_intensity_g_per_kwh=lambda t: 400.0 + 10.0 * math.sin(t),
        )

        edge_active, edge_util = self._edge_feature_vectors()
        obs = self._build_observation(
            time_ratio=0.0,
            avg_util=0.0,
            active_ratio=1.0 if len(self.edge_list) > 0 else 0.0,
            max_util=0.0,
            min_util=0.0,
            p95_util=0.0,
            dropped_prev=0.0,
            num_active_edges=float(len(self.edge_list)),
            near_saturated_edges=0.0,
            edge_active=edge_active,
            edge_util=edge_util,
        )
        info: Dict[str, Any] = {"metrics": None}
        return obs, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        if self.simulator is None:
            raise RuntimeError("Simulator not initialized; call reset() first.")
        if getattr(self, "_global_toggle_cooldown_remaining", 0) > 0:
            self._global_toggle_cooldown_remaining -= 1

        # Action diagnostics (helps distinguish true no-op vs invalid index).
        try:
            action_int = int(action)
        except Exception:
            action_int = 0
        action_is_noop = action_int <= 0
        idx = action_int - 1
        action_is_invalid = (action_int > 0) and (idx < 0 or idx >= len(self.edge_list) or self.simulator is None or not self.edge_list)

        toggle_cost, toggle_applied, toggle_reverted, toggle_blocked, toggle_blocked_high_util, toggle_blocked_global = self._apply_action(action)
        
        flows = self._generate_flows()
        metrics = self.simulator.step(flows)

        time_ratio = min(1.0, self._step_count / self.config.max_steps)
        active_ratio = self._active_ratio()
        util_values = list(self.simulator.utilization.values()) if self.simulator else []
        max_util = max(util_values, default=0.0)
        min_util = min(util_values, default=0.0)
        p95_util = float(np.percentile(util_values, 95)) if util_values else 0.0
        near_sat = sum(1 for u in util_values if u >= self.config.saturation_util_threshold)
        edge_active, edge_util = self._edge_feature_vectors()
        obs = self._build_observation(
            time_ratio=time_ratio,
            avg_util=metrics.avg_utilization,
            active_ratio=active_ratio,
            max_util=max_util,
            min_util=min_util,
            p95_util=p95_util,
            dropped_prev=self._prev_dropped,
            num_active_edges=float(self._active_edges_count()),
            near_saturated_edges=float(near_sat),
            edge_active=edge_active,
            edge_util=edge_util,
        )
        self._prev_dropped = metrics.dropped

        drop_component = metrics.dropped
        if self.config.normalize_drop:
            drop_component = metrics.dropped / max(metrics.delivered + metrics.dropped + 1e-9, 1e-9)

        # ENERGY WEIGHT: Increase this multiplier to make the agent care more about saving energy.
        # Decrease it if energy dominates and the agent becomes too aggressive about switching links off.
        reward_energy = -self.config.energy_weight * metrics.energy_kwh

        reward_drop = -self.config.drop_penalty_lambda * drop_component
        reward_toggle = -toggle_cost
        reward = reward_energy + reward_drop + reward_toggle
        terminated = False
        truncated = self._step_count >= self.config.max_steps
        info: Dict[str, Any] = {
            "metrics": metrics,
            "flows": flows,
            "toggle_applied": toggle_applied,
            "toggle_reverted": toggle_reverted,
            "toggle_blocked_cooldown": toggle_blocked,
            "toggle_blocked_high_util": toggle_blocked_high_util,
            "toggle_blocked_global_cooldown": toggle_blocked_global,
            "reward_energy": reward_energy,
            "reward_drop": reward_drop,
            "reward_toggle": reward_toggle,
            "total_reward": reward,
            "action_int": action_int,
            "action_is_noop": action_is_noop,
            "action_is_invalid": action_is_invalid,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _generate_flows(self) -> Tuple[Flow, ...]:
        if self.simulator is None:
            return tuple()

        flows: List[Flow] = []
        node_count = self.simulator.graph.number_of_nodes()

        for _ in range(self.config.flows_per_step):
            src = int(self.np_random.integers(0, node_count))
            dst = int(self.np_random.integers(0, node_count))
            while dst == src:
                dst = int(self.np_random.integers(0, node_count))

            demand = float(self.np_random.uniform(self.config.demand_min, self.config.demand_max))
            flows.append(Flow(src, dst, demand))

        return tuple(flows)

    def _build_observation(
        self,
        time_ratio: float,
        avg_util: float,
        active_ratio: float,
        max_util: float,
        min_util: float,
        p95_util: float,
        dropped_prev: float,
        num_active_edges: float,
        near_saturated_edges: float,
        edge_active: np.ndarray,
        edge_util: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "time": np.array([np.float32(time_ratio)], dtype=np.float32),
            "avg_util": np.array([np.float32(avg_util)], dtype=np.float32),
            "active_ratio": np.array([np.float32(active_ratio)], dtype=np.float32),
            "max_util": np.array([np.float32(max_util)], dtype=np.float32),
            "min_util": np.array([np.float32(min_util)], dtype=np.float32),
            "p95_util": np.array([np.float32(p95_util)], dtype=np.float32),
            "dropped_prev": np.array([np.float32(dropped_prev)], dtype=np.float32),
            "num_active_edges": np.array([np.float32(num_active_edges)], dtype=np.float32),
            "near_saturated_edges": np.array([np.float32(near_saturated_edges)], dtype=np.float32),
            "edge_active": edge_active.astype(np.float32, copy=False),
            "edge_util": edge_util.astype(np.float32, copy=False),
        }

    def _active_ratio(self) -> float:
        if self.simulator is None:
            return 0.0
        total_edges = self.simulator.graph.number_of_edges()
        if total_edges <= 0:
            return 0.0
        active_edges = sum(1 for _, is_active in self.simulator.active.items() if is_active)
        return active_edges / float(total_edges)

    def _active_edges_count(self) -> int:
        if self.simulator is None:
            return 0
        return sum(1 for _, is_active in self.simulator.active.items() if is_active)

    def _edge_feature_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (edge_active, edge_util) aligned with self.edge_list."""
        edge_active = np.zeros(self.E, dtype=np.float32)
        edge_util = np.zeros(self.E, dtype=np.float32)

        if self.simulator is None:
            return edge_active, edge_util

        for idx, (u, v) in enumerate(self.edge_list):
            key = self._edge_key(u, v)
            is_active = bool(self.simulator.active.get(key, True))
            util = float(self.simulator.utilization.get(key, 0.0))
            edge_active[idx] = 1.0 if is_active else 0.0
            edge_util[idx] = float(np.clip(util, 0.0, 1.0))

        return edge_active, edge_util

    def _topology_config(self, seed: int | None) -> TopologyConfig:
        return TopologyConfig(
            node_count=self.config.node_count,
            edge_prob=self.config.edge_prob,
            directed=self.config.directed,
            seed=seed,
        )

    def _build_topology(self):
        if not self.config.topology_randomize:
            return self._base_graph.copy()

        seed = self.config.topology_seed
        if self.config.topology_randomize:
            if self.config.topology_seeds:
                # Choose from provided seed set deterministically w.r.t. env RNG.
                idx = int(self.np_random.integers(0, len(self.config.topology_seeds)))  # type: ignore[arg-type]
                seed = self.config.topology_seeds[idx]
            else:
                # Use env RNG to derive a seed for variety across episodes.
                seed = int(self.np_random.integers(0, 1_000_000))
        return build_random_topology(self._topology_config(seed))

    def _edge_key(self, u: int, v: int) -> Tuple[int, int]:
        if self.simulator and self.simulator.graph.is_directed():
            return (u, v)
        return (u, v) if u <= v else (v, u)

    def _apply_action(self, action: Any) -> Tuple[float, bool, bool, bool, bool, bool]:
        """Toggle an edge if requested.

        Returns (toggle_cost, applied, reverted, cooldown_blocked, high_util_blocked, global_cooldown_blocked).
        """
        if self.simulator is None or not self.edge_list:
            return 0.0, False, False, False, False, False

        try:
            action_int = int(action)
        except Exception:
            action_int = 0

        if action_int <= 0:
            return 0.0, False, False, False, False, False

        idx = action_int - 1
        if idx < 0 or idx >= len(self.edge_list):
            return 0.0, False, False, False, False, False

        edge = self.edge_list[idx]
        key = self._edge_key(edge[0], edge[1])

        # If the chosen edge isn't present in the current graph (possible when topology_randomize=True),
        # treat the action as a safe no-op.
        if not self.simulator.graph.has_edge(*edge):
            return 0.0, False, False, False, False, False

        current_state = bool(self.simulator.active.get(key, True))

        # Global cooldown: block rapid toggles ONLY when turning an edge OFF.
        # Turning an edge ON is always allowed so the agent can recover quickly.
        if current_state and getattr(self, "_global_toggle_cooldown_remaining", 0) > 0:
            penalty = self.config.toggle_penalty * self.config.revert_penalty_scale
            return penalty, False, False, False, False, True

        # Safety gate: block turning OFF a highly utilized edge.
        # (Turning ON an edge is always allowed.)
        if current_state:
            util = float(self.simulator.utilization.get(key, 0.0))
            if util > self.config.util_block_threshold:
                penalty = self.config.toggle_penalty * self.config.revert_penalty_scale
                return penalty, False, False, False, True, False

        # Cooldown: prevent rapid flapping of the same edge.
        last_toggled = self.simulator.graph.edges[edge].get("last_toggled", -self.config.toggle_cooldown_steps)
        if (self._step_count - last_toggled) < self.config.toggle_cooldown_steps:
            penalty = self.config.toggle_penalty * self.config.revert_penalty_scale
            return penalty, False, False, True, False, False

        new_state = not current_state

        # Apply toggle.
        self.simulator.active[key] = new_state
        if self.simulator.graph.has_edge(*edge):
            self.simulator.graph.edges[edge]["active"] = new_state
            self.simulator.graph.edges[edge]["last_toggled"] = self._step_count

        # Safety: ensure graph stays connected via active edges.
        if not self._is_active_graph_connected():
            # Revert
            self.simulator.active[key] = current_state
            if self.simulator.graph.has_edge(*edge):
                self.simulator.graph.edges[edge]["active"] = current_state
            penalty = self.config.toggle_penalty * self.config.revert_penalty_scale
            self._global_toggle_cooldown_remaining = int(self.config.global_toggle_cooldown_steps)
            return penalty, False, True, False, False, False

        toggle_cost = self.config.toggle_penalty

        # Start global cooldown ONLY after successfully turning an edge OFF.
        if new_state is False:
            self._global_toggle_cooldown_remaining = int(self.config.global_toggle_cooldown_steps)

        return toggle_cost, True, False, False, False, False

    def _is_active_graph_connected(self) -> bool:
        if self.simulator is None:
            return True

        g = self.simulator.graph
        active_edges = [
            (u, v)
            for (u, v) in g.edges()
            if self.simulator.active.get(self._edge_key(u, v), True)
        ]

        if g.number_of_nodes() == 0:
            return True

        if g.is_directed():
            H = nx.DiGraph()
            H.add_nodes_from(g.nodes())
            H.add_edges_from(active_edges)
            return nx.is_weakly_connected(H) if H.number_of_edges() > 0 else False

        H = nx.Graph()
        H.add_nodes_from(g.nodes())
        H.add_edges_from(active_edges)
        if H.number_of_edges() == 0:
            return False
        return nx.is_connected(H)

    def _compute_toggleable_edges(self, graph: nx.Graph) -> List[Tuple[int, int]]:
        """Return edges that are not bridges (safe to toggle without guaranteed disconnect)."""
        if graph.number_of_edges() == 0:
            return []

        undirected = graph if not graph.is_directed() else graph.to_undirected()
        try:
            bridges = set(nx.bridges(undirected))
        except Exception:
            bridges = set()

        toggleable: List[Tuple[int, int]] = []
        for u, v in graph.edges():
            e = (u, v) if graph.is_directed() else (min(u, v), max(u, v))
            if e in bridges or (e[1], e[0]) in bridges:
                continue
            toggleable.append((u, v))
        toggleable.sort()
        return toggleable


if __name__ == "__main__":
    env = GreenNetEnv()
    obs, info = env.reset()
    print("reset OK. obs keys:", list(obs.keys()), "info:", info)

    for t in range(10):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        print(f"t={t} r={r:.3f} term={terminated} trunc={truncated} avg_util={obs['avg_util'][0]:.3f}")

    print("Env smoke OK ✅")
