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
from greennet.traffic import (ConstantTrafficGenerator, StochasticTrafficConfig, StochasticTrafficGenerator, TrafficBurst, TrafficGenerator,)


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

    base_capacity: float = 10.0
    base_latency_ms: float = 10.0

    flows_per_step: int = 6
    demand_min: float = 1.0
    demand_max: float = 5.0

    # Traffic model selection:
    # - "uniform": current per-step random src/dst + uniform demand
    # - "stochastic": use greennet.traffic.StochasticTrafficGenerator
    traffic_model: str = "uniform"

    # Seed for traffic generator; if None, we derive it from env.reset(seed=...).
    traffic_seed: int | None = None

    # Optional hotspot pairs for stochastic traffic: (src, dst, weight)
    traffic_hotspots: Tuple[Tuple[int, int, float], ...] = ()

    # Stochastic traffic tuning
    traffic_avg_bursts_per_step: float = 3.0
    traffic_p_elephant: float = 0.15
    traffic_mice_size_range: Tuple[int, int] = (1, 5)
    traffic_elephant_size_range: Tuple[int, int] = (10, 50)
    traffic_duration_range: Tuple[int, int] = (1, 6)
    traffic_spike_prob: float = 0.01
    traffic_spike_multiplier_range: Tuple[float, float] = (2.0, 6.0)
    traffic_spike_duration_range: Tuple[int, int] = (3, 12)

    drop_penalty_lambda: float = 5.0
    # ENERGY WEIGHT: Increase to prioritize energy savings more; decrease to prioritize performance (drops).
    energy_weight: float = 50.0

    # QoS constraint: target maximum normalized drop ratio (dropped / (delivered + dropped)).
    # If norm_drop exceeds this, we add an extra penalty (linear by default) so QoS stays stable while optimizing energy.
    qos_target_norm_drop: float = 0.0720

    # Gate QoS penalty until enough traffic volume has been observed.
    qos_min_volume: float = 3000.0

    # Strength of the extra penalty when norm_drop exceeds qos_target_norm_drop.
    # For linear QoS penalty, this should be small (tens), otherwise it will dominate reward.
    qos_violation_penalty_scale: float = 30.0

    # When norm_drop_total is close to qos_target_norm_drop, block turning edges OFF.
    qos_guard_margin: float = 0.002  # headroom; tune 0.001–0.005
    qos_guard_penalty_scale: float = 1.0  # multiplier for blocked OFF attempts due to QoS

    toggle_penalty: float = 0.01

    blocked_action_penalty: float = 0.01
    revert_penalty_scale: float = 0.5
    top_k_edge_utils: int = 0  # not used yet; placeholder for richer observations
    normalize_drop: bool = True
    saturation_util_threshold: float = 0.9  # for counting near-saturated edges
    toggle_cooldown_steps: int = 10  # minimum steps between toggles of the same edge
    util_block_threshold: float = 0.2  # block turning OFF an edge if its utilization is above this threshold
    global_toggle_cooldown_steps: int = 30


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
        self._traffic_generator: TrafficGenerator | None = None
        self._traffic_by_step: Dict[int, List[TrafficBurst]] = {}
        self._active_bursts: List[Tuple[int, int, float, int]] = []  # (src, dst, demand, remaining_steps)
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

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None,) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self._step_count = 0
        self._prev_dropped = 0.0
        # Track cumulative metrics (some Simulator implementations return cumulative totals).
        # We compute per-step deltas for reward so we don't double-count cumulative values.
        self._prev_energy_kwh_total = 0.0
        self._prev_delivered_total = 0.0
        self._prev_dropped_total = 0.0
        self._global_toggle_cooldown_remaining = 0
        self._last_norm_drop_total = 0.0

        # ---- Traffic generator setup ----
        self._traffic_by_step = {}
        self._active_bursts = []

        # Derive a deterministic traffic seed when not explicitly provided.
        derived_seed: int | None = self.config.traffic_seed
        if derived_seed is None and seed is not None:
            derived_seed = int(seed) + 10_000  # keep it separate from topology/model seeding

        if self.config.traffic_model.lower() == "stochastic":
            tcfg = StochasticTrafficConfig(
                node_count=int(self.config.node_count),
                avg_bursts_per_step=float(self.config.traffic_avg_bursts_per_step),
                hotspots=tuple(self.config.traffic_hotspots),
                p_elephant=float(self.config.traffic_p_elephant),
                mice_size_range=tuple(self.config.traffic_mice_size_range),
                elephant_size_range=tuple(self.config.traffic_elephant_size_range),
                duration_range=tuple(self.config.traffic_duration_range),
                spike_prob=float(self.config.traffic_spike_prob),
                spike_multiplier_range=tuple(self.config.traffic_spike_multiplier_range),
                spike_duration_range=tuple(self.config.traffic_spike_duration_range),
            )
            self._traffic_generator = StochasticTrafficGenerator(tcfg, seed=derived_seed)

            # Precompute bursts and bucket them by integer step.
            for burst in self._traffic_generator.generate(int(self.config.max_steps)):
                step_idx = int(burst.start_time)
                if step_idx < 0 or step_idx >= int(self.config.max_steps):
                    continue
                self._traffic_by_step.setdefault(step_idx, []).append(burst)
        else:
            # Keep old behavior by default.
            self._traffic_generator = ConstantTrafficGenerator(rate=1.0)

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
                f" traffic_model={self.config.traffic_model}"
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
        
        blocked_any = bool(toggle_blocked or toggle_blocked_high_util or toggle_blocked_global)
        if blocked_any:
            toggle_cost = float(toggle_cost) + float(self.config.blocked_action_penalty)

        flows = self._generate_flows()
        metrics = self.simulator.step(flows)

        # ---- Convert simulator metrics into per-step deltas (robust to cumulative vs per-step APIs) ----
        # Some Simulator implementations expose cumulative totals (episode-to-date). If we use those
        # directly in the reward each step, we double-count and the episode reward blows up.
        energy_total = float(metrics.energy_kwh)
        delivered_total = float(metrics.delivered)
        dropped_total = float(metrics.dropped)

        prev_e = float(getattr(self, "_prev_energy_kwh_total", 0.0))
        prev_del = float(getattr(self, "_prev_delivered_total", 0.0))
        prev_drop = float(getattr(self, "_prev_dropped_total", 0.0))

        # Default: treat metrics as cumulative totals.
        delta_energy = energy_total - prev_e
        delta_delivered = delivered_total - prev_del
        delta_dropped = dropped_total - prev_drop

        # Detect per-step vs cumulative reporting.
        # Default assumption is cumulative totals, but some Simulator implementations report per-step
        # values (especially for energy). In that case, treating them as cumulative makes deltas go to
        # ~0 after step 1 (because prev == current).
        eps = 1e-12

        # ENERGY: if energy_total stays roughly constant across steps (delta ~ 0) while energy_total > 0,
        # interpret it as a per-step value and accumulate our own episode-to-date total.
        if prev_e > 0.0 and energy_total > 0.0 and delta_energy <= eps:
            delta_energy = energy_total
            energy_total = prev_e + delta_energy

        # DELIVERED/DROPPED: if deltas are negative (beyond tiny numerical noise), treat totals as per-step.
        if delta_delivered < -eps:
            delta_delivered = delivered_total
            delivered_total = prev_del + delta_delivered
        if delta_dropped < -eps:
            delta_dropped = dropped_total
            dropped_total = prev_drop + delta_dropped

        # Clamp tiny negatives caused by float noise.
        delta_energy = float(max(0.0, delta_energy))
        delta_delivered = float(max(0.0, delta_delivered))
        delta_dropped = float(max(0.0, delta_dropped))

        # Persist totals for next step.
        self._prev_energy_kwh_total = float(energy_total)
        self._prev_delivered_total = float(delivered_total)
        self._prev_dropped_total = float(dropped_total)

        # Per-step normalized drop ratio.
        norm_drop_step = delta_dropped / max(delta_delivered + delta_dropped + 1e-9, 1e-9)

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
        # Store previous-step drops for the next observation (use per-step delta).
        self._prev_dropped = float(delta_dropped)

        # Normalized drop ratios:
        # - `norm_drop_total` uses cumulative totals (good for a "whole-episode" QoS constraint)
        # - `norm_drop_step` uses per-step deltas (good for shaping reward each step)
        norm_drop_total = float(dropped_total) / max(float(delivered_total) + float(dropped_total) + 1e-9, 1e-9)
        self._last_norm_drop_total = float(norm_drop_total)

        drop_component = float(delta_dropped)
        if self.config.normalize_drop:
            drop_component = float(norm_drop_step)

        # Use per-step energy delta for reward (prevents double-counting cumulative totals).
        reward_energy = -float(self.config.energy_weight) * float(delta_energy)

        reward_drop = -self.config.drop_penalty_lambda * drop_component

        # Extra QoS penalty if drop ratio exceeds the target (after enough volume is observed).
        volume = float(delivered_total + dropped_total)
        qos_excess = 0.0
        if volume >= float(self.config.qos_min_volume):
            qos_excess = max(0.0, float(norm_drop_total) - float(self.config.qos_target_norm_drop))
        # Cap overshoot to prevent rare spikes from nuking reward (stabilizes training).
        qos_excess = min(qos_excess, 0.05)
        # Linear QoS penalty (less spiky than quadratic; prevents QoS term from dominating reward).
        reward_qos = -float(self.config.qos_violation_penalty_scale) * float(qos_excess)

        reward_toggle = -toggle_cost
        reward = reward_energy + reward_drop + reward_qos + reward_toggle
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
            "toggle_blocked_any": blocked_any,
            "reward_energy": reward_energy,
            "reward_drop": reward_drop,
            "delta_energy_kwh": float(delta_energy),
            "delta_delivered": float(delta_delivered),
            "delta_dropped": float(delta_dropped),
            "norm_drop_step": float(norm_drop_step),
            "norm_drop": float(norm_drop_total),
            "reward_qos": reward_qos,
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



    def get_action_mask(self) -> np.ndarray:
        """Return a boolean mask over actions (Discrete(E+1)).
        
        mask[a] == True means action 'a' is currently allowed.

        We mask out actions that would be blocked by the same gates used in '_apply_action':
            - edge mising (when topology_randomize=True)
            - global cooldown (only blocks turning OFF)
            - high utilization (only blocks turning OFF)
            - per-edge cooldown (blocks both directions)

        Note: we do NOT try to predict connectivity reverts here (too expensive to evaluate for every action each step). Reverts remain possible.
        """
        n = int(self.action_space.n)
        mask = np.ones((n,), dtype=bool)
        if n <= 0:
            return mask
        
        # Always allow NOOP
        mask[0] = True

        if self.simulator is None or not self.edge_list:
            mask[1:] = False
            return mask
        
        global_cd = int(getattr(self, "_global_toggle_cooldown_remaining", 0))

        for a in range(1, n):
            idx = a - 1
            if idx < 0 or idx >= len(self.edge_list):
                mask[a] = False
                continue
        
            edge = self.edge_list[idx]
            if not self.simulator.graph.has_edge(*edge):
                mask[a] = False
                continue

            key = self._edge_key(edge[0], edge[1])
            current_state = bool(self.simulator.active.get(key, True))

            # Global cooldown blocks only when trying to turn an edge OFF
            if current_state and global_cd > 0:
                mask[a] = False
                continue

            # High-util gate blocks only when trying to turn an edge OFF
            if current_state:
                # QoS guard: block turning an edge OFF when QoS is near/over target.
                last_nd = float(getattr(self, "_last_norm_drop_total", 0.0))
                target = float(self.config.qos_target_norm_drop)
                margin = float(getattr(self.config, "qos_guard_margin", 0.0))
                if last_nd >= (target - margin):
                    mask[a] = False
                    continue
                util = float(self.simulator.utilization.get(key, 0.0))
                if util > float(self.config.util_block_threshold):
                    mask[a] = False
                    continue
            
            # Per-edge cooldown blocks both directions
            last_toggled = self.simulator.graph.edges[edge].get(
                "last_toggled", -self.config.toggle_cooldown_steps
            )
            if (self._step_count - last_toggled) < int(self.config.toggle_cooldown_steps):
                mask[a] = False
                continue
            mask[a] = True
        return mask



    def _generate_flows(self) -> Tuple[Flow, ...]:
        if self.simulator is None:
            return tuple()

        # Stochastic traffic mode: convert scheduled bursts into flows.
        if self.config.traffic_model.lower() == "stochastic":
            step_idx = max(0, self._step_count - 1)  # step() increments before generating flows

            # Add newly starting bursts for this step.
            for b in self._traffic_by_step.get(step_idx, []):
                demand = float(max(0, int(b.size)))
                remaining = max(1, int(b.duration))
                self._active_bursts.append((int(b.source), int(b.destination), demand, remaining))

            flows: List[Flow] = []
            still_active: List[Tuple[int, int, float, int]] = []

            # Emit one flow per active burst per step.
            for src, dst, demand, remaining in self._active_bursts:
                if src == dst or demand <= 0.0:
                    continue
                flows.append(Flow(int(src), int(dst), float(demand)))
                remaining -= 1
                if remaining > 0:
                    still_active.append((src, dst, demand, remaining))

            self._active_bursts = still_active
            return tuple(flows)

        # Default/uniform traffic mode (existing behavior).
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

        # QoS guard: when norm_drop_total is near/above target, block turning edges OFF.
        if current_state:
            last_nd = float(getattr(self, "_last_norm_drop_total", 0.0))
            target = float(self.config.qos_target_norm_drop)
            margin = float(getattr(self.config, "qos_guard_margin", 0.0))
            if last_nd >= (target - margin):
                penalty = (
                    self.config.toggle_penalty
                    * self.config.revert_penalty_scale
                    * float(getattr(self.config, "qos_guard_penalty_scale", 1.0))
                )
                return penalty, False, False, False, False, False

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
