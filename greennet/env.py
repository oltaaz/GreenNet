"""Gymnasium environment wrapper for GreenNet."""
from __future__ import annotations
from greennet.forecasting import DemandForecastConfig, EmaDemandForecaster

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from greennet.routing import ShortestPathPolicy
from greennet.simulator import Flow, Simulator
from greennet.topology import TopologyConfig, build_random_topology
from greennet.traffic import (
    ConstantTrafficGenerator,
    StochasticTrafficConfig,
    StochasticTrafficGenerator,
    TrafficBurst,
    TrafficGenerator,
    apply_traffic_scenario,
)


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

    # Canonical traffic scenarios (optional). If set, scenario presets override base traffic fields.
    traffic_scenario: str | None = None
    traffic_scenario_version: int = 2
    traffic_scenario_intensity: float = 1.0
    traffic_scenario_duration: float = 1.0
    traffic_scenario_frequency: float = 1.0

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

    drop_penalty_lambda: float = 40.0
    # ENERGY WEIGHT: Increase to prioritize energy savings more; decrease to prioritize performance (drops).
    energy_weight: float = 25.0

    # QoS constraint: target maximum normalized drop ratio (dropped / (delivered + dropped)).
    # If norm_drop exceeds this, we add an extra penalty (linear by default) so QoS stays stable while optimizing energy.
    qos_target_norm_drop: float = 0.0720

    # Gate QoS penalty until enough traffic volume has been observed.
    qos_min_volume: float = 500.0

    # Strength of the extra penalty when norm_drop exceeds qos_target_norm_drop.
    # For linear QoS penalty, this should be small (tens), otherwise it will dominate reward.
    qos_violation_penalty_scale: float = 80.0

    # When norm_drop_total is close to qos_target_norm_drop, block turning edges OFF.
    qos_guard_margin: float = 0.004  # headroom; tune 0.001–0.005
    qos_guard_penalty_scale: float = 1.0  # multiplier for blocked OFF attempts due to QoS

    toggle_penalty: float = 0.02
    off_toggle_penalty_scale: float = 1.5
    toggle_apply_penalty: float = 0.02
    toggle_on_penalty_scale: float = 0.2
    toggle_off_penalty_scale: float = 5.0
    # Penalize any non-NOOP toggle attempt (encourages the agent to choose NOOP unless confident)
    toggle_attempt_penalty: float = 0.0
    # Small per-step reward when choosing NOOP (encourages the agent to not spam actions)
    noop_bonus: float = 0.0
    debug_logs: bool = False

    blocked_action_penalty: float = 0.0
    revert_penalty_scale: float = 0.5
    top_k_edge_utils: int = 0  # not used yet; placeholder for richer observations
    normalize_drop: bool = True
    saturation_util_threshold: float = 0.9  # for counting near-saturated edges
    toggle_cooldown_steps: int = 10  # minimum steps between toggles of the same edge
    util_block_threshold: float = 0.85  # block turning OFF an edge if its utilization is above this threshold
    global_toggle_cooldown_steps: int = 5  # after any toggle, block all toggles for this many steps (prevents rapid toggling that can lead to instability)
    decision_interval_steps: int = 10  # allow toggles only every N steps; NOOP always allowed
    max_off_toggles_per_episode: int = 6
    max_total_toggles_per_episode: int = 10
    disable_off_actions: bool = False
    initial_off_edges: int = 0
    initial_off_seed: int | None = None



    #configs for forecasting
    enable_forecasting: bool = True
    forecast_alpha: float = 0.3
    forecast_horizon_steps: int = 3
    demand_norm_scale: float = 0.0  # scale factor to keep demand values in a reasonable range for the forecaster


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
    HOLD_ACTION = 0

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.debug_logs = bool(self.config.debug_logs)

        self._step_count: int = 0
        self.simulator: Simulator | None = None
        self._debug_logged: bool = False
        self._printed_toggle_debug: bool = False
        self._printed_mask_debug: bool = False
        self._last_mask_reason_counts: Dict[str, int] = {}
        self._last_valid_on_actions: int = 0
        self._last_valid_off_actions: int = 0
        self._last_valid_toggle_actions: int = 0
        self._initial_off_requested: int = 0
        self._initial_off_applied: int = 0
        self._edge_universe_size: int = 0
        self._traffic_generator: TrafficGenerator | None = None
        self._traffic_by_step: Dict[int, List[TrafficBurst]] = {}
        self._active_bursts: List[Tuple[int, int, float, int]] = []  # (src, dst, demand, remaining_steps)
        # ---- Forecasting state ----
        self._demand_forecaster: EmaDemandForecaster | None = None
        self._demand_norm_scale: float = 1.0
        self._last_demand_now_norm: float = 0.0
        self._last_demand_forecast_norm: float = 0.0
        self._off_toggles_used: int = 0
        self._total_toggles_used: int = 0
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
        # Stable action-edge mapping for the whole run.
        self._edge_universe: List[Tuple[int, int]] = sorted(
            {self._edge_key(int(u), int(v)) for (u, v) in self.edge_list}
        )
        self._edge_to_index: Dict[Tuple[int, int], int] = {
            e: i for i, e in enumerate(self._edge_universe)
        }
        self._edge_universe_size = int(len(self._edge_universe))
        # Backward-compatible alias used by eval diagnostics.
        self.edge_list = list(self._edge_universe)
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
                "demand_now": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "demand_forecast": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
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
        self._last_demand_now_norm = 0.0
        self._last_demand_forecast_norm = 0.0
        self._off_toggles_used = 0
        self._total_toggles_used = 0
        self._printed_toggle_debug = False
        self._printed_mask_debug = False
        self._last_mask_reason_counts = {}
        self._last_valid_on_actions = 0
        self._last_valid_off_actions = 0
        self._last_valid_toggle_actions = 0
        self._initial_off_requested = 0
        self._initial_off_applied = 0
        self._edge_universe_size = int(len(self._edge_universe))

        # ---- Traffic generator setup ----
        self._traffic_by_step = {}
        self._active_bursts = []

        # Derive a deterministic traffic seed when not explicitly provided.
        derived_seed: int | None = self.config.traffic_seed
        if derived_seed is None and seed is not None:
            derived_seed = int(seed) + 10_000  # keep it separate from topology/model seeding

        use_stochastic = self.config.traffic_model.lower() == "stochastic" or bool(self.config.traffic_scenario)
        if use_stochastic:
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
            if self.config.traffic_scenario:
                tcfg = apply_traffic_scenario(
                    tcfg,
                    self.config.traffic_scenario,
                    intensity=float(self.config.traffic_scenario_intensity),
                    duration=float(self.config.traffic_scenario_duration),
                    frequency=float(self.config.traffic_scenario_frequency),
                    version=int(self.config.traffic_scenario_version),
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

        # ---- Forecasting setup ----
        # Normalize demand features to [0,1] so they play nicely with other observations.
        scale = float(getattr(self.config, "demand_norm_scale", 0.0) or 0.0)
        if scale <= 0.0:
            if use_stochastic:
                # Approximate a high-ish offered-demand scale from stochastic config.
                mice_max = float(self.config.traffic_mice_size_range[1])
                ele_max = float(self.config.traffic_elephant_size_range[1])
                p_ele = float(self.config.traffic_p_elephant)
                exp_size = (1.0 - p_ele) * mice_max + p_ele * ele_max
                scale = max(1.0, float(self.config.traffic_avg_bursts_per_step) * exp_size * 2.0)
            else:
                # Uniform mode: flows_per_step * demand_max is a reasonable upper-ish bound.
                scale = max(1.0, float(self.config.flows_per_step) * float(self.config.demand_max))

        self._demand_norm_scale = float(scale)

        if bool(getattr(self.config, "enable_forecasting", True)):
            self._demand_forecaster = EmaDemandForecaster(
                DemandForecastConfig(
                    alpha=float(getattr(self.config, "forecast_alpha", 0.3)),
                    horizon_steps=int(getattr(self.config, "forecast_horizon_steps", 1)),
                )
            )
            self._demand_forecaster.reset(initial=0.0)
        else:
            self._demand_forecaster = None

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

        # Ensure action-edge universe exists in the topology graph so action index -> edge
        # mapping is stable and never "missing" during the episode.
        self._ensure_edge_universe_on_graph(graph)

        if self.debug_logs and not self._debug_logged:
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
            return 30.0 + 30.0 * active_edges

        self.simulator = Simulator(
            graph,
            routing_policy=ShortestPathPolicy(weight="weight"),
            dt_seconds=1.0,
            default_capacity=self.config.base_capacity,
            default_latency_ms=self.config.base_latency_ms,
            power_model_watts=_power_model_watts,
            carbon_intensity_g_per_kwh=lambda t: 400.0 + 10.0 * math.sin(t),
        )

        # Optional initial-condition random OFF edges while preserving connectivity.
        # This creates available ON actions when OFF actions are disabled.
        self._initial_off_requested = int(getattr(self.config, "initial_off_edges", 0) or 0)
        self._initial_off_applied = int(self._apply_safe_initial_off_edges(seed=seed))
        # Initial OFF edges are episode setup, not agent toggles: clear per-edge cooldown
        # so repair ON actions are immediately available on decision steps.
        self._clear_all_edge_cooldowns()
        self._edge_universe_size = int(len(self._edge_universe))
        for (u, v) in self._edge_universe:
            key = self._edge_key(u, v)
            if key not in self.simulator.active:
                self.simulator.active[key] = False
            if key not in self.simulator.capacity:
                self.simulator.capacity[key] = float(self.config.base_capacity)
            if key not in self.simulator.utilization:
                self.simulator.utilization[key] = 0.0
        if self.debug_logs:
            assert len(self._edge_universe) > 0
            assert all(e in self.simulator.active for e in self._edge_universe)

        edge_active, edge_util = self._edge_feature_vectors()
        obs = self._build_observation(
            time_ratio=0.0,
            avg_util=0.0,
            active_ratio=1.0 if len(self.edge_list) > 0 else 0.0,
            max_util=0.0,
            min_util=0.0,
            p95_util=0.0,
            demand_now=0.0,
            demand_forecast=0.0,
            dropped_prev=0.0,
            num_active_edges=float(len(self.edge_list)),
            near_saturated_edges=0.0,
            edge_active=edge_active,
            edge_util=edge_util,
        )
        info: Dict[str, Any] = {"metrics": None}
        return obs, info

    def _apply_safe_initial_off_edges(self, seed: int | None) -> int:
        if self.simulator is None:
            return 0
        requested = int(getattr(self.config, "initial_off_edges", 0) or 0)
        if requested <= 0:
            return 0

        g = self.simulator.graph
        if g.number_of_edges() <= 0:
            return 0

        # Build a protected connected backbone from an undirected view.
        if g.is_directed():
            und = g.to_undirected()
        else:
            und = nx.Graph(g)

        protected_keys: set[Tuple[int, int]] = set()
        if und.number_of_nodes() > 0 and und.number_of_edges() > 0:
            try:
                for (u, v) in nx.minimum_spanning_tree(und).edges():
                    protected_keys.add(self._edge_key(int(u), int(v)))
            except Exception:
                pass

        candidate_edges = [
            (int(u), int(v))
            for (u, v) in self._edge_universe
            if self._edge_key(int(u), int(v)) not in protected_keys
        ]
        if not candidate_edges:
            return 0

        if self.config.initial_off_seed is not None:
            rng = random.Random(int(self.config.initial_off_seed))
            rng.shuffle(candidate_edges)
        else:
            # Derive deterministic randomness from reset seed when provided.
            if seed is not None:
                rng = random.Random(int(seed))
                rng.shuffle(candidate_edges)
            else:
                self.np_random.shuffle(candidate_edges)  # type: ignore[arg-type]

        k = min(int(requested), len(candidate_edges))
        off_applied = 0
        for (u, v) in candidate_edges[:k]:
            key = self._edge_key(u, v)
            prev_state = bool(self.simulator.active.get(key, True))
            if not prev_state:
                continue
            self.simulator.active[key] = False
            if g.has_edge(u, v):
                g.edges[u, v]["active"] = False
                # Keep ON action immediately available at episode start.
                g.edges[u, v]["last_toggled"] = -int(self.config.toggle_cooldown_steps)
            if not self._is_active_graph_connected():
                self.simulator.active[key] = prev_state
                if g.has_edge(u, v):
                    g.edges[u, v]["active"] = prev_state
                continue
            off_applied += 1
            if off_applied >= k:
                break
        return int(off_applied)

    def _clear_all_edge_cooldowns(self) -> None:
        if self.simulator is None:
            return
        reset_last_toggled = -max(1, int(getattr(self.config, "toggle_cooldown_steps", 1) or 1))
        g = self.simulator.graph
        for (u, v) in self._edge_universe:
            if g.has_edge(u, v):
                g.edges[u, v]["last_toggled"] = int(reset_last_toggled)

    def _ensure_edge_universe_on_graph(self, graph: nx.Graph) -> None:
        for (u, v) in self._edge_universe:
            if graph.has_edge(u, v):
                continue
            graph.add_edge(u, v)
            graph.edges[u, v]["capacity"] = float(self.config.base_capacity)
            graph.edges[u, v]["latency_ms"] = float(self.config.base_latency_ms)
            graph.edges[u, v]["weight"] = 1.0
            graph.edges[u, v]["active"] = True
            graph.edges[u, v]["last_toggled"] = -int(self.config.toggle_cooldown_steps)

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        is_decision = self._is_decision_step()
        self._step_count += 1
        if self.debug_logs:
            k = int(getattr(self.config, "decision_interval_steps", 1) or 1)
            expected_now = True if k <= 1 else (int(self._step_count) % k) == 0
            assert bool(is_decision) == bool(expected_now), (
                f"decision-step mismatch at step={self._step_count}: "
                f"is_decision={is_decision} expected={expected_now}"
            )
        if self.simulator is None:
            raise RuntimeError("Simulator not initialized; call reset() first.")
        if getattr(self, "_global_toggle_cooldown_remaining", 0) > 0:
            self._global_toggle_cooldown_remaining -= 1

        # Action diagnostics (helps distinguish true no-op vs invalid index).
        try:
            action_int = int(action)
        except Exception:
            action_int = 0
        # Decision interval safety net (hard guard):
        # Only allow non-NOOP actions every N steps; otherwise force NOOP.
        if not is_decision:
            action_int = 0
        action_is_noop = action_int == self.HOLD_ACTION or action_int <= 0
        idx = action_int - 1
        action_is_invalid = (action_int > 0) and (idx < 0 or idx >= len(self.edge_list) or self.simulator is None or not self.edge_list)

        (
            toggle_cost,
            toggle_applied,
            toggle_reverted,
            toggle_blocked,
            toggle_blocked_high_util,
            toggle_blocked_global,
            toggled_on,
            toggled_off,
            toggle_blocked_off_budget,
            toggle_blocked_total_budget,
        ) = self._apply_action(action_int)
        
        mask_total_cap_blocked = int(
            (action_int != 0)
            and (
                int(getattr(self, "_last_mask_reason_counts", {}).get("total_cap", 0))
                > 0
            )
        )
        toggle_blocked_total_budget = bool(toggle_blocked_total_budget or mask_total_cap_blocked)

        blocked_any = bool(
            toggle_blocked
            or toggle_blocked_high_util
            or toggle_blocked_global
            or toggle_blocked_off_budget
            or toggle_blocked_total_budget
        )
        if blocked_any:
            toggle_cost = float(toggle_cost) + float(self.config.blocked_action_penalty)

        toggles_attempted_count = int((not action_is_noop) and (not action_is_invalid))
        applied_count = int(toggle_applied) if isinstance(toggle_applied, (bool, int, np.integer)) else int(bool(toggle_applied))
        applied_count = max(0, applied_count)
        toggle_penalty_part = float(toggle_cost)
        allowed_toggle_count = int(bool(toggles_attempted_count) and (not blocked_any))
        attempt_penalty_part = float(allowed_toggle_count) * float(self.config.toggle_attempt_penalty)
        toggle_cost = float(toggle_cost) + attempt_penalty_part
        toggles_applied_count = applied_count
        blocked_by_util_count = int(bool(toggle_blocked_high_util))
        blocked_by_cooldown_count = int(bool(toggle_blocked or toggle_blocked_global))
        blocked_by_off_budget_count = int(bool(toggle_blocked_off_budget))
        blocked_by_total_budget_count = int(bool(toggle_blocked_total_budget))

        # ---- Forecasting update (based on offered demand) ----
        flows = self._generate_flows()

        total_demand = float(sum(getattr(f, "demand", 0.0) for f in flows)) if flows else 0.0
        scale = float(getattr(self, "_demand_norm_scale", 1.0) or 1.0)
        demand_now_norm = float(np.clip(total_demand / max(scale, 1e-9), 0.0, 1.0))

        demand_forecast_norm = 0.0
        if bool(getattr(self.config, "enable_forecasting", True)) and self._demand_forecaster is not None:
            self._demand_forecaster.update(total_demand)
            pred = float(self._demand_forecaster.predict())
            demand_forecast_norm = float(np.clip(pred / max(scale, 1e-9), 0.0, 1.0))
        self._last_demand_now_norm = float(demand_now_norm)
        self._last_demand_forecast_norm = float(demand_forecast_norm)

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
            demand_now=demand_now_norm,
            demand_forecast=demand_forecast_norm,
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
        qos_violation = False
        if volume >= float(self.config.qos_min_volume):
            qos_excess = max(0.0, float(norm_drop_total) - float(self.config.qos_target_norm_drop))
            qos_violation = qos_excess > 0.0
        # Cap overshoot to prevent rare spikes from nuking reward (stabilizes training).
        qos_excess = min(qos_excess, 0.05)
        # Linear QoS penalty (less spiky than quadratic; prevents QoS term from dominating reward).
        reward_qos = -float(self.config.qos_violation_penalty_scale) * float(qos_excess)

        toggle_apply_cost = float(self.config.toggle_apply_penalty) * float(toggles_applied_count)
        toggle_on_penalty_applied = float(self.config.toggle_on_penalty_scale) if bool(toggled_on) else 0.0
        toggle_off_penalty_applied = float(self.config.toggle_off_penalty_scale) if bool(toggled_off) else 0.0
        reward_toggle_on = -float(toggle_on_penalty_applied)
        reward_toggle_off = -float(toggle_off_penalty_applied)
        reward_toggle = -(
            float(toggle_cost)
            + float(toggle_apply_cost)
            + float(toggle_on_penalty_applied)
            + float(toggle_off_penalty_applied)
        )
        reward = reward_energy + reward_drop + reward_qos + reward_toggle
        if action_is_noop:
            reward = float(reward) + float(self.config.noop_bonus)
        terminated = False
        truncated = self._step_count >= self.config.max_steps
        if self.debug_logs and not self._printed_toggle_debug:
            print(
                f"[toggle-debug] applied_count={applied_count} attempted_count={toggles_attempted_count} "
                f"toggle_penalty_part={toggle_penalty_part:.6f} attempt_penalty_part={attempt_penalty_part:.6f} "
                f"toggle_cost={float(toggle_cost):.6f}"
            )
            self._printed_toggle_debug = True
        info: Dict[str, Any] = {
            "metrics": metrics,
            "flows": flows,
            "toggle_applied": toggle_applied,
            "toggle_reverted": toggle_reverted,
            "toggle_blocked_cooldown": toggle_blocked,
            "toggle_blocked_high_util": toggle_blocked_high_util,
            "toggle_blocked_global_cooldown": toggle_blocked_global,
            "toggle_blocked_off_budget": bool(toggle_blocked_off_budget),
            "toggle_blocked_total_budget": int(bool(toggle_blocked_total_budget)),
            "toggle_blocked_any": blocked_any,
            "toggled_on": bool(toggled_on),
            "toggled_off": bool(toggled_off),
            "blocked_by_util_count": blocked_by_util_count,
            "blocked_by_cooldown_count": blocked_by_cooldown_count,
            "blocked_by_off_budget_count": blocked_by_off_budget_count,
            "blocked_by_total_budget_count": blocked_by_total_budget_count,
            "allowed_toggle_count": allowed_toggle_count,
            "toggles_attempted_count": toggles_attempted_count,
            "toggles_applied_count": toggles_applied_count,
            "off_toggles_used": int(getattr(self, "_off_toggles_used", 0)),
            "off_toggles_cap": int(getattr(self.config, "max_off_toggles_per_episode", 0)),
            "total_toggles_used": int(getattr(self, "_total_toggles_used", 0)),
            "total_toggles_cap": int(getattr(self.config, "max_total_toggles_per_episode", 0)),
            "on_edges_count": int(sum(1 for e in self._edge_universe if bool(self.simulator.active.get(e, False)))),
            "off_edges_count": int(sum(1 for e in self._edge_universe if not bool(self.simulator.active.get(e, False)))),
            "toggle_budget_remaining": (
                (int(getattr(self.config, "max_total_toggles_per_episode", 0)) - int(getattr(self, "_total_toggles_used", 0)))
                if int(getattr(self.config, "max_total_toggles_per_episode", 0)) > 0
                else None
            ),
            "edge_universe_size": int(getattr(self, "_edge_universe_size", len(self._edge_universe))),
            "initial_off_requested": int(getattr(self, "_initial_off_requested", 0)),
            "initial_off_applied": int(getattr(self, "_initial_off_applied", 0)),
            "reward_energy": reward_energy,
            "reward_drop": reward_drop,
            "delta_energy_kwh": float(delta_energy),
            "delta_delivered": float(delta_delivered),
            "delta_dropped": float(delta_dropped),
            "norm_drop_step": float(norm_drop_step),
            "norm_drop": float(norm_drop_total),
            "reward_qos": reward_qos,
            "reward_toggle": reward_toggle,
            "reward_toggle_on": reward_toggle_on,
            "reward_toggle_off": reward_toggle_off,
            "toggle_on_penalty_applied": float(toggle_on_penalty_applied),
            "toggle_off_penalty_applied": float(toggle_off_penalty_applied),
            "total_reward": reward,
            "qos_excess": float(qos_excess),
            "qos_violation": bool(qos_violation),
            "action_int": action_int,
            "action_is_noop": action_is_noop,
            "noop_action_valid": 1,
            "noop_chosen": int(bool(action_is_noop)),
            "action_is_invalid": action_is_invalid,
            "is_decision_step": bool(is_decision),
            "mask_reason_counts": dict(getattr(self, "_last_mask_reason_counts", {})),
            "valid_on_actions": int(getattr(self, "_last_valid_on_actions", 0)),
            "valid_off_actions": int(getattr(self, "_last_valid_off_actions", 0)),
            "valid_toggle_actions": int(getattr(self, "_last_valid_toggle_actions", 0)),
            "valid_noop_actions": 1,
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
        if self.debug_logs and not self._printed_mask_debug:
            print("[mask-debug] get_action_mask() called")
            self._printed_mask_debug = True
        n = int(self.action_space.n)
        mask = np.ones((n,), dtype=bool)
        reasons: Dict[str, int] = {
            "interval": 0,
            "sim_none_or_no_edges": 0,
            "invalid_index": 0,
            "missing_edge": 0,
            "bridge": 0,
            "global_cd": 0,
            "off_disabled": 0,
            "off_start_guard": 0,
            "off_cap": 0,
            "demand_gate": 0,
            "qos_guard": 0,
            "util_gate": 0,
            "edge_cooldown": 0,
            "total_cap": 0,
            "on_disabled_by_interval": 0,
            "on_disabled_by_missing_edge": 0,
            "on_disabled_by_invalid_index": 0,
            "on_disabled_by_edge_cooldown": 0,
            "on_disabled_by_global_cd": 0,
            "on_disabled_by_sim_none_or_no_edges": 0,
        }
        if n <= 0:
            self._last_mask_reason_counts = reasons
            self._last_valid_on_actions = 0
            self._last_valid_off_actions = 0
            self._last_valid_toggle_actions = 0
            return mask
        
        # Always allow NOOP
        mask[0] = True
        # Decision interval: only allow toggles every N steps to prevent action spam.
        if not self._is_decision_step():
            mask[1:] = False
            reasons["interval"] = max(0, n - 1)
            reasons["on_disabled_by_interval"] = max(0, n - 1)
            self._last_mask_reason_counts = reasons
            self._last_valid_on_actions = 0
            self._last_valid_off_actions = 0
            self._last_valid_toggle_actions = 0
            return mask

        if self.simulator is None or not self._edge_universe:
            mask[1:] = False
            reasons["sim_none_or_no_edges"] = max(0, n - 1)
            reasons["on_disabled_by_sim_none_or_no_edges"] = max(0, n - 1)
            self._last_mask_reason_counts = reasons
            self._last_valid_on_actions = 0
            self._last_valid_off_actions = 0
            self._last_valid_toggle_actions = 0
            return mask

        total_cap = int(getattr(self.config, "max_total_toggles_per_episode", 0))
        total_used = int(getattr(self, "_total_toggles_used", 0))
        if total_cap > 0 and total_used >= total_cap:
            mask[1:] = False
            reasons["total_cap"] = max(0, n - 1)
            self._last_mask_reason_counts = reasons
            self._last_valid_on_actions = 0
            self._last_valid_off_actions = 0
            self._last_valid_toggle_actions = 0
            return mask
        
        global_cd = int(getattr(self, "_global_toggle_cooldown_remaining", 0))
        util_values = list(self.simulator.utilization.values()) if self.simulator else []
        max_util = max(util_values, default=0.0)
        for a in range(1, n):
            idx = a - 1
            if idx < 0 or idx >= len(self._edge_universe):
                mask[a] = False
                reasons["invalid_index"] += 1
                reasons["on_disabled_by_invalid_index"] += 1
                continue
        
            edge = self._edge_universe[idx]
            key = self._edge_key(edge[0], edge[1])
            if key not in self.simulator.active:
                mask[a] = False
                reasons["missing_edge"] += 1
                reasons["on_disabled_by_missing_edge"] += 1
                if self.debug_logs and reasons["on_disabled_by_missing_edge"] <= 5:
                    rev_exists = self._edge_key(edge[1], edge[0]) in self.simulator.active
                    print(
                        f"[on-missing] action={a} edge={edge} key={key} reversed_exists={rev_exists}"
                    )
                continue
            current_state = bool(self.simulator.active.get(key, False))
            # Global cooldown blocks only when trying to turn an edge OFF
            if current_state and global_cd > 0:
                mask[a] = False
                reasons["global_cd"] += 1
                continue

            # High-util gate blocks only when trying to turn an edge OFF
            if current_state:
                # In normal all-on mode (no seeded OFF edges), keep links ON by default.
                # This prevents rare harmful OFF toggles that make policy worse than NOOP.
                if (
                    int(getattr(self.config, "initial_off_edges", 0) or 0) <= 0
                    and (not bool(getattr(self.config, "disable_off_actions", False)))
                ):
                    mask[a] = False
                    reasons["off_start_guard"] += 1
                    continue
                if bool(getattr(self.config, "disable_off_actions", False)):
                    mask[a] = False
                    reasons["off_disabled"] += 1
                    continue
                # OFF budget: if we've already used our allowed OFF toggles this episode,
                # block further OFF actions.
                cap = int(getattr(self.config, "max_off_toggles_per_episode", 0))
                used = int(getattr(self, "_off_toggles_used", 0))
                if cap > 0 and used >= cap:
                    mask[a] = False
                    reasons["off_cap"] += 1
                    continue
                # NEW: Demand/forecast gate — only allow OFF when demand is low.
                dn = float(getattr(self, "_last_demand_now_norm", 0.0))
                df = float(getattr(self, "_last_demand_forecast_norm", 0.0))

                # If current OR forecast demand is high, do not allow turning edges OFF.
                if (dn >= 0.70) or (df >= 0.70):
                    mask[a] = False
                    reasons["demand_gate"] += 1
                    continue
                # QoS guard: block turning an edge OFF when QoS is near/over target.
                last_nd = float(getattr(self, "_last_norm_drop_total", 0.0))
                target = float(self.config.qos_target_norm_drop)
                margin = float(getattr(self.config, "qos_guard_margin", 0.0))
                if last_nd >= (target - margin):
                    mask[a] = False
                    reasons["qos_guard"] += 1
                    continue
                util = float(self.simulator.utilization.get(key, 0.0))
                # Block turning OFF when utilization is ABOVE the threshold.
                if util > float(self.config.util_block_threshold):
                    mask[a] = False
                    reasons["util_gate"] += 1
                    continue
            
            # Per-edge cooldown: for ON-only experiments, allow ON actions immediately.
            apply_cooldown = current_state or (not bool(getattr(self.config, "disable_off_actions", False)))
            if apply_cooldown:
                last_toggled = self.simulator.graph.edges[edge].get(
                    "last_toggled", -self.config.toggle_cooldown_steps
                )
                if (self._step_count - last_toggled) < int(self.config.toggle_cooldown_steps):
                    mask[a] = False
                    reasons["edge_cooldown"] += 1
                    if not current_state:
                        reasons["on_disabled_by_edge_cooldown"] += 1
                    continue
            mask[a] = True
        valid_on = 0
        valid_off = 0
        for a in range(1, n):
            if not bool(mask[a]):
                continue
            idx = a - 1
            if idx < 0 or idx >= len(self._edge_universe):
                continue
            edge = self._edge_universe[idx]
            key = self._edge_key(edge[0], edge[1])
            current_state = bool(self.simulator.active.get(key, False))
            if current_state:
                valid_off += 1
            else:
                valid_on += 1
        self._last_valid_on_actions = int(valid_on)
        self._last_valid_off_actions = int(valid_off)
        self._last_valid_toggle_actions = int(valid_on + valid_off)
        self._last_mask_reason_counts = reasons
        return mask

    def _is_decision_step(self) -> bool:
        k = int(getattr(self.config, "decision_interval_steps", 1) or 1)
        if k <= 1:
            return True
        # IMPORTANT: mask is queried BEFORE step(); step() increments _step_count.
        # So decision should be based on the next step index.
        next_step = int(self._step_count) + 1
        return (next_step % k) == 0



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
        demand_now: float,
        demand_forecast: float,
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
            "demand_now": np.array([np.float32(demand_now)], dtype=np.float32),
            "demand_forecast": np.array([np.float32(demand_forecast)], dtype=np.float32),
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

        for idx, (u, v) in enumerate(self._edge_universe):
            key = self._edge_key(u, v)
            is_active = bool(self.simulator.active.get(key, False))
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

    def _graph_has_edge_any(self, u: int, v: int) -> bool:
        if self.simulator is None:
            return False
        g = self.simulator.graph
        if g.has_edge(u, v):
            return True
        if g.has_edge(v, u):
            return True
        return False

    def _apply_action(self, action: Any) -> Tuple[float, bool, bool, bool, bool, bool, bool, bool, bool, bool]:
        """Toggle an edge if requested.

        Returns:
          (toggle_cost, applied, reverted, cooldown_blocked, high_util_blocked,
           global_cooldown_blocked, toggled_on, toggled_off, off_budget_blocked,
           total_budget_blocked).
        """
        if self.simulator is None or not self._edge_universe:
            return 0.0, False, False, False, False, False, False, False, False, False

        try:
            action_int = int(action)
        except Exception:
            action_int = 0

        if action_int <= 0:
            return 0.0, False, False, False, False, False, False, False, False, False

        idx = action_int - 1
        if idx < 0 or idx >= len(self._edge_universe):
            return 0.0, False, False, False, False, False, False, False, False, False

        edge = self._edge_universe[idx]
        key = self._edge_key(edge[0], edge[1])

        if key not in self.simulator.active:
            return 0.0, False, False, False, False, False, False, False, False, False

        current_state = bool(self.simulator.active.get(key, False))
        toggled_on = False
        toggled_off = False
        off_budget_blocked = False
        total_budget_blocked = False

        # Hard safety mirror of the mask: in normal all-on mode, do not allow OFF toggles.
        if (
            current_state
            and int(getattr(self.config, "initial_off_edges", 0) or 0) <= 0
            and (not bool(getattr(self.config, "disable_off_actions", False)))
        ):
            return 0.0, False, False, False, False, False, False, False, False, False

        # Total toggle budget hard gate: treat as NOOP if exhausted.
        total_cap = int(getattr(self.config, "max_total_toggles_per_episode", 0))
        total_used = int(getattr(self, "_total_toggles_used", 0))
        if total_cap > 0 and total_used >= total_cap:
            total_budget_blocked = True
            return 0.0, False, False, False, False, False, False, False, False, total_budget_blocked

        # OFF budget hard gate: if edge is currently ON, this action would turn it OFF.
        if current_state:
            cap = int(getattr(self.config, "max_off_toggles_per_episode", 0))
            used = int(getattr(self, "_off_toggles_used", 0))
            if cap > 0 and used >= cap:
                off_budget_blocked = True
                penalty = self.config.toggle_penalty * self.config.revert_penalty_scale
                return penalty, False, False, False, False, False, False, False, off_budget_blocked, False

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
                return penalty, False, False, False, False, False, False, False, False, False

        # Global cooldown: block rapid toggles ONLY when turning an edge OFF.
        # Turning an edge ON is always allowed so the agent can recover quickly.
        if current_state and getattr(self, "_global_toggle_cooldown_remaining", 0) > 0:
            penalty = self.config.toggle_penalty * self.config.revert_penalty_scale
            return penalty, False, False, False, False, True, False, False, False, False

        # Safety gate: block turning OFF a highly utilized edge.
        # (Turning ON an edge is always allowed.)
        if current_state:
            util = float(self.simulator.utilization.get(key, 0.0))
            # Block turning OFF when utilization is ABOVE the threshold.
            if util > self.config.util_block_threshold:
                penalty = self.config.toggle_penalty * self.config.revert_penalty_scale
                return penalty, False, False, False, True, False, False, False, False, False

        # Cooldown: prevent rapid flapping of the same edge.
        last_toggled = self.simulator.graph.edges[edge].get("last_toggled", -self.config.toggle_cooldown_steps)
        if (self._step_count - last_toggled) < self.config.toggle_cooldown_steps:
            penalty = self.config.toggle_penalty * self.config.revert_penalty_scale
            return penalty, False, False, True, False, False, False, False, False, False

        new_state = not current_state

        # Apply toggle.
        self.simulator.active[key] = new_state
        if self._graph_has_edge_any(*edge):
            self.simulator.graph.edges[edge]["active"] = new_state
            self.simulator.graph.edges[edge]["last_toggled"] = self._step_count

        # Safety: ensure graph stays connected via active edges.
        if not self._is_active_graph_connected():
            # Revert
            self.simulator.active[key] = current_state
            if self._graph_has_edge_any(*edge):
                self.simulator.graph.edges[edge]["active"] = current_state
            penalty = self.config.toggle_penalty * self.config.revert_penalty_scale
            self._global_toggle_cooldown_remaining = int(self.config.global_toggle_cooldown_steps)
            return penalty, False, True, False, False, False, False, False, False, False

        toggle_cost = self.config.toggle_penalty
        # Turning OFF is riskier than turning ON -> apply extra cost on OFF only.
        if new_state is False:
            toggle_cost = float(toggle_cost) * float(getattr(self.config, "off_toggle_penalty_scale", 1.0))

        # Start global cooldown ONLY after successfully turning an edge OFF.
        if new_state is False:
            self._global_toggle_cooldown_remaining = int(self.config.global_toggle_cooldown_steps)
            toggled_off = True
            self._off_toggles_used += 1
        else:
            toggled_on = True

        self._total_toggles_used += 1
        return toggle_cost, True, False, False, False, False, toggled_on, toggled_off, off_budget_blocked, False

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
