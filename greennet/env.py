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

    drop_penalty_lambda: float = 50.0
    # ENERGY WEIGHT: Increase to prioritize energy savings more; decrease to prioritize performance (drops).
    energy_weight: float = 40.0

    # QoS constraint: target maximum normalized drop ratio (dropped / (delivered + dropped)).
    # If norm_drop exceeds this, we add an extra penalty (linear by default) so QoS stays stable while optimizing energy.
    qos_target_norm_drop: float = 0.0720

    # Gate QoS penalty until enough traffic volume has been observed.
    qos_min_volume: float = 500.0

    # Strength of the extra penalty when norm_drop exceeds qos_target_norm_drop.
    # For linear QoS penalty, this should be small (tens), otherwise it will dominate reward.
    qos_violation_penalty_scale: float = 120.0

    # When norm_drop_total is close to qos_target_norm_drop, block turning edges OFF.
    qos_guard_margin: float = 0.004  # headroom; tune 0.001–0.005
    # Split margins:
    # - OFF brake should trigger earlier (larger margin)
    # - ON recovery should trigger easier (smaller margin)
    qos_guard_margin_off: float = 0.004
    qos_guard_margin_on: float = 0.002
    qos_guard_penalty_scale: float = 4.5  # multiplier for blocked OFF attempts due to QoS

    toggle_penalty: float = 0.02
    off_toggle_penalty_scale: float = 1.5
    toggle_apply_penalty: float = 0.02
    toggle_on_penalty_scale: float = 0.2
    # Discount ON-toggle costs when QoS is violated so recovery actions are not over-penalized.
    qos_toggle_discount_on: float = 0.25
    # Increase OFF-toggle costs when the network is calm to discourage unnecessary toggles.
    calm_toggle_multiplier_off: float = 2.0
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
    max_util_off_allow_threshold: float = 0.80  # block turning OFF any edge when global max utilization is high
    util_unblock_threshold: float = 0.90  # allow turning ON an edge only when utilization is above this threshold (or QoS violated)
    global_toggle_cooldown_steps: int = 5  # after any toggle, block all toggles for this many steps (prevents rapid toggling that can lead to instability)
    decision_interval_steps: int = 10  # allow toggles only every N steps; NOOP always allowed
    max_off_toggles_per_episode: int = 1
    max_total_toggles_per_episode: int = 4
    max_emergency_on_toggles_per_episode: int = 8
    off_calm_steps_required: int = 20
    disable_off_actions: bool = False
    initial_off_edges: int = 3
    initial_off_seed: int | None = 123
    off_start_guard_decision_steps: int = 10  # block OFF actions for first N decision steps when starting all-on




    #configs for forecasting
    enable_forecasting: bool = True
    forecast_alpha: float = 0.6
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
        self._episode_initial_off_edges: int = 0
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
        self._emergency_on_toggles_used: int = 0
        self._calm_streak: int = 0
        self._decision_step_count: int = 0
        self._mask_calls: int = 0
        self._last_qos_viol_step: bool = False
        self._dbg_cd_block_mask_val_last: int = -999
        self._dbg_cd_block_mask_samples: int = 0
        self._dbg_cd_block_mask_mismatches: int = 0
        self._dbg_block_edge_cd: int = 0
        self._dbg_block_global_cd: int = 0
        self._dbg_block_budget: int = 0
        self._dbg_block_util: int = 0
        self._dbg_block_off_stress: int = 0
        self._dbg_block_qos_off: int = 0
        self._dbg_block_qos_on: int = 0


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
        self._last_norm_drop_step = 0.0
        self._last_qos_allow_on = False
        self._last_qos_viol_step = False
        self._last_max_util = 0.0
        self._last_demand_now_norm = 0.0
        self._last_demand_forecast_norm = 0.0
        self._off_toggles_used = 0
        self._total_toggles_used = 0
        self._emergency_on_toggles_used = 0
        self._calm_streak = 0
        self._decision_step_count = 0
        self._mask_calls = 0
        self._dbg_cd_block_mask_val_last = -999
        self._dbg_cd_block_mask_samples = 0
        self._dbg_cd_block_mask_mismatches = 0
        self._dbg_block_edge_cd = 0
        self._dbg_block_global_cd = 0
        self._dbg_block_budget = 0
        self._dbg_block_util = 0
        self._dbg_block_off_stress = 0
        self._dbg_block_qos_off = 0
        self._dbg_block_qos_on = 0
        self._printed_toggle_debug = False
        self._printed_mask_debug = False
        self._last_mask_reason_counts = {}
        self._last_valid_on_actions = 0
        self._last_valid_off_actions = 0
        self._last_valid_toggle_actions = 0
        self._initial_off_requested = 0
        self._initial_off_applied = 0
        self._episode_initial_off_edges = 0
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
            # Stronger signal so energy savings can beat toggle cost when OFF persists.
            active_edges = sum(1 for (a, b) in g.edges() if g.edges[a, b].get("active", True))
            return 40.0 + 200.0 * active_edges

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
        self._episode_initial_off_edges = int(self._initial_off_applied)
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
        if bool(is_decision):
            self._decision_step_count = int(getattr(self, "_decision_step_count", 0)) + 1

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
        toggle_blocked_global_off_stress = bool(
            getattr(self, "_last_toggle_blocked_global_off_stress", False)
        )
        qos_allow_on = bool(getattr(self, "_last_qos_allow_on", False))
        emergency_on_applied = bool(getattr(self, "_last_emergency_on_applied", False))
        dbg_cd_block_mask_val_last = int(getattr(self, "_dbg_cd_block_mask_val_last", -999))
        dbg_cd_block_mask_samples = int(getattr(self, "_dbg_cd_block_mask_samples", 0))
        dbg_cd_block_mask_mismatches = int(getattr(self, "_dbg_cd_block_mask_mismatches", 0))
        
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
        blocked_by_global_off_stress_count = int(bool(toggle_blocked_global_off_stress))
        blocked_by_budget_count = int(bool(toggle_blocked_off_budget or toggle_blocked_total_budget))
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
        self._last_max_util = float(max_util)
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
        self._last_norm_drop_step = float(norm_drop_step)
        qos_target = float(getattr(self.config, "qos_target_norm_drop", 0.0))

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
            qos_excess = max(0.0, float(norm_drop_total) - float(qos_target))
            qos_violation = qos_excess > 0.0
        # Canonical per-step QoS-violation flag used by both penalty and ON-recovery gates.
        self._last_qos_viol_step = bool(qos_violation)
        self._last_qos_allow_on = bool(self._last_qos_viol_step)
        calm_now = (not self._last_qos_viol_step) and (
            float(getattr(self, "_last_max_util", 0.0))
            < float(getattr(self.config, "util_block_threshold", 0.85))
        )
        self._calm_streak = int(getattr(self, "_calm_streak", 0) + 1) if calm_now else 0
        # Cap overshoot to prevent rare spikes from nuking reward (stabilizes training).
        qos_excess = min(qos_excess, 0.05)
        # Linear QoS penalty (less spiky than quadratic; prevents QoS term from dominating reward).
        reward_qos = -float(self.config.qos_violation_penalty_scale) * float(qos_excess)

        toggle_apply_cost = float(self.config.toggle_apply_penalty) * float(toggles_applied_count)
        toggle_on_penalty_applied = float(self.config.toggle_on_penalty_scale) if bool(toggled_on) else 0.0
        toggle_off_penalty_applied = float(self.config.toggle_off_penalty_scale) if bool(toggled_off) else 0.0
        # In QoS-violation regime, discount ON-toggle costs to favor recovery over prolonged violations.
        qos_toggle_discount_on = float(getattr(self.config, "qos_toggle_discount_on", 1.0))
        discount_on = bool(getattr(self, "_last_qos_viol_step", False)) and bool(toggled_on)
        if discount_on and qos_toggle_discount_on < 1.0:
            toggle_cost = float(toggle_cost) * float(qos_toggle_discount_on)
            toggle_apply_cost = float(toggle_apply_cost) * float(qos_toggle_discount_on)
            toggle_on_penalty_applied = float(toggle_on_penalty_applied) * float(qos_toggle_discount_on)
        calm_toggle_multiplier_off = float(getattr(self.config, "calm_toggle_multiplier_off", 1.0))
        calm_for_off = (not bool(getattr(self, "_last_qos_viol_step", False))) and (
            float(getattr(self, "_last_max_util", 0.0))
            < float(getattr(self.config, "util_block_threshold", 0.85))
        )
        discount_off_calm = bool(calm_for_off and bool(toggled_off) and calm_toggle_multiplier_off > 1.0)
        if discount_off_calm:
            toggle_cost = float(toggle_cost) * float(calm_toggle_multiplier_off)
            toggle_apply_cost = float(toggle_apply_cost) * float(calm_toggle_multiplier_off)
            toggle_off_penalty_applied = float(toggle_off_penalty_applied) * float(calm_toggle_multiplier_off)
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
        reward_other = float(reward) - (
            float(reward_energy)
            + float(reward_drop)
            + float(reward_qos)
            + float(reward_toggle)
        )
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
            "toggle_blocked_global_off_stress": bool(toggle_blocked_global_off_stress),
            "toggle_blocked_off_budget": bool(toggle_blocked_off_budget),
            "toggle_blocked_total_budget": int(bool(toggle_blocked_total_budget)),
            "toggle_blocked_any": blocked_any,
            "qos_allow_on": bool(qos_allow_on),
            "dbg_cd_block_mask_val": int(dbg_cd_block_mask_val_last),
            "dbg_cd_block_mask_val_last": int(dbg_cd_block_mask_val_last),
            "dbg_cd_block_mask_samples": int(dbg_cd_block_mask_samples),
            "dbg_cd_block_mask_mismatches": int(dbg_cd_block_mask_mismatches),
            "cd_block_mask_mismatch_count": int(dbg_cd_block_mask_mismatches),
            "attempted_toggle": int(toggles_attempted_count),
            "toggle_allowed": int(allowed_toggle_count),
            "toggle_blocked_cd": int(bool(toggle_blocked or toggle_blocked_global)),
            "toggle_cost": float(toggle_cost),
            "toggle_apply_cost": float(toggle_apply_cost),
            "on_penalty": float(toggle_on_penalty_applied),
            "off_penalty": float(toggle_off_penalty_applied),
            "qos_toggle_discount_on_applied": int(bool(discount_on and qos_toggle_discount_on < 1.0)),
            "qos_toggle_discount_on": float(qos_toggle_discount_on),
            "calm_toggle_multiplier_off_applied": int(bool(discount_off_calm)),
            "calm_toggle_multiplier_off": float(calm_toggle_multiplier_off),
            "dbg_block_edge_cd": int(getattr(self, "_dbg_block_edge_cd", 0)),
            "dbg_block_global_cd": int(getattr(self, "_dbg_block_global_cd", 0)),
            "dbg_block_budget": int(getattr(self, "_dbg_block_budget", 0)),
            "dbg_block_util": int(getattr(self, "_dbg_block_util", 0)),
            "dbg_block_off_stress": int(getattr(self, "_dbg_block_off_stress", 0)),
            "dbg_block_qos_off": int(getattr(self, "_dbg_block_qos_off", 0)),
            "dbg_block_qos_on": int(getattr(self, "_dbg_block_qos_on", 0)),
            "toggled_on": bool(toggled_on),
            "toggled_off": bool(toggled_off),
            "blocked_by_util_count": blocked_by_util_count,
            "blocked_by_cooldown_count": blocked_by_cooldown_count,
            "blocked_by_global_off_stress_count": blocked_by_global_off_stress_count,
            "blocked_by_budget_count": blocked_by_budget_count,
            "blocked_by_off_budget_count": blocked_by_off_budget_count,
            "blocked_by_total_budget_count": blocked_by_total_budget_count,
            "allowed_toggle_count": allowed_toggle_count,
            "toggles_attempted_count": toggles_attempted_count,
            "toggles_applied_count": toggles_applied_count,
            "emergency_on_applied_count": int(bool(emergency_on_applied)),
            "off_toggles_used": int(getattr(self, "_off_toggles_used", 0)),
            "off_toggles_cap": int(getattr(self.config, "max_off_toggles_per_episode", 0)),
            "total_toggles_used": int(getattr(self, "_total_toggles_used", 0)),
            "total_toggles_cap": int(getattr(self.config, "max_total_toggles_per_episode", 0)),
            "emergency_on_toggles_used": int(getattr(self, "_emergency_on_toggles_used", 0)),
            "emergency_on_toggles_cap": int(getattr(self.config, "max_emergency_on_toggles_per_episode", 0)),
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
            "max_util": float(self._last_max_util),
            "total_reward": reward,
            "r_total": float(reward),
            "r_energy": float(reward_energy),
            "r_drop": float(reward_drop),
            "r_qos": float(reward_qos),
            "r_toggle": float(reward_toggle),
            "r_other": float(reward_other),
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
            "mask_calls": int(getattr(self, "_mask_calls", 0)),
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
        self._mask_calls = int(getattr(self, "_mask_calls", 0)) + 1
        if self.debug_logs and not self._printed_mask_debug:
            print("[mask-debug] get_action_mask() called")
            self._printed_mask_debug = True
        n = int(self.action_space.n)
        mask = np.ones((n,), dtype=bool)
        decision_idx = int(getattr(self, "_decision_step_count", 0))
        warmup_decisions = int(getattr(self.config, "off_start_guard_decision_steps", 10))

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
            "off_all_on_calm": 0,
            "off_calm_streak": 0,
            "demand_gate": 0,
            "qos_guard": 0,
            "off_stress_gate": 0,
            "util_gate": 0,
            "edge_cooldown": 0,
            "total_cap": 0,
            "on_disabled_by_interval": 0,
            "on_disabled_by_missing_edge": 0,
            "on_disabled_by_invalid_index": 0,
            "on_disabled_by_edge_cooldown": 0,
            "on_disabled_by_global_cd": 0,
            "on_disabled_by_qos_gate": 0,
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
        emerg_cap = int(getattr(self.config, "max_emergency_on_toggles_per_episode", 8))
        emerg_used = int(getattr(self, "_emergency_on_toggles_used", 0))
        
        global_cd = int(getattr(self, "_global_toggle_cooldown_remaining", 0))
        nd_signal = float(getattr(self, "_last_norm_drop_step", getattr(self, "_last_norm_drop_total", 0.0)))
        target = float(self.config.qos_target_norm_drop)
        margin_off = float(getattr(self.config, "qos_guard_margin_off", getattr(self.config, "qos_guard_margin", 0.0)))
        qos_viol_step = bool(getattr(self, "_last_qos_viol_step", False))
        qos_allow_on = bool(qos_viol_step)
        qos_block_off = bool(nd_signal > (target - margin_off))
        last_max_util = float(getattr(self, "_last_max_util", 0.0))
        calm_streak = int(getattr(self, "_calm_streak", 0))
        off_calm_steps_required = int(getattr(self.config, "off_calm_steps_required", 0))
        max_util_off_allow_threshold = float(getattr(self.config, "max_util_off_allow_threshold", 0.80))
        # "No-first-OFF in calm all-on state":
        # if episode started all-on and we are still all-on + calm, mask OFF actions.
        episode_started_all_on = int(getattr(self, "_episode_initial_off_edges", 0)) == 0
        off_edges_now = int(
            sum(1 for e in self._edge_universe if not bool(self.simulator.active.get(e, True)))
        )
        all_on_and_calm = bool(
            episode_started_all_on
            and off_edges_now == 0
            and (not qos_viol_step)
            and (last_max_util < float(getattr(self.config, "util_block_threshold", 0.80)))
        )
        util_unblock_threshold = float(
            getattr(
                self.config,
                "util_unblock_threshold",
                getattr(self.config, "util_block_threshold", 0.80),
            )
        )
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
            next_state = (not current_state)
            is_turning_on = bool((not current_state) and next_state)
            emergency_on_candidate = bool(is_turning_on and qos_viol_step)
            emergency_cap_ok = bool(emerg_cap <= 0 or emerg_used < emerg_cap)
            # During QoS emergency, keep OFF->ON visible even if cooldowns are active.
            emergency_on_mask_bypass = bool(is_turning_on and qos_allow_on and emergency_cap_ok)
            if total_cap > 0 and total_used >= total_cap and not emergency_on_candidate:
                mask[a] = False
                reasons["total_cap"] += 1
                if not current_state:
                    reasons["on_disabled_by_global_cd"] += 1
                continue
            if emergency_on_candidate and emerg_cap > 0 and emerg_used >= emerg_cap:
                mask[a] = False
                reasons["total_cap"] += 1
                if not current_state:
                    reasons["on_disabled_by_global_cd"] += 1
                continue
            # Global cooldown blocks only OFF toggles; emergency ON can bypass it.
            if global_cd > 0 and current_state and not emergency_on_mask_bypass:
                mask[a] = False
                reasons["global_cd"] += 1
                if not current_state:
                    reasons["on_disabled_by_global_cd"] += 1
                continue

            # High-util gate blocks only when trying to turn an edge OFF
            if current_state:
                if all_on_and_calm:
                    mask[a] = False
                    reasons["off_all_on_calm"] += 1
                    continue
                # In normal all-on mode (no seeded OFF edges), keep links ON by default.
                # This prevents rare harmful OFF toggles that make policy worse than NOOP.
                # OFF start guard: when starting all-on (initial_off_edges==0),
                # keep OFF actions masked only for the first N decision steps.
                if int(getattr(self.config, "initial_off_edges", 0)) == 0 and decision_idx < warmup_decisions:
                    for a in range(1, n):
                        if not bool(mask[a]):
                            continue
                        idx = int(a) - 1
                        if idx < 0 or idx >= len(self.edge_list):
                            continue
                        if self.simulator is None:
                            continue
                        u, v = self.edge_list[idx]
                        key = self._edge_key(int(u), int(v))
                        current_state = bool(self.simulator.active.get(key, True))
                        # current_state==True means this toggle would turn the edge OFF
                        if current_state:
                            mask[a] = False
                            reasons["off_start_guard"] = reasons.get("off_start_guard", 0) + 1
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
                if off_calm_steps_required > 0 and calm_streak < off_calm_steps_required:
                    mask[a] = False
                    reasons["off_calm_streak"] += 1
                    continue
                # Hard OFF safety: when QoS is violated, never allow turning edges OFF.
                if qos_viol_step:
                    mask[a] = False
                    reasons["qos_guard"] += 1
                    continue
                # Global calm OFF gate at mask-time to avoid unsafe OFF attempts.
                if last_max_util >= max_util_off_allow_threshold:
                    mask[a] = False
                    reasons["off_stress_gate"] += 1
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
                if nd_signal > (target - margin_off):
                    mask[a] = False
                    reasons["qos_guard"] += 1
                    continue
                util = float(self.simulator.utilization.get(key, 0.0))
                # Block turning OFF when utilization is ABOVE the threshold.
                if util > float(self.config.util_block_threshold):
                    mask[a] = False
                    reasons["util_gate"] += 1
                    continue
            
            # Per-edge cooldown blocks all normal toggles; emergency ON can bypass it.
            if not emergency_on_mask_bypass:
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
        toggle_cost = 0.0
        toggle_applied = False
        toggle_reverted = False
        toggle_blocked = False
        toggle_blocked_high_util = False
        toggle_blocked_global = False
        toggle_blocked_global_off_stress = False
        toggled_on = False
        toggled_off = False
        off_budget_blocked = False
        total_budget_blocked = False

        idx = -1
        edge: Tuple[int, int] | None = None
        key: Tuple[int, int] | None = None
        prev_state: bool = True
        next_state: bool = True
        qos_allow_on = bool(getattr(self, "_last_qos_viol_step", False))
        emergency_on = False
        emergency_on_applied = False

        try:
            action_int = int(action)
        except Exception:
            action_int = 0

        action_invalid = bool(
            action_int > 0
            and (
                self.simulator is None
                or not self.edge_list
                or (action_int - 1) < 0
                or (action_int - 1) >= len(self.edge_list)
            )
        )

        def _finish() -> Tuple[float, bool, bool, bool, bool, bool, bool, bool, bool, bool]:
            self._last_toggle_blocked_global_off_stress = bool(toggle_blocked_global_off_stress)
            self._last_qos_allow_on = bool(qos_allow_on)
            self._last_emergency_on_applied = bool(emergency_on_applied)
            if self.debug_logs and action_int > 0 and (not action_invalid):
                blocked_any = bool(
                    toggle_blocked
                    or toggle_blocked_high_util
                    or toggle_blocked_global
                    or off_budget_blocked
                    or total_budget_blocked
                )
                if (not blocked_any) and (not toggle_applied):
                    print(
                        "[toggle-debug] unexpected no-op "
                        f"action_int={action_int} idx={idx} edge={edge} key={key} "
                        f"prev_state={prev_state} next_state={next_state} "
                        f"blocked={toggle_blocked} blocked_high_util={toggle_blocked_high_util} "
                        f"blocked_global={toggle_blocked_global} blocked_global_off_stress={toggle_blocked_global_off_stress} "
                        f"blocked_off_budget={off_budget_blocked} blocked_total_budget={total_budget_blocked}"
                    )
            return (
                float(toggle_cost),
                bool(toggle_applied),
                bool(toggle_reverted),
                bool(toggle_blocked),
                bool(toggle_blocked_high_util),
                bool(toggle_blocked_global),
                bool(toggled_on),
                bool(toggled_off),
                bool(off_budget_blocked),
                bool(total_budget_blocked),
            )

        def _record_cd_block_debug(a_int: int) -> None:
            try:
                m = self.get_action_mask()
                a = int(a_int)
                mask_val = int(bool(m[a])) if 0 <= a < len(m) else -998
            except Exception:
                mask_val = -997
            self._dbg_cd_block_mask_val_last = int(mask_val)
            self._dbg_cd_block_mask_samples = int(getattr(self, "_dbg_cd_block_mask_samples", 0)) + 1
            if int(mask_val) == 1:
                self._dbg_cd_block_mask_mismatches = int(getattr(self, "_dbg_cd_block_mask_mismatches", 0)) + 1

        if self.simulator is None or not self.edge_list:
            return _finish()
        if action_int <= 0:
            return _finish()

        idx = int(action_int) - 1
        if idx < 0 or idx >= len(self.edge_list):
            return _finish()

        u, v = self.edge_list[idx]
        u = int(u)
        v = int(v)
        edge = (u, v)
        key = self._edge_key(u, v)
        prev_state = bool(self.simulator.active.get(key, True))
        next_state = not prev_state
        norm_drop_signal = float(getattr(self, "_last_norm_drop_step", getattr(self, "_last_norm_drop_total", 0.0)))
        qos_target = float(getattr(self.config, "qos_target_norm_drop", 0.0))
        qos_margin_off = float(getattr(self.config, "qos_guard_margin_off", getattr(self.config, "qos_guard_margin", 0.0)))
        qos_block_off = bool(norm_drop_signal > (qos_target - qos_margin_off))
        qos_viol_step = bool(getattr(self, "_last_qos_viol_step", False))
        # Canonical ON-recovery gate: use the exact QoS-violation flag that drives reward_qos.
        qos_allow_on = bool(qos_viol_step)
        is_turning_on = bool((not prev_state) and next_state)
        # Emergency ON is strictly tied to active QoS violation.
        emergency_on = bool(is_turning_on and qos_viol_step)

        # Total toggle budget hard gate: treat as NOOP if exhausted.
        total_cap = int(getattr(self.config, "max_total_toggles_per_episode", 0))
        total_used = int(getattr(self, "_total_toggles_used", 0))
        if (not emergency_on) and total_cap > 0 and total_used >= total_cap:
            total_budget_blocked = True
            self._dbg_block_budget = int(getattr(self, "_dbg_block_budget", 0)) + 1
            return _finish()
        if emergency_on:
            emerg_cap = int(getattr(self.config, "max_emergency_on_toggles_per_episode", 8))
            emerg_used = int(getattr(self, "_emergency_on_toggles_used", 0))
            if emerg_cap > 0 and emerg_used >= emerg_cap:
                total_budget_blocked = True
                toggle_blocked = True
                toggle_cost = float(getattr(self.config, "toggle_attempt_penalty", 0.0))
                self._dbg_block_budget = int(getattr(self, "_dbg_block_budget", 0)) + 1
                return _finish()

        # OFF budget hard gate: if edge is currently ON, this action would turn it OFF.
        if prev_state:
            cap = int(getattr(self.config, "max_off_toggles_per_episode", 0))
            used = int(getattr(self, "_off_toggles_used", 0))
            if cap > 0 and used >= cap:
                off_budget_blocked = True
                toggle_cost = self.config.toggle_penalty * self.config.revert_penalty_scale
                self._dbg_block_budget = int(getattr(self, "_dbg_block_budget", 0)) + 1
                return _finish()

        # QoS OFF-brake: block turning edges OFF when QoS is near/over the OFF threshold.
        if prev_state:
            if qos_block_off:
                toggle_blocked = True
                toggle_cost = (
                    self.config.toggle_penalty
                    * self.config.revert_penalty_scale
                    * float(getattr(self.config, "qos_guard_penalty_scale", 1.0))
                )
                self._dbg_block_qos_off = int(getattr(self, "_dbg_block_qos_off", 0)) + 1
                return _finish()

        # Global cooldown: block rapid toggles ONLY when turning an edge OFF.
        # Turning an edge ON is always allowed so the agent can recover quickly.
        if prev_state and getattr(self, "_global_toggle_cooldown_remaining", 0) > 0 and not emergency_on:
            toggle_blocked_global = True
            toggle_cost = self.config.toggle_penalty * self.config.revert_penalty_scale
            _record_cd_block_debug(action_int)
            self._dbg_block_global_cd = int(getattr(self, "_dbg_block_global_cd", 0)) + 1
            return _finish()

        # Global stress gate for OFF actions: allow ON->OFF only when the network is globally calm.
        # IMPORTANT: apply only if this toggle would turn an edge OFF.
        is_turning_off = bool(prev_state and (not next_state))
        if is_turning_off:
            episode_started_all_on = int(getattr(self, "_episode_initial_off_edges", 0)) == 0
            off_edges_now = int(
                sum(1 for e in self._edge_universe if not bool(self.simulator.active.get(e, True)))
            )
            all_on_now = bool(off_edges_now == 0)
            calm_now = bool(
                (not bool(getattr(self, "_last_qos_viol_step", False)))
                and (
                    float(getattr(self, "_last_max_util", 0.0))
                    < float(getattr(self.config, "util_block_threshold", 0.80))
                )
            )
            if episode_started_all_on and all_on_now and calm_now:
                toggle_blocked = True
                toggle_cost = float(getattr(self.config, "toggle_attempt_penalty", 0.0))
                return _finish()

            req_calm = int(getattr(self.config, "off_calm_steps_required", 0))
            if req_calm > 0 and int(getattr(self, "_calm_streak", 0)) < req_calm:
                toggle_blocked = True
                toggle_cost = float(getattr(self.config, "toggle_attempt_penalty", 0.0))
                return _finish()
            max_util = float(getattr(self, "_last_max_util", 0.0))
            thr = float(getattr(self.config, "max_util_off_allow_threshold", 0.80))
            if max_util >= thr:
                toggle_blocked_global_off_stress = True
                toggle_blocked = True
                toggle_cost = float(getattr(self.config, "toggle_attempt_penalty", 0.0))
                self._dbg_block_off_stress = int(getattr(self, "_dbg_block_off_stress", 0)) + 1
                return _finish()

        # Safety gate: block turning OFF a highly utilized edge.
        # (Turning ON an edge is always allowed.)
        if prev_state:
            util = float(self.simulator.utilization.get(key, 0.0))
            # Block turning OFF when utilization is ABOVE the threshold.
            if util > self.config.util_block_threshold:
                toggle_blocked_high_util = True
                toggle_cost = self.config.toggle_penalty * self.config.revert_penalty_scale
                self._dbg_block_util = int(getattr(self, "_dbg_block_util", 0)) + 1
                return _finish()
        else:
            # ON actions are always allowed. OFF safety gates protect QoS.
            pass

        # Cooldown: prevent rapid flapping of the same edge.
        g = self.simulator.graph
        if g.has_edge(u, v):
            graph_edge = (u, v)
        elif g.has_edge(v, u):
            graph_edge = (v, u)
        else:
            graph_edge = None
        last_toggled = (
            g.edges[graph_edge].get("last_toggled", -self.config.toggle_cooldown_steps)
            if graph_edge is not None
            else -self.config.toggle_cooldown_steps
        )
        if (self._step_count - int(last_toggled)) < int(self.config.toggle_cooldown_steps) and not emergency_on:
            toggle_blocked = True
            toggle_cost = self.config.toggle_penalty * self.config.revert_penalty_scale
            _record_cd_block_debug(action_int)
            self._dbg_block_edge_cd = int(getattr(self, "_dbg_block_edge_cd", 0)) + 1
            return _finish()

        # Apply toggle to simulator state.
        self.simulator.active[key] = next_state

        # Sync graph attributes where edge exists.
        prev_last_toggled = None
        if graph_edge is not None:
            prev_last_toggled = g.edges[graph_edge].get("last_toggled", -self.config.toggle_cooldown_steps)
            g.edges[graph_edge]["active"] = next_state
            g.edges[graph_edge]["last_toggled"] = int(self._step_count)

        # Safety: ensure graph stays connected via active edges.
        if not self._is_active_graph_connected():
            self.simulator.active[key] = prev_state
            if graph_edge is not None:
                g.edges[graph_edge]["active"] = prev_state
                if prev_last_toggled is not None:
                    g.edges[graph_edge]["last_toggled"] = int(prev_last_toggled)
            toggle_cost = self.config.toggle_penalty * self.config.revert_penalty_scale
            toggle_reverted = True
            self._global_toggle_cooldown_remaining = int(self.config.global_toggle_cooldown_steps)
            return _finish()

        # Successful toggle.
        toggle_applied = True
        toggled_off = bool(prev_state and (not next_state))
        toggled_on = bool((not prev_state) and next_state)

        toggle_cost = self.config.toggle_penalty
        # Turning OFF is riskier than turning ON -> apply extra cost on OFF only.
        if toggled_off:
            toggle_cost = float(toggle_cost) * float(getattr(self.config, "off_toggle_penalty_scale", 1.0))
            self._global_toggle_cooldown_remaining = int(self.config.global_toggle_cooldown_steps)
            self._off_toggles_used += 1

        if emergency_on and toggled_on:
            self._emergency_on_toggles_used += 1
            emergency_on_applied = True
        else:
            self._total_toggles_used += 1
        return _finish()

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
