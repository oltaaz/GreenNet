"""Traffic generators and burst models.

This module intentionally keeps traffic generation *simple but extensible*.
We want:
- Reproducibility via seeds (important for baseline + RL training stability)
- A minimum-viable "realistic" model: diurnal intensity, hotspots, and bursty flows

The simulator/environment can consume an iterator of TrafficBurst events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence
import math
import random
import re

NAMED_TRAFFIC_DIR = Path(__file__).resolve().parent / "data" / "traffic"


class TrafficValidationError(ValueError):
    """Raised when a traffic replay input is malformed or unsupported."""


@dataclass(frozen=True)
class TrafficBurst:
    """A burst of traffic between two nodes.

    Notes:
      - `size` is an abstract amount of traffic (e.g., packets, bytes, or demand units)
      - `start_time` is a step-index expressed as float for compatibility
      - `duration` is in steps; default 1 means it only applies to that step
    """

    source: int
    destination: int
    size: float
    start_time: float
    duration: int = 1


class TrafficGenerator:
    """Base traffic generator.

    Subclasses should override `generate`.
    """

    def generate(self, horizon: int) -> Iterator[TrafficBurst]:
        raise NotImplementedError


@dataclass(frozen=True)
class ReplayTrafficConfig:
    """Configuration for replaying traffic bursts from a file-backed profile."""

    node_count: int
    bursts: tuple[TrafficBurst, ...]
    cycle_length: int | None = None


class ReplayTrafficGenerator(TrafficGenerator):
    """Replay a validated burst schedule, optionally repeating it on a fixed cycle."""

    def __init__(self, config: ReplayTrafficConfig):
        if config.node_count < 2:
            raise ValueError("node_count must be >= 2")
        if not config.bursts:
            raise ValueError("Replay traffic requires at least one burst.")
        if config.cycle_length is not None and int(config.cycle_length) <= 0:
            raise ValueError("cycle_length must be > 0 when provided.")

        self.config = ReplayTrafficConfig(
            node_count=int(config.node_count),
            bursts=tuple(sorted(config.bursts, key=lambda burst: (burst.start_time, burst.source, burst.destination))),
            cycle_length=(int(config.cycle_length) if config.cycle_length is not None else None),
        )

    def generate(self, horizon: int) -> Iterator[TrafficBurst]:
        if horizon <= 0:
            return

        cycle_length = self.config.cycle_length
        if cycle_length is None:
            for burst in self.config.bursts:
                if 0 <= int(burst.start_time) < int(horizon):
                    yield burst
            return

        offset = 0
        while offset < int(horizon):
            for burst in self.config.bursts:
                start_time = int(burst.start_time) + offset
                if start_time >= int(horizon):
                    continue
                yield TrafficBurst(
                    source=int(burst.source),
                    destination=int(burst.destination),
                    size=float(burst.size),
                    start_time=float(start_time),
                    duration=int(burst.duration),
                )
            offset += int(cycle_length)


@dataclass
class ConstantTrafficGenerator(TrafficGenerator):
    """Your original simple generator, kept for compatibility.

    Generates a single unit flow (0 -> 1) at every step.
    """

    rate: float = 1.0

    def generate(self, horizon: int) -> Iterator[TrafficBurst]:
        for step in range(horizon):
            yield TrafficBurst(source=0, destination=1, size=1, start_time=float(step))


@dataclass
class StochasticTrafficConfig:
    """Configuration for a minimum-viable realistic traffic model."""

    node_count: int

    # Average number of *new* bursts per step at midday intensity=1.0.
    avg_bursts_per_step: float = 3.0

    # Diurnal control: intensity multiplier over time.
    # If None, a smooth sinusoid is used.
    diurnal_profile: Sequence[float] | None = None

    # Hotspot pairs (src, dst, weight). Weights bias selection.
    # If empty, pairs are chosen uniformly at random.
    hotspots: Sequence[tuple[int, int, float]] = ()

    # Burst size mixture: mice vs elephant.
    p_elephant: float = 0.15
    mice_size_range: tuple[int, int] = (1, 5)
    elephant_size_range: tuple[int, int] = (10, 50)

    # Duration (steps) range for bursts.
    duration_range: tuple[int, int] = (1, 6)

    # Optional anomaly/spike injection.
    # Probability per step to trigger a temporary intensity spike.
    spike_prob: float = 0.01
    spike_multiplier_range: tuple[float, float] = (2.0, 6.0)
    spike_duration_range: tuple[int, int] = (3, 12)


_SCENARIO_ALIASES = {
    "normal diurnal": "diurnal",
    "normal/diurnal": "diurnal",
    "failure": "anomaly",
    "flash crowd": "flash_crowd",
    "flashcrowd": "flash_crowd",
    "spike": "flash_crowd",
    "multi peak": "multi_peak",
}


def _default_hotspots_for_node_count(node_count: int) -> tuple[tuple[int, int, float], ...]:
    if node_count <= 2:
        return ()
    if node_count == 6:
        return ((0, 3, 3.0), (1, 4, 2.0))
    if node_count == 8:
        return ((0, 5, 3.0), (2, 7, 2.0))
    if node_count >= 12:
        return ((0, 6, 3.0), (2, 9, 2.5), (4, 11, 2.0))

    center = max(1, node_count // 2)
    return (
        (0, center, 3.0),
        (max(1, node_count // 3), node_count - 1, 2.0),
    )


def _diurnal_profile_single_peak() -> tuple[float, ...]:
    return (0.30, 0.35, 0.45, 0.60, 0.80, 0.95, 1.00, 0.90, 0.75, 0.60, 0.45, 0.35)


def _diurnal_profile_multi_peak() -> tuple[float, ...]:
    return (0.28, 0.42, 0.70, 0.95, 0.72, 0.48, 0.40, 0.62, 0.92, 0.88, 0.55, 0.35)


def _resolve_scenario_overrides(
    base: StochasticTrafficConfig,
    preset: dict[str, object],
) -> dict[str, object]:
    resolved: dict[str, object] = {}
    for key, value in preset.items():
        resolved[key] = value(base) if callable(value) else value
    return resolved


_SCENARIO_PRESETS: dict[str, dict[int, dict[str, object]]] = {
    "normal": {
        2: {
            "avg_bursts_per_step": 3.0,
            "spike_prob": 0.01,
            "spike_multiplier_range": (1.5, 3.0),
            "spike_duration_range": (2, 6),
            "hotspots": (),
        }
    },
    "diurnal": {
        2: {
            "avg_bursts_per_step": 2.8,
            "diurnal_profile": _diurnal_profile_single_peak(),
            "spike_prob": 0.01,
            "spike_multiplier_range": (1.4, 2.6),
            "spike_duration_range": (2, 5),
            "hotspots": lambda cfg: _default_hotspots_for_node_count(int(cfg.node_count)),
        }
    },
    "burst": {
        2: {
            "avg_bursts_per_step": 4.0,
            "spike_prob": 0.05,
            "spike_multiplier_range": (2.0, 6.0),
            "spike_duration_range": (3, 12),
            "hotspots": (),
        }
    },
    "hotspot": {
        2: {
            "avg_bursts_per_step": 3.0,
            "spike_prob": 0.01,
            "hotspots": lambda cfg: _default_hotspots_for_node_count(int(cfg.node_count)),
        }
    },
    "anomaly": {
        2: {
            "avg_bursts_per_step": 3.0,
            "spike_prob": 0.15,
            "spike_multiplier_range": (3.0, 8.0),
            "spike_duration_range": (5, 20),
            "hotspots": (),
        }
    },
    "flash_crowd": {
        2: {
            "avg_bursts_per_step": 3.4,
            "diurnal_profile": (0.35, 0.40, 0.45, 0.55, 0.65, 0.75, 0.90, 1.0, 0.95, 0.70, 0.50, 0.40),
            "hotspots": lambda cfg: _default_hotspots_for_node_count(int(cfg.node_count)),
            "spike_prob": 0.10,
            "spike_multiplier_range": (3.0, 9.0),
            "spike_duration_range": (4, 14),
            "duration_range": (2, 8),
            "p_elephant": 0.20,
        }
    },
    "multi_peak": {
        2: {
            "avg_bursts_per_step": 3.2,
            "diurnal_profile": _diurnal_profile_multi_peak(),
            "spike_prob": 0.02,
            "spike_multiplier_range": (1.8, 4.0),
            "spike_duration_range": (2, 8),
            "hotspots": lambda cfg: _default_hotspots_for_node_count(int(cfg.node_count)),
        }
    },
}


def list_named_traffic_profiles() -> list[str]:
    """Return packaged traffic replay profiles available to the simulator."""
    if not NAMED_TRAFFIC_DIR.exists():
        return []
    return sorted(path.stem for path in NAMED_TRAFFIC_DIR.glob("*.json"))


def load_named_traffic_profile(name: str, *, node_count: int) -> ReplayTrafficConfig:
    """Load a packaged traffic replay profile by name."""
    normalized = str(name or "").strip()
    if not normalized:
        raise TrafficValidationError("Traffic profile name must be a non-empty string.")

    path = NAMED_TRAFFIC_DIR / f"{normalized}.json"
    if not path.exists():
        available = ", ".join(list_named_traffic_profiles()) or "<none>"
        raise TrafficValidationError(
            f"Unknown traffic profile '{normalized}'. Available named traffic profiles: {available}."
        )
    try:
        return load_traffic_profile_from_file(path, node_count=node_count)
    except TrafficValidationError as exc:
        raise TrafficValidationError(
            f"Named traffic profile '{normalized}' is not compatible with node_count={int(node_count)}: {exc}"
        ) from exc


def load_traffic_profile_from_file(path: str | Path, *, node_count: int) -> ReplayTrafficConfig:
    """Load and validate a replay traffic profile from JSON."""
    resolved = _resolve_input_path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrafficValidationError(f"Traffic file does not exist: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise TrafficValidationError(f"Traffic file is not valid JSON: {resolved}") from exc
    except OSError as exc:
        raise TrafficValidationError(f"Failed to read traffic file '{resolved}': {exc}") from exc
    return load_traffic_profile_from_dict(payload, node_count=node_count, source=str(resolved))


def load_traffic_profile_from_dict(
    payload: object,
    *,
    node_count: int,
    source: str = "<memory>",
) -> ReplayTrafficConfig:
    """Load and validate a replay traffic profile from a decoded JSON object."""
    if not isinstance(payload, dict):
        raise TrafficValidationError(f"Traffic payload from {source} must be a JSON object.")

    format_version = payload.get("format_version", 1)
    if format_version != 1:
        raise TrafficValidationError(
            f"Unsupported traffic format_version={format_version!r} in {source}; expected 1."
        )

    expected_node_count = int(node_count)
    file_node_count = payload.get("node_count")
    if file_node_count is not None:
        parsed_node_count = _coerce_int(file_node_count, field="node_count", source=source, minimum=2)
        if parsed_node_count != expected_node_count:
            raise TrafficValidationError(
                f"Traffic input '{source}' declares node_count={parsed_node_count}, "
                f"but the active topology uses node_count={expected_node_count}."
            )

    has_bursts = "bursts" in payload
    has_matrices = "matrices" in payload
    if has_bursts == has_matrices:
        raise TrafficValidationError(
            f"Traffic input '{source}' must define exactly one of 'bursts' or 'matrices'."
        )

    repeat = bool(payload.get("repeat", False))
    raw_cycle_length = payload.get("cycle_length")
    cycle_length = None
    if raw_cycle_length is not None:
        cycle_length = _coerce_int(raw_cycle_length, field="cycle_length", source=source, minimum=1)

    if has_bursts:
        bursts = _parse_bursts(payload.get("bursts"), node_count=expected_node_count, source=source)
        if not bursts:
            raise TrafficValidationError(f"Traffic input '{source}' must contain at least one burst.")
        if repeat and cycle_length is None:
            cycle_length = max(1, max(int(burst.start_time) for burst in bursts) + 1)
        return ReplayTrafficConfig(
            node_count=expected_node_count,
            bursts=tuple(bursts),
            cycle_length=cycle_length if repeat else None,
        )

    matrices = _parse_matrices(payload.get("matrices"), node_count=expected_node_count, source=source)
    bursts = _bursts_from_matrices(matrices)
    if not bursts:
        raise TrafficValidationError(
            f"Traffic input '{source}' matrices must contain at least one positive non-diagonal demand value."
        )
    if repeat and cycle_length is None:
        cycle_length = len(matrices)
    return ReplayTrafficConfig(
        node_count=expected_node_count,
        bursts=tuple(bursts),
        cycle_length=cycle_length if repeat else None,
    )


def _normalize_scenario_name(name: str) -> tuple[str, int | None]:
    raw = name.strip().lower().replace("_", " ").replace("-", " ").replace("/", " ")
    raw = re.sub(r"\s+", " ", raw)
    match = re.match(r"^(.*?)(?:\s+v(\d+))?$", raw)
    if match:
        base = match.group(1).strip()
        ver = int(match.group(2)) if match.group(2) is not None else None
    else:
        base = raw.strip()
        ver = None
    base = _SCENARIO_ALIASES.get(base, base)
    return base, ver


def _scale_int_range(rng: tuple[int, int], factor: float) -> tuple[int, int]:
    lo, hi = rng
    lo = max(1, int(round(float(lo) * factor)))
    hi = max(lo, int(round(float(hi) * factor)))
    return lo, hi


def apply_traffic_scenario(
    base: StochasticTrafficConfig,
    scenario: str | None,
    *,
    intensity: float | None = None,
    duration: float | None = None,
    frequency: float | None = None,
    version: int | None = None,
) -> StochasticTrafficConfig:
    if scenario is None or str(scenario).strip() == "":
        return base

    name, parsed_ver = _normalize_scenario_name(str(scenario))
    use_ver = int(version) if version is not None else (parsed_ver or 2)

    presets = _SCENARIO_PRESETS.get(name)
    if presets is None or use_ver not in presets:
        raise ValueError(f"Unknown traffic scenario '{scenario}' (version={use_ver})")

    cfg = replace(base, **_resolve_scenario_overrides(base, presets[use_ver]))

    if intensity is not None:
        cfg.avg_bursts_per_step = max(0.0, float(cfg.avg_bursts_per_step) * float(intensity))

    if frequency is not None:
        scaled = float(cfg.spike_prob) * float(frequency)
        cfg.spike_prob = min(1.0, max(0.0, scaled))

    if duration is not None:
        factor = max(0.1, float(duration))
        cfg.duration_range = _scale_int_range(cfg.duration_range, factor)
        cfg.spike_duration_range = _scale_int_range(cfg.spike_duration_range, factor)

    return cfg


class StochasticTrafficGenerator(TrafficGenerator):
    """A reproducible stochastic traffic generator.

    Model features:
      - Diurnal intensity (sinusoid or provided profile)
      - Bursty arrivals (Poisson-like via exponential inter-arrivals)
      - Hotspots bias (elephant flows often occur on a few popular pairs)
      - Mice/elephant size mixture
      - Optional spikes/anomalies

    This is intentionally lightweight: stdlib only.
    """

    def __init__(self, config: StochasticTrafficConfig, *, seed: int | None = None):
        self.config = config
        self.rng = random.Random(seed)

        if config.node_count < 2:
            raise ValueError("node_count must be >= 2")

        # Precompute hotspot cumulative distribution for fast sampling.
        self._hotspot_cdf: list[tuple[float, tuple[int, int]]] = []
        total_w = 0.0
        for (s, d, w) in config.hotspots:
            if s == d:
                continue
            if not (0 <= s < config.node_count and 0 <= d < config.node_count):
                continue
            if w <= 0:
                continue
            total_w += float(w)
            self._hotspot_cdf.append((total_w, (s, d)))
        self._hotspot_total = total_w

    def _diurnal_intensity(self, step: int, horizon: int) -> float:
        prof = self.config.diurnal_profile
        if prof is not None and len(prof) > 0:
            # Map step -> profile index.
            idx = int((step / max(1, horizon - 1)) * (len(prof) - 1))
            return max(0.0, float(prof[idx]))

        # Smooth sinusoid: low at night (~0.4), high at midday (~1.0)
        # Step normalized to [0, 2pi].
        x = (step / max(1, horizon)) * 2.0 * math.pi
        base = 0.7 + 0.3 * math.sin(x - math.pi / 2.0)  # peak around mid-horizon
        return max(0.05, base)

    def _sample_pair(self) -> tuple[int, int]:
        # Prefer hotspots some of the time when defined.
        if self._hotspot_total > 0.0 and self.rng.random() < 0.7:
            r = self.rng.random() * self._hotspot_total
            for c, pair in self._hotspot_cdf:
                if r <= c:
                    return pair

        # Uniform random pair (excluding self).
        n = self.config.node_count
        s = self.rng.randrange(n)
        d = self.rng.randrange(n - 1)
        if d >= s:
            d += 1
        return s, d

    def _sample_size(self) -> int:
        cfg = self.config
        if self.rng.random() < cfg.p_elephant:
            lo, hi = cfg.elephant_size_range
        else:
            lo, hi = cfg.mice_size_range
        lo = max(1, int(lo))
        hi = max(lo, int(hi))
        return self.rng.randint(lo, hi)

    def _sample_duration(self) -> int:
        lo, hi = self.config.duration_range
        lo = max(1, int(lo))
        hi = max(lo, int(hi))
        return self.rng.randint(lo, hi)

    def generate(self, horizon: int) -> Iterator[TrafficBurst]:
        cfg = self.config

        # Spike state
        spike_remaining = 0
        spike_mult = 1.0

        for step in range(horizon):
            intensity = self._diurnal_intensity(step, horizon)

            # Possibly trigger an anomaly/spike.
            if spike_remaining <= 0 and self.rng.random() < cfg.spike_prob:
                spike_remaining = self.rng.randint(
                    cfg.spike_duration_range[0], cfg.spike_duration_range[1]
                )
                spike_mult = self.rng.uniform(
                    cfg.spike_multiplier_range[0], cfg.spike_multiplier_range[1]
                )

            if spike_remaining > 0:
                intensity *= spike_mult
                spike_remaining -= 1

            # Poisson-like number of bursts per step.
            lam = max(0.0, cfg.avg_bursts_per_step * intensity)
            k = _poisson(self.rng, lam)

            for _ in range(k):
                s, d = self._sample_pair()
                size = self._sample_size()
                dur = self._sample_duration()
                yield TrafficBurst(
                    source=s,
                    destination=d,
                    size=size,
                    start_time=float(step),
                    duration=dur,
                )


def _poisson(rng: random.Random, lam: float) -> int:
    """Sample from a Poisson distribution using Knuth's method.

    Works well for small/moderate lam (which is our use case per-step).
    """

    if lam <= 0.0:
        return 0
    # Knuth
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def _resolve_input_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    return resolved


def _parse_bursts(raw_bursts: object, *, node_count: int, source: str) -> list[TrafficBurst]:
    if not isinstance(raw_bursts, list) or not raw_bursts:
        raise TrafficValidationError(f"Traffic input '{source}' must define a non-empty 'bursts' list.")

    bursts: list[TrafficBurst] = []
    for idx, raw_burst in enumerate(raw_bursts):
        if not isinstance(raw_burst, dict):
            raise TrafficValidationError(f"Traffic input '{source}' burst #{idx} must be an object.")

        src = _coerce_int(raw_burst.get("source"), field=f"bursts[{idx}].source", source=source, minimum=0)
        dst = _coerce_int(raw_burst.get("destination"), field=f"bursts[{idx}].destination", source=source, minimum=0)
        if src == dst:
            raise TrafficValidationError(f"Traffic input '{source}' burst #{idx} cannot use source==destination.")
        if src >= node_count or dst >= node_count:
            raise TrafficValidationError(
                f"Traffic input '{source}' burst #{idx} references node IDs outside 0..{node_count - 1}."
            )

        size = _coerce_float(
            raw_burst.get("size"),
            field=f"bursts[{idx}].size",
            source=source,
            minimum=0.0,
            strictly_positive=True,
        )
        duration = _coerce_int(raw_burst.get("duration", 1), field=f"bursts[{idx}].duration", source=source, minimum=1)
        start_time = _coerce_int(
            raw_burst.get("start_time", 0),
            field=f"bursts[{idx}].start_time",
            source=source,
            minimum=0,
        )
        bursts.append(
            TrafficBurst(
                source=src,
                destination=dst,
                size=float(size),
                start_time=float(start_time),
                duration=duration,
            )
        )
    return bursts


def _parse_matrices(raw_matrices: object, *, node_count: int, source: str) -> list[list[list[float]]]:
    if not isinstance(raw_matrices, list) or not raw_matrices:
        raise TrafficValidationError(f"Traffic input '{source}' must define a non-empty 'matrices' list.")

    matrices: list[list[list[float]]] = []
    for matrix_idx, raw_matrix in enumerate(raw_matrices):
        if not isinstance(raw_matrix, list) or len(raw_matrix) != node_count:
            raise TrafficValidationError(
                f"Traffic input '{source}' matrix #{matrix_idx} must have exactly {node_count} rows."
            )

        parsed_rows: list[list[float]] = []
        for row_idx, raw_row in enumerate(raw_matrix):
            if not isinstance(raw_row, list) or len(raw_row) != node_count:
                raise TrafficValidationError(
                    f"Traffic input '{source}' matrix #{matrix_idx} row #{row_idx} "
                    f"must have exactly {node_count} columns."
                )
            parsed_row: list[float] = []
            for col_idx, raw_value in enumerate(raw_row):
                value = _coerce_float(
                    raw_value,
                    field=f"matrices[{matrix_idx}][{row_idx}][{col_idx}]",
                    source=source,
                    minimum=0.0,
                    strictly_positive=False,
                )
                if row_idx == col_idx and value > 0.0:
                    raise TrafficValidationError(
                        f"Traffic input '{source}' matrix #{matrix_idx} must have a zero diagonal."
                    )
                parsed_row.append(value)
            parsed_rows.append(parsed_row)
        matrices.append(parsed_rows)
    return matrices


def _bursts_from_matrices(matrices: Sequence[Sequence[Sequence[float]]]) -> list[TrafficBurst]:
    bursts: list[TrafficBurst] = []
    for step_idx, matrix in enumerate(matrices):
        for src, row in enumerate(matrix):
            for dst, demand in enumerate(row):
                if src == dst or float(demand) <= 0.0:
                    continue
                bursts.append(
                    TrafficBurst(
                        source=int(src),
                        destination=int(dst),
                        size=float(demand),
                        start_time=float(step_idx),
                        duration=1,
                    )
                )
    return bursts


def _coerce_int(value: object, *, field: str, source: str, minimum: int) -> int:
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TrafficValidationError(f"Traffic input '{source}' field '{field}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TrafficValidationError(f"Traffic input '{source}' field '{field}' must be an integer.")
    if coerced < minimum:
        raise TrafficValidationError(
            f"Traffic input '{source}' field '{field}' must be at least {minimum}."
        )
    return coerced


def _coerce_float(
    value: object,
    *,
    field: str,
    source: str,
    minimum: float,
    strictly_positive: bool,
) -> float:
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TrafficValidationError(f"Traffic input '{source}' field '{field}' must be numeric.") from exc
    if coerced != coerced or coerced in (float("inf"), float("-inf")):
        raise TrafficValidationError(f"Traffic input '{source}' field '{field}' must be finite.")
    if strictly_positive and coerced <= minimum:
        raise TrafficValidationError(
            f"Traffic input '{source}' field '{field}' must be greater than {minimum}."
        )
    if (not strictly_positive) and coerced < minimum:
        raise TrafficValidationError(
            f"Traffic input '{source}' field '{field}' must be at least {minimum}."
        )
    return coerced


# Backward-compatible alias: existing code that imports TrafficGenerator and
# instantiates it directly can switch to ConstantTrafficGenerator.
# Prefer explicit classes in new code.
__all__ = [
    "TrafficBurst",
    "TrafficGenerator",
    "ReplayTrafficConfig",
    "ReplayTrafficGenerator",
    "ConstantTrafficGenerator",
    "StochasticTrafficConfig",
    "StochasticTrafficGenerator",
    "apply_traffic_scenario",
    "list_named_traffic_profiles",
    "load_named_traffic_profile",
    "load_traffic_profile_from_file",
    "load_traffic_profile_from_dict",
]
