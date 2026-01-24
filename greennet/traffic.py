"""Traffic generators and burst models.

This module intentionally keeps traffic generation *simple but extensible*.
We want:
- Reproducibility via seeds (important for baseline + RL training stability)
- A minimum-viable "realistic" model: diurnal intensity, hotspots, and bursty flows

The simulator/environment can consume an iterator of TrafficBurst events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence
import math
import random


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
    size: int
    start_time: float
    duration: int = 1


class TrafficGenerator:
    """Base traffic generator.

    Subclasses should override `generate`.
    """

    def generate(self, horizon: int) -> Iterator[TrafficBurst]:
        raise NotImplementedError


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


# Backward-compatible alias: existing code that imports TrafficGenerator and
# instantiates it directly can switch to ConstantTrafficGenerator.
# Prefer explicit classes in new code.
__all__ = [
    "TrafficBurst",
    "TrafficGenerator",
    "ConstantTrafficGenerator",
    "StochasticTrafficConfig",
    "StochasticTrafficGenerator",
]
