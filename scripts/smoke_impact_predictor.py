#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np

from greennet.env import EnvConfig, GreenNetEnv


def _unpack_reset(ret):
    if isinstance(ret, tuple):
        if len(ret) >= 2:
            return ret[0], ret[1] if isinstance(ret[1], dict) else {}
        if len(ret) == 1:
            return ret[0], {}
    return ret, {}


def _unpack_step(ret):
    if len(ret) == 5:
        return ret[0], float(ret[1]), bool(ret[2]), bool(ret[3]), ret[4] if isinstance(ret[4], dict) else {}
    return ret[0], float(ret[1]), bool(ret[2]), False, ret[3] if isinstance(ret[3], dict) else {}


def _pick_action(env: GreenNetEnv, rng: np.random.Generator, p_off: float) -> int:
    if not bool(env._is_decision_step()):  # type: ignore[attr-defined]
        return 0
    if float(rng.random()) >= float(p_off):
        return 0
    off_actions = []
    for a in range(1, int(env.action_space.n)):
        try:
            if bool(env._is_off_toggle_action(a)):  # type: ignore[attr-defined]
                off_actions.append(a)
        except Exception:
            continue
    if not off_actions:
        return 0
    return int(off_actions[int(rng.integers(0, len(off_actions)))])


def main() -> None:
    model_dir = Path("models/impact_predictor")
    if not (model_dir / "meta.json").exists():
        print(f"[skip] missing model directory: {model_dir}")
        return

    cfg = EnvConfig(max_steps=200, cost_estimator_enabled=True, cost_estimator_model_dir=str(model_dir))
    env = GreenNetEnv(cfg)
    rng = np.random.default_rng(0)

    off_candidates = 0
    off_masked = 0
    for ep in range(2):
        obs, _ = _unpack_reset(env.reset(seed=ep))
        done = False
        while not done:
            action = _pick_action(env, rng, p_off=0.30)
            obs, _r, term, trunc, info = _unpack_step(env.step(action))
            off_candidates += int(info.get("is_off_candidate", 0))
            off_masked += int(info.get("forced_noop_by_cost_estimator", 0))
            done = bool(term or trunc)

    env.close()
    print(f"off_candidates={off_candidates} off_masked={off_masked}")
    assert off_candidates > 0, "off_candidates should be > 0"


if __name__ == "__main__":
    main()
