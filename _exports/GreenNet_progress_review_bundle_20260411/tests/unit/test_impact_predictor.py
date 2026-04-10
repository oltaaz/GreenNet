from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from greennet.impact_predictor import EdgeAttentionMLP, ImpactPredictor


def _write_zeroed_model(
    path: Path,
    *,
    edge_dim: int,
    global_dim: int,
    hidden_dim: int,
    qos_logit: float,
    ddrop: float,
    denergy: float,
) -> None:
    model = EdgeAttentionMLP(edge_dim=edge_dim, global_dim=global_dim, hidden_dim=hidden_dim)
    state = model.state_dict()
    for tensor in state.values():
        tensor.zero_()
    state["qos_head.bias"].fill_(float(qos_logit))
    state["ddrop_head.bias"].fill_(float(ddrop))
    state["denergy_head.bias"].fill_(float(denergy))
    torch.save(state, path)


def test_predict_from_state_aggregates_ensemble_outputs_and_risk(tmp_path: Path) -> None:
    model_dir = tmp_path / "impact_predictor"
    model_dir.mkdir()
    meta = {
        "global_feature_names": ["g0", "g1", "g2"],
        "edge_feature_names": ["e0", "e1"],
        "global_dim": 3,
        "edge_dim": 2,
        "hidden_dim": 4,
        "temperature": 2.0,
        "risk_defaults": {"k": 1.5, "w_drop": 2.0, "w_energy": 0.5},
    }
    (model_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    _write_zeroed_model(model_dir / "model_0.pt", edge_dim=2, global_dim=3, hidden_dim=4, qos_logit=0.0, ddrop=0.1, denergy=-1.0)
    _write_zeroed_model(model_dir / "model_1.pt", edge_dim=2, global_dim=3, hidden_dim=4, qos_logit=2.0, ddrop=0.3, denergy=-0.2)

    predictor = ImpactPredictor(model_dir=model_dir, device="cpu")
    prediction = predictor.predict_from_state(
        Xg=np.array([9.0, 8.0, 7.0, 6.0], dtype=np.float32),
        Xe=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        action_edge_id=99,
    )

    probs = np.array([0.5, 1.0 / (1.0 + np.exp(-1.0))], dtype=np.float64)
    ddrop_vals = np.array([0.1, 0.3], dtype=np.float64)
    denergy_vals = np.array([-1.0, -0.2], dtype=np.float64)
    expected_risk = (
        (float(probs.mean()) + 1.5 * float(probs.std()))
        + 2.0 * max(0.0, float(ddrop_vals.mean()) + 1.5 * float(ddrop_vals.std()))
        - 0.5 * max(0.0, -(float(denergy_vals.mean()) - 1.5 * float(denergy_vals.std())))
    )

    assert prediction["p_qos_mean"] == pytest.approx(float(probs.mean()))
    assert prediction["p_qos_std"] == pytest.approx(float(probs.std()))
    assert prediction["ddrop_mean"] == pytest.approx(float(ddrop_vals.mean()))
    assert prediction["ddrop_std"] == pytest.approx(float(ddrop_vals.std()))
    assert prediction["denergy_mean"] == pytest.approx(float(denergy_vals.mean()))
    assert prediction["denergy_std"] == pytest.approx(float(denergy_vals.std()))
    assert prediction["risk_score"] == pytest.approx(expected_risk)


def test_predictor_requires_meta_and_weights(tmp_path: Path) -> None:
    empty_dir = tmp_path / "missing_model"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        ImpactPredictor(model_dir=empty_dir, device="cpu")
