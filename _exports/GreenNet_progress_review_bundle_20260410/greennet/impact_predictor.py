"""Graph-aware lightweight Impact Predictor with uncertainty and calibration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn


class EdgeAttentionMLP(nn.Module):
    """Small edge-encoder + pooled network embedding + action-edge conditioning."""

    def __init__(self, edge_dim: int, global_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.global_dim = int(global_dim)
        self.hidden_dim = int(hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.edge_attn = nn.Linear(self.hidden_dim, 1)

        trunk_in = self.hidden_dim * 4 + self.global_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.qos_head = nn.Linear(32, 1)
        self.ddrop_head = nn.Linear(32, 1)
        self.denergy_head = nn.Linear(32, 1)

    def forward(
        self, xg: torch.Tensor, xe: torch.Tensor, action_edge_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # xg: [B, G], xe: [B, E, F], action_edge_id: [B]
        edge_emb = self.edge_mlp(xe)

        mean_pool = edge_emb.mean(dim=1)
        max_pool = edge_emb.max(dim=1).values

        attn_logits = self.edge_attn(edge_emb).squeeze(-1)
        attn_w = torch.softmax(attn_logits, dim=1)
        attn_pool = (edge_emb * attn_w.unsqueeze(-1)).sum(dim=1)

        bsz = edge_emb.shape[0]
        edge_count = max(1, int(edge_emb.shape[1]))
        idx = torch.clamp(action_edge_id.long(), min=0, max=edge_count - 1)
        batch_idx = torch.arange(bsz, device=edge_emb.device)
        action_emb = edge_emb[batch_idx, idx]

        valid_action = (action_edge_id >= 0).float().unsqueeze(-1)
        action_emb = action_emb * valid_action

        h = torch.cat([mean_pool, max_pool, attn_pool, action_emb, xg], dim=-1)
        z = self.trunk(h)

        qos_logit = self.qos_head(z).squeeze(-1)
        ddrop = self.ddrop_head(z).squeeze(-1)
        denergy = self.denergy_head(z).squeeze(-1)
        return qos_logit, ddrop, denergy


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ImpactPredictor:
    """Runtime ensemble predictor for OFF-action impact estimates."""

    def __init__(self, model_dir: str | Path = "models/impact_predictor", device: str | None = None) -> None:
        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta file: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.global_feature_names = [str(x) for x in self.meta["global_feature_names"]]
        self.edge_feature_names = [str(x) for x in self.meta["edge_feature_names"]]
        self.global_dim = int(self.meta["global_dim"])
        self.edge_dim = int(self.meta["edge_dim"])
        self.hidden_dim = int(self.meta.get("hidden_dim", 32))
        self.temperature = max(1e-3, _safe_float(self.meta.get("temperature", 1.0), 1.0))

        risk_defaults = self.meta.get("risk_defaults", {})
        self.default_k = _safe_float(risk_defaults.get("k", 1.0), 1.0)
        self.default_w_drop = _safe_float(risk_defaults.get("w_drop", 1.0), 1.0)
        self.default_w_energy = _safe_float(risk_defaults.get("w_energy", 0.2), 0.2)

        self.device = _pick_device(device)
        self.models: List[EdgeAttentionMLP] = []

        model_paths = sorted(self.model_dir.glob("model_*.pt"))
        if not model_paths:
            raise FileNotFoundError(f"No ensemble weights found in {self.model_dir}")

        for path in model_paths:
            model = EdgeAttentionMLP(
                edge_dim=self.edge_dim,
                global_dim=self.global_dim,
                hidden_dim=self.hidden_dim,
            )
            payload = torch.load(path, map_location=self.device)
            if isinstance(payload, dict) and "state_dict" in payload:
                state_dict = payload["state_dict"]
            else:
                state_dict = payload
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)

    def _align_xg(self, xg: np.ndarray) -> np.ndarray:
        out = np.zeros((self.global_dim,), dtype=np.float32)
        n = min(self.global_dim, int(xg.size))
        if n > 0:
            out[:n] = xg[:n]
        return out

    def _align_xe(self, xe: np.ndarray) -> np.ndarray:
        if xe.ndim != 2:
            xe = np.asarray(xe, dtype=np.float32).reshape(-1, self.edge_dim)
        out = np.zeros((int(xe.shape[0]), self.edge_dim), dtype=np.float32)
        n = min(self.edge_dim, int(xe.shape[1]))
        if n > 0:
            out[:, :n] = xe[:, :n]
        return out

    def predict_from_state(
        self,
        Xg: Sequence[float] | np.ndarray,
        Xe: Sequence[Sequence[float]] | np.ndarray,
        action_edge_id: int,
        *,
        k: float | None = None,
        w_drop: float | None = None,
        w_energy: float | None = None,
    ) -> Dict[str, float]:
        xg_np = self._align_xg(np.asarray(Xg, dtype=np.float32).reshape(-1))
        xe_np = self._align_xe(np.asarray(Xe, dtype=np.float32))

        xg_t = torch.from_numpy(xg_np).to(self.device).unsqueeze(0)
        xe_t = torch.from_numpy(xe_np).to(self.device).unsqueeze(0)
        a_t = torch.tensor([int(action_edge_id)], dtype=torch.long, device=self.device)

        logits: List[float] = []
        ddrop_vals: List[float] = []
        denergy_vals: List[float] = []

        with torch.no_grad():
            for model in self.models:
                q_logit, ddrop, denergy = model(xg_t, xe_t, a_t)
                logits.append(_safe_float(q_logit.item()))
                ddrop_vals.append(_safe_float(ddrop.item()))
                denergy_vals.append(_safe_float(denergy.item()))

        logits_np = np.asarray(logits, dtype=np.float64)
        ddrop_np = np.asarray(ddrop_vals, dtype=np.float64)
        denergy_np = np.asarray(denergy_vals, dtype=np.float64)

        probs = 1.0 / (1.0 + np.exp(-(logits_np / float(self.temperature))))
        p_qos_mean = _safe_float(np.mean(probs))
        p_qos_std = _safe_float(np.std(probs))
        ddrop_mean = _safe_float(np.mean(ddrop_np))
        ddrop_std = _safe_float(np.std(ddrop_np))
        denergy_mean = _safe_float(np.mean(denergy_np))
        denergy_std = _safe_float(np.std(denergy_np))

        k_v = self.default_k if k is None else _safe_float(k, self.default_k)
        w_drop_v = self.default_w_drop if w_drop is None else _safe_float(w_drop, self.default_w_drop)
        w_energy_v = self.default_w_energy if w_energy is None else _safe_float(w_energy, self.default_w_energy)

        risk_score = (
            (p_qos_mean + k_v * p_qos_std)
            + w_drop_v * max(0.0, ddrop_mean + k_v * ddrop_std)
            - w_energy_v * max(0.0, -(denergy_mean - k_v * denergy_std))
        )

        return {
            "p_qos_mean": p_qos_mean,
            "p_qos_std": p_qos_std,
            "ddrop_mean": ddrop_mean,
            "ddrop_std": ddrop_std,
            "denergy_mean": denergy_mean,
            "denergy_std": denergy_std,
            "risk_score": _safe_float(risk_score),
        }
