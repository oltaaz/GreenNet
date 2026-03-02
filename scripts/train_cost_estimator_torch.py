#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from greennet.impact_predictor import EdgeAttentionMLP

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None  # type: ignore[assignment]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_npz(
    path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    np.ndarray,
    Dict[str, np.ndarray],
]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path, allow_pickle=True)

    xg = np.asarray(data["Xg"], dtype=np.float32)
    xe = np.asarray(data["Xe"], dtype=np.float32)
    a = np.asarray(data["a"], dtype=np.int64)
    y_energy = np.asarray(data["y_delta_energy"], dtype=np.float32)
    y_ddrop = np.asarray(data["y_delta_norm_drop"], dtype=np.float32)
    y_qos = np.asarray(data["y_qos"], dtype=np.float32)

    is_off = np.asarray(data["is_off"], dtype=np.int64) if "is_off" in data else np.ones((xg.shape[0],), dtype=np.int64)
    extras: Dict[str, np.ndarray] = {}
    for key in (
        "scenario_id",
        "scenario_names",
        "topology_seed",
        "traffic_seed",
        "demand_scale",
        "capacity_scale",
        "flows_scale",
    ):
        if key in data:
            extras[key] = np.asarray(data[key])
    for key in ("qos_delay_ms", "qos_drop_max"):
        if key in data:
            extras[key] = np.asarray(data[key])

    global_names = [str(x) for x in np.asarray(data["global_feature_names"]).tolist()]
    edge_names = [str(x) for x in np.asarray(data["edge_feature_names"]).tolist()]

    if xg.ndim != 2:
        raise ValueError(f"Xg must be 2D, got shape={xg.shape}")
    if xe.ndim != 3:
        raise ValueError(f"Xe must be 3D, got shape={xe.shape}")
    if xg.shape[0] != xe.shape[0] or xg.shape[0] != a.shape[0]:
        raise ValueError("Mismatched sample counts in dataset")

    return xg, xe, a, y_energy, y_ddrop, y_qos, global_names, edge_names, is_off, extras


def _split_indices(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_n = max(1, int(n * val_frac))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    if train_idx.size == 0:
        train_idx = val_idx
    return train_idx, val_idx


def _split_indices_group(
    groups: np.ndarray, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups).reshape(-1)
    n = int(groups.shape[0])
    if n <= 1:
        return _split_indices(n, val_frac, seed)
    uniq = np.unique(groups)
    if uniq.size <= 1:
        return _split_indices(n, val_frac, seed)
    rng = np.random.default_rng(seed)
    uniq_shuffled = np.asarray(uniq).copy()
    rng.shuffle(uniq_shuffled)
    val_g = int(round(float(uniq_shuffled.size) * float(val_frac)))
    val_g = max(1, min(int(uniq_shuffled.size) - 1, val_g))
    val_groups = set(uniq_shuffled[:val_g].tolist())
    val_mask = np.asarray([g in val_groups for g in groups], dtype=bool)
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    if train_idx.size == 0 or val_idx.size == 0:
        return _split_indices(n, val_frac, seed)
    return train_idx, val_idx


def _make_loader(
    xg: np.ndarray,
    xe: np.ndarray,
    a: np.ndarray,
    y_energy: np.ndarray,
    y_ddrop: np.ndarray,
    y_qos: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(xg[indices]),
        torch.from_numpy(xe[indices]),
        torch.from_numpy(a[indices]),
        torch.from_numpy(y_energy[indices]),
        torch.from_numpy(y_ddrop[indices]),
        torch.from_numpy(y_qos[indices]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _eval_outputs(
    model: EdgeAttentionMLP,
    xg: np.ndarray,
    xe: np.ndarray,
    a: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    out_q: list[np.ndarray] = []
    out_d: list[np.ndarray] = []
    out_e: list[np.ndarray] = []
    with torch.no_grad():
        loader = _make_loader(
            xg,
            xe,
            a,
            np.zeros((xg.shape[0],), dtype=np.float32),
            np.zeros((xg.shape[0],), dtype=np.float32),
            np.zeros((xg.shape[0],), dtype=np.float32),
            indices,
            batch_size=512,
            shuffle=False,
        )
        for xb_g, xb_e, xb_a, _, _, _ in loader:
            xb_g = xb_g.to(device)
            xb_e = xb_e.to(device)
            xb_a = xb_a.to(device)
            qos_logit, ddrop, denergy = model(xb_g, xb_e, xb_a)
            out_q.append(qos_logit.detach().cpu().numpy())
            out_d.append(ddrop.detach().cpu().numpy())
            out_e.append(denergy.detach().cpu().numpy())
    if not out_q:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z
    return (
        np.concatenate(out_q, axis=0).astype(np.float32, copy=False),
        np.concatenate(out_d, axis=0).astype(np.float32, copy=False),
        np.concatenate(out_e, axis=0).astype(np.float32, copy=False),
    )


def _fit_temperature(logits: np.ndarray, y_true: np.ndarray, device: torch.device) -> float:
    if logits.size == 0:
        return 1.0

    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_true, dtype=torch.float32, device=device)

    log_t = nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
    opt = torch.optim.Adam([log_t], lr=0.05)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(250):
        opt.zero_grad(set_to_none=True)
        t = torch.exp(log_t).clamp(0.05, 20.0)
        loss = bce(logits_t / t, y_t)
        loss.backward()
        opt.step()

    t_final = float(torch.exp(log_t).detach().cpu().item())
    if not np.isfinite(t_final) or t_final <= 0.0:
        return 1.0
    return max(0.05, min(20.0, t_final))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    if y_true.size < 2:
        return None
    uniq = np.unique((y_true > 0.5).astype(np.int8))
    if uniq.size < 2:
        return None
    if roc_auc_score is None:
        return None
    try:
        return float(roc_auc_score((y_true > 0.5).astype(np.int32), y_score))
    except Exception:
        return None


def train_one(
    *,
    seed: int,
    xg: np.ndarray,
    xe: np.ndarray,
    a: np.ndarray,
    y_energy: np.ndarray,
    y_ddrop: np.ndarray,
    y_qos: np.ndarray,
    train_idx: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
) -> EdgeAttentionMLP:
    _set_seed(seed)
    model = EdgeAttentionMLP(edge_dim=xe.shape[2], global_dim=xg.shape[1], hidden_dim=hidden_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(np.sum(y_qos[train_idx] > 0.5))
    neg = float(max(0, int(train_idx.size) - int(pos)))
    pos_weight_value = float(neg / max(pos, 1.0)) if pos > 0.0 else 1.0
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    s1 = nn.SmoothL1Loss()

    rng = np.random.default_rng(seed)

    for _epoch in range(int(epochs)):
        boot_idx = rng.choice(train_idx, size=train_idx.size, replace=True)
        loader = _make_loader(
            xg, xe, a, y_energy, y_ddrop, y_qos, boot_idx, batch_size=batch_size, shuffle=True
        )
        model.train()
        for xb_g, xb_e, xb_a, yb_e, yb_d, yb_q in loader:
            xb_g = xb_g.to(device)
            xb_e = xb_e.to(device)
            xb_a = xb_a.to(device)
            yb_e = yb_e.to(device)
            yb_d = yb_d.to(device)
            yb_q = yb_q.to(device)

            q_logit, ddrop, denergy = model(xb_g, xb_e, xb_a)
            loss = bce(q_logit, yb_q) + s1(ddrop, yb_d) + s1(denergy, yb_e)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train graph-aware Impact Predictor ensemble.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/cost_estimator/ds_graph.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/impact_predictor"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--off-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="auto",
        choices=("auto", "group", "random"),
        help="Validation split mode. 'group' uses topology_seed groups if dataset provides them.",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu|mps|cuda")
    args = parser.parse_args()

    (
        xg,
        xe,
        a,
        y_energy,
        y_ddrop,
        y_qos,
        global_names,
        edge_names,
        is_off,
        extras,
    ) = _load_npz(args.dataset)
    n_raw = int(xg.shape[0])
    qos_pos_raw = int(np.sum(y_qos > 0.5))
    print(
        f"[train] qos class balance (raw): positives={qos_pos_raw} "
        f"negatives={max(0, n_raw - qos_pos_raw)} rate={float(np.mean(y_qos > 0.5)):.4f}"
    )

    if args.off_only:
        off_mask = is_off.astype(bool)
        if int(np.sum(off_mask)) >= 64:
            xg = xg[off_mask]
            xe = xe[off_mask]
            a = a[off_mask]
            y_energy = y_energy[off_mask]
            y_ddrop = y_ddrop[off_mask]
            y_qos = y_qos[off_mask]
            for key, value in list(extras.items()):
                if value.ndim >= 1 and value.shape[0] == off_mask.shape[0]:
                    extras[key] = value[off_mask]
            print(f"[train] using OFF-only samples: {xg.shape[0]}")
        else:
            print(f"[train] OFF-only requested but only {int(np.sum(off_mask))} samples; using all rows")
    n_used = int(xg.shape[0])
    qos_pos_used = int(np.sum(y_qos > 0.5))
    print(
        f"[train] qos class balance (used): positives={qos_pos_used} "
        f"negatives={max(0, n_used - qos_pos_used)} rate={float(np.mean(y_qos > 0.5)):.4f}"
    )

    n = int(xg.shape[0])
    if n < 32:
        raise RuntimeError(f"Dataset too small after filtering: {n} rows")

    device = _pick_device(args.device)
    topology_groups = extras.get("topology_seed")
    has_topology_groups = (
        topology_groups is not None
        and topology_groups.ndim >= 1
        and topology_groups.shape[0] == n
        and np.unique(topology_groups).size > 1
    )
    split_mode_used = "random"
    if str(args.split_mode) == "group" and not has_topology_groups:
        print("[train] split-mode=group requested but dataset has no usable topology_seed groups; using random split")
    if str(args.split_mode) in {"auto", "group"} and has_topology_groups:
        train_idx, val_idx = _split_indices_group(
            np.asarray(topology_groups).reshape(-1),
            float(args.val_frac),
            int(args.seed),
        )
        split_mode_used = "group_topology_seed"
    else:
        train_idx, val_idx = _split_indices(n, float(args.val_frac), int(args.seed))
        split_mode_used = "random"
    print(f"[train] split mode: {split_mode_used} train_rows={int(train_idx.size)} val_rows={int(val_idx.size)}")

    models: list[EdgeAttentionMLP] = []
    val_logits_by_model: list[np.ndarray] = []
    val_ddrop_by_model: list[np.ndarray] = []
    val_denergy_by_model: list[np.ndarray] = []

    for m in range(int(args.ensemble_size)):
        seed_m = int(args.seed) + m
        model = train_one(
            seed=seed_m,
            xg=xg,
            xe=xe,
            a=a,
            y_energy=y_energy,
            y_ddrop=y_ddrop,
            y_qos=y_qos,
            train_idx=train_idx,
            device=device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            hidden_dim=int(args.hidden_dim),
        )
        models.append(model)
        v_logit, v_ddrop, v_denergy = _eval_outputs(model, xg, xe, a, val_idx, device)
        val_logits_by_model.append(v_logit)
        val_ddrop_by_model.append(v_ddrop)
        val_denergy_by_model.append(v_denergy)
        print(f"[train] model_{m} done")

    val_logits_mean = np.mean(np.stack(val_logits_by_model, axis=0), axis=0).astype(np.float32)
    val_ddrop_pred = np.mean(np.stack(val_ddrop_by_model, axis=0), axis=0).astype(np.float32)
    val_denergy_pred = np.mean(np.stack(val_denergy_by_model, axis=0), axis=0).astype(np.float32)
    val_targets = y_qos[val_idx].astype(np.float32)
    val_ddrop_true = y_ddrop[val_idx].astype(np.float32)
    val_denergy_true = y_energy[val_idx].astype(np.float32)
    temperature = _fit_temperature(val_logits_mean, val_targets, device)
    probs_uncal = 1.0 / (1.0 + np.exp(-val_logits_mean))
    probs_cal = 1.0 / (1.0 + np.exp(-(val_logits_mean / max(1e-6, temperature))))
    brier_uncal = float(np.mean((probs_uncal - val_targets) ** 2)) if val_targets.size else 0.0
    brier_cal = float(np.mean((probs_cal - val_targets) ** 2)) if val_targets.size else 0.0
    overall_auc = _safe_auc(val_targets, probs_cal)
    overall_ddrop_mae = float(np.mean(np.abs(val_ddrop_pred - val_ddrop_true))) if val_ddrop_true.size else 0.0
    overall_denergy_mae = (
        float(np.mean(np.abs(val_denergy_pred - val_denergy_true))) if val_denergy_true.size else 0.0
    )

    scenario_val = np.zeros((val_idx.size,), dtype=np.int64)
    scenario_names_map: Dict[int, str] = {0: "all"}
    if "scenario_id" in extras and extras["scenario_id"].shape[0] == n:
        scenario_val = np.asarray(extras["scenario_id"], dtype=np.int64)[val_idx]
        scenario_names_map = {}
        if "scenario_names" in extras:
            names = [str(x) for x in np.asarray(extras["scenario_names"]).tolist()]
            scenario_names_map = {i: names[i] for i in range(len(names))}
        for sid in np.unique(scenario_val).tolist():
            if int(sid) not in scenario_names_map:
                scenario_names_map[int(sid)] = f"scenario_{int(sid)}"

    scenario_metrics: Dict[str, Dict[str, float | None]] = {}
    for sid in sorted(np.unique(scenario_val).tolist()):
        sid_int = int(sid)
        m = scenario_val == sid_int
        if int(np.sum(m)) <= 0:
            continue
        name = scenario_names_map.get(sid_int, f"scenario_{sid_int}")
        p = probs_cal[m]
        y = val_targets[m]
        d_pred = val_ddrop_pred[m]
        d_true = val_ddrop_true[m]
        e_pred = val_denergy_pred[m]
        e_true = val_denergy_true[m]
        sm = {
            "rows": int(np.sum(m)),
            "qos_positive_rate": float(np.mean(y > 0.5)),
            "qos_auc": _safe_auc(y, p),
            "qos_brier": float(np.mean((p - y) ** 2)),
            "ddrop_mae": float(np.mean(np.abs(d_pred - d_true))),
            "denergy_mae": float(np.mean(np.abs(e_pred - e_true))),
        }
        scenario_metrics[str(name)] = sm
        auc_text = f"{sm['qos_auc']:.4f}" if sm["qos_auc"] is not None else "NA"
        print(
            f"[val][{name}] rows={sm['rows']} auc={auc_text} "
            f"brier={sm['qos_brier']:.6f} ddrop_mae={sm['ddrop_mae']:.6f} denergy_mae={sm['denergy_mae']:.6f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, model in enumerate(models):
        out_path = args.out_dir / f"model_{i}.pt"
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "seed": int(args.seed) + i,
            },
            out_path,
        )
        print(f"[saved] {out_path}")

    ds_qos_delay_ms = (
        float(np.asarray(extras["qos_delay_ms"]).reshape(-1)[0]) if "qos_delay_ms" in extras else None
    )
    ds_qos_drop_max = (
        float(np.asarray(extras["qos_drop_max"]).reshape(-1)[0]) if "qos_drop_max" in extras else None
    )
    seeds = [int(args.seed) + i for i in range(int(args.ensemble_size))]
    split_report: Dict[str, object] = {
        "mode": split_mode_used,
        "val_frac": float(args.val_frac),
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
    }
    if has_topology_groups and topology_groups is not None and topology_groups.shape[0] == n:
        topo_all = np.asarray(topology_groups).reshape(-1)
        train_groups = sorted(np.unique(topo_all[train_idx]).astype(int).tolist())
        val_groups = sorted(np.unique(topo_all[val_idx]).astype(int).tolist())
        split_report["train_topology_seeds"] = train_groups
        split_report["val_topology_seeds"] = val_groups
        split_report["topology_seed_overlap"] = sorted(set(train_groups).intersection(val_groups))

    meta: Dict[str, object] = {
        "model_type": "impact_predictor_ensemble",
        "ensemble_size": int(args.ensemble_size),
        "dataset_path": str(args.dataset),
        "dataset_qos_delay_ms": ds_qos_delay_ms,
        "dataset_qos_drop_max": ds_qos_drop_max,
        "rows_raw": int(n_raw),
        "rows_used": int(n_used),
        "y_qos_positive_rate_used": float(np.mean(y_qos > 0.5)),
        "global_dim": int(xg.shape[1]),
        "edge_dim": int(xe.shape[2]),
        "num_edges": int(xe.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "global_feature_names": list(global_names),
        "edge_feature_names": list(edge_names),
        "temperature": float(temperature),
        "training_args": {
            "seed": int(args.seed),
            "seeds": seeds,
            "ensemble_size": int(args.ensemble_size),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "hidden_dim": int(args.hidden_dim),
            "val_frac": float(args.val_frac),
            "off_only": bool(args.off_only),
        },
        "calibration": {
            "temperature": float(temperature),
            "brier_uncal": float(brier_uncal),
            "brier_cal": float(brier_cal),
            "val_rows": int(val_targets.size),
        },
        "split": split_report,
        "validation_metrics": {
            "qos_auc": overall_auc,
            "qos_brier": float(brier_cal),
            "ddrop_mae": float(overall_ddrop_mae),
            "denergy_mae": float(overall_denergy_mae),
            "qos_positive_rate": float(np.mean(val_targets > 0.5)) if val_targets.size else 0.0,
        },
        "scenario_metrics": scenario_metrics,
        "risk_defaults": {
            "k": 1.0,
            "w_drop": 1.0,
            "w_energy": 0.2,
        },
        "threshold_defaults": {
            "tau": 0.11,
            "p_qos_max": 0.11,
            "ddrop_max": 0.001,
        },
    }

    meta_path = args.out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"[saved] {meta_path}")

    report = {
        "dataset_path": str(args.dataset),
        "split": split_report,
        "calibration": {
            "temperature": float(temperature),
            "brier_uncal": float(brier_uncal),
            "brier_cal": float(brier_cal),
            "val_rows": int(val_targets.size),
        },
        "overall_metrics": {
            "qos_auc": overall_auc,
            "qos_brier": float(brier_cal),
            "ddrop_mae": float(overall_ddrop_mae),
            "denergy_mae": float(overall_denergy_mae),
            "qos_positive_rate": float(np.mean(val_targets > 0.5)) if val_targets.size else 0.0,
        },
        "scenario_metrics": scenario_metrics,
    }
    report_path = args.out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"[saved] {report_path}")
    print(f"[val] rows={val_targets.size} temp={temperature:.4f} brier_uncal={brier_uncal:.6f} brier_cal={brier_cal:.6f}")


if __name__ == "__main__":
    main()
