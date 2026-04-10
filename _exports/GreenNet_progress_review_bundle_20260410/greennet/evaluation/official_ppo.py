from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_TOPOLOGY_ALIASES = {
    "small": "small",
    "regional_ring": "small",
    "medium": "medium",
    "metro_hub": "medium",
    "large": "large",
    "backbone_large": "large",
}
OFFICIAL_TOPOLOGY_ORDER = ("small", "medium", "large")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_official_topology_name(topology_name: str | None) -> str | None:
    if topology_name is None:
        return None
    return OFFICIAL_TOPOLOGY_ALIASES.get(str(topology_name).strip(), None)


def canonical_official_ppo_family_dir() -> Path:
    return _repo_root() / "artifacts" / "models" / "official_acceptance_v1"


def canonical_official_ppo_dir(topology_name: str | None = None) -> Path:
    normalized = normalize_official_topology_name(topology_name)
    base = canonical_official_ppo_family_dir()
    if normalized is None:
        return base
    return base / normalized


def canonical_official_ppo_model_path(topology_name: str | None = None) -> Path:
    return canonical_official_ppo_dir(topology_name) / "ppo_greennet.zip"


def canonical_official_ppo_metadata_path(topology_name: str | None = None) -> Path:
    return canonical_official_ppo_dir(topology_name) / "checkpoint_metadata.json"


def official_ppo_exists(topology_name: str | None = None) -> bool:
    return canonical_official_ppo_model_path(topology_name).exists()


def missing_official_ppo_topologies() -> list[str]:
    return [name for name in OFFICIAL_TOPOLOGY_ORDER if not official_ppo_exists(name)]


def install_official_ppo_from_run(
    source_run_dir: Path,
    *,
    topology_name: str,
    output_dir: Path | None = None,
    config_path: Path | None = None,
    total_timesteps: int | None = None,
    note: str | None = None,
) -> Path:
    normalized = normalize_official_topology_name(topology_name)
    if normalized is None:
        raise ValueError(f"Unsupported official topology name: {topology_name!r}")

    source_run_dir = source_run_dir.expanduser().resolve()
    model_path = source_run_dir / "ppo_greennet.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected PPO artifact at {model_path}")

    family_dir = (output_dir or canonical_official_ppo_family_dir()).expanduser().resolve()
    target_dir = family_dir / normalized
    target_dir.mkdir(parents=True, exist_ok=True)
    target_model_path = target_dir / "ppo_greennet.zip"
    target_metadata_path = target_dir / "checkpoint_metadata.json"
    shutil.copy2(model_path, target_model_path)

    copied_files = []
    for name in ("env_config.json", "train_config.json", "config.json", "requirements.txt"):
        src = source_run_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)
            copied_files.append(name)

    metadata: dict[str, Any] = {
        "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "topology_name": normalized,
        "source_run_dir": str(source_run_dir),
        "model_path": str(target_model_path),
        "copied_files": copied_files,
        "config_path": None if config_path is None else str(config_path.expanduser().resolve()),
        "total_timesteps": total_timesteps,
        "note": note,
        "observation_compatibility": {
            "status": "current_env_only",
            "details": (
                "This checkpoint was regenerated against the current GreenNetEnv Dict observation space "
                f"for the official {normalized} topology class."
            ),
        },
    }
    target_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    family_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "family_dir": str(family_dir),
        "topologies": {
            name: {
                "model_path": str(family_dir / name / "ppo_greennet.zip"),
                "metadata_path": str(family_dir / name / "checkpoint_metadata.json"),
                "exists": (family_dir / name / "ppo_greennet.zip").exists(),
            }
            for name in OFFICIAL_TOPOLOGY_ORDER
        },
    }
    (family_dir / "checkpoint_family.json").write_text(json.dumps(family_manifest, indent=2), encoding="utf-8")
    return target_model_path
