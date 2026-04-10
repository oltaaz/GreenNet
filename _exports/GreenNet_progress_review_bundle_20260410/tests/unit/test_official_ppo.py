from __future__ import annotations

import json
from pathlib import Path

from greennet.evaluation import official_ppo


def test_normalize_official_topology_name_accepts_size_classes_and_aliases() -> None:
    assert official_ppo.normalize_official_topology_name("small") == "small"
    assert official_ppo.normalize_official_topology_name("regional_ring") == "small"
    assert official_ppo.normalize_official_topology_name("medium") == "medium"
    assert official_ppo.normalize_official_topology_name("metro_hub") == "medium"
    assert official_ppo.normalize_official_topology_name("large") == "large"
    assert official_ppo.normalize_official_topology_name("backbone_large") == "large"
    assert official_ppo.normalize_official_topology_name("unknown") is None


def test_install_official_ppo_from_run_writes_topology_metadata_and_family_manifest(tmp_path: Path) -> None:
    source_run_dir = tmp_path / "runs" / "demo"
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "ppo_greennet.zip").write_bytes(b"demo")
    (source_run_dir / "env_config.json").write_text('{"topology_name":"small"}', encoding="utf-8")
    (source_run_dir / "train_config.json").write_text('{"total_timesteps": 123}', encoding="utf-8")

    output_dir = tmp_path / "artifacts" / "models" / "official_acceptance_v1"
    model_path = official_ppo.install_official_ppo_from_run(
        source_run_dir,
        topology_name="small",
        output_dir=output_dir,
        total_timesteps=123,
        note="unit-test",
    )

    assert model_path == output_dir / "small" / "ppo_greennet.zip"
    assert model_path.exists()

    metadata = json.loads((output_dir / "small" / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assert metadata["topology_name"] == "small"
    assert metadata["source_run_dir"] == str(source_run_dir.resolve())
    assert metadata["total_timesteps"] == 123
    assert metadata["note"] == "unit-test"

    manifest = json.loads((output_dir / "checkpoint_family.json").read_text(encoding="utf-8"))
    assert manifest["topologies"]["small"]["exists"] is True
    assert manifest["topologies"]["medium"]["exists"] is False
    assert manifest["topologies"]["large"]["exists"] is False
