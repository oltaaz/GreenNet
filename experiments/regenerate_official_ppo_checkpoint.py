#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from greennet.evaluation.official_ppo import (
    OFFICIAL_TOPOLOGY_ORDER,
    canonical_official_ppo_family_dir,
    install_official_ppo_from_run,
)


def _extract_run_dir(stdout: str) -> Path:
    marker = "Saved to "
    for line in reversed(stdout.splitlines()):
        if marker in line:
            return Path(line.split(marker, 1)[1].strip()).expanduser().resolve()
    raise SystemExit("Could not determine training output run directory from train.py output.")


def _config_with_topology(config_path: Path, topology_name: str) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Config file must contain a JSON object: {config_path}")
    env_block = payload.get("env")
    if not isinstance(env_block, dict):
        env_block = {}
        payload["env"] = env_block
    env_block["topology_name"] = topology_name
    env_block.pop("topology_path", None)
    tmp = tempfile.NamedTemporaryFile("w", suffix=f"_{topology_name}.json", delete=False, encoding="utf-8")
    with tmp:
        json.dump(payload, tmp, indent=2)
    return Path(tmp.name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh PPO checkpoint with the current GreenNet codebase and install it as the "
            "canonical official acceptance-matrix PPO artifact."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train_official_ppo.json"))
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--topology-name", choices=list(OFFICIAL_TOPOLOGY_ORDER), default=None)
    parser.add_argument("--all-topologies", action="store_true")
    args = parser.parse_args(argv)

    config_path = (REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if not config_path.exists():
        raise SystemExit(f"--config does not exist: {config_path}")

    if args.all_topologies and args.topology_name is not None:
        raise SystemExit("Use either --topology-name or --all-topologies, not both.")

    requested_topologies = list(OFFICIAL_TOPOLOGY_ORDER) if args.all_topologies or args.topology_name is None else [args.topology_name]
    output_dir = args.output_dir or canonical_official_ppo_family_dir()

    for topology_name in requested_topologies:
        effective_config = _config_with_topology(config_path, topology_name)
        try:
            cmd = [
                args.python,
                str(REPO_ROOT / "train.py"),
                "--config",
                str(effective_config),
                "--timesteps",
                str(int(args.timesteps)),
            ]
            print(f"[regenerate_official_ppo] training ({topology_name}):", " ".join(cmd))
            result = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            print(result.stdout, end="")
            if result.returncode != 0:
                if result.stderr:
                    print(result.stderr, file=sys.stderr, end="")
                raise SystemExit(result.returncode)

            run_dir = _extract_run_dir(result.stdout)
            target_model = install_official_ppo_from_run(
                run_dir,
                topology_name=topology_name,
                output_dir=output_dir,
                config_path=config_path,
                total_timesteps=int(args.timesteps),
                note="Canonical PPO artifact for the official acceptance-matrix rerun path.",
            )
            print(f"[regenerate_official_ppo] installed canonical PPO artifact at {target_model}")
        finally:
            effective_config.unlink(missing_ok=True)

    print(f"[regenerate_official_ppo] canonical family directory: {output_dir}")


if __name__ == "__main__":
    main()
