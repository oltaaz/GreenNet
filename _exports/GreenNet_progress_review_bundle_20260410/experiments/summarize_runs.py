#!/usr/bin/env python3
import csv
import json
import re
import sys
from pathlib import Path


RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")
CSV_COLUMNS = [
    "run_id",
    "has_model",
    "model_path",
    "has_env_config",
    "has_train_config",
    "timesteps",
    "seed",
    "topology_randomize",
    "topology_seed",
    "energy_weight",
    "drop_penalty_lambda",
    "qos_target_norm_drop",
    "qos_min_volume",
    "qos_violation_penalty_scale",
    "qos_guard_margin",
    "toggle_penalty",
    "blocked_action_penalty",
    "base_capacity",
    "flows_per_step",
    "traffic_model",
    "has_robustness",
    "files_count",
    "folder_size_mb",
]
ENV_FIELDS = [
    "topology_randomize",
    "topology_seed",
    "energy_weight",
    "drop_penalty_lambda",
    "qos_target_norm_drop",
    "qos_min_volume",
    "qos_violation_penalty_scale",
    "qos_guard_margin",
    "toggle_penalty",
    "blocked_action_penalty",
    "base_capacity",
    "flows_per_step",
    "traffic_model",
]


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Warning: failed to parse {path}", file=sys.stderr)
        return None


def normalize_cell(value):
    if value is None:
        return ""
    return value


def run_folder_stats(run_dir: Path):
    files_count = 0
    total_size = 0
    for path in run_dir.rglob("*"):
        try:
            if path.is_file():
                files_count += 1
                total_size += path.stat().st_size
        except OSError:
            continue
    return files_count, total_size


def find_repo_root(start_path: Path):
    path = start_path
    if path.is_file():
        path = path.parent
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists():
            return candidate
        if (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "setup.cfg").exists():
            return candidate
    return path


def main():
    repo_root = find_repo_root(Path(__file__).resolve())
    runs_dir = repo_root / "runs"
    output_path = repo_root / "experiments" / "training_history.csv"

    run_dirs = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and RUN_ID_RE.match(path.name)
    ] if runs_dir.exists() else []
    run_dirs.sort(key=lambda p: p.name)

    rows = []
    size_index = []
    runs_with_model = 0
    runs_missing_env = 0
    runs_with_robustness = 0

    for run_dir in run_dirs:
        run_id = run_dir.name
        env_config = load_json(run_dir / "env_config.json")
        train_config = load_json(run_dir / "train_config.json")
        has_env_config = env_config is not None
        has_train_config = train_config is not None

        model_path = ""
        has_model = False
        for candidate in ("ppo_greennet.zip", "ppo_greennet"):
            path = run_dir / candidate
            if path.exists():
                has_model = True
                try:
                    model_path = str(path.relative_to(repo_root))
                except ValueError:
                    model_path = str(path)
                break

        robustness_path = run_dir / "robustness_eval.csv"
        has_robustness = robustness_path.exists()

        files_count, size_bytes = run_folder_stats(run_dir)
        folder_size_mb = round(size_bytes / (1024 * 1024), 2)

        if has_model:
            runs_with_model += 1
        if not has_env_config:
            runs_missing_env += 1
        if has_robustness:
            runs_with_robustness += 1

        row = {
            "run_id": run_id,
            "has_model": has_model,
            "model_path": model_path,
            "has_env_config": has_env_config,
            "has_train_config": has_train_config,
            "timesteps": normalize_cell(
                train_config.get("total_timesteps") if isinstance(train_config, dict) else None
            ),
            "seed": normalize_cell(
                train_config.get("seed") if isinstance(train_config, dict) else None
            ),
            "has_robustness": has_robustness,
            "files_count": files_count,
            "folder_size_mb": folder_size_mb,
        }

        for field in ENV_FIELDS:
            value = env_config.get(field) if isinstance(env_config, dict) else None
            row[field] = normalize_cell(value)

        row["has_model"] = normalize_cell(row["has_model"])
        row["has_env_config"] = normalize_cell(row["has_env_config"])
        row["has_train_config"] = normalize_cell(row["has_train_config"])
        row["has_robustness"] = normalize_cell(row["has_robustness"])

        rows.append(row)
        size_index.append((run_id, folder_size_mb))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_cell(row.get(key)) for key in CSV_COLUMNS})

    size_index.sort(key=lambda item: item[1], reverse=True)
    top_largest = size_index[:10]

    print(f"Total runs: {len(rows)}")
    print(f"Runs with model: {runs_with_model}")
    print(f"Runs missing env_config: {runs_missing_env}")
    print(f"Runs with robustness outputs: {runs_with_robustness}")
    print("Top 10 largest run folders:")
    for run_id, size_mb in top_largest:
        print(f"  {run_id}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
