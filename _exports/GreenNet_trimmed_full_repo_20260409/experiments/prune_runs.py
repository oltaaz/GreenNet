#!/usr/bin/env python3
import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path


RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")
MODEL_CANDIDATES = ("ppo_greennet.zip", "ppo_greennet")
ROBUSTNESS_FILES = ("robustness_eval.csv", "robustness_energy_vs_drop.png")
EXTRA_FILES = ("requirements.txt", "config.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Plan and prune Greennet run folders.")
    parser.add_argument("--runs_dir", default="runs", help="Directory containing run folders.")
    parser.add_argument("--keep", action="append", default=[], help="Run ID to keep (repeatable).")
    parser.add_argument("--keep-file", dest="keep_file", help="Path to a file with run IDs to keep.")
    parser.add_argument("--keep-latest", type=int, default=0, help="Keep latest N runs with models.")
    parser.add_argument(
        "--keep-robustness",
        action="store_true",
        help="Keep runs containing robustness artifacts.",
    )
    parser.add_argument(
        "--strip-only",
        action="store_true",
        help="Strip model artifacts from non-kept runs.",
    )
    parser.add_argument(
        "--strip-extras",
        action="store_true",
        help="Also remove requirements.txt and config.json from STRIP runs.",
    )
    parser.add_argument(
        "--delete-empty",
        action="store_true",
        help="Delete tiny runs missing models and env_config.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply the planned changes.")
    return parser.parse_args()


def load_keep_file(path: Path):
    keep_ids = []
    if not path:
        return keep_ids
    if not path.exists():
        print(f"Warning: keep file not found: {path}")
        return keep_ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            keep_ids.append(value)
    return keep_ids


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


def path_size_bytes(path: Path):
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    if path.is_dir():
        total = 0
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
        return total
    return 0


def collect_model_files(run_dir: Path):
    files = set()
    for name in MODEL_CANDIDATES:
        path = run_dir / name
        if path.exists():
            files.add(path)
    for path in run_dir.glob("*.zip"):
        files.add(path)
    return sorted(files)


def collect_extra_files(run_dir: Path):
    files = []
    for name in EXTRA_FILES:
        path = run_dir / name
        if path.exists():
            files.append(path)
    return files


def classify_runs(runs, keep_set, delete_empty):
    keep_list = []
    strip_list = []
    delete_list = []

    for run in runs:
        run_id = run["run_id"]
        if run_id in keep_set:
            keep_list.append(run)
        elif run["has_model"]:
            strip_list.append(run)
        elif delete_empty and (not run["has_model"]) and (not run["has_env_config"]) and run["size_bytes"] < 1024 * 1024:
            delete_list.append(run)
        else:
            keep_list.append(run)

    return keep_list, strip_list, delete_list


def print_plan(title, runs):
    print(title)
    if not runs:
        print("  (none)")
        return
    for run in runs:
        print(f"  {run['run_id']}")


def remove_path(path: Path):
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True, ""
    except FileNotFoundError:
        return False, "missing"
    except OSError as exc:
        return False, str(exc)


def main():
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = repo_root / runs_dir

    if not runs_dir.exists():
        print(f"No runs directory found at {runs_dir}")
        return

    run_dirs = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and RUN_ID_RE.match(path.name)
    ]
    run_dirs.sort(key=lambda p: p.name)

    runs = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        has_env_config = (run_dir / "env_config.json").exists()
        has_model = any((run_dir / name).exists() for name in MODEL_CANDIDATES)
        has_robustness = any((run_dir / name).exists() for name in ROBUSTNESS_FILES)
        files_count, size_bytes = run_folder_stats(run_dir)
        model_files = collect_model_files(run_dir)
        model_bytes = sum(path_size_bytes(path) for path in model_files)
        extra_files = collect_extra_files(run_dir)
        extra_bytes = sum(path_size_bytes(path) for path in extra_files)

        runs.append(
            {
                "run_id": run_id,
                "path": run_dir,
                "has_env_config": has_env_config,
                "has_model": has_model,
                "has_robustness": has_robustness,
                "files_count": files_count,
                "size_bytes": size_bytes,
                "model_files": model_files,
                "model_bytes": model_bytes,
                "extra_files": extra_files,
                "extra_bytes": extra_bytes,
            }
        )

    keep_set = set(args.keep or [])
    keep_set.update(load_keep_file(Path(args.keep_file)) if args.keep_file else [])

    if args.keep_robustness:
        keep_set.update(run["run_id"] for run in runs if run["has_robustness"])

    if args.keep_latest and args.keep_latest > 0:
        model_runs = [run["run_id"] for run in runs if run["has_model"]]
        model_runs.sort()
        keep_set.update(model_runs[-args.keep_latest :])

    run_ids = {run["run_id"] for run in runs}
    missing_keeps = sorted(keep_set - run_ids)
    if missing_keeps:
        print(f"Warning: keep IDs not found: {', '.join(missing_keeps)}")

    keep_list, strip_list, delete_list = classify_runs(runs, keep_set, args.delete_empty)

    strip_bytes = sum(
        run["model_bytes"] + (run["extra_bytes"] if args.strip_extras else 0)
        for run in strip_list
    )
    reclaim_bytes = strip_bytes + sum(run["size_bytes"] for run in delete_list)
    reclaim_mb = reclaim_bytes / (1024 * 1024)

    print_plan("KEEP:", keep_list)
    print_plan("STRIP:", strip_list)
    print_plan("DELETE:", delete_list)
    print(f"Estimated space to reclaim: {reclaim_mb:.2f} MB")
    if strip_list and not args.strip_only:
        print("Note: STRIP actions require --strip-only to execute.")

    log_dir = repo_root / "experiments"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"prune_log_{timestamp}.txt"

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"timestamp={timestamp}\n")
        log.write(f"runs_dir={runs_dir}\n")
        log.write(f"apply={args.apply}\n")
        log.write(f"strip_only={args.strip_only}\n")
        log.write(f"strip_extras={args.strip_extras}\n")
        log.write(f"delete_empty={args.delete_empty}\n")
        log.write(f"keep_ids={sorted(keep_set)}\n")
        log.write(f"estimated_reclaim_mb={reclaim_mb:.2f}\n")
        log.write(f"KEEP={','.join(run['run_id'] for run in keep_list)}\n")
        log.write(f"STRIP={','.join(run['run_id'] for run in strip_list)}\n")
        log.write(f"DELETE={','.join(run['run_id'] for run in delete_list)}\n")

        if not args.apply:
            log.write("DRY_RUN=1\n")
            print(f"Wrote log to {log_path}")
            return

        if strip_list and not args.strip_only:
            log.write("STRIP_SKIPPED=missing_strip_only\n")

        for run in strip_list:
            if not args.strip_only:
                continue
            for path in run["model_files"]:
                ok, err = remove_path(path)
                log.write(f"STRIP {run['run_id']} {path} ok={ok} err={err}\n")
            if args.strip_extras:
                for path in run["extra_files"]:
                    ok, err = remove_path(path)
                    log.write(f"STRIP_EXTRA {run['run_id']} {path} ok={ok} err={err}\n")

        for run in delete_list:
            ok, err = remove_path(run["path"])
            log.write(f"DELETE {run['run_id']} {run['path']} ok={ok} err={err}\n")

    print(f"Wrote log to {log_path}")


if __name__ == "__main__":
    main()
