#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/lock_run.sh --scenario <normal|burst|hotspot> --run-id <RUN_ID> [options]

Required:
  --scenario <name>      Training scenario label for locked path.
  --run-id <RUN_ID>      Run folder id under runs/ (e.g. 20260220_111755).

Options:
  --runs-dir <path>      Source runs root. Default: runs
  --locked-root <path>   Locked artifacts root. Default: artifacts/locked
  --eval-source <path>   Optional file/dir to copy eval logs from.
  --eval-pattern <glob>  Pattern used when --eval-source is a directory.
                         Default: eval_*.txt
  --train-config <path>  Optional explicit training config to copy as train_config.json.
  -h, --help             Show this help.

Example:
  scripts/lock_run.sh --scenario normal --run-id 20260220_111755 \
    --eval-source artifacts/locked/normal/20260220_111755
USAGE
}

SCENARIO=""
RUN_ID=""
RUNS_DIR="runs"
LOCKED_ROOT="artifacts/locked"
EVAL_SOURCE=""
EVAL_PATTERN="eval_*.txt"
TRAIN_CONFIG_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario)
      SCENARIO="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --runs-dir)
      RUNS_DIR="${2:-}"
      shift 2
      ;;
    --locked-root)
      LOCKED_ROOT="${2:-}"
      shift 2
      ;;
    --eval-source)
      EVAL_SOURCE="${2:-}"
      shift 2
      ;;
    --eval-pattern)
      EVAL_PATTERN="${2:-}"
      shift 2
      ;;
    --train-config)
      TRAIN_CONFIG_OVERRIDE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SCENARIO" || -z "$RUN_ID" ]]; then
  echo "--scenario and --run-id are required." >&2
  usage
  exit 1
fi

case "$SCENARIO" in
  normal|burst|hotspot)
    ;;
  *)
    echo "Warning: scenario '$SCENARIO' is not one of normal|burst|hotspot." >&2
    ;;
esac

RUN_DIR="$RUNS_DIR/$RUN_ID"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

LOCK_DIR="$LOCKED_ROOT/$SCENARIO/$RUN_ID"
mkdir -p "$LOCK_DIR"

MODEL_SRC=""
if [[ -f "$RUN_DIR/ppo_greennet.zip" ]]; then
  MODEL_SRC="$RUN_DIR/ppo_greennet.zip"
elif [[ -f "$RUN_DIR/ppo_greennet" ]]; then
  MODEL_SRC="$RUN_DIR/ppo_greennet"
fi

if [[ -z "$MODEL_SRC" ]]; then
  echo "Missing model artifact in $RUN_DIR (expected ppo_greennet.zip)." >&2
  exit 1
fi

cp "$MODEL_SRC" "$LOCK_DIR/ppo_greennet.zip"

if [[ -f "$RUN_DIR/env_config.json" ]]; then
  cp "$RUN_DIR/env_config.json" "$LOCK_DIR/env_config.json"
else
  echo "Missing env_config.json in $RUN_DIR" >&2
  exit 1
fi

if [[ -n "$TRAIN_CONFIG_OVERRIDE" ]]; then
  if [[ ! -f "$TRAIN_CONFIG_OVERRIDE" ]]; then
    echo "train config override not found: $TRAIN_CONFIG_OVERRIDE" >&2
    exit 1
  fi
  cp "$TRAIN_CONFIG_OVERRIDE" "$LOCK_DIR/train_config.json"
elif [[ -f "$RUN_DIR/train_config.json" ]]; then
  cp "$RUN_DIR/train_config.json" "$LOCK_DIR/train_config.json"
elif [[ -f "$RUN_DIR/config.json" ]]; then
  cp "$RUN_DIR/config.json" "$LOCK_DIR/train_config.json"
else
  echo "Missing train_config.json/config.json in $RUN_DIR" >&2
  exit 1
fi

copied_eval=0
if [[ -n "$EVAL_SOURCE" ]]; then
  if [[ -f "$EVAL_SOURCE" ]]; then
    cp "$EVAL_SOURCE" "$LOCK_DIR/"
    copied_eval=$((copied_eval + 1))
  elif [[ -d "$EVAL_SOURCE" ]]; then
    shopt -s nullglob
    eval_files=("$EVAL_SOURCE"/$EVAL_PATTERN)
    for f in "${eval_files[@]}"; do
      cp "$f" "$LOCK_DIR/"
      copied_eval=$((copied_eval + 1))
    done
    shopt -u nullglob
  else
    echo "eval source not found: $EVAL_SOURCE" >&2
    exit 1
  fi
fi

echo "Locked run created: $LOCK_DIR"
echo "Copied: ppo_greennet.zip, env_config.json, train_config.json"
if [[ "$copied_eval" -gt 0 ]]; then
  echo "Copied eval logs: $copied_eval"
else
  echo "Copied eval logs: 0 (use --eval-source to include eval_*.txt)"
fi
