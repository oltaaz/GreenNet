#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/eval_matrix.sh --model <path/to/ppo_greennet.zip> [options]

This runs a fixed eval matrix:
  - scenarios: normal,burst,hotspot
  - initial_off_edges: 0,2,4
  - held-out topology seeds: 10..19

Defaults:
  --episodes 10
  --scenarios normal,burst,hotspot
  --off-edges 0,2,4
  --topology-seeds 10,11,12,13,14,15,16,17,18,19

Outputs:
  eval_<scenario>_off<k>.txt logs under:
    artifacts/locked/<lock_scenario>/<run_id>/

Options:
  --model <path>           Required model zip path.
  --episodes <int>         Episodes per matrix cell (default: 10).
  --scenarios <csv>        Scenario list (default: normal,burst,hotspot).
  --off-edges <csv>        Off-edge list (default: 0,2,4).
  --topology-seeds <csv>   Held-out topology seeds (default: 10..19).
  --config <path>          Optional config JSON passed to train.py --config.
  --eval-traffic-seed <n>  Optional fixed traffic seed override.
  --eval-drop-lambda <f>   Eval drop lambda override (default: train.py default).
  --lock-scenario <name>   Locked scenario folder name; auto-inferred from env_config when omitted (required if inference fails).
  --locked-root <path>     Locked root (default: artifacts/locked).
  --out-dir <path>         Explicit output dir; overrides locked-root/lock-scenario/run_id.
  --python <bin>           Python executable (default: python3).
  --dry-run                Print commands without executing.
  -h, --help               Show this help.

Example:
  scripts/eval_matrix.sh \
    --model runs/20260220_111755/ppo_greennet.zip \
    --episodes 10
USAGE
}

MODEL_PATH=""
EPISODES="10"
SCENARIOS_CSV="normal,burst,hotspot"
OFF_EDGES_CSV="0,2,4"
TOPOLOGY_SEEDS_CSV="10,11,12,13,14,15,16,17,18,19"
CONFIG_PATH=""
EVAL_TRAFFIC_SEED=""
EVAL_DROP_LAMBDA=""
LOCK_SCENARIO=""
LOCKED_ROOT="artifacts/locked"
OUT_DIR=""
PYTHON_BIN="python3"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="${2:-}"
      shift 2
      ;;
    --episodes)
      EPISODES="${2:-}"
      shift 2
      ;;
    --scenarios)
      SCENARIOS_CSV="${2:-}"
      shift 2
      ;;
    --off-edges)
      OFF_EDGES_CSV="${2:-}"
      shift 2
      ;;
    --topology-seeds)
      TOPOLOGY_SEEDS_CSV="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --eval-traffic-seed)
      EVAL_TRAFFIC_SEED="${2:-}"
      shift 2
      ;;
    --eval-drop-lambda)
      EVAL_DROP_LAMBDA="${2:-}"
      shift 2
      ;;
    --lock-scenario)
      LOCK_SCENARIO="${2:-}"
      shift 2
      ;;
    --locked-root)
      LOCKED_ROOT="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
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

if [[ -z "$MODEL_PATH" ]]; then
  echo "--model is required." >&2
  usage
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi

MODEL_ABS="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
RUN_DIR="$(dirname "$MODEL_ABS")"
RUN_ID="$(basename "$RUN_DIR")"

# Default eval config: use the run's train_config.json when available.
if [[ -z "$CONFIG_PATH" ]]; then
  if [[ -f "$RUN_DIR/train_config.json" ]]; then
    CONFIG_PATH="$RUN_DIR/train_config.json"
  elif [[ -f "$RUN_DIR/config.json" ]]; then
    CONFIG_PATH="$RUN_DIR/config.json"
  fi
fi

if [[ -n "$CONFIG_PATH" && ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

# Infer lock scenario from env_config.json unless explicitly provided.
if [[ -z "$LOCK_SCENARIO" && -f "$RUN_DIR/env_config.json" ]]; then
  LOCK_SCENARIO="$($PYTHON_BIN - "$RUN_DIR/env_config.json" <<'PY'
import json
import pathlib
import sys
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
scenario = data.get("traffic_scenario")
print("" if scenario is None else str(scenario))
PY
)"
fi

if [[ -z "$LOCK_SCENARIO" ]]; then
  echo "Could not infer training scenario from $RUN_DIR/env_config.json." >&2
  echo "Pass --lock-scenario normal|burst|hotspot." >&2
  exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$LOCKED_ROOT/$LOCK_SCENARIO/$RUN_ID"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$OUT_DIR"

  # Ensure locked snapshot contains required artifacts.
  cp "$MODEL_ABS" "$OUT_DIR/ppo_greennet.zip"
  if [[ -f "$RUN_DIR/env_config.json" ]]; then
    cp "$RUN_DIR/env_config.json" "$OUT_DIR/env_config.json"
  fi
  if [[ -f "$RUN_DIR/train_config.json" ]]; then
    cp "$RUN_DIR/train_config.json" "$OUT_DIR/train_config.json"
  elif [[ -f "$RUN_DIR/config.json" ]]; then
    cp "$RUN_DIR/config.json" "$OUT_DIR/train_config.json"
  fi
fi

IFS=',' read -r -a SCENARIOS <<< "$SCENARIOS_CSV"
IFS=',' read -r -a OFF_EDGES <<< "$OFF_EDGES_CSV"

echo "Matrix output dir: $OUT_DIR"
echo "Model: $MODEL_ABS"
echo "Lock scenario: $LOCK_SCENARIO"

for scenario in "${SCENARIOS[@]}"; do
  scenario_trimmed="$(echo "$scenario" | xargs)"
  [[ -z "$scenario_trimmed" ]] && continue

  for off in "${OFF_EDGES[@]}"; do
    off_trimmed="$(echo "$off" | xargs)"
    [[ -z "$off_trimmed" ]] && continue

    log_file="$OUT_DIR/eval_${scenario_trimmed}_off${off_trimmed}.txt"

    cmd=(
      "$PYTHON_BIN" train.py
      --eval
      --model "$MODEL_ABS"
      --episodes "$EPISODES"
      --eval-traffic-scenario "$scenario_trimmed"
      --eval-initial-off-edges "$off_trimmed"
      --eval-topology-seeds "$TOPOLOGY_SEEDS_CSV"
    )

    if [[ -n "$CONFIG_PATH" ]]; then
      cmd+=(--config "$CONFIG_PATH")
    fi
    if [[ -n "$EVAL_TRAFFIC_SEED" ]]; then
      cmd+=(--eval-traffic-seed "$EVAL_TRAFFIC_SEED")
    fi
    if [[ -n "$EVAL_DROP_LAMBDA" ]]; then
      cmd+=(--eval-drop-lambda "$EVAL_DROP_LAMBDA")
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[dry-run] $log_file"
      printf '          '
      printf '%q ' "${cmd[@]}"
      echo
      continue
    fi

    {
      echo "# generated_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo "# model=$MODEL_ABS"
      echo "# run_id=$RUN_ID"
      echo "# lock_scenario=$LOCK_SCENARIO"
      echo "# eval_scenario=$scenario_trimmed"
      echo "# initial_off_edges=$off_trimmed"
      echo "# topology_seeds=$TOPOLOGY_SEEDS_CSV"
      echo "# episodes=$EPISODES"
      if [[ -n "$CONFIG_PATH" ]]; then
        echo "# config=$CONFIG_PATH"
      fi
      if [[ -n "$EVAL_TRAFFIC_SEED" ]]; then
        echo "# eval_traffic_seed=$EVAL_TRAFFIC_SEED"
      fi
      if [[ -n "$EVAL_DROP_LAMBDA" ]]; then
        echo "# eval_drop_lambda=$EVAL_DROP_LAMBDA"
      fi
      printf '# cmd='
      printf '%q ' "${cmd[@]}"
      echo
      echo
    } > "$log_file"

    echo "[eval] scenario=$scenario_trimmed off=$off_trimmed -> $log_file"
    "${cmd[@]}" | tee -a "$log_file"
  done
done

echo "Matrix evaluation complete. Logs written to: $OUT_DIR"
