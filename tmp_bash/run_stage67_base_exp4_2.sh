#!/usr/bin/env bash
set -Eeuo pipefail

# Stage 6-7 formal rerun launcher: B1 -> V1 -> V2 -> A0 -> A1 -> forced Eval.
# The five controller runs overwrite their existing official paths atomically.
# Interrupted runs retain *.partial.jsonl evidence; rerunning starts that run at row 0.

BRICKNET_ROOT=/home/jiahao/task/BrickNet
BRICKNET_PY=/home/jiahao/miniconda3/envs/bricknet/bin/python
BRICKNET_DATASET="$BRICKNET_ROOT/outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl"
STAGE5_REPORT="$BRICKNET_ROOT/outputs_preprocess/BrickNet-MM-Act-SFT/stage5/Stage5-full-replay-report.json"
CONTRACT_CONFIG="$BRICKNET_ROOT/configs/agentic_stage67_exp4_2.json"
OUTPUT_ROOT="$BRICKNET_ROOT/outputs_val/qwen35_08b"
MODEL=Qwen/Qwen3.5-0.8B
MODEL_REVISION=2fc06364715b967f1860aea9cf38778875588b17

export BRICKNET_DATA="$BRICKNET_ROOT/data/bricknet_datasets"
export PYTHONPATH="$BRICKNET_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  printf 'Usage: %s [--preflight-only]\n' "$0"
  printf '  no argument       Wait for an idle GPU 0, run B1, V1, V2, A0, A1, then rebuild all evaluations.\n'
  printf '  --preflight-only  Validate the frozen config and local provenance without loading the GPU model.\n'
}

PREFLIGHT_ONLY=0
case "${1:-}" in
  "") ;;
  --preflight-only) PREFLIGHT_ONLY=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

trap 'status=$?; printf "Stage 6-7 launcher failed at line %s (exit %s).\n" "$LINENO" "$status" >&2; exit "$status"' ERR

cd "$BRICKNET_ROOT"

if [[ ! -x "$BRICKNET_PY" ]]; then
  printf 'BrickNet Python is not executable: %s\n' "$BRICKNET_PY" >&2
  exit 1
fi

printf 'Validating frozen Stage 6-7 config, hashes, adapters, data, media, software, and output mapping...\n'
"$BRICKNET_PY" - "$CONTRACT_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from bricknet.agentic_contract import canonical_json_sha256
from scripts.evaluate_bricknet_agentic_stage67 import validate_config

config_path = Path(sys.argv[1]).resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
validate_config(config)
contract = config["inference_contract"]
print(f"Static contract OK: {canonical_json_sha256(contract)}")
for experiment in contract["experiments"]:
    print(f"  {experiment['mode']}: {experiment['audit']}")
PY

if (( PREFLIGHT_ONLY )); then
  printf 'Preflight-only validation passed; no inference or evaluation was started.\n'
  exit 0
fi

wait_for_idle_gpu0() {
  local poll_count=0
  local state
  local utilization
  local used_memory
  local compute_pids

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi is required for the formal cuda:0 launcher.\n' >&2
    return 1
  fi
  while true; do
    state=$(nvidia-smi \
      --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits \
      --id=0)
    IFS=',' read -r utilization used_memory <<< "$state"
    utilization=${utilization//[[:space:]]/}
    used_memory=${used_memory//[[:space:]]/}
    compute_pids=$(nvidia-smi \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits \
      --id=0 | tr -d '[:space:]')

    if [[ -z "$compute_pids" ]] && (( utilization < 20 && used_memory < 4096 )); then
      printf '[%s] GPU 0 is idle (%s%% utilization, %s MiB used); starting the formal rerun.\n' \
        "$(date --iso-8601=seconds)" "$utilization" "$used_memory"
      return 0
    fi
    if (( poll_count % 20 == 0 )); then
      printf '[%s] GPU 0 is busy (%s%% utilization, %s MiB used, compute PID(s): %s); checking again in 30 seconds.\n' \
        "$(date --iso-8601=seconds)" "$utilization" "$used_memory" "${compute_pids:-none}"
    fi
    poll_count=$((poll_count + 1))
    sleep 30
  done
}

wait_for_idle_gpu0

COMMON_ARGS=(
  --input "$BRICKNET_DATASET"
  --backend hf
  --prompt-protocol exp4_2-stepwise
  --seed 42
  --model "$MODEL"
  --model-revision "$MODEL_REVISION"
  --contract-config "$CONTRACT_CONFIG"
  --stage5-report "$STAGE5_REPORT"
  --media-dir "$BRICKNET_ROOT"
  --device cuda:0
  --dtype bfloat16
  --temperature 1.0
  --top-k 20
  --top-p 0.95
  --candidates-per-round 8
  --max-rounds-per-state 4
  --max-backtrack-depth 3
  --max-expansions 0
  --max-turns 0
  --max-action-tokens 256
)

run_controller() {
  local slug=$1
  local mode=$2
  local output=$3

  printf '[%s] Starting %s (%s); official artifacts will be atomically overwritten on success.\n' \
    "$(date --iso-8601=seconds)" "${slug^^}" "$mode"
  "$BRICKNET_PY" scripts/run_bricknet_agentic_inference.py \
    "${COMMON_ARGS[@]}" \
    --output "$output" \
    --mode "$mode"

  printf '[%s] Validating the completed %s artifacts before continuing.\n' \
    "$(date --iso-8601=seconds)" "${slug^^}"
  "$BRICKNET_PY" scripts/evaluate_bricknet_agentic_stage67.py \
    --config "$CONTRACT_CONFIG" \
    --action preflight \
    --runs "$slug"
}

run_controller \
  b1 \
  b1-post-hoc \
  "$OUTPUT_ROOT/agentic_exp4_2_b1/controller_audit.jsonl"

run_controller \
  v1 \
  v1-silent-retry \
  "$OUTPUT_ROOT/agentic_exp4_2_v1/controller_audit.jsonl"

run_controller \
  v2 \
  v2-silent-dfs \
  "$OUTPUT_ROOT/agentic_exp4_2_v2/controller_audit.jsonl"

run_controller \
  a0 \
  a0-act-feedback \
  "$OUTPUT_ROOT/agentic_exp4_2_a0/controller_audit.jsonl"

run_controller \
  a1 \
  a1-feedback-search \
  "$OUTPUT_ROOT/agentic_exp4_2_a1/controller_audit.jsonl"

printf '[%s] All controller artifacts passed preflight; forcing complete final/diagnostic evaluation and bootstrap rebuild.\n' \
  "$(date --iso-8601=seconds)"
"$BRICKNET_PY" scripts/evaluate_bricknet_agentic_stage67.py \
  --config "$CONTRACT_CONFIG" \
  --action all \
  --execute \
  --force

STATS="$OUTPUT_ROOT/agentic_exp4_2_stage67_statistics.json"
"$BRICKNET_PY" - "$STATS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
stats = json.loads(path.read_text(encoding="utf-8"))

print("Final-system scores:")
for name, run in stats["runs"].items():
    final = run["final_system"]
    controller = run["controller"]
    print(
        f"  {name}: strict={final['task_strict_success'] * 100:.2f}% "
        f"dense={final['dense_reward']:.6f} "
        f"hard_valid={controller['controller_hard_valid_success']}/{stats['expected_count']}"
    )

print("Paired final-system strict deltas (candidate - baseline):")
for name, comparison in stats["comparisons"].items():
    metric = comparison["final_system"]["task_strict_success"]
    print(
        f"  {name}: delta={metric['candidate_minus_baseline'] * 100:+.2f} pp "
        f"95% CI=[{metric['ci95_low'] * 100:+.2f}, {metric['ci95_high'] * 100:+.2f}] pp"
    )

for name, comparison in stats.get("external_comparisons", {}).items():
    metric = comparison["task_strict_success"]
    print(
        f"  {name}: delta={metric['candidate_minus_baseline'] * 100:+.2f} pp "
        f"95% CI=[{metric['ci95_low'] * 100:+.2f}, {metric['ci95_high'] * 100:+.2f}] pp"
    )

print(f"Full experiment table: {path.with_name('agentic_exp4_2_stage67_results.md')}")
PY

printf '[%s] Stage 6-7 rerun and evaluation completed successfully.\n' "$(date --iso-8601=seconds)"
