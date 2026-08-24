#!/usr/bin/env bash
# Isolated PT-exp2 MM v2-no-meta pipeline:
# e1 train -> e1 predict -> e1 base+alignment evaluate -> e2 ... -> e3 evaluate.
#
# Start from the repository root with:
#   nohup bash tmp_bash/run_pt_exp2_mm_e1_e2_e3_v2.sh >/dev/null 2>&1 &
# All nine stages use physical CUDA device 1. GPU-idle waiting remains disabled.

set -Eeuo pipefail
umask 002

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BRICKNET_ROOT="/data/jiahao/task/BrickNet"
PYTHON_BIN="/home/jiahao/miniconda3/envs/llamafactory/bin/python"
BRICKNET_PYTHON="/home/jiahao/miniconda3/envs/bricknet/bin/python"
MS_SWIFT_ROOT="/data/jiahao/task/ms-swift"
MS_SWIFT_EVALUATOR="${MS_SWIFT_ROOT}/examples/train/grpo/plugin/bricknet/evaluate_experiment.py"
ALIGNMENT_DATASET="${BRICKNET_ROOT}/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl"
LAUNCHER="${REPO_ROOT}/scripts/launch_bricknet_pt_exp2_mm_v2.py"
BUILDER="${REPO_ROOT}/scripts/build_bricknet_pt_exp2_mm_train_view_v2.py"
SAVE_ROOT="${REPO_ROOT}/saves/Qwen3.5-0.8B-Thinking/lora"
EVAL_ROOT="${BRICKNET_ROOT}/outputs_val/qwen35_08b"
LDVIEW_BIN="/home/jiahao/.local/bin/ldview"
LDRAW_DIR="${BRICKNET_ROOT}/data/bricknet_datasets/ldraw"
BRICKNET_DATA="${PT_EXP2_MM_V2_BRICKNET_DATA:-/home/jiahao/.local/share/bricknet}"

LOG_FILE="${PT_EXP2_MM_V2_LOG_FILE:-${SCRIPT_DIR}/pt_exp2_mm_e1_e2_e3_v2.nohup.log}"
LOCK_FILE="${SCRIPT_DIR}/pt_exp2_mm_e1_e2_e3_v2.lock"
MIN_FREE_GIB="${PT_EXP2_MM_V2_MIN_FREE_GIB:-40}"

mkdir -p -- "${SCRIPT_DIR}"
touch -- "${LOG_FILE}"
exec >>"${LOG_FILE}" 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%F %T %z')" "$*"
}

die() {
  log "FATAL: $*"
  exit 1
}

on_error() {
  local exit_code="$1"
  local line_no="$2"
  local command_text="$3"
  log "ERROR: exit=${exit_code} line=${line_no} command=${command_text}"
  exit "${exit_code}"
}

trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap 'log "Received termination signal; stopping after current process exits."; exit 130' INT TERM

export PATH="/home/jiahao/miniconda3/bin:${PATH}"
export LDVIEW_BIN LDRAW_DIR BRICKNET_DATA

[[ "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]] || die "PT_EXP2_MM_V2_MIN_FREE_GIB must be a non-negative integer"
(( MIN_FREE_GIB > 0 )) || die "PT_EXP2_MM_V2_MIN_FREE_GIB must be positive"

command -v flock >/dev/null 2>&1 || die "flock is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
command -v conda >/dev/null 2>&1 || die "conda is required"
[[ -x "${PYTHON_BIN}" ]] || die "missing LlamaFactory Python: ${PYTHON_BIN}"
[[ -x "${BRICKNET_PYTHON}" ]] || die "missing BrickNet Python: ${BRICKNET_PYTHON}"
[[ -f "${LAUNCHER}" ]] || die "missing v2 launcher: ${LAUNCHER}"
[[ -f "${BUILDER}" ]] || die "missing v2 builder: ${BUILDER}"
[[ -d "${BRICKNET_ROOT}" ]] || die "missing BrickNet repository: ${BRICKNET_ROOT}"
[[ -x "${LDVIEW_BIN}" ]] || die "missing LDView executable: ${LDVIEW_BIN}"
[[ -d "${LDRAW_DIR}" ]] || die "missing LDraw library: ${LDRAW_DIR}"
[[ -d "${BRICKNET_DATA}/inset" ]] || die "missing BrickNet collision meshes: ${BRICKNET_DATA}/inset"

exec 9>"${LOCK_FILE}"
flock -n 9 || die "another PT-exp2 MM v2 batch holds ${LOCK_FILE}"

declare -A TRAIN_DIRS=(
  [mm-e1]="${SAVE_ROOT}/train_PT_exp2_mm_e1_v2_nometa_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400"
  [mm-e2]="${SAVE_ROOT}/train_PT_exp2_mm_e2_v2_nometa_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400"
  [mm-e3]="${SAVE_ROOT}/train_PT_exp2_mm_e3_v2_nometa_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400"
)

declare -A PREDICT_DIRS=(
  [mm-e1]="${SAVE_ROOT}/eval_PT_exp2_mm_e1_v2_nometa_ptval_in4096_out4096_p95_t1_k20"
  [mm-e2]="${SAVE_ROOT}/eval_PT_exp2_mm_e2_v2_nometa_ptval_in4096_out4096_p95_t1_k20"
  [mm-e3]="${SAVE_ROOT}/eval_PT_exp2_mm_e3_v2_nometa_ptval_in4096_out4096_p95_t1_k20"
)

declare -A EVAL_DIRS=(
  [mm-e1]="${EVAL_ROOT}/eval_PT_exp2_mm_e1_v2_nometa_ptval_in4096_out4096_p95_t1_k20"
  [mm-e2]="${EVAL_ROOT}/eval_PT_exp2_mm_e2_v2_nometa_ptval_in4096_out4096_p95_t1_k20"
  [mm-e3]="${EVAL_ROOT}/eval_PT_exp2_mm_e3_v2_nometa_ptval_in4096_out4096_p95_t1_k20"
)

check_free_space() {
  local path available_bytes required_bytes
  required_bytes=$((MIN_FREE_GIB * 1024 * 1024 * 1024))
  for path in "${REPO_ROOT}" "${BRICKNET_ROOT}"; do
    available_bytes="$(df -PB1 -- "${path}" | awk 'NR == 2 {print $4}')"
    [[ "${available_bytes}" =~ ^[0-9]+$ ]] || die "could not determine free space for ${path}"
    (( available_bytes >= required_bytes )) || die "free space below ${MIN_FREE_GIB} GiB for ${path}"
  done
}

validate_static_inputs() {
  local report epoch
  "${PYTHON_BIN}" "${BUILDER}" --verify-only >/dev/null
  [[ -f "${REPO_ROOT}/data/BrickNet-MM_PT_VAL.json" ]] || die "missing BrickNet-MM PT VAL data"
  [[ -f "${BRICKNET_ROOT}/scripts/evaluate_experiment.py" ]] || die "missing BrickNet evaluator"
  [[ -f "${MS_SWIFT_EVALUATOR}" ]] || die "missing BrickNet alignment evaluator"
  [[ -f "${ALIGNMENT_DATASET}" ]] || die "missing BrickNet alignment VAL512 dataset"
  [[ -x "${BRICKNET_ROOT}/scripts/render_ldview_8views.sh" ]] || die "missing LDView render script"
  [[ -f "${BRICKNET_ROOT}/hf_checkpoints/timm/PE-Core-bigG-14-448/open_clip_model.safetensors" ]] || die "missing PE checkpoint"
  [[ -f "${BRICKNET_ROOT}/hf_checkpoints/google/siglip2-giant-opt-patch16-384/model.safetensors.index.json" ]] || die "missing SigLIP2 checkpoint"
  [[ -f "${BRICKNET_ROOT}/hf_checkpoints/Qwen/Qwen2-VL-7B-Instruct/model.safetensors.index.json" ]] || die "missing VQAScore checkpoint"
  for epoch in e1 e2 e3; do
    report="${BRICKNET_ROOT}/outputs_preprocess/BrickNet-MM-PT-exp2/reports/token_audit_mm6400_v2/${epoch}/BrickNet-MM-Reasoning_token_audit_report.json"
    [[ -f "${report}" ]] || die "missing v2 token audit report: ${report}"
    "${PYTHON_BIN}" - "${report}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not (
    report.get("is_full_pool") is True
    and report.get("zero_errors") is True
    and report.get("zero_truncation") is True
    and report.get("training_eligible") is True
):
    raise SystemExit(f"v2 token audit is not eligible: {sys.argv[1]}")
PY
  done
}

train_complete() {
  local output_dir="${TRAIN_DIRS[$1]}"
  [[ -f "${output_dir}/adapter_config.json" ]] || return 1
  [[ -f "${output_dir}/adapter_model.safetensors" || -f "${output_dir}/adapter_model.bin" ]] || return 1
  [[ -f "${output_dir}/trainer_state.json" && -f "${output_dir}/train_results.json" ]] || return 1
  "${PYTHON_BIN}" - "${output_dir}/trainer_state.json" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = int(state.get("global_step", -1))
max_steps = int(state.get("max_steps", -2))
if step <= 0 or step != max_steps:
    raise SystemExit(1)
PY
}

validate_jsonl_count() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
count = 0
with path.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            json.loads(line)
            count += 1
if count != expected:
    raise SystemExit(f"{path}: expected {expected} JSONL rows, found {count}")
PY
}

predict_complete() {
  local output_dir="${PREDICT_DIRS[$1]}"
  [[ -f "${output_dir}/generated_predictions.jsonl" ]] || return 1
  [[ -f "${output_dir}/predict_results.json" ]] || return 1
  validate_jsonl_count "${output_dir}/generated_predictions.jsonl" 512
}

evaluate_complete() {
  local output_dir="${EVAL_DIRS[$1]}"
  [[ -f "${output_dir}/metrics.json" ]] || return 1
  [[ -f "${output_dir}/evaluation_manifest.json" ]] || return 1
  [[ -f "${output_dir}/alignment_input.jsonl" ]] || return 1
  [[ -f "${output_dir}/alignment.jsonl" ]] || return 1
  [[ -f "${output_dir}/alignment_manifest.json" ]] || return 1
  [[ -f "${output_dir}/metrics.md" ]] || return 1
  [[ -f "${output_dir}/scored.jsonl" ]] || return 1
  [[ -f "${PREDICT_DIRS[$1]}/generated_predictions.jsonl" ]] || return 1
  "${PYTHON_BIN}" - \
    "$1" \
    "${output_dir}/metrics.json" \
    "${output_dir}/metrics.md" \
    "${output_dir}/evaluation_manifest.json" \
    "${output_dir}/alignment_input.jsonl" \
    "${output_dir}/alignment.jsonl" \
    "${output_dir}/alignment_manifest.json" \
    "${output_dir}/scored.jsonl" \
    "${PREDICT_DIRS[$1]}/generated_predictions.jsonl" \
    "${ALIGNMENT_DATASET}" \
    "${MS_SWIFT_EVALUATOR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_name = sys.argv[1]
metrics_path = Path(sys.argv[2])
metrics_md = Path(sys.argv[3])
base_manifest_path = Path(sys.argv[4])
alignment_input = Path(sys.argv[5])
alignment = Path(sys.argv[6])
alignment_manifest = json.loads(Path(sys.argv[7]).read_text(encoding="utf-8"))
scored = Path(sys.argv[8])
predictions = Path(sys.argv[9])
dataset = Path(sys.argv[10])
evaluator = Path(sys.argv[11])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))

def count_jsonl(path):
    count = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise SystemExit(1)
                count += 1
    return count

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

artifacts = alignment_manifest.get("artifacts", {})
task_alignment = metrics.get("task_alignment", {})
condition_generation = metrics.get("condition_generation", {})
expected_identity = {
    "run": run_name,
    "expected_samples": 512,
    "predictions": str(predictions.resolve()),
    "predictions_sha256": sha256(predictions),
    "scored": str(scored.resolve()),
    "scored_sha256": sha256(scored),
    "alignment_dataset": str(dataset.resolve()),
    "alignment_dataset_sha256": sha256(dataset),
    "alignment_evaluator": str(evaluator.resolve()),
    "alignment_evaluator_sha256": sha256(evaluator),
    "base_evaluation_manifest": str(base_manifest_path.resolve()),
    "base_evaluation_manifest_sha256": sha256(base_manifest_path),
    "pose_tolerances": {
        "translation": 0.5,
        "rotation_degrees": 5.0,
        "success_threshold": 1.0,
    },
    "reward_weights": {
        "parse_prefix": 0.2,
        "inventory_f1": 0.2,
        "length_score": 0.1,
        "collision_prefix": 0.2,
        "pose_match": 0.3,
    },
}
if not (
    base_manifest.get("status") == "complete"
    and base_manifest.get("samples") == 512
    and metrics.get("artifacts", {}).get("predictions") == 512
    and metrics.get("structure", {}).get("samples") == 512
    and task_alignment.get("samples") == 512
    and task_alignment.get("weights") == expected_identity["reward_weights"]
    and task_alignment.get("pose_tolerances") == expected_identity["pose_tolerances"]
    and isinstance(task_alignment.get("dense_reward_mean"), (int, float))
    and isinstance(task_alignment.get("strict_success"), int)
    and isinstance(task_alignment.get("strict_success_rate"), (int, float))
    and condition_generation.get("samples") == 512
    and condition_generation.get("dense_reward") == task_alignment.get("dense_reward_mean")
    and condition_generation.get("strict_success_num") == task_alignment.get("strict_success")
    and condition_generation.get("strict_success_rate") == task_alignment.get("strict_success_rate")
    and alignment_manifest.get("schema_version") == "bricknet-pt-exp2-mm-alignment-v1"
    and alignment_manifest.get("status") == "complete"
    and alignment_manifest.get("identity") == expected_identity
    and count_jsonl(alignment_input) == 512
    and count_jsonl(alignment) == 512
    and artifacts.get("alignment_input_sha256") == sha256(alignment_input)
    and artifacts.get("alignment_sha256") == sha256(alignment)
    and artifacts.get("metrics_json_sha256") == sha256(metrics_path)
    and artifacts.get("metrics_md_sha256") == sha256(metrics_md)
):
    raise SystemExit(1)
PY
}

stage_complete() {
  case "$1" in
    train) train_complete "$2" ;;
    predict) predict_complete "$2" ;;
    evaluate) evaluate_complete "$2" ;;
    *) die "unknown action: $1" ;;
  esac
}

assert_target_safe() {
  local action="$1" run="$2" target
  case "${action}" in
    train) target="${TRAIN_DIRS[${run}]}" ;;
    predict) target="${PREDICT_DIRS[${run}]}" ;;
    evaluate) target="${EVAL_DIRS[${run}]}" ;;
    *) die "unknown action: ${action}" ;;
  esac
  if [[ "${action}" == evaluate && -e "${target}" ]]; then
    log "Evaluation output exists but is incomplete; evaluator may resume only with a matching manifest: ${target}"
  elif [[ -e "${target}" ]]; then
    die "incomplete ${action} output requires manual review: ${target}"
  fi
}

validate_launcher_ready() {
  local payload="$1"
  printf '%s\n' "${payload}"
  printf '%s\n' "${payload}" | "${PYTHON_BIN}" -c '
import json
import sys
payload = json.load(sys.stdin)
if payload.get("ready") is not True or payload.get("blockers"):
    print("launcher blockers:", payload.get("blockers"), file=sys.stderr)
    raise SystemExit(1)
' || die "v2 launcher dry-run is not ready"
}

run_stage() {
  local action="$1" run="$2" dry_run
  local -a args=(--gpus 1 --action "${action}" --run "${run}")
  if stage_complete "${action}" "${run}"; then
    log "SKIP complete stage: ${run} ${action}"
    return
  fi
  assert_target_safe "${action}" "${run}"
  check_free_space

  # User-approved exception: physical CUDA 1 is fixed for train/predict/evaluate.
  # GPU-idle waiting is intentionally disabled; the launcher only reports occupants.
  # wait_for_gpus 1

  log "Dry-run preflight: ${run} ${action}"
  dry_run="$("${PYTHON_BIN}" "${LAUNCHER}" "${args[@]}")"
  validate_launcher_ready "${dry_run}"
  log "START: ${run} ${action}"
  "${PYTHON_BIN}" "${LAUNCHER}" "${args[@]}" --execute
  stage_complete "${action}" "${run}" || die "completion artifacts failed validation: ${run} ${action}"
  log "COMPLETE: ${run} ${action}"
}

main() {
  local run
  cd -- "${REPO_ROOT}"
  log "Batch start: isolated PT-exp2 MM v2-no-meta on physical CUDA 1"
  log "Sequence: e1 train/predict/evaluate -> e2 -> e3"
  log "Log file: ${LOG_FILE}"
  log "GPU-idle waiting is disabled by user approval."
  validate_static_inputs
  check_free_space
  for run in mm-e1 mm-e2 mm-e3; do
    run_stage train "${run}"
    run_stage predict "${run}"
    run_stage evaluate "${run}"
  done
  log "Batch complete: all nine v2 stages validated; PT-exp2-v2 alias selection remains manual."
}

main "$@"
