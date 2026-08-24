#!/usr/bin/env bash
# Sequential PT-exp2 MM pipeline:
# e1 train -> e1 predict -> e1 evaluate -> e2 ... -> e3 evaluate.
#
# Start from the repository root with:
#   nohup bash tmp_batch/run_pt_exp2_mm_e1_e2_e3.sh >/dev/null 2>&1 &
# The script writes its own append-only log next to this file. All nine stages
# are pinned to physical CUDA device 1; GPU-idle waiting is intentionally disabled.

set -Eeuo pipefail
umask 002

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BRICKNET_ROOT="/data/jiahao/task/BrickNet"
PYTHON_BIN="/home/jiahao/miniconda3/envs/llamafactory/bin/python"
BRICKNET_PYTHON="/home/jiahao/miniconda3/envs/bricknet/bin/python"
LAUNCHER="${REPO_ROOT}/scripts/launch_bricknet_pt_exp2.py"
SAVE_ROOT="${REPO_ROOT}/saves/Qwen3.5-0.8B-Thinking/lora"
EVAL_ROOT="${BRICKNET_ROOT}/outputs_val/qwen35_08b"
LDVIEW_BIN="/home/jiahao/.local/bin/ldview"
LDRAW_DIR="${BRICKNET_ROOT}/data/bricknet_datasets/ldraw"
BRICKNET_DATA="${PT_EXP2_MM_BRICKNET_DATA:-/home/jiahao/.local/share/bricknet}"

LOG_FILE="${PT_EXP2_MM_LOG_FILE:-${SCRIPT_DIR}/pt_exp2_mm_e1_e2_e3.nohup.log}"
LOCK_FILE="${SCRIPT_DIR}/pt_exp2_mm_e1_e2_e3.lock"
GPU_POLL_SECONDS="${PT_EXP2_MM_GPU_POLL_SECONDS:-60}"
GPU_WAIT_TIMEOUT_SECONDS="${PT_EXP2_MM_GPU_WAIT_TIMEOUT_SECONDS:-0}"
MIN_FREE_GIB="${PT_EXP2_MM_MIN_FREE_GIB:-40}"

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

for numeric_value in "${GPU_POLL_SECONDS}" "${GPU_WAIT_TIMEOUT_SECONDS}" "${MIN_FREE_GIB}"; do
  [[ "${numeric_value}" =~ ^[0-9]+$ ]] || die "numeric runtime settings must be non-negative integers"
done
(( GPU_POLL_SECONDS > 0 )) || die "PT_EXP2_MM_GPU_POLL_SECONDS must be positive"
(( MIN_FREE_GIB > 0 )) || die "PT_EXP2_MM_MIN_FREE_GIB must be positive"

command -v flock >/dev/null 2>&1 || die "flock is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
command -v conda >/dev/null 2>&1 || die "conda is required"
[[ -x "${PYTHON_BIN}" ]] || die "missing LlamaFactory Python: ${PYTHON_BIN}"
[[ -x "${BRICKNET_PYTHON}" ]] || die "missing BrickNet Python: ${BRICKNET_PYTHON}"
[[ -f "${LAUNCHER}" ]] || die "missing launcher: ${LAUNCHER}"
[[ -d "${BRICKNET_ROOT}" ]] || die "missing BrickNet repository: ${BRICKNET_ROOT}"
[[ -x "${LDVIEW_BIN}" ]] || die "missing LDView executable: ${LDVIEW_BIN}"
[[ -d "${LDRAW_DIR}" ]] || die "missing LDraw library: ${LDRAW_DIR}"
[[ -d "${BRICKNET_DATA}/inset" ]] || die "missing BrickNet collision meshes: ${BRICKNET_DATA}/inset"

exec 9>"${LOCK_FILE}"
flock -n 9 || die "another PT-exp2 MM batch script already holds ${LOCK_FILE}"

declare -A TRAIN_DIRS=(
  [mm-e1]="${SAVE_ROOT}/train_PT_exp2_mm_e1_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400"
  [mm-e2]="${SAVE_ROOT}/train_PT_exp2_mm_e2_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400"
  [mm-e3]="${SAVE_ROOT}/train_PT_exp2_mm_e3_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400"
)

declare -A PREDICT_DIRS=(
  [mm-e1]="${SAVE_ROOT}/eval_PT_exp2_mm_e1_ptval_in4096_out4096_p95_t1_k20"
  [mm-e2]="${SAVE_ROOT}/eval_PT_exp2_mm_e2_ptval_in4096_out4096_p95_t1_k20"
  [mm-e3]="${SAVE_ROOT}/eval_PT_exp2_mm_e3_ptval_in4096_out4096_p95_t1_k20"
)

declare -A EVAL_DIRS=(
  [mm-e1]="${EVAL_ROOT}/eval_PT_exp2_mm_e1_ptval_in4096_out4096_p95_t1_k20"
  [mm-e2]="${EVAL_ROOT}/eval_PT_exp2_mm_e2_ptval_in4096_out4096_p95_t1_k20"
  [mm-e3]="${EVAL_ROOT}/eval_PT_exp2_mm_e3_ptval_in4096_out4096_p95_t1_k20"
)

check_free_space() {
  local path available_bytes required_bytes
  required_bytes=$((MIN_FREE_GIB * 1024 * 1024 * 1024))
  for path in "${REPO_ROOT}" "${BRICKNET_ROOT}"; do
    available_bytes="$(df -PB1 -- "${path}" | awk 'NR == 2 {print $4}')"
    [[ "${available_bytes}" =~ ^[0-9]+$ ]] || die "could not determine free space for ${path}"
    if (( available_bytes < required_bytes )); then
      die "free space below ${MIN_FREE_GIB} GiB on filesystem containing ${path}"
    fi
  done
}

validate_static_inputs() {
  local report run
  [[ -f "${REPO_ROOT}/data/bricknet_pt_exp2/mm/manifest.json" ]] || die "missing MM manifest"
  [[ -f "${REPO_ROOT}/data/BrickNet-MM_PT_VAL.json" ]] || die "missing BrickNet-MM PT VAL data"
  [[ -f "${BRICKNET_ROOT}/scripts/evaluate_experiment.py" ]] || die "missing BrickNet evaluator"
  [[ -x "${BRICKNET_ROOT}/scripts/render_ldview_8views.sh" ]] || die "missing LDView render script"
  [[ -f "${BRICKNET_ROOT}/hf_checkpoints/timm/PE-Core-bigG-14-448/open_clip_model.safetensors" ]] || die "missing PE checkpoint"
  [[ -f "${BRICKNET_ROOT}/hf_checkpoints/google/siglip2-giant-opt-patch16-384/model.safetensors.index.json" ]] || die "missing SigLIP2 checkpoint"
  [[ -f "${BRICKNET_ROOT}/hf_checkpoints/Qwen/Qwen2-VL-7B-Instruct/model.safetensors.index.json" ]] || die "missing VQAScore checkpoint"
  "${PYTHON_BIN}" - "${REPO_ROOT}/data/BrickNet-MM_PT_VAL.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
if len(rows) != 512:
    raise SystemExit(f"expected 512 PT VAL rows, found {len(rows)}")
PY
  for run in e1 e2 e3; do
    report="${BRICKNET_ROOT}/outputs_preprocess/BrickNet-MM-PT-exp2/reports/token_audit_mm6400/${run}/BrickNet-MM-Reasoning_token_audit_report.json"
    [[ -f "${report}" ]] || die "missing token audit report: ${report}"
    "${PYTHON_BIN}" - "${report}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
if not (
    report.get("is_full_pool") is True
    and report.get("zero_errors") is True
    and report.get("zero_truncation") is True
    and report.get("training_eligible") is True
):
    raise SystemExit(f"token audit is not eligible: {path}")
PY
  done
}

gpu_process_snapshot() {
  local gpu raw
  for gpu in "$@"; do
    if ! raw="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits 2>&1)"; then
      die "nvidia-smi failed for GPU ${gpu}: ${raw}"
    fi
    if [[ -n "${raw//[[:space:]]/}" ]]; then
      while IFS= read -r line; do
        [[ -n "${line}" ]] && printf 'gpu=%s %s\n' "${gpu}" "${line}"
      done <<<"${raw}"
    fi
  done
}

wait_for_gpus() {
  local start_epoch now_epoch elapsed snapshot gpu_list
  gpu_list="$*"
  start_epoch="$(date +%s)"
  while true; do
    snapshot="$(gpu_process_snapshot "$@")"
    if [[ -z "${snapshot//[[:space:]]/}" ]]; then
      log "GPU(s) ${gpu_list} are free."
      return
    fi
    now_epoch="$(date +%s)"
    elapsed=$((now_epoch - start_epoch))
    if (( GPU_WAIT_TIMEOUT_SECONDS > 0 && elapsed >= GPU_WAIT_TIMEOUT_SECONDS )); then
      die "GPU wait timed out after ${elapsed}s; active processes: ${snapshot//$'\n'/; }"
    fi
    log "Waiting for GPU(s) ${gpu_list}; active processes: ${snapshot//$'\n'/; }"
    sleep "${GPU_POLL_SECONDS}"
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
    for line_no, line in enumerate(handle, 1):
        if not line.strip():
            continue
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
  [[ -f "${output_dir}/metrics.json" && -f "${output_dir}/evaluation_manifest.json" ]] || return 1
  "${PYTHON_BIN}" - "${output_dir}/metrics.json" "${output_dir}/evaluation_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if manifest.get("status") != "complete":
    raise SystemExit(1)
if metrics.get("artifacts", {}).get("predictions") != 512:
    raise SystemExit(1)
PY
}

stage_complete() {
  local action="$1" run="$2"
  case "${action}" in
    train) train_complete "${run}" ;;
    predict) predict_complete "${run}" ;;
    evaluate) evaluate_complete "${run}" ;;
    *) die "unknown action: ${action}" ;;
  esac
}

assert_target_safe() {
  local action="$1" run="$2" target
  case "${action}" in
    train)
      target="${TRAIN_DIRS[${run}]}"
      [[ ! -e "${target}" ]] || die "incomplete training output requires manual review: ${target}"
      ;;
    predict)
      target="${PREDICT_DIRS[${run}]}"
      [[ ! -e "${target}" ]] || die "incomplete prediction output requires manual review: ${target}"
      ;;
    evaluate)
      target="${EVAL_DIRS[${run}]}"
      if [[ -e "${target}" ]]; then
        log "Evaluation output exists but is incomplete; evaluator will resume only if its manifest matches: ${target}"
      fi
      ;;
    *) die "unknown action: ${action}" ;;
  esac
}

launcher_args() {
  local action="$1" run="$2"
  case "${action}" in
    train) printf '%s\0' --gpus 1 --action train --run "${run}" ;;
    predict) printf '%s\0' --gpus 1 --action predict --run "${run}" ;;
    evaluate) printf '%s\0' --gpus 1 --action evaluate --run "${run}" ;;
    *) die "unknown action: ${action}" ;;
  esac
}

validate_launcher_ready() {
  local payload="$1"
  printf '%s\n' "${payload}"
  if ! printf '%s\n' "${payload}" | "${PYTHON_BIN}" -c '
import json
import sys
payload = json.load(sys.stdin)
if payload.get("ready") is not True or payload.get("blockers"):
    print("launcher blockers:", payload.get("blockers"), file=sys.stderr)
    raise SystemExit(1)
'; then
    die "launcher dry-run is not ready"
  fi
}

run_stage() {
  local action="$1" run="$2" dry_run
  local -a args=()

  if stage_complete "${action}" "${run}"; then
    log "SKIP complete stage: ${run} ${action}"
    return
  fi

  assert_target_safe "${action}" "${run}"
  check_free_space

  # User-approved exception: all stages use physical CUDA device 1 even when
  # nvidia-smi reports an existing compute process. Keep these calls disabled.
  # case "${action}" in
  #   train|predict|evaluate) wait_for_gpus 1 ;;
  # esac

  while IFS= read -r -d '' item; do
    args+=("${item}")
  done < <(launcher_args "${action}" "${run}")

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
  log "Batch start on physical CUDA device 1: e1 train/predict/evaluate -> e2 -> e3"
  log "Log file: ${LOG_FILE}"
  log "GPU-idle waiting is disabled by user approval; existing GPU 1 processes will not block launch."
  validate_static_inputs
  check_free_space

  for run in mm-e1 mm-e2 mm-e3; do
    run_stage train "${run}"
    run_stage predict "${run}"
    run_stage evaluate "${run}"
  done

  log "Batch complete: all nine stages validated. PT-exp2 final alias selection is intentionally not included."
}

main "$@"
