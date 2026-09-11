#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot supervisor for the official BrickNet-Render PT render.
#
# The supervisor owns only its own launcher process group and its own physical
# CUDA 0/1 sampler.  It never kills an unrelated CUDA process.  A fresh pilot
# and the formal render use nine workers per physical GPU (18 total).  The
# config keeps eight as its baseline worker value; the supervisor supplies the
# audited nine-worker override to every pilot/preflight/formal invocation.
# The selected topology and worker count are written into the PT identity and
# must not change on ordinary resume.
#
# A clean post-format run creates a fresh dual-GPU [0,1]/9 identity.  The
# historical single-GPU [1]/9 -> dual-GPU [0,1]/9 transition remains available
# only if that exact old identity is deliberately restored; it never triggers
# for an empty production tree and arbitrary refreezes are rejected.
#
# A successful formal launcher is intentionally reported as
# RENDER_COMPLETE_PENDING_VERIFY.  The supervisor does not claim strict
# image verification or generate a transfer gate.

SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BRICKNET_ROOT="/data/jiahao/task/BrickNet"
RENDER_REPO="/data/jiahao/task/BrickNet-Render"
OFFICIAL_LAUNCHER="${BRICKNET_ROOT}/scripts/render_bricknet_render_8views_official_pt.sh"
DEFAULT_CONFIG="${BRICKNET_ROOT}/configs/bricknet_mm_image_v2_official_pt_render.json"
RENDER_DRIVER="${BRICKNET_ROOT}/scripts/render_bricknet_render_8views.py"
PT_NPZ="${BRICKNET_ROOT}/data/bricknet_datasets/pt.npz"

EXPECTED_CONFIG_SHA256="61ec07b40077e842df8c4b768ebee5ae2b391fb35d4f0ff23e36be982ca7918a"
EXPECTED_LAUNCHER_SHA256="770a355731944cd711644beb932cc9f4143ccd292e32b6bb1d7dfd85c9c740ce"
EXPECTED_DRIVER_SHA256="0d36ef76c802f5ce76f4c0c839800cd7f1c3cd644074c30e257ebddae6f205ec"
EXPECTED_PT_NPZ_SHA256="a3c9ebe27fa49c97a3dce2c76152522aad0bf1b9c6c1859ee639e23935597ab6"
EXPECTED_RENDERER_COMMIT="b49e8733782a5fa7040cc776b5fba8c55b0b894e"
EXPECTED_OLD_PT_IDENTITY_SHA256="de713eafb76cd637cd8c3c63c39da933eab775a1c363cbd834f9611bf9e15016"
ALLOWED_IDENTITY_MISMATCH_FIELDS="config_sha256,render_parameters_sha256,gpu_ids"
TOPOLOGY_REFREEZE_OLD_GPU_IDS_CSV="1"
TOPOLOGY_REFREEZE_OLD_WORKERS=9
TOPOLOGY_REFREEZE_NEW_GPU_IDS_CSV="0,1"
TOPOLOGY_REFREEZE_NEW_WORKERS=9

TMP_BASH="${SCRIPT_DIR}"
LOCK_FILE="${TMP_BASH}/supervise_official_pt_render.lock"
STATUS_FILE="${TMP_BASH}/supervise_official_pt_render.status"
PID_FILE="${TMP_BASH}/supervise_official_pt_render.pid"

RAW_ROOT="${BRICKNET_ROOT}/outputs_gt/pt_8view_renders_v2_official_rowids"
COLLAGE_ROOT="${BRICKNET_ROOT}/outputs_preprocess/BrickNet-MM/images_v2_official/PT"
METADATA_NAMESPACE=".bricknet_render_v2_official_pt"
METADATA_ROOT="${BRICKNET_ROOT}/outputs_gt/${METADATA_NAMESPACE}"

MIN_DATA_START_BYTES=280000000000
DISK_LOW_WATERMARK_BYTES=30000000000
# The audited PT pilot uses a stratified, higher-than-average-node subset.
# Keep a fixed 50 GB reserve on top of the direct full-row media projection.
DATA_RESERVE_BYTES=50000000000
FULL_PT_ROWS=135051
PILOT_PT_ROWS=32

GPU_PILOT_WORKERS_PER_GPU=9
GPU_DEFAULT_FORMAL_WORKERS_PER_GPU=9
# A 10-worker/GPU dual-card trial reached 96,529 MiB used on GPU 1 with only
# 721 MiB free; nine workers/GPU is therefore the hard candidate ceiling.
GPU_MAX_FORMAL_WORKERS_PER_GPU=9
# The audited empty-card dual9 pilot peaked at +44,007 MiB on GPU 0 and
# +42,795 MiB on GPU 1.  Require the larger observed delta plus the 8 GiB
# post-pilot reserve before starting another pilot; this still permits
# coexistence with unrelated jobs when they leave genuinely sufficient room.
GPU_PILOT_START_MIN_FREE_MIB=52200
GPU_PILOT_MIN_FREE_MIB=8192
GPU_PILOT_HARD_MIN_FREE_MIB=2048
GPU_FORMAL_SAFETY_MIB=8192
GPU_MAX_USED_MIB=95000
GPU_SAMPLE_INTERVAL_SECONDS=1

PILOT_ROW_IDS=(
  3 18466 36768 54758 72867 90859 108708 127497
  146162 8604 36674 65416 93203 121607 150801 40290
  95094 150780 49743 110312 22376 133916 102302 82485
  104384 44578 84695 132940 97971 61706 82776 14588
)

if (( ${#PILOT_ROW_IDS[@]} != PILOT_PT_ROWS )); then
  printf 'internal pilot row-id count mismatch: %s != %s\n' \
    "${#PILOT_ROW_IDS[@]}" "${PILOT_PT_ROWS}" >&2
  exit 2
fi

SELF_TEST=0
if (( $# > 0 )); then
  if (( $# == 1 )) && [[ "$1" == "--self-test" ]]; then
    SELF_TEST=1
  else
    printf 'usage: %s [--self-test]\n' "$0" >&2
    exit 2
  fi
fi

if (( SELF_TEST == 1 )); then
  LOCK_FILE="${TMP_BASH}/supervise_official_pt_render.selftest.lock"
  STATUS_FILE="${TMP_BASH}/supervise_official_pt_render.selftest.status"
  PID_FILE="${TMP_BASH}/supervise_official_pt_render.selftest.pid"
fi

mkdir -p -- "${TMP_BASH}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] another official PT supervisor already holds %s\n' \
    "$(date --iso-8601=seconds)" "${LOCK_FILE}" >&2
  exit 75
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SUPERVISOR_LOG="${TMP_BASH}/supervise_official_pt_render-${RUN_ID}.log"
FORMAL_GPU_SAMPLE_LOG="${TMP_BASH}/supervise_official_pt_render-${RUN_ID}.gpu.tsv"
GPU_PEAK_LOG="${TMP_BASH}/supervise_official_pt_render-${RUN_ID}.gpu-peaks.tsv"
FORMAL_GPU_SAMPLE_ERROR="${TMP_BASH}/supervise_official_pt_render-${RUN_ID}.gpu.error"
PILOT_GPU_SAMPLE_LOG="${TMP_BASH}/supervise_official_pt_render-${RUN_ID}.pilot.gpu.tsv"
PILOT_GPU_SAMPLE_ERROR="${TMP_BASH}/supervise_official_pt_render-${RUN_ID}.pilot.gpu.error"
GPU_SAMPLE_LOG="${FORMAL_GPU_SAMPLE_LOG}"
GPU_SAMPLE_ERROR="${FORMAL_GPU_SAMPLE_ERROR}"
PILOT_EVIDENCE_GPU_LOG="${PILOT_GPU_SAMPLE_LOG}"
EXTERNAL_PILOT_ROOT="${BRICKNET_OFFICIAL_PT_PILOT_ROOT:-}"
PILOT_REUSED=0
PILOT_ROOT="${EXTERNAL_PILOT_ROOT:-${RENDER_REPO}/tmp/official_pt_pilot_${RUN_ID}}"
if [[ -n "${EXTERNAL_PILOT_ROOT}" ]]; then
  PILOT_REUSED=1
fi
PILOT_RENDER_ROOT="${PILOT_ROOT}/render_output"
PILOT_IMAGES_ROOT="${PILOT_ROOT}/images_v2"
PILOT_WORK_ROOT="${PILOT_ROOT}/work"
PILOT_ID_MANIFEST="${PILOT_ROOT}/row_ids.txt"
PILOT_EXTERNAL_LOG="${PILOT_ROOT}/pilot.log"
PILOT_EXTERNAL_GPU_LOG="${PILOT_ROOT}/gpu.tsv"

CONFIG="${BRICKNET_OFFICIAL_PT_RENDER_CONFIG:-${DEFAULT_CONFIG}}"
PYTHON_BIN="${BRICKNET_PYTHON:-/home/jiahao/miniconda3/envs/bricknet/bin/python}"
WORKERS_OVERRIDE="${BRICKNET_OFFICIAL_PT_WORKERS_PER_GPU:-}"
REQUESTED_FORMAL_WORKERS="${WORKERS_OVERRIDE:-${GPU_DEFAULT_FORMAL_WORKERS_PER_GPU}}"
STOP_BEFORE_FORMAL="${BRICKNET_OFFICIAL_PT_STOP_BEFORE_FORMAL:-0}"
REFREEZE_WORKERS="${BRICKNET_OFFICIAL_PT_REFREEZE_WORKERS:-}"
# Export the supervisor-selected dual-GPU config so every child, including
# pilots and preflight, is bound to the same frozen contract.
export BRICKNET_OFFICIAL_PT_RENDER_CONFIG="${CONFIG}"
CONFIG_WORKERS_PER_GPU="unknown"
GPU_IDS_CSV="unknown"
GPU_IDS=()
CURRENT_STAGE="INIT"
STATUS_FINAL_WRITTEN=0
GPU_SAMPLER_PID=""
CHILD_PID=""
CHILD_KIND=""
GPU_SAMPLER_MODE=""
DATA_FREE_BYTES="unknown"
DATA_REQUIRED_BYTES="unknown"
DATA_PROJECTED_BYTES="unknown"
DATA_PILOT_BYTES="unknown"
DATA_EXISTING_COMPLETE_BYTES="unknown"
DATA_REMAINING_BYTES="unknown"
FORMAL_WORKERS="unknown"
FORMAL_WORKER_FORMULA="unknown"
GPU_PILOT_START_READY="unknown"
NEEDS_IDENTITY_ADOPTION=0
TOPOLOGY_REFREEZE_REQUESTED=0
TOPOLOGY_REFREEZE_COMPLETED=0
IDENTITY_REFREEZE_COMPLETED=0
IDENTITY_OLD_WORKERS="unknown"
IDENTITY_ADOPTION_ARCHIVE_ROOT="/data/jiahao/bricknet_render_work/images_v2_official_pt/failed/pt/reports"
IDENTITY_ADOPTION_ARCHIVE_COUNT_BEFORE=0
declare -A SNAP_MEM=()
declare -A SNAP_FREE=()
declare -A SNAP_UTIL=()
declare -A PILOT_BASELINE_MEM=()
declare -A PILOT_PEAK_MEM=()
declare -A PILOT_PER_WORKER_MIB=()
SNAPSHOT_TEXT=""

printf '%s\n' "$$" >"${PID_FILE}"
exec >>"${SUPERVISOR_LOG}" 2>&1

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

write_status() {
  local stage="$1"
  local result="$2"
  local detail="${3:-}"
  local now tmp_path resume_command pilot_resume_env=""
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  tmp_path="${STATUS_FILE}.tmp.$$"
  detail="${detail//$'\n'/ | }"
  # Once a pilot root exists, carry its exact path into the resumable command.
  # This lets STOP_BEFORE_FORMAL=1 validate a fresh pilot first, then resume
  # formal rendering without silently creating a different pilot.
  if [[ -d "${PILOT_ROOT}" && ! -L "${PILOT_ROOT}" ]]; then
    pilot_resume_env="BRICKNET_OFFICIAL_PT_PILOT_ROOT=${PILOT_ROOT}"
  fi
  if (( SELF_TEST == 1 )); then
    resume_command="none"
  elif (( NEEDS_IDENTITY_ADOPTION != 0 )); then
    resume_command="BRICKNET_OFFICIAL_PT_RENDER_CONFIG=${CONFIG} BRICKNET_OFFICIAL_PT_WORKERS_PER_GPU=${FORMAL_WORKERS} BRICKNET_OFFICIAL_PT_REFREEZE_WORKERS=${REFREEZE_WORKERS} ${pilot_resume_env} ${SCRIPT_PATH}"
  else
    resume_command="BRICKNET_OFFICIAL_PT_RENDER_CONFIG=${CONFIG} BRICKNET_OFFICIAL_PT_WORKERS_PER_GPU=${FORMAL_WORKERS} ${pilot_resume_env} ${SCRIPT_PATH}"
  fi
  {
    printf 'run_id=%s\n' "${RUN_ID}"
    if (( SELF_TEST == 1 )); then
      printf 'run_kind=SELF_TEST\n'
    else
      printf 'run_kind=PRODUCTION\n'
    fi
    printf 'pid=%s\n' "$$"
    printf 'stage=%s\n' "${stage}"
    printf 'result=%s\n' "${result}"
    printf 'updated_utc=%s\n' "${now}"
    printf 'config=%s\n' "${CONFIG}"
    printf 'config_sha256=%s\n' "${EXPECTED_CONFIG_SHA256}"
    printf 'launcher=%s\n' "${OFFICIAL_LAUNCHER}"
    printf 'launcher_sha256=%s\n' "${EXPECTED_LAUNCHER_SHA256}"
    printf 'driver=%s\n' "${RENDER_DRIVER}"
    printf 'driver_sha256=%s\n' "${EXPECTED_DRIVER_SHA256}"
    printf 'pt_npz=%s\n' "${PT_NPZ}"
    printf 'pt_npz_sha256=%s\n' "${EXPECTED_PT_NPZ_SHA256}"
    printf 'renderer_repo=%s\n' "${RENDER_REPO}"
    printf 'renderer_commit=%s\n' "${EXPECTED_RENDERER_COMMIT}"
    printf 'gpu_ids=%s\n' "${GPU_IDS_CSV}"
    printf 'config_workers_per_gpu=%s\n' "${CONFIG_WORKERS_PER_GPU}"
    printf 'pilot_workers_per_gpu=%s\n' "${GPU_PILOT_WORKERS_PER_GPU}"
    printf 'formal_workers_per_gpu=%s\n' "${FORMAL_WORKERS}"
    printf 'formal_worker_formula=%s\n' "${FORMAL_WORKER_FORMULA}"
    printf 'gpu_pilot_start_ready=%s\n' "${GPU_PILOT_START_READY}"
    printf 'gpu_pilot_start_min_free_mib=%s\n' "${GPU_PILOT_START_MIN_FREE_MIB}"
    printf 'refreeze_workers_requested=%s\n' "${REFREEZE_WORKERS:-none}"
    printf 'topology_refreeze_requested=%s\n' "${TOPOLOGY_REFREEZE_REQUESTED}"
    printf 'topology_refreeze_completed=%s\n' "${TOPOLOGY_REFREEZE_COMPLETED}"
    printf 'identity_old_workers=%s\n' "${IDENTITY_OLD_WORKERS}"
    printf 'needs_identity_adoption=%s\n' "${NEEDS_IDENTITY_ADOPTION}"
    printf 'identity_refreeze_completed=%s\n' "${IDENTITY_REFREEZE_COMPLETED}"
    printf 'identity_adoption_archive_root=%s\n' "${IDENTITY_ADOPTION_ARCHIVE_ROOT}"
    printf 'adoption_expected_old_identity_sha256=%s\n' "${EXPECTED_OLD_PT_IDENTITY_SHA256}"
    printf 'adoption_allowed_mismatch_fields=%s\n' "${ALLOWED_IDENTITY_MISMATCH_FIELDS}"
    printf 'raw_root=%s\n' "${RAW_ROOT}"
    printf 'collage_root=%s\n' "${COLLAGE_ROOT}"
    printf 'metadata_namespace=%s\n' "${METADATA_NAMESPACE}"
    printf 'pilot_root=%s\n' "${PILOT_ROOT}"
    printf 'pilot_reused=%s\n' "${PILOT_REUSED}"
    printf 'pilot_log=%s\n' "${PILOT_EXTERNAL_LOG}"
    printf 'pilot_external_gpu_log=%s\n' "${PILOT_EXTERNAL_GPU_LOG}"
    printf 'pilot_evidence_gpu_log=%s\n' "${PILOT_EVIDENCE_GPU_LOG}"
    printf 'pilot_gpu_log=%s\n' "${PILOT_EVIDENCE_GPU_LOG}"
    printf 'pilot_raw_root=%s\n' "${PILOT_RENDER_ROOT}/pt_8view_renders_v2_official_rowids"
    printf 'pilot_collage_root=%s\n' "${PILOT_IMAGES_ROOT}/PT"
    printf 'supervisor_log=%s\n' "${SUPERVISOR_LOG}"
    printf 'gpu_sample_log=%s\n' "${FORMAL_GPU_SAMPLE_LOG}"
    printf 'gpu_peak_log=%s\n' "${GPU_PEAK_LOG}"
    printf 'gpu_sample_error=%s\n' "${FORMAL_GPU_SAMPLE_ERROR}"
    printf 'pilot_gpu_sample_error=%s\n' "${PILOT_GPU_SAMPLE_ERROR}"
    printf 'pid_file=%s\n' "${PID_FILE}"
    printf 'data_available_bytes=%s\n' "${DATA_FREE_BYTES}"
    printf 'data_pilot_media_bytes=%s\n' "${DATA_PILOT_BYTES}"
    printf 'data_projected_media_bytes=%s\n' "${DATA_PROJECTED_BYTES}"
    printf 'data_existing_complete_media_bytes=%s\n' "${DATA_EXISTING_COMPLETE_BYTES}"
    printf 'data_remaining_media_bytes=%s\n' "${DATA_REMAINING_BYTES}"
    printf 'data_reserve_bytes=%s\n' "${DATA_RESERVE_BYTES}"
    printf 'data_required_bytes=%s\n' "${DATA_REQUIRED_BYTES}"
    printf 'data_start_min_bytes=%s\n' "${MIN_DATA_START_BYTES}"
    printf 'disk_low_watermark_bytes=%s\n' "${DISK_LOW_WATERMARK_BYTES}"
    printf 'resume_command=%s\n' "${resume_command}"
    printf 'detail=%s\n' "${detail}"
  } >"${tmp_path}"
  mv -f -- "${tmp_path}" "${STATUS_FILE}"
}

set_stage() {
  CURRENT_STAGE="$1"
  write_status "$1" RUNNING "${2:-}"
  log "stage=${1} result=RUNNING ${2:-}"
}

stage_ok() {
  CURRENT_STAGE="$1"
  write_status "$1" OK "${2:-}"
  log "stage=${1} result=OK ${2:-}"
}

fail_stage() {
  local message="$1"
  local exit_code="${2:-1}"
  STATUS_FINAL_WRITTEN=1
  write_status "${CURRENT_STAGE}" FAILED "${message} exit_code=${exit_code}"
  log "stage=${CURRENT_STAGE} result=FAILED ${message} exit_code=${exit_code}"
  exit "${exit_code}"
}

terminate_child_group() {
  local pid="${CHILD_PID:-}"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    log "sending TERM to owned ${CHILD_KIND} process group pgid=${pid}"
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  fi
  set +e
  wait "${pid}" 2>/dev/null
  set -e
  CHILD_PID=""
  CHILD_KIND=""
}

cleanup_gpu_sampler() {
  local sampler_pid="${GPU_SAMPLER_PID:-}"
  [[ -n "${sampler_pid}" ]] || return 0
  if kill -0 "${sampler_pid}" 2>/dev/null; then
    kill "${sampler_pid}" 2>/dev/null || true
  fi
  set +e
  wait "${sampler_pid}" 2>/dev/null
  set -e
  GPU_SAMPLER_PID=""
}

on_exit() {
  local exit_code="$?"
  trap - EXIT
  set +e
  if [[ -n "${CHILD_PID:-}" ]]; then
    terminate_child_group
  fi
  cleanup_gpu_sampler
  if (( exit_code != 0 )) && (( STATUS_FINAL_WRITTEN == 0 )); then
    STATUS_FINAL_WRITTEN=1
    write_status "${CURRENT_STAGE:-UNKNOWN}" FAILED "supervisor_exit exit_code=${exit_code}"
    log "supervisor exit status=${exit_code}"
  fi
  exit "${exit_code}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    fail_stage "${label} must be a regular non-symlink file: ${path}" 1
  fi
  actual="$(sha256sum -- "${path}" | awk '{print $1}')" \
    || fail_stage "unable to hash ${label}: ${path}" 1
  if [[ "${actual}" != "${expected}" ]]; then
    fail_stage "${label} SHA-256 drift: expected=${expected} actual=${actual} path=${path}" 1
  fi
}

read_data_free_bytes() {
  local value
  value="$(df -B1 --output=avail /data | awk 'NR == 2 {print $1}')" || return 1
  [[ "${value}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${value}"
}

gpu_snapshot() {
  local gpu_id line line_count idx mem free util
  SNAPSHOT_TEXT=""
  SNAP_MEM=()
  SNAP_FREE=()
  SNAP_UTIL=()
  for gpu_id in "${GPU_IDS[@]}"; do
    if ! line="$(nvidia-smi --id="${gpu_id}" \
      --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits 2>&1)"; then
      return 1
    fi
    line_count="$(printf '%s\n' "${line}" | awk 'NF {count++} END {print count + 0}')"
    if [[ "${line_count}" != 1 ]]; then
      return 1
    fi
    IFS=',' read -r idx mem free util <<<"${line}"
    idx="$(printf '%s' "${idx}" | tr -d '[:space:]%')"
    mem="$(printf '%s' "${mem}" | tr -d '[:space:]%')"
    free="$(printf '%s' "${free}" | tr -d '[:space:]%')"
    util="$(printf '%s' "${util}" | tr -d '[:space:]%')"
    if [[ "${idx}" != "${gpu_id}" || ! "${mem}" =~ ^[0-9]+$ || \
      ! "${free}" =~ ^[0-9]+$ || ! "${util}" =~ ^[0-9]+$ ]]; then
      return 1
    fi
    SNAP_MEM["${gpu_id}"]="${mem}"
    SNAP_FREE["${gpu_id}"]="${free}"
    SNAP_UTIL["${gpu_id}"]="${util}"
    SNAPSHOT_TEXT+="gpu=${gpu_id},used_mib=${mem},free_mib=${free},utilization_pct=${util};"
  done
}

verify_renderer_checkout() {
  local actual status_output
  [[ -d "${RENDER_REPO}" ]] || fail_stage "renderer checkout missing: ${RENDER_REPO}" 1
  actual="$(git -C "${RENDER_REPO}" rev-parse HEAD 2>/dev/null)" \
    || fail_stage "unable to resolve BrickNet-Render commit" 1
  if [[ "${actual}" != "${EXPECTED_RENDERER_COMMIT}" ]]; then
    fail_stage "BrickNet-Render commit mismatch: expected=${EXPECTED_RENDERER_COMMIT} actual=${actual}" 1
  fi
  status_output="$(git -C "${RENDER_REPO}" status --porcelain=v1 --untracked-files=all)" \
    || fail_stage "unable to inspect BrickNet-Render checkout" 1
  if [[ -n "${status_output}" ]]; then
    fail_stage "BrickNet-Render checkout is dirty: ${status_output//$'\n'/ | }" 1
  fi
}

verify_config_contract() {
  local actual_config_workers gpu_config_raw
  if ! jq -e \
    --arg raw_root "${RAW_ROOT}" \
    --arg collage_root "${BRICKNET_ROOT}/outputs_preprocess/BrickNet-MM/images_v2_official" \
    --arg namespace "${METADATA_NAMESPACE}" \
    --arg raw_name "pt_8view_renders_v2_official_rowids" \
    '(
      .allowed_splits == ["PT"] and
      .render_output_root == "/data/jiahao/task/BrickNet/outputs_gt" and
      .images_v2_root == $collage_root and
      .metadata_namespace == $namespace and
      .render_output_dir_name == $raw_name and
      .engine == "CYCLES" and
      .execution_mode == "cli" and
      .renderer_backend == "bricknet-render-cli" and
      .views == 8 and
      .samples == 256 and
      .random_seed == 0 and
      .resolution == "512x512" and
      .format == "png8" and
      .compute == "OPTIX" and
      .cuda_visible_devices == "0,1" and
      .workers_per_gpu == 8 and
      .renderer_commit == "b49e8733782a5fa7040cc776b5fba8c55b0b894e" and
      .skip_existing == true and
      .overwrite_images == false
    )' "${CONFIG}" >/dev/null; then
    fail_stage "PT config contract failed" 1
  fi
  actual_config_workers="$(jq -er '.workers_per_gpu' "${CONFIG}")" \
    || fail_stage "unable to read PT config workers_per_gpu" 1
  CONFIG_WORKERS_PER_GPU="${actual_config_workers}"
  gpu_config_raw="$(jq -er '.cuda_visible_devices' "${CONFIG}")" \
    || fail_stage "unable to read PT config cuda_visible_devices" 1
  if [[ "${gpu_config_raw}" != "0,1" ]]; then
    fail_stage "PT config must bind physical CUDA 0 and 1: ${gpu_config_raw}" 1
  fi
  GPU_IDS=(0 1)
  GPU_IDS_CSV="0,1"
}

validate_worker_request() {
  if [[ "${REQUESTED_FORMAL_WORKERS}" == "auto" ]]; then
    return 0
  fi
  if [[ "${REQUESTED_FORMAL_WORKERS}" != "${TOPOLOGY_REFREEZE_NEW_WORKERS}" ]]; then
    fail_stage "the frozen fresh/resume topology requires BRICKNET_OFFICIAL_PT_WORKERS_PER_GPU=auto or ${TOPOLOGY_REFREEZE_NEW_WORKERS}; got ${REQUESTED_FORMAL_WORKERS}" 2
  fi
}

validate_refreeze_request() {
  [[ -n "${REFREEZE_WORKERS}" ]] || return 0
  if [[ ! "${REFREEZE_WORKERS}" =~ ^[1-9][0-9]*$ ]] || \
    (( REFREEZE_WORKERS > GPU_MAX_FORMAL_WORKERS_PER_GPU )); then
    fail_stage "BRICKNET_OFFICIAL_PT_REFREEZE_WORKERS must be an integer in 1..${GPU_MAX_FORMAL_WORKERS_PER_GPU}; got ${REFREEZE_WORKERS}" 2
  fi
  if [[ "${WORKERS_OVERRIDE}" == "auto" ]]; then
    fail_stage "BRICKNET_OFFICIAL_PT_REFREEZE_WORKERS requires an exact worker count, not BRICKNET_OFFICIAL_PT_WORKERS_PER_GPU=auto" 2
  fi
  if [[ -n "${WORKERS_OVERRIDE}" && "${WORKERS_OVERRIDE}" != "${REFREEZE_WORKERS}" ]]; then
    fail_stage "refreeze workers=${REFREEZE_WORKERS} conflicts with requested workers=${WORKERS_OVERRIDE}" 2
  fi
  if [[ "${GPU_IDS_CSV}" != "${TOPOLOGY_REFREEZE_NEW_GPU_IDS_CSV}" ||
    "${REFREEZE_WORKERS}" != "${TOPOLOGY_REFREEZE_NEW_WORKERS}" ]]; then
    fail_stage "only the audited topology refreeze ${TOPOLOGY_REFREEZE_OLD_GPU_IDS_CSV}/${TOPOLOGY_REFREEZE_OLD_WORKERS}->${TOPOLOGY_REFREEZE_NEW_GPU_IDS_CSV}/${TOPOLOGY_REFREEZE_NEW_WORKERS} is permitted" 2
  fi
  TOPOLOGY_REFREEZE_REQUESTED=1
}

validate_stop_mode() {
  if [[ "${STOP_BEFORE_FORMAL}" != "0" && "${STOP_BEFORE_FORMAL}" != "1" ]]; then
    fail_stage "BRICKNET_OFFICIAL_PT_STOP_BEFORE_FORMAL must be 0 or 1; got ${STOP_BEFORE_FORMAL}" 2
  fi
}

verify_frozen_inputs() {
  require_sha256 "${CONFIG}" "${EXPECTED_CONFIG_SHA256}" config
  require_sha256 "${OFFICIAL_LAUNCHER}" "${EXPECTED_LAUNCHER_SHA256}" launcher
  require_sha256 "${RENDER_DRIVER}" "${EXPECTED_DRIVER_SHA256}" driver
  require_sha256 "${PT_NPZ}" "${EXPECTED_PT_NPZ_SHA256}" pt_npz
  verify_renderer_checkout
}

initial_gpu_capacity_gate() {
  GPU_PILOT_START_READY=1
  if ! gpu_snapshot; then
    fail_stage "initial GPU query failed; refusing to infer available capacity" 1
  fi
  log "initial GPU snapshot ${SNAPSHOT_TEXT}"
  for gpu_id in "${GPU_IDS[@]}"; do
    if (( SNAP_FREE[${gpu_id}] < GPU_PILOT_START_MIN_FREE_MIB )); then
      GPU_PILOT_START_READY=0
      if (( SELF_TEST == 0 )); then
        fail_stage "GPU ${gpu_id} has only ${SNAP_FREE[${gpu_id}]} MiB free; audited ${GPU_PILOT_WORKERS_PER_GPU}-worker pilot requires at least ${GPU_PILOT_START_MIN_FREE_MIB} MiB before launch" 1
      fi
      log "self-test capacity note gpu=${gpu_id} free_mib=${SNAP_FREE[${gpu_id}]} start_required_mib=${GPU_PILOT_START_MIN_FREE_MIB} pilot_not_ready=1"
    fi
    if (( SNAP_MEM[${gpu_id}] >= GPU_MAX_USED_MIB )); then
      GPU_PILOT_START_READY=0
      if (( SELF_TEST == 0 )); then
        fail_stage "GPU ${gpu_id} already uses ${SNAP_MEM[${gpu_id}]} MiB; refusing pilot at ${GPU_MAX_USED_MIB} MiB ceiling" 1
      fi
      log "self-test capacity note gpu=${gpu_id} used_mib=${SNAP_MEM[${gpu_id}]} ceiling_mib=${GPU_MAX_USED_MIB} pilot_not_ready=1"
    fi
    PILOT_BASELINE_MEM["${gpu_id}"]="${SNAP_MEM[${gpu_id}]}"
  done
}

gpu_sampler_loop() {
  local timestamp gpu_id
  set +e
  while :; do
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if ! gpu_snapshot; then
      printf '%s sampler_query_failed\n' "${timestamp}" >>"${GPU_SAMPLE_ERROR}"
      return 1
    fi
    for gpu_id in "${GPU_IDS[@]}"; do
      printf '%s\t%s\t%s\t%s\t%s\n' "${timestamp}" "${gpu_id}" \
        "${SNAP_MEM[${gpu_id}]}" "${SNAP_FREE[${gpu_id}]}" "${SNAP_UTIL[${gpu_id}]}" \
        >>"${GPU_SAMPLE_LOG}"
      if [[ "${GPU_SAMPLER_MODE}" == "pilot" || "${GPU_SAMPLER_MODE}" == "formal" ]] && \
        { (( SNAP_MEM[${gpu_id}] >= GPU_MAX_USED_MIB )) || \
          (( SNAP_FREE[${gpu_id}] < GPU_PILOT_HARD_MIN_FREE_MIB )); }; then
        printf '%s %s_capacity_breach gpu=%s used_mib=%s free_mib=%s\n' \
          "${timestamp}" "${GPU_SAMPLER_MODE}" "${gpu_id}" \
          "${SNAP_MEM[${gpu_id}]}" "${SNAP_FREE[${gpu_id}]}" \
          >>"${GPU_SAMPLE_ERROR}"
        return 1
      fi
    done
    sleep "${GPU_SAMPLE_INTERVAL_SECONDS}" || return 1
  done
}

start_gpu_sampler() {
  GPU_SAMPLER_MODE="$1"
  if [[ "${GPU_SAMPLER_MODE}" == "pilot" ]]; then
    GPU_SAMPLE_LOG="${PILOT_GPU_SAMPLE_LOG}"
    GPU_SAMPLE_ERROR="${PILOT_GPU_SAMPLE_ERROR}"
    PILOT_EVIDENCE_GPU_LOG="${PILOT_GPU_SAMPLE_LOG}"
  elif [[ "${GPU_SAMPLER_MODE}" == "formal" ]]; then
    GPU_SAMPLE_LOG="${FORMAL_GPU_SAMPLE_LOG}"
    GPU_SAMPLE_ERROR="${FORMAL_GPU_SAMPLE_ERROR}"
  else
    fail_stage "unsupported GPU sampler mode: ${GPU_SAMPLER_MODE}" 2
  fi
  printf 'timestamp_utc\tgpu_id\tmemory_used_mib\tmemory_free_mib\tutilization_gpu_pct\n' >"${GPU_SAMPLE_LOG}"
  : >"${GPU_SAMPLE_ERROR}"
  gpu_sampler_loop &
  GPU_SAMPLER_PID="$!"
  if ! kill -0 "${GPU_SAMPLER_PID}" 2>/dev/null; then
    fail_stage "GPU sampler exited before ${GPU_SAMPLER_MODE} launch" 1
  fi
}

pilot_peak_summary() {
  local gpu_id peak_line sample_count peak_memory peak_util minimum_free
  printf 'gpu_id\tsample_count\tpeak_memory_used_mib\tpeak_utilization_gpu_pct\tminimum_free_memory_mib\n' >"${GPU_PEAK_LOG}"
  for gpu_id in "${GPU_IDS[@]}"; do
    peak_line="$(awk -F '\t' -v target="${gpu_id}" '
      NR == 1 { next }
      $2 == target {
        count += 1
        memory = $3 + 0
        free = $4 + 0
        util = $5 + 0
        if (count == 1 || memory > peak_memory) peak_memory = memory
        if (count == 1 || util > peak_util) peak_util = util
        if (count == 1 || free < minimum_free) minimum_free = free
      }
      END {
        if (count == 0) exit 1
        printf "%d\t%d\t%d\t%d", count, peak_memory, peak_util, minimum_free
      }
    ' "${PILOT_EVIDENCE_GPU_LOG}")" || fail_stage "no valid GPU samples recorded for GPU ${gpu_id}" 1
    IFS=$'\t' read -r sample_count peak_memory peak_util minimum_free <<<"${peak_line}"
    printf '%s\t%s\t%s\t%s\t%s\n' "${gpu_id}" "${sample_count}" \
      "${peak_memory}" "${peak_util}" "${minimum_free}" >>"${GPU_PEAK_LOG}"
    PILOT_PEAK_MEM["${gpu_id}"]="${peak_memory}"
    if (( peak_memory >= GPU_MAX_USED_MIB )); then
      fail_stage "pilot GPU ${gpu_id} peak ${peak_memory} MiB reached ceiling ${GPU_MAX_USED_MIB} MiB" 1
    fi
    if (( minimum_free < GPU_PILOT_MIN_FREE_MIB )); then
      fail_stage "pilot GPU ${gpu_id} minimum free ${minimum_free} MiB is below the required ${GPU_PILOT_MIN_FREE_MIB} MiB (8 GiB) pilot headroom" 1
    fi
  done
}

logical_png_bytes() {
  local root="$1"
  [[ -d "${root}" && ! -L "${root}" ]] || return 1
  find -P "${root}" -type f -name '*.png' -printf '%s\n' | \
    awk '{sum += $1} END {printf "%.0f\n", sum + 0}'
}

complete_production_media_bytes() {
  # Count only production rows that already have eight non-empty raw views and
  # one non-empty collage.  This makes the resume disk gate charge only for
  # media still missing, rather than rejecting the current free space against
  # the full projected tree a second time.
  local raw_root="${RAW_ROOT}"
  local collage_root="${COLLAGE_ROOT}"
  [[ -d "${raw_root}" && ! -L "${raw_root}" ]] || { printf '0\n'; return 0; }
  [[ -d "${collage_root}" && ! -L "${collage_root}" ]] || { printf '0\n'; return 0; }
  {
    find -P "${raw_root}" -mindepth 2 -maxdepth 2 -type f -name '*.png' \
      -printf 'R\t%h\t%s\n'
    find -P "${collage_root}" -mindepth 1 -maxdepth 1 -type f -name '*.png' \
      -printf 'C\t%f\t%s\n'
  } | awk -F '\t' '
    $1 == "R" {
      row = $2
      sub(/^.*\//, "", row)
      raw_count[row] += 1
      raw_bytes[row] += $3 + 0
      if (($3 + 0) > 0) raw_nonempty[row] += 1
      next
    }
    $1 == "C" {
      row = $2
      sub(/\.png$/, "", row)
      collage_bytes[row] = $3 + 0
      next
    }
    END {
      total = 0
      for (row in raw_count) {
        if (raw_count[row] == 8 && raw_nonempty[row] == 8 && collage_bytes[row] > 0)
          total += raw_bytes[row] + collage_bytes[row]
      }
      printf "%.0f\n", total + 0
    }
  '
}

check_pilot_outputs() {
  local pilot_summary pilot_view_root pilot_collage_root view_count collage_count expected actual
  pilot_view_root="${PILOT_RENDER_ROOT}/pt_8view_renders_v2_official_rowids"
  pilot_collage_root="${PILOT_IMAGES_ROOT}/PT"
  pilot_summary="${PILOT_RENDER_ROOT}/${METADATA_NAMESPACE}/pt_summary.json"
  [[ -s "${pilot_summary}" ]] || fail_stage "pilot summary is missing or empty: ${pilot_summary}" 1
  if ! jq -e \
    '(.split == "PT" and .rows_requested == 32 and .stats.failed == 0 and (.stats.rendered + .stats.skipped + .stats.stitched) == 32)' \
    "${pilot_summary}" >/dev/null; then
    fail_stage "pilot summary does not prove 32 successful PT rows: ${pilot_summary}" 1
  fi
  [[ -d "${pilot_view_root}" && -d "${pilot_collage_root}" ]] || \
    fail_stage "pilot output roots are missing" 1
  view_count="$(find -P "${pilot_view_root}" -mindepth 1 -maxdepth 1 -type d -print | wc -l)"
  collage_count="$(find -P "${pilot_collage_root}" -mindepth 1 -maxdepth 1 -type f -name '*.png' -print | wc -l)"
  if [[ "${view_count}" != "${PILOT_PT_ROWS}" || "${collage_count}" != "${PILOT_PT_ROWS}" ]]; then
    fail_stage "pilot output count mismatch: view_dirs=${view_count} collages=${collage_count}" 1
  fi
  expected="$(printf '%s\n' "${PILOT_ROW_IDS[@]}" | sort -n | paste -sd ' ' -)"
  actual="$(find -P "${pilot_view_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | paste -sd ' ' -)"
  if [[ "${actual}" != "${expected}" ]]; then
    fail_stage "pilot row-id output mismatch: expected=${expected} actual=${actual}" 1
  fi
  for row_id in "${PILOT_ROW_IDS[@]}"; do
    local row_dir="${pilot_view_root}/${row_id}"
    local views
    views="$(find -P "${row_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.png' -print | wc -l)"
    [[ "${views}" == 8 ]] || fail_stage "pilot row ${row_id} does not have exactly 8 raw PNGs" 1
  done
  if ! "${PYTHON_BIN}" - "${pilot_view_root}" "${pilot_collage_root}" "${PILOT_ROW_IDS[@]}" <<'PY'
from pathlib import Path
import sys

from PIL import Image

raw_root = Path(sys.argv[1])
collage_root = Path(sys.argv[2])
for raw_row_id in sys.argv[3:]:
    row_id = int(raw_row_id)
    row_dir = raw_root / str(row_id)
    expected = [row_dir / f"{row_id}_{view:04d}.png" for view in range(8)]
    actual = sorted(row_dir.glob("*.png"))
    if actual != expected:
        raise SystemExit(f"pilot row {row_id} raw-view names differ from 0000..0007")
    for path in expected:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"pilot raw view is missing, empty, or symlinked: {path}")
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.size != (512, 512):
                raise SystemExit(f"pilot raw view has invalid format/size: {path}")
    collage = collage_root / f"{row_id}.png"
    if collage.is_symlink() or not collage.is_file() or collage.stat().st_size <= 0:
        raise SystemExit(f"pilot collage is missing, empty, or symlinked: {collage}")
    with Image.open(collage) as image:
        image.load()
        if image.format != "PNG" or image.size != (1024, 512) or image.mode != "RGB":
            raise SystemExit(f"pilot collage has invalid format/size/mode: {collage}")
PY
  then
    fail_stage "pilot image decode/dimension check failed" 1
  fi
}

validate_external_pilot_scope() {
  local pilot_abs repo_abs prefix
  [[ -n "${EXTERNAL_PILOT_ROOT}" ]] || return 0
  [[ -d "${PILOT_ROOT}" && ! -L "${PILOT_ROOT}" ]] || \
    fail_stage "BRICKNET_OFFICIAL_PT_PILOT_ROOT must be an existing real directory: ${PILOT_ROOT}" 1
  pilot_abs="$(realpath -e -- "${PILOT_ROOT}")" || \
    fail_stage "unable to resolve external pilot root: ${PILOT_ROOT}" 1
  repo_abs="$(realpath -e -- "${RENDER_REPO}")" || \
    fail_stage "unable to resolve renderer repository for pilot scope" 1
  prefix="${repo_abs}/tmp/"
  if [[ "${pilot_abs}" != "${prefix}"* ]]; then
    fail_stage "external pilot root escaped BrickNet-Render/tmp: ${pilot_abs}" 1
  fi
  PILOT_ROOT="${pilot_abs}"
  PILOT_RENDER_ROOT="${PILOT_ROOT}/render_output"
  PILOT_IMAGES_ROOT="${PILOT_ROOT}/images_v2"
  PILOT_WORK_ROOT="${PILOT_ROOT}/work"
  PILOT_ID_MANIFEST="${PILOT_ROOT}/row_ids.txt"
  PILOT_EXTERNAL_LOG="${PILOT_ROOT}/pilot.log"
  PILOT_EXTERNAL_GPU_LOG="${PILOT_ROOT}/gpu.tsv"
  log "using existing external pilot root=${PILOT_ROOT}; no new pilot will be launched"
}

load_external_pilot_baselines() {
  local gpu_id baseline
  for gpu_id in "${GPU_IDS[@]}"; do
    baseline="$(awk -F '\t' -v target="${gpu_id}" '
      NR > 1 && $2 == target {print $3; exit}
    ' "${PILOT_EVIDENCE_GPU_LOG}")"
    if [[ ! "${baseline}" =~ ^[0-9]+$ ]]; then
      fail_stage "external pilot GPU log has no baseline sample for GPU ${gpu_id}" 1
    fi
    PILOT_BASELINE_MEM["${gpu_id}"]="${baseline}"
  done
}

reuse_existing_pilot() {
  local identity_path summary_path timing_path expected actual
  local pilot_view_root="${PILOT_RENDER_ROOT}/pt_8view_renders_v2_official_rowids"
  local pilot_collage_root="${PILOT_IMAGES_ROOT}/PT"
  identity_path="${PILOT_RENDER_ROOT}/${METADATA_NAMESPACE}/pt_identity.json"
  summary_path="${PILOT_RENDER_ROOT}/${METADATA_NAMESPACE}/pt_summary.json"
  timing_path="${PILOT_RENDER_ROOT}/${METADATA_NAMESPACE}/pt_timings.jsonl"
  set_stage PILOT_REUSE "validating existing external pilot root=${PILOT_ROOT}"
  [[ -s "${PILOT_EXTERNAL_LOG}" ]] || fail_stage "external pilot log is missing: ${PILOT_EXTERNAL_LOG}" 1
  [[ -s "${PILOT_EXTERNAL_GPU_LOG}" ]] || fail_stage "external pilot GPU log is missing: ${PILOT_EXTERNAL_GPU_LOG}" 1
  if ! grep -Fxq 'PILOT_EXIT=0' "${PILOT_EXTERNAL_LOG}"; then
    fail_stage "external pilot log does not contain the required successful exit marker: ${PILOT_EXTERNAL_LOG}" 1
  fi
  [[ -s "${identity_path}" && -s "${summary_path}" && -s "${timing_path}" ]] || \
    fail_stage "external pilot is not complete yet; pt identity/summary/timings are required before formal rendering" 1
  if grep -Eiq 'out of memory|cuda.*memory|optix.*error|memory allocation|PILOT_EXIT=[1-9]' "${PILOT_EXTERNAL_LOG}"; then
    fail_stage "external pilot log contains OOM/renderer failure evidence; no formal render started" 1
  fi
  if ! jq -e \
    --arg config_sha "${EXPECTED_CONFIG_SHA256}" \
    --arg npz_sha "${EXPECTED_PT_NPZ_SHA256}" \
    --arg namespace "${METADATA_NAMESPACE}" \
    '(
      .split == "PT" and
      .config_sha256 == $config_sha and
      .input_npz_sha256 == $npz_sha and
      .metadata_namespace == $namespace and
      .gpu_ids == [0,1] and
      .render_parameters.gpu_ids == [0,1] and
      .render_parameters.cuda_visible_devices == "0,1" and
      .workers_per_gpu == 9 and
      .render_parameters.workers_per_gpu == 9 and
      .render_parameters.render_output_dir_name == "pt_8view_renders_v2_official_rowids" and
      .renderer_commit == "b49e8733782a5fa7040cc776b5fba8c55b0b894e" and
      .selected_row_count == 135051
    )' "${identity_path}" >/dev/null; then
    fail_stage "external pilot identity does not match PT dual-CUDA config/NPZ/namespace/worker=9 contract" 1
  fi
  if ! jq -e \
    '(
      .split == "PT" and
      .rows_requested == 32 and
      .rows_in_identity == 135051 and
      .stats.failed == 0 and
      (.stats.rendered + .stats.skipped + .stats.stitched) == 32 and
      (.identity_mismatch_fields == [])
    )' "${summary_path}" >/dev/null; then
    fail_stage "external pilot summary does not prove 32 successful PT rows" 1
  fi
  if ! jq -s -e \
    '(
      length == 32 and
      all(.[]; (.status == "rendered" or .status == "skipped" or .status == "stitched"))
    )' "${timing_path}" >/dev/null; then
    fail_stage "external pilot timing JSONL is not exactly 32 successful row records" 1
  fi
  if awk -F '\t' '
    NR > 1 {
      if ($2 != 0 && $2 != 1) bad=1
      seen[$2]=1
    }
    END {exit bad || !seen[0] || !seen[1]}
  ' "${PILOT_EXTERNAL_GPU_LOG}"; then
    :
  else
    fail_stage "external pilot GPU evidence must contain only and both physical CUDA 0 and 1" 1
  fi
  expected="$(printf '%s\n' "${PILOT_ROW_IDS[@]}" | sort -n | paste -sd ' ' -)"
  actual="$(jq -s -r '[.[].row_id] | sort | join(" ")' "${timing_path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    fail_stage "external pilot timing row IDs mismatch: expected=${expected} actual=${actual}" 1
  fi
  # Keep the externally supplied pilot evidence immutable.  Use a run-local
  # copy for peak calculations because start_gpu_sampler() truncates its
  # target when formal monitoring begins.
  cp -- "${PILOT_EXTERNAL_GPU_LOG}" "${PILOT_GPU_SAMPLE_LOG}" \
    || fail_stage "unable to copy external pilot GPU evidence into run-local log" 1
  PILOT_EVIDENCE_GPU_LOG="${PILOT_GPU_SAMPLE_LOG}"
  : >"${PILOT_GPU_SAMPLE_ERROR}"
  load_external_pilot_baselines
  CURRENT_STAGE="PILOT_GPU_CHECK"
  pilot_peak_summary
  CURRENT_STAGE="PILOT_OUTPUT_CHECK"
  check_pilot_outputs
  stage_ok PILOT_OK "reused external pilot rows=${PILOT_PT_ROWS} raw_dirs=${PILOT_PT_ROWS} collages=${PILOT_PT_ROWS} workers_per_gpu=${GPU_PILOT_WORKERS_PER_GPU} gpu_ids=${GPU_IDS_CSV} peak_log=${GPU_PEAK_LOG}"
}

estimate_space_after_pilot() {
  local pilot_raw_root="${PILOT_RENDER_ROOT}/pt_8view_renders_v2_official_rowids"
  local pilot_collage_root="${PILOT_IMAGES_ROOT}/PT"
  local raw_bytes collage_bytes pilot_bytes projected_bytes existing_bytes remaining_bytes required_bytes
  raw_bytes="$(logical_png_bytes "${pilot_raw_root}")" || fail_stage "unable to measure pilot raw logical bytes" 1
  collage_bytes="$(logical_png_bytes "${pilot_collage_root}")" || fail_stage "unable to measure pilot collage logical bytes" 1
  pilot_bytes=$((raw_bytes + collage_bytes))
  projected_bytes=$(( (pilot_bytes * FULL_PT_ROWS + PILOT_PT_ROWS - 1) / PILOT_PT_ROWS ))
  existing_bytes="$(complete_production_media_bytes)" || \
    fail_stage "unable to measure complete production raw+collage bytes" 1
  remaining_bytes=$((projected_bytes - existing_bytes))
  (( remaining_bytes < 0 )) && remaining_bytes=0
  required_bytes=$((remaining_bytes + DATA_RESERVE_BYTES))
  DATA_PILOT_BYTES="${pilot_bytes}"
  DATA_PROJECTED_BYTES="${projected_bytes}"
  DATA_EXISTING_COMPLETE_BYTES="${existing_bytes}"
  DATA_REMAINING_BYTES="${remaining_bytes}"
  DATA_REQUIRED_BYTES="${required_bytes}"
  DATA_FREE_BYTES="$(read_data_free_bytes)" || fail_stage "unable to query /data free space after pilot" 1
  log "pilot_space raw_bytes=${raw_bytes} collage_bytes=${collage_bytes} pilot_media_bytes=${pilot_bytes} projected_full_media_bytes=${projected_bytes} existing_complete_media_bytes=${existing_bytes} remaining_media_bytes=${remaining_bytes} reserve_bytes=${DATA_RESERVE_BYTES} required_bytes=${required_bytes} available_bytes=${DATA_FREE_BYTES}"
  CURRENT_STAGE="PILOT_SPACE_CHECK"
  write_status PILOT_SPACE_CHECK RUNNING "raw_bytes=${raw_bytes} collage_bytes=${collage_bytes} projected_full_media_bytes=${projected_bytes} existing_complete_media_bytes=${existing_bytes} remaining_media_bytes=${remaining_bytes} reserve_bytes=${DATA_RESERVE_BYTES} required_bytes=${required_bytes} available_bytes=${DATA_FREE_BYTES}"
  local minimum_required="${MIN_DATA_START_BYTES}"
  if (( required_bytes > minimum_required )); then
    minimum_required="${required_bytes}"
  fi
  if (( DATA_FREE_BYTES < minimum_required )); then
    fail_stage "post-pilot /data space gate failed: available=${DATA_FREE_BYTES} minimum_required=${minimum_required} projected_required=${required_bytes}" 1
  fi
  stage_ok PILOT_SPACE_OK "available=${DATA_FREE_BYTES} minimum_required=${minimum_required} projected_required=${required_bytes} existing_complete=${existing_bytes} remaining=${remaining_bytes}"
}

choose_formal_workers() {
  local gpu_id delta per_worker current_used current_free budget capacity
  local min_capacity=999999
  local requested_cap production_root production_entry
  local existing_identity="${METADATA_ROOT}/pt_identity.json"
  if ! gpu_snapshot; then
    fail_stage "formal preflight GPU query failed" 1
  fi
  # Recompute capacity from this pilot even on resume.  The persisted identity
  # fixes the worker count, while the fresh pilot proves that the same count
  # still fits beside whatever CUDA jobs are present now.
  for gpu_id in "${GPU_IDS[@]}"; do
    delta=$((PILOT_PEAK_MEM[${gpu_id}] - PILOT_BASELINE_MEM[${gpu_id}]))
    (( delta < 0 )) && delta=0
    per_worker=$(( (delta + GPU_PILOT_WORKERS_PER_GPU - 1) / GPU_PILOT_WORKERS_PER_GPU ))
    (( per_worker < 1 )) && per_worker=1
    PILOT_PER_WORKER_MIB["${gpu_id}"]="${per_worker}"
    current_used="${SNAP_MEM[${gpu_id}]}"
    current_free="${SNAP_FREE[${gpu_id}]}"
    # Capacity is deliberately derived from free memory, reserving 8 GiB for
    # unrelated CUDA work and runtime overhead.  The sampler still enforces
    # the independent hard stop at used>=95000/free<2048 during execution.
    budget=$((current_free - GPU_FORMAL_SAFETY_MIB))
    capacity=$((budget / per_worker))
    if (( capacity < min_capacity )); then
      min_capacity="${capacity}"
    fi
    log "formal_capacity gpu=${gpu_id} baseline_used_mib=${PILOT_BASELINE_MEM[${gpu_id}]} pilot_peak_used_mib=${PILOT_PEAK_MEM[${gpu_id}]} pilot_delta_mib=${delta} estimated_per_worker_increment_mib=${per_worker} current_used_mib=${current_used} current_free_mib=${current_free} budget_mib=${budget} capacity_workers=${capacity}"
  done
  if (( min_capacity < 1 )); then
    fail_stage "formal GPU capacity is below one worker after ${GPU_FORMAL_SAFETY_MIB} MiB safety reserve: capacity=${min_capacity}" 1
  fi
  if [[ -s "${existing_identity}" ]]; then
    local existing_workers existing_gpu_ids identity_sha archived_identity_count=0
    existing_workers="$(jq -er '.workers_per_gpu' "${existing_identity}")" \
      || fail_stage "existing PT identity has no workers_per_gpu: ${existing_identity}" 1
    existing_gpu_ids="$(jq -er '[.gpu_ids[]] | join(",")' "${existing_identity}")" \
      || fail_stage "existing PT identity has no gpu_ids: ${existing_identity}" 1
    if [[ ! "${existing_workers}" =~ ^[1-9][0-9]*$ ]]; then
      fail_stage "existing PT identity workers_per_gpu is invalid: ${existing_workers}" 1
    fi
    IDENTITY_OLD_WORKERS="${existing_workers}"
    if [[ -n "${REFREEZE_WORKERS}" ]]; then
      # Idempotent retry after the driver already completed the one-time
      # adoption is allowed only when its archive proves that transition.
      if [[ "${existing_gpu_ids}" == "${TOPOLOGY_REFREEZE_NEW_GPU_IDS_CSV}" &&
        "${existing_workers}" == "${TOPOLOGY_REFREEZE_NEW_WORKERS}" ]]; then
        if [[ -d "${IDENTITY_ADOPTION_ARCHIVE_ROOT}" ]]; then
          archived_identity_count="$(find -P "${IDENTITY_ADOPTION_ARCHIVE_ROOT}" -maxdepth 1 -type f -name 'pt_identity-*.json' -printf 'x\n' | wc -l)"
        fi
        if (( archived_identity_count == 0 )); then
          fail_stage "topology refreeze appears complete but no adopted old identity archive proves it" 1
        fi
        FORMAL_WORKERS="${existing_workers}"
        IDENTITY_REFREEZE_COMPLETED=1
        TOPOLOGY_REFREEZE_COMPLETED=1
        FORMAL_WORKER_FORMULA="topology_refreeze_already_complete old_gpu_ids=${TOPOLOGY_REFREEZE_OLD_GPU_IDS_CSV} old_workers=${TOPOLOGY_REFREEZE_OLD_WORKERS} new_gpu_ids=${existing_gpu_ids} new_workers=${FORMAL_WORKERS} archived_identity_count=${archived_identity_count}"
        log "topology refreeze already complete at gpu_ids=${existing_gpu_ids} workers=${FORMAL_WORKERS}; archive count=${archived_identity_count}"
      elif [[ "${existing_gpu_ids}" == "${TOPOLOGY_REFREEZE_OLD_GPU_IDS_CSV}" &&
        "${existing_workers}" == "${TOPOLOGY_REFREEZE_OLD_WORKERS}" &&
        "${REFREEZE_WORKERS}" == "${TOPOLOGY_REFREEZE_NEW_WORKERS}" ]]; then
        identity_sha="$(sha256sum -- "${existing_identity}" | awk '{print $1}')" \
          || fail_stage "unable to hash existing PT identity for topology refreeze" 1
        if [[ "${identity_sha}" != "${EXPECTED_OLD_PT_IDENTITY_SHA256}" ]]; then
          fail_stage "topology refreeze old identity SHA-256 mismatch: expected=${EXPECTED_OLD_PT_IDENTITY_SHA256} actual=${identity_sha}" 1
        fi
        FORMAL_WORKERS="${REFREEZE_WORKERS}"
        NEEDS_IDENTITY_ADOPTION=1
        TOPOLOGY_REFREEZE_REQUESTED=1
        if [[ -d "${IDENTITY_ADOPTION_ARCHIVE_ROOT}" ]]; then
          IDENTITY_ADOPTION_ARCHIVE_COUNT_BEFORE="$(find -P "${IDENTITY_ADOPTION_ARCHIVE_ROOT}" -maxdepth 1 -type f -name 'pt_identity-*.json' -printf 'x\n' | wc -l)"
        fi
        if (( FORMAL_WORKERS > min_capacity )); then
          fail_stage "topology refreeze workers=${FORMAL_WORKERS} exceeds current minimum dual-GPU capacity=${min_capacity}; outputs preserved, retry when more memory is free" 1
        fi
        FORMAL_WORKER_FORMULA="topology_refreeze old_gpu_ids=${existing_gpu_ids} old_workers=${existing_workers} new_gpu_ids=${GPU_IDS_CSV} new_workers=${FORMAL_WORKERS} capacity=${min_capacity}; explicit_adopt_existing=1"
        log "topology refreeze selected old_gpu_ids=${existing_gpu_ids} old_workers=${existing_workers} new_gpu_ids=${GPU_IDS_CSV} new_workers=${FORMAL_WORKERS}; driver will archive old identity under ${IDENTITY_ADOPTION_ARCHIVE_ROOT}"
      else
        fail_stage "refreeze request is not the exact audited transition ${TOPOLOGY_REFREEZE_OLD_GPU_IDS_CSV}/${TOPOLOGY_REFREEZE_OLD_WORKERS}->${TOPOLOGY_REFREEZE_NEW_GPU_IDS_CSV}/${TOPOLOGY_REFREEZE_NEW_WORKERS}; existing=${existing_gpu_ids}/${existing_workers} requested=${REFREEZE_WORKERS}" 1
      fi
    else
      if [[ "${existing_gpu_ids}" != "${TOPOLOGY_REFREEZE_NEW_GPU_IDS_CSV}" ||
        "${existing_workers}" != "${TOPOLOGY_REFREEZE_NEW_WORKERS}" ]]; then
        fail_stage "ordinary dual-CUDA resume requires adopted identity gpu_ids=${TOPOLOGY_REFREEZE_NEW_GPU_IDS_CSV} workers=${TOPOLOGY_REFREEZE_NEW_WORKERS}; existing=${existing_gpu_ids}/${existing_workers}; run the one-time topology refreeze first" 1
      fi
      if [[ -n "${WORKERS_OVERRIDE}" && "${WORKERS_OVERRIDE}" != "auto" && "${WORKERS_OVERRIDE}" != "${TOPOLOGY_REFREEZE_NEW_WORKERS}" ]]; then
        fail_stage "resume requires fixed dual-CUDA workers_per_gpu=${TOPOLOGY_REFREEZE_NEW_WORKERS}; requested=${WORKERS_OVERRIDE}" 1
      fi
      FORMAL_WORKERS="${existing_workers}"
      if (( FORMAL_WORKERS > min_capacity )); then
        fail_stage "resume identity requires workers_per_gpu=${FORMAL_WORKERS}, but current GPU capacity permits only ${min_capacity}; outputs preserved, retry when more memory is free" 1
      fi
      FORMAL_WORKER_FORMULA="resume_existing_identity_workers=${existing_workers} capacity=${min_capacity}"
      log "resume identity fixes formal workers_per_gpu=${FORMAL_WORKERS}"
    fi
  else
    if [[ -n "${REFREEZE_WORKERS}" ]]; then
      fail_stage "BRICKNET_OFFICIAL_PT_REFREEZE_WORKERS requires an existing PT identity; refusing to adopt without one" 1
    fi
    # Without an identity this must be a truly fresh production tree.  Missing
    # identity plus residual output is ambiguous after a crash or manual copy;
    # never silently bless it as a new canonical run.
    for production_root in "${RAW_ROOT}" "${COLLAGE_ROOT}" "${METADATA_ROOT}"; do
      if [[ -e "${production_root}" || -L "${production_root}" ]]; then
        [[ -d "${production_root}" && ! -L "${production_root}" ]] || \
          fail_stage "fresh PT root is not a real directory: ${production_root}" 1
        production_entry="$(find -P "${production_root}" -mindepth 1 -maxdepth 1 -print -quit)"
        [[ -z "${production_entry}" ]] || \
          fail_stage "PT identity is missing but production output is not empty: ${production_entry}; inspect/recover explicitly before fresh launch" 1
      fi
    done
    requested_cap="${TOPOLOGY_REFREEZE_NEW_WORKERS}"
    if (( min_capacity < requested_cap )); then
      fail_stage "fresh dual topology is frozen at ${requested_cap} workers/GPU, but current pilot-derived capacity is ${min_capacity}; wait for GPU memory instead of changing the identity" 1
    fi
    FORMAL_WORKERS="${requested_cap}"
    FORMAL_WORKER_FORMULA="fresh_frozen_dual_workers=${FORMAL_WORKERS} capacity=${min_capacity}; pilot_workers=${GPU_PILOT_WORKERS_PER_GPU}; safety_mib=${GPU_FORMAL_SAFETY_MIB}"
  fi
  if (( FORMAL_WORKERS > min_capacity )); then
    fail_stage "selected formal workers=${FORMAL_WORKERS} exceeds current GPU capacity=${min_capacity}; outputs preserved, retry when more memory is free" 1
  fi
  if (( FORMAL_WORKERS < 1 || FORMAL_WORKERS > GPU_MAX_FORMAL_WORKERS_PER_GPU )); then
    fail_stage "selected formal workers out of range: ${FORMAL_WORKERS}" 1
  fi
  write_status FORMAL_WORKERS_SELECTED RUNNING "${FORMAL_WORKER_FORMULA}"
  log "${FORMAL_WORKER_FORMULA} selected=${FORMAL_WORKERS}"
}

formal_gpu_gate() {
  local gpu_id per_worker required_free
  if ! gpu_snapshot; then
    fail_stage "formal preflight GPU query failed" 1
  fi
  for gpu_id in "${GPU_IDS[@]}"; do
    per_worker="${PILOT_PER_WORKER_MIB[${gpu_id}]:-1}"
    required_free=$((FORMAL_WORKERS * per_worker + GPU_FORMAL_SAFETY_MIB))
    if (( required_free < GPU_FORMAL_SAFETY_MIB )); then
      required_free="${GPU_FORMAL_SAFETY_MIB}"
    fi
    if (( SNAP_FREE[${gpu_id}] < required_free )); then
      fail_stage "formal GPU ${gpu_id} free memory ${SNAP_FREE[${gpu_id}]} MiB is below conservative threshold ${required_free} MiB" 1
    fi
    if (( SNAP_MEM[${gpu_id}] + GPU_FORMAL_SAFETY_MIB >= GPU_MAX_USED_MIB )); then
      fail_stage "formal GPU ${gpu_id} current used memory ${SNAP_MEM[${gpu_id}]} MiB leaves less than ${GPU_FORMAL_SAFETY_MIB} MiB below ceiling" 1
    fi
  done
  log "formal GPU gate passed ${SNAPSHOT_TEXT} required_free_formula=formal_workers*estimated_per_worker_increment+${GPU_FORMAL_SAFETY_MIB}MiB"
}

mark_identity_refreeze_complete() {
  local identity_path="${METADATA_ROOT}/pt_identity.json"
  local current_workers current_gpu_ids archive_count
  (( NEEDS_IDENTITY_ADOPTION != 0 )) || return 0
  [[ -s "${identity_path}" ]] || return 1
  current_workers="$(jq -er '.workers_per_gpu' "${identity_path}" 2>/dev/null || true)"
  [[ "${current_workers}" == "${FORMAL_WORKERS}" ]] || return 1
  current_gpu_ids="$(jq -er '[.gpu_ids[]] | join(",")' "${identity_path}" 2>/dev/null || true)"
  [[ "${current_gpu_ids}" == "${TOPOLOGY_REFREEZE_NEW_GPU_IDS_CSV}" ]] || return 1
  archive_count=0
  if [[ -d "${IDENTITY_ADOPTION_ARCHIVE_ROOT}" ]]; then
    archive_count="$(find -P "${IDENTITY_ADOPTION_ARCHIVE_ROOT}" -maxdepth 1 -type f -name 'pt_identity-*.json' -printf 'x\n' | wc -l)"
  fi
  if (( archive_count <= IDENTITY_ADOPTION_ARCHIVE_COUNT_BEFORE )); then
    return 1
  fi
  NEEDS_IDENTITY_ADOPTION=0
  IDENTITY_REFREEZE_COMPLETED=1
  TOPOLOGY_REFREEZE_COMPLETED=1
  log "topology refreeze completed old_gpu_ids=${TOPOLOGY_REFREEZE_OLD_GPU_IDS_CSV} old_workers=${IDENTITY_OLD_WORKERS} new_gpu_ids=${current_gpu_ids} new_workers=${FORMAL_WORKERS}; archived_old_identity_count=${archive_count} archive_root=${IDENTITY_ADOPTION_ARCHIVE_ROOT}; subsequent resume command is fixed dual-CUDA workers=${FORMAL_WORKERS} without refreeze"
  write_status FORMAL_RENDER RUNNING "topology_refreeze_completed old_gpu_ids=${TOPOLOGY_REFREEZE_OLD_GPU_IDS_CSV} old_workers=${IDENTITY_OLD_WORKERS} new_gpu_ids=${current_gpu_ids} new_workers=${FORMAL_WORKERS} archive_root=${IDENTITY_ADOPTION_ARCHIVE_ROOT} subsequent resume uses dual-CUDA workers=${FORMAL_WORKERS} without refreeze"
  return 0
}

run_isolated_pilot() {
  local pilot_rc=0 pilot_interrupted=0
  if (( PILOT_REUSED == 1 )); then
    reuse_existing_pilot
    return 0
  fi
  set_stage PILOT_SETUP "pilot_root=${PILOT_ROOT} rows=${PILOT_PT_ROWS} workers_per_gpu=${GPU_PILOT_WORKERS_PER_GPU} gpu_ids=${GPU_IDS_CSV}"
  if [[ -e "${PILOT_ROOT}" ]]; then
    fail_stage "run-local pilot path unexpectedly exists: ${PILOT_ROOT}" 1
  fi
  mkdir -p -- "${PILOT_RENDER_ROOT}" "${PILOT_IMAGES_ROOT}" "${PILOT_WORK_ROOT}"
  if [[ -L "${PILOT_ROOT}" || -L "${PILOT_RENDER_ROOT}" || -L "${PILOT_IMAGES_ROOT}" || -L "${PILOT_WORK_ROOT}" ]]; then
    fail_stage "pilot output path must not contain a symlink at mutable roots" 1
  fi
  if [[ "${PILOT_ROOT}" != "${RENDER_REPO}/tmp/"* ]]; then
    fail_stage "pilot output escaped BrickNet-Render/tmp: ${PILOT_ROOT}" 1
  fi
  printf '%s\n' "${PILOT_ROW_IDS[@]}" >"${PILOT_ID_MANIFEST}"
  : >"${PILOT_EXTERNAL_LOG}"
  start_gpu_sampler pilot
  set_stage PILOT "PT row-id pilot rows=${PILOT_PT_ROWS} workers_per_gpu=${GPU_PILOT_WORKERS_PER_GPU} output=${PILOT_ROOT}"
  set +e
  env BRICKNET_OFFICIAL_PT_RENDER_CONFIG="${CONFIG}" setsid -- "${OFFICIAL_LAUNCHER}" \
    --pilot-root "${PILOT_ROOT}" \
    --workers-per-gpu "${GPU_PILOT_WORKERS_PER_GPU}" \
    --row-ids "${PILOT_ROW_IDS[@]}" >"${PILOT_EXTERNAL_LOG}" 2>&1 &
  CHILD_PID="$!"
  CHILD_KIND="pilot"
  while kill -0 "${CHILD_PID}" 2>/dev/null; do
    if [[ -s "${GPU_SAMPLE_ERROR}" ]]; then
      log "pilot monitor reported GPU capacity/query failure; stopping only owned pilot process group"
      kill -TERM -- "-${CHILD_PID}" 2>/dev/null || kill -TERM "${CHILD_PID}" 2>/dev/null || true
      pilot_interrupted=1
      break
    fi
    sleep 1
  done
  wait "${CHILD_PID}" 2>/dev/null
  pilot_rc="$?"
  set -e
  CHILD_PID=""
  CHILD_KIND=""
  cleanup_gpu_sampler
  [[ -f "${PILOT_GPU_SAMPLE_LOG}" ]] || \
    fail_stage "fresh pilot GPU sampler evidence is missing: ${PILOT_GPU_SAMPLE_LOG}" 1
  cp -- "${PILOT_GPU_SAMPLE_LOG}" "${PILOT_EXTERNAL_GPU_LOG}" || \
    fail_stage "unable to persist fresh pilot GPU evidence: ${PILOT_EXTERNAL_GPU_LOG}" 1
  printf 'PILOT_EXIT=%s\n' "${pilot_rc}" >>"${PILOT_EXTERNAL_LOG}"
  if (( pilot_interrupted != 0 )); then
    fail_stage "pilot stopped after GPU capacity/query safety failure; no formal render started" 1
  fi
  if (( pilot_rc != 0 )); then
    if grep -Eiq 'out of memory|cuda.*memory|optix.*error|memory allocation' "${GPU_SAMPLE_ERROR}" "${PILOT_EXTERNAL_LOG}" "${SUPERVISOR_LOG}" 2>/dev/null; then
      fail_stage "PT pilot failed with likely GPU capacity/OOM evidence; no formal render started" "${pilot_rc}"
    fi
    fail_stage "PT pilot failed with exit=${pilot_rc}; no formal render started" "${pilot_rc}"
  fi
  if [[ -s "${GPU_SAMPLE_ERROR}" ]]; then
    fail_stage "PT pilot GPU sampler reported an error; no formal render started" 1
  fi
  stage_ok PILOT_RENDER_OK "pilot_root=${PILOT_ROOT} workers_per_gpu=${GPU_PILOT_WORKERS_PER_GPU}"
  CURRENT_STAGE="PILOT_GPU_CHECK"
  pilot_peak_summary
  CURRENT_STAGE="PILOT_OUTPUT_CHECK"
  check_pilot_outputs
  stage_ok PILOT_OK "rows=${PILOT_PT_ROWS} raw_dirs=${PILOT_PT_ROWS} collages=${PILOT_PT_ROWS} peak_log=${GPU_PEAK_LOG}"
}

run_formal_render() {
  local formal_rc=0 current_free watchdog_seconds=0 adoption_wait=0
  local launch_args=(--workers-per-gpu "${FORMAL_WORKERS}")
  formal_gpu_gate
  set_stage FORMAL_PREFLIGHT "launcher=${OFFICIAL_LAUNCHER} workers_per_gpu=${FORMAL_WORKERS}"
  if env BRICKNET_OFFICIAL_PT_RENDER_CONFIG="${CONFIG}" "${OFFICIAL_LAUNCHER}" --preflight-only --workers-per-gpu "${FORMAL_WORKERS}"; then
    stage_ok FORMAL_PREFLIGHT_OK "workers_per_gpu=${FORMAL_WORKERS}"
  else
    formal_rc="$?"
    fail_stage "formal PT preflight failed" "${formal_rc}"
  fi
  DATA_FREE_BYTES="$(read_data_free_bytes)" || fail_stage "unable to query /data before formal render" 1
  local formal_min_data_bytes="${MIN_DATA_START_BYTES}"
  if [[ "${DATA_REQUIRED_BYTES}" =~ ^[0-9]+$ ]] && (( DATA_REQUIRED_BYTES > formal_min_data_bytes )); then
    formal_min_data_bytes="${DATA_REQUIRED_BYTES}"
  fi
  if (( DATA_FREE_BYTES < formal_min_data_bytes )); then
    fail_stage "formal /data start gate failed: available=${DATA_FREE_BYTES} minimum_required=${formal_min_data_bytes} projected_required=${DATA_REQUIRED_BYTES}" 1
  fi
  start_gpu_sampler formal
  if (( NEEDS_IDENTITY_ADOPTION != 0 )); then
    launch_args+=(--adopt-existing)
    set_stage FORMAL_RENDER "launcher=${OFFICIAL_LAUNCHER} workers_per_gpu=${FORMAL_WORKERS} identity_refreeze=${IDENTITY_OLD_WORKERS}->${FORMAL_WORKERS} adopt_existing=1 archive_root=${IDENTITY_ADOPTION_ARCHIVE_ROOT} gpu_ids=${GPU_IDS_CSV} raw=${RAW_ROOT} collage=${COLLAGE_ROOT}"
  else
    set_stage FORMAL_RENDER "launcher=${OFFICIAL_LAUNCHER} workers_per_gpu=${FORMAL_WORKERS} gpu_ids=${GPU_IDS_CSV} raw=${RAW_ROOT} collage=${COLLAGE_ROOT}"
  fi
  set +e
  if (( NEEDS_IDENTITY_ADOPTION != 0 )); then
    env \
      BRICKNET_OFFICIAL_PT_ALLOW_ADOPT_EXISTING=1 \
      BRICKNET_OFFICIAL_PT_SUPERVISOR_ADOPTION_CONTRACT=1 \
      BRICKNET_OFFICIAL_PT_SUPERVISOR_EXPECTED_OLD_IDENTITY_SHA256="${EXPECTED_OLD_PT_IDENTITY_SHA256}" \
      BRICKNET_OFFICIAL_PT_SUPERVISOR_ALLOWED_IDENTITY_MISMATCH_FIELDS="${ALLOWED_IDENTITY_MISMATCH_FIELDS}" \
      env BRICKNET_OFFICIAL_PT_RENDER_CONFIG="${CONFIG}" setsid -- "${OFFICIAL_LAUNCHER}" "${launch_args[@]}" &
  else
    env BRICKNET_OFFICIAL_PT_RENDER_CONFIG="${CONFIG}" setsid -- "${OFFICIAL_LAUNCHER}" "${launch_args[@]}" &
  fi
  CHILD_PID="$!"
  CHILD_KIND="formal"
  if (( NEEDS_IDENTITY_ADOPTION != 0 )); then
    # The driver archives the old single-GPU/9 identity and atomically writes
    # the dual-GPU/9 identity before scheduling rows.  Confirm that transition
    # at startup so any later recoverable status can use the ordinary dual-GPU/9
    # resume command rather than attempting a second topology adoption.
    adoption_wait=0
    while (( adoption_wait < 60 )) && kill -0 "${CHILD_PID}" 2>/dev/null; do
      if mark_identity_refreeze_complete; then
        break
      fi
      sleep 1
      adoption_wait=$((adoption_wait + 1))
    done
    if (( NEEDS_IDENTITY_ADOPTION != 0 )); then
      log "identity refreeze not yet observable after ${adoption_wait}s; retaining explicit adoption state while monitoring owned formal process"
    fi
  fi
  while kill -0 "${CHILD_PID}" 2>/dev/null; do
    sleep 1
    if (( NEEDS_IDENTITY_ADOPTION != 0 )); then
      mark_identity_refreeze_complete || true
    fi
    if ! kill -0 "${CHILD_PID}" 2>/dev/null; then
      break
    fi
    if [[ -s "${GPU_SAMPLE_ERROR}" ]]; then
      log "formal GPU monitor reported a capacity/query failure; stopping only the owned formal process group and preserving resumable outputs"
      kill -TERM -- "-${CHILD_PID}" 2>/dev/null || kill -TERM "${CHILD_PID}" 2>/dev/null || true
      wait "${CHILD_PID}" 2>/dev/null
      formal_rc="$?"
      set -e
      CHILD_PID=""
      CHILD_KIND=""
      cleanup_gpu_sampler
      CURRENT_STAGE="RECOVERABLE_GPU_PRESSURE"
      STATUS_FINAL_WRITTEN=1
      if (( NEEDS_IDENTITY_ADOPTION != 0 )); then
        write_status RECOVERABLE_GPU_PRESSURE FAILED "GPU capacity/query guard stopped the formal process; outputs preserved; identity refreeze old_workers=${IDENTITY_OLD_WORKERS} new_workers=${FORMAL_WORKERS} was not completed; archive_root=${IDENTITY_ADOPTION_ARCHIVE_ROOT}; retry with explicit refreeze workers=${FORMAL_WORKERS}"
        log "stage=RECOVERABLE_GPU_PRESSURE result=FAILED formal_rc=${formal_rc} identity_refreeze=${IDENTITY_OLD_WORKERS}->${FORMAL_WORKERS} archive_root=${IDENTITY_ADOPTION_ARCHIVE_ROOT} evidence=${GPU_SAMPLE_ERROR}"
      else
        write_status RECOVERABLE_GPU_PRESSURE FAILED "GPU capacity/query guard stopped the formal process; outputs preserved; retry the same workers=${FORMAL_WORKERS} when sufficient GPU memory is available"
        log "stage=RECOVERABLE_GPU_PRESSURE result=FAILED formal_rc=${formal_rc} resume_workers=${FORMAL_WORKERS} evidence=${GPU_SAMPLE_ERROR}"
      fi
      exit 76
    fi
    watchdog_seconds=$((watchdog_seconds + 1))
    if (( watchdog_seconds < 60 )); then
      continue
    fi
    watchdog_seconds=0
    current_free="$(read_data_free_bytes 2>/dev/null || true)"
    if [[ ! "${current_free}" =~ ^[0-9]+$ ]]; then
      log "formal disk watchdog could not read /data free bytes; stopping owned formal process group safely"
      kill -TERM -- "-${CHILD_PID}" 2>/dev/null || kill -TERM "${CHILD_PID}" 2>/dev/null || true
      wait "${CHILD_PID}" 2>/dev/null
      formal_rc="$?"
      set -e
      CHILD_PID=""
      CHILD_KIND=""
      cleanup_gpu_sampler
      CURRENT_STAGE="RECOVERABLE_DISK_LOW"
      STATUS_FINAL_WRITTEN=1
      write_status RECOVERABLE_DISK_LOW FAILED "disk watchdog query failed; formal process terminated; outputs preserved; rerun resume command with the same workers=${FORMAL_WORKERS}"
      log "stage=RECOVERABLE_DISK_LOW result=FAILED disk watchdog query failed formal_rc=${formal_rc}"
      exit 74
    fi
    DATA_FREE_BYTES="${current_free}"
    write_status FORMAL_RENDER RUNNING "disk_watchdog_available_bytes=${DATA_FREE_BYTES} low_watermark_bytes=${DISK_LOW_WATERMARK_BYTES} formal_pid=${CHILD_PID}"
    log "formal disk watchdog available_bytes=${DATA_FREE_BYTES} low_watermark_bytes=${DISK_LOW_WATERMARK_BYTES}"
    if (( DATA_FREE_BYTES < DISK_LOW_WATERMARK_BYTES )); then
      log "formal disk low watermark reached; TERM only owned formal process group; preserving resumable outputs"
      kill -TERM -- "-${CHILD_PID}" 2>/dev/null || kill -TERM "${CHILD_PID}" 2>/dev/null || true
      wait "${CHILD_PID}" 2>/dev/null
      formal_rc="$?"
      set -e
      CHILD_PID=""
      CHILD_KIND=""
      cleanup_gpu_sampler
      CURRENT_STAGE="RECOVERABLE_DISK_LOW"
      STATUS_FINAL_WRITTEN=1
      write_status RECOVERABLE_DISK_LOW FAILED "available_bytes=${DATA_FREE_BYTES} low_watermark_bytes=${DISK_LOW_WATERMARK_BYTES} formal process terminated; outputs preserved; complete raw+collage rows resume via skip_existing and partial rows are atomically retried"
      log "stage=RECOVERABLE_DISK_LOW result=FAILED formal_rc=${formal_rc} resume_workers=${FORMAL_WORKERS}"
      exit 74
    fi
  done
  wait "${CHILD_PID}" 2>/dev/null
  formal_rc="$?"
  set -e
  CHILD_PID=""
  CHILD_KIND=""
  cleanup_gpu_sampler
  if (( formal_rc != 0 )); then
    if grep -Eiq 'out of memory|cuda.*memory|optix.*error|memory allocation' "${SUPERVISOR_LOG}" 2>/dev/null; then
      fail_stage "formal PT render failed with likely GPU capacity/OOM evidence; outputs preserved for resume" "${formal_rc}"
    fi
    fail_stage "formal PT render failed with exit=${formal_rc}; outputs preserved for resume" "${formal_rc}"
  fi
  DATA_FREE_BYTES="$(read_data_free_bytes 2>/dev/null || printf '%s' unknown)"
  STATUS_FINAL_WRITTEN=1
  if (( NEEDS_IDENTITY_ADOPTION != 0 )); then
    write_status RENDER_COMPLETE_PENDING_VERIFY OK "formal launcher exited successfully; identity refreeze old_workers=${IDENTITY_OLD_WORKERS} new_workers=${FORMAL_WORKERS}; adopted identity archive_root=${IDENTITY_ADOPTION_ARCHIVE_ROOT}; strict --verify-only and PT completion gate remain required; outputs=${RAW_ROOT},${COLLAGE_ROOT}"
    log "stage=RENDER_COMPLETE_PENDING_VERIFY result=OK formal render exited; identity_refreeze=${IDENTITY_OLD_WORKERS}->${FORMAL_WORKERS} archive_root=${IDENTITY_ADOPTION_ARCHIVE_ROOT}; strict verify and gate are still pending"
  else
    write_status RENDER_COMPLETE_PENDING_VERIFY OK "formal launcher exited successfully; strict --verify-only and PT completion gate remain required; outputs=${RAW_ROOT},${COLLAGE_ROOT}"
    log "stage=RENDER_COMPLETE_PENDING_VERIFY result=OK formal render exited; strict verify and gate are still pending"
  fi
  exit 0
}

set_stage INIT "run_id=${RUN_ID} self_test=${SELF_TEST}"
set_stage PREREQUISITES "checking PT launcher/config/driver/NPZ hashes, renderer commit, config contract, and tools"

[[ -x "${OFFICIAL_LAUNCHER}" ]] || fail_stage "official PT launcher is not executable: ${OFFICIAL_LAUNCHER}" 1
[[ -d "${BRICKNET_ROOT}" && -d "${RENDER_REPO}" ]] || fail_stage "required repository directory is missing" 1
[[ -x "${PYTHON_BIN}" ]] || fail_stage "BrickNet Python is not executable: ${PYTHON_BIN}" 1
command -v nvidia-smi >/dev/null 2>&1 || fail_stage "nvidia-smi is unavailable" 1
command -v jq >/dev/null 2>&1 || fail_stage "jq is unavailable" 1
command -v sha256sum >/dev/null 2>&1 || fail_stage "sha256sum is unavailable" 1
command -v flock >/dev/null 2>&1 || fail_stage "flock is unavailable" 1
command -v setsid >/dev/null 2>&1 || fail_stage "setsid is unavailable" 1
[[ -f "${CONFIG}" && ! -L "${CONFIG}" ]] || fail_stage "PT config must be a regular non-symlink file: ${CONFIG}" 1
verify_config_contract
validate_worker_request
validate_refreeze_request
validate_stop_mode
verify_frozen_inputs
validate_external_pilot_scope

DATA_FREE_BYTES="$(read_data_free_bytes)" || fail_stage "unable to query /data available bytes" 1
if (( DATA_FREE_BYTES < MIN_DATA_START_BYTES )); then
  fail_stage "initial /data space gate failed: available=${DATA_FREE_BYTES} minimum=${MIN_DATA_START_BYTES}" 1
fi
initial_gpu_capacity_gate
stage_ok PREREQUISITES "config=${CONFIG} config_workers=${CONFIG_WORKERS_PER_GPU} pilot_workers=${GPU_PILOT_WORKERS_PER_GPU} requested_formal_workers=${REQUESTED_FORMAL_WORKERS} gpu_ids=${GPU_IDS_CSV} available_bytes=${DATA_FREE_BYTES}"

set_stage PREFLIGHT "official PT preflight with dual CUDA 0,1 workers_per_gpu=${GPU_PILOT_WORKERS_PER_GPU}"
if env BRICKNET_OFFICIAL_PT_RENDER_CONFIG="${CONFIG}" "${OFFICIAL_LAUNCHER}" --preflight-only --workers-per-gpu "${GPU_PILOT_WORKERS_PER_GPU}"; then
  stage_ok PREFLIGHT_OK "official PT preflight passed"
else
  preflight_rc="$?"
  fail_stage "official PT preflight failed" "${preflight_rc}"
fi

if (( SELF_TEST == 1 )); then
  STATUS_FINAL_WRITTEN=1
  write_status SELF_TEST_PASS OK "read-only self-test passed; gpu_pilot_start_ready=${GPU_PILOT_START_READY}; pilot and formal render were not started"
  log "stage=SELF_TEST_PASS result=OK gpu_pilot_start_ready=${GPU_PILOT_START_READY} pilot/full render not started"
  exit 0
fi

run_isolated_pilot
estimate_space_after_pilot
choose_formal_workers
if [[ "${STOP_BEFORE_FORMAL}" == "1" ]]; then
  STATUS_FINAL_WRITTEN=1
  write_status PILOT_VALIDATED_NO_FORMAL OK "pilot validated and space/GPU/formal-worker gates passed; formal render intentionally not started"
  log "stage=PILOT_VALIDATED_NO_FORMAL result=OK formal render intentionally not started"
  exit 0
fi
run_formal_render
