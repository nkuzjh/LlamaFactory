#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot, fail-closed supervisor for the official BrickNet SFT render.
#
# The supervisor deliberately does not accept positional arguments.  The
# official launcher itself is the only render entry point used here; it fixes
# the split to SFT.  The pilot gets a unique, run-local output tree below
# BrickNet-Render/tmp, while the final exec uses the official config's formal
# output roots.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BRICKNET_ROOT="/data/jiahao/task/BrickNet"
RENDER_REPO="/data/jiahao/task/BrickNet-Render"
OFFICIAL_LAUNCHER="${BRICKNET_ROOT}/scripts/render_bricknet_render_8views_official.sh"
DEFAULT_CONFIG="${BRICKNET_ROOT}/configs/bricknet_mm_image_v2_official_render.json"
RENDER_DRIVER="${BRICKNET_ROOT}/scripts/render_bricknet_render_8views.py"
SFT_NPZ="${BRICKNET_ROOT}/data/bricknet_datasets/sft.npz"
EXPECTED_CONFIG_SHA256="0f19f471e945f76b25cb7ba4cf85358a0d7be49dd52d2c0c02faacb214777be9"
EXPECTED_LAUNCHER_SHA256="f8bc3cb3be0e9250144e20fb750d5728a7517e2d59d8451739c0343cc8c3ca47"
EXPECTED_DRIVER_SHA256="74cd93f2d2bc418e583f8b1821d32066020b13036337819adebcd526756789fc"
EXPECTED_SFT_NPZ_SHA256="fa8f22f73a04e8abd0bb388c4c49df764a1f54b23ad6e0dc464ecea47ef350d8"
TMP_BASH="${SCRIPT_DIR}"
LOCK_FILE="${TMP_BASH}/supervise_official_sft_render.lock"
STATUS_FILE="${TMP_BASH}/supervise_official_sft_render.status"
MIN_DATA_FREE_BYTES=220000000000
GPU_MEMORY_LIMIT_MIB=2048
GPU_UTIL_LIMIT_PCT=5
GPU_IDLE_SAMPLES=3
GPU_IDLE_INTERVAL_SECONDS=10
GPU_SAMPLE_INTERVAL_SECONDS=1
GPU_FORMAL_MAX_PEAK_MIB=95000

PILOT_ROW_IDS=(
  53244 53273 53835 54368 55407 55947 56063 56509
  56547 57047 57610 57722 57848 58455 58993 59246
  59475 59553 60035 60847 61621 62578 62863 63467
  63776 64056 64211 65345 66142 66395 66632 66731
)

if (( $# != 0 )); then
  printf 'usage: %s\n' "$0" >&2
  exit 2
fi

mkdir -p -- "${TMP_BASH}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] another official SFT supervisor already holds %s\n' \
    "$(date --iso-8601=seconds)" "${LOCK_FILE}" >&2
  exit 75
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SUPERVISOR_LOG="${TMP_BASH}/supervise_official_sft_render-${RUN_ID}.log"
GPU_SAMPLE_LOG="${TMP_BASH}/supervise_official_sft_render-${RUN_ID}.gpu.tsv"
GPU_PEAK_LOG="${TMP_BASH}/supervise_official_sft_render-${RUN_ID}.gpu-peaks.tsv"
GPU_SAMPLE_ERROR="${TMP_BASH}/supervise_official_sft_render-${RUN_ID}.gpu.error"
PILOT_ROOT="${RENDER_REPO}/tmp/official_sft_pilot_${RUN_ID}"
PILOT_RENDER_ROOT="${PILOT_ROOT}/render_output"
PILOT_IMAGES_ROOT="${PILOT_ROOT}/images_v2"
PILOT_WORK_ROOT="${PILOT_ROOT}/work"
PILOT_ID_MANIFEST="${PILOT_ROOT}/row_ids.txt"

exec >>"${SUPERVISOR_LOG}" 2>&1

STATUS_READY=1
CURRENT_STAGE="INIT"
GPU_SAMPLER_PID=""
GPU_IDS=()
GPU_IDS_CSV=""
declare -A SNAP_MEM=()
declare -A SNAP_UTIL=()
SNAPSHOT_TEXT=""
GPU_ALL_IDLE=0

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

write_status() {
  local stage="$1"
  local result="$2"
  local detail="${3:-}"
  local now
  local tmp_path="${STATUS_FILE}.tmp.$$"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  detail="${detail//$'\n'/ | }"
  {
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'pid=%s\n' "$$"
    printf 'stage=%s\n' "${stage}"
    printf 'result=%s\n' "${result}"
    printf 'updated_utc=%s\n' "${now}"
    printf 'config=%s\n' "${CONFIG:-unknown}"
    printf 'gpu_ids=%s\n' "${GPU_IDS_CSV:-unknown}"
    printf 'config_workers_per_gpu=%s\n' "${CONFIG_WORKERS_PER_GPU:-unknown}"
    printf 'workers_per_gpu=%s\n' "${WORKERS_PER_GPU:-unknown}"
    printf 'pilot_root=%s\n' "${PILOT_ROOT:-unknown}"
    printf 'supervisor_log=%s\n' "${SUPERVISOR_LOG:-unknown}"
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
  if [[ "${STATUS_READY:-0}" == 1 ]]; then
    write_status "${CURRENT_STAGE}" FAILED "${message} exit_code=${exit_code}"
  fi
  log "stage=${CURRENT_STAGE} result=FAILED ${message} exit_code=${exit_code}"
  exit "${exit_code}"
}

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

verify_frozen_inputs() {
  require_sha256 "${CONFIG}" "${EXPECTED_CONFIG_SHA256}" config
  require_sha256 "${OFFICIAL_LAUNCHER}" "${EXPECTED_LAUNCHER_SHA256}" launcher
  require_sha256 "${RENDER_DRIVER}" "${EXPECTED_DRIVER_SHA256}" driver
  require_sha256 "${SFT_NPZ}" "${EXPECTED_SFT_NPZ_SHA256}" sft_npz
}

cleanup_gpu_sampler() {
  local sampler_pid="${GPU_SAMPLER_PID:-}"
  [[ -n "${sampler_pid}" ]] || return 0
  # This PID is assigned only to the sampler process started below.  No GPU
  # occupant or renderer PID is ever discovered, signalled, or reaped here.
  if kill -0 "${sampler_pid}" 2>/dev/null; then
    kill "${sampler_pid}" 2>/dev/null || true
  fi
  wait "${sampler_pid}" 2>/dev/null || true
  GPU_SAMPLER_PID=""
}

on_exit() {
  local exit_code="$?"
  trap - EXIT
  set +e
  cleanup_gpu_sampler
  if (( exit_code != 0 )) && [[ "${STATUS_READY:-0}" == 1 ]]; then
    write_status "${CURRENT_STAGE:-UNKNOWN}" FAILED "supervisor_exit exit_code=${exit_code}"
    log "supervisor exit status=${exit_code}"
  fi
  exit "${exit_code}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set_stage INIT "run_id=${RUN_ID}"
set_stage PREREQUISITES "checking launcher, config, tools, GPU selection, and /data free space"

if [[ ! -x "${OFFICIAL_LAUNCHER}" ]]; then
  fail_stage "official launcher is not executable: ${OFFICIAL_LAUNCHER}" 1
fi
if [[ ! -d "${BRICKNET_ROOT}" || ! -d "${RENDER_REPO}" ]]; then
  fail_stage "required repository directory is missing" 1
fi

PYTHON_BIN="${BRICKNET_PYTHON:-/home/jiahao/miniconda3/envs/bricknet/bin/python}"
export BRICKNET_PYTHON="${PYTHON_BIN}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  fail_stage "BrickNet Python is not executable: ${PYTHON_BIN}" 1
fi

CONFIG="${BRICKNET_OFFICIAL_RENDER_CONFIG:-${DEFAULT_CONFIG}}"
export BRICKNET_OFFICIAL_RENDER_CONFIG="${CONFIG}"
if [[ ! -f "${CONFIG}" || -L "${CONFIG}" ]]; then
  fail_stage "official render config must be a regular file: ${CONFIG}" 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail_stage "nvidia-smi is unavailable" 1
fi
if ! command -v jq >/dev/null 2>&1; then
  fail_stage "jq is unavailable for config audit" 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  fail_stage "sha256sum is unavailable for frozen-input validation" 1
fi

WORKERS_PER_GPU="${BRICKNET_OFFICIAL_WORKERS_PER_GPU:-16}"
if [[ ! "${WORKERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  fail_stage "BRICKNET_OFFICIAL_WORKERS_PER_GPU must be a positive integer; got ${WORKERS_PER_GPU}" 2
fi
CONFIG_WORKERS_PER_GPU="$(jq -er 'if has("workers_per_gpu") then (.workers_per_gpu | tostring) else "<default>" end' "${CONFIG}")" \
  || fail_stage "unable to read workers_per_gpu from config" 1

GPU_CONFIG_RAW="$(jq -er 'if has("cuda_visible_devices") then (.cuda_visible_devices | tostring) else empty end' "${CONFIG}")" \
  || fail_stage "config must explicitly set cuda_visible_devices" 1
GPU_CONFIG_SPACED="${GPU_CONFIG_RAW//,/ }"
read -r -a GPU_IDS <<<"${GPU_CONFIG_SPACED}"
if (( ${#GPU_IDS[@]} != 2 )); then
  fail_stage "config must select exactly two physical GPUs; got ${GPU_CONFIG_RAW}" 1
fi
for gpu_id in "${GPU_IDS[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    fail_stage "config cuda_visible_devices contains a non-numeric GPU ID: ${GPU_CONFIG_RAW}" 1
  fi
done
if [[ "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
  fail_stage "config selects the same physical GPU twice: ${GPU_CONFIG_RAW}" 1
fi
GPU_IDS_CSV="$(IFS=,; printf '%s' "${GPU_IDS[*]}")"

# SFT is the only permitted split in this supervisor's config contract.  The
# launcher also fixes SFT in its argv, but rejecting a broader config keeps a
# changed config from silently widening the run's scope.
if ! jq -e '(.allowed_splits | type == "array" and length == 1 and .[0] == "SFT")' "${CONFIG}" >/dev/null; then
  fail_stage "config allowed_splits must be exactly [SFT]" 1
fi
verify_frozen_inputs

if ! DATA_FREE_BYTES="$(df -B1 --output=avail /data | awk 'NR == 2 {print $1}')"; then
  fail_stage "unable to query free bytes on /data" 1
fi
if [[ ! "${DATA_FREE_BYTES}" =~ ^[0-9]+$ ]]; then
  fail_stage "df returned a non-numeric /data available-byte count: ${DATA_FREE_BYTES:-empty}" 1
fi
if (( DATA_FREE_BYTES < MIN_DATA_FREE_BYTES )); then
  fail_stage "/data free-byte gate failed: available=${DATA_FREE_BYTES} minimum=${MIN_DATA_FREE_BYTES}" 1
fi

stage_ok PREREQUISITES "config=${CONFIG} config_gpu_ids=${GPU_IDS_CSV} config_workers_per_gpu=${CONFIG_WORKERS_PER_GPU} workers_per_gpu=${WORKERS_PER_GPU} data_available_bytes=${DATA_FREE_BYTES}"

gpu_snapshot() {
  local gpu_id line line_count idx mem util
  local all_idle=1
  SNAPSHOT_TEXT=""
  SNAP_MEM=()
  SNAP_UTIL=()

  for gpu_id in "${GPU_IDS[@]}"; do
    if ! line="$(nvidia-smi --id="${gpu_id}" \
      --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1)"; then
      return 1
    fi
    if [[ "${line}" == *$'\n'* || -z "${line//[[:space:]]/}" ]]; then
      return 1
    fi
    line_count="$(printf '%s\n' "${line}" | awk 'NF {count++} END {print count + 0}')"
    if [[ "${line_count}" != 1 ]]; then
      return 1
    fi
    IFS=',' read -r idx mem util <<<"${line}"
    idx="$(printf '%s' "${idx}" | tr -d '[:space:]%')"
    mem="$(printf '%s' "${mem}" | tr -d '[:space:]%')"
    util="$(printf '%s' "${util}" | tr -d '[:space:]%')"
    if [[ "${idx}" != "${gpu_id}" || ! "${mem}" =~ ^[0-9]+$ || ! "${util}" =~ ^[0-9]+$ ]]; then
      return 1
    fi
    SNAP_MEM["${gpu_id}"]="${mem}"
    SNAP_UTIL["${gpu_id}"]="${util}"
    SNAPSHOT_TEXT+="gpu=${gpu_id},memory_used_mib=${mem},utilization_pct=${util};"
    if (( mem >= GPU_MEMORY_LIMIT_MIB || util > GPU_UTIL_LIMIT_PCT )); then
      all_idle=0
    fi
  done
  GPU_ALL_IDLE="${all_idle}"
}

wait_for_idle() {
  local consecutive=0 sample=0 timestamp
  set_stage GPU_WAIT "need=${GPU_IDLE_SAMPLES} consecutive samples interval_seconds=${GPU_IDLE_INTERVAL_SECONDS} memory_lt_mib=${GPU_MEMORY_LIMIT_MIB} util_lte_pct=${GPU_UTIL_LIMIT_PCT} gpu_ids=${GPU_IDS_CSV}"
  while (( consecutive < GPU_IDLE_SAMPLES )); do
    sample=$((sample + 1))
    timestamp="$(date --iso-8601=seconds)"
    if ! gpu_snapshot; then
      fail_stage "GPU query failed; refusing to infer idleness" 1
    fi
    if (( GPU_ALL_IDLE == 1 )); then
      consecutive=$((consecutive + 1))
    else
      consecutive=0
    fi
    log "gpu_wait sample=${sample} consecutive=${consecutive}/${GPU_IDLE_SAMPLES} ${SNAPSHOT_TEXT}"
    write_status GPU_WAIT RUNNING "sample=${sample} consecutive=${consecutive}/${GPU_IDLE_SAMPLES} ${SNAPSHOT_TEXT}"
    if (( consecutive < GPU_IDLE_SAMPLES )); then
      sleep "${GPU_IDLE_INTERVAL_SECONDS}"
    fi
  done
  stage_ok GPU_IDLE_CONFIRMED "samples=${GPU_IDLE_SAMPLES} interval_seconds=${GPU_IDLE_INTERVAL_SECONDS} ${SNAPSHOT_TEXT}"
}

wait_for_idle

verify_frozen_inputs
set_stage PREFLIGHT "official SFT preflight launcher=${OFFICIAL_LAUNCHER} workers_per_gpu=${WORKERS_PER_GPU}"
if "${OFFICIAL_LAUNCHER}" --preflight-only --workers-per-gpu "${WORKERS_PER_GPU}"; then
  stage_ok PREFLIGHT_OK "official SFT preflight passed"
else
  preflight_rc="$?"
  fail_stage "official SFT preflight failed" "${preflight_rc}"
fi

# Preflight is CPU-side and leaves a race window in which another job can
# acquire a card.  Re-establish the same three-sample gate before pilot.
wait_for_idle

verify_frozen_inputs
set_stage PILOT_SETUP "pilot_root=${PILOT_ROOT} row_count=${#PILOT_ROW_IDS[@]} workers_per_gpu=${WORKERS_PER_GPU} gpu_ids=${GPU_IDS_CSV}"
if [[ -e "${PILOT_ROOT}" ]]; then
  fail_stage "run-local pilot path unexpectedly exists: ${PILOT_ROOT}" 1
fi
mkdir -p -- "${PILOT_RENDER_ROOT}" "${PILOT_IMAGES_ROOT}" "${PILOT_WORK_ROOT}"
if [[ -L "${PILOT_ROOT}" || -L "${PILOT_RENDER_ROOT}" || -L "${PILOT_IMAGES_ROOT}" || -L "${PILOT_WORK_ROOT}" ]]; then
  fail_stage "pilot output path must not contain a symlink at its mutable roots" 1
fi
if [[ "${PILOT_ROOT}" != "${RENDER_REPO}/tmp/"* ]]; then
  fail_stage "pilot output escaped BrickNet-Render/tmp: ${PILOT_ROOT}" 1
fi
printf 'timestamp_utc\tgpu_id\tmemory_used_mib\tutilization_gpu_pct\n' >"${GPU_SAMPLE_LOG}"
: >"${GPU_SAMPLE_ERROR}"
printf '%s\n' "${PILOT_ROW_IDS[@]}" >"${PILOT_ID_MANIFEST}"

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
      printf '%s\t%s\t%s\t%s\n' "${timestamp}" "${gpu_id}" "${SNAP_MEM[${gpu_id}]}" "${SNAP_UTIL[${gpu_id}]}"
    done
    sleep "${GPU_SAMPLE_INTERVAL_SECONDS}" || return 1
  done
}

gpu_sampler_loop >>"${GPU_SAMPLE_LOG}" 2>>"${GPU_SAMPLE_ERROR}" &
GPU_SAMPLER_PID="$!"
if ! kill -0 "${GPU_SAMPLER_PID}" 2>/dev/null; then
  fail_stage "GPU sampler exited before pilot launch" 1
fi
stage_ok PILOT_SETUP "pilot_root=${PILOT_ROOT} raw_gpu_log=${GPU_SAMPLE_LOG} sampler_pid=${GPU_SAMPLER_PID} row_count=${#PILOT_ROW_IDS[@]} workers_per_gpu=${WORKERS_PER_GPU} gpu_ids=${GPU_IDS_CSV}"

set_stage PILOT "official SFT row-id pilot row_count=${#PILOT_ROW_IDS[@]} workers_per_gpu=${WORKERS_PER_GPU} gpu_ids=${GPU_IDS_CSV} output=${PILOT_ROOT}"
if "${OFFICIAL_LAUNCHER}" \
  --workers-per-gpu "${WORKERS_PER_GPU}" \
  --row-ids "${PILOT_ROW_IDS[@]}" \
  --render-output-root "${PILOT_RENDER_ROOT}" \
  --images-v2-root "${PILOT_IMAGES_ROOT}" \
  --tmp-dir "${PILOT_WORK_ROOT}"; then
  pilot_rc=0
else
  pilot_rc="$?"
fi

sampler_stop_failed=0
if ! cleanup_gpu_sampler; then
  sampler_stop_failed=1
fi
if [[ -s "${GPU_SAMPLE_ERROR}" ]]; then
  sampler_stop_failed=1
fi
if (( sampler_stop_failed != 0 )); then
  fail_stage "GPU sampler failed; refusing formal start (see ${GPU_SAMPLE_ERROR})" 1
fi
if (( pilot_rc != 0 )); then
  fail_stage "official SFT row-id pilot failed; refusing formal start" "${pilot_rc}"
fi

PILOT_SUMMARY="${PILOT_RENDER_ROOT}/.bricknet_render_v2_official/sft_summary.json"
PILOT_VIEW_ROOT="${PILOT_RENDER_ROOT}/sft_8view_renders_v2_rowids"
PILOT_COLLAGE_ROOT="${PILOT_IMAGES_ROOT}/SFT"
if [[ ! -s "${PILOT_SUMMARY}" ]]; then
  fail_stage "pilot summary is missing or empty: ${PILOT_SUMMARY}" 1
fi
if ! jq -e \
  '(.split == "SFT" and .rows_requested == 32 and .stats.failed == 0 and (.stats.rendered + .stats.skipped + .stats.stitched) == 32)' \
  "${PILOT_SUMMARY}" >/dev/null; then
  fail_stage "pilot summary does not prove 32 successful SFT rows: ${PILOT_SUMMARY}" 1
fi
if [[ ! -d "${PILOT_VIEW_ROOT}" || ! -d "${PILOT_COLLAGE_ROOT}" ]]; then
  fail_stage "pilot output roots are missing" 1
fi
PILOT_VIEW_COUNT="$(find "${PILOT_VIEW_ROOT}" -mindepth 1 -maxdepth 1 -type d -print | wc -l)"
PILOT_COLLAGE_COUNT="$(find "${PILOT_COLLAGE_ROOT}" -mindepth 1 -maxdepth 1 -type f -name '*.png' -print | wc -l)"
if [[ "${PILOT_VIEW_COUNT}" != 32 || "${PILOT_COLLAGE_COUNT}" != 32 ]]; then
  fail_stage "pilot output count mismatch: view_dirs=${PILOT_VIEW_COUNT} collages=${PILOT_COLLAGE_COUNT}" 1
fi
EXPECTED_PILOT_IDS="$(printf '%s\n' "${PILOT_ROW_IDS[@]}" | sort -n | paste -sd ' ' -)"
ACTUAL_PILOT_IDS="$(find "${PILOT_VIEW_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | paste -sd ' ' -)"
if [[ "${ACTUAL_PILOT_IDS}" != "${EXPECTED_PILOT_IDS}" ]]; then
  fail_stage "pilot row-id output mismatch: expected=${EXPECTED_PILOT_IDS} actual=${ACTUAL_PILOT_IDS}" 1
fi

printf 'gpu_id\tsample_count\tpeak_memory_used_mib\tpeak_utilization_gpu_pct\n' >"${GPU_PEAK_LOG}"
for gpu_id in "${GPU_IDS[@]}"; do
  peak_line="$(awk -F '\t' -v target="${gpu_id}" '
    NR == 1 { next }
    $2 == target {
      count += 1
      memory = $3 + 0
      util = $4 + 0
      if (count == 1 || memory > peak_memory) peak_memory = memory
      if (count == 1 || util > peak_util) peak_util = util
    }
    END {
      if (count == 0) exit 1
      printf "%d\t%d\t%d\n", count, peak_memory, peak_util
    }
  ' "${GPU_SAMPLE_LOG}")" || fail_stage "no valid GPU samples recorded for GPU ${gpu_id}" 1
  IFS=$'\t' read -r sample_count peak_memory peak_util <<<"${peak_line}"
  printf '%s\t%s\t%s\t%s\n' "${gpu_id}" "${sample_count}" "${peak_memory}" "${peak_util}" >>"${GPU_PEAK_LOG}"
  if (( peak_memory >= GPU_FORMAL_MAX_PEAK_MIB )); then
    fail_stage "pilot GPU ${gpu_id} peak ${peak_memory} MiB reached the ${GPU_FORMAL_MAX_PEAK_MIB} MiB safety ceiling" 1
  fi
done
stage_ok PILOT_OK "summary=${PILOT_SUMMARY} row_id_manifest=${PILOT_ID_MANIFEST} rows=32 view_dirs=${PILOT_VIEW_COUNT} collages=${PILOT_COLLAGE_COUNT} gpu_samples=${GPU_SAMPLE_LOG} gpu_peaks=${GPU_PEAK_LOG} workers_per_gpu=${WORKERS_PER_GPU} gpu_ids=${GPU_IDS_CSV}"

# A different job may claim a card between the pilot and formal render.  Make
# the ownership boundary explicit again instead of assuming the pilot still
# owns the devices after its child processes exit.
wait_for_idle

verify_frozen_inputs
set_stage FORMAL_RENDER "official SFT launcher=${OFFICIAL_LAUNCHER} workers_per_gpu=${WORKERS_PER_GPU} gpu_ids=${GPU_IDS_CSV} formal_log=${SUPERVISOR_LOG}"
log "pilot passed; official full SFT render begins; output is inherited by ${SUPERVISOR_LOG}"
if "${OFFICIAL_LAUNCHER}" --workers-per-gpu "${WORKERS_PER_GPU}"; then
  stage_ok COMPLETE "official SFT render completed successfully"
  exit 0
else
  formal_rc="$?"
  fail_stage "official SFT render failed" "${formal_rc}"
fi
