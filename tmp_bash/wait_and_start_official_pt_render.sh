#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for both physical GPUs to have enough stable free memory, then replace
# this process with the frozen official PT render supervisor.  This watcher
# never signals or otherwise modifies unrelated processes.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SUPERVISOR="${SCRIPT_DIR}/supervise_official_pt_render.sh"
CONFIG="/data/jiahao/task/BrickNet/configs/bricknet_mm_image_v2_official_pt_render.json"
LAUNCHER="/data/jiahao/task/BrickNet/scripts/render_bricknet_render_8views_official_pt.sh"
DRIVER="/data/jiahao/task/BrickNet/scripts/render_bricknet_render_8views.py"
PT_NPZ="/data/jiahao/task/BrickNet/data/bricknet_datasets/pt.npz"
RENDER_REPO="/data/jiahao/task/BrickNet-Render"

EXPECTED_SUPERVISOR_SHA256="9584b3d84c096c1bf9e2c007e35fd8c86d557e152b9e396cee8622363e1d4996"
EXPECTED_CONFIG_SHA256="61ec07b40077e842df8c4b768ebee5ae2b391fb35d4f0ff23e36be982ca7918a"
EXPECTED_LAUNCHER_SHA256="770a355731944cd711644beb932cc9f4143ccd292e32b6bb1d7dfd85c9c740ce"
EXPECTED_DRIVER_SHA256="0d36ef76c802f5ce76f4c0c839800cd7f1c3cd644074c30e257ebddae6f205ec"
EXPECTED_PT_NPZ_SHA256="a3c9ebe27fa49c97a3dce2c76152522aad0bf1b9c6c1859ee639e23935597ab6"
EXPECTED_RENDERER_COMMIT="b49e8733782a5fa7040cc776b5fba8c55b0b894e"

GPU_IDS=(0 1)
MIN_FREE_MIB=52200
POLL_SECONDS=30
REQUIRED_STABLE_SAMPLES=2
MIN_DATA_START_BYTES=280000000000

LOCK_FILE="${SCRIPT_DIR}/wait_and_start_official_pt_render.lock"
PID_FILE="${SCRIPT_DIR}/wait_and_start_official_pt_render.pid"
STATUS_FILE="${SCRIPT_DIR}/wait_and_start_official_pt_render.status"
LOG_FILE="${SCRIPT_DIR}/wait_and_start_official_pt_render.log"
SUPERVISOR_LOCK="${SCRIPT_DIR}/supervise_official_pt_render.lock"

declare -A GPU_USED=()
declare -A GPU_FREE=()
declare -A GPU_UTIL=()
stable_samples=0
last_free_0="unknown"
last_free_1="unknown"
CURRENT_STAGE="INIT"

log() {
  local message="$*"
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "${message}" | tee -a -- "${LOG_FILE}"
}

write_status() {
  local stage="$1"
  local result="$2"
  local detail="${3:-}"
  local tmp_path="${STATUS_FILE}.tmp.$$"
  detail="${detail//$'\n'/ | }"
  {
    printf 'pid=%s\n' "$$"
    printf 'stage=%s\n' "${stage}"
    printf 'result=%s\n' "${result}"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'gpu_ids=0,1\n'
    printf 'minimum_free_mib_per_gpu=%s\n' "${MIN_FREE_MIB}"
    printf 'poll_seconds=%s\n' "${POLL_SECONDS}"
    printf 'required_stable_samples=%s\n' "${REQUIRED_STABLE_SAMPLES}"
    printf 'stable_samples=%s\n' "${stable_samples}"
    printf 'gpu0_free_mib=%s\n' "${last_free_0}"
    printf 'gpu1_free_mib=%s\n' "${last_free_1}"
    printf 'supervisor=%s\n' "${SUPERVISOR}"
    printf 'detail=%s\n' "${detail}"
  } >"${tmp_path}"
  mv -f -- "${tmp_path}" "${STATUS_FILE}"
}

fail() {
  local message="$1"
  local code="${2:-1}"
  CURRENT_STAGE="FAILED"
  write_status FAILED FAILED "${message}; exit_code=${code}"
  log "stage=FAILED result=FAILED ${message} exit_code=${code}"
  exit "${code}"
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  [[ -f "${path}" && ! -L "${path}" ]] || \
    fail "${label} must be a regular non-symlink file: ${path}"
  actual="$(sha256sum -- "${path}" | awk '{print $1}')" || \
    fail "unable to hash ${label}: ${path}"
  [[ "${actual}" == "${expected}" ]] || \
    fail "${label} SHA-256 drift: expected=${expected} actual=${actual} path=${path}"
}

verify_frozen_inputs() {
  local renderer_commit renderer_status data_free
  require_sha256 "${SUPERVISOR}" "${EXPECTED_SUPERVISOR_SHA256}" supervisor
  require_sha256 "${CONFIG}" "${EXPECTED_CONFIG_SHA256}" config
  require_sha256 "${LAUNCHER}" "${EXPECTED_LAUNCHER_SHA256}" launcher
  require_sha256 "${DRIVER}" "${EXPECTED_DRIVER_SHA256}" driver
  require_sha256 "${PT_NPZ}" "${EXPECTED_PT_NPZ_SHA256}" pt_npz

  [[ -d "${RENDER_REPO}" && ! -L "${RENDER_REPO}" ]] || \
    fail "renderer checkout is missing or is a symlink: ${RENDER_REPO}"
  renderer_commit="$(git -C "${RENDER_REPO}" rev-parse HEAD 2>/dev/null)" || \
    fail "unable to resolve BrickNet-Render commit"
  [[ "${renderer_commit}" == "${EXPECTED_RENDERER_COMMIT}" ]] || \
    fail "BrickNet-Render commit drift: expected=${EXPECTED_RENDERER_COMMIT} actual=${renderer_commit}"
  renderer_status="$(git -C "${RENDER_REPO}" status --porcelain=v1 --untracked-files=all)" || \
    fail "unable to inspect BrickNet-Render checkout"
  [[ -z "${renderer_status}" ]] || \
    fail "BrickNet-Render checkout is dirty: ${renderer_status//$'\n'/ | }"

  data_free="$(df -B1 --output=avail /data | awk 'NR == 2 {print $1}')" || \
    fail "unable to query /data available bytes"
  [[ "${data_free}" =~ ^[0-9]+$ ]] || fail "invalid /data available-byte value: ${data_free}"
  (( data_free >= MIN_DATA_START_BYTES )) || \
    fail "/data start gate failed: available=${data_free} minimum=${MIN_DATA_START_BYTES}"
}

supervisor_is_active() {
  local probe_fd
  exec {probe_fd}>"${SUPERVISOR_LOCK}"
  if flock -n "${probe_fd}"; then
    flock -u "${probe_fd}"
    exec {probe_fd}>&-
    return 1
  fi
  exec {probe_fd}>&-
  return 0
}

gpu_snapshot() {
  local gpu_id line line_count idx used free util
  GPU_USED=()
  GPU_FREE=()
  GPU_UTIL=()
  for gpu_id in "${GPU_IDS[@]}"; do
    line="$(nvidia-smi --id="${gpu_id}" \
      --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits 2>&1)" || return 1
    line_count="$(printf '%s\n' "${line}" | awk 'NF {count++} END {print count + 0}')"
    [[ "${line_count}" == 1 ]] || return 1
    IFS=',' read -r idx used free util <<<"${line}"
    idx="$(printf '%s' "${idx}" | tr -d '[:space:]%')"
    used="$(printf '%s' "${used}" | tr -d '[:space:]%')"
    free="$(printf '%s' "${free}" | tr -d '[:space:]%')"
    util="$(printf '%s' "${util}" | tr -d '[:space:]%')"
    [[ "${idx}" == "${gpu_id}" && "${used}" =~ ^[0-9]+$ && \
      "${free}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] || return 1
    GPU_USED["${gpu_id}"]="${used}"
    GPU_FREE["${gpu_id}"]="${free}"
    GPU_UTIL["${gpu_id}"]="${util}"
  done
}

on_exit() {
  local code="$?"
  trap - EXIT
  if (( code != 0 )) && [[ "${CURRENT_STAGE}" != "FAILED" ]]; then
    write_status "${CURRENT_STAGE}" FAILED "watcher exited unexpectedly; exit_code=${code}"
    log "stage=${CURRENT_STAGE} result=FAILED watcher exited unexpectedly exit_code=${code}"
  fi
  rm -f -- "${PID_FILE}"
  exit "${code}"
}

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is unavailable"
command -v flock >/dev/null 2>&1 || fail "flock is unavailable"
command -v git >/dev/null 2>&1 || fail "git is unavailable"

mkdir -p -- "${SCRIPT_DIR}"
exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
  printf '[%s] another PT GPU watcher already holds %s\n' \
    "$(date --iso-8601=seconds)" "${LOCK_FILE}" >&2
  exit 75
fi
printf '%s\n' "$$" >"${PID_FILE}"
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

: >>"${LOG_FILE}"
CURRENT_STAGE="VALIDATING"
write_status VALIDATING RUNNING "checking frozen inputs before monitoring"
verify_frozen_inputs

if supervisor_is_active; then
  CURRENT_STAGE="ALREADY_RUNNING"
  write_status ALREADY_RUNNING OK "an official PT supervisor already owns its production lock; no duplicate launched"
  log "stage=ALREADY_RUNNING result=OK existing supervisor detected; no duplicate launched"
  exit 0
fi

CURRENT_STAGE="WAITING_GPU"
log "stage=WAITING_GPU result=RUNNING gpu_ids=0,1 minimum_free_mib_per_gpu=${MIN_FREE_MIB} stable_samples_required=${REQUIRED_STABLE_SAMPLES} poll_seconds=${POLL_SECONDS}"

while :; do
  if supervisor_is_active; then
    CURRENT_STAGE="ALREADY_RUNNING"
    write_status ALREADY_RUNNING OK "an official PT supervisor started while this watcher was waiting; no duplicate launched"
    log "stage=ALREADY_RUNNING result=OK existing supervisor detected; no duplicate launched"
    exit 0
  fi

  if ! gpu_snapshot; then
    stable_samples=0
    last_free_0="unknown"
    last_free_1="unknown"
    write_status WAITING_GPU RUNNING "GPU query failed; stable counter reset"
    log "stage=WAITING_GPU result=RUNNING gpu_query_failed=1 stable_samples=0"
    sleep "${POLL_SECONDS}"
    continue
  fi

  last_free_0="${GPU_FREE[0]}"
  last_free_1="${GPU_FREE[1]}"
  if (( GPU_FREE[0] >= MIN_FREE_MIB && GPU_FREE[1] >= MIN_FREE_MIB )); then
    stable_samples=$((stable_samples + 1))
  else
    stable_samples=0
  fi
  write_status WAITING_GPU RUNNING \
    "gpu0_used=${GPU_USED[0]} gpu0_util=${GPU_UTIL[0]} gpu1_used=${GPU_USED[1]} gpu1_util=${GPU_UTIL[1]}"
  log "stage=WAITING_GPU result=RUNNING gpu0_free_mib=${GPU_FREE[0]} gpu0_used_mib=${GPU_USED[0]} gpu0_util_pct=${GPU_UTIL[0]} gpu1_free_mib=${GPU_FREE[1]} gpu1_used_mib=${GPU_USED[1]} gpu1_util_pct=${GPU_UTIL[1]} stable_samples=${stable_samples}/${REQUIRED_STABLE_SAMPLES}"

  if (( stable_samples >= REQUIRED_STABLE_SAMPLES )); then
    CURRENT_STAGE="FINAL_VALIDATION"
    write_status FINAL_VALIDATION RUNNING "stable GPU admission reached; rechecking immutable inputs and supervisor lock"
    verify_frozen_inputs
    if supervisor_is_active; then
      CURRENT_STAGE="ALREADY_RUNNING"
      write_status ALREADY_RUNNING OK "an official PT supervisor won the launch race; no duplicate launched"
      log "stage=ALREADY_RUNNING result=OK launch race resolved without duplicate"
      exit 0
    fi

    CURRENT_STAGE="LAUNCHING"
    write_status LAUNCHING RUNNING \
      "executing frozen dual-GPU supervisor with workers_per_gpu=9; watcher PID becomes supervisor PID"
    log "stage=LAUNCHING result=RUNNING supervisor=${SUPERVISOR} gpu_ids=0,1 workers_per_gpu=9"
    trap - EXIT INT TERM
    exec env \
      BRICKNET_OFFICIAL_PT_RENDER_CONFIG="${CONFIG}" \
      BRICKNET_OFFICIAL_PT_WORKERS_PER_GPU=9 \
      "${SUPERVISOR}"
  fi

  sleep "${POLL_SECONDS}"
done
