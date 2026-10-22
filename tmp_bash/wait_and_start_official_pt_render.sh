#!/usr/bin/env bash
set -Eeuo pipefail

# Adaptive, resumable launcher for the official PT BrickNet-Render job.
#
# The immutable render identity is capped at seven workers per physical GPU.
# Runtime concurrency starts at seven and may only move downward after the
# one-shot supervisor returns the explicit RECOVERABLE_GPU_PRESSURE status.
# This watcher observes unrelated CUDA jobs but never signals them. Its only
# signal target is its direct child supervisor; that supervisor owns and
# validates its own formal launcher process group.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SUPERVISOR="${SCRIPT_DIR}/supervise_official_pt_render.sh"
SUPERVISOR_STATUS="${SCRIPT_DIR}/supervise_official_pt_render.status"
CONFIG="/data/jiahao/task/BrickNet/configs/bricknet_mm_image_v2_official_pt_render.json"
LAUNCHER="/data/jiahao/task/BrickNet/scripts/render_bricknet_render_8views_official_pt.sh"
DRIVER="/data/jiahao/task/BrickNet/scripts/render_bricknet_render_8views.py"
PT_NPZ="/data/jiahao/task/BrickNet/data/bricknet_datasets/pt.npz"
PT_IDENTITY="/data/jiahao/task/BrickNet/outputs_gt/.bricknet_render_v2_official_pt/pt_identity.json"
PILOT_ROOT="/data/jiahao/task/BrickNet-Render/tmp/official_pt_pilot_20260910T205638Z-485442"
RENDER_REPO="/data/jiahao/task/BrickNet-Render"

# Filled after the supervisor is finalized; the watcher refuses to launch if
# any frozen implementation byte changes.
EXPECTED_SUPERVISOR_SHA256="00fea1302eae6925777e29a3645103ba8d302842335e2063a02d5491b4bb012a"
EXPECTED_CONFIG_SHA256="61ec07b40077e842df8c4b768ebee5ae2b391fb35d4f0ff23e36be982ca7918a"
EXPECTED_LAUNCHER_SHA256="4a79b248112794fde72915b1204045e32392ab4d4ba78753cddf7aa47d369855"
EXPECTED_DRIVER_SHA256="e9e883f1d0e3282c14db0a5e7713a60f55ead88ce41575069f1e3fd86bb1876e"
EXPECTED_PT_NPZ_SHA256="a3c9ebe27fa49c97a3dce2c76152522aad0bf1b9c6c1859ee639e23935597ab6"
EXPECTED_RENDERER_COMMIT="b49e8733782a5fa7040cc776b5fba8c55b0b894e"
EXPECTED_OLD_IDENTITY_SHA256="9af34d313b6d021c07c5dbfd6869de7d480e94f48fe0ecc44e92a8cafc02c0a5"

GPU_IDS=(0 1)
IDENTITY_WORKERS_PER_GPU=7
MAX_ACTIVE_WORKERS_PER_GPU=7
MIN_ACTIVE_WORKERS_PER_GPU=1
# Frozen nine-worker pilot: ceil(43884 MiB / 9) = 4876 MiB/worker.
PER_WORKER_MIB=4876
ADAPTIVE_RESERVE_MIB=16384
GPU_MAX_USED_MIB=95000
POLL_SECONDS=5
REQUIRED_STABLE_SAMPLES=3
MIN_DATA_START_BYTES=280000000000

LOCK_FILE="${SCRIPT_DIR}/wait_and_start_official_pt_render.lock"
PID_FILE="${SCRIPT_DIR}/wait_and_start_official_pt_render.pid"
STATUS_FILE="${SCRIPT_DIR}/wait_and_start_official_pt_render.status"
LOG_FILE="${SCRIPT_DIR}/wait_and_start_official_pt_render.log"
ATTEMPT_MANIFEST="${SCRIPT_DIR}/wait_and_start_official_pt_render.attempts.jsonl"
STATE_FILE="${SCRIPT_DIR}/wait_and_start_official_pt_render.state"
SUPERVISOR_LOCK="${SCRIPT_DIR}/supervise_official_pt_render.lock"

declare -A GPU_USED=()
declare -A GPU_FREE=()
declare -A GPU_UTIL=()
CURRENT_STAGE="INIT"
CURRENT_ACTIVE_WORKERS="${MAX_ACTIVE_WORKERS_PER_GPU}"
STABLE_SAMPLES=0
LAST_FREE_0="unknown"
LAST_FREE_1="unknown"
LAST_CAPACITY="unknown"
ATTEMPT_NUMBER=0
CHILD_PID=""
SELECTED_ACTIVE_WORKERS=""
WRAPPER_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SELF_TEST=0

if (( $# > 0 )); then
  if (( $# == 1 )) && [[ "$1" == "--self-test" ]]; then
    SELF_TEST=1
  else
    printf 'usage: %s [--self-test]\n' "$0" >&2
    exit 2
  fi
fi

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a -- "${LOG_FILE}"
}

status_get() {
  local key="$1" path="${2:-${SUPERVISOR_STATUS}}"
  [[ -f "${path}" && ! -L "${path}" ]] || return 1
  awk -F= -v key="${key}" '$1 == key {sub(/^[^=]*=/, ""); print; found=1; exit} END {exit !found}' "${path}"
}

manifest_sha256() {
  if [[ -f "${ATTEMPT_MANIFEST}" && ! -L "${ATTEMPT_MANIFEST}" ]]; then
    sha256sum -- "${ATTEMPT_MANIFEST}" | awk '{print $1}'
  else
    printf 'none\n'
  fi
}

write_status() {
  local stage="$1" result="$2" detail="${3:-}" tmp_path manifest_sha
  tmp_path="${STATUS_FILE}.tmp.$$"
  detail="${detail//$'\n'/ | }"
  manifest_sha="$(manifest_sha256)"
  {
    printf 'run_id=%s\n' "${WRAPPER_RUN_ID}"
    printf 'pid=%s\n' "$$"
    printf 'stage=%s\n' "${stage}"
    printf 'result=%s\n' "${result}"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'gpu_ids=0,1\n'
    printf 'identity_workers_per_gpu=%s\n' "${IDENTITY_WORKERS_PER_GPU}"
    printf 'active_workers_per_gpu=%s\n' "${CURRENT_ACTIVE_WORKERS}"
    printf 'worker_floor=%s\n' "${CURRENT_ACTIVE_WORKERS}"
    printf 'worker_policy=identity_cap_plus_runtime_downshift\n'
    printf 'per_worker_mib=%s\n' "${PER_WORKER_MIB}"
    printf 'adaptive_reserve_mib=%s\n' "${ADAPTIVE_RESERVE_MIB}"
    printf 'poll_seconds=%s\n' "${POLL_SECONDS}"
    printf 'required_stable_samples=%s\n' "${REQUIRED_STABLE_SAMPLES}"
    printf 'stable_samples=%s\n' "${STABLE_SAMPLES}"
    printf 'gpu0_free_mib=%s\n' "${LAST_FREE_0}"
    printf 'gpu1_free_mib=%s\n' "${LAST_FREE_1}"
    printf 'candidate_capacity_per_gpu=%s\n' "${LAST_CAPACITY}"
    printf 'attempt_number=%s\n' "${ATTEMPT_NUMBER}"
    printf 'child_pid=%s\n' "${CHILD_PID:-none}"
    printf 'supervisor=%s\n' "${SUPERVISOR}"
    printf 'supervisor_status=%s\n' "${SUPERVISOR_STATUS}"
    printf 'pilot_root=%s\n' "${PILOT_ROOT}"
    printf 'adaptive_attempt_manifest=%s\n' "${ATTEMPT_MANIFEST}"
    printf 'adaptive_attempt_manifest_sha256=%s\n' "${manifest_sha}"
    printf 'state_file=%s\n' "${STATE_FILE}"
    printf 'identity_base_sha256=%s\n' "${EXPECTED_OLD_IDENTITY_SHA256}"
    printf 'detail=%s\n' "${detail}"
  } >"${tmp_path}"
  mv -f -- "${tmp_path}" "${STATUS_FILE}"
}

fail() {
  local message="$1" code="${2:-1}"
  CURRENT_STAGE="FAILED"
  write_status FAILED FAILED "${message}; exit_code=${code}"
  log "stage=FAILED result=FAILED ${message} exit_code=${code}"
  exit "${code}"
}

require_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" && ! -L "${path}" ]] || fail "${label} must be a regular non-symlink file: ${path}"
  actual="$(sha256sum -- "${path}" | awk '{print $1}')" || fail "unable to hash ${label}: ${path}"
  [[ "${actual}" == "${expected}" ]] || fail "${label} SHA-256 drift: expected=${expected} actual=${actual} path=${path}"
}

verify_frozen_inputs() {
  local renderer_commit renderer_status data_free
  require_sha256 "${SUPERVISOR}" "${EXPECTED_SUPERVISOR_SHA256}" supervisor
  require_sha256 "${CONFIG}" "${EXPECTED_CONFIG_SHA256}" config
  require_sha256 "${LAUNCHER}" "${EXPECTED_LAUNCHER_SHA256}" launcher
  require_sha256 "${DRIVER}" "${EXPECTED_DRIVER_SHA256}" driver
  require_sha256 "${PT_NPZ}" "${EXPECTED_PT_NPZ_SHA256}" pt_npz
  [[ -d "${PILOT_ROOT}" && ! -L "${PILOT_ROOT}" ]] || fail "frozen reusable pilot root is unavailable: ${PILOT_ROOT}"
  grep -Fxq 'PILOT_EXIT=0' "${PILOT_ROOT}/pilot.log" || fail "frozen reusable pilot is not successful: ${PILOT_ROOT}/pilot.log"
  [[ -d "${RENDER_REPO}" && ! -L "${RENDER_REPO}" ]] || fail "renderer checkout is missing or a symlink: ${RENDER_REPO}"
  renderer_commit="$(git -C "${RENDER_REPO}" rev-parse HEAD 2>/dev/null)" || fail "unable to resolve BrickNet-Render commit"
  [[ "${renderer_commit}" == "${EXPECTED_RENDERER_COMMIT}" ]] || fail "BrickNet-Render commit drift: expected=${EXPECTED_RENDERER_COMMIT} actual=${renderer_commit}"
  renderer_status="$(git -C "${RENDER_REPO}" status --porcelain=v1 --untracked-files=all)" || fail "unable to inspect BrickNet-Render checkout"
  [[ -z "${renderer_status}" ]] || fail "BrickNet-Render checkout is dirty: ${renderer_status//$'\n'/ | }"
  data_free="$(df -B1 --output=avail /data | awk 'NR == 2 {print $1}')" || fail "unable to query /data available bytes"
  [[ "${data_free}" =~ ^[0-9]+$ ]] || fail "invalid /data available-byte value: ${data_free}"
  (( data_free >= MIN_DATA_START_BYTES )) || fail "/data start gate failed: available=${data_free} minimum=${MIN_DATA_START_BYTES}"
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
  local gpu_id line count idx used free util
  GPU_USED=(); GPU_FREE=(); GPU_UTIL=()
  for gpu_id in "${GPU_IDS[@]}"; do
    line="$(nvidia-smi --id="${gpu_id}" --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1)" || return 1
    count="$(printf '%s\n' "${line}" | awk 'NF {n++} END {print n + 0}')"
    [[ "${count}" == 1 ]] || return 1
    IFS=',' read -r idx used free util <<<"${line}"
    idx="$(printf '%s' "${idx}" | tr -d '[:space:]%')"
    used="$(printf '%s' "${used}" | tr -d '[:space:]%')"
    free="$(printf '%s' "${free}" | tr -d '[:space:]%')"
    util="$(printf '%s' "${util}" | tr -d '[:space:]%')"
    [[ "${idx}" == "${gpu_id}" && "${used}" =~ ^[0-9]+$ && "${free}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] || return 1
    GPU_USED["${gpu_id}"]="${used}"
    GPU_FREE["${gpu_id}"]="${free}"
    GPU_UTIL["${gpu_id}"]="${util}"
  done
}

capacity_from_free() {
  local free_mib="$1" capacity
  if (( free_mib <= ADAPTIVE_RESERVE_MIB )); then
    printf '0\n'
    return 0
  fi
  capacity=$(( (free_mib - ADAPTIVE_RESERVE_MIB) / PER_WORKER_MIB ))
  (( capacity > MAX_ACTIVE_WORKERS_PER_GPU )) && capacity="${MAX_ACTIVE_WORKERS_PER_GPU}"
  (( capacity < 0 )) && capacity=0
  printf '%s\n' "${capacity}"
}

snapshot_capacity() {
  local cap0 cap1
  cap0="$(capacity_from_free "${GPU_FREE[0]}")"
  cap1="$(capacity_from_free "${GPU_FREE[1]}")"
  if (( cap0 < cap1 )); then printf '%s\n' "${cap0}"; else printf '%s\n' "${cap1}"; fi
}

required_free_for_active() {
  local active="$1"
  printf '%s\n' "$((active * PER_WORKER_MIB + ADAPTIVE_RESERVE_MIB))"
}

persist_active_floor() {
  local value="$1" tmp_path="${STATE_FILE}.tmp.$$"
  printf 'active_workers_per_gpu=%s\n' "${value}" >"${tmp_path}"
  mv -f -- "${tmp_path}" "${STATE_FILE}"
}

load_active_floor() {
  local value=""
  if [[ -e "${STATE_FILE}" || -L "${STATE_FILE}" ]]; then
    [[ -f "${STATE_FILE}" && ! -L "${STATE_FILE}" ]] || fail "adaptive state must be a regular non-symlink file: ${STATE_FILE}"
    value="$(status_get active_workers_per_gpu "${STATE_FILE}" || true)"
    [[ "${value}" =~ ^[1-7]$ ]] || fail "adaptive state has invalid active_workers_per_gpu: ${value}"
    CURRENT_ACTIVE_WORKERS="${value}"
  fi
}

append_manifest_event() {
  local event="$1" rc="${2:-null}" supervisor_run_id="${3:-none}" supervisor_stage="${4:-none}" supervisor_result="${5:-none}" note="${6:-}"
  local now data_free encoded gpu0_json gpu1_json candidate_json
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  data_free="$(df -B1 --output=avail /data 2>/dev/null | awk 'NR == 2 {print $1}')"
  [[ "${data_free}" =~ ^[0-9]+$ ]] || data_free=0
  gpu0_json="${LAST_FREE_0}"; gpu1_json="${LAST_FREE_1}"; candidate_json="${LAST_CAPACITY}"
  [[ "${gpu0_json}" =~ ^[0-9]+$ ]] || gpu0_json=0
  [[ "${gpu1_json}" =~ ^[0-9]+$ ]] || gpu1_json=0
  [[ "${candidate_json}" =~ ^[0-9]+$ ]] || candidate_json=0
  encoded="$(jq -cn \
    --arg schema bricknet_pt_adaptive_attempt_v1 \
    --arg wrapper_run_id "${WRAPPER_RUN_ID}" \
    --arg event "${event}" \
    --arg created_utc "${now}" \
    --arg supervisor_run_id "${supervisor_run_id}" \
    --arg supervisor_stage "${supervisor_stage}" \
    --arg supervisor_result "${supervisor_result}" \
    --arg note "${note}" \
    --argjson attempt "${ATTEMPT_NUMBER}" \
    --argjson identity_workers "${IDENTITY_WORKERS_PER_GPU}" \
    --argjson active_workers "${CURRENT_ACTIVE_WORKERS}" \
    --argjson gpu0_free "${gpu0_json}" \
    --argjson gpu1_free "${gpu1_json}" \
    --argjson candidate "${candidate_json}" \
    --argjson data_free "${data_free}" \
    --argjson exit_code "${rc}" \
    '{schema:$schema,wrapper_run_id:$wrapper_run_id,event:$event,created_utc:$created_utc,attempt_number:$attempt,identity_workers_per_gpu:$identity_workers,active_workers_per_gpu:$active_workers,gpu0_free_mib:$gpu0_free,gpu1_free_mib:$gpu1_free,candidate_capacity_per_gpu:$candidate,data_available_bytes:$data_free,supervisor_run_id:$supervisor_run_id,supervisor_stage:$supervisor_stage,supervisor_result:$supervisor_result,exit_code:$exit_code,note:$note}')" || fail "unable to encode adaptive attempt manifest event"
  printf '%s\n' "${encoded}" >>"${ATTEMPT_MANIFEST}"
}

wait_for_start_capacity() {
  local required_free capacity
  required_free="$(required_free_for_active "${CURRENT_ACTIVE_WORKERS}")"
  STABLE_SAMPLES=0
  CURRENT_STAGE="WAITING_GPU"
  while :; do
    if supervisor_is_active; then
      fail "another official PT supervisor owns its production lock" 75
    fi
    if ! gpu_snapshot; then
      STABLE_SAMPLES=0
      LAST_FREE_0="unknown"; LAST_FREE_1="unknown"; LAST_CAPACITY="unknown"
      write_status WAITING_GPU RUNNING "GPU query failed; stable counter reset"
      log "stage=WAITING_GPU gpu_query_failed=1 active_workers=${CURRENT_ACTIVE_WORKERS} stable=0/${REQUIRED_STABLE_SAMPLES}"
      sleep "${POLL_SECONDS}"
      continue
    fi
    LAST_FREE_0="${GPU_FREE[0]}"; LAST_FREE_1="${GPU_FREE[1]}"
    capacity="$(snapshot_capacity)"; LAST_CAPACITY="${capacity}"
    if (( capacity >= CURRENT_ACTIVE_WORKERS && GPU_USED[0] < GPU_MAX_USED_MIB && GPU_USED[1] < GPU_MAX_USED_MIB )); then
      STABLE_SAMPLES=$((STABLE_SAMPLES + 1))
    else
      STABLE_SAMPLES=0
    fi
    write_status WAITING_GPU RUNNING "required_free_mib=${required_free} gpu0_used=${GPU_USED[0]} gpu1_used=${GPU_USED[1]}"
    log "stage=WAITING_GPU active_workers=${CURRENT_ACTIVE_WORKERS} required_free_mib=${required_free} gpu0_free=${GPU_FREE[0]} gpu1_free=${GPU_FREE[1]} candidate=${capacity} stable=${STABLE_SAMPLES}/${REQUIRED_STABLE_SAMPLES}"
    (( STABLE_SAMPLES >= REQUIRED_STABLE_SAMPLES )) && return 0
    sleep "${POLL_SECONDS}"
  done
}

select_after_pressure() {
  local previous="$1" capacity candidate_min=999 target
  SELECTED_ACTIVE_WORKERS=""
  STABLE_SAMPLES=0
  CURRENT_STAGE="ADAPTIVE_GPU_RECOVERY"
  while :; do
    if ! gpu_snapshot; then
      STABLE_SAMPLES=0; candidate_min=999
      LAST_FREE_0="unknown"; LAST_FREE_1="unknown"; LAST_CAPACITY="unknown"
      write_status ADAPTIVE_GPU_RECOVERY RUNNING "GPU query failed; recovery samples reset"
      sleep "${POLL_SECONDS}"
      continue
    fi
    LAST_FREE_0="${GPU_FREE[0]}"; LAST_FREE_1="${GPU_FREE[1]}"
    capacity="$(snapshot_capacity)"; LAST_CAPACITY="${capacity}"
    if (( capacity < MIN_ACTIVE_WORKERS_PER_GPU )); then
      STABLE_SAMPLES=0; candidate_min=999
      write_status ADAPTIVE_GPU_RECOVERY RUNNING "no worker fits the coexistence reserve; waiting"
      log "stage=ADAPTIVE_GPU_RECOVERY previous=${previous} candidate=0 gpu0_free=${GPU_FREE[0]} gpu1_free=${GPU_FREE[1]} action=wait"
      sleep "${POLL_SECONDS}"
      continue
    fi
    (( capacity < candidate_min )) && candidate_min="${capacity}"
    STABLE_SAMPLES=$((STABLE_SAMPLES + 1))
    write_status ADAPTIVE_GPU_RECOVERY RUNNING "previous_active=${previous} candidate_min=${candidate_min}"
    log "stage=ADAPTIVE_GPU_RECOVERY previous=${previous} candidate=${capacity} candidate_min=${candidate_min} stable=${STABLE_SAMPLES}/${REQUIRED_STABLE_SAMPLES}"
    if (( STABLE_SAMPLES >= REQUIRED_STABLE_SAMPLES )); then
      if (( previous > MIN_ACTIVE_WORKERS_PER_GPU )); then
        target=$((previous - 1))
        (( candidate_min < target )) && target="${candidate_min}"
      else
        target="${MIN_ACTIVE_WORKERS_PER_GPU}"
      fi
      (( target < MIN_ACTIVE_WORKERS_PER_GPU )) && target="${MIN_ACTIVE_WORKERS_PER_GPU}"
      SELECTED_ACTIVE_WORKERS="${target}"
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

owned_child_is_safe() {
  local child="$1" ppid cmdline
  [[ "${child}" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "${child}" 2>/dev/null || return 1
  ppid="$(ps -o ppid= -p "${child}" 2>/dev/null | tr -d '[:space:]')"
  [[ "${ppid}" == "$$" ]] || return 1
  cmdline="$(tr '\0' ' ' <"/proc/${child}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"${SUPERVISOR}"* ]]
}

on_signal() {
  local code="$1"
  trap - INT TERM
  if [[ -n "${CHILD_PID}" ]] && owned_child_is_safe "${CHILD_PID}"; then
    log "stage=${CURRENT_STAGE} signal_exit=${code} action=TERM_owned_supervisor child_pid=${CHILD_PID}"
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
    wait "${CHILD_PID}" 2>/dev/null || true
  else
    log "stage=${CURRENT_STAGE} signal_exit=${code} no_verified_owned_child=1"
  fi
  CHILD_PID=""
  CURRENT_STAGE="STOPPED"
  write_status STOPPED FAILED "watcher stopped by signal; exit_code=${code}"
  exit "${code}"
}

on_exit() {
  local code="$?"
  trap - EXIT
  rm -f -- "${PID_FILE}"
  if (( code != 0 )) && [[ "${CURRENT_STAGE}" != "FAILED" && "${CURRENT_STAGE}" != "STOPPED" ]]; then
    write_status "${CURRENT_STAGE}" FAILED "adaptive watcher exited; exit_code=${code}"
  fi
  exit "${code}"
}

self_test() {
  local actual
  actual="$(capacity_from_free 56158)"; [[ "${actual}" == 7 ]] || return 1
  actual="$(capacity_from_free 40000)"; [[ "${actual}" == 4 ]] || return 1
  actual="$(capacity_from_free 20000)"; [[ "${actual}" == 0 ]] || return 1
  [[ $((7 - 1)) == 6 ]]
  (( (40000 - ADAPTIVE_RESERVE_MIB) / PER_WORKER_MIB == 4 ))
  [[ 76 -eq 76 && "RECOVERABLE_GPU_PRESSURE" == RECOVERABLE_GPU_PRESSURE && "FAILED" == FAILED ]]
  printf 'adaptive PT watcher self-test passed: capacity=7/4/0, monotonic floor, exact rc=76 contract\n'
}

if (( SELF_TEST == 1 )); then
  self_test
  exit 0
fi

for required_command in nvidia-smi sha256sum flock git jq awk df; do
  command -v "${required_command}" >/dev/null 2>&1 || fail "required command is unavailable: ${required_command}"
done

mkdir -p -- "${SCRIPT_DIR}"
exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
  printf '[%s] another adaptive PT watcher already holds %s\n' "$(date --iso-8601=seconds)" "${LOCK_FILE}" >&2
  exit 75
fi
printf '%s\n' "$$" >"${PID_FILE}"
trap on_exit EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
: >>"${LOG_FILE}"
: >>"${ATTEMPT_MANIFEST}"
load_active_floor

CURRENT_STAGE="VALIDATING"
write_status VALIDATING RUNNING "checking frozen inputs before adaptive monitoring"
verify_frozen_inputs

while :; do
  wait_for_start_capacity
  CURRENT_STAGE="FINAL_VALIDATION"
  write_status FINAL_VALIDATION RUNNING "stable coexistence capacity reached; rechecking frozen inputs"
  verify_frozen_inputs

  identity_workers="$(jq -er '.workers_per_gpu' "${PT_IDENTITY}" 2>/dev/null || true)"
  if [[ "${identity_workers}" != 9 && "${identity_workers}" != "${IDENTITY_WORKERS_PER_GPU}" ]]; then
    fail "PT identity workers must be the frozen old 9 or adopted cap ${IDENTITY_WORKERS_PER_GPU}; got ${identity_workers}"
  fi

  ATTEMPT_NUMBER=$((ATTEMPT_NUMBER + 1))
  append_manifest_event attempt_start null none none none "identity_before_workers=${identity_workers}"
  CURRENT_STAGE="SUPERVISOR_RUNNING"
  write_status SUPERVISOR_RUNNING RUNNING "launching attempt=${ATTEMPT_NUMBER} identity_workers=${IDENTITY_WORKERS_PER_GPU} active_workers=${CURRENT_ACTIVE_WORKERS}"
  log "stage=SUPERVISOR_RUNNING attempt=${ATTEMPT_NUMBER} identity_before_workers=${identity_workers} identity_workers=${IDENTITY_WORKERS_PER_GPU} active_workers=${CURRENT_ACTIVE_WORKERS}"

  supervisor_env=(
    BRICKNET_OFFICIAL_PT_RENDER_CONFIG="${CONFIG}"
    BRICKNET_OFFICIAL_PT_WORKERS_PER_GPU="${IDENTITY_WORKERS_PER_GPU}"
    BRICKNET_OFFICIAL_PT_ACTIVE_WORKERS_PER_GPU="${CURRENT_ACTIVE_WORKERS}"
    BRICKNET_OFFICIAL_PT_PILOT_ROOT="${PILOT_ROOT}"
  )
  if [[ "${identity_workers}" == 9 ]]; then
    supervisor_env+=(BRICKNET_OFFICIAL_PT_REFREEZE_WORKERS="${IDENTITY_WORKERS_PER_GPU}")
  fi

  set +e
  env "${supervisor_env[@]}" "${SUPERVISOR}" &
  CHILD_PID="$!"
  attempt_child_pid="${CHILD_PID}"
  write_status SUPERVISOR_RUNNING RUNNING "attempt=${ATTEMPT_NUMBER} child_pid=${attempt_child_pid} identity_workers=${IDENTITY_WORKERS_PER_GPU} active_workers=${CURRENT_ACTIVE_WORKERS}"
  log "stage=SUPERVISOR_RUNNING attempt=${ATTEMPT_NUMBER} child_pid=${attempt_child_pid} supervisor_started=1"
  wait "${CHILD_PID}"
  supervisor_rc="$?"
  set -e
  CHILD_PID=""

  supervisor_run_id="$(status_get run_id || printf none)"
  supervisor_pid="$(status_get pid || printf missing)"
  supervisor_run_kind="$(status_get run_kind || printf missing)"
  supervisor_stage="$(status_get stage || printf missing)"
  supervisor_result="$(status_get result || printf missing)"
  supervisor_active="$(status_get active_workers_per_gpu || printf missing)"
  if gpu_snapshot; then
    LAST_FREE_0="${GPU_FREE[0]}"; LAST_FREE_1="${GPU_FREE[1]}"; LAST_CAPACITY="$(snapshot_capacity)"
  fi
  append_manifest_event attempt_end "${supervisor_rc}" "${supervisor_run_id}" "${supervisor_stage}" "${supervisor_result}" "supervisor_active=${supervisor_active}"

  [[ "${supervisor_pid}" == "${attempt_child_pid}" ]] || fail "supervisor status does not belong to this attempt: child_pid=${attempt_child_pid} status_pid=${supervisor_pid}"
  [[ "${supervisor_run_kind}" == PRODUCTION ]] || fail "supervisor status run_kind is not PRODUCTION: ${supervisor_run_kind}"

  if (( supervisor_rc == 0 )); then
    [[ "${supervisor_stage}" == RENDER_COMPLETE_PENDING_VERIFY && "${supervisor_result}" == OK ]] || fail "supervisor exited 0 without accepted terminal status: stage=${supervisor_stage} result=${supervisor_result}"
    [[ "${supervisor_active}" == "${CURRENT_ACTIVE_WORKERS}" ]] || fail "supervisor terminal active worker mismatch: expected=${CURRENT_ACTIVE_WORKERS} actual=${supervisor_active}"
    CURRENT_STAGE="COMPLETED"
    write_status COMPLETED OK "supervisor_run_id=${supervisor_run_id}; strict verify and completion gate remain pending"
    log "stage=COMPLETED result=OK attempt=${ATTEMPT_NUMBER} supervisor_run_id=${supervisor_run_id} active_workers=${CURRENT_ACTIVE_WORKERS}"
    exit 0
  fi

  if (( supervisor_rc == 76 )) && [[ "${supervisor_stage}" == RECOVERABLE_GPU_PRESSURE && "${supervisor_result}" == FAILED ]]; then
    [[ "${supervisor_active}" == "${CURRENT_ACTIVE_WORKERS}" ]] || fail "recoverable supervisor active worker mismatch: expected=${CURRENT_ACTIVE_WORKERS} actual=${supervisor_active}"
    previous_active="${CURRENT_ACTIVE_WORKERS}"
    select_after_pressure "${previous_active}"
    next_active="${SELECTED_ACTIVE_WORKERS}"
    [[ "${next_active}" =~ ^[1-7]$ ]] || fail "adaptive recovery selected invalid active worker count: ${next_active}"
    (( next_active <= previous_active )) || fail "adaptive policy attempted to increase workers: previous=${previous_active} next=${next_active}"
    CURRENT_ACTIVE_WORKERS="${next_active}"
    persist_active_floor "${CURRENT_ACTIVE_WORKERS}"
    append_manifest_event downshift null "${supervisor_run_id}" "${supervisor_stage}" "${supervisor_result}" "previous_active=${previous_active};next_active=${CURRENT_ACTIVE_WORKERS}"
    write_status RETRY_SCHEDULED RUNNING "GPU protection pause accepted; previous_active=${previous_active} next_active=${CURRENT_ACTIVE_WORKERS}"
    log "stage=RETRY_SCHEDULED event=gpu_pressure previous_active=${previous_active} next_active=${CURRENT_ACTIVE_WORKERS} action=resume"
    continue
  fi

  CURRENT_STAGE="FAILED"
  write_status FAILED FAILED "supervisor exit is not auto-retryable; rc=${supervisor_rc} stage=${supervisor_stage} result=${supervisor_result}"
  log "stage=FAILED result=FAILED supervisor_rc=${supervisor_rc} supervisor_stage=${supervisor_stage} supervisor_result=${supervisor_result} active_workers=${CURRENT_ACTIVE_WORKERS} action=no_retry"
  exit "${supervisor_rc}"
done
