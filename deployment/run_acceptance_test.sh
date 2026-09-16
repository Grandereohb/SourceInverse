#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EPOCHS="${1:-50}"
INPUT_PATH="${2:-}"
IMAGE="${SOURCE_INVERSION_IMAGE:-source-inversion:protected}"
API_TOKEN="${SOURCE_INVERSION_ACCEPTANCE_TOKEN:-acceptance-test-token}"
POLL_SECONDS="${SOURCE_INVERSION_ACCEPTANCE_POLL_SECONDS:-2}"
MAX_WAIT_SECONDS="${SOURCE_INVERSION_ACCEPTANCE_MAX_WAIT_SECONDS:-43200}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="acceptance-${STAMP}"
RESULT_DIR="${REPO_ROOT}/deployment_runs/${RUN_NAME}"
JOBS_DIR="${RESULT_DIR}/jobs"
CALLBACK_DIR="${RESULT_DIR}/callback"
DOWNLOAD_DIR="${RESULT_DIR}/download"
NETWORK="source-inversion-acceptance"
APP_CONTAINER="source-inversion-app-${STAMP}"
CALLBACK_CONTAINER="source-inversion-callback-${STAMP}"

if [[ "${EPOCHS}" != "full" && ! "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 <positive-epochs|full> [input.json]" >&2
    exit 2
fi

if [[ -z "${INPUT_PATH}" ]]; then
    INPUT_PATH="$(find "${REPO_ROOT}" -maxdepth 1 -type f -name 'historical*input.json' -print -quit)"
fi
if [[ -z "${INPUT_PATH}" || ! -f "${INPUT_PATH}" ]]; then
    echo "Historical input JSON was not found. Pass its path as the second argument." >&2
    exit 2
fi
INPUT_PATH="$(readlink -f "${INPUT_PATH}")"

for command in docker curl python3; do
    command -v "${command}" >/dev/null || {
        echo "Required command is unavailable: ${command}" >&2
        exit 2
    }
done
docker image inspect "${IMAGE}" >/dev/null
python3 -m json.tool "${INPUT_PATH}" >/dev/null

mkdir -p "${JOBS_DIR}" "${CALLBACK_DIR}" "${DOWNLOAD_DIR}"
chmod 0777 "${JOBS_DIR}" "${CALLBACK_DIR}"
printf "timestamp\tcpu\tmemory\tpids\n" >"${RESULT_DIR}/resources.tsv"

cleanup() {
    docker logs "${APP_CONTAINER}" >"${RESULT_DIR}/app.log" 2>&1 || true
    docker logs "${CALLBACK_CONTAINER}" >"${RESULT_DIR}/callback.log" 2>&1 || true
    docker rm -f "${APP_CONTAINER}" "${CALLBACK_CONTAINER}" >/dev/null 2>&1 || true
    if [[ "$(id -u)" == "0" && -n "${SUDO_UID:-}" ]]; then
        chown -R "${SUDO_UID}:${SUDO_GID}" "${RESULT_DIR}" || true
    fi
}
trap cleanup EXIT

docker network inspect "${NETWORK}" >/dev/null 2>&1 \
    || docker network create "${NETWORK}" >/dev/null

docker run -d \
    --name "${CALLBACK_CONTAINER}" \
    --network "${NETWORK}" \
    -e CALLBACK_OUTPUT_PATH=/callback-output/output.json.gz \
    -v "${SCRIPT_DIR}/test_callback_receiver.py:/callback.py:ro" \
    -v "${CALLBACK_DIR}:/callback-output" \
    python:3.11-slim python /callback.py >/dev/null

APP_ARGS=(
    run -d
    --name "${APP_CONTAINER}"
    --network "${NETWORK}"
    -p 127.0.0.1::8000
    -e "SOURCE_INVERSION_API_TOKEN=${API_TOKEN}"
    -e "SOURCE_INVERSION_CALLBACK_URL_OVERRIDE=http://${CALLBACK_CONTAINER}:9000/result"
    -e SOURCE_INVERSION_CALLBACK_TIMEOUT_SECONDS=10
    -e SOURCE_INVERSION_CALLBACK_RETRY_DELAYS_SECONDS=0
    -v "${JOBS_DIR}:/data/jobs"
)
if [[ "${EPOCHS}" != "full" ]]; then
    APP_ARGS+=(-e "SOURCE_INVERSION_TEST_EPOCHS=${EPOCHS}")
fi
APP_ARGS+=("${IMAGE}")
docker "${APP_ARGS[@]}" >/dev/null

PORT="$(docker port "${APP_CONTAINER}" 8000/tcp | awk -F: 'NR == 1 {print $NF}')"
API="http://127.0.0.1:${PORT}"
for _ in $(seq 1 60); do
    if curl -fsS "${API}/api/v1/health" >/dev/null; then
        break
    fi
    sleep 1
done
curl -fsS "${API}/api/v1/health" >/dev/null

REQUEST_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["request_id"])' "${INPUT_PATH}")"
START_SECONDS="$(date +%s)"
curl -fsS -X POST "${API}/api/v1/jobs" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${API_TOKEN}" \
    -H "Idempotency-Key: ${REQUEST_ID}" \
    --data-binary "@${INPUT_PATH}" \
    -o "${RESULT_DIR}/submit.json"
JOB_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["job_id"])' "${RESULT_DIR}/submit.json")"

echo "Run: ${RUN_NAME}"
echo "Epochs: ${EPOCHS}"
echo "Job: ${JOB_ID}"
echo "Results: ${RESULT_DIR}"

while true; do
    NOW="$(date +%s)"
    if (( NOW - START_SECONDS > MAX_WAIT_SECONDS )); then
        echo "Acceptance job exceeded ${MAX_WAIT_SECONDS} seconds." >&2
        exit 1
    fi
    docker stats --no-stream \
        --format "$(date +%Y-%m-%dT%H:%M:%S)\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.PIDs}}" \
        "${APP_CONTAINER}" >>"${RESULT_DIR}/resources.tsv"
    curl -fsS -H "Authorization: Bearer ${API_TOKEN}" \
        "${API}/api/v1/jobs/${JOB_ID}" -o "${RESULT_DIR}/status.json"
    STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "${RESULT_DIR}/status.json")"
    printf '\rStatus: %-18s elapsed: %4ss' "${STATUS}" "$((NOW - START_SECONDS))"
    if [[ "${STATUS}" == "completed" || "${STATUS}" == "failed" ]]; then
        echo
        break
    fi
    sleep "${POLL_SECONDS}"
done

curl -fsS -H "Authorization: Bearer ${API_TOKEN}" \
    "${API}/api/v1/jobs/${JOB_ID}/output" -o "${DOWNLOAD_DIR}/output.json" || true
curl -fsS -H "Authorization: Bearer ${API_TOKEN}" \
    "${API}/api/v1/jobs/${JOB_ID}/result" -o "${DOWNLOAD_DIR}/result.zip" || true

END_SECONDS="$(date +%s)"
echo "Elapsed: $((END_SECONDS - START_SECONDS)) seconds"
echo "Final status: ${STATUS}"
echo "Artifacts: ${RESULT_DIR}"
[[ "${STATUS}" == "completed" ]]
