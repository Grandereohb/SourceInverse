#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${1:-source-inversion:protected}"
CONTAINER_ID=""
TEMP_DIR="$(mktemp -d)"

cleanup() {
    if [[ -n "${CONTAINER_ID}" ]]; then
        docker rm -f "${CONTAINER_ID}" >/dev/null 2>&1 || true
    fi
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

docker image inspect "${IMAGE}" >/dev/null

USER_NAME="$(docker image inspect "${IMAGE}" --format '{{.Config.User}}')"
if [[ -z "${USER_NAME}" || "${USER_NAME}" == "0" || "${USER_NAME}" == "root" ]]; then
    echo "Runtime image must use a non-root user; found '${USER_NAME:-unset}'." >&2
    exit 1
fi

CONTAINER_ID="$(docker create "${IMAGE}")"
docker export "${CONTAINER_ID}" --output "${TEMP_DIR}/filesystem.tar"
tar -tf "${TEMP_DIR}/filesystem.tar" >"${TEMP_DIR}/files.txt"

FORBIDDEN_PATTERN='(^|/)opt/source-inversion/(deployment|pinn_source)/.*\.py[co]?$|(^|/)opt/source-inversion/source_inversion_launcher\.py[co]?$|(^|/)opt/source-inversion/(deployment/)?tests?(/|$)|(^|/)opt/source-inversion/\.git(/|$)|(^|/)opt/source-inversion/.*\.build(/|$)|(^|/)opt/source-inversion/nuitka-report\.xml$'
if grep -E "${FORBIDDEN_PATTERN}" "${TEMP_DIR}/files.txt" >"${TEMP_DIR}/forbidden.txt"; then
    echo "Protected image contains forbidden first-party source or build files:" >&2
    sed -n '1,80p' "${TEMP_DIR}/forbidden.txt" >&2
    exit 1
fi

docker run --rm --entrypoint /opt/source-inversion/source-inversion \
    "${IMAGE}" --help >/dev/null

echo "Runtime image scan passed: ${IMAGE} (user=${USER_NAME})"
