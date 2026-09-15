#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-source-inversion}"
IMAGE_TAG="${IMAGE_TAG:-protected}"
BASE_IMAGE="${BASE_IMAGE:-python:3.11-slim}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
BUILD_REVISION="${BUILD_REVISION:-$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || printf unknown)}"
BUILD_DATE="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || {
    echo "Base image is not loaded: ${BASE_IMAGE}" >&2
    echo "Load the offline archive and tag it before building." >&2
    exit 1
}

echo "Building ${FULL_IMAGE} from local base ${BASE_IMAGE}"
docker build \
    --pull=false \
    --file "${REPO_ROOT}/deployment/Dockerfile.protected" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
    --build-arg "BUILD_REVISION=${BUILD_REVISION}" \
    --build-arg "BUILD_DATE=${BUILD_DATE}" \
    --tag "${FULL_IMAGE}" \
    "${REPO_ROOT}"

"${SCRIPT_DIR}/scan_runtime_image.sh" "${FULL_IMAGE}"

echo "Protected image ready: ${FULL_IMAGE}"
