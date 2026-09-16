#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VERSION="${1:-$(date +%Y.%m.%d)}"
IMAGE="${SOURCE_INVERSION_IMAGE:-source-inversion:protected}"
OUTPUT_ROOT="${SOURCE_INVERSION_DELIVERY_ROOT:-${REPO_ROOT}/deployment_artifacts/delivery}"
PACKAGE_NAME="source-inversion-${VERSION}-linux-amd64"
PACKAGE_DIR="${OUTPUT_ROOT}/${PACKAGE_NAME}"
ARCHIVE_NAME="${PACKAGE_NAME}.docker.tar.gz"
ARCHIVE_PATH="${PACKAGE_DIR}/${ARCHIVE_NAME}"

docker image inspect "${IMAGE}" >/dev/null
mkdir -p "${PACKAGE_DIR}/proto"

echo "Exporting ${IMAGE} to ${ARCHIVE_PATH}"
docker save "${IMAGE}" | gzip -1 >"${ARCHIVE_PATH}"
(
    cd "${PACKAGE_DIR}"
    sha256sum "${ARCHIVE_NAME}" >"${ARCHIVE_NAME}.sha256"
)

cp "${SCRIPT_DIR}/docker-compose.runtime.yml" "${PACKAGE_DIR}/"
cp "${SCRIPT_DIR}/runtime.env.example" "${PACKAGE_DIR}/"
cp "${SCRIPT_DIR}/CLIENT_DEPLOYMENT.md" "${PACKAGE_DIR}/"
cp "${REPO_ROOT}/proto/API_PROTOCOL.md" "${PACKAGE_DIR}/proto/"
cp "${REPO_ROOT}/proto/input.json" "${PACKAGE_DIR}/proto/"
cp "${REPO_ROOT}/proto/output.json" "${PACKAGE_DIR}/proto/"

IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
IMAGE_PLATFORM="$(docker image inspect "${IMAGE}" --format '{{.Os}}/{{.Architecture}}')"
IMAGE_REVISION="$(docker image inspect "${IMAGE}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
GIT_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)"
cat >"${PACKAGE_DIR}/manifest.txt" <<EOF
package=${PACKAGE_NAME}
created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
image=${IMAGE}
image_id=${IMAGE_ID}
image_platform=${IMAGE_PLATFORM}
image_build_revision=${IMAGE_REVISION}
repository_revision=${GIT_REVISION}
archive=${ARCHIVE_NAME}
archive_sha256=$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')
EOF

if [[ "$(id -u)" == "0" && -n "${SUDO_UID:-}" ]]; then
    chown -R "${SUDO_UID}:${SUDO_GID}" "${PACKAGE_DIR}"
fi

echo "Delivery package ready: ${PACKAGE_DIR}"
find "${PACKAGE_DIR}" -maxdepth 2 -type f -printf '%P\t%k KB\n' | sort
