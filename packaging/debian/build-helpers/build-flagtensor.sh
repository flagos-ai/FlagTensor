#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
BASE_IMAGE="${1:-nvidia/cuda:12.8.0-devel-ubuntu22.04}"
IMAGE_TAG="flagtensor-deb:nvidia"
OUTPUT_DIR="${PROJECT_DIR}/debian-packages"

docker build --network=host \
    -f "${SCRIPT_DIR}/Dockerfile.deb" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$IMAGE_TAG" "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"
find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -delete
CID=$(docker create "$IMAGE_TAG")
docker cp "$CID:/output/." "$OUTPUT_DIR/"
docker rm "$CID" > /dev/null

echo ""; echo ">>> Output:"; ls -lh "$OUTPUT_DIR"
