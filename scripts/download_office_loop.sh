#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ZIP="${ROOT_DIR}/office_loop.zip"
DEST_DIR="${ROOT_DIR}/office_loop"

if [[ -d "${DEST_DIR}" ]]; then
  echo "office_loop already exists at ${DEST_DIR}"
  exit 0
fi

if [[ ! -f "${DEST_ZIP}" ]]; then
  echo "Downloading office_loop.zip..."
  curl -L -o "${DEST_ZIP}" "https://raw.githubusercontent.com/MIT-SPARK/VGGT-SLAM/main/office_loop.zip"
fi

echo "Extracting ${DEST_ZIP} to ${DEST_DIR}..."
unzip -q "${DEST_ZIP}" -d "${ROOT_DIR}"

echo "Done. Dataset available at ${DEST_DIR}"
