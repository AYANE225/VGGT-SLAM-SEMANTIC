#!/usr/bin/env bash
set -euo pipefail

# Run any command and tee stdout/stderr to <repo>/LOG/<name>__YYYYmmdd_HHMMSS.log
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${repo_root}/LOG"

name="${1:-run}"
shift || true
ts="$(date +"%Y%m%d_%H%M%S")"
log="${repo_root}/LOG/${name}__${ts}.log"

echo "[LOG] ${log}"
echo "[CMD] $*" | tee -a "${log}"
# 合并 stdout+stderr，并实时打印 + 写文件
"$@" 2>&1 | tee -a "${log}"
