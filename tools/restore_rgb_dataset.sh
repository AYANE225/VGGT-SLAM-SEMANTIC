#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

PART_PREFIX="DATA/rgb.tar.gz.part_"
CHECKSUM_FILE="DATA/rgb.tar.gz.sha256"
ARCHIVE="DATA/rgb.tar.gz"

echo "[restore] repo: $REPO_DIR"

# 1) 检查分卷和校验文件是否存在
shopt -s nullglob
parts=( ${PART_PREFIX}* )
shopt -u nullglob
if [[ ${#parts[@]} -eq 0 ]]; then
  echo "[restore] 未找到分卷 ${PART_PREFIX}*" >&2
  exit 1
fi
if [[ ! -f "$CHECKSUM_FILE" ]]; then
  echo "[restore] 未找到校验文件 $CHECKSUM_FILE" >&2
  exit 1
fi
echo "[restore] 找到分卷 ${#parts[@]} 个"

# 2) 校验分卷
echo "[restore] 正在校验分卷..."
sha256sum -c "$CHECKSUM_FILE"
echo "[restore] 校验通过"

# 3) 合并分卷
echo "[restore] 合并分卷到 $ARCHIVE"
cat ${PART_PREFIX}* > "$ARCHIVE"

# 4) 解压数据集
echo "[restore] 解压到仓库相对路径 DATA/rgb"
tar -xzf "$ARCHIVE"

echo "[restore] 完成。示例列出前10个文件："
ls -lh DATA/rgb | head || true