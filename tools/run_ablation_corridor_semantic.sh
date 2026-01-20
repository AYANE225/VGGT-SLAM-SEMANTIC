#!/usr/bin/env bash
set -e

PY="$(which python)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/DATA/rgb"

TAG="ablation_corridor_uniq_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT}/RUNS/${TAG}"
LOG_DIR="${ROOT}/LOG"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

if [ ! -d "${DATA}" ]; then
  echo "[ERR] DATA not found: ${DATA}"
  echo "      Please put corridor images under: ${DATA}"
  exit 1
fi

COMMON=(
  --image_folder "${DATA}"
  --log_results
  --log_path "${RUN_ROOT}"
  --submap_size 16
  --overlapping_window_size 1
  --downsample_factor 1
  --max_loops 1
  --min_disparity 0
  --conf_threshold 25.0
  --vis_stride 1
  --vis_point_size 0.003
)

SEM_COMMON=(
  --use_semantic_backend
  --semantic_backend_cfg ""
  --semantic_min_sim 0.60
)

UNIQ=(
  --semantic_u_enable
  --semantic_u_topk_submaps 8
  --semantic_u_m0 0.05
  --semantic_u_min 0.25
)

run_one () {
  local name="$1"; shift
  local out="${RUN_ROOT}/${name}"
  local log="${LOG_DIR}/${name}.log"
  local edge="${out}/edge_stats.csv"
  mkdir -p "${out}"

  echo "===================="
  echo "[RUN] ${name}"
  echo "PY   : ${PY}"
  echo "ROOT : ${ROOT}"
  echo "DATA : ${DATA}"
  echo "log  : ${log}"
  echo "edge : ${edge}"
  echo "out  : ${out}"
  echo "===================="

  "${PY}" "${ROOT}/main.py" \
    "${COMMON[@]}" \
    --log_path "${out}" \
    --edge_stats_path "${edge}" \
    "$@" \
    2>&1 | tee "${log}"
}

run_one "03_all_edges" \
  "${SEM_COMMON[@]}" \
  "${UNIQ[@]}" \
  --semantic_gate_mode off \
  --semantic_weight_mode all_edges \
  --semantic_w_min 0.25 \
  --semantic_w_max 2.0 \
  --semantic_w_s0 0.60 \
  --semantic_w_gamma 2.0 \
  --semantic_w_degen_beta 0.0

echo
echo "[DONE] runs at: ${RUN_ROOT}"
echo "  head -n 20 ${RUN_ROOT}/03_all_edges/edge_stats.csv"
