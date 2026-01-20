#!/usr/bin/env bash
set -euo pipefail

echo "===== grep: detected_loops / loop closure ====="
grep -RIn --exclude-dir=.git --exclude=*.pyc \
  -E "detected_loops|loop_closure|LoopClosure|retrieve_best_score_frame|get_num_loops|get_frames_from_loops" \
  vggt_slam | head -n 120

echo
echo "===== grep: where loop_closure module is used ====="
grep -RIn --exclude-dir=.git --exclude=*.pyc \
  -E "import vggt_slam.loop_closure|from vggt_slam.loop_closure|loop_closure\." \
  vggt_slam | head -n 120

echo
echo "===== grep: graph optimize / add loop factors ====="
grep -RIn --exclude-dir=.git --exclude=*.pyc \
  -E "add_loop|addLoop|BetweenFactor|loop factor|detected_loop" \
  vggt_slam | head -n 120
