# -*- coding: utf-8 -*-
import os
import glob
import argparse

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")

parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
parser.add_argument("--max_loops", type=int, default=1, help="Maximum number of loop closures per submap")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--vis_stride", type=int, default=1, help="Stride interval in the 3D point cloud image for visualization.")
parser.add_argument("--vis_point_size", type=float, default=0.003, help="Visualization point size")

# --- semantic backend ---
parser.add_argument("--use_semantic_backend", action="store_true", help="Enable semantic backend.")
parser.add_argument("--semantic_backend_cfg", type=str, default="", help="Optional semantic backend config path.")
parser.add_argument("--semantic_min_sim", type=float, default=0.25, help="Semantic similarity threshold.")

# --- semantic gate (ablation) ---
# 兼容旧参数：filter_loops / gate_retrieved
# Solver 支持：off / filter / retrieved / both
parser.add_argument(
    "--semantic_gate_mode",
    type=str,
    default="both",
    choices=["off", "filter", "retrieved", "both", "filter_loops", "gate_retrieved"],
    help="Semantic gate mode: off|filter|retrieved|both (legacy: filter_loops|gate_retrieved)",
)

# ✅ 兼容你现在脚本在传的开关：--disable_semantic_gate
parser.add_argument("--disable_semantic_gate", action="store_true",
                    help="Alias: disable semantic gate/filter even if semantic backend enabled.")

# ✅ 兼容你现在脚本在传的参数：--edge_stats_path
parser.add_argument("--edge_stats_path", type=str, default="",
                    help="Optional: path to save edge stats csv (sim/w/n_good...). Empty = disable.")

# --- semantic factor reweighting ---
parser.add_argument("--semantic_weight_mode", type=str, default="loop_only",
                    choices=["off", "loop_only", "all_edges"],
                    help="semantic factor reweighting: off|loop_only|all_edges")
parser.add_argument("--semantic_w_min", type=float, default=0.25)
parser.add_argument("--semantic_w_max", type=float, default=4.0)
parser.add_argument("--semantic_w_s0", type=float, default=0.25, help="similarity threshold start")
parser.add_argument("--semantic_w_gamma", type=float, default=2.0, help="weight mapping curve")
parser.add_argument("--semantic_w_degen_beta", type=float, default=0.0, help="degeneracy boost beta (0=off)")
parser.add_argument("--semantic_w_degen_ref_good", type=int, default=2000, help="ref good_mask size for degeneracy")

# --- corridor uniqueness down-weighting (NEW) ---

parser.add_argument("--semantic_loop_topk", type=int, default=1,
                    help="Keep only top-K loop candidates after semantic scoring.")
parser.add_argument("--semantic_loop_margin_thr", type=float, default=0.02,
                    help="If (best-second_best)<thr, treat as ambiguous and down-weight.")
parser.add_argument("--semantic_u_enable", action="store_true",
                    help="Enable corridor uniqueness down-weighting (recommended for long similar corridors).")
parser.add_argument("--semantic_u_topk_submaps", type=int, default=8,
                    help="How many recent submaps to use for uniqueness estimation.")
parser.add_argument("--semantic_u_m0", type=float, default=0.05,
                    help="Scale for margin->u. smaller=more sensitive down-weighting.")
parser.add_argument("--semantic_u_min", type=float, default=0.25,
                    help="Lower bound of uniqueness factor u in [u_min, 1].")

parser.add_argument("--disable_loop_closure", action="store_true", help="Disable loop closure entirely (ablation)")


def _normalize_semantic_gate_mode(mode: str, disable: bool) -> str:
    """把 legacy gate_mode 映射到 Solver 支持的 off/filter/retrieved/both。"""
    if disable:
        return "off"
    m = (mode or "").strip().lower()
    if m in ("filter_loops", "filterloop", "filterloops"):
        return "filter"
    if m in ("gate_retrieved", "gateretrieved"):
        return "retrieved"
    if m in ("off", "filter", "retrieved", "both"):
        return m
    return "both"


def main():
    args = parser.parse_args()

    # 兼容：disable_semantic_gate 覆盖 semantic_gate_mode；并把 legacy 名字映射到 Solver 接受的名字
    args.semantic_gate_mode = _normalize_semantic_gate_mode(args.semantic_gate_mode, args.disable_semantic_gate)

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False,
        vis_stride=args.vis_stride,
        vis_point_size=args.vis_point_size,

        use_semantic_backend=args.use_semantic_backend,
        semantic_backend_cfg=args.semantic_backend_cfg,
        semantic_min_sim=args.semantic_min_sim,

        semantic_gate_mode=args.semantic_gate_mode,
        disable_semantic_gate=args.disable_semantic_gate,

        edge_stats_path=args.edge_stats_path,

        semantic_weight_mode=args.semantic_weight_mode,
        semantic_w_min=args.semantic_w_min,
        semantic_w_max=args.semantic_w_max,
        semantic_w_s0=args.semantic_w_s0,
        semantic_w_gamma=args.semantic_w_gamma,
        semantic_w_degen_beta=args.semantic_w_degen_beta,
        semantic_w_degen_ref_good=args.semantic_w_degen_ref_good,

        # NEW: corridor uniqueness
        semantic_u_enable=args.semantic_u_enable,
        semantic_u_topk_submaps=args.semantic_u_topk_submaps,
        semantic_u_m0=args.semantic_u_m0,
        semantic_u_min=args.semantic_u_min,
        semantic_loop_topk=args.semantic_loop_topk,
        semantic_loop_margin_thr=args.semantic_loop_margin_thr,

    )

    print("Initializing and loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    print(f"Loading images from {args.image_folder}...")
    image_names = [
        f for f in glob.glob(os.path.join(args.image_folder, "*"))
        if "depth" not in os.path.basename(f).lower()
        and "txt" not in os.path.basename(f).lower()
        and "db" not in os.path.basename(f).lower()
    ]
    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    data = []

    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            print(image_names_subset)
            max_loops_eff = 0 if args.disable_loop_closure else args.max_loops

            predictions = solver.run_predictions(image_names_subset, model, max_loops_eff)
            data.append(predictions["intrinsic"][:, 0, 0])

            solver.add_points(predictions)
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            loop_closure_detected = len(predictions.get("detected_loops", [])) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    if not args.vis_map:
        solver.update_all_submap_vis()

    if args.log_results:
        if os.path.isdir(args.log_path):
            args.log_path = os.path.join(args.log_path, "poses.txt")
        parent = os.path.dirname(args.log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        solver.map.write_poses_to_file(args.log_path)

        if not args.skip_dense_log:
            solver.map.save_framewise_pointclouds(args.log_path.replace(".txt", "_logs"))

    if args.plot_focal_lengths:
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        plt.figure(figsize=(8, 6))
        for i, values in enumerate(data):
            y = values
            x = [i] * len(values)
            plt.scatter(x, y, color=colors[i], label=f"List {i+1}")
        plt.xlabel("poses")
        plt.ylabel("Focal lengths")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
