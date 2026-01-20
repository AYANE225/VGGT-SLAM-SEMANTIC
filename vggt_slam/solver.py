# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import open3d as o3d
import viser
import viser.transforms as viser_tf
from termcolor import colored

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective
from vggt_slam.gradio_viewer import TrimeshViewer

def color_point_cloud_by_confidence(pcd, confidence, cmap="viridis"):
    """
    Color a point cloud based on per-point confidence values.
    Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud.
        confidence (np.ndarray): Confidence values, shape (N,).
        cmap (str): Matplotlib colormap name.
    """
    assert len(confidence) == len(pcd.points), "Confidence length must match number of points"
    confidence_normalized = (confidence - np.min(confidence)) / (np.ptp(confidence) + 1e-8)
    colormap = plt.get_cmap(cmap)
    colors = colormap(confidence_normalized)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

class Viewer:
    """纯可视化：不要在这里初始化 semantic backend（避免重复加载/副作用）"""

    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_show_frames.on_update(self._on_update_show_frames)

        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}

        num_rand_colors = 250
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int, image_scale: float = 0.5) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            self.submap_frames[submap_id] = []
            self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            img_resized = cv2.resize(
                img,
                (int(img.shape[1] * image_scale), int(img.shape[0] * image_scale)),
                interpolation=cv2.INTER_AREA,
            )

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img_resized,
                line_width=3.0,
                color=self.random_colors[submap_id],
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible

@dataclass
class EdgeStat:
    """记录每条图边（里程计/回环）的语义相似度与权重，便于论文做 ablation & 可解释性。"""
    time: str
    edge_type: str  # odom|loop
    src: int
    dst: int
    sim: float
    w: float
    n_good: int
    mask_thr: float
    mask_fallback_or: int
    weight_mode: str
    # --- corridor anti-aliasing additions ---
    margin: float = 0.0   # top1-top2 similarity gap (retrieval space)
    u: float = 1.0        # uniqueness factor in [u_min, 1]

class Solver:
    def __init__(
        self,
        init_conf_threshold: float,
        use_point_map: bool = False,
        visualize_global_map: bool = False,
        use_sim3: bool = False,
        gradio_mode: bool = False,
        vis_stride: int = 1,
        vis_point_size: float = 0.001,
        # --- semantic backend ---
        use_semantic_backend: bool = False,
        semantic_backend_cfg: str = "",
        semantic_min_sim: float = 0.25,
        # --- semantic gate（筛选 loop candidate / retrieved frames）---
        semantic_gate_mode: str = "both",   # off|filter|retrieved|both
        disable_semantic_gate: bool = False,
        # --- semantic factor reweighting（核心：语义参与图优化）---
        semantic_weight_mode: str = "loop_only",  # off|loop_only|all_edges
        semantic_w_min: float = 0.25,
        semantic_w_max: float = 4.0,
        semantic_w_gamma: float = 2.0,
        semantic_w_s0: float = 0.25,
        semantic_w_degen_beta: float = 0.0,
        semantic_w_degen_ref_good: int = 2000,
        # --- corridor anti-aliasing (uniqueness) ---
        semantic_u_enable: bool = True,
        semantic_u_topk_submaps: int = 8,
        semantic_u_m0: float = 0.05,
        semantic_u_min: float = 0.25,
        # --- loop ambiguity control (within retrieved loop candidates) ---
        # Keep only top-K candidates after semantic scoring; additionally compute
        # a margin (best-second best) and down-weight ambiguous loops.
        semantic_loop_topk: int = 1,
        semantic_loop_margin_thr: float = 0.02,
        # --- stats logging ---
        edge_stats_path: str = "",
        **kwargs,
    ):
        # semantic backend
        self.use_semantic_backend = bool(use_semantic_backend)
        self.semantic_backend_cfg = semantic_backend_cfg
        self.semantic_min_sim = float(semantic_min_sim)
        self.semantic_backend = None
        if self.use_semantic_backend:
            try:
                from vggt_slam.semantic_backend import SemanticBackend
                self.semantic_backend = SemanticBackend(cfg_path=self.semantic_backend_cfg)
            except Exception as e:
                print(f"[WARN] semantic backend init failed -> disabled: {e}")
                self.use_semantic_backend = False
                self.semantic_backend = None

        # semantic gate
        self.semantic_gate_mode = str(semantic_gate_mode)
        if self.semantic_gate_mode not in ("off", "filter", "retrieved", "both"):
            print(f"[WARN] unknown semantic_gate_mode={self.semantic_gate_mode}, fallback to both")
            self.semantic_gate_mode = "both"
        self.disable_semantic_gate = bool(disable_semantic_gate)
        if (not self.use_semantic_backend) or (self.semantic_backend is None) or self.disable_semantic_gate:
            self.semantic_gate_mode = "off"

        # loop ambiguity control (only meaningful if semantic backend is enabled)
        self.semantic_loop_topk = int(semantic_loop_topk)
        self.semantic_loop_margin_thr = float(semantic_loop_margin_thr)

        # semantic weights
        self.semantic_weight_mode = str(semantic_weight_mode)
        if self.semantic_weight_mode not in ("off", "loop_only", "all_edges"):
            print(f"[WARN] unknown semantic_weight_mode={self.semantic_weight_mode}, fallback to off")
            self.semantic_weight_mode = "off"

        self.semantic_w_min = float(semantic_w_min)
        self.semantic_w_max = float(semantic_w_max)
        self.semantic_w_gamma = float(semantic_w_gamma)
        self.semantic_w_s0 = float(semantic_w_s0)
        self.semantic_w_degen_beta = float(semantic_w_degen_beta)
        self.semantic_w_degen_ref_good = int(semantic_w_degen_ref_good)

        if (not self.use_semantic_backend) or (self.semantic_backend is None):
            # 没有语义后端就不要开权重（避免 baseline 行为被不小心改变）
            self.semantic_weight_mode = "off"

        # corridor anti-aliasing (uniqueness)
        self.semantic_u_enable = bool(semantic_u_enable)
        self.semantic_u_topk_submaps = int(semantic_u_topk_submaps)
        self.semantic_u_m0 = float(semantic_u_m0)
        self.semantic_u_min = float(semantic_u_min)

        # edge stats
        self.edge_stats_path = str(edge_stats_path).strip()
        self._edge_stats_fp = None
        self._edge_stats_writer = None
        self._edge_stats_init_if_needed()

        self.init_conf_threshold = init_conf_threshold
        self.use_point_map = use_point_map
        self.gradio_mode = gradio_mode

        if self.gradio_mode:
            self.viewer = TrimeshViewer()
        else:
            self.viewer = Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.use_sim3 = use_sim3

        if self.use_sim3:
            from vggt_slam.graph_se3 import PoseGraph
        else:
            from vggt_slam.graph import PoseGraph
        self.graph = PoseGraph()

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.first_edge = True
        self.T_w_kf_minus = None

        self.prior_pcd = None
        self.prior_conf = None

        self.vis_stride = vis_stride
        self.vis_point_size = vis_point_size

        # --- 参数安全检查（中文注释：防止配置不合法导致运行中崩溃） ---
        self._sanitize_semantic_params()

    def _sanitize_semantic_params(self) -> None:
        """
        确保语义相关参数处于合理范围，避免异常配置导致数值不稳定或崩溃。
        该逻辑不会改变主流程，仅做安全兜底与警告。
        """
        # semantic weights
        if self.semantic_w_min > self.semantic_w_max:
            print(
                f"[WARN] semantic_w_min({self.semantic_w_min}) > semantic_w_max({self.semantic_w_max}), swap them."
            )
            self.semantic_w_min, self.semantic_w_max = self.semantic_w_max, self.semantic_w_min

        self.semantic_w_min = max(0.0, float(self.semantic_w_min))
        self.semantic_w_max = max(self.semantic_w_min, float(self.semantic_w_max))
        self.semantic_w_gamma = max(0.0, float(self.semantic_w_gamma))
        self.semantic_w_s0 = max(0.0, float(self.semantic_w_s0))

        # uniqueness params
        self.semantic_u_min = float(np.clip(self.semantic_u_min, 0.0, 1.0))
        self.semantic_u_m0 = max(0.0, float(self.semantic_u_m0))
        if self.semantic_loop_topk < 1:
            print("[WARN] semantic_loop_topk < 1, fallback to 1.")
            self.semantic_loop_topk = 1

    # -------------------------
    # stats logging
    # -------------------------
    def _set_submap_attr(self, submap, attr_name: str, value):
        # Compatible setter for different Submap method names.
        candidates = [
            f'set_all_{attr_name}',
            f'add_all_{attr_name}',
            f'set_{attr_name}',
            f'add_{attr_name}',
        ]
        for m in candidates:
            fn = getattr(submap, m, None)
            if callable(fn):
                fn(value)
                return
        setattr(submap, attr_name, value)

    def _edge_stats_init_if_needed(self):
        if not self.edge_stats_path:
            return
        p = Path(self.edge_stats_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        is_new = (not p.exists())
        self._edge_stats_fp = p.open("a", newline="", encoding="utf-8")
        self._edge_stats_writer = csv.DictWriter(
            self._edge_stats_fp,
            fieldnames=[
                "time", "edge_type", "src", "dst", "sim", "w",
                "n_good", "mask_thr", "mask_fallback_or", "weight_mode",
                "margin", "u",
            ],
        )
        if is_new:
            self._edge_stats_writer.writeheader()
            self._edge_stats_fp.flush()

    def _edge_stats_log(self, stat: EdgeStat):
        if self._edge_stats_writer is None:
            return
        self._edge_stats_writer.writerow({
            "time": stat.time,
            "edge_type": stat.edge_type,
            "src": int(stat.src),
            "dst": int(stat.dst),
            "sim": f"{stat.sim:.6f}",
            "w": f"{stat.w:.6f}",
            "n_good": int(stat.n_good),
            "mask_thr": f"{stat.mask_thr:.6f}",
            "mask_fallback_or": int(stat.mask_fallback_or),
            "weight_mode": str(stat.weight_mode),
            "margin": f"{float(stat.margin):.6f}",
            "u": f"{float(stat.u):.6f}",
        })
        self._edge_stats_fp.flush()

    # -------------------------
    # semantic weight helpers
    # -------------------------
    def _semantic_sim_to_weight(self, sim: float) -> float:
        """
        sim∈[0,1] -> w
        约定：w=1 表示不改变；w>1 表示更相信（噪声更小）；w<1 表示更不相信（噪声更大）
        在 graph.py 内部建议按 sigma' = sigma0 / sqrt(w) 方式实现。
        """
        try:
            s = float(sim)
        except Exception:
            return 1.0
        if not np.isfinite(s):
            return 1.0
        s = max(0.0, min(1.0, s))

        s0 = max(0.0, min(0.999, float(self.semantic_w_s0)))
        if s <= s0:
            t = 0.0
        else:
            t = (s - s0) / (1.0 - s0)
        t = max(0.0, min(1.0, t))

        gamma = max(1e-6, float(self.semantic_w_gamma))
        wmin = float(self.semantic_w_min)
        wmax = float(self.semantic_w_max)
        if wmax < wmin:
            wmin, wmax = wmax, wmin

        w = wmin + (wmax - wmin) * (t ** gamma)
        if (not np.isfinite(w)) or w <= 0:
            return 1.0
        return float(w)

    def _apply_degeneracy_boost(self, w: float, n_good: int) -> float:
        """
        退化增强（可选）：
        d = 1 - clip(n_good/ref,0,1)
        w' = w*(1 + beta*d)
        """
        beta = float(self.semantic_w_degen_beta)
        if beta <= 0:
            return float(w)
        ref = float(max(1, int(self.semantic_w_degen_ref_good)))
        ng = float(max(0, int(n_good)))
        d = 1.0 - min(1.0, ng / ref)
        return float(w) * (1.0 + beta * d)

    # -------------------------
    # corridor anti-aliasing helpers (uniqueness)
    # -------------------------
    def _get_frame_retrieval_vec(self, submap: Submap, frame_idx: int) -> Optional[np.ndarray]:
        """取 retrieval embedding（归一化后）"""
        try:
            vecs = getattr(submap, "retrieval_vectors", None)
            if vecs is None:
                # 兼容不同实现：可能是 get_all_retrieval_vectors()
                if hasattr(submap, "get_all_retrieval_vectors"):
                    vecs = submap.get_all_retrieval_vectors()
            if vecs is None:
                return None
            v = np.asarray(vecs[int(frame_idx)], dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(v) + 1e-8)
            return v / n
        except Exception:
            return None

    def _get_keyframe_retrieval_vec(self, submap: Submap, frame_idx: int) -> Optional[np.ndarray]:
        return self._get_frame_retrieval_vec(submap, frame_idx)

    def _compute_uniqueness_from_map(
        self,
        q_vec: Optional[np.ndarray],
        exclude_sid: int,
    ) -> Tuple[float, float, float, float]:
        """
        返回 (margin, u, top1, top2)
        margin = top1 - top2（越小越歧义）
        u ∈ [u_min, 1]（越小越降权）
        """
        if (not self.semantic_u_enable) or (q_vec is None):
            return 0.0, 1.0, 0.0, 0.0

        cands: List[np.ndarray] = []
        try:
            submaps = self.map.get_submaps()
            # 取最近若干个 submap 的关键帧 embedding
            for sm in submaps[-int(max(1, self.semantic_u_topk_submaps)):]:
                sid = int(sm.get_id())
                if sid == int(exclude_sid):
                    continue
                idx = int(sm.get_last_non_loop_frame_index())
                v = self._get_keyframe_retrieval_vec(sm, idx)
                if v is not None:
                    cands.append(v)
        except Exception:
            pass

        if len(cands) < 2:
            return 0.0, 1.0, 0.0, 0.0

        C = np.stack(cands, axis=0)  # (N,D) normalized
        sims = C @ q_vec             # cosine sims

        s_sorted = np.sort(sims)[::-1]
        top1 = float(s_sorted[0])
        top2 = float(s_sorted[1])
        margin = float(top1 - top2)

        u = float(self._compute_u(margin))
        return margin, u, top1, top2

    def _compute_u(self, margin: float) -> float:
        """Map margin -> u in [u_min, 1]."""
        if not self.semantic_u_enable:
            return 1.0
        m0 = float(max(1e-6, self.semantic_u_m0))
        u_min = float(np.clip(self.semantic_u_min, 0.0, 1.0))
        try:
            m = float(margin)
        except Exception:
            return 1.0
        if not np.isfinite(m):
            return 1.0
        return float(np.clip(m / m0, u_min, 1.0))

    # -------------------------
    # visualization helpers
    # -------------------------
    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.gradio_mode:
            self.viewer.add_point_cloud(points_in_world_frame, points_colors)
        else:
            self.viewer.server.scene.add_point_cloud(
                name="pcd_" + name,
                points=points_in_world_frame,
                colors=points_colors,
                point_size=point_size,
                point_shape="circle",
            )

    def set_submap_point_cloud(self, submap):
        points_in_world_frame = submap.get_points_in_world_frame(stride=self.vis_stride)
        points_colors = submap.get_points_colors(stride=self.vis_stride)
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, self.vis_point_size)

    def set_submap_poses(self, submap):
        extrinsics = submap.get_all_poses_world()
        if self.gradio_mode:
            for i in range(extrinsics.shape[0]):
                self.viewer.add_camera_pose(extrinsics[i])
        else:
            images = submap.get_all_frames()
            self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def export_3d_scene(self, output_path="output.glb"):
        return self.viewer.export(output_path)

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    # -------------------------
    # core SLAM
    # -------------------------
    def add_points(self, pred_dict):
        """
        pred_dict keys:
            images, extrinsic, intrinsic, detected_loops,
            world_points/world_points_conf or depth/depth_conf
        """
        images = pred_dict["images"]                         # (S, 3, H, W)
        extrinsics_cam = pred_dict["extrinsic"]              # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]              # (S, 3, 3)
        detected_loops = pred_dict.get("detected_loops", [])

        if self.use_point_map:
            world_points = pred_dict["world_points"]         # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]            # (S, H, W)
        else:
            depth_map = pred_dict["depth"]                   # (S, H, W, 1)
            conf = pred_dict["depth_conf"]                   # (S, H, W)
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # (S, 4, 4)

        assert self.current_working_submap is not None
        new_pcd_num = self.current_working_submap.get_id()

        # --- first edge ---
        if self.first_edge:
            self.first_edge = False

            self.prior_pcd = world_points[-1, ...].reshape(-1, 3)
            self.prior_conf = conf[-1, ...].reshape(-1)

            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
            H_relative = np.eye(4)

        # --- subsequent edges ---
        else:
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)

            current_pts = world_points[0, ...].reshape(-1, 3)
            curr_conf = conf[0, ...].reshape(-1)

            thr = float(prior_submap.get_conf_threshold())
            if thr > 1.0:
                pc = self.prior_conf
                cc = curr_conf
                print(
                    f"[DBG][thr>1] {prior_pcd_num}->{new_pcd_num} thr={thr:.6f} "
                    f"prior(min={pc.min():.6f},p99={np.percentile(pc,99):.6f},max={pc.max():.6f}) "
                    f"curr(min={cc.min():.6f},p99={np.percentile(cc,99):.6f},max={cc.max():.6f})"
                )


            good_mask = (self.prior_conf > thr) & (curr_conf > thr)

            min_good = 2000
            n_good0 = int(good_mask.sum())
            fallback_or = 0
            if n_good0 < min_good:
                fallback_or = 1
                print(f"[WARN] good_mask too small ({n_good0} < {min_good}), fallback to OR mask")
                good_mask = (self.prior_conf > thr) | (curr_conf > thr)
            n_good = int(good_mask.sum())

            if self.use_sim3:
                idx = prior_submap.get_last_non_loop_frame_index()
                R_temp = prior_submap.poses[idx][0:3, 0:3]
                t_temp = prior_submap.poses[idx][0:3, 3]

                T_temp = np.eye(4)
                T_temp[0:3, 0:3] = R_temp
                T_temp[0:3, 3] = t_temp
                T_temp = np.linalg.inv(T_temp)

                scale_factor = np.mean(
                    np.linalg.norm((T_temp[0:3, 0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3, 3], axis=1) /
                    (np.linalg.norm(current_pts[good_mask], axis=1) + 1e-8)
                )
                print(colored("scale factor", "green"), scale_factor)

                H_relative = np.eye(4)
                H_relative[0:3, 0:3] = R_temp
                H_relative[0:3, 3] = t_temp

                world_points *= scale_factor
                cam_to_world[:, 0:3, 3] *= scale_factor
            else:
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])

            # Update prior pcd/conf using last non-loop frame of current submap
            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame, ...].reshape(-1, 3)
            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame, ...].reshape(-1)

            # Add node to graph
            self.graph.add_homography(new_pcd_num, np.eye(4))

            # Add between factor (prev -> new)
            use_sem_w_odom = (
                self.semantic_weight_mode == "all_edges"
                and self.use_semantic_backend
                and (self.semantic_backend is not None)
                and hasattr(self.graph, "add_between_factor_weighted")
            )

            if use_sem_w_odom:
                sim = 0.0
                w = 1.0
                margin = 0.0
                u = 1.0
                try:
                    p_idx = prior_submap.get_last_non_loop_frame_index()
                    prior_frame = prior_submap.get_frame_at_index(p_idx)
                    curr_frame0 = self.current_working_submap.get_frame_at_index(0)

                    # semantic sim (DINO metric head)
                    sim = float(self.semantic_backend.similarity(curr_frame0, prior_frame))
                    w_sim = self._semantic_sim_to_weight(sim)

                    # uniqueness (retrieval embedding; anti-aliasing for corridor)
                    q_vec = self._get_keyframe_retrieval_vec(self.current_working_submap, 0)
                    margin, u, top1, top2 = self._compute_uniqueness_from_map(
                        q_vec=q_vec,
                        exclude_sid=int(self.current_working_submap.get_id()),
                    )

                    w = float(w_sim) * float(u)
                    w = self._apply_degeneracy_boost(w, n_good)

                    self.graph.add_between_factor_weighted(prior_pcd_num, new_pcd_num, H_relative, w)
                    print(f"[sem_weight][odom] sim={sim:.3f} margin={margin:.3f} u={u:.2f} n_good={n_good} w={w:.2f} {prior_pcd_num}->{new_pcd_num}")
                except Exception as e:
                    print(f"[WARN] odom semantic weight failed -> fallback: {e}")
                    self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)

                self._edge_stats_log(EdgeStat(
                    time=datetime.now().isoformat(timespec="seconds"),
                    edge_type="odom",
                    src=int(prior_pcd_num),
                    dst=int(new_pcd_num),
                    sim=float(sim),
                    w=float(w),
                    n_good=int(n_good),
                    mask_thr=float(thr),
                    mask_fallback_or=int(fallback_or),
                    weight_mode=str(self.semantic_weight_mode),
                    margin=float(margin),
                    u=float(u),
                ))
            else:
                self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)

            print("added between factor", prior_pcd_num, new_pcd_num)
            H_w_submap = prior_submap.get_reference_homography() @ H_relative

        # --- create/update current submap ---
        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)
        self.current_working_submap.add_all_points(
            world_points,
            colors,
            conf,
            self.init_conf_threshold,
            intrinsics_cam,
        )
        self.current_working_submap.set_conf_masks(conf)

        # --- loop closures (if any) ---
        if detected_loops is not None and len(detected_loops) > 0:
            for index, loop in enumerate(detected_loops):
                if hasattr(loop, "query_submap_id"):
                    assert loop.query_submap_id == self.current_working_submap.get_id()

                loop_index = self.current_working_submap.get_last_non_loop_frame_index() + index + 1

                # ---- long-range loop only (avoid treating overlap as loop) ----
                min_loop_submap_gap = 10
                query_sid = int(getattr(loop, "query_submap_id", self.current_working_submap.get_id()))
                det_sid = int(loop.detected_submap_id)
                if abs(det_sid - query_sid) < min_loop_submap_gap:
                    continue

                if self.use_sim3:
                    pose_world_detected = self.map.get_submap(loop.detected_submap_id).get_pose_subframe(loop.detected_submap_frame)
                    pose_world_query = self.current_working_submap.get_pose_subframe(loop_index)
                    pose_world_detected = gtsam.Pose3(pose_world_detected)
                    pose_world_query = gtsam.Pose3(pose_world_query)
                    H_relative_lc = pose_world_detected.between(pose_world_query).matrix()
                else:
                    points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                    points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                    H_relative_lc = ransac_projective(points_world_query, points_world_detected)

                # 关键：避免重复加边（weighted 就不要再 add normal）
                use_sem_w_loop = (
                    self.semantic_weight_mode in ("loop_only", "all_edges")
                    and self.use_semantic_backend
                    and (self.semantic_backend is not None)
                    and hasattr(self.graph, "add_between_factor_weighted")
                )

                if use_sem_w_loop:
                    sim = 0.0
                    w = 1.0
                    loop_margin = float(getattr(loop, "semantic_margin", 0.0))
                    loop_u = float(getattr(loop, "semantic_u", 1.0))
                    try:
                        sim_cached = getattr(loop, "semantic_sim", None)
                        if sim_cached is None:
                            q_frame = self.current_working_submap.get_frame_at_index(loop_index)
                            d_submap = self.map.get_submap(loop.detected_submap_id)
                            d_frame = d_submap.get_frame_at_index(loop.detected_submap_frame)
                            sim = float(self.semantic_backend.similarity(q_frame, d_frame))
                        else:
                            sim = float(sim_cached)

                        # Down-weight ambiguous loops using the semantic margin-derived
                        # uniqueness factor (u) computed during loop gating.
                        loop_margin = float(getattr(loop, "semantic_margin", 0.0))
                        loop_u = float(getattr(loop, "semantic_u", 1.0))

                        w = float(self._semantic_sim_to_weight(sim)) * float(loop_u)
                        self.graph.add_between_factor_weighted(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, w)
                        print(
                            f"[sem_weight][loop] sim={sim:.3f} margin={loop_margin:.3f} u={loop_u:.2f} w={w:.2f} "
                            f"{loop.detected_submap_id}->{loop.query_submap_id}"
                        )
                    except Exception as e:
                        print(f"[WARN] loop semantic weight failed -> fallback: {e}")
                        self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)

                    self._edge_stats_log(EdgeStat(
                        time=datetime.now().isoformat(timespec="seconds"),
                        edge_type="loop",
                        src=int(loop.detected_submap_id),
                        dst=int(loop.query_submap_id),
                        sim=float(sim),
                        w=float(w),
                        n_good=0,
                        mask_thr=0.0,
                        mask_fallback_or=0,
                        weight_mode=str(self.semantic_weight_mode),
                        margin=float(getattr(loop, "semantic_margin", 0.0)),
                        u=float(getattr(loop, "semantic_u", 1.0)),
                    ))
                else:
                    self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)

                self.graph.increment_loop_closure()
                print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id)

                print(
                    "homography between nodes estimated to be",
                    np.linalg.inv(self.map.get_submap(loop.detected_submap_id).get_reference_homography()) @ H_w_submap,
                )

        # finally, add submap into map
        self.map.add_submap(self.current_working_submap)

    def sample_pixel_coordinates(self, H, W, n):
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    # -------------------------
    # semantic gating
    # -------------------------
    def _filter_loops_by_semantic(self, new_submap, detected_loops):
        """
        Filter detected_loops using semantic similarity between query frame and retrieved frame.
        会把 sim 缓存到 loop.semantic_sim，后续 loop reweighting 直接用。
        """
        if self.semantic_gate_mode not in ("filter", "both"):
            return detected_loops
        if (not self.use_semantic_backend) or (self.semantic_backend is None):
            return detected_loops
        if detected_loops is None or len(detected_loops) == 0:
            return detected_loops

        kept = []
        thr = float(self.semantic_min_sim)

        for loop in detected_loops:
            try:
                q_idx = getattr(loop, "query_submap_frame", None)
                if q_idx is None:
                    q_idx = getattr(loop, "query_frame", None)
                if q_idx is None:
                    q_idx = len(new_submap.get_frame_ids()) - 1

                d_sid = getattr(loop, "detected_submap_id", None)
                d_fid = getattr(loop, "detected_submap_frame", None)
                if (d_sid is None) or (d_fid is None):
                    kept.append(loop)
                    continue

                q_frame = new_submap.get_frame_at_index(int(q_idx))
                d_frame = self.map.get_submap(d_sid).get_frame_at_index(int(d_fid))

                sim = float(self.semantic_backend.similarity(q_frame, d_frame))
                setattr(loop, "semantic_sim", sim)

                if sim >= thr:
                    kept.append(loop)
            except Exception:
                kept.append(loop)

        return kept

    def run_predictions(self, image_names, model, max_loops):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = load_and_preprocess_images(image_names).to(device)
        print(f"Preprocessed images shape: {images.shape}")

        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)

        # retrieval vectors for loop closure
        self._set_submap_attr(new_submap, 'retrieval_vectors', self.image_retrieval.get_all_submap_embeddings(new_submap))

        # IMPORTANT: set before loop closure to keep frame indexing consistent
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)

        detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops)

        # semantic filter on loop candidates
        if self.semantic_gate_mode in ("filter", "both") and len(detected_loops) > 0:
            before = len(detected_loops)
            detected_loops = self._filter_loops_by_semantic(new_submap, detected_loops)
            after = len(detected_loops)
            print(colored("semantic_loop_filter", "cyan"), f"{before}->{after}")

        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)

        retrieved_frames = self.map.get_frames_from_loops(detected_loops)

        # semantic gate on retrieved frames (query frame vs retrieved frame)
        if self.semantic_gate_mode in ("retrieved", "both") and len(detected_loops) > 0 and len(retrieved_frames) > 0:
            try:
                q_ids = new_submap.get_frame_ids()
                q_idx = len(q_ids) - 1
                q_frame = new_submap.get_frame_at_index(q_idx)

                sims = [float(self.semantic_backend.similarity(q_frame, rf)) for rf in retrieved_frames]
                thr = float(self.semantic_min_sim)

                # cache sim to loop object if possible
                for i, sim in enumerate(sims):
                    try:
                        setattr(detected_loops[i], "semantic_sim", float(sim))
                    except Exception:
                        pass

                # 1) threshold filter
                cand = [(i, sims[i]) for i in range(len(sims)) if sims[i] >= thr]
                before = len(detected_loops)
                if len(cand) == 0:
                    detected_loops, retrieved_frames = [], []
                    after = 0
                    print(f"[semantic_loop_gate] min_sim={thr:.3f} {before}->{after} (no candidate) sims={[round(x,3) for x in sims]}")
                else:
                    # 2) rank by semantic sim; compute uniqueness margin using retrieval embeddings
                    cand.sort(key=lambda x: x[1], reverse=True)
                    best_i, best_sim = cand[0]
                    second_sim = cand[1][1] if len(cand) > 1 else -1.0

                    q_vec = self._get_frame_retrieval_vec(new_submap, q_idx)
                    retrieval_sims: List[Tuple[int, float]] = []
                    if q_vec is not None:
                        for i, _ in cand:
                            loop = detected_loops[i]
                            d_sid = getattr(loop, "detected_submap_id", None)
                            d_fid = getattr(loop, "detected_submap_frame", None)
                            if (d_sid is None) or (d_fid is None):
                                continue
                            d_submap = self.map.get_submap(d_sid)
                            d_vec = self._get_frame_retrieval_vec(d_submap, int(d_fid))
                            if d_vec is None:
                                continue
                            retrieval_sims.append((i, float(np.dot(q_vec, d_vec))))

                    margin_valid = False
                    if len(retrieval_sims) >= 2:
                        retrieval_sims.sort(key=lambda x: x[1], reverse=True)
                        top1 = retrieval_sims[0][1]
                        top2 = retrieval_sims[1][1]
                        margin = float(top1 - top2)
                        margin_valid = True
                    elif len(retrieval_sims) == 1:
                        margin = 1.0
                        margin_valid = True
                    else:
                        margin = 0.0

                    # Map margin -> u in (u_min, 1]; small margin => ambiguous => down-weight.
                    if margin_valid:
                        u = float(self._compute_u(margin))
                        if margin < float(self.semantic_loop_margin_thr):
                            u = float(min(u, self.semantic_u_min))
                    else:
                        u = 1.0

                    topk = max(1, int(self.semantic_loop_topk))
                    keep = [i for (i, _) in cand[:topk]]

                    # Attach margin/u to kept loop objects for downstream weighting & logging.
                    for i in keep:
                        try:
                            setattr(detected_loops[i], "semantic_margin", float(margin))
                            setattr(detected_loops[i], "semantic_u", float(u))
                        except Exception:
                            pass

                    detected_loops = [detected_loops[i] for i in keep]
                    retrieved_frames = [retrieved_frames[i] for i in keep]
                    after = len(detected_loops)
                    print(
                        f"[semantic_loop_gate] min_sim={thr:.3f} topk={topk} {before}->{after} "
                        f"best={best_sim:.3f} second={second_sim:.3f} margin={margin:.3f} "
                        f"u={u:.3f} margin_thr={float(self.semantic_loop_margin_thr):.3f} "
                        f"keep={keep} sims_head={[round(x,3) for x in sims[:10]]}"
                    )
            except Exception as e:
                print(f"[WARN] semantic_loop_gate failed (keep loops): {e}")

        num_loop_frames = len(retrieved_frames)

        if num_loop_frames > 0:
            image_tensor = torch.stack(retrieved_frames)      # (n, 3, H, W)
            images = torch.cat([images, image_tensor], dim=0) # (S+n, 3, H, W)
            new_submap.add_all_frames(images)

        self.current_working_submap = new_submap

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        predictions["detected_loops"] = detected_loops

        for key in list(predictions.keys()):
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        return predictions
