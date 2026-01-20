import re
import os
import cv2
import torch
import numpy as np
import open3d as o3d


class Submap:
    def __init__(self, submap_id):
        self.submap_id = submap_id

        # Geometry / pose
        self.H_world_map = None            # (4,4)
        self.R_world_map = None
        self.poses = None                  # (N,4,4)

        # Frame-level data
        self.frames = None                 # (N,H,W,3)
        self.frame_ids = None
        self.last_non_loop_frame_index = None

        # Camera / intrinsics
        self.vggt_intrinscs = None          # (3,4) or (3,3)

        # Semantic / retrieval
        self.retrieval_vectors = None

        # Dense geometry
        self.pointclouds = None             # (N,H,W,3)
        self.colors = None                  # (N,H,W,3)
        self.conf = None                    # (N,H,W)
        self.conf_masks = None              # (N,H,W)
        self.conf_threshold = None

        # Cached
        self.voxelized_points = None

    # ------------------------------------------------------------------
    # Basic setters
    # ------------------------------------------------------------------

    def add_all_poses(self, poses):
        self.poses = poses

    def add_all_frames(self, frames):
        self.frames = frames

    def set_all_retrieval_vectors(self, retrieval_vectors):
    # 兼容 solver.py 里调用的旧/新接口命名
        self.add_all_retrieval_vectors(retrieval_vectors)

    def get_all_retrieval_vectors(self):
        return self.retrieval_vectors


    def add_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors

    def add_all_points(self, points, colors, conf,
                       conf_threshold_percentile,
                       intrinsics):
        """
        points: (N,H,W,3)
        colors: (N,H,W,3)
        conf:   (N,H,W)
        """
        self.pointclouds = points
        self.colors = colors
        self.conf = conf
        self.conf_threshold = np.percentile(conf, conf_threshold_percentile)
        self.vggt_intrinscs = intrinsics

    # ------------------------------------------------------------------
    # ID / meta
    # ------------------------------------------------------------------

    def get_id(self):
        return self.submap_id

    def set_frame_ids(self, file_paths):
        """
        Extract numeric frame ids from filenames.
        Example: frame_00321.jpg -> 321
        """
        frame_ids = []
        for path in file_paths:
            name = os.path.basename(path)
            match = re.search(r"\d+(?:\.\d+)?", name)
            if match:
                frame_ids.append(float(match.group()))
            else:
                raise ValueError(f"No number found in filename: {name}")
        self.frame_ids = frame_ids

    def get_frame_ids(self):
        return self.frame_ids

    def set_last_non_loop_frame_index(self, idx):
        self.last_non_loop_frame_index = idx

    def get_last_non_loop_frame_index(self):
        return self.last_non_loop_frame_index

    # ------------------------------------------------------------------
    # Reference transform
    # ------------------------------------------------------------------

    def set_reference_homography(self, H_world_map):
        self.H_world_map = H_world_map

    def get_reference_homography(self):
        return self.H_world_map

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame_at_index(self, index):
        return self.frames[index]

    def get_all_frames(self):
        return self.frames

    # ------------------------------------------------------------------
    # Pose utilities
    # ------------------------------------------------------------------

    def get_pose_subframe(self, pose_index):
        return np.linalg.inv(self.poses[pose_index])

    def get_all_poses_world(self, ignore_loop_closure_frames=False):
        """
        Recover camera poses in world frame using projection decomposition.
        """
        proj_mats = (
            self.vggt_intrinscs
            @ np.linalg.inv(self.poses)[:, 0:3, :]
            @ np.linalg.inv(self.H_world_map)
        )

        poses_world = []
        for i, P in enumerate(proj_mats):
            K, R, t = cv2.decomposeProjectionMatrix(P)[0:3]
            t = t / t[3, 0]

            T = np.eye(4)
            T[0:3, 0:3] = np.linalg.inv(R)
            T[0:3, 3] = t[0:3, 0]

            poses_world.append(T)

            if ignore_loop_closure_frames and i == self.last_non_loop_frame_index:
                break

        return np.stack(poses_world, axis=0)

    # ------------------------------------------------------------------
    # Confidence filtering
    # ------------------------------------------------------------------

    def set_conf_masks(self, conf_masks):
        self.conf_masks = conf_masks

    def get_conf_threshold(self):
        return self.conf_threshold

    def filter_data_by_confidence(self, data, stride=1):
        """
        Generic confidence filter for points / colors.
        """
        if stride == 1:
            mask = self.conf >= self.conf_threshold
            return data[mask]
        else:
            conf_sub = self.conf[:, ::stride, ::stride]
            data_sub = data[:, ::stride, ::stride, :]
            mask = conf_sub >= self.conf_threshold
            return data_sub[mask]

    # ------------------------------------------------------------------
    # Point cloud access
    # ------------------------------------------------------------------

    def get_frame_pointcloud(self, pose_index):
        return self.pointclouds[pose_index]

    def get_points_colors(self, stride=1):
        colors = self.filter_data_by_confidence(self.colors, stride)
        return colors.reshape(-1, 3)

    def get_points_in_world_frame(self, stride=1):
        points = self.filter_data_by_confidence(self.pointclouds, stride)

        pts = points.reshape(-1, 3)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])

        pts_w = (self.H_world_map @ pts_h.T).T
        return pts_w[:, :3] / pts_w[:, 3:]

    def get_points_list_in_world_frame(self, ignore_loop_closure_frames=False):
        """
        Return per-frame world points + frame ids + confidence masks
        """
        point_list = []
        frame_id_list = []
        frame_conf_mask = []

        for i, pts in enumerate(self.pointclouds):
            pts_flat = pts.reshape(-1, 3)
            pts_h = np.hstack([pts_flat, np.ones((pts_flat.shape[0], 1))])
            pts_w = (self.H_world_map @ pts_h.T).T
            pts_w = (pts_w[:, :3] / pts_w[:, 3:]).reshape(pts.shape)

            point_list.append(pts_w)
            frame_id_list.append(self.frame_ids[i])
            frame_conf_mask.append(self.conf_masks[i] >= self.conf_threshold)

            if ignore_loop_closure_frames and i == self.last_non_loop_frame_index:
                break

        return point_list, frame_id_list, frame_conf_mask

    # ------------------------------------------------------------------
    # Voxelized point cloud (Open3D)
    # ------------------------------------------------------------------

    def get_voxel_points_in_world_frame(
        self,
        voxel_size,
        nb_points=8,
        factor_for_outlier_rejection=2.0,
    ):
        if self.voxelized_points is None:
            if voxel_size <= 0:
                raise RuntimeError("voxel_size must be > 0")

            pts = self.filter_data_by_confidence(self.pointclouds)
            cols = self.filter_data_by_confidence(self.colors)

            pts = pts.reshape(-1, 3)
            cols = (cols.reshape(-1, 3)) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)

            pcd = pcd.voxel_down_sample(voxel_size)

            if nb_points > 0:
                pcd, _ = pcd.remove_radius_outlier(
                    nb_points=nb_points,
                    radius=voxel_size * factor_for_outlier_rejection,
                )

            self.voxelized_points = pcd

        pts = np.asarray(self.voxelized_points.points)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_w = (self.H_world_map @ pts_h.T).T

        pcd_w = o3d.geometry.PointCloud()
        pcd_w.points = o3d.utility.Vector3dVector(
            pts_w[:, :3] / pts_w[:, 3:]
        )
        pcd_w.colors = self.voxelized_points.colors

        return pcd_w
