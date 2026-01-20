#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_poses_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read poses.txt with format:
      frame_id x y z qx qy qz qw
    Returns:
      frame_ids: (N,)
      xyz: (N,3)
    """
    frame_ids = []
    xyz = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                fid = float(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                continue
            frame_ids.append(fid)
            xyz.append([x, y, z])

    if len(xyz) == 0:
        return np.array([], dtype=float), np.zeros((0, 3), dtype=float)

    return np.array(frame_ids, dtype=float), np.asarray(xyz, dtype=float)


def pca_main_direction(xyz: np.ndarray) -> np.ndarray:
    """
    Compute main direction vector (unit) using PCA (SVD) on 3D positions.
    """
    if xyz.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0], dtype=float)

    center = xyz.mean(axis=0, keepdims=True)
    X = xyz - center
    # SVD: X = U S Vt
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


def lateral_drift_stats(xyz: np.ndarray) -> Dict[str, float]:
    """
    Lateral drift = distance to main axis line (PCA first component).
    For each point p:
      d = || (p - p0) - dot(p - p0, v)*v ||
    Returns rmse/p95/max (meters) + path length etc.
    """
    if xyz.shape[0] < 3:
        return {
            "lateral_rmse_m": float("nan"),
            "lateral_p95_m": float("nan"),
            "lateral_max_m": float("nan"),
            "along_length_m": float("nan"),
        }

    v = pca_main_direction(xyz)
    p0 = xyz[0]
    dp = xyz - p0[None, :]

    # along-track scalar
    s = dp @ v
    # perpendicular component
    perp = dp - s[:, None] * v[None, :]
    lateral = np.linalg.norm(perp, axis=1)

    lateral_rmse = float(np.sqrt(np.mean(lateral ** 2)))
    lateral_p95 = float(np.percentile(lateral, 95))
    lateral_max = float(np.max(lateral))
    along_len = float(np.max(s) - np.min(s))  # extent along main direction

    return {
        "lateral_rmse_m": lateral_rmse,
        "lateral_p95_m": lateral_p95,
        "lateral_max_m": lateral_max,
        "along_length_m": along_len,
    }


def make_plots(xyz: np.ndarray, out_dir: str, prefix: str = "") -> None:
    """
    Save:
      - lateral_profile.png: along-track s vs lateral distance
      - pca_topdown.png: trajectory projected to (PC1, PC2) plane
    """
    if xyz.shape[0] < 3:
        return

    os.makedirs(out_dir, exist_ok=True)

    # PCA basis
    center = xyz.mean(axis=0, keepdims=True)
    X = xyz - center
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    pc1 = vt[0] / (np.linalg.norm(vt[0]) + 1e-12)
    pc2 = vt[1] / (np.linalg.norm(vt[1]) + 1e-12)

    p0 = xyz[0]
    dp = xyz - p0[None, :]

    s = dp @ pc1
    perp = dp - s[:, None] * pc1[None, :]
    lateral = np.linalg.norm(perp, axis=1)

    # Figure 1: lateral profile
    plt.figure(figsize=(8, 4))
    plt.plot(s, lateral, marker="o", linewidth=1)
    plt.xlabel("Along-track (m)  [projection onto PC1]")
    plt.ylabel("Lateral distance (m)  [distance to main axis]")
    plt.title("Lateral drift profile")
    plt.grid(True, alpha=0.3)
    fn1 = os.path.join(out_dir, f"{prefix}lateral_profile.png")
    plt.tight_layout()
    plt.savefig(fn1, dpi=160)
    plt.close()

    # Figure 2: topdown in PCA plane
    x1 = (xyz - center) @ pc1
    x2 = (xyz - center) @ pc2
    plt.figure(figsize=(5, 5))
    plt.plot(x1, x2, marker="o", linewidth=1)
    plt.xlabel("PC1 (m)")
    plt.ylabel("PC2 (m)")
    plt.title("Trajectory in PCA plane (top-down)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    fn2 = os.path.join(out_dir, f"{prefix}pca_topdown.png")
    plt.tight_layout()
    plt.savefig(fn2, dpi=160)
    plt.close()


def eval_single(poses_file: str, out_dir: Optional[str], write_plots: bool) -> Dict[str, float]:
    fids, xyz = read_poses_txt(poses_file)
    stats = lateral_drift_stats(xyz)
    if write_plots and out_dir:
        make_plots(xyz, out_dir=out_dir, prefix="")
    return stats


def update_metrics_csv(metrics_csv: str, out_csv: str, plot: bool) -> None:
    df = pd.read_csv(metrics_csv)

    # ensure columns
    for c in ["lateral_rmse_m", "lateral_p95_m", "lateral_max_m", "along_length_m"]:
        if c not in df.columns:
            df[c] = np.nan

    n = len(df)
    for i in range(n):
        poses_file = str(df.loc[i, "poses_file"]) if "poses_file" in df.columns else ""
        run_dir = str(df.loc[i, "run_dir"]) if "run_dir" in df.columns else ""

        if not poses_file or not os.path.isfile(poses_file):
            continue

        try:
            stats = eval_single(
                poses_file=poses_file,
                out_dir=run_dir if (plot and run_dir) else None,
                write_plots=plot
            )
            for k, v in stats.items():
                df.loc[i, k] = v
        except Exception as e:
            print(f"[WARN] failed on row {i} poses={poses_file}: {e}")
            continue

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] wrote: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=str, default="", help="single poses.txt to evaluate")
    ap.add_argument("--metrics_csv", type=str, default="", help="batch update: corridor_compare_metrics.csv")
    ap.add_argument("--out_csv", type=str, default="", help="output csv (for batch mode)")
    ap.add_argument("--plot", action="store_true", help="save plots into run_dir (batch) or out_dir (single)")
    ap.add_argument("--out_dir", type=str, default="", help="single mode: output dir for plots")
    args = ap.parse_args()

    if args.poses:
        out_dir = args.out_dir if args.out_dir else (os.path.dirname(args.poses) or ".")
        stats = eval_single(args.poses, out_dir=out_dir, write_plots=args.plot)
        print("[LATERAL]")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if args.plot:
            print(f"[PLOTS] saved into: {out_dir}")
        return

    if args.metrics_csv:
        out_csv = args.out_csv if args.out_csv else args.metrics_csv.replace(".csv", "_plus.csv")
        update_metrics_csv(metrics_csv=args.metrics_csv, out_csv=out_csv, plot=args.plot)
        return

    raise SystemExit("Provide --poses or --metrics_csv")


if __name__ == "__main__":
    main()
