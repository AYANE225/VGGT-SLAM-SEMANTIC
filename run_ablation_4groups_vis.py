# run_ablation_4groups_vis.py
# 用法示例：
# python run_ablation_4groups_vis.py \
#   --data /media/omnisky/18/cst1/project/VGGT-SLAM/DATA/corridor_0300-0399 \
#   --out_dir /media/omnisky/18/cst1/project/VGGT-SLAM/RUNS/ablation_20260116 \
#   --submap_size 16 --max_loops 1 \
#   --min_disparity 50 --conf_threshold 25 \
#   --semantic_backend_cfg /path/to/semantic_backend.yaml \
#   --semantic_min_sim 0.25
#
# 产物：
# out_dir/
#   all_runs.csv
#   edge_stats_all.csv
#   baseline/ edge_stats.csv poses.npy run_meta.json
#   gate_only/ ...
#   weight_only/ ...
#   gate_weight/ ...
#   images/ (可选：输入样例图)

from __future__ import annotations

import os
import re
import csv
import time
import json
import glob
import shutil
import zipfile
import argparse
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import cv2

from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT


# -----------------------------
# utils: list & sort images
# -----------------------------
def _extract_numeric_key(p: str) -> Tuple:
    name = os.path.basename(p)
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    if m:
        return (0, float(m.group(1)), name)
    return (1, name)


def list_images(data_path: str, tmp_dir: str) -> List[str]:
    """
    data_path 支持：
      - 文件夹：递归找 jpg/png/jpeg
      - zip：解压到 tmp_dir 后递归找图
    """
    p = Path(data_path)
    if p.is_file() and p.suffix.lower() == ".zip":
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        with zipfile.ZipFile(str(p), "r") as zf:
            zf.extractall(tmp_dir)
        root = tmp_dir
    else:
        root = str(p)

    imgs = [
        f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and "depth" not in os.path.basename(f).lower()
    ]
    imgs.sort(key=_extract_numeric_key)
    return imgs


def save_sample_images(image_paths: List[str], out_dir: str, max_samples: int = 8) -> None:
    """
    将输入序列中的部分图像保存到输出目录，便于可视化对比。
    """
    if not image_paths:
        return
    os.makedirs(out_dir, exist_ok=True)
    take = min(max_samples, len(image_paths))
    stride = max(1, len(image_paths) // take)
    picked = image_paths[::stride][:take]
    for idx, path in enumerate(picked):
        try:
            img = Image.open(path).convert("RGB")
            img.save(os.path.join(out_dir, f"sample_{idx:02d}.png"))
        except Exception:
            continue


# -----------------------------
# trajectory & metrics
# -----------------------------
def collect_trajectory_from_map(solver: Solver) -> Tuple[np.ndarray, List[float]]:
    """
    收集所有 submap 的非 loop 帧世界位姿（4x4），按 frame_id 排序并去重。
    """
    submaps = solver.map.get_submaps()

    id2pose: Dict[float, np.ndarray] = {}
    id_list: List[float] = []

    for sm in submaps:
        poses_world = sm.get_all_poses_world(ignore_loop_closure_frames=True)  # (K,4,4)
        fids = sm.get_frame_ids()  # 不含 loop 帧

        if fids is None:
            continue

        K = min(len(fids), poses_world.shape[0])
        for i in range(K):
            fid = float(fids[i])
            if fid not in id2pose:
                id2pose[fid] = poses_world[i].copy()
                id_list.append(fid)

    id_list.sort()
    poses = np.stack([id2pose[fid] for fid in id_list], axis=0) if len(id_list) > 0 else np.zeros((0, 4, 4))
    return poses, id_list


def compute_pose_metrics(poses: np.ndarray) -> Dict[str, float]:
    """
    无 GT 的稳定可比较指标：
      - path_length_m：相邻位姿平移增量之和
      - loop_trans_err_m：起点到终点位移（回环序列越小越好；直行序列接近 path_length 很正常）
      - drift_ratio：loop_trans_err_m / path_length_m（回环序列越小越好）
    """
    n = int(poses.shape[0])
    if n < 2:
        return dict(
            n_poses=float(n),
            path_length_m=0.0,
            loop_trans_err_m=0.0,
            drift_ratio=0.0,
            step_trans_mean_m=0.0,
            step_trans_p95_m=0.0,
            step_trans_max_m=0.0,
        )

    t = poses[:, 0:3, 3]
    steps = np.linalg.norm(t[1:] - t[:-1], axis=1)
    path_len = float(np.sum(steps))
    loop_err = float(np.linalg.norm(t[-1] - t[0]))
    drift = float(loop_err / (path_len + 1e-9))

    return dict(
        n_poses=float(n),
        path_length_m=path_len,
        loop_trans_err_m=loop_err,
        drift_ratio=drift,
        step_trans_mean_m=float(np.mean(steps)),
        step_trans_p95_m=float(np.percentile(steps, 95)),
        step_trans_max_m=float(np.max(steps)),
    )


# -----------------------------
# edge_stats summarization
# -----------------------------
def _safe_float_series(values: List) -> np.ndarray:
    out = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def summarize_edge_stats(edge_stats_csv: str) -> Dict[str, float]:
    """
    从 edge_stats.csv 提取“论文级”统计，直接写进 all_runs.csv，便于对比。
    字段来自你 solver.py 的 EdgeStat：
      time,edge_type,src,dst,sim,w,n_good,mask_thr,mask_fallback_or,weight_mode,margin,u
    """
    p = Path(edge_stats_csv)
    if (not p.exists()) or p.stat().st_size == 0:
        return {}

    try:
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return {}

    if len(rows) == 0:
        return {}

    out: Dict[str, float] = {}

    # 全部边数量
    out["n_edges_all"] = float(len(rows))

    def collect(edge_type: str) -> List[Dict]:
        return [r for r in rows if str(r.get("edge_type", "")).strip() == edge_type]

    for et in ["loop", "odom"]:
        d = collect(et)
        out[f"n_edges_{et}"] = float(len(d))

        if len(d) == 0:
            continue

        sim = _safe_float_series([r.get("sim", np.nan) for r in d])
        w = _safe_float_series([r.get("w", np.nan) for r in d])

        out[f"{et}_sim_mean"] = float(np.nanmean(sim))
        out[f"{et}_sim_p50"] = float(np.nanmedian(sim))
        out[f"{et}_w_mean"] = float(np.nanmean(w))
        out[f"{et}_w_min"] = float(np.nanmin(w))
        out[f"{et}_w_max"] = float(np.nanmax(w))

        # corridor anti-aliasing stats（如果有）
        if "margin" in d[0]:
            margin = _safe_float_series([r.get("margin", np.nan) for r in d])
            out[f"{et}_margin_mean"] = float(np.nanmean(margin))
            out[f"{et}_margin_p50"] = float(np.nanmedian(margin))
        if "u" in d[0]:
            u = _safe_float_series([r.get("u", np.nan) for r in d])
            out[f"{et}_u_mean"] = float(np.nanmean(u))
            out[f"{et}_u_p50"] = float(np.nanmedian(u))

        # 仅 odom：fallback_or 比例（几何退化信号）
        if et == "odom" and ("mask_fallback_or" in d[0]):
            fb = _safe_float_series([r.get("mask_fallback_or", 0) for r in d])
            out["odom_fallback_or_rate"] = float(np.mean(fb > 0))

        # 仅 odom：n_good（有效点数）统计（可选但很有用）
        if et == "odom" and ("n_good" in d[0]):
            ng = _safe_float_series([r.get("n_good", np.nan) for r in d])
            out["odom_n_good_mean"] = float(np.nanmean(ng))
            out["odom_n_good_p50"] = float(np.nanmedian(ng))

    return out


# -----------------------------
# core run
# -----------------------------
def build_solver(
    conf_threshold: float,
    use_point_map: bool,
    use_sim3: bool,
    use_semantic_backend: bool,
    semantic_backend_cfg: str,
    semantic_gate_mode: str,
    semantic_weight_mode: str,
    semantic_min_sim: float,
    edge_stats_path: str,
) -> Solver:
    solver = Solver(
        init_conf_threshold=float(conf_threshold),
        use_point_map=bool(use_point_map),
        use_sim3=bool(use_sim3),

        # 自动跑实验：避免 viser server 冲突/开端口
        gradio_mode=True,

        # semantic backend
        use_semantic_backend=bool(use_semantic_backend),
        semantic_backend_cfg=str(semantic_backend_cfg),
        semantic_min_sim=float(semantic_min_sim),

        # gate / weight
        semantic_gate_mode=str(semantic_gate_mode),
        semantic_weight_mode=str(semantic_weight_mode),

        # stats
        edge_stats_path=str(edge_stats_path),
    )
    return solver


def load_model(device: str, model_ckpt: str = "") -> torch.nn.Module:
    model = VGGT()
    if model_ckpt and os.path.exists(model_ckpt):
        sd = torch.load(model_ckpt, map_location="cpu")
        model.load_state_dict(sd)
    else:
        # 与 app.py 一致：从 URL 拉权重
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        sd = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
        model.load_state_dict(sd)

    model.eval()
    model = model.to(device)
    return model


def warmup_model(model: torch.nn.Module, device: str):
    """
    可选：做一次短 warmup，减少 baseline first-run 慢导致的 seconds 偏差。
    """
    try:
        dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=torch.float16)
        with torch.no_grad():
            _ = model(dummy)
    except Exception:
        pass


def run_one_group(
    group_name: str,
    image_paths: List[str],
    out_dir: str,
    model: torch.nn.Module,
    use_optical_flow_downsample: bool,
    min_disparity: float,
    submap_size: int,
    max_loops: int,
    conf_threshold: float,
    use_point_map: bool,
    use_sim3: bool,
    semantic_backend_cfg: str,
    semantic_min_sim: float,
    semantic_gate_mode: str,
    semantic_weight_mode: str,
    use_semantic_backend: bool,
    dataset_id: str,
    cmd_str: str,
) -> Dict[str, object]:

    gdir = Path(out_dir) / group_name
    gdir.mkdir(parents=True, exist_ok=True)

    edge_stats_path = str(gdir / "edge_stats.csv")
    poses_path = str(gdir / "poses.npy")
    meta_path = str(gdir / "run_meta.json")

    solver = build_solver(
        conf_threshold=conf_threshold,
        use_point_map=use_point_map,
        use_sim3=use_sim3,
        use_semantic_backend=use_semantic_backend,
        semantic_backend_cfg=semantic_backend_cfg,
        semantic_gate_mode=semantic_gate_mode,
        semantic_weight_mode=semantic_weight_mode,
        semantic_min_sim=semantic_min_sim,
        edge_stats_path=edge_stats_path,
    )

    subset: List[str] = []
    start = time.time()

    n_submaps = 0
    n_opt = 0
    n_selected = 0

    for idx, img_path in enumerate(image_paths):
        if use_optical_flow_downsample:
            img = cv2.imread(img_path)
            enough = solver.flow_tracker.compute_disparity(img, float(min_disparity), False)
            if enough:
                subset.append(img_path)
                n_selected += 1
        else:
            subset.append(img_path)
            n_selected += 1

        is_last = (idx == len(image_paths) - 1)
        if len(subset) == (int(submap_size) + 1) or is_last:
            if len(subset) < 2:
                subset = subset[-1:]
                continue

            predictions = solver.run_predictions(subset, model, int(max_loops))
            solver.add_points(predictions)
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            n_submaps += 1
            n_opt += 1

            # overlap：保留最后一帧
            subset = subset[-1:]

    seconds = float(time.time() - start)

    poses, frame_ids = collect_trajectory_from_map(solver)
    np.save(poses_path, poses)

    metrics = compute_pose_metrics(poses)
    edge_summary = summarize_edge_stats(edge_stats_path)

    # 关闭 edge_stats 文件句柄（建议）
    try:
        fp = getattr(solver, "_edge_stats_fp", None)
        if fp is not None:
            fp.close()
    except Exception:
        pass

    meta = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "group": group_name,
        "dataset": dataset_id,
        "n_images_total": int(len(image_paths)),
        "n_images_selected": int(n_selected),
        "n_submaps": int(n_submaps),
        "seconds": seconds,
        "poses_path": poses_path,
        "edge_stats_path": edge_stats_path,
        "cmd": cmd_str,
        "semantic": {
            "use_semantic_backend": bool(use_semantic_backend),
            "semantic_backend_cfg": str(semantic_backend_cfg),
            "semantic_min_sim": float(semantic_min_sim),
            "semantic_gate_mode": str(semantic_gate_mode),
            "semantic_weight_mode": str(semantic_weight_mode),
        },
        "metrics": metrics,
        "edge_summary": edge_summary,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    row: Dict[str, object] = {
        "time": meta["time"],
        "group": group_name,
        "returncode": 0,
        "seconds": seconds,
        "dataset": dataset_id,
        "data": str(Path(out_dir).resolve()),
        "run_dir": str(gdir.resolve()),
        "poses_file": poses_path,
        "edge_stats_file": edge_stats_path,
        "n_images_total": int(len(image_paths)),
        "n_images_selected": int(n_selected),

        **metrics,

        "semantic_gate_mode": str(semantic_gate_mode),
        "semantic_weight_mode": str(semantic_weight_mode),
        "use_semantic_backend": int(bool(use_semantic_backend)),

        # 关键：把 loop/odom 的证据直接写进 all_runs
        **edge_summary,

        "cmd": cmd_str,
    }
    return row


def merge_edge_stats(out_dir: str, groups: List[str]) -> str:
    """
    合并各组 edge_stats.csv -> edge_stats_all.csv，并加 group 列
    """
    out_path = str(Path(out_dir) / "edge_stats_all.csv")
    rows = []
    header: Optional[List[str]] = None

    for g in groups:
        p = Path(out_dir) / g / "edge_stats.csv"
        if (not p.exists()) or p.stat().st_size == 0:
            continue
        with p.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if header is None:
                header = ["group"] + list(r.fieldnames or [])
            for item in r:
                item2 = {"group": g}
                item2.update(item)
                rows.append(item2)

    if header is None:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return out_path

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for it in rows:
            w.writerow(it)

    return out_path


def write_all_runs_csv(out_dir: str, rows: List[Dict[str, object]]) -> str:
    """
    写 all_runs.csv：字段用“所有 row 的 key 并集”，保证不会丢字段。
    """
    out_path = str(Path(out_dir) / "all_runs.csv")
    keys = []
    seen = set()
    # 让重要字段排前面
    preferred = [
        "time","group","returncode","seconds","dataset","data","run_dir",
        "poses_file","edge_stats_file","n_images_total","n_images_selected",
        "n_poses","path_length_m","loop_trans_err_m","drift_ratio",
        "step_trans_mean_m","step_trans_p95_m","step_trans_max_m",
        "semantic_gate_mode","semantic_weight_mode","use_semantic_backend",
        "n_edges_all","n_edges_odom","n_edges_loop",
        "loop_sim_mean","loop_sim_p50","loop_w_mean","loop_w_min","loop_w_max",
        "odom_sim_mean","odom_sim_p50","odom_w_mean","odom_w_min","odom_w_max",
        "odom_fallback_or_rate","odom_n_good_mean","odom_n_good_p50",
        "odom_margin_mean","odom_margin_p50","odom_u_mean","odom_u_p50",
        "loop_margin_mean","loop_margin_p50","loop_u_mean","loop_u_p50",
        "cmd",
    ]
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k); seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return out_path


def build_cmd_str(args: argparse.Namespace) -> str:
    # 记录本次运行命令（方便复现实验）
    kv = vars(args).copy()
    # 删掉可能很长的东西也行，这里保留
    return "python run_ablation_4groups_vis.py " + " ".join([f"--{k} {kv[k]}" if not isinstance(kv[k], bool) else (f"--{k}" if kv[k] else "") for k in kv]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="image folder OR .zip")
    ap.add_argument("--out_dir", type=str, required=True, help="output directory")
    ap.add_argument("--dataset_id", type=str, default="", help="写进CSV的dataset标识（不填则用 data 路径）")

    ap.add_argument("--submap_size", type=int, default=16)
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--min_disparity", type=float, default=50.0)
    ap.add_argument("--conf_threshold", type=float, default=25.0)
    ap.add_argument("--use_point_map", type=int, default=0)
    ap.add_argument("--use_sim3", type=int, default=0)
    ap.add_argument("--no_flow_downsample", action="store_true")

    ap.add_argument("--semantic_backend_cfg", type=str, default="", help="semantic backend cfg path")
    ap.add_argument("--semantic_min_sim", type=float, default=0.25)
    ap.add_argument("--save_images", action="store_true", help="保存输入样例图到输出目录")

    ap.add_argument("--model_ckpt", type=str, default="", help="local model.pt (optional)")
    ap.add_argument("--warmup", action="store_true", help="do a tiny model warmup to reduce first-run bias")
    args = ap.parse_args()

    out_dir = str(Path(args.out_dir).resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    dataset_id = args.dataset_id.strip() if args.dataset_id.strip() else str(Path(args.data).resolve())
    tmp_dir = str(Path(out_dir) / "_tmp_images")

    image_paths = list_images(args.data, tmp_dir=tmp_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found from: {args.data}")

    if args.save_images:
        save_sample_images(image_paths, out_dir=os.path.join(out_dir, "images"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device, model_ckpt=args.model_ckpt)

    if args.warmup:
        warmup_model(model, device=device)

    # 四组定义（固定）
    groups = [
        ("baseline",     dict(use_semantic_backend=False, semantic_gate_mode="off",  semantic_weight_mode="off")),
        ("gate_only",    dict(use_semantic_backend=True,  semantic_gate_mode="both", semantic_weight_mode="off")),
        ("weight_only",  dict(use_semantic_backend=True,  semantic_gate_mode="off",  semantic_weight_mode="loop_only")),
        ("gate_weight",  dict(use_semantic_backend=True,  semantic_gate_mode="both", semantic_weight_mode="loop_only")),
    ]

    cmd_str = build_cmd_str(args)

    rows: List[Dict[str, object]] = []
    for gname, cfg in groups:
        print(f"\n==================== RUN {gname} ====================\n")
        try:
            row = run_one_group(
                group_name=gname,
                image_paths=image_paths,
                out_dir=out_dir,
                model=model,
                use_optical_flow_downsample=(not args.no_flow_downsample),
                min_disparity=float(args.min_disparity),
                submap_size=int(args.submap_size),
                max_loops=int(args.max_loops),
                conf_threshold=float(args.conf_threshold),
                use_point_map=bool(int(args.use_point_map)),
                use_sim3=bool(int(args.use_sim3)),
                semantic_backend_cfg=str(args.semantic_backend_cfg),
                semantic_min_sim=float(args.semantic_min_sim),
                semantic_gate_mode=str(cfg["semantic_gate_mode"]),
                semantic_weight_mode=str(cfg["semantic_weight_mode"]),
                use_semantic_backend=bool(cfg["use_semantic_backend"]),
                dataset_id=dataset_id,
                cmd_str=cmd_str,
            )
        except Exception as e:
            # 不中断其它组，写一行失败记录
            row = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "group": gname,
                "returncode": 1,
                "seconds": 0.0,
                "dataset": dataset_id,
                "data": str(Path(out_dir).resolve()),
                "run_dir": str((Path(out_dir) / gname).resolve()),
                "poses_file": "",
                "edge_stats_file": "",
                "error": repr(e),
                "semantic_gate_mode": str(cfg["semantic_gate_mode"]),
                "semantic_weight_mode": str(cfg["semantic_weight_mode"]),
                "use_semantic_backend": int(bool(cfg["use_semantic_backend"])),
                "cmd": cmd_str,
            }
            print(f"[ERROR] group={gname} failed: {e}")

        rows.append(row)

    all_runs = write_all_runs_csv(out_dir, rows)
    edge_all = merge_edge_stats(out_dir, [g for g, _ in groups])

    print("\n==================== DONE ====================")
    print("all_runs.csv     :", all_runs)
    print("edge_stats_all.csv:", edge_all)


if __name__ == "__main__":
    main()
