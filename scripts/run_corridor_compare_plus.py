#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, os, re, shlex, sys, time, subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

# ---------------------------
# Utils
# ---------------------------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def repo_root_from_script() -> Path:
    # scripts/xxx.py -> repo root
    return Path(__file__).resolve().parents[1]

def safe_float(x: str):
    try:
        return float(x)
    except Exception:
        return None

def stream_subprocess(cmd, cwd: Path, log_path: Path, env: Dict[str, str]) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.write(f"[CWD] {cwd}\n")
        f.write(f"[TIME] {datetime.now().isoformat()}\n")
        f.write("-"*80 + "\n")
        f.flush()

        proc = subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True, env=env
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line); f.flush()
        return proc.wait()

# ---------------------------
# Trajectory parsing + metrics
# poses.txt format (from map.write_poses_to_file):
# frame_id x y z qx qy qz qw
# ---------------------------
@dataclass
class Traj:
    t: np.ndarray
    p: np.ndarray
    q: Optional[np.ndarray] = None  # xyzw

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n

def quat_conj(q: np.ndarray) -> np.ndarray:
    qc = q.copy()
    qc[..., :3] *= -1.0
    return qc

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[...,0], a[...,1], a[...,2], a[...,3]
    bx, by, bz, bw = b[...,0], b[...,1], b[...,2], b[...,3]
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return np.stack([x,y,z,w], axis=-1)

def quat_to_rot_angle_deg(q_rel: np.ndarray) -> np.ndarray:
    q_rel = quat_normalize(q_rel)
    w = np.clip(q_rel[...,3], -1.0, 1.0)
    ang = 2.0 * np.arccos(w)
    return np.degrees(ang)

def parse_poses_txt(path: Path) -> Optional[Traj]:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"[,\s]+", s)
        vals = [safe_float(x) for x in parts]
        if any(v is None for v in vals):
            continue
        rows.append([float(v) for v in vals])
    if len(rows) < 2:
        return None
    arr = np.array(rows, dtype=np.float64)
    if arr.shape[1] >= 8:
        t = arr[:,0]
        p = arr[:,1:4]
        q = arr[:,4:8]  # xyzw
        return Traj(t=t, p=p, q=q)
    if arr.shape[1] == 4:
        return Traj(t=arr[:,0], p=arr[:,1:4], q=None)
    return None

def compute_metrics(traj: Traj, jump_trans_thresh: float = 0.5, jump_rot_thresh_deg: float = 30.0) -> Dict[str, object]:
    p = traj.p
    N = p.shape[0]
    out = {
        "n_poses": int(N),
        "path_length_m": "",
        "loop_trans_err_m": "",
        "loop_rot_err_deg": "",
        "drift_ratio": "",
        "step_trans_mean_m": "",
        "step_trans_p95_m": "",
        "step_trans_max_m": "",
        "step_rot_mean_deg": "",
        "step_rot_p95_deg": "",
        "step_rot_max_deg": "",
        "jump_trans_count": "",
        "jump_rot_count": "",
    }
    if N < 2:
        return out

    dp = p[1:] - p[:-1]
    step_trans = np.linalg.norm(dp, axis=1)
    path_len = float(step_trans.sum())
    loop_trans = float(np.linalg.norm(p[-1] - p[0]))

    out["path_length_m"] = path_len
    out["loop_trans_err_m"] = loop_trans
    out["drift_ratio"] = (loop_trans / (path_len + 1e-12)) if path_len > 0 else ""

    out["step_trans_mean_m"] = float(np.mean(step_trans))
    out["step_trans_p95_m"]  = float(np.percentile(step_trans, 95))
    out["step_trans_max_m"]  = float(np.max(step_trans))
    out["jump_trans_count"]  = int(np.sum(step_trans > jump_trans_thresh))

    if traj.q is not None and traj.q.shape[0] == N:
        q = quat_normalize(traj.q)
        q_rel = quat_mul(q[1:], quat_conj(q[:-1]))
        step_rot = quat_to_rot_angle_deg(q_rel)
        q_loop = quat_mul(q[-1], quat_conj(q[0]))
        out["loop_rot_err_deg"] = float(quat_to_rot_angle_deg(q_loop)[()])

        out["step_rot_mean_deg"] = float(np.mean(step_rot))
        out["step_rot_p95_deg"]  = float(np.percentile(step_rot, 95))
        out["step_rot_max_deg"]  = float(np.max(step_rot))
        out["jump_rot_count"]    = int(np.sum(step_rot > jump_rot_thresh_deg))

    return out

def parse_submaps_and_loops_from_log(log_file: Path) -> Tuple[str, str]:
    """
    Parse:
      Total number of submaps in map X
      Total number of loop closures in map Y
    """
    if not log_file.exists():
        return "", ""
    txt = log_file.read_text(encoding="utf-8", errors="ignore")
    m1 = re.findall(r"Total number of submaps in map\s+(\d+)", txt)
    m2 = re.findall(r"Total number of loop closures in map\s+(\d+)", txt)
    n_submaps = m1[-1] if m1 else ""
    n_loops   = m2[-1] if m2 else ""
    return n_submaps, n_loops

def append_csv(csv_path: Path, row: Dict[str, object], fieldnames):
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})

# ---------------------------
# Main runner
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default="", help="VGGT-SLAM repo root (default: auto)")
    ap.add_argument("--main_py", type=str, default="main.py")
    ap.add_argument("--dataset", type=str, default="corridor_0300-0399", help="Only run this dataset folder under DATA/")
    ap.add_argument("--tag", type=str, default="corridor_sem_vs_base", help="RUNS/<tag>/..., LOG/<tag>__*.log")
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--semantic_min_sim", type=float, default=None, help="if provided, pass --semantic_min_sim X to main.py")
    ap.add_argument("--disable_loop_closure", action="store_true", help="pass --disable_loop_closure to main.py")
    ap.add_argument("--jump_trans_thresh", type=float, default=0.5)
    ap.add_argument("--jump_rot_thresh_deg", type=float, default=30.0)
    ap.add_argument("--metrics_csv", type=str, default="LOG/corridor_compare_metrics.csv")

    # pass-through for extra main.py args: use "-- <args...>"
    ap.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    main_py = (repo_root / args.main_py).resolve()
    if not main_py.exists():
        print(f"[ERROR] main.py not found: {main_py}", file=sys.stderr)
        sys.exit(2)

    data_dir = (repo_root / "DATA" / args.dataset).resolve()
    if not data_dir.exists():
        print(f"[ERROR] dataset not found: {data_dir}", file=sys.stderr)
        sys.exit(2)

    log_dir = (repo_root / "LOG").resolve(); ensure_dir(log_dir)
    run_root = (repo_root / "RUNS" / args.tag).resolve(); ensure_dir(run_root)
    metrics_csv = (repo_root / args.metrics_csv).resolve()

    # count images (rough)
    n_images = len([p for p in data_dir.glob("*") if p.is_file() and p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    passthrough = args.passthrough
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    # two modes only
    modes = [
        ("baseline", []),
        ("semantic", ["--use_semantic_backend"]),
    ]

    common_extra = ["--max_loops", str(args.max_loops)]
    if args.semantic_min_sim is not None:
        common_extra += ["--semantic_min_sim", str(args.semantic_min_sim)]
    if args.disable_loop_closure:
        common_extra += ["--disable_loop_closure"]

    fieldnames = [
        "time","tag","mode","returncode","seconds",
        "dataset","n_images",
        "run_dir","log_file","poses_file",
        "n_submaps","n_loops",
        "n_poses","path_length_m","loop_trans_err_m","loop_rot_err_deg","drift_ratio",
        "step_trans_mean_m","step_trans_p95_m","step_trans_max_m",
        "step_rot_mean_deg","step_rot_p95_deg","step_rot_max_deg",
        "jump_trans_count","jump_rot_count",
        "cmd",
    ]

    results = {}

    for mode, mode_args in modes:
        run_stamp = ts()
        run_dir = run_root / mode / run_stamp
        ensure_dir(run_dir)
        poses_file = run_dir / "poses.txt"
        log_file = log_dir / f"{args.tag}__{mode}__{run_stamp}.log"

        cmd = [sys.executable, str(main_py), "--image_folder", str(data_dir)]
        cmd += common_extra
        cmd += mode_args
        cmd += ["--log_results", "--log_path", str(poses_file)]
        cmd += passthrough

        print("\n" + "="*80)
        print(f"[RUN] tag={args.tag} mode={mode}")
        print(f"[DATA] {data_dir}")
        print(f"[RUN_DIR] {run_dir}")
        print(f"[LOG] {log_file}")
        print(f"[CMD] {' '.join(cmd)}")
        print("="*80 + "\n")

        t0 = time.time()
        rc = stream_subprocess(cmd, cwd=repo_root, log_path=log_file, env=env)
        sec = time.time() - t0

        traj = parse_poses_txt(poses_file) if rc == 0 else None
        met = compute_metrics(traj, args.jump_trans_thresh, args.jump_rot_thresh_deg) if traj else {k:"" for k in [
            "n_poses","path_length_m","loop_trans_err_m","loop_rot_err_deg","drift_ratio",
            "step_trans_mean_m","step_trans_p95_m","step_trans_max_m",
            "step_rot_mean_deg","step_rot_p95_deg","step_rot_max_deg",
            "jump_trans_count","jump_rot_count",
        ]}
        n_submaps, n_loops = parse_submaps_and_loops_from_log(log_file)

        row = {
            "time": now_iso(),
            "tag": args.tag,
            "mode": mode,
            "returncode": rc,
            "seconds": f"{sec:.3f}",
            "dataset": str(data_dir),
            "n_images": n_images,
            "run_dir": str(run_dir),
            "log_file": str(log_file),
            "poses_file": str(poses_file),
            "n_submaps": n_submaps,
            "n_loops": n_loops,
            "cmd": " ".join(cmd),
        }
        row.update(met)
        append_csv(metrics_csv, row, fieldnames)

        results[mode] = row

        if rc != 0:
            print(f"[WARN] run failed (rc={rc}). see log: {log_file}")
        elif traj is None:
            print(f"[WARN] poses.txt not parsed or too short -> {poses_file}")

    # print delta (semantic - baseline)
    if "baseline" in results and "semantic" in results:
        b = results["baseline"]
        s = results["semantic"]
        keys = ["loop_trans_err_m","drift_ratio","path_length_m","seconds","n_loops","n_poses"]
        print("\n=== semantic - baseline (delta) ===")
        for k in keys:
            try:
                fb = float(b[k])
                fs = float(s[k])
                print(f"{k:16s}: {fs - fb:+.6f}")
            except Exception:
                print(f"{k:16s}: (skip) baseline={b.get(k,'')} semantic={s.get(k,'')}")
        print(f"\n[DONE] metrics -> {metrics_csv}")
        print(f"[DONE] runs    -> {run_root}")
        print(f"[DONE] logs    -> {log_dir}")

if __name__ == "__main__":
    main()
