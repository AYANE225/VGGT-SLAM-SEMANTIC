#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, csv, os, re, shlex, sys, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Optional, Tuple, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_float(x: str):
    try: return float(x)
    except: return None

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

def compute_basic_metrics(traj: Traj, jump_trans_thresh: float, jump_rot_thresh_deg: float) -> dict:
    p = traj.p
    N = p.shape[0]
    if N < 2:
        return {"n_poses": N}

    dp = p[1:] - p[:-1]
    step_trans = np.linalg.norm(dp, axis=1)
    out = {
        "n_poses": int(N),
        "path_length_m": float(step_trans.sum()),
        "loop_trans_err_m": float(np.linalg.norm(p[-1] - p[0])),
        "step_trans_mean_m": float(np.mean(step_trans)),
        "step_trans_p95_m": float(np.percentile(step_trans, 95)),
        "step_trans_max_m": float(np.max(step_trans)),
        "jump_trans_count": int(np.sum(step_trans > jump_trans_thresh)),
        "loop_rot_err_deg": "",
        "step_rot_mean_deg": "",
        "step_rot_p95_deg": "",
        "step_rot_max_deg": "",
        "jump_rot_count": "",
    }

    if traj.q is not None and traj.q.shape[0] == N:
        q = quat_normalize(traj.q)
        q_rel = quat_mul(q[1:], quat_conj(q[:-1]))
        step_rot = quat_to_rot_angle_deg(q_rel)
        q_loop = quat_mul(q[-1], quat_conj(q[0]))
        out["loop_rot_err_deg"] = float(quat_to_rot_angle_deg(q_loop)[()])
        out["step_rot_mean_deg"] = float(np.mean(step_rot))
        out["step_rot_p95_deg"] = float(np.percentile(step_rot, 95))
        out["step_rot_max_deg"] = float(np.max(step_rot))
        out["jump_rot_count"] = int(np.sum(step_rot > jump_rot_thresh_deg))

    return out

TRAJ_PATTERNS = [
    r".*traj.*\.(txt|csv)$",
    r".*trajectory.*\.(txt|csv)$",
    r".*pose.*\.(txt|csv)$",
    r".*poses.*\.(txt|csv)$",
    r".*tum.*\.(txt|csv)$",
    r".*kitti.*\.(txt|csv)$",
    r".*estimated.*\.(txt|csv)$",
    r".*cam.*pose.*\.(txt|csv)$",
]

def list_recent_files(root: Path, start_epoch: float, regex_list: List[str]) -> List[Path]:
    regs = [re.compile(r, re.IGNORECASE) for r in regex_list]
    hits = []
    if not root.exists():
        return hits
    for p in root.rglob("*"):
        if not p.is_file(): continue
        try:
            if p.stat().st_mtime < start_epoch: continue
        except: 
            continue
        if any(r.match(p.name) for r in regs):
            hits.append(p)
    hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return hits

def try_parse_traj_file(path: Path) -> Optional[Traj]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
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

    if arr.shape[1] >= 8:  # TUM
        return Traj(t=arr[:,0], p=arr[:,1:4], q=arr[:,4:8])
    if arr.shape[1] == 12: # KITTI
        p = np.stack([arr[:,3], arr[:,7], arr[:,11]], axis=1)
        t = np.arange(p.shape[0], dtype=np.float64)
        return Traj(t=t, p=p, q=None)
    if arr.shape[1] == 4:  # t xyz
        return Traj(t=arr[:,0], p=arr[:,1:4], q=None)
    if arr.shape[1] == 3:  # xyz
        p = arr[:,0:3]
        t = np.arange(p.shape[0], dtype=np.float64)
        return Traj(t=t, p=p, q=None)

    return None

def discover_and_parse_traj(run_dir: Path, start_epoch: float) -> Tuple[str, Optional[Traj]]:
    cand = list_recent_files(run_dir, start_epoch, TRAJ_PATTERNS)
    for f in cand[:50]:
        traj = try_parse_traj_file(f)
        if traj is not None and traj.p.shape[0] >= 2:
            return str(f), traj
    return "", None

def save_plots(traj: Traj, out_dir: Path, prefix: str) -> dict:
    out = {"plot_traj":"", "plot_step_trans":"", "plot_step_rot":""}
    if not HAS_PLT:
        return out
    ensure_dir(out_dir)
    p = traj.p
    x = p[:,0]
    y = p[:,2] if p.shape[1] >= 3 else p[:,1]

    fig = plt.figure()
    plt.plot(x, y)
    plt.axis("equal")
    plt.title("Trajectory (top-down)")
    plt.xlabel("x")
    plt.ylabel("z" if p.shape[1] >= 3 else "y")
    fp = out_dir / f"{prefix}__traj_xy.png"
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    out["plot_traj"] = str(fp)

    dp = p[1:] - p[:-1]
    step_trans = np.linalg.norm(dp, axis=1)
    fig = plt.figure()
    plt.plot(step_trans)
    plt.title("Step translation (m)")
    plt.xlabel("step idx")
    plt.ylabel("m")
    fp = out_dir / f"{prefix}__step_trans.png"
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    out["plot_step_trans"] = str(fp)

    if traj.q is not None and traj.q.shape[0] == p.shape[0]:
        q = quat_normalize(traj.q)
        q_rel = quat_mul(q[1:], quat_conj(q[:-1]))
        step_rot = quat_to_rot_angle_deg(q_rel)
        fig = plt.figure()
        plt.plot(step_rot)
        plt.title("Step rotation (deg)")
        plt.xlabel("step idx")
        plt.ylabel("deg")
        fp = out_dir / f"{prefix}__step_rot.png"
        fig.savefig(fp, dpi=200, bbox_inches="tight")
        plt.close(fig)
        out["plot_step_rot"] = str(fp)

    return out

def stream_subprocess(cmd: list[str], cwd: Path, log_path: Path, env: dict[str,str]) -> int:
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

def append_csv(csv_path: Path, row: dict, fieldnames: List[str]) -> None:
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default="")
    ap.add_argument("--main_py", type=str, default="main.py")
    ap.add_argument("--data_root", type=str, default="DATA")
    ap.add_argument("--datasets", nargs="+", default=["control_0100-0199", "corridor_0300-0399"])
    ap.add_argument("--modes", nargs="+", default=["baseline", "semantic"])
    ap.add_argument("--baseline_args", type=str, default="")
    ap.add_argument("--semantic_args", type=str, default="")
    ap.add_argument("--tag", type=str, default="no_tag")
    ap.add_argument("--log_dir", type=str, default="LOG")
    ap.add_argument("--run_root", type=str, default="RUNS")
    ap.add_argument("--metrics_csv", type=str, default="LOG/semantic_suite_metrics.csv")
    ap.add_argument("--jump_trans_thresh", type=float, default=0.5)
    ap.add_argument("--jump_rot_thresh_deg", type=float, default=30.0)

    # 关键升级：强制让 main.py 输出落盘（利用 main.py 自带 --log_results/--log_path）
    ap.add_argument("--force_log_results", action="store_true", default=True,
                    help="Force append --log_results --log_path <run_dir> to main.py.")
    ap.add_argument("--no_force_log_results", action="store_true",
                    help="Disable force log_results/log_path injection.")
    ap.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    main_py = (repo_root / args.main_py).resolve()
    if not main_py.exists():
        print(f"[ERROR] main.py not found: {main_py}", file=sys.stderr)
        sys.exit(2)

    data_root = (repo_root / args.data_root).resolve()
    if not data_root.exists():
        print(f"[ERROR] data_root not found: {data_root}", file=sys.stderr)
        sys.exit(2)

    log_dir = (repo_root / args.log_dir).resolve(); ensure_dir(log_dir)
    run_root = (repo_root / args.run_root).resolve(); ensure_dir(run_root)
    metrics_csv = (repo_root / args.metrics_csv).resolve()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    passthrough = args.passthrough
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    mode_to_args = {"baseline": args.baseline_args, "semantic": args.semantic_args}

    fieldnames = [
        "time","tag","dataset","mode","returncode","seconds",
        "image_folder","run_dir","log_file","traj_file",
        "n_poses","path_length_m","loop_trans_err_m","loop_rot_err_deg",
        "step_trans_mean_m","step_trans_p95_m","step_trans_max_m",
        "step_rot_mean_deg","step_rot_p95_deg","step_rot_max_deg",
        "jump_trans_count","jump_rot_count",
        "plot_traj","plot_step_trans","plot_step_rot",
        "cmd",
    ]

    for ds in args.datasets:
        ds_dir = data_root / ds
        if not ds_dir.exists():
            print(f"[WARN] dataset not found: {ds_dir}")
            continue

        for mode in args.modes:
            extra = mode_to_args.get(mode, "")
            extra_list = shlex.split(extra)

            run_stamp = ts()
            run_dir = run_root / args.tag / ds / mode / run_stamp
            ensure_dir(run_dir)

            log_path = log_dir / f"{args.tag}__{ds}__{mode}__{run_stamp}.log"

            cmd = [sys.executable, str(main_py), "--image_folder", str(ds_dir)]
            cmd += extra_list
            cmd += passthrough

            # 注入 log_results/log_path，让 main.py 把轨迹/结果稳定输出到 run_dir
            force = args.force_log_results and (not args.no_force_log_results)
            if force:
                if "--log_results" not in cmd:
                    cmd += ["--log_results"]
                if "--log_path" not in cmd:
                    cmd += ["--log_path", str(run_dir / "poses.txt")]

            print("\n" + "="*80)
            print(f"[RUN] tag={args.tag} dataset={ds} mode={mode}")
            print(f"[RUN_DIR] {run_dir}")
            print(f"[LOG] {log_path}")
            print(f"[CMD] {' '.join(cmd)}")
            print("="*80 + "\n")

            start_epoch = time.time()
            t0 = time.time()
            rc = stream_subprocess(cmd, cwd=repo_root, log_path=log_path, env=env)
            sec = time.time() - t0

            traj_file, traj = discover_and_parse_traj(run_dir, start_epoch)

            metrics = {
                "n_poses":"","path_length_m":"","loop_trans_err_m":"","loop_rot_err_deg":"",
                "step_trans_mean_m":"","step_trans_p95_m":"","step_trans_max_m":"",
                "step_rot_mean_deg":"","step_rot_p95_deg":"","step_rot_max_deg":"",
                "jump_trans_count":"","jump_rot_count":"",
            }
            plots = {"plot_traj":"","plot_step_trans":"","plot_step_rot":""}

            if traj is not None:
                metrics = compute_basic_metrics(traj, args.jump_trans_thresh, args.jump_rot_thresh_deg)
                plots = save_plots(traj, run_dir, f"{ds}__{mode}")

            row = {
                "time": now_iso(),
                "tag": args.tag,
                "dataset": ds,
                "mode": mode,
                "returncode": rc,
                "seconds": f"{sec:.3f}",
                "image_folder": str(ds_dir),
                "run_dir": str(run_dir),
                "log_file": str(log_path),
                "traj_file": traj_file,
                "cmd": " ".join(cmd),
            }
            row.update(metrics)
            row.update(plots)
            append_csv(metrics_csv, row, fieldnames)

            if traj is None:
                print(f"[WARN] 仍未解析到轨迹文件。请到 run_dir 看 main.py 实际输出了什么文件：{run_dir}")

    print(f"\n[DONE] {metrics_csv}")

if __name__ == "__main__":
    main()
