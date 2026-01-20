#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, csv, os, sys, time
from datetime import datetime
from pathlib import Path
import subprocess
import numpy as np
import re


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def stream(cmd: list[str], cwd: Path, log_path: Path) -> int:
    ensure_dir(log_path.parent)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.write(f"[CWD] {cwd}\n")
        f.write(f"[TIME] {datetime.now().isoformat()}\n")
        f.write("-" * 80 + "\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        return proc.wait()


def parse_poses_txt(p: Path):
    if not p.exists():
        return None
    rows = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"[,\s]+", s)
        try:
            vals = [float(x) for x in parts]
        except Exception:
            continue
        rows.append(vals)
    if len(rows) < 2:
        return None
    arr = np.array(rows, dtype=np.float64)
    # frame_id x y z qx qy qz qw
    if arr.shape[1] >= 4:
        return arr[:, 1:4]
    return None


def metrics_from_xyz(xyz: np.ndarray) -> dict:
    N = xyz.shape[0]
    if N < 2:
        return {"n_poses": int(N)}

    dp = xyz[1:] - xyz[:-1]
    step = np.linalg.norm(dp, axis=1)
    path_len = float(step.sum())
    loop_err = float(np.linalg.norm(xyz[-1] - xyz[0]))
    drift_ratio = float(loop_err / (path_len + 1e-12))

    # ---- Corridor-sensitive: lateral drift via PCA line fit in 3D ----
    p0 = xyz[0]
    X = xyz - p0
    # PCA: first principal direction
    # SVD of centered points
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    v = vt[0]
    nv = np.linalg.norm(v)
    if not np.isfinite(nv) or nv < 1e-12:
        v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        v = v / nv

    t = (xyz - p0) @ v
    proj = p0[None, :] + t[:, None] * v[None, :]
    lateral = np.linalg.norm(xyz - proj, axis=1)
    lateral_rmse = float(np.sqrt(np.mean(lateral**2)))
    lateral_p95 = float(np.percentile(lateral, 95))
    lateral_max = float(np.max(lateral))
    along_length = float(np.max(t) - np.min(t))

    return {
        "n_poses": int(N),
        "path_length_m": path_len,
        "loop_trans_err_m": loop_err,
        "drift_ratio": drift_ratio,
        "step_trans_mean_m": float(step.mean()),
        "step_trans_p95_m": float(np.percentile(step, 95)),
        "step_trans_max_m": float(step.max()),
        "lateral_rmse_m": lateral_rmse,
        "lateral_p95_m": lateral_p95,
        "lateral_max_m": lateral_max,
        "along_length_m": along_length,
    }


def append_csv(csv_path: Path, row: dict, fieldnames: list[str]) -> None:
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _strip_flag(args: list[str], flag: str, has_value: bool = False) -> list[str]:
    """Remove all occurrences of a flag (optionally with its value)."""
    out = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            if has_value and i + 1 < len(args):
                i += 2
            else:
                i += 1
            continue
        out.append(args[i])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--data", type=str, default="DATA/corridor_0300-0399")
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--repo_root", type=str, default=".")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--main_py", type=str, default="main.py")
    ap.add_argument("--metrics_csv", type=str, default="LOG/corridor_compare_metrics.csv")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["base", "sem_gate", "sem_weight", "both"],
        choices=["base", "sem_gate", "sem_weight", "both", "baseline", "semantic"],
        help="Ablations: base(no semantic), sem_gate(only gate), sem_weight(only weight), both(gate+weight). "
             "Aliases: baseline->base, semantic->both",
    )
    ap.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    data = (repo / args.data).resolve()
    main_py = (repo / args.main_py).resolve()
    log_dir = repo / "LOG"
    run_root = repo / "RUNS" / args.tag
    ensure_dir(log_dir)
    ensure_dir(run_root)

    fieldnames = [
        "time", "tag", "mode", "returncode", "seconds",
        "dataset", "run_dir", "log_file", "poses_file",
        "n_poses", "path_length_m", "loop_trans_err_m", "drift_ratio",
        "step_trans_mean_m", "step_trans_p95_m", "step_trans_max_m",
        "cmd",
        "lateral_rmse_m", "lateral_p95_m", "lateral_max_m", "along_length_m",
    ]

    passthrough = args.passthrough
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    # Runner is authoritative for semantic backend enable; prevent accidental contamination.
    passthrough = _strip_flag(passthrough, "--use_semantic_backend", has_value=False)

    # You can still pass these, but runner will override per-mode by appending force_args at the end.
    # (Argparse takes the last occurrence.)

    modes = []
    for m in args.modes:
        if m == "baseline":
            modes.append("base")
        elif m == "semantic":
            modes.append("both")
        else:
            modes.append(m)

    for mode in modes:
        stamp = ts()
        run_dir = run_root / mode / stamp
        ensure_dir(run_dir)

        log_file = log_dir / f"{args.tag}__{mode}__{stamp}.log"
        poses_file = run_dir / "poses.txt"
        edge_stats_file = run_dir / "edge_stats.csv"

        cmd = [
            args.python, str(main_py),
            "--image_folder", str(data),
            "--max_loops", str(args.max_loops),
            "--log_results",
            "--log_path", str(poses_file),
            "--edge_stats_path", str(edge_stats_file),
        ]

        # semantic backend toggle by mode
        if mode in ("sem_gate", "sem_weight", "both"):
            cmd += ["--use_semantic_backend"]

        # common passthrough (sanitized)
        cmd += list(passthrough)

        # force args per ablation (placed last => wins)
        force_args: list[str] = []
        if mode == "base":
            force_args += ["--semantic_weight_mode", "off", "--disable_semantic_gate"]
        elif mode == "sem_gate":
            force_args += ["--semantic_weight_mode", "off"]   # gate only
        elif mode == "sem_weight":
            force_args += ["--disable_semantic_gate"]         # weight only (keep your weight_mode in passthrough)
        elif mode == "both":
            pass

        cmd += force_args

        print("\n" + "=" * 80)
        print(f"[RUN] tag={args.tag} mode={mode}")
        print(f"[DATA] {data}")
        print(f"[RUN_DIR] {run_dir}")
        print(f"[LOG] {log_file}")
        print(f"[CMD] {' '.join(cmd)}")
        print("=" * 80 + "\n")

        t0 = time.time()
        rc = stream(cmd, cwd=repo, log_path=log_file)
        sec = time.time() - t0

        xyz = parse_poses_txt(poses_file) if rc == 0 else None
        met = {}
        if xyz is not None:
            met = metrics_from_xyz(xyz)

        row = {
            "time": now_iso(),
            "tag": args.tag,
            "mode": mode,
            "returncode": rc,
            "seconds": f"{sec:.3f}",
            "dataset": str(data),
            "run_dir": str(run_dir),
            "log_file": str(log_file),
            "poses_file": str(poses_file) if poses_file.exists() else "",
            "cmd": " ".join(cmd),
        }
        row.update(met)
        append_csv((repo / args.metrics_csv).resolve(), row, fieldnames)

        if rc != 0:
            print(f"[WARN] run failed (rc={rc}). see log: {log_file}")
        elif xyz is None:
            print(f"[WARN] poses exists but not parsed: {poses_file}")

    print(f"\n[DONE] metrics -> {repo / args.metrics_csv}")
    print(f"[DONE] runs    -> {run_root}")
    print(f"[DONE] logs    -> {log_dir}")


if __name__ == "__main__":
    main()
