#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run baseline vs semantic (or any two modes) on selected datasets, and ALWAYS save logs into <repo>/LOG/.

Typical usage:
  python scripts/run_semantic_suite.py \
    --data_root DATA \
    --datasets control_0100-0199 corridor_0300-0399 \
    --modes baseline semantic \
    --baseline_args "--max_loops 1" \
    --semantic_args "--max_loops 1 --use_semantic_backend" \
    --tag sem_v1

Notes:
- This script streams main.py stdout/stderr to terminal AND to a timestamped log file under LOG/.
- It does NOT assume main.py supports an output-dir flag. If you later add one, just include it in *_args.
"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
import subprocess


def repo_root_from_script() -> Path:
    # scripts/run_semantic_suite.py -> repo root = parent of scripts/
    return Path(__file__).resolve().parents[1]


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stream_subprocess(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

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


def append_summary(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    fieldnames = [
        "time",
        "tag",
        "dataset",
        "mode",
        "image_folder",
        "returncode",
        "seconds",
        "log_file",
        "cmd",
    ]

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Run main.py on datasets with baseline/semantic modes and log to <repo>/LOG/")
    parser.add_argument("--repo_root", type=str, default="", help="VGGT-SLAM repo root. Default: auto-detect from this script.")
    parser.add_argument("--main_py", type=str, default="main.py", help="Path to main.py (relative to repo_root).")

    parser.add_argument("--data_root", type=str, default="DATA", help="Data root under repo.")
    parser.add_argument("--datasets", nargs="+", default=["control_0100-0199", "corridor_0300-0399"], help="Dataset folder names under data_root.")

    parser.add_argument("--modes", nargs="+", default=["baseline", "semantic"], help="Run modes (e.g., baseline semantic).")

    parser.add_argument("--baseline_args", type=str, default="", help='Extra args string for baseline, e.g. "--max_loops 1"')
    parser.add_argument("--semantic_args", type=str, default="", help='Extra args string for semantic, e.g. "--max_loops 1 --use_semantic_backend"')

    parser.add_argument("--tag", type=str, default="", help="Experiment tag for log/summary naming.")
    parser.add_argument("--log_dir", type=str, default="LOG", help="Log directory under repo.")
    parser.add_argument("--summary_csv", type=str, default="LOG/semantic_suite_summary.csv", help="Summary CSV path under repo.")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands, do not execute.")

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    main_py = (repo_root / args.main_py).resolve()
    if not main_py.exists():
        print(f"[ERROR] main.py not found: {main_py}", file=sys.stderr)
        sys.exit(2)

    log_dir = (repo_root / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = (repo_root / args.summary_csv).resolve()

    data_root = (repo_root / args.data_root).resolve()
    if not data_root.exists():
        print(f"[ERROR] data_root not found: {data_root}", file=sys.stderr)
        sys.exit(2)

    mode_to_args = {
        "baseline": args.baseline_args,
        "semantic": args.semantic_args,
    }

    # Ensure real-time flush from python + downstream
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for ds in args.datasets:
        image_folder = data_root / ds
        if not image_folder.exists():
            print(f"[WARN] dataset not found, skip: {image_folder}")
            continue

        for mode in args.modes:
            extra = mode_to_args.get(mode, "")
            extra_list = shlex.split(extra)

            tag = args.tag.strip() or "no_tag"
            log_name = f"{tag}__{ds}__{mode}__{ts()}.log"
            log_path = log_dir / log_name

            cmd = [sys.executable, str(main_py), "--image_folder", str(image_folder)] + extra_list

            print("\n" + "=" * 80)
            print(f"[RUN] dataset={ds} mode={mode} tag={tag}")
            print(f"[LOG] {log_path}")
            print(f"[CMD] {' '.join(cmd)}")
            print("=" * 80 + "\n")

            if args.dry_run:
                continue

            t0 = time.time()
            rc = stream_subprocess(cmd, cwd=repo_root, log_path=log_path, env=env)
            sec = time.time() - t0

            append_summary(summary_csv, {
                "time": datetime.now().isoformat(timespec="seconds"),
                "tag": tag,
                "dataset": ds,
                "mode": mode,
                "image_folder": str(image_folder),
                "returncode": rc,
                "seconds": f"{sec:.3f}",
                "log_file": str(log_path),
                "cmd": " ".join(cmd),
            })

            if rc != 0:
                print(f"[ERROR] main.py returned {rc} (see log: {log_path})", file=sys.stderr)

    print(f"\n[DONE] Summary CSV: {summary_csv}")
    print(f"[DONE] Logs directory: {log_dir}")


if __name__ == "__main__":
    main()
