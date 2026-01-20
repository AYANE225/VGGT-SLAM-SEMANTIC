#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunked runner + visualization for VGGT-SLAM.

Features:
  1) Natural-sort images in --image_folder
  2) Split into chunks (default 100)
  3) Create per-chunk folder with images (symlink by default; fallback to copy)
  4) Run: python main.py --image_folder <chunk_images> + passthrough args
     - stream stdout in real-time to terminal AND write to run.log
  5) Compute simple texture/quality stats per chunk
  6) (Optional) Save per-chunk input visualization (RGB + gradient magnitude)
  7) Write chunk_summary.csv and summary plots (line plot + scatter)

IMPORTANT:
  - Wrapper args (like --viz) are NOT passed to main.py.
  - Only arguments after `--` are passed to main.py.
    Example:  ... --viz -- --max_loops 1

Usage:
  cd /path/to/VGGT-SLAM

  python run_chunked_vggt_slam.py \
    --repo_root . \
    --image_folder ./office_loop \
    --chunk_size 100 \
    --out_root runs_chunk100_viz \
    --prefer_symlink \
    --viz --samples_per_chunk 12 \
    -- --max_loops 1
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image

# headless plotting (safe on servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(s: str):
    """Natural sort key: frame_2 < frame_10"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder: Path) -> List[Path]:
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort(key=lambda p: natural_key(p.name))
    return imgs


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, prefer_symlink: bool = True):
    """Symlink preferred; fallback to copy if symlink not permitted."""
    if dst.exists():
        return
    try:
        if prefer_symlink:
            os.symlink(src.resolve(), dst)
        else:
            shutil.copy2(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_gray_uint8(img_path: Path) -> np.ndarray:
    im = Image.open(img_path).convert("L")
    return np.asarray(im, dtype=np.uint8)


def texture_stats(img_paths: List[Path]) -> Dict[str, float]:
    """
    Simple texture/blur proxies (cheap, dependency-free):
      - lap_var_mean: variance of discrete Laplacian (higher => sharper/more texture)
      - grad_mean: mean abs gradient magnitude (higher => more edges)
      - brightness_mean/std
      - blur_score: inverse of lap_var_mean (higher => blurrier)
    """
    lap_vars, grad_means, means, stds = [], [], [], []

    for p in img_paths:
        g = load_gray_uint8(p).astype(np.float32)

        gx = np.abs(g[:, 1:] - g[:, :-1])
        gy = np.abs(g[1:, :] - g[:-1, :])
        grad_means.append(float(0.5 * (gx.mean() + gy.mean())))

        # 4-neighbor Laplacian: -4I + N+S+E+W
        lap = (-4.0 * g
               + np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0)
               + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1))
        lap_vars.append(float(lap.var()))

        means.append(float(g.mean()))
        stds.append(float(g.std()))

    lap_var_mean = float(np.mean(lap_vars)) if lap_vars else 0.0
    return {
        "lap_var_mean": lap_var_mean,
        "grad_mean": float(np.mean(grad_means)) if grad_means else 0.0,
        "brightness_mean": float(np.mean(means)) if means else 0.0,
        "brightness_std": float(np.mean(stds)) if stds else 0.0,
        "blur_score": float(1.0 / (lap_var_mean + 1e-6)),
    }


def parse_generic_costs(log_text: str) -> Dict[str, Any]:
    """
    Best-effort: extract last numeric value from lines containing keywords.
    VGGT-SLAM may not print these; often returns found=False.
    """
    keywords = ["loss", "cost", "error", "chi2", "chi^2", "objective", "residual"]
    hits = []
    for ln in log_text.splitlines():
        low = ln.lower()
        if any(k in low for k in keywords):
            vals = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", ln)
            if vals:
                try:
                    hits.append(float(vals[-1]))
                except Exception:
                    pass
    if not hits:
        return {"found": False}
    return {
        "found": True,
        "count": len(hits),
        "last": float(hits[-1]),
        "min": float(np.min(hits)),
        "max": float(np.max(hits)),
        "mean": float(np.mean(hits)),
    }


def save_input_viz(img_paths: List[Path], out_path: Path, n_samples: int = 12, thumb_w: int = 320):
    """
    Save a contact sheet:
      Row 1: RGB
      Row 2: gradient magnitude (edge strength)
    """
    if not img_paths:
        return

    n = len(img_paths)
    n_samples = max(2, min(n_samples, n))
    idxs = np.linspace(0, n - 1, n_samples).round().astype(int).tolist()

    cols = min(6, n_samples)
    rows = int(np.ceil(n_samples / cols))

    fig = plt.figure(figsize=(cols * 3.2, rows * 3.4))

    for k, i in enumerate(idxs):
        p = img_paths[i]
        rgb = Image.open(p).convert("RGB")
        w, h = rgb.size
        scale = thumb_w / max(1, w)
        rgb = rgb.resize((int(w * scale), int(h * scale)))

        g = np.asarray(rgb.convert("L"), dtype=np.float32)
        gx = np.abs(g[:, 1:] - g[:, :-1])
        gy = np.abs(g[1:, :] - g[:-1, :])
        gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
        gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
        grad = (gx + gy) * 0.5
        grad = grad / (grad.max() + 1e-6)

        ax1 = plt.subplot(rows * 2, cols, k + 1)
        ax1.imshow(rgb)
        ax1.set_title(p.name, fontsize=8)
        ax1.axis("off")

        ax2 = plt.subplot(rows * 2, cols, cols * rows + k + 1)
        ax2.imshow(grad, cmap="gray")
        ax2.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_summary_plots(summary_csv: Path, out_root: Path):
    """Generate summary plots for chunk texture proxies."""
    rows = []
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return

    chunk_id = np.array([int(r["chunk_id"]) for r in rows], dtype=int)
    lap = np.array([float(r["lap_var_mean"]) for r in rows], dtype=float)
    grad = np.array([float(r["grad_mean"]) for r in rows], dtype=float)
    blur = np.array([float(r["blur_score"]) for r in rows], dtype=float)

    # 1) Line plot over chunks
    fig1 = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(chunk_id, lap, marker="o", label="lap_var_mean")
    ax.plot(chunk_id, grad, marker="o", label="grad_mean")
    ax.plot(chunk_id, blur, marker="o", label="blur_score")
    ax.set_xlabel("chunk_id")
    ax.set_ylabel("value")
    ax.set_title("Texture/Blur proxies over chunks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(out_root / "summary_texture_over_chunks.png", dpi=160)
    plt.close(fig1)

    # 2) Scatter lap vs grad, annotate chunk ids
    fig2 = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.scatter(lap, grad)
    for i in range(len(chunk_id)):
        ax.annotate(str(chunk_id[i]), (lap[i], grad[i]), fontsize=9)
    ax.set_xlabel("lap_var_mean")
    ax.set_ylabel("grad_mean")
    ax.set_title("lap_var_mean vs grad_mean (annotated by chunk_id)")
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_root / "scatter_lap_vs_grad.png", dpi=160)
    plt.close(fig2)


@dataclass
class ChunkResult:
    chunk_id: int
    start_idx: int
    end_idx: int
    n_images: int
    out_dir: Path
    ok: bool
    returncode: int
    seconds: float
    tex: Dict[str, float]
    costs: Dict[str, Any]


def strip_duplicate_image_folder_args(extra: List[str]) -> List[str]:
    """Remove accidental duplicate --image_folder ... from passthrough args."""
    out = []
    skip_next = False
    for a in extra:
        if skip_next:
            skip_next = False
            continue
        if a == "--image_folder":
            skip_next = True
            continue
        out.append(a)
    return out


def find_main_py(repo_root: Path) -> Path:
    """Find main.py in repo_root (direct), else recursively if unique."""
    direct = repo_root / "main.py"
    if direct.exists():
        return direct

    candidates = list(repo_root.rglob("main.py"))
    if len(candidates) == 1:
        return candidates[0]

    msg = f"Cannot find main.py at: {direct}\n"
    msg += f"Found candidates: {[str(c) for c in candidates]}\n"
    msg += "Please pass correct --repo_root (the folder that contains main.py)."
    raise FileNotFoundError(msg)


def split_wrapper_and_passthrough(argv: List[str], ap: argparse.ArgumentParser):
    """
    Only pass arguments AFTER `--` to main.py.
    Wrapper args are everything BEFORE `--`.
    """
    if "--" in argv:
        cut = argv.index("--")
        known_argv = argv[:cut]
        passthrough = argv[cut + 1:]
        args = ap.parse_args(known_argv)
        extra = passthrough
    else:
        # backward compatible: unknown args are forwarded (not recommended)
        args, extra = ap.parse_known_args(argv)
    return args, extra


def run_one_chunk(
    repo_root: Path,
    main_py: Path,
    chunk_imgs: List[Path],
    out_dir: Path,
    prefer_symlink: bool,
    python_bin: str,
    passthrough_args: List[str],
    make_viz: bool = False,
    samples_per_chunk: int = 12,
    echo_main_output: bool = True,
) -> ChunkResult:
    safe_mkdir(out_dir)
    chunk_img_dir = out_dir / "images"
    safe_mkdir(chunk_img_dir)

    # materialize images as symlinks/copies
    for p in chunk_imgs:
        link_or_copy(p, chunk_img_dir / p.name, prefer_symlink=prefer_symlink)

    cmd = [python_bin, str(main_py), "--image_folder", str(chunk_img_dir)] + passthrough_args

    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    collected = []
    with log_path.open("a", encoding="utf-8") as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            collected.append(line)
            f.write(line)
            f.flush()
            if echo_main_output:
                sys.stdout.write(line)
                sys.stdout.flush()

    proc.wait()
    secs = time.time() - t0
    returncode = int(proc.returncode)

    log_text = "".join(collected)
    tex = texture_stats(chunk_imgs)
    costs = parse_generic_costs(log_text)

    (out_dir / "stats.json").write_text(
        json.dumps(
            {
                "texture": tex,
                "costs": costs,
                "returncode": returncode,
                "seconds": secs,
                "cmd": cmd,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if make_viz:
        try:
            save_input_viz(chunk_imgs, out_dir / "inputs_viz.png", n_samples=samples_per_chunk)
        except Exception as e:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n[VIZ_ERROR] {repr(e)}\n")

    return ChunkResult(
        chunk_id=-1,
        start_idx=-1,
        end_idx=-1,
        n_images=len(chunk_imgs),
        out_dir=out_dir,
        ok=(returncode == 0),
        returncode=returncode,
        seconds=secs,
        tex=tex,
        costs=costs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, required=True, help="Path to VGGT-SLAM repo (contains main.py).")
    ap.add_argument("--image_folder", type=str, required=True, help="Folder containing your images.")
    ap.add_argument("--out_root", type=str, default="runs_chunked", help="Where to store outputs.")
    ap.add_argument("--chunk_size", type=int, default=100)
    ap.add_argument("--start_idx", type=int, default=0, help="Start index in sorted image list (inclusive).")
    ap.add_argument("--end_idx", type=int, default=-1, help="End index in sorted image list (inclusive). -1 means last.")
    ap.add_argument("--prefer_symlink", action="store_true", help="Use symlinks (fast, no duplication).")

    ap.add_argument("--viz", action="store_true", help="Save per-chunk input visualizations and summary plots.")
    ap.add_argument("--samples_per_chunk", type=int, default=12, help="How many sample images to visualize per chunk.")
    ap.add_argument("--no_echo", action="store_true", help="Do not echo main.py output to terminal (still written to run.log).")

    ap.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable to run main.py.")

    args, extra = split_wrapper_and_passthrough(sys.argv[1:], ap)
    passthrough_args = strip_duplicate_image_folder_args(extra)

    repo_root = Path(args.repo_root).resolve()
    main_py = find_main_py(repo_root)

    img_folder = Path(args.image_folder).resolve()
    imgs_all = list_images(img_folder)
    if not imgs_all:
        raise RuntimeError(f"No images found in {img_folder} (extensions: {sorted(IMG_EXTS)})")

    start = max(0, int(args.start_idx))
    end = (len(imgs_all) - 1) if int(args.end_idx) < 0 else min(len(imgs_all) - 1, int(args.end_idx))
    if start > end:
        raise ValueError(f"start_idx({start}) > end_idx({end})")

    imgs = imgs_all[start:end + 1]
    chunk_size = max(1, int(args.chunk_size))

    out_root = Path(args.out_root).resolve()
    safe_mkdir(out_root)

    n = len(imgs)
    n_chunks = (n + chunk_size - 1) // chunk_size
    results: List[ChunkResult] = []

    print(f"[INFO] repo_root   : {repo_root}")
    print(f"[INFO] main.py     : {main_py}")
    print(f"[INFO] image_folder: {img_folder}")
    print(f"[INFO] images      : {len(imgs_all)} total, using [{start}:{end}] => {len(imgs)}")
    print(f"[INFO] chunk_size  : {chunk_size} => {n_chunks} chunks")
    print(f"[INFO] out_root    : {out_root}")
    print(f"[INFO] passthrough : {' '.join(passthrough_args) if passthrough_args else '(none)'}")
    print("")

    for ci in range(n_chunks):
        s = ci * chunk_size
        e = min(n, (ci + 1) * chunk_size) - 1
        chunk_imgs = imgs[s:e + 1]

        chunk_name = f"chunk_{ci:02d}_idx_{start + s:04d}-{start + e:04d}"
        out_dir = out_root / chunk_name

        print(f"\n========== RUN {ci+1}/{n_chunks}: {chunk_name} (n={len(chunk_imgs)}) ==========")
        r = run_one_chunk(
            repo_root=repo_root,
            main_py=main_py,
            chunk_imgs=chunk_imgs,
            out_dir=out_dir,
            prefer_symlink=bool(args.prefer_symlink),
            python_bin=args.python_bin,
            passthrough_args=passthrough_args,
            make_viz=bool(args.viz),
            samples_per_chunk=int(args.samples_per_chunk),
            echo_main_output=(not args.no_echo),
        )
        r.chunk_id = ci
        r.start_idx = start + s
        r.end_idx = start + e
        results.append(r)

        print(f"[CHUNK DONE] returncode={r.returncode}  seconds={r.seconds:.1f}  out_dir={r.out_dir}")

    summary_path = out_root / "chunk_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "chunk_id", "start_idx", "end_idx", "n_images",
            "ok", "returncode", "seconds",
            "lap_var_mean", "grad_mean", "brightness_mean", "brightness_std", "blur_score",
            "cost_found", "cost_last", "cost_min", "cost_max", "cost_mean",
            "out_dir"
        ])
        for r in results:
            cost_found = r.costs.get("found", False)
            w.writerow([
                r.chunk_id, r.start_idx, r.end_idx, r.n_images,
                r.ok, r.returncode, f"{r.seconds:.3f}",
                r.tex.get("lap_var_mean", 0.0),
                r.tex.get("grad_mean", 0.0),
                r.tex.get("brightness_mean", 0.0),
                r.tex.get("brightness_std", 0.0),
                r.tex.get("blur_score", 0.0),
                cost_found,
                r.costs.get("last", ""),
                r.costs.get("min", ""),
                r.costs.get("max", ""),
                r.costs.get("mean", ""),
                str(r.out_dir),
            ])

    print(f"\n[DONE] Wrote: {summary_path}")
    print(f"       Logs per chunk : {out_root}/chunk_*/run.log")
    print(f"       Per-chunk stats: {out_root}/chunk_*/stats.json")
    if args.viz:
        save_summary_plots(summary_path, out_root)
        print(f"[VIZ] Wrote: {out_root / 'summary_texture_over_chunks.png'}")
        print(f"[VIZ] Wrote: {out_root / 'scatter_lap_vs_grad.png'}")
        print(f"[VIZ] Per-chunk: {out_root}/chunk_*/inputs_viz.png")


if __name__ == "__main__":
    main()
