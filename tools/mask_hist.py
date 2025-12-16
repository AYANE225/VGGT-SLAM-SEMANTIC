import os, glob
import argparse
import numpy as np
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--suffix", default=".png")
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.mask_dir, f"*{args.suffix}")))
    if not paths:
        raise SystemExit(f"No masks found in {args.mask_dir} with suffix {args.suffix}")

    hist = {}
    total = 0
    for p in paths:
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[..., 0]
        m = m.astype(np.int64)
        vals, cnts = np.unique(m, return_counts=True)
        for v, c in zip(vals, cnts):
            hist[int(v)] = hist.get(int(v), 0) + int(c)
        total += int(m.size)

    items = sorted(hist.items(), key=lambda x: x[1], reverse=True)
    print(f"masks={len(paths)} total_pixels={total}")
    print("top ids:")
    for i, (k, v) in enumerate(items[:args.topk]):
        print(f"  id={k:4d}  count={v:12d}  ratio={v/total:.4f}")

if __name__ == "__main__":
    main()
