from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

try:
    import torch
except Exception:
    torch = None


def _parse_ids(s: str | None) -> list[int]:
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    parts = s.replace(",", " ").split()
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return out


def _split_keys(s: str | None) -> list[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    # 支持 "a,b" 或 "a b" 或 "a,b c"
    s = s.replace(",", " ")
    return [k.strip() for k in s.split() if k.strip()]


def _find_conf_key(pred: dict) -> str | None:
    for k in ("world_points_conf", "depth_conf", "confidence", "conf", "weights", "weight"):
        if k in pred:
            return k
    return None


def _is_torch(x) -> bool:
    return torch is not None and "torch" in str(type(x))


def _shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        return None


def _infer_hw(conf):
    shp = _shape(conf)
    if shp is None or len(shp) < 2:
        return None
    return shp[-2], shp[-1]


def apply_semantic_filter(predictions: dict, image_paths: list[str], args):
    sem_dir = getattr(args, "semantic_dir", "")
    if not sem_dir:
        return predictions

    sem_dir = Path(sem_dir)
    suffix = getattr(args, "semantic_suffix", ".png")
    ignore_ids = set(_parse_ids(getattr(args, "semantic_ignore_ids", "")))
    keep_ids = set(_parse_ids(getattr(args, "semantic_keep_ids", "")))
    missing_ok = bool(getattr(args, "semantic_missing_ok", False))

    debug_dir = getattr(args, "semantic_debug_dir", "")
    debug_stride = int(getattr(args, "semantic_debug_stride", 20))
    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
    else:
        debug_dir = None

    forced = _split_keys(getattr(args, "semantic_conf_key", ""))
    conf_keys = [k for k in forced if k in predictions]
    if not conf_keys:
        k = _find_conf_key(predictions)
        conf_keys = [k] if k else []

    if not conf_keys:
        print("[semantic] WARNING: cannot find any conf key in predictions, skip semantic filtering.")
        return predictions

    # 用第一个 key 推断 H/W（其余 key 默认同分辨率）
    hw0 = _infer_hw(predictions[conf_keys[0]])
    if hw0 is None:
        print(f"[semantic] WARNING: key '{conf_keys[0]}' has no H/W dims, skip semantic filtering.")
        return predictions
    H0, W0 = hw0

    # 先把所有 conf_out clone/copy 好
    conf_out_map = {}
    for ck in conf_keys:
        c = predictions[ck]
        conf_out_map[ck] = c.clone() if _is_torch(c) else np.asarray(c).copy()

    keep_ratios = []
    applied = 0

    for i, img_path in enumerate(image_paths):
        stem = Path(img_path).stem
        mpath = sem_dir / f"{stem}{suffix}"

        if not mpath.exists():
            if missing_ok:
                continue
            raise FileNotFoundError(f"[semantic] mask not found: {mpath}")

        m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if m is None:
            if missing_ok:
                continue
            raise RuntimeError(f"[semantic] failed to read mask: {mpath}")

        if m.ndim == 3:
            m = m[..., 0]
        m = m.astype(np.int64)

        if m.shape != (H0, W0):
            m = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST)

        # keep=True 表示保留（静态/可信）；推荐办公室：keep_ids="0"（只保留背景）
        if len(keep_ids) > 0:
            keep = np.isin(m, list(keep_ids))
        else:
            keep = ~np.isin(m, list(ignore_ids))

        keep_ratios.append(float(keep.mean()))

        if debug_dir is not None and (debug_stride <= 1 or (i % debug_stride) == 0):
            cv2.imwrite(str(debug_dir / f"{stem}_keep.png"), (keep.astype(np.uint8) * 255))
            cv2.imwrite(str(debug_dir / f"{stem}_label.png"), m.astype(np.uint16))

        # 对每个 conf key 都乘一次 keep
        for ck in conf_keys:
            conf_out = conf_out_map[ck]
            is_t = _is_torch(conf_out)
            if is_t:
                kt = torch.from_numpy(keep).to(device=conf_out.device)
                if conf_out.ndim == 4:
                    conf_out[i, 0] = conf_out[i, 0] * kt
                elif conf_out.ndim == 3:
                    conf_out[i] = conf_out[i] * kt
                elif conf_out.ndim == 2:
                    conf_out = conf_out * kt
                else:
                    conf_out[i, ...] = conf_out[i, ...] * kt
                conf_out_map[ck] = conf_out
            else:
                if conf_out.ndim == 4:
                    conf_out[i, 0] *= keep
                elif conf_out.ndim == 3:
                    conf_out[i] *= keep
                elif conf_out.ndim == 2:
                    conf_out *= keep
                else:
                    conf_out[i, ...] *= keep
                conf_out_map[ck] = conf_out

        applied += 1

    # 写回 predictions
    for ck in conf_keys:
        predictions[ck] = conf_out_map[ck]

    predictions["_semantic_applied_frames"] = applied
    if keep_ratios:
        predictions["_semantic_keep_ratio_mean"] = float(np.mean(keep_ratios))
        predictions["_semantic_keep_ratio_min"] = float(np.min(keep_ratios))
        print(f"[semantic] keys={conf_keys} applied={applied} keep_mean={predictions['_semantic_keep_ratio_mean']:.3f} keep_min={predictions['_semantic_keep_ratio_min']:.3f}")

    return predictions
