from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image
except Exception as e:
    raise ImportError("PIL(Pillow) is required for semantic masks. Try: pip install pillow") from e


ArrayLike = Union["np.ndarray", "torch.Tensor"]


def load_semantic_mask(mask_path: Union[str, Path], device: Optional[str] = None) -> ArrayLike:
    """
    Load a semantic label mask (H,W) from an image file.
    - Accepts PNG/JPG. Assumes pixel values are integer class IDs.
    - Returns:
        torch.LongTensor(H,W) if torch is available, else np.int64(H,W).
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Semantic mask not found: {mask_path}")

    m = Image.open(mask_path)
    # 语义标签通常是单通道；如果是RGB调色板，也先转成L再读值
    m = m.convert("L")
    arr = np.array(m, dtype=np.int64)

    if torch is not None:
        t = torch.from_numpy(arr).long()
        if device is not None:
            t = t.to(device)
        return t
    return arr


def build_bool_mask(
    semantic: ArrayLike,
    *,
    ignore_ids: Sequence[int] = (),
    keep_ids: Sequence[int] = (),
) -> ArrayLike:
    """
    Build a boolean mask (H,W):
      - if keep_ids is provided: True where class in keep_ids
      - else if ignore_ids is provided: True where class in ignore_ids
    返回 True 的位置表示“命中集合”（后续可选择过滤/保留）。
    """
    if keep_ids and ignore_ids:
        raise ValueError("Use either keep_ids or ignore_ids, not both.")

    if torch is not None and hasattr(semantic, "dtype") and str(getattr(semantic, "device", "")) != "":
        # torch branch
        if keep_ids:
            ids = torch.tensor(list(keep_ids), device=semantic.device, dtype=semantic.dtype)
            return (semantic[..., None] == ids).any(dim=-1)
        if ignore_ids:
            ids = torch.tensor(list(ignore_ids), device=semantic.device, dtype=semantic.dtype)
            return (semantic[..., None] == ids).any(dim=-1)
        return torch.zeros_like(semantic, dtype=torch.bool)

    # numpy branch
    semantic_np = np.asarray(semantic)
    if keep_ids:
        return np.isin(semantic_np, np.asarray(list(keep_ids), dtype=semantic_np.dtype))
    if ignore_ids:
        return np.isin(semantic_np, np.asarray(list(ignore_ids), dtype=semantic_np.dtype))
    return np.zeros_like(semantic_np, dtype=bool)


def apply_mask_to_confidence(
    conf: ArrayLike,
    hit_mask: ArrayLike,
    *,
    mode: str = "zero",  # "zero" or "nan"
) -> ArrayLike:
    """
    Apply boolean mask to confidence map / weight map.
    - hit_mask True 的位置会被处理：
        mode="zero": conf=0
        mode="nan" : conf=NaN (仅对 float 有意义)
    conf 形状支持 (H,W) 或 (H,W,1) 或 (1,H,W) 等可广播情况。
    """
    if mode not in ("zero", "nan"):
        raise ValueError("mode must be 'zero' or 'nan'")

    # torch
    if torch is not None and hasattr(conf, "clone") and hasattr(hit_mask, "dtype"):
        out = conf.clone()
        m = hit_mask.bool()
        if mode == "zero":
            out = out.masked_fill(m, 0.0)
        else:
            out = out.masked_fill(m, float("nan"))
        return out

    # numpy
    out = np.array(conf, copy=True)
    m = np.asarray(hit_mask).astype(bool)
    if mode == "zero":
        out[m] = 0.0
    else:
        out[m] = np.nan
    return out
