import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict

# Optional deps
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_OK = False

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    Image = None
    PIL_OK = False


@dataclass
class SemanticBackendCfg:
    # Deep backend (DINOv2 + optional metric head)
    use_deep: bool = True
    dinov2_repo: str = ""           # optional: /path/to/facebookresearch_dinov2_main
    dinov2_name: str = "dinov2_vits14"
    dinov2_ckpt: str = ""           # optional (not used in this loader)
    metric_head_ckpt: str = ""      # optional: path to .pth produced by your training script
    deep_img_size: int = 224

    # HOG fallback
    hog_img_size: int = 128
    hog_winsize: int = 128
    hog_blocksize: int = 16
    hog_blockstride: int = 8
    hog_cellsize: int = 8
    hog_nbins: int = 9

    # Cache
    cache_size: int = 4096


if TORCH_OK:
    class MetricHead(nn.Module):
        """Simple MLP metric head."""
        def __init__(self, dim: int = 384, out_dim: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, out_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
else:
    class MetricHead(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for MetricHead")

        def forward(self, x):
            raise RuntimeError("torch is required for MetricHead")


class SemanticBackend:
    """Semantic backend for gating and reweighting edges.

    - If deep backend is available (DINOv2 + optional metric head), uses deep embeddings.
    - Otherwise falls back to HOG features.

    IMPORTANT: `similarity(a, b)` accepts:
      - file path (str/Path)
      - numpy image (H,W,3) or (H,W)
      - torch tensor (C,H,W) or (H,W,C)
      - PIL Image
    """

    def __init__(self, cfg_path: str = ""):
        self.cfg = SemanticBackendCfg()
        self.device = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"

        # cache for path-based embeddings
        self._cache = OrderedDict()  # key -> np.ndarray

        # Try read cfg_path (optional)
        self._apply_cfg_path(cfg_path)

        self.use_deep = bool(self.cfg.use_deep and TORCH_OK)
        self.deep_model = None
        self.metric_head = None

        if self.use_deep:
            ok = self._init_deep()
            if ok:
                head_flag = (self.metric_head is not None)
                if head_flag:
                    print(f"[SemanticBackend] deep enabled (dinov2+metric_head) on {self.device}")
                else:
                    print(f"[SemanticBackend] deep enabled (dinov2 backbone only) on {self.device}")
            else:
                self.use_deep = False
                print("[SemanticBackend] deep disabled (missing dinov2 repo), fallback to HOG")
        else:
            if not TORCH_OK:
                print("[SemanticBackend] deep disabled (missing torch), fallback to HOG")
            else:
                print("[SemanticBackend] deep disabled (cfg.use_deep=False), fallback to HOG")

        # HOG descriptor (always available as fallback)
        ws = (int(self.cfg.hog_winsize), int(self.cfg.hog_winsize))
        bs = (int(self.cfg.hog_blocksize), int(self.cfg.hog_blocksize))
        bstr = (int(self.cfg.hog_blockstride), int(self.cfg.hog_blockstride))
        cs = (int(self.cfg.hog_cellsize), int(self.cfg.hog_cellsize))
        self.hog = cv2.HOGDescriptor(ws, bs, bstr, cs, int(self.cfg.hog_nbins))

    # -------------------------
    # Config helpers
    # -------------------------

    def _apply_cfg_path(self, cfg_path: str):
        if not cfg_path:
            return
        p = Path(cfg_path)
        if not p.exists():
            # Allow passing a directory containing semantic_backend.yaml
            if p.is_dir():
                cand = p / "semantic_backend.yaml"
                if cand.exists():
                    p = cand
                else:
                    return
            else:
                return
        try:
            import yaml
            cfg_dict = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return

        # shallow merge
        for k, v in cfg_dict.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    def _init_deep(self) -> bool:
        # Allow missing metric_head_ckpt: enable dinov2 backbone-only
        repo = str(self.cfg.dinov2_repo or "").strip()

        # If repo not provided, try torch hub cache path (offline-friendly if already cached)
        if not repo:
            try:
                hub_dir = Path(torch.hub.get_dir())
                cand = hub_dir / "facebookresearch_dinov2_main"
                if cand.exists():
                    repo = str(cand)
            except Exception:
                repo = ""

        if (not repo) or (not Path(repo).exists()):
            return False

        # Make repo importable
        if repo not in sys.path:
            sys.path.insert(0, repo)

        # Load DINOv2 backbone
        try:
            from dinov2.hub.backbones import (  # type: ignore
                dinov2_vits14,
                dinov2_vitb14,
                dinov2_vitl14,
                dinov2_vitg14,
            )
            name = self.cfg.dinov2_name
            fn = {
                "dinov2_vits14": dinov2_vits14,
                "dinov2_vitb14": dinov2_vitb14,
                "dinov2_vitl14": dinov2_vitl14,
                "dinov2_vitg14": dinov2_vitg14,
            }.get(name, dinov2_vits14)
            self.deep_model = fn(pretrained=True)
        except Exception:
            # fallback: torch.hub (may fail offline if repo not cached)
            try:
                self.deep_model = torch.hub.load(repo, self.cfg.dinov2_name, source="local", pretrained=True)
            except Exception:
                return False

        self.deep_model.eval().to(self.device)

        # Load metric head if provided (optional)
        head_ckpt = str(self.cfg.metric_head_ckpt or "").strip()
        if head_ckpt and Path(head_ckpt).exists():
            ckpt = torch.load(head_ckpt, map_location="cpu")
            dim = int(ckpt.get("dim", 384))
            out_dim = int(ckpt.get("out_dim", 128))
            self.metric_head = MetricHead(dim=dim, out_dim=out_dim)

            state = ckpt.get("state_dict", ckpt)
            fixed_state = {k.replace("module.", ""): v for k, v in state.items()}
            self.metric_head.load_state_dict(fixed_state, strict=False)
            self.metric_head.eval().to(self.device)
        else:
            self.metric_head = None

        return True

    # -------------------------
    # Input conversion
    # -------------------------

    def _read_bgr(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imread failed: {path}")
        return img

    def _to_bgr(self, frame) -> np.ndarray:
        """Convert input to uint8 BGR (OpenCV order)."""
        # Robustly handle numpy scalar / size-1 arrays that accidentally contain a path
        try:
            if isinstance(frame, np.ndarray) and getattr(frame, "dtype", None) is not None:
                if frame.dtype.kind in ("U", "S", "O"):
                    if frame.ndim == 0:
                        frame = str(frame.item())
                    elif frame.size == 1:
                        frame = str(frame.reshape(-1)[0])
        except Exception:
            pass

        # 1) path
        if isinstance(frame, (str, Path)):
            return self._read_bgr(str(frame))

        # 2) PIL
        if PIL_OK and isinstance(frame, Image.Image):
            rgb = np.array(frame.convert("RGB"), dtype=np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        x = frame

        # 3) torch tensor
        if TORCH_OK and torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        # 4) numpy
        x = np.asarray(x)

        # grayscale
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)

        # CHW -> HWC
        if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
            x = np.transpose(x, (1, 2, 0))

        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"unsupported frame shape: {x.shape}")

        # dtype normalize
        if x.dtype != np.uint8:
            x = x.astype(np.float32)
            mx = float(np.max(x)) if x.size else 0.0
            if mx > 1.5:
                x = np.clip(x, 0.0, 255.0)
            else:
                x = np.clip(x * 255.0, 0.0, 255.0)
            x = x.astype(np.uint8)

        # Assume input is RGB unless it clearly comes from cv2; convert RGB->BGR best-effort
        try:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
        return x

    # -------------------------
    # Embeddings
    # -------------------------

    def _resize_square(self, bgr: np.ndarray, size: int) -> np.ndarray:
        return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)

    def _hog_embedding(self, bgr: np.ndarray) -> np.ndarray:
        bgr = self._resize_square(bgr, int(self.cfg.hog_img_size))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        feat = self.hog.compute(gray)
        feat = feat.reshape(-1).astype(np.float32)
        n = np.linalg.norm(feat) + 1e-12
        return feat / n

    def _deep_embedding(self, bgr: np.ndarray) -> np.ndarray:
        if not (self.use_deep and self.deep_model is not None):
            raise RuntimeError("deep backend not initialized")

        # bgr -> rgb
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = self._resize_square(rgb, int(self.cfg.deep_img_size))
        x = torch.from_numpy(rgb).float().to(self.device) / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            y = self.deep_model(x)
            if isinstance(y, dict):
                for k in ("x_norm_clstoken", "x_norm", "cls", "pooled"):
                    if k in y:
                        y = y[k]
                        break
            if isinstance(y, (list, tuple)):
                y = y[0]
            y = y.reshape(y.shape[0], -1)
            if self.metric_head is not None:
                y = self.metric_head(y)
            y = F.normalize(y, dim=-1)

        return y.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def embedding(self, frame) -> np.ndarray:
        cache_key = None
        if isinstance(frame, (str, Path)):
            cache_key = str(frame)
            if cache_key in self._cache:
                v = self._cache.pop(cache_key)
                self._cache[cache_key] = v
                return v

        bgr = self._to_bgr(frame)
        if self.use_deep:
            emb = self._deep_embedding(bgr)
        else:
            emb = self._hog_embedding(bgr)

        if cache_key is not None:
            self._cache[cache_key] = emb
            if len(self._cache) > int(self.cfg.cache_size):
                self._cache.popitem(last=False)
        return emb

    # -------------------------
    # Public API
    # -------------------------

    def similarity(self, frame_a, frame_b) -> float:
        ea = self.embedding(frame_a)
        eb = self.embedding(frame_b)
        s = float(np.dot(ea, eb) / ((np.linalg.norm(ea) + 1e-12) * (np.linalg.norm(eb) + 1e-12)))
        s = max(-1.0, min(1.0, s))
        # HOG similarity can be slightly negative; map to [0,1]
        if not self.use_deep:
            s = 0.5 * (s + 1.0)
        return float(s)
