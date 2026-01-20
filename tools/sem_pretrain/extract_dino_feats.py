#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import torchvision.transforms as T


# ======= 你的离线路径（按你提供的）=======
DINO_REPO_DIR = "/media/omnisky/18/cst1/cache/torch/hub/facebookresearch_dinov2_main"
DINO_CKPT = "/media/omnisky/18/cst1/cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
DINO_MODEL_NAME = "dinov2_vitb14"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(image_dir: str, recursive: bool = True) -> List[str]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    if root.is_file():
        # allow single image
        if root.suffix.lower() in IMG_EXTS:
            return [str(root)]
        raise ValueError(f"Provided image_dir is a file but not an image: {image_dir}")

    pattern = "**/*" if recursive else "*"
    files = []
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(str(p))

    files.sort()
    if len(files) == 0:
        raise RuntimeError(f"No images found under: {image_dir}")
    return files


def build_transform(image_size: int) -> T.Compose:
    # DINOv2 常用 ImageNet normalize
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_dinov2(device: str = "cuda") -> torch.nn.Module:
    """
    关键点：
    - torch.hub.load(..., source="local") -> 不联网、不访问 GitHub
    - pretrained=False -> 不触发在线权重下载
    - 手动 load_state_dict 读取本地 ckpt
    """
    if not os.path.isdir(DINO_REPO_DIR):
        raise FileNotFoundError(
            f"DINO repo dir not found: {DINO_REPO_DIR}\n"
            f"Expected hubconf.py under it."
        )
    if not os.path.isfile(os.path.join(DINO_REPO_DIR, "hubconf.py")):
        raise FileNotFoundError(
            f"hubconf.py not found under: {DINO_REPO_DIR}\n"
            f"Make sure this is the dinov2 repo root."
        )
    if not os.path.isfile(DINO_CKPT):
        raise FileNotFoundError(f"DINO checkpoint not found: {DINO_CKPT}")

    model = torch.hub.load(
        DINO_REPO_DIR,
        DINO_MODEL_NAME,
        source="local",
        pretrained=False,
    )

    state = torch.load(DINO_CKPT, map_location="cpu")
    # 有些权重是 {"model": state_dict, ...}
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print(f"[WARN] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    if len(missing) > 0:
        print(f"[WARN] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    model = model.to(device).eval()
    return model


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    image_paths: List[str],
    device: str,
    image_size: int,
    batch_size: int,
) -> Tuple[np.ndarray, List[str]]:
    tfm = build_transform(image_size)

    feats = []
    ok_paths = []

    iterator = range(0, len(image_paths), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(image_paths) + batch_size - 1) // batch_size, desc="Extract")

    for i in iterator:
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(tfm(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"[WARN] skip unreadable image: {p} ({e})")

        if len(imgs) == 0:
            continue

        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)

        # DINOv2 推荐接口：forward_features -> dict，取 x_norm_clstoken
        if hasattr(model, "forward_features"):
            out = model.forward_features(x)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                f = out["x_norm_clstoken"]
            elif isinstance(out, dict) and "x_prenorm" in out:
                # 兜底：不保证存在
                f = out["x_prenorm"][:, 0]
            else:
                # 再兜底
                f = model(x)
        else:
            f = model(x)

        # 统一到 [B, D]
        if f.dim() == 3:
            # [B, T, D] -> cls token
            f = f[:, 0, :]
        f = f.detach().float().cpu().numpy()

        feats.append(f)
        ok_paths.extend(valid_paths)

    feats = np.concatenate(feats, axis=0) if len(feats) > 0 else np.zeros((0, 0), dtype=np.float32)
    return feats, ok_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image folder (recursive) or a single image")
    parser.add_argument("--out", type=str, required=True, help="output .npy path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / cuda:0 ...")
    parser.add_argument("--image_size", type=int, default=518, help="resize+crop size (DINOv2 commonly 518)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--no_recursive", action="store_true", help="disable recursive scan")
    parser.add_argument("--save_paths", action="store_true", help="also save a .txt with image paths next to .npy")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        print("[WARN] CUDA not available, falling back to cpu")
        device = "cpu"

    image_paths = list_images(args.image_dir, recursive=(not args.no_recursive))
    print(f"[INFO] found {len(image_paths)} images")

    model = load_dinov2(device=device)
    feats, ok_paths = extract_features(
        model=model,
        image_paths=image_paths,
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), feats.astype(np.float32))
    print(f"[INFO] saved feats: {feats.shape} -> {out_path}")

    if args.save_paths:
        txt_path = out_path.with_suffix(".paths.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for p in ok_paths:
                f.write(p + "\n")
        print(f"[INFO] saved paths -> {txt_path}")


if __name__ == "__main__":
    main()
