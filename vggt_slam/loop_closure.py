import os
import heapq
from pathlib import Path as _Path
from typing import NamedTuple, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from salad.eval import load_model  # SALAD


def _patch_torch_hub_offline():
    """
    Force torch.hub.load() to use local cached repos for offline environments.
    Looks for:
      - <hub_dir>/serizba_salad_main
      - <hub_dir>/facebookresearch_dinov2_main
    """
    import torch as _torch

    hub_dir = _Path(_torch.hub.get_dir())
    mapping = {}
    salad = hub_dir / "serizba_salad_main"
    dinov2 = hub_dir / "facebookresearch_dinov2_main"
    if salad.exists():
        mapping["serizba/salad"] = salad
    if dinov2.exists():
        mapping["facebookresearch/dinov2"] = dinov2

    if not mapping:
        return {}

    _orig = _torch.hub.load

    def _load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir in mapping:
            kwargs["source"] = "local"
            repo_or_dir = str(mapping[repo_or_dir])
        return _orig(repo_or_dir, model, *args, **kwargs)

    _torch.hub.load = _load
    return mapping


device = "cuda" if torch.cuda.is_available() else "cpu"

tensor_to_pil = T.ToPILImage()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def input_transform(image_size=None):
    transform_list = [T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
    if image_size:
        transform_list.insert(0, T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))
    return T.Compose(transform_list)


class LoopMatch(NamedTuple):
    score: float
    query_submap_id: int
    query_submap_frame: int
    detected_submap_id: int
    detected_submap_frame: int


class LoopMatchQueue:
    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.heap: List[Tuple[float, LoopMatch]] = []

    def add(self, match: LoopMatch):
        # heap stores (-score, match) so that "largest negative" == smallest score is at end after sort
        item = (-float(match.score), match)
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
        else:
            heapq.heappushpop(self.heap, item)

    def get_matches(self):
        # return sorted by score ascending (best first)
        return [m for _, m in sorted(self.heap, reverse=True)]


class ImageRetrieval:
    def __init__(self, input_size=224):
        _patch_torch_hub_offline()

        # Ensure salad repo is present in hub cache (offline-safe with patch)
        try:
            torch.hub.load("serizba/salad", "dinov2_salad")
        except Exception:
            # ignore; we will load checkpoint below anyway
            pass

        ckpt_pth = os.path.join(torch.hub.get_dir(), "checkpoints/dino_salad.ckpt")
        self.model = load_model(ckpt_pth)
        self.model.eval()
        self.model.to(device)

        self.transform = input_transform((input_size, input_size))

    @torch.no_grad()
    def get_batch_descriptors(self, imgs):
        """
        imgs: torch.Tensor (B,3,H,W) OR list of torch.Tensor/PIL images.
        return: torch.Tensor (B,D) on CPU.
        """
        # to list of PIL
        pil_list = []
        if isinstance(imgs, torch.Tensor):
            # (B,3,H,W)
            for i in range(imgs.shape[0]):
                im = imgs[i].detach().cpu()
                # if normalized already, just convert; else assume 0..1
                pil_list.append(tensor_to_pil(im))
        else:
            for im in imgs:
                if isinstance(im, Image.Image):
                    pil_list.append(im)
                elif isinstance(im, torch.Tensor):
                    pil_list.append(tensor_to_pil(im.detach().cpu()))
                else:
                    # numpy HWC BGR/RGB
                    arr = np.asarray(im)
                    if arr.ndim == 3 and arr.shape[2] == 3:
                        pil_list.append(Image.fromarray(arr.astype(np.uint8)))
                    else:
                        raise ValueError("Unsupported image type for retrieval")

        batch = torch.stack([self.transform(im) for im in pil_list], dim=0).to(device)

        out = self.model(batch)

        # robust unpack
        if isinstance(out, dict):
            if "global_descriptor" in out:
                desc = out["global_descriptor"]
            elif "descriptor" in out:
                desc = out["descriptor"]
            else:
                desc = next(iter(out.values()))
        elif isinstance(out, (tuple, list)):
            desc = out[0]
        else:
            desc = out

        desc = desc.detach()
        # normalize shape -> (B,D)
        if desc.dim() == 1:
            desc = desc.unsqueeze(0)
        elif desc.dim() == 3:
            # common: (B,1,D) or (1,B,D)
            if desc.shape[1] == 1:
                desc = desc.squeeze(1)
            elif desc.shape[0] == 1:
                desc = desc.squeeze(0)
            else:
                desc = desc.reshape(desc.shape[0], -1)
        elif desc.dim() > 3:
            desc = desc.reshape(desc.shape[0], -1)

        desc = desc.float()
        # optional normalize (helps dot-product / euclid stability)
        desc = torch.nn.functional.normalize(desc, dim=-1)

        return desc.cpu()

    def get_all_submap_embeddings(self, submap):
        frames = submap.get_all_frames()
        return self.get_batch_descriptors(frames)

    def find_loop_closures(
        self,
        map,
        submap,
        max_similarity_thres: float = 0.8,
        max_loop_closures: int = 0,
    ):
        """
        NOTE:
          - best_score is distance (smaller better) if map uses L2.
          - keep condition: best_score < max_similarity_thres
        """
        if max_loop_closures <= 0:
            return []

        matches_queue = LoopMatchQueue(max_loop_closures)

        query_id = 0
        for query_vector in submap.get_all_retrieval_vectors():
            try:
                qv = query_vector
                if isinstance(qv, torch.Tensor):
                    qv = qv.detach().cpu().float().reshape(-1)
                else:
                    qv = torch.as_tensor(qv).float().reshape(-1)

                best_score, best_submap_id, best_frame_id = map.retrieve_best_score_frame(
                    qv,
                    submap.get_id(),
                    ignore_last_submap=True,
                )

                if best_score < float(max_similarity_thres):
                    matches_queue.add(
                        LoopMatch(
                            float(best_score),
                            int(submap.get_id()),
                            int(query_id),
                            int(best_submap_id),
                            int(best_frame_id),
                        )
                    )

            except Exception as e:
                # 关键兜底：任何 embedding 维度/类型异常，不要炸整个 SLAM
                # 直接跳过这个 query frame
                # print(f"[WARN][loop_closure] skip query {query_id}: {e}")
                pass

            query_id += 1

        return matches_queue.get_matches()


def is_point_in_fov(K, T_wc, point_world, image_size, fov_padding=0.0):
    """
    Check if a 3D point is inside the camera frustum defined by K and T_wc.
    """
    T_cw = np.linalg.inv(T_wc)  # World to camera
    point_cam = T_cw[:3, :3] @ point_world + T_cw[:3, 3]
    if point_cam[2] <= 1e-8:
        return False

    x = K[0, 0] * (point_cam[0] / point_cam[2]) + K[0, 2]
    y = K[1, 1] * (point_cam[1] / point_cam[2]) + K[1, 2]

    W, H = image_size[0], image_size[1]
    pad = float(fov_padding)
    return (-pad <= x <= W + pad) and (-pad <= y <= H + pad)


def frustums_overlap(K1, T1, K2, T2, image_size):
    """
    Approximate overlap check by sampling points on near plane of one frustum and checking in other.
    """
    # sample a few points in image plane for camera1
    W, H = image_size[0], image_size[1]
    pts = [
        np.array([0.0, 0.0, 1.0]),
        np.array([W, 0.0, 1.0]),
        np.array([0.0, H, 1.0]),
        np.array([W, H, 1.0]),
        np.array([W / 2.0, H / 2.0, 1.0]),
    ]

    K1_inv = np.linalg.inv(K1)
    # convert to world points (arbitrary depth=1)
    for p in pts:
        ray = K1_inv @ p
        pw = (T1[:3, :3] @ ray) + T1[:3, 3]
        if is_point_in_fov(K2, T2, pw, image_size, fov_padding=0.0):
            return True
    return False
