import os, glob, argparse
import numpy as np
import cv2
import torch

# COCO 80类（用于打印/核对）
COCO = [
"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
"elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
"skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
"broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
"dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
"toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def load_model(device: str):
    import torchvision
    try:
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        try:
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            model = maskrcnn_resnet50_fpn(weights=weights)
            preprocess = weights.transforms()
        except Exception:
            model = maskrcnn_resnet50_fpn(pretrained=True)
            preprocess = None
    except Exception as e:
        raise RuntimeError("torchvision maskrcnn_resnet50_fpn not available. Check torchvision install.") from e

    model.eval().to(device)
    return model, preprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score", type=float, default=0.6)
    ap.add_argument("--stride", type=int, default=1, help="process every N-th image")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model, preprocess = load_model(args.device)

    paths = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    paths = [p for p in paths if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]]
    paths = paths[::max(1, args.stride)]
    if not paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    # 统计：办公室重点类
    # person=1 tv=63 laptop=64 mouse=65 remote=66 keyboard=67 cell phone=68 potted plant=59
    print("[COCO ids] person=1 tv=63 laptop=64 mouse=65 remote=66 keyboard=67 cell_phone=68 plant=59 chair=57")

    for idx, p in enumerate(paths):
        bname = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(args.out_dir, bname + ".png")

        if os.path.exists(out_path):
            continue


        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # to tensor (C,H,W), float in [0,1]
        x = torch.from_numpy(img_rgb).to(args.device).permute(2,0,1).float() / 255.0
        if preprocess is not None:
            x = preprocess(x)

        with torch.no_grad():
            out = model([x])[0]

        H, W = img_rgb.shape[:2]
        label_map = np.zeros((H, W), dtype=np.uint8)
        score_map = np.zeros((H, W), dtype=np.float32)

        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        masks = out["masks"].detach().cpu().numpy()  # (N,1,H,W)

        keep = scores >= args.score
        scores = scores[keep]
        labels = labels[keep]
        masks = masks[keep]

        # 逐实例写入：同一像素取更高 score 的实例
        for s, lab, m in zip(scores, labels, masks):
            m2 = (m[0] >= 0.5)
            upd = m2 & (s > score_map)
            score_map[upd] = float(s)
            label_map[upd] = int(lab) if int(lab) <= 255 else 255

        cv2.imwrite(out_path, label_map)

        if idx % 50 == 0:
            # 打印一下这一帧主要检测到什么
            uniq = np.unique(label_map)
            names = []
            for u in uniq:
                if u == 0: 
                    continue
                if 1 <= u <= len(COCO):
                    names.append(f"{u}:{COCO[u-1]}")
                else:
                    names.append(str(u))
            print(f"[{idx}/{len(paths)}] wrote {out_path}  ids={names[:10]}")

    print("DONE. masks saved to:", args.out_dir)

if __name__ == "__main__":
    main()
