import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class PairDataset(Dataset):
    def __init__(self, feats, pos_win=2, neg_min=50, hard_k=20):
        self.feats = torch.from_numpy(feats).float()
        self.feats = nn.functional.normalize(self.feats, dim=-1)
        self.N = self.feats.shape[0]
        self.pos_win = int(pos_win)
        self.neg_min = int(neg_min)

        # hard negatives: top-k similar but far in time
        sim = (self.feats @ self.feats.t()).cpu().numpy()
        self.hard_negs = []
        for i in range(self.N):
            cand = np.argsort(-sim[i])
            hard = []
            for j in cand:
                if abs(i - j) >= self.neg_min:
                    hard.append(int(j))
                if len(hard) >= hard_k:
                    break
            if len(hard) == 0:
                hard = [min(self.N - 1, i + self.neg_min)]
            self.hard_negs.append(hard)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        # positive: nearby frame
        lo = max(0, i - self.pos_win)
        hi = min(self.N - 1, i + self.pos_win)
        j = np.random.randint(lo, hi + 1)
        if j == i:
            j = min(self.N - 1, i + 1)

        # negative: mostly hard negative
        if np.random.rand() < 0.8:
            k = int(np.random.choice(self.hard_negs[i]))
        else:
            while True:
                k = np.random.randint(0, self.N)
                if abs(i - k) >= self.neg_min:
                    break

        return self.feats[i], self.feats[j], self.feats[k]


class MetricHead(nn.Module):
    def __init__(self, dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return nn.functional.normalize(x, dim=-1)


def info_nce(a, p, n, tau=0.07):
    pos = (a * p).sum(-1, keepdim=True) / tau
    neg = (a * n).sum(-1, keepdim=True) / tau
    logits = torch.cat([pos, neg], dim=1)  # [B,2]
    labels = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
    return nn.CrossEntropyLoss()(logits, labels)


def main(feats_npy, out_ckpt, device="cuda", epochs=10, bs=64, lr=1e-3, num_workers=2):
    data = np.load(feats_npy, allow_pickle=True).item()
    feats = data["feats"]
    dim = feats.shape[1]
    N = feats.shape[0]

    neg_min = max(20, N // 5)
    ds = PairDataset(feats, pos_win=2, neg_min=neg_min, hard_k=min(50, max(5, N // 2)))

    # 关键：别让 batch_size > N 且 drop_last=True 导致 0 step
    bs = int(min(bs, N))
    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=True,
    )

    head = MetricHead(dim, out_dim=256).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(lr))

    head.train()
    for ep in range(int(epochs)):
        losses = []
        for a, p, n in tqdm(dl, desc=f"ep{ep}"):
            a, p, n = a.to(device), p.to(device), n.to(device)
            za, zp, zn = head(a), head(p), head(n)
            loss = info_nce(za, zp, zn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        if len(losses) == 0:
            raise RuntimeError(
                f"No training steps executed. N={N}, bs={bs}. "
                f"Check DataLoader settings."
            )

        print(f"epoch {ep} loss={np.mean(losses):.4f} (steps={len(losses)})")

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "dim": dim}, out_ckpt)
    print("saved:", out_ckpt)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    main(args.feats, args.out, args.device, args.epochs, args.bs, args.lr, args.num_workers)
