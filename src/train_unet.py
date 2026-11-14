"""Train a 2D U-Net on Parihaka facies using the frozen blind-block split.

Design choices
- Training samples are 2D sections taken PERPENDICULAR to the split axis (the
  split is along x, so sections are z-y planes). A patch therefore never
  crosses the buffer gap between train and val/test.
- All sections of each split are held in RAM (normalised float16), so random
  patch sampling is fast. This needs ~1.5 GB for the Parihaka volume.
- Rare classes (train share < 10%) are oversampled: half the time we draw
  several random crops and keep the one with the most rare-class pixels.
- Loss = class-weighted cross-entropy + Dice. Model selection uses val mean IoU
  only. The test strip is evaluated once, at the end, with the best checkpoint.

Usage (repo root, GPU machine):
    python src/train_unet.py --epochs 30 --steps-per-epoch 100 --tag unet_r34
Smoke test (any machine, tiny):
    python src/train_unet.py --epochs 1 --steps-per-epoch 3 --batch 4 --patch 64 \
        --no-pretrained --workers 0 --tag smoke
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp

CLASS_NAMES = ["Basement/Other", "Slope Mudstone A", "Mass Transport Deposit",
               "Slope Mudstone B", "Slope Valley", "Submarine Canyon"]


# --------------------------------------------------------------------------- data
class SectionBank:
    """All sections of one split, perpendicular to the split axis, in RAM.

    seis: (n, Z, L) float16, normalised (z-score, clipped to +-3 sigma)
    lab : (n, Z, L) uint8
    """

    def __init__(self, root, split, name, mean, std, block=32):
        axis = split["axis"]
        seis, lab = root["seismic"], root["labels"]
        s_parts, l_parts, idx = [], [], []
        for a, b in split["ranges"][name]:
            for s0 in range(a, b, block):
                s1 = min(s0 + block, b)
                if axis == "x":
                    s = np.asarray(seis[:, s0:s1, :]).transpose(1, 0, 2)
                    l = np.asarray(lab[:, s0:s1, :]).transpose(1, 0, 2)
                else:
                    s = np.asarray(seis[:, :, s0:s1]).transpose(2, 0, 1)
                    l = np.asarray(lab[:, :, s0:s1]).transpose(2, 0, 1)
                s_parts.append(np.clip((s - mean) / std, -3, 3).astype(np.float16))
                l_parts.append(np.ascontiguousarray(l))
                idx.extend(range(s0, s1))
        self.seis = np.concatenate(s_parts)
        self.lab = np.concatenate(l_parts)
        self.index = np.asarray(idx)          # original trace index of each section
        del s_parts, l_parts

    def __len__(self):
        return self.seis.shape[0]


class PatchDataset(Dataset):
    def __init__(self, bank, patch, n_samples, rare, tries=4, p_rare=0.5):
        self.bank, self.P, self.n = bank, patch, n_samples
        self.rare, self.tries, self.p_rare = np.asarray(rare), tries, p_rare
        self._rng = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._rng is None:   # per-worker, per-epoch stream (seed differs each epoch)
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        rng, P = self._rng, self.P
        n, Z, L = self.bank.seis.shape
        tries = self.tries if rng.random() < self.p_rare else 1
        best = None
        for _ in range(tries):
            k = int(rng.integers(n))
            z0 = int(rng.integers(0, Z - P + 1))
            l0 = int(rng.integers(0, L - P + 1))
            lab = self.bank.lab[k, z0:z0 + P, l0:l0 + P]
            score = int(np.isin(lab, self.rare).sum())
            if best is None or score > best[0]:
                best = (score, k, z0, l0)
        _, k, z0, l0 = best
        x = self.bank.seis[k, z0:z0 + P, l0:l0 + P].astype(np.float32)
        y = self.bank.lab[k, z0:z0 + P, l0:l0 + P].astype(np.int64)
        # augmentation: lateral flip, amplitude scaling, light noise (never flip depth)
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        x = x * rng.uniform(0.8, 1.2) + rng.normal(0, 0.05, x.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(np.ascontiguousarray(y))


# ------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, x, device):
    """x: (B,1,Z,L) float tensor -> logits (B,C,Z,L). Pads to a multiple of 32."""
    Z, L = x.shape[-2:]
    pz, pl = (-Z) % 32, (-L) % 32
    xp = F.pad(x, (0, pl, 0, pz), mode="reflect") if (pz or pl) else x
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
        logits = model(xp)
    return logits[..., :Z, :L].float()


def confusion(model, bank, C, device, stride=1, batch=4):
    model.eval()
    cm = np.zeros((C, C), dtype=np.int64)
    ids = list(range(0, len(bank), stride))
    for i in range(0, len(ids), batch):
        sel = ids[i:i + batch]
        x = torch.from_numpy(bank.seis[sel].astype(np.float32))[:, None].to(device)
        pred = predict(model, x, device).argmax(1).cpu().numpy().astype(np.int64)
        truth = bank.lab[sel].astype(np.int64)
        cm += np.bincount((truth * C + pred).ravel(), minlength=C * C).reshape(C, C)
    return cm


def metrics(cm):
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return {"per_class_iou": [None if np.isnan(v) else float(v) for v in iou],
            "mean_iou": float(np.nanmean(iou)),
            "pixel_accuracy": float(tp.sum() / cm.sum())}


def save_gallery(model, bank, device, path, n, C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(plt.get_cmap("tab10").colors[:C])
    picks = np.linspace(0, len(bank) - 1, n).astype(int)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.4 * n), squeeze=False)
    model.eval()
    for r, k in enumerate(picks):
        x = torch.from_numpy(bank.seis[[k]].astype(np.float32))[:, None].to(device)
        pred = predict(model, x, device).argmax(1)[0].cpu().numpy()
        truth = bank.lab[k]
        miou = metrics(np.bincount((truth.astype(np.int64) * C + pred).ravel(),
                                   minlength=C * C).reshape(C, C))["mean_iou"]
        axes[r, 0].imshow(bank.seis[k].astype(np.float32), cmap="gray", vmin=-3, vmax=3, aspect="auto")
        axes[r, 1].imshow(truth, cmap=cmap, vmin=0, vmax=C - 1, aspect="auto", interpolation="nearest")
        axes[r, 2].imshow(pred, cmap=cmap, vmin=0, vmax=C - 1, aspect="auto", interpolation="nearest")
        axes[r, 3].imshow(pred != truth, cmap="Reds", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for c, t in enumerate(["seismic", "truth", f"prediction (section mIoU {miou:.2f})", "errors"]):
            axes[r, c].set_title(f"{t}" + (f" | trace {bank.index[k]}" if c == 0 else ""), fontsize=9)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.legend(handles=[Patch(color=cmap(i), label=f"{i} {CLASS_NAMES[i] if i < len(CLASS_NAMES) else ''}")
                        for i in range(C)], loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/zarr/parihaka.zarr")
    ap.add_argument("--split", default="configs/split.json")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-stride", type=int, default=4, help="evaluate every k-th val section each epoch")
    ap.add_argument("--figs", type=int, default=4, help="test sections in the gallery (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="unet")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                                 "  -- no GPU, this will be very slow"))

    root = zarr.open_group(args.zarr, mode="r")
    split = json.load(open(args.split))
    C = int(root.attrs["n_classes"])
    glob = np.asarray(root.attrs["class_fractions"])
    assert np.allclose(glob, split["class_fractions"]["global"], atol=1e-6), \
        "split.json was made from different labels than this Zarr store"
    mean, std = float(root.attrs["seismic_mean"]), float(root.attrs["seismic_std"])

    t0 = time.time()
    banks = {n: SectionBank(root, split, n, mean, std) for n in ("train", "val", "test")}
    print("sections in RAM: " + ", ".join(f"{n} {len(b)}" for n, b in banks.items())
          + f"  ({time.time() - t0:.0f}s)")

    train_frac = np.asarray(split["class_fractions"]["train"])
    rare = [i for i, f in enumerate(train_frac) if f < 0.10]
    w = 1.0 / np.sqrt(np.maximum(train_frac, 1e-4)); w = w / w.mean()
    print(f"rare classes oversampled: {rare} | CE weights: {np.round(w, 2).tolist()}")

    try:
        model = smp.Unet(args.encoder, encoder_weights=None if args.no_pretrained else "imagenet",
                         in_channels=1, classes=C)
    except Exception as e:  # offline or weights unavailable
        print(f"WARNING: could not load pretrained weights ({e}); training from scratch")
        model = smp.Unet(args.encoder, encoder_weights=None, in_channels=1, classes=C)
    model.to(device)

    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    dice = smp.losses.DiceLoss(mode="multiclass")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ckpt = Path("checkpoints") / f"{args.tag}_seed{args.seed}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    history, best = [], (-1.0, -1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ds = PatchDataset(banks["train"], args.patch, args.steps_per_epoch * args.batch, rare)
        dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, drop_last=True,
                        pin_memory=device.type == "cuda")
        losses, te = [], time.time()
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(x)
            logits = logits.float()
            loss = ce(logits, y) + dice(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        vm = metrics(confusion(model, banks["val"], C, device, stride=args.val_stride))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_mean_iou": vm["mean_iou"], "val_pixel_accuracy": vm["pixel_accuracy"]})
        flag = ""
        if vm["mean_iou"] > best[0]:
            best = (vm["mean_iou"], epoch)
            torch.save(model.state_dict(), ckpt); flag = "  * best"
        print(f"epoch {epoch:3d}/{args.epochs}  loss {np.mean(losses):.3f}  "
              f"val mIoU {vm['mean_iou']:.3f}  pixAcc {vm['pixel_accuracy']:.3f}  "
              f"({time.time() - te:.0f}s){flag}", flush=True)

    # ---- final evaluation with the best checkpoint (val: all sections; test: once) ----
    model.load_state_dict(torch.load(ckpt, map_location=device))
    res = {}
    for name in ("val", "test"):
        cm = confusion(model, banks[name], C, device)
        res[name] = {**metrics(cm), "confusion_matrix": cm.tolist()}

    base = {}
    if Path("results/baselines.json").exists():
        b = json.load(open("results/baselines.json"))["results"]
        base = {k: v["mean_iou"] for k, v in b.items()}

    print(f"\nBest epoch {best[1]} (val mIoU on subsample {best[0]:.3f}). Final blind-strip results:")
    print(f"{'split':<6}{'mIoU':>7}{'pixAcc':>8}   per-class IoU")
    for name in ("val", "test"):
        r = res[name]
        print(f"{name:<6}{r['mean_iou']:7.3f}{r['pixel_accuracy']:8.3f}   "
              + " ".join("  n/a" if v is None else f"{v:5.2f}" for v in r["per_class_iou"]))
    if base:
        print(f"reference mIoU  depth-only val {base.get('depth_val', float('nan')):.3f} "
              f"test {base.get('depth_test', float('nan')):.3f} | majority val "
              f"{base.get('majority_val', float('nan')):.3f} test {base.get('majority_test', float('nan')):.3f}")

    out = Path("results") / f"{args.tag}_seed{args.seed}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args), "best_epoch": best[1], "history": history, "class_names": CLASS_NAMES,
        "results": res, "baseline_mean_iou": base, "train_seconds": time.time() - t0,
        "torch": torch.__version__, "smp": smp.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }, indent=2))
    print(f"Wrote {out}")
    if args.figs > 0:
        fig = Path("results/figures") / f"{args.tag}_seed{args.seed}_test_gallery.png"
        save_gallery(model, banks["test"], device, fig, args.figs, C)
        print(f"Wrote {fig}")


if __name__ == "__main__":
    main()
