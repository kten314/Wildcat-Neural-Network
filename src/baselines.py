"""Floor baselines on the frozen blind-block split. No seismic amplitudes used.

1. majority : predict the most common training class everywhere.
2. depth    : predict the most common training class at each depth sample (z).
              Facies are layered, so this shows how much "free" signal depth
              alone gives. Any neural network must clearly beat both.

Metrics are per-class IoU and mean IoU (classes absent from both the truth and
the prediction are skipped), plus pixel accuracy for reference.

Usage (repo root):
    python src/baselines.py
Writes results/baselines.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import zarr

AXIS = {"x": 1, "y": 2}


def sections(lab, axis, ranges, step=16):
    """Yield label blocks (z, n, y) or (z, x, n) covering the given index ranges."""
    for a, b in ranges:
        for s in range(a, b, step):
            e = min(s + step, b)
            yield np.asarray(lab[:, s:e, :] if axis == "x" else lab[:, :, s:e])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/zarr/parihaka.zarr")
    ap.add_argument("--split", default="configs/split.json")
    ap.add_argument("--out", default="results/baselines.json")
    args = ap.parse_args()

    split = json.load(open(args.split))
    axis = split["axis"]
    root = zarr.open_group(args.zarr, mode="r")
    lab = root["labels"]
    Z = lab.shape[0]
    C = int(root.attrs["n_classes"])

    # ---- fit on train: class counts per depth sample ----
    zc = np.zeros((Z, C), dtype=np.int64)
    zoff = (np.arange(Z, dtype=np.int32) * C)[:, None, None]
    for blk in sections(lab, axis, split["ranges"]["train"]):
        idx = zoff + blk.astype(np.int32)
        zc += np.bincount(idx.ravel(), minlength=Z * C).reshape(Z, C)
    majority = int(zc.sum(axis=0).argmax())
    depth_table = zc.argmax(axis=1).astype(np.uint8)          # (Z,)
    print(f"majority class (train): {majority}")

    # ---- evaluate on val and test ----
    results = {}
    for name in ("val", "test"):
        cms = {"majority": np.zeros((C, C), np.int64), "depth": np.zeros((C, C), np.int64)}
        for blk in sections(lab, axis, split["ranges"][name]):
            truth = blk.astype(np.int32)
            preds = {
                "majority": np.full(blk.shape, majority, dtype=np.int32),
                "depth": np.broadcast_to(depth_table[:, None, None].astype(np.int32), blk.shape),
            }
            for k, p in preds.items():
                cms[k] += np.bincount((truth * C + p).ravel(),
                                      minlength=C * C).reshape(C, C)
        for k, cm in cms.items():
            tp = np.diag(cm).astype(float)
            denom = cm.sum(axis=1) + cm.sum(axis=0) - tp
            iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
            results[f"{k}_{name}"] = {
                "per_class_iou": [None if np.isnan(v) else float(v) for v in iou],
                "mean_iou": float(np.nanmean(iou)),
                "pixel_accuracy": float(tp.sum() / cm.sum()),
            }

    print(f"\n{'baseline':<10}{'split':<6}{'mIoU':>7}{'pixAcc':>8}   per-class IoU")
    for k in ("majority", "depth"):
        for name in ("val", "test"):
            r = results[f"{k}_{name}"]
            ious = " ".join("  n/a" if v is None else f"{v:5.2f}" for v in r["per_class_iou"])
            print(f"{k:<10}{name:<6}{r['mean_iou']:7.3f}{r['pixel_accuracy']:8.3f}   {ious}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"majority_class": majority, "results": results}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
