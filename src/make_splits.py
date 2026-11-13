"""Create a frozen blind-block train/val/test split for the Parihaka volume.

The survey is cut into contiguous strips along ONE lateral axis (x or y, never
depth z: facies are layered, so a depth split would put different geology in
each set). Strips assigned to val/test are separated from train by a buffer
gap, because neighbouring traces are nearly identical and would leak labels.

Strips are chosen from label statistics only (class coverage), BEFORE any
model is trained. Then the result is frozen in a JSON file. Never change the
split after looking at model results.

Usage (repo root):
    python src/make_splits.py --blocks 5 --buffer 64
Writes configs/split.json
"""
import argparse
import json
import itertools
from pathlib import Path

import numpy as np
import zarr

SPLITS = ("train", "val", "test")


def lateral_class_counts(lab, n_classes):
    """Return class counts per x index (X, C) and per y index (Y, C)."""
    z, X, Y = lab.shape
    cx = np.zeros((X, n_classes), dtype=np.int64)
    cy = np.zeros((Y, n_classes), dtype=np.int64)
    step = lab.chunks[2]
    for y0 in range(0, Y, step):
        blk = np.asarray(lab[:, :, y0:y0 + step])
        for s0 in range(0, blk.shape[2], 16):
            sub = blk[:, :, s0:s0 + 16].astype(np.int32)
            n = sub.shape[2]
            idx = sub + n_classes * np.arange(n, dtype=np.int32)[None, None, :]
            cy[y0 + s0:y0 + s0 + n] = np.bincount(
                idx.ravel(), minlength=n * n_classes).reshape(n, n_classes)
            idx = sub + n_classes * np.arange(X, dtype=np.int32)[None, :, None]
            cx += np.bincount(idx.ravel(), minlength=X * n_classes).reshape(X, n_classes)
    return cx, cy


def assign_indices(n, n_blocks, test_block, val_block, buffer):
    """Label every index along the axis: 0 train, 1 val, 2 test, -1 buffer."""
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    block_of = np.zeros(n, dtype=int)
    for k in range(n_blocks):
        block_of[edges[k]:edges[k + 1]] = k
    a = np.zeros(n, dtype=int)
    a[block_of == val_block] = 1
    a[block_of == test_block] = 2
    kernel = np.ones(2 * buffer + 1, dtype=int)
    near_test = np.convolve((a == 2).astype(int), kernel, mode="same") > 0
    near_val = np.convolve((a == 1).astype(int), kernel, mode="same") > 0
    orig = a.copy()
    a[near_test & (orig != 2)] = -1          # test keeps full width
    a[near_val & (orig == 0)] = -1           # val only eats into train
    return a, edges


def runs(mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append([start, i]); start = None
    if start is not None:
        out.append([start, len(mask)])
    return out


def evaluate(counts_axis, a, glob_frac, min_rel):
    """Return (ok, score, per-split fractions) for one assignment."""
    fr, ok, score = {}, True, 0.0
    for code, name in enumerate(SPLITS):
        c = counts_axis[a == code].sum(axis=0)
        if c.sum() == 0:
            return False, np.inf, fr
        f = c / c.sum()
        fr[name] = f
        if (f < min_rel * glob_frac).any():
            ok = False
        if name != "train":
            score += 0.5 * np.abs(f - glob_frac).sum()   # total variation distance
    return ok, score, fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/zarr/parihaka.zarr")
    ap.add_argument("--out", default="configs/split.json")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--buffer", type=int, default=64,
                    help="gap (in traces) between train and val/test; use >= patch width")
    ap.add_argument("--min-rel", type=float, default=0.3,
                    help="every class must reach this share of its global fraction in every split")
    args = ap.parse_args()

    root = zarr.open_group(args.zarr, mode="r")
    lab = root["labels"]
    C = int(root.attrs["n_classes"])
    glob = np.asarray(root.attrs["class_fractions"], dtype=float)
    print(f"labels {lab.shape}, {C} classes; counting classes per x and y index...")
    cx, cy = lateral_class_counts(lab, C)
    assert cx.sum() == cy.sum() == int(np.prod(lab.shape))

    candidates = []
    for axis, counts in (("x", cx), ("y", cy)):
        n = counts.shape[0]
        for t, v in itertools.permutations(range(args.blocks), 2):
            a, edges = assign_indices(n, args.blocks, t, v, args.buffer)
            ok, score, fr = evaluate(counts, a, glob, args.min_rel)
            if ok:
                candidates.append((score, axis, t, v, a, edges, fr))
    if not candidates:
        raise SystemExit("No split keeps every class represented in every split. "
                         "Try --blocks 4, a smaller --buffer, or a lower --min-rel.")
    candidates.sort(key=lambda c: c[0])

    print("\nBest candidate splits (lower score = closer to global class mix):")
    for score, axis, t, v, *_ in candidates[:5]:
        print(f"  axis {axis}: test block {t}, val block {v}, score {score:.3f}")

    score, axis, t, v, a, edges, fr = candidates[0]
    n = len(a)
    # self-check: gap between train and val/test is at least the buffer
    tr = np.where(a == 0)[0]
    ev = np.where(a >= 1)[0]
    gap = np.abs(tr[:, None] - ev[None, :]).min()
    assert gap > args.buffer, f"buffer violated: gap {gap}"

    ranges = {name: runs(a == code) for code, name in enumerate(SPLITS)}
    ranges["buffer"] = runs(a == -1)
    other = int(np.prod(lab.shape) // n)
    print(f"\nChosen: split along {axis} ({n} traces), {args.blocks} blocks, "
          f"buffer {args.buffer}, test block {t}, val block {v}")
    print("Index ranges [start, stop) along", axis)
    for name in (*SPLITS, "buffer"):
        total = sum(b - s for s, b in ranges[name])
        print(f"  {name:<6} {ranges[name]}  ({total} traces, {100 * total / n:.0f}% of axis)")
    print("Class fractions (global | train | val | test):")
    for i in range(C):
        print(f"  class {i}: {glob[i]:.4f} | {fr['train'][i]:.4f} | "
              f"{fr['val'][i]:.4f} | {fr['test'][i]:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "axis": axis, "axis_length": int(n), "n_blocks": args.blocks,
        "buffer": args.buffer, "block_edges": [int(e) for e in edges],
        "test_block": int(t), "val_block": int(v), "ranges": ranges,
        "class_fractions": {"global": glob.tolist(),
                            **{k: f.tolist() for k, f in fr.items()}},
        "note": "Chosen from label class coverage only, before any modelling. "
                "FROZEN: do not edit after seeing model results. Training patches "
                "must lie entirely inside a train range along the split axis.",
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
