"""Leave-one-strip-out folds along the same axis and blocks as the primary split.

Fold t holds out block t as TEST and uses block (t-1) mod K as VALIDATION, with
the same buffer gaps as make_splits.py. With the default settings, the fold whose
test block equals the primary split's test block reproduces the primary split
exactly. Every fold is written to configs/folds/fold<t>.json in the same schema
as configs/split.json, so train_unet.py and ensemble_eval.py accept it through
--split without changes.

Usage (repo root):
    python src/make_folds.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from make_splits import SPLITS, assign_indices, evaluate, lateral_class_counts, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/zarr/parihaka.zarr")
    ap.add_argument("--split", default="configs/split.json", help="primary split (gives axis, blocks, buffer)")
    ap.add_argument("--out-dir", default="configs/folds")
    ap.add_argument("--min-rel", type=float, default=0.3)
    args = ap.parse_args()

    primary = json.load(open(args.split))
    axis, K, buffer, n = primary["axis"], primary["n_blocks"], primary["buffer"], primary["axis_length"]
    root = zarr.open_group(args.zarr, mode="r")
    C = int(root.attrs["n_classes"])
    glob = np.asarray(root.attrs["class_fractions"], dtype=float)
    print(f"counting classes along {axis} ({n} traces)...")
    cx, cy = lateral_class_counts(root["labels"], C)
    counts = cx if axis == "x" else cy

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{K} folds, buffer {buffer}, axis {axis}")
    for t in range(K):
        v = (t - 1) % K
        a, edges = assign_indices(n, K, t, v, buffer)
        ok, _, fr = evaluate(counts, a, glob, args.min_rel)
        ranges = {name: runs(a == code) for code, name in enumerate(SPLITS)}
        ranges["buffer"] = runs(a == -1)
        same = (t == primary["test_block"] and v == primary["val_block"]
                and ranges["test"] == primary["ranges"]["test"] and ranges["train"] == primary["ranges"]["train"])
        (out_dir / f"fold{t}.json").write_text(json.dumps({
            "fold": t, "axis": axis, "axis_length": int(n), "n_blocks": K, "buffer": buffer,
            "block_edges": [int(e) for e in edges], "test_block": t, "val_block": int(v),
            "ranges": ranges, "coverage_ok": bool(ok),
            "class_fractions": {"global": glob.tolist(), **{k: f.tolist() for k, f in fr.items()}},
            "note": "Leave-one-strip-out fold. FROZEN once experiments start; do not edit.",
        }, indent=2))
        tr = sum(b - s for s, b in ranges["train"])
        print(f"fold {t}: test {ranges['test']} val {ranges['val']} train {tr} traces"
              + ("  [= primary split]" if same else "") + ("" if ok else "  WARNING: a class is under-represented"))
        print("   test class fractions:", " ".join(f"{x:.3f}" for x in fr["test"]))
    print(f"\nWrote {K} files to {out_dir}/")


if __name__ == "__main__":
    main()
