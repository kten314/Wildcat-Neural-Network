"""Convert the Parihaka facies .npz files into one compressed Zarr store.

The .npz arrays are Fortran-ordered, shape (Z, X, Y) = (1006, 782, 590).
In that layout each y-slice is contiguous on disk, so we stream a few
y-slices at a time and never hold the full 1.9 GB volume in memory.

Usage (from the repo root, venv active):
    pip install zarr numpy
    python src/convert_parihaka.py

Output: data/zarr/parihaka.zarr with
    seismic  float32 (z, x, y)
    labels   uint8   (z, x, y), shifted so classes start at 0
plus attributes holding normalization statistics and class frequencies.
"""
import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import zarr

try:  # zarr >= 3
    from zarr.codecs import BloscCodec

    ZARR3 = True
except ImportError:  # zarr 2.x
    from numcodecs import Blosc

    ZARR3 = False


def make_array(group, name, shape, chunks, dtype):
    if ZARR3:
        return group.create_array(
            name, shape=shape, chunks=chunks, dtype=dtype,
            compressors=BloscCodec(cname="zstd", clevel=3, shuffle="shuffle"),
        )
    return group.create_dataset(
        name, shape=shape, chunks=chunks, dtype=dtype,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
    )


def read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        part = f.read(n - len(buf))
        if not part:
            raise EOFError("Unexpected end of file while reading array data")
        buf += part
    return bytes(buf)


def npy_header(zf, member):
    f = zf.open(member)
    version = np.lib.format.read_magic(f)
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, fortran, dtype = reader(f)
    return f, shape, np.dtype(dtype), fortran


def array_info(npz_path, key):
    with zipfile.ZipFile(npz_path) as zf:
        f, shape, dtype, fortran = npy_header(zf, key + ".npy")
        f.close()
    return shape, dtype, fortran


def iter_slabs(npz_path, key, slab):
    """Yield (y0, block) where block has logical shape (Z, X, n) in (z, x, y) order."""
    with zipfile.ZipFile(npz_path) as zf:
        f, shape, dtype, fortran = npy_header(zf, key + ".npy")
        if not fortran:
            raise NotImplementedError(
                f"{npz_path}:{key} is C-ordered; this script expects Fortran order")
        z, x, y = shape
        for y0 in range(0, y, slab):
            n = min(slab, y - y0)
            raw = read_exact(f, n * x * z * dtype.itemsize)
            # Fortran order: z fastest, then x, then y -> C array [y, x, z]
            block = np.frombuffer(raw, dtype=dtype).reshape(n, x, z)
            yield y0, np.ascontiguousarray(block.transpose(2, 1, 0))
        f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/parihaka")
    ap.add_argument("--out", default="data/zarr/parihaka.zarr")
    ap.add_argument("--slab", type=int, default=8, help="y-slices read at a time")
    ap.add_argument("--chunk", type=int, default=128, help="Zarr chunk edge length")
    args = ap.parse_args()

    raw = Path(args.raw)
    data_npz = raw / "parihaka_data_train.npz"
    lab_npz = raw / "parihaka_labels_train.npz"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    d_shape, d_dtype, _ = array_info(data_npz, "data")
    l_shape, l_dtype, _ = array_info(lab_npz, "labels")
    if d_shape != l_shape:
        raise SystemExit(f"Shape mismatch: data {d_shape} vs labels {l_shape}")
    print(f"data {d_shape} {d_dtype} | labels {l_shape} {l_dtype}")

    # Pass 1: which label values exist? (labels are small, this is quick)
    t0 = time.time()
    values = set()
    for _, blk in iter_slabs(lab_npz, "labels", args.slab):
        values.update(np.unique(blk).tolist())
    values = sorted(values)
    offset = values[0] if values[0] in (0, 1) else 0
    n_classes = values[-1] - offset + 1
    print(f"Label values found: {values} -> shifting by {offset}, {n_classes} classes "
          f"({time.time() - t0:.0f}s)")

    # Pass 2: convert and collect statistics
    chunks = tuple(min(args.chunk, s) for s in d_shape)
    root = zarr.open_group(str(out), mode="w")
    seis = make_array(root, "seismic", d_shape, chunks, "float32")
    labs = make_array(root, "labels", d_shape, chunks, "uint8")

    n_vox = 0
    s1 = s2 = 0.0
    n_bad = 0
    sample = []
    counts = np.zeros(n_classes, dtype=np.int64)
    first = last = None

    t0 = time.time()
    y_total = d_shape[2]
    slabs = zip(iter_slabs(data_npz, "data", args.slab),
                iter_slabs(lab_npz, "labels", args.slab))
    for (y0, block), (y0_l, lblock) in slabs:
        assert y0 == y0_l
        n = block.shape[2]
        finite = np.isfinite(block)
        if not finite.all():
            n_bad += int((~finite).sum())
            block = np.where(finite, block, 0).astype(np.float32)
        lblock = (lblock.astype(np.int16) - offset).astype(np.uint8)

        n_vox += block.size
        s1 += float(block.sum(dtype=np.float64))
        s2 += float(np.square(block, dtype=np.float64).sum())
        sample.append(block[::4, ::4, ::2].ravel())
        counts += np.bincount(lblock.ravel(), minlength=n_classes)[:n_classes]

        seis[:, :, y0:y0 + n] = block
        labs[:, :, y0:y0 + n] = lblock
        if y0 == 0:
            first = (n, block.copy(), lblock.copy())
        last = (y0, n, block.copy(), lblock.copy())

        if (y0 // args.slab) % 10 == 0:
            print(f"  y {y0 + n}/{y_total}  ({time.time() - t0:.0f}s)", flush=True)

    mean = s1 / n_vox
    std = (s2 / n_vox - mean ** 2) ** 0.5
    p1, p99 = np.percentile(np.concatenate(sample), [1, 99])

    # Verify: read back the first and last slabs and compare with the source
    n0, b0, l0 = first
    ly0, ln, bl, ll = last
    assert np.array_equal(seis[:, :, 0:n0], b0), "seismic mismatch (first slab)"
    assert np.array_equal(labs[:, :, 0:n0], l0), "label mismatch (first slab)"
    assert np.array_equal(seis[:, :, ly0:ly0 + ln], bl), "seismic mismatch (last slab)"
    assert np.array_equal(labs[:, :, ly0:ly0 + ln], ll), "label mismatch (last slab)"
    assert counts.sum() == n_vox, "class counts do not add up"

    root.attrs.update({
        "axes": ["z", "x", "y"],
        "shape": list(d_shape),
        "source": "parihaka_data_train.npz / parihaka_labels_train.npz "
                  "(Mendeley Data DOI 10.17632/gnvyh3msrj.1, from AIcrowd Seismic "
                  "Facies Identification Challenge)",
        "original_label_values": [int(v) for v in values],
        "label_offset": int(offset),
        "n_classes": int(n_classes),
        "class_counts": [int(c) for c in counts],
        "class_fractions": [float(c / n_vox) for c in counts],
        "seismic_mean": float(mean),
        "seismic_std": float(std),
        "seismic_p1": float(p1),
        "seismic_p99": float(p99),
        "n_nonfinite_replaced_with_zero": int(n_bad),
    })

    print("\nDone. Verification passed (first and last slabs match the source).")
    print(f"Seismic: mean {mean:.4g} | std {std:.4g} | p1 {p1:.4g} | p99 {p99:.4g} "
          f"| non-finite values replaced: {n_bad}")
    print("Class fractions (class index: share of voxels):")
    for i, c in enumerate(counts):
        print(f"  {i}: {c / n_vox:.4f}  ({int(c):,})")
    print(f"Store written to {out}")


if __name__ == "__main__":
    main()
