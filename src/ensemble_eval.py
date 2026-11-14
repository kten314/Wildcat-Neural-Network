"""Evaluate a deep ensemble of U-Nets on the frozen blind val/test strips.

Loads every checkpoints/<tag>_seed*.pt, averages the members' softmax outputs,
and reports:
  1. ensemble vs single-model mean IoU, NLL and calibration error (ECE)
  2. whether uncertainty flags errors (entropy and mutual information):
     AUROC, error rate per uncertainty bin, accuracy vs coverage
  3. accuracy vs distance from the nearest TRAINING section, per section
     (does the model get worse the further it extrapolates?)
It also draws an uncertainty gallery for a few test sections.

All statistics are plain NumPy (see accumulate_batch and the summary helpers);
only model loading and forward passes touch PyTorch.

Usage (repo root, GPU machine, after training several seeds with one --tag):
    python src/ensemble_eval.py --tag ens
Writes results/ensemble_<tag>.json and figures in results/figures/.
"""
import argparse
import glob
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import zarr
import segmentation_models_pytorch as smp

from train_unet import CLASS_NAMES, SectionBank, predict

NB_CONF = 15                                                    # confidence bins for ECE
EDGES = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 120)])   # uncertainty bin upper edges
NBINS = len(EDGES) + 1


# ============================================================ NumPy statistics
def entropy(p, axis):
    return -(p * np.log(np.maximum(p, 1e-12))).sum(axis=axis)


class Acc:
    """Running totals for one split."""

    def __init__(self, n_members, n_classes):
        M, C = n_members, n_classes
        self.M, self.C = M, C
        self.cm_ens = np.zeros((C, C), np.int64)
        self.cm_mem = np.zeros((M, C, C), np.int64)
        self.sec_cm, self.sec_H, self.sec_MI, self.sec_err = [], [], [], []
        self.h_tot = {"H": np.zeros(NBINS, np.int64), "MI": np.zeros(NBINS, np.int64)}
        self.h_err = {"H": np.zeros(NBINS, np.int64), "MI": np.zeros(NBINS, np.int64)}
        self.conf_ens = np.zeros((3, NB_CONF))        # rows: count, correct, sum of confidence
        self.conf_mem = np.zeros((M, 3, NB_CONF))
        self.nll_ens = 0.0
        self.nll_mem = np.zeros(M)
        self.n_vox = 0


def _conf_bins(conf, correct):
    cb = np.minimum((conf.ravel() * NB_CONF).astype(np.int64), NB_CONF - 1)
    return np.stack([np.bincount(cb, minlength=NB_CONF).astype(float),
                     np.bincount(cb, weights=correct.ravel().astype(float), minlength=NB_CONF),
                     np.bincount(cb, weights=conf.ravel().astype(float), minlength=NB_CONF)])


def accumulate_batch(acc, P, truth):
    """P: (M, B, C, Z, L) member softmax probabilities; truth: (B, Z, L) int labels."""
    M, B, C = P.shape[:3]
    Pbar = P.mean(axis=0)                               # (B, C, Z, L)
    conf, pred = Pbar.max(axis=1), Pbar.argmax(axis=1)  # (B, Z, L)
    logC = np.log(C)
    H = entropy(Pbar, 1) / logC                         # predictive entropy, 0..1
    MI = np.maximum(H - entropy(P, 2).mean(axis=0) / logC, 0.0)   # member disagreement
    err = pred != truth

    for b in range(B):
        cm_b = np.bincount((truth[b] * C + pred[b]).ravel(), minlength=C * C).reshape(C, C)
        acc.cm_ens += cm_b
        acc.sec_cm.append(cm_b)
        acc.sec_H.append(float(H[b].mean()))
        acc.sec_MI.append(float(MI[b].mean()))
        acc.sec_err.append(float(err[b].mean()))

    pm = P.argmax(axis=2)                               # (M, B, Z, L)
    pm_conf = P.max(axis=2)
    for m in range(M):
        acc.cm_mem[m] += np.bincount((truth * C + pm[m]).ravel(), minlength=C * C).reshape(C, C)
        acc.conf_mem[m] += _conf_bins(pm_conf[m], pm[m] == truth)
    acc.conf_ens += _conf_bins(conf, ~err)

    for key, u in (("H", H), ("MI", MI)):
        bi = np.searchsorted(EDGES, u.ravel(), side="left")
        acc.h_tot[key] += np.bincount(bi, minlength=NBINS)
        acc.h_err[key] += np.bincount(bi[err.ravel()], minlength=NBINS)

    p_true = np.take_along_axis(Pbar, truth[:, None], axis=1)
    acc.nll_ens += float(-np.log(np.maximum(p_true, 1e-12)).sum(dtype=np.float64))
    pm_true = np.take_along_axis(P, truth[None, :, None], axis=2)
    acc.nll_mem += -np.log(np.maximum(pm_true, 1e-12)).sum(axis=(1, 2, 3, 4), dtype=np.float64)
    acc.n_vox += truth.size


def iou_from_cm(cm):
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1) - tp
    return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)


def miou(cm):
    iou = iou_from_cm(cm)
    return float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan")


def pixacc(cm):
    return float(np.diag(cm).sum() / max(cm.sum(), 1))


def ece_from_bins(bins):
    count, correct, conf_sum = bins
    m = count > 0
    acc_b = np.where(m, correct / np.maximum(count, 1), np.nan)
    conf_b = np.where(m, conf_sum / np.maximum(count, 1), np.nan)
    ece = float((count[m] / count.sum() * np.abs(acc_b[m] - conf_b[m])).sum())
    return ece, acc_b, conf_b


def uncertainty_curves(tot, err):
    """Error-detection quality from uncertainty histograms (higher bin = more uncertain)."""
    tot, err = tot.astype(float), err.astype(float)
    ok = tot - err
    E, K = err.sum(), ok.sum()
    before = np.cumsum(ok) - ok
    auroc = float((np.sum(err * before) + 0.5 * np.sum(err * ok)) / (E * K)) if E > 0 and K > 0 else float("nan")
    cum = np.cumsum(tot)
    return {"auroc": auroc, "coverage": cum / tot.sum(),
            "accuracy": 1.0 - np.cumsum(err) / np.maximum(cum, 1),
            "error_rate": np.where(tot > 0, err / np.maximum(tot, 1), np.nan)}


def accuracy_at_coverage(curves, target):
    i = min(int(np.searchsorted(curves["coverage"], target)), len(curves["coverage"]) - 1)
    return float(curves["accuracy"][i])


def nearest_distance(idx, train_idx):
    t = np.sort(np.asarray(train_idx))
    idx = np.asarray(idx)
    pos = np.searchsorted(t, idx)
    left = t[np.clip(pos - 1, 0, len(t) - 1)]
    right = t[np.clip(pos, 0, len(t) - 1)]
    return np.minimum(np.abs(idx - left), np.abs(right - idx))


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a)); ranks[order] = np.arange(len(a))
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=ranks) / counts)[inv]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[m]), rankdata(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(acc, index, train_idx):
    C, M = acc.C, acc.M
    cm = acc.cm_ens
    mem_miou = [miou(c) for c in acc.cm_mem]
    ens_ece, rel_acc, rel_conf = ece_from_bins(acc.conf_ens)
    mem_ece = [ece_from_bins(b)[0] for b in acc.conf_mem]
    cur = {k: uncertainty_curves(acc.h_tot[k], acc.h_err[k]) for k in ("H", "MI")}

    dist = nearest_distance(index, train_idx)
    s_miou = np.array([miou(c) for c in acc.sec_cm])
    s_acc = np.array([pixacc(c) for c in acc.sec_cm])
    order = np.argsort(dist, kind="stable")
    quartiles = []
    for g in np.array_split(order, 4):
        quartiles.append({"distance_mean": float(dist[g].mean()), "distance_min": int(dist[g].min()),
                          "distance_max": int(dist[g].max()), "n_sections": int(len(g)),
                          "mean_iou": float(np.nanmean(s_miou[g])), "pixel_accuracy": float(s_acc[g].mean()),
                          "mean_entropy": float(np.mean(np.array(acc.sec_H)[g]))})
    return {
        "n_members": M,
        "ensemble": {"mean_iou": miou(cm), "pixel_accuracy": pixacc(cm),
                     "per_class_iou": [None if np.isnan(v) else float(v) for v in iou_from_cm(cm)],
                     "nll": acc.nll_ens / acc.n_vox, "ece": ens_ece, "confusion_matrix": cm.tolist()},
        "members": {"mean_iou": mem_miou, "nll": (acc.nll_mem / acc.n_vox).tolist(), "ece": mem_ece,
                    "per_class_iou_mean": np.nanmean([iou_from_cm(c) for c in acc.cm_mem], axis=0).tolist()},
        "reliability": {"accuracy": rel_acc.tolist(), "confidence": rel_conf.tolist(),
                        "member0_accuracy": ece_from_bins(acc.conf_mem[0])[1].tolist(),
                        "member0_confidence": ece_from_bins(acc.conf_mem[0])[2].tolist()},
        "uncertainty": {k: {"auroc": cur[k]["auroc"],
                            "accuracy_at_coverage": {str(c): accuracy_at_coverage(cur[k], c) for c in (0.8, 0.9, 0.95)}}
                        for k in cur},
        "overall_accuracy": pixacc(cm),
        "distance": {"spearman_miou": spearman(dist, s_miou), "spearman_pixel_accuracy": spearman(dist, s_acc),
                     "spearman_entropy": spearman(dist, acc.sec_H), "quartiles": quartiles},
        "sections": [{"trace": int(t), "distance_to_train": int(d), "mean_iou": None if np.isnan(a) else float(a),
                      "pixel_accuracy": float(p), "mean_entropy": float(h), "mean_mutual_information": float(mi)}
                     for t, d, a, p, h, mi in zip(index, dist, s_miou, s_acc, acc.sec_H, acc.sec_MI)],
        "_curves": cur,   # kept for plotting, removed before JSON export
    }


# ================================================================== PyTorch part
@torch.no_grad()
def member_probs(models, x, device):
    """x: (B,1,Z,L) -> softmax probabilities (M,B,C,Z,L) as a NumPy array."""
    return np.stack([torch.softmax(predict(m, x, device), dim=1).cpu().numpy() for m in models])


def evaluate_split(models, bank, C, device, batch):
    acc = Acc(len(models), C)
    n, t0 = len(bank), time.time()
    for i in range(0, n, batch):
        sel = list(range(i, min(i + batch, n)))
        x = torch.from_numpy(bank.seis[sel].astype(np.float32))[:, None].to(device)
        accumulate_batch(acc, member_probs(models, x, device), bank.lab[sel].astype(np.int64))
        if (i // batch) % 20 == 0:
            print(f"    {sel[-1] + 1}/{n} sections ({time.time() - t0:.0f}s)", flush=True)
    return acc


# ====================================================================== figures
def make_figures(res, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, v = res["test"], res["val"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for r, color, label in ((v, "tab:blue", "val"), (t, "tab:red", "test")):
        d = [s["distance_to_train"] for s in r["sections"]]
        ax[0].scatter(d, [np.nan if s["mean_iou"] is None else s["mean_iou"] for s in r["sections"]],
                      s=10, alpha=0.5, color=color, label=f"{label} sections")
        ax[1].scatter(d, [s["mean_entropy"] for s in r["sections"]], s=10, alpha=0.5, color=color, label=label)
        q = r["distance"]["quartiles"]
        ax[0].plot([x["distance_mean"] for x in q], [x["mean_iou"] for x in q], "o-", color=color, lw=2,
                   label=f"{label} quartile means")
    ax[0].set_xlabel("distance to nearest training trace"); ax[0].set_ylabel("per-section mean IoU")
    ax[0].set_title(f"Accuracy vs distance (test Spearman {t['distance']['spearman_miou']:.2f})", fontsize=10)
    ax[1].set_xlabel("distance to nearest training trace"); ax[1].set_ylabel("mean ensemble entropy (0-1)")
    ax[1].set_title(f"Uncertainty vs distance (test Spearman {t['distance']['spearman_entropy']:.2f})", fontsize=10)
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{prefix}_distance.png", dpi=110); plt.close(fig)

    cur = t["_curves"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    for k, c in (("H", "tab:green"), ("MI", "tab:purple")):
        lab = "entropy" if k == "H" else "member disagreement (MI)"
        ax[0, 0].plot(EDGES[1:], cur[k]["error_rate"][1:len(EDGES)], color=c, label=f"{lab}, AUROC {cur[k]['auroc']:.3f}")
        ax[0, 1].plot(cur[k]["coverage"], cur[k]["accuracy"], color=c, label=lab)
    ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("normalised uncertainty bin"); ax[0, 0].set_ylabel("error rate")
    ax[0, 0].set_title("Error rate by uncertainty (test)", fontsize=10); ax[0, 0].legend(fontsize=7)
    ax[0, 1].axhline(t["overall_accuracy"], color="gray", ls="--", label="no filtering")
    ax[0, 1].set_xlabel("coverage (fraction of voxels kept, most certain first)"); ax[0, 1].set_ylabel("pixel accuracy")
    ax[0, 1].set_xlim(0.5, 1.0); ax[0, 1].set_title("Accuracy vs coverage (test)", fontsize=10); ax[0, 1].legend(fontsize=7)
    rel = t["reliability"]
    ax[1, 0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1, 0].plot(rel["confidence"], rel["accuracy"], "o-", color="tab:red", label=f"ensemble (ECE {t['ensemble']['ece']:.4f})")
    ax[1, 0].plot(rel["member0_confidence"], rel["member0_accuracy"], "s-", color="tab:blue",
                  label=f"single model (ECE {t['members']['ece'][0]:.4f})")
    ax[1, 0].set_xlabel("confidence"); ax[1, 0].set_ylabel("accuracy"); ax[1, 0].set_title("Reliability (test)", fontsize=10)
    ax[1, 0].legend(fontsize=7)
    w = 0.38; xs = np.arange(len(t["ensemble"]["per_class_iou"]))
    ax[1, 1].bar(xs - w / 2, t["members"]["per_class_iou_mean"], w, label="single (mean of members)")
    ax[1, 1].bar(xs + w / 2, [0 if v is None else v for v in t["ensemble"]["per_class_iou"]], w, label="ensemble")
    ax[1, 1].set_xticks(xs); ax[1, 1].set_xticklabels([n[:12] for n in CLASS_NAMES[:len(xs)]], rotation=30, fontsize=8)
    ax[1, 1].set_ylim(0, 1.15); ax[1, 1].set_ylabel("IoU")
    ax[1, 1].set_title("Per-class IoU (test)", fontsize=10); ax[1, 1].legend(fontsize=7, ncol=2, loc="upper center")
    fig.tight_layout(); fig.savefig(f"{prefix}_uncertainty.png", dpi=110); plt.close(fig)


def uncertainty_gallery(models, bank, dist, device, path, n, C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(plt.get_cmap("tab10").colors[:C])
    picks = np.linspace(0, len(bank) - 1, n).astype(int)
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.4 * n), squeeze=False)
    for r, k in enumerate(picks):
        x = torch.from_numpy(bank.seis[[k]].astype(np.float32))[:, None].to(device)
        P = member_probs(models, x, device)
        Pbar = P.mean(axis=0)[0]
        pred = Pbar.argmax(0)
        H = entropy(Pbar, 0) / np.log(C)
        truth = bank.lab[k]
        panels = [(bank.seis[k].astype(np.float32), dict(cmap="gray", vmin=-3, vmax=3), f"seismic | trace {bank.index[k]}"),
                  (truth, dict(cmap=cmap, vmin=0, vmax=C - 1, interpolation="nearest"), "truth"),
                  (pred, dict(cmap=cmap, vmin=0, vmax=C - 1, interpolation="nearest"),
                   f"ensemble prediction ({len(models)} members)"),
                  (H, dict(cmap="magma", vmin=0, vmax=max(0.05, float(np.percentile(H, 99.5)))),
                   f"uncertainty (entropy) | {dist[k]} traces from training"),
                  (pred != truth, dict(cmap="Reds", vmin=0, vmax=1, interpolation="nearest"), "errors")]
        for c, (img, kw, title) in enumerate(panels):
            axes[r, c].imshow(img, aspect="auto", **kw)
            axes[r, c].set_title(title, fontsize=9); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110); plt.close(fig)


# ======================================================================== main
def fmt(v):
    return "  n/a" if v is None else f"{v:5.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ens")
    ap.add_argument("--zarr", default="data/zarr/parihaka.zarr")
    ap.add_argument("--split", default="configs/split.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--figs", type=int, default=3, help="sections in the uncertainty gallery (0 = none)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = zarr.open_group(args.zarr, mode="r")
    split = json.load(open(args.split))
    C = int(root.attrs["n_classes"])
    mean, std = float(root.attrs["seismic_mean"]), float(root.attrs["seismic_std"])

    paths = sorted(glob.glob(f"{args.ckpt_dir}/{args.tag}_seed*.pt"))
    if not paths:
        raise SystemExit(f"No checkpoints matching {args.ckpt_dir}/{args.tag}_seed*.pt")
    print(f"device {device} | {len(paths)} ensemble members: {[Path(p).name for p in paths]}")
    if len(paths) < 2:
        print("NOTE: with one member, ensemble disagreement (MI) is zero and 'ensemble' = single model.")
    models = []
    for p in paths:
        m = smp.Unet(args.encoder, encoder_weights=None, in_channels=1, classes=C)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        models.append(m.to(device).eval())

    banks = {n: SectionBank(root, split, n, mean, std) for n in ("val", "test")}
    train_idx = np.concatenate([np.arange(a, b) for a, b in split["ranges"]["train"]])
    res = {}
    for name in ("val", "test"):
        print(f"evaluating {name} ({len(banks[name])} sections)...")
        res[name] = summarize(evaluate_split(models, banks[name], C, device, args.batch),
                              banks[name].index, train_idx)

    # ---------------------------------------------------------------- report
    print("\n=== Ensemble vs single model ===")
    print(f"{'split':<6}{'single mIoU (mean±sd)':>26}{'ensemble mIoU':>15}{'pixAcc':>8}"
          f"{'NLL single→ens':>18}{'ECE single→ens':>20}")
    for name in ("val", "test"):
        r = res[name]; mm = r["members"]
        print(f"{name:<6}{np.mean(mm['mean_iou']):>17.3f} ± {np.std(mm['mean_iou']):.3f}"
              f"{r['ensemble']['mean_iou']:>15.3f}{r['ensemble']['pixel_accuracy']:>8.3f}"
              f"{np.mean(mm['nll']):>10.4f} → {r['ensemble']['nll']:.4f}"
              f"{np.mean(mm['ece']):>12.4f} → {r['ensemble']['ece']:.4f}")
    t = res["test"]
    print("\nEnsemble per-class IoU (test): " + " ".join(fmt(v) for v in t["ensemble"]["per_class_iou"]))
    print("\n=== Does uncertainty flag errors? (test) ===")
    for k, lab in (("H", "entropy"), ("MI", "member disagreement")):
        u = t["uncertainty"][k]; a = u["accuracy_at_coverage"]
        print(f"{lab:<20} AUROC {u['auroc']:.3f} | accuracy keeping 95%/90%/80% most certain voxels: "
              f"{a['0.95']:.3f} / {a['0.9']:.3f} / {a['0.8']:.3f}  (no filtering: {t['overall_accuracy']:.3f})")
    print("\n=== Accuracy vs distance from nearest training trace ===")
    for name in ("val", "test"):
        d = res[name]["distance"]
        print(f"{name}: Spearman(distance, mIoU) {d['spearman_miou']:.2f} | (distance, pixAcc) "
              f"{d['spearman_pixel_accuracy']:.2f} | (distance, entropy) {d['spearman_entropy']:.2f}")
    print(f"{'test quartile':<16}{'distance':>14}{'sections':>10}{'mIoU':>8}{'pixAcc':>8}{'entropy':>9}")
    for i, q in enumerate(t["distance"]["quartiles"], 1):
        print(f"{'Q' + str(i):<16}{str(q['distance_min']) + '-' + str(q['distance_max']):>14}{q['n_sections']:>10}"
              f"{q['mean_iou']:>8.3f}{q['pixel_accuracy']:>8.3f}{q['mean_entropy']:>9.4f}")

    # ---------------------------------------------------------------- outputs
    curves = {n: res[n]["_curves"] for n in res}
    for n in res:
        res[n].pop("_curves")
    out = Path("results") / f"ensemble_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"tag": args.tag, "members": [Path(p).name for p in paths],
                               "class_names": CLASS_NAMES, "results": res}, indent=2))
    for n in res:
        res[n]["_curves"] = curves[n]
    prefix = f"results/figures/ensemble_{args.tag}"
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    make_figures(res, prefix)
    print(f"\nWrote {out}, {prefix}_distance.png, {prefix}_uncertainty.png")
    if args.figs > 0:
        td = nearest_distance(banks["test"].index, train_idx)
        uncertainty_gallery(models, banks["test"], td, device, f"{prefix}_gallery.png", args.figs, C)
        print(f"Wrote {prefix}_gallery.png")


if __name__ == "__main__":
    main()
