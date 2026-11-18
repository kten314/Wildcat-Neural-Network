"""Summarise leave-one-strip-out (LOSO) runs: one ensemble_eval JSON per fold.

Reads results/ensemble_<prefix><k>.json for k = 0, 1, 2, ... and reports, across folds:
  * ensemble vs single-model accuracy, calibration, per-class IoU
  * how well each uncertainty score flags errors (voxel AUROC, with and without
    the layer-contact voxels; region-level rank correlation with boundary density
    partialled out), including the single-model baselines
  * an indicative comparison with the provisional H2 criteria (NOT locked)

Usage (repo root):
    python src/aggregate_folds.py --prefix fold
Writes results/loso_summary.json and appends an entry to results/registry.jsonl.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from registry import log_event

SCORES = ("ens_entropy", "ens_mi", "single_entropy", "single_maxprob")
LABEL = {"ens_entropy": "ensemble entropy", "ens_mi": "ensemble disagreement (MI)",
         "single_entropy": "single-model entropy", "single_maxprob": "single-model max-prob"}
BASELINES = ("single_entropy", "single_maxprob")


def ms(values):
    v = np.array([np.nan if x is None else x for x in values], float)
    ok = np.isfinite(v)
    if not ok.any():
        return float("nan"), float("nan"), 0
    return float(v[ok].mean()), float(v[ok].std(ddof=1)) if ok.sum() > 1 else float("nan"), int(ok.sum())


def fmt(m, s, n, digits=3):
    if n == 0:
        return "n/a"
    return f"{m:.{digits}f}" + (f" ± {s:.{digits}f}" if n > 1 else "") + (f"  (n={n})" if n < 5 else "")


def load_folds(results_dir, prefix):
    out, k = [], 0
    while (Path(results_dir) / f"ensemble_{prefix}{k}.json").exists():
        out.append(json.load(open(Path(results_dir) / f"ensemble_{prefix}{k}.json")))
        k += 1
    return out


def fold_row(doc):
    r = doc["results"]["test"]
    rc = r["reliability_controls"]
    reg = rc["region"]["scores"]
    return {
        "ens_miou": r["ensemble"]["mean_iou"],
        "single_miou": float(np.mean(r["members"]["mean_iou"])),
        "ens_ece": r["ensemble"]["ece"], "single_ece": float(np.mean(r["members"]["ece"])),
        "ens_nll": r["ensemble"]["nll"], "single_nll": float(np.mean(r["members"]["nll"])),
        "per_class_iou": r["ensemble"]["per_class_iou"],
        "share_errors_at_boundaries": rc["share_of_errors_at_boundaries"],
        "voxel_auroc": {k: {"all": rc["voxel_auroc"][k]["all"]["mean"], "int": rc["voxel_auroc"][k]["int"]["mean"]}
                        for k in SCORES},
        "region": {k: {"spearman_all": reg[k]["spearman_all"], "spearman_int": reg[k]["spearman_int"],
                       "partial_int": reg[k]["partial_int"]} for k in SCORES},
        "n_regions": rc["region"]["n_regions"],
        "distance_spearman_miou": r["distance"]["spearman_miou"],
        "class_names": doc.get("class_names"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="fold")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/loso_summary.json")
    args = ap.parse_args()

    docs = load_folds(args.results_dir, args.prefix)
    if not docs:
        raise SystemExit(f"No files named {args.results_dir}/ensemble_{args.prefix}<k>.json found")
    rows = [fold_row(d) for d in docs]
    K = len(rows)
    print(f"{K} folds found: " + ", ".join(f"{args.prefix}{k}" for k in range(K)))

    print(f"\n{'fold':<6}{'single mIoU':>12}{'ensemble':>10}{'ECE s→e':>16}{'AUROC all|int (ens entropy)':>30}"
          f"{'regions':>9}")
    for k, r in enumerate(rows):
        a = r["voxel_auroc"]["ens_entropy"]
        print(f"{k:<6}{r['single_miou']:>12.3f}{r['ens_miou']:>10.3f}{r['single_ece']:>10.3f} → {r['ens_ece']:.3f}"
              f"{a['all']:>14.3f} | {a['int']:.3f}{r['n_regions']:>11}")

    print("\n=== Across folds (mean ± sd) ===")
    print(f"single-model mIoU       {fmt(*ms([r['single_miou'] for r in rows]))}")
    print(f"ensemble mIoU           {fmt(*ms([r['ens_miou'] for r in rows]))}")
    print(f"ensemble gain           {fmt(*ms([r['ens_miou'] - r['single_miou'] for r in rows]))}")
    print(f"calibration error       {fmt(*ms([r['single_ece'] for r in rows]))} → {fmt(*ms([r['ens_ece'] for r in rows]))}")
    print(f"NLL                     {fmt(*ms([r['single_nll'] for r in rows]))} → {fmt(*ms([r['ens_nll'] for r in rows]))}")
    names = rows[0]["class_names"] or [str(i) for i in range(len(rows[0]["per_class_iou"]))]
    print("per-class IoU (ensemble):")
    per_class = {}
    for i, n in enumerate(names):
        per_class[n] = ms([r["per_class_iou"][i] for r in rows])
        print(f"  {n:<26}{fmt(*per_class[n])}")
    print(f"share of errors at label contacts: {fmt(*ms([r['share_errors_at_boundaries'] for r in rows]), digits=2)}")
    print(f"accuracy vs distance, Spearman (per fold): {[round(r['distance_spearman_miou'], 2) for r in rows]}")

    print("\n=== Do uncertainty scores flag errors? (mean across folds) ===")
    print(f"{'score':<28}{'voxel AUROC all':>17}{'interior only':>17}{'region ρ (interior, partial)':>32}")
    summary_scores = {}
    for k in SCORES:
        va = ms([r["voxel_auroc"][k]["all"] for r in rows])
        vi = ms([r["voxel_auroc"][k]["int"] for r in rows])
        rp = ms([r["region"][k]["partial_int"] for r in rows])
        summary_scores[k] = {"voxel_auroc_all": va, "voxel_auroc_interior": vi, "region_partial_interior": rp}
        print(f"{LABEL[k]:<28}{fmt(*va):>17}{fmt(*vi):>17}{fmt(*rp):>32}")

    # indicative check against the provisional H2 criteria (thresholds are NOT locked)
    ens_p = np.array([r["region"]["ens_entropy"]["partial_int"] for r in rows], float)
    best_base = np.array([np.nanmax([r["region"][b]["partial_int"] for b in BASELINES]) for r in rows], float)
    diff = ens_p - best_base
    ok = np.isfinite(diff)
    n_thr = int((ens_p[np.isfinite(ens_p)] >= 0.5).sum())
    n_beat = int((diff[ok] >= 0.1).sum())
    print("\n=== Indicative check against the PROVISIONAL H2 criteria (not locked) ===")
    print(f"region-level rank correlation >= 0.5 in {n_thr} of {int(np.isfinite(ens_p).sum())} folds")
    print(f"ensemble entropy beats the best single-model baseline by >= 0.1 in {n_beat} of {int(ok.sum())} folds "
          f"(mean difference {fmt(*ms(diff))})")
    print("Caveat: few, spatially correlated regions per fold; five folds are not five independent experiments.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "prefix": args.prefix, "n_folds": K, "folds": rows,
        "across_folds": {"single_miou": ms([r["single_miou"] for r in rows]), "ensemble_miou": ms([r["ens_miou"] for r in rows]),
                         "per_class_iou": {n: list(v) for n, v in per_class.items()},
                         "scores": {k: {a: list(b) for a, b in v.items()} for k, v in summary_scores.items()}},
        "h2_indicative": {"folds_region_rho_ge_0.5": n_thr, "folds_beating_best_baseline_by_0.1": n_beat,
                          "paired_difference_per_fold": diff.tolist()},
    }, indent=2, default=float))
    log_event("aggregate_folds", {"prefix": args.prefix, "n_folds": K,
                                  "ensemble_miou_mean": ms([r["ens_miou"] for r in rows])[0]}, files=[args.out])
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
