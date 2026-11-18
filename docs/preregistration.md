# Pre-registration (DRAFT, not locked)

Project: reliability of seismic facies interpretation under geological and acquisition shift.
Status: **DRAFT.** Lock date: ____ . Commit hash at lock: ____ .
Keep this file in `docs/` and in sync with the research plan. Once locked, this file governs.

## 1. Hypotheses and primary metrics

| | Hypothesis | Primary metric (one per hypothesis) | Supported if (PROVISIONAL) |
|---|---|---|---|
| H1 | The generalization gap shrinks as training diversity grows | Gap = within-survey blind-block mIoU minus held-out-condition mIoU, same model | Gap at 3+ training surveys is smaller than at 1 by more than the run-to-run range seen in the pilot (about 0.03 mIoU) |
| H2 | Ensemble uncertainty ranks held-out regions by true error better than simple baselines | Region-level Spearman correlation between ensemble predictive entropy and region error (interior voxels only, boundary density partialled out; region = 10 sections; contact radius k = 3) | rho >= 0.5 in most tested ladder levels AND exceeds the better of {single-model entropy, single-model max-probability} by >= 0.1, bootstrap interval over regions excludes zero |
| H3 | The signal holds under covariate shift, weakens under geological novelty, fails under label-convention shift | Region-level AUROC for flagging above-median-error regions | AUROC >= 0.7 under covariate shift (L2a) and <= 0.6 under label-convention shift (L2b) |
| H4 | Uncertainty-guided selection needs fewer labels than random | Labelled sections needed to reach 95% of fully supervised mIoU in a new region | At least 20% fewer labelled sections than random, outside seed spread |

All thresholds above are placeholders. They are reviewed after the leave-one-strip-out results (plan week 6) and then locked. A refuted hypothesis is reported as a result.

## 2. Protocol commitments

- Model: ResNet-34 U-Net on 2D sections; training settings as in `src/train_unet.py` at the lock commit. Architecture changes only as a labelled sensitivity check.
- At least 5 seeds per condition; ensembles of 5 members.
- Splits and folds are frozen in `configs/` and hashed in `results/registry.jsonl`. Nothing is tuned on a test condition; model selection uses validation data only.
- The region, not the voxel, is the unit of analysis for reliability claims. Voxel-level AUROC is reported together with its interior-only version.
- Analyses run with the scripts at the lock commit. Any analysis added later is labelled **exploratory**.
- Software versions and the git commit are recorded for every run (`src/registry.py`).

## 3. Shift ladder (levels tested)

L0 blind strip (done) · L1 leave-one-strip-out · L2a simulated acquisition shift · L2b simulated label-convention shift · L2c leave-one-class-out · L3 new survey · L4 new basin (stretch).

## 4. Limitations stated in advance

One interpreter's labels; few labelled surveys; neighbouring sections are not independent; distance from training data is confounded with geology; run-to-run variation can exceed seed spread across platforms.

## 5. Amendments

| Date | Change | Reason |
|---|---|---|
| | | |

## 6. Lock checklist

- [ ] Leave-one-strip-out results reviewed; thresholds revised if clearly unrealistic
- [ ] Literature check complete (no identical prior study)
- [ ] Data and licence register up to date
- [ ] Commit with message "Lock pre-registration v1"; record the hash above
