#nothing to see here lmao lock in 🤨😀😅🙃
## Data
Parihaka seismic and facies labels come from a Mendeley Data redistribution
(DOI 10.17632/gnvyh3msrj.1) of the AIcrowd / SEAM AI Parihaka Facies Challenge data.
The challenge page describes the labels (Chevron) as CC BY-SA 4.0, while the
Mendeley record says CC BY 4.0. Assume the stricter share-alike terms until reviewed.
Raw data and the Zarr store are not in git (see .gitignore).

## Pipeline
1. `python src/convert_parihaka.py` -> data/zarr/parihaka.zarr
2. `python src/make_splits.py --blocks 5 --buffer 64` -> configs/split.json (frozen, do not edit)
3. `python src/baselines.py` -> results/baselines.json

## Baselines (blind val/test strips, mean IoU)
Majority class: 0.072 / 0.066. Depth-only lookup: 0.293 / 0.246.

## Results (5-seed ResNet-34 U-Net ensemble, frozen blind val/test strips)
| | Val mIoU | Test mIoU |
|---|---|---|
| Majority class | 0.072 | 0.066 |
| Depth-only lookup | 0.293 | 0.246 |
| Single model (mean, range of 5 seeds) | 0.812 (0.798-0.822) | 0.815 (0.810-0.828) |
| Ensemble of 5 | 0.826 | 0.836 |

Test per-class IoU (ensemble): basement 0.95, mudstone A 0.88, mass transport 0.73,
mudstone B 0.97, slope valley 0.60, canyon 0.88. Probability quality improves with ensembling
(test NLL 0.216 -> 0.156, ECE 0.035 -> 0.013). Ensemble entropy detects errors with AUROC 0.90;
keeping the most certain 80% of voxels gives 98.9% accuracy (95.1% unfiltered).
Accuracy declines gradually with distance from training sections (test mIoU 0.875 -> 0.804
across 65-220 traces) while uncertainty rises.

Caveats: one survey, one split; the test strip is no longer unseen (not tuned on). Run-to-run
variation can exceed the seed spread. Valley and mass-transport bodies are more often missed
than invented. Labels are one interpreter's picks. Distance effects are confounded with geology.
