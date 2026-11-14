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
