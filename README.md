# crop-disease-classification
# Crop Disease Detection using Vision Transformers

Trained and compared 5 Vision Transformer architectures on a plant disease
classification dataset, trained on Kaggle Notebooks (free GPU) since local
hardware wasn't sufficient for training.

## Project Status: Complete

All 5 architectures from the proposal have been trained and evaluated on
the full validation set:

| Model    | timm identifier                | Accuracy | Precision | Recall | F1     | ROC AUC |
|----------|----------------------------------|----------|-----------|--------|--------|---------|
| CAFormer | caformer_s18                    | 99.54%   | 99.42%    | 99.31% | 99.36% | 1.0000  |
| Swin     | swin_base_patch4_window7_224    | 99.22%   | 99.15%    | 99.02% | 99.06% | 1.0000  |
| DeiT     | deit_base_patch16_224           | 99.14%   | 98.99%    | 99.00% | 98.98% | 1.0000  |
| PiT      | pit_b_224                       | 98.74%   | 98.10%    | 98.26% | 98.12% | 1.0000  |
| ViT      | vit_base_patch16_224            | 97.67%   | 97.20%    | 97.25% | 97.16% | 0.9999  |

**Notes:**
- All 5 models performed strongly (under 2 percentage points spread top to
  bottom), so results should be read as "all excellent, with modest
  differentiation" rather than any model failing.
- CAFormer used the smaller **s18** variant rather than b36 — the larger
  b36 model caused a CUDA out-of-memory error on Kaggle's free-tier GPU at
  batch_size=32. s18 is still the CAFormer architecture family named in the
  original proposal, just a lighter parameter size — and notably, it also
  achieved the *highest* accuracy of all 5, which is a meaningful finding
  given this project's low-resource-hardware focus, not just a footnote
  about the substitution.
- ROC AUC is ~0.9999-1.0000 across all models — essentially saturated at
  this dataset size, so accuracy/F1 do the real discriminative work in
  this comparison.

Full per-model confusion matrices, the accuracy comparison chart, and the
raw CSV are in `outputs/` after running `compare_all_metrics.py` /
`full_compare_with_progress.py`.

Trained checkpoints (`.pth` files) are **not stored in this repo** — they're
uploaded as Kaggle Models and referenced by path in `config.py`. Also kept
locally as a backup.

## Dataset

[Plant Disease Dataset](https://www.kaggle.com/datasets/alimransonet/plant-disease-dataset)
(alimransonet) — 51 classes across multiple crops (Apple, Banana, Bean,
Corn, Grape, Mango, Pepper, Potato, Rice, Strawberry, Tomato, etc.), split
into `train/`, `val/`, `test/`.

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings: dataset paths, hyperparameters, model selection, trained checkpoint paths |
| `dataset.py` | Loads the dataset via `ImageFolder`, builds train/val/test dataloaders |
| `model.py` | Builds any of the 5 architectures via `timm` |
| `train.py` | Trains whichever model is set as `CURRENT_MODEL` in `config.py`; `--dry-run` flag runs a quick 2-batch sanity check first |
| `evaluate.py` | Computes accuracy, precision, recall, F1, ROC AUC; used internally by `train.py` |
| `predict.py` | Single image, single model — whichever is set as `CURRENT_MODEL` |
| `predict_single_image_all_models.py` | Single image, tested against all 5 trained models, ranked by confidence |
| `compare_all_metrics.py` / `full_compare_with_progress.py` | Full validation-set evaluation across all 5 models — the aggregate results table, confusion matrices, and accuracy chart |

## Setup

```bash
pip install torch torchvision timm scikit-learn seaborn matplotlib tqdm
```

## Switching between models

In `config.py`:
```python
CURRENT_MODEL = "vit"   # or "deit", "pit", "swin", "caformer"
```
Everything else (`train.py`, `predict.py`, etc.) picks this up automatically
— no other code changes needed.

## Running on Kaggle

This project was trained on Kaggle Notebooks (free GPU) since local hardware
isn't sufficient. Typical session setup:

```python
!git clone https://github.com/sadiqfahim/crop-disease-classification.git
%cd crop-disease-classification
```

Then:
1. Attach the dataset via **Add Input**
2. Enable GPU in **Settings → Accelerator**
3. Run any script above, e.g.:
   ```python
   !python train.py --dry-run   # sanity check first
   !python train.py             # full training
   ```

**Note:** paths in `config.py` (`DATASET_ROOT`, `MODEL_PATHS`) are
Kaggle-specific (`/kaggle/input/...`). Running outside Kaggle requires
updating these paths.

## License

Apache 2.0
