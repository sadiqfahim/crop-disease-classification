# crop-disease-classification
Crop disease detection using Vision Transformers

## Project Status

All 5 architectures from the proposal have been trained and evaluated on the
plant disease dataset (kaggle.com/datasets/alimransonet/plant-disease-dataset):

| Model    | timm identifier                  |
|----------|-----------------------------------|
| ViT      | vit_base_patch16_224              |
| DeiT     | deit_base_patch16_224             |
| PiT      | pit_b_224                         |
| Swin     | swin_base_patch4_window7_224      |
| CAFormer | caformer_s18                      |

Note: CAFormer used the smaller **s18** variant rather than b36 — the larger
b36 model caused a CUDA out-of-memory error on Kaggle's free-tier GPU at
batch_size=32. s18 is still the CAFormer architecture family named in the
proposal, just a lighter parameter size, consistent with this project's
low-resource-hardware focus.

Trained checkpoints (`.pth` files) are not stored in this repo — they're
uploaded as Kaggle Models and referenced by path in `config.py`.

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
| `compare_all_metrics.py` | Full validation-set evaluation across all 5 models — the aggregate results table (accuracy/precision/recall/F1/ROC AUC per model) |

## Running on Kaggle

This project was trained on Kaggle Notebooks (free GPU) since local hardware
isn't sufficient. Typical session setup:

```python
!git clone https://github.com/sadiqfahim/crop-disease-classification.git
%cd crop-disease-classification
```
Then attach the dataset via "Add Input", enable GPU in Settings → Accelerator,
and run any script above.

**Note:** paths in `config.py` (`DATASET_ROOT`, `MODEL_PATHS`) are Kaggle-
specific (`/kaggle/input/...`). Running outside Kaggle requires updating
these paths.
