# Cityscapes Semantic Segmentation

A portfolio project for 19-class semantic segmentation on Cityscapes, beginning with a custom U-Net baseline and followed by controlled experiments, error analysis, final held-out evaluation, and inference benchmarking.

## Project status

Repository setup, official label integration, and the local dataset audit are complete. The audit findings are in [reports/dataset_audit.md](reports/dataset_audit.md). The raw dataset is kept locally and ignored by Git. No model training or evaluation results are claimed.

## Dataset

Use the official [Cityscapes dataset](https://www.cityscapes-dataset.com/) `leftImg8bit` images and `gtFine` annotations. The project uses [cityscapesscripts](https://github.com/mcordts/cityscapesScripts) for authoritative label definitions. Dataset files and archives stay outside Git and are not redistributed here.

Local layout:

```text
<dataset-root>/
├── leftImg8bit/{train,val,test}/<city>/...
└── gtFine/{train,val,test}/<city>/...
```

Official test labels are not public. The planned evaluation split is:

- Official `train` → development train and development validation.
- Official `val` → final held-out labeled evaluation.
- Official `test` → optional qualitative inspection or benchmark submission only.

## Development split

The tracked [city-level manifest](configs/splits/cityscapes_dev_split.json) divides official `train` into **2,662 development-training images** and **313 development-validation images** from **jena, krefeld, and ulm**. City separation reduces scene and location leakage compared with randomly splitting images. `src/cityscapes_segmentation/data/splits.py` validates the city lists, image membership, and semantic-mask paths against a dataset root supplied by the caller. Official `val` remains untouched for final held-out labeled evaluation; official `test` is not used for local semantic metrics.

## Task definition

The target is a 19-class semantic mask with train IDs `0..18` and ignore index `255`. Raw `*_labelIds.png` masks must be converted with `map_raw_ids_to_train_ids`; already converted `*_labelTrainIds.png` masks must not be converted a second time. Unknown raw IDs raise an error. The planned baseline input resolution is **512 × 256** (width × height), and the initial loss is `CrossEntropyLoss(ignore_index=255)`.

## Data pipeline

The Dataset takes image paths selected by the development split utility. It reads RGB images from `leftImg8bit` and semantic raw-ID masks from `gtFine/*_labelIds.png`, then maps raw IDs to train IDs `0..18` or ignore index `255`. Baseline transforms resize to **512×256** using bilinear interpolation for RGB and nearest-neighbor interpolation for masks. Images become normalized `float32` tensors with ImageNet mean/std and shape `[3,256,512]`; categorical targets remain unnormalized `int64` tensors with shape `[256,512]`. Training may apply one synchronized horizontal flip to image and mask; development validation is deterministic.

## Model baseline

The custom U-Net has four encoder stages (64, 128, 256, 512 channels), a 1024-channel bottleneck, and four decoder stages (512, 256, 128, 64 channels) with skip concatenation. It has no pretrained encoder. For input `[B,3,256,512]`, it returns **raw logits** `[B,19,256,512]` for a later cross-entropy loss; forward does not apply softmax, sigmoid, or argmax. Other input heights and widths must be at least 32 and divisible by 16.

## Planned class mapping

The [official Cityscapes label table](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py) is loaded at runtime; this table documents its 19 evaluated train IDs.

| Train ID | Class | Raw ID |
| ---: | --- | ---: |
| 0 | road | 7 |
| 1 | sidewalk | 8 |
| 2 | building | 11 |
| 3 | wall | 12 |
| 4 | fence | 13 |
| 5 | pole | 17 |
| 6 | traffic light | 19 |
| 7 | traffic sign | 20 |
| 8 | vegetation | 21 |
| 9 | terrain | 22 |
| 10 | sky | 23 |
| 11 | person | 24 |
| 12 | rider | 25 |
| 13 | car | 26 |
| 14 | truck | 27 |
| 15 | bus | 28 |
| 16 | train | 31 |
| 17 | motorcycle | 32 |
| 18 | bicycle | 33 |

Official ignored labels, including raw ID `-1` for license plate, become `255` in training targets. An input `255` ignore sentinel remains `255`.

## Metrics

Primary: mean intersection over union (mIoU). Secondary: per-class IoU, mean/per-class Dice, and pixel accuracy. These are computed from one accumulated confusion matrix across batches, with target ignore index `255` excluded from all counts. A class absent from both targets and predictions has undefined (NaN) IoU/Dice and is omitted from macro means. Inference speed will be measured later; no model results are claimed yet.

## Training engine

`train_one_epoch` and `validate_one_epoch` use sample-weighted mean batch loss. Validation derives mIoU and secondary metrics from one accumulated confusion matrix, and the entry point selects `best.pt` only when **development-validation mIoU** improves; it also writes `last.pt`. Checkpoints contain model and optimizer state, epoch, best mIoU, and config under the Git-ignored `checkpoints/` directory. `scripts/train_baseline.py` wires the city-separated development split to this engine. Python, NumPy, and PyTorch are seeded (default 42); fully deterministic CUDA algorithms are not forced. Only a bounded infrastructure smoke run has been performed so far; its metrics are not model-quality results. Official `val` remains untouched.

## Repository structure

```text
scripts/inspect_cityscapes.py             Future dataset audit entry point
src/cityscapes_segmentation/data/labels.py Official label mapping helpers
src/cityscapes_segmentation/data/splits.py City-level development split validation
src/cityscapes_segmentation/data/dataset.py RGB/semantic-mask sample loading
src/cityscapes_segmentation/data/transforms.py Paired baseline transforms
src/cityscapes_segmentation/analysis/metrics.py Confusion-matrix metrics
src/cityscapes_segmentation/models/unet.py Custom U-Net baseline
src/cityscapes_segmentation/engine/train.py One-epoch train/validation loops
src/cityscapes_segmentation/engine/checkpoint.py Checkpoint save/load
scripts/train_baseline.py                  Development-split training entry point
configs/splits/cityscapes_dev_split.json   Tracked development split manifest
reports/dataset_audit.md                  Local dataset audit findings
src/cityscapes_segmentation/models/        Planned model definitions
src/cityscapes_segmentation/engine/        Planned training and evaluation
src/cityscapes_segmentation/analysis/      Planned experiment analysis
src/cityscapes_segmentation/utils/         Shared utilities
tests/test_labels.py                       Synthetic mapping tests
tests/test_splits.py                       Split unit and optional local integration tests
tests/test_dataset.py                      Dataset unit and optional local integration tests
tests/test_transforms.py                   Synthetic transform tests
tests/test_metrics.py                      Synthetic metric tests
tests/test_unet.py                         U-Net shape and gradient checks
tests/test_engine.py                       Synthetic engine checks
tests/test_checkpoint.py                   Checkpoint roundtrip checks
```

## Environment setup

Use Python 3.12 from a standalone installation, not the Anaconda base interpreter. On the verified Windows setup, `py -3.12` selects Python 3.12.10 from the Microsoft Store installation. Check `py -3.12 -c "import sys; print(sys.executable)"` before creating the venv. The NVIDIA setup was verified with the [official PyTorch CUDA 13.0 pair](https://pytorch.org/get-started/previous-versions/): torch 2.13.0 and torchvision 0.28.0.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

Install the matching PyTorch pair before the requirements files so their version floors retain the CUDA wheels. The development requirements include the runtime requirements and `pytest`. For other hardware or drivers, choose the appropriate official command from the [PyTorch selector](https://pytorch.org/get-started/locally/).

## Dataset download guidance

Register or sign in at the [official Cityscapes website](https://www.cityscapes-dataset.com/) and download `leftImg8bit` and `gtFine` there. Extract them under a local dataset root outside the repository or an ignored local `data/` directory. Do not commit raw data or downloaded archives. The `cityscapesscripts` package provides the label reference and optional preparation tools.

The local dataset has been audited; see [reports/dataset_audit.md](reports/dataset_audit.md). `scripts/inspect_cityscapes.py` remains a preflight skeleton and does not regenerate that report.

## Next milestone

Review the model baseline before implementing the training engine.
