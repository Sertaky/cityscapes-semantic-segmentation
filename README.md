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

## Planned metrics

The primary metric is mean intersection over union (mIoU) across the 19 evaluated classes, excluding ignored pixels. Per-class IoU and inference speed will be reported in later milestones. There are no measurements yet.

## Repository structure

```text
scripts/inspect_cityscapes.py             Future dataset audit entry point
src/cityscapes_segmentation/data/labels.py Official label mapping helpers
src/cityscapes_segmentation/data/splits.py City-level development split validation
configs/splits/cityscapes_dev_split.json   Tracked development split manifest
reports/dataset_audit.md                  Local dataset audit findings
src/cityscapes_segmentation/models/        Planned model definitions
src/cityscapes_segmentation/engine/        Planned training and evaluation
src/cityscapes_segmentation/analysis/      Planned experiment analysis
src/cityscapes_segmentation/utils/         Shared utilities
tests/test_labels.py                       Synthetic mapping tests
tests/test_splits.py                       Split unit and optional local integration tests
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

Review the development split and its tests before implementing the Dataset class and custom U-Net baseline.
