# Cityscapes Semantic Segmentation

A reproducible 19-class semantic-segmentation project built around Cityscapes and a custom U-Net trained from scratch. The repository covers the path from raw-data validation and official label integration through deterministic splitting, paired transforms, confusion-matrix metrics, checkpointed training, controlled experiments, and failure analysis.

The current best **development** result is Baseline 02 with **0.551079 mIoU** on a 313-image, city-separated development-validation split. The official Cityscapes validation set remains untouched, so this is not a final or official benchmark result.

## Key results

| Experiment | Loss | Best epoch | Dev-val mIoU | Mean Dice | Pixel accuracy |
|---|---|---:|---:|---:|---:|
| Baseline 01 | Cross-entropy | 26 | 0.524634 | 0.641598 | 0.928686 |
| Baseline 02 | Sqrt-inverse-frequency weighted cross-entropy | 28 | **0.551079** | **0.684990** | 0.912909 |
| Change | One controlled loss-weighting change | — | **+0.026445** | **+0.043392** | -0.015777 |

Weighted cross-entropy substantially improved rare-class recognition, including rider, motorcycle, train, truck, and bus. It also exposed a clear tradeoff: some common classes regressed, and traffic-light predictions became overly frequent. The project treats that tradeoff as evidence to investigate, not as a reason to report a single aggregate metric without context.

## Objective

The engineering objective is to build a transparent segmentation baseline whose data membership, label semantics, transforms, metrics, training state, and experiment changes can all be audited. The task uses the 19 standard Cityscapes train IDs (`0..18`), with `255` as the ignore index.

This repository does not claim production readiness, official test performance, or final held-out performance.

## Dataset and evaluation discipline

The project uses Cityscapes `leftImg8bit` RGB images and `gtFine` semantic annotations. Raw label IDs are interpreted through [`cityscapesscripts`](https://github.com/mcordts/cityscapesScripts), the authoritative label source used by the project.

| Official split | Images | Project use |
|---|---:|---|
| `train` | 2,975 | Divided by city into development train and development validation |
| `val` | 500 | Untouched; reserved for final held-out labeled evaluation |
| `test` | 1,525 | Not used for local semantic metrics |

The deterministic development split is defined in [`configs/splits/cityscapes_dev_split.json`](configs/splits/cityscapes_dev_split.json):

- **Development train:** 2,662 images from 15 cities.
- **Development validation:** 313 images from **jena, krefeld, and ulm**.
- **Official validation:** all 500 images remain untouched.

City-level separation reduces scene and location leakage compared with a random image split. Every metric in the experiment sections below is a **development-validation metric**, not a test result.

### Audit findings

The read-only audit verified:

- all official train/validation image-mask pairs;
- 2048x1024 image and annotation dimensions throughout;
- no missing or malformed pairs;
- no duplicate image content across official splits;
- no city overlap across official train, validation, and test splits;
- raw-ID behavior against `cityscapesscripts`;
- train IDs `0..18` and ignore index `255`;
- approximately 11.47% ignored pixels in official train and 12.55% in official validation;
- substantial class imbalance, especially among human and vehicle subclasses.

The locally supplied test `labelIds` contain only ignored/non-evaluation raw classes, producing no usable train-ID pixels. They are not used for local semantic evaluation. Detailed counts and class distributions are available in [`reports/dataset_audit.md`](reports/dataset_audit.md), [`reports/class_pixel_frequencies.csv`](reports/class_pixel_frequencies.csv), and [`reports/city_class_distribution.csv`](reports/city_class_distribution.csv).

## Data pipeline

`CityscapesSegmentationDataset` loads membership resolved by the split utility rather than duplicating split rules inside the Dataset.

| Component | Contract |
|---|---|
| RGB source | `leftImg8bit` |
| Semantic target | `gtFine/*_gtFine_labelIds.png` |
| Label conversion | Official raw ID to train ID `0..18`; ignored labels to `255` |
| Image tensor | `[3, 256, 512]`, `torch.float32` |
| Target tensor | `[256, 512]`, `torch.int64` |
| Image resize | Bilinear interpolation |
| Mask resize | Nearest-neighbor interpolation |
| Normalization | ImageNet mean and standard deviation |
| Training geometry | Resize plus synchronized horizontal flip, `p=0.5` |
| Validation geometry | Deterministic resize, no random augmentation |

Nearest-neighbor interpolation preserves categorical mask IDs; continuous interpolation could create invalid class values. Unknown raw IDs raise explicitly instead of being silently remapped.

## Model architecture

The baseline is a custom U-Net with no pretrained encoder:

```text
Input       [B, 3, 256, 512]
Encoder     64 -> 128 -> 256 -> 512
Bottleneck  1024
Decoder     512 -> 256 -> 128 -> 64
Classifier  64 -> 19
Output      [B, 19, 256, 512]
```

It uses four encoder and four decoder stages, `Conv2d -> BatchNorm2d -> ReLU` blocks, max-pooling, transposed-convolution upsampling, and channel-wise skip concatenation. The model returns raw logits for cross-entropy loss; it does not apply softmax or argmax in `forward()`.

- Base channels: 64
- Trainable parameters: **31,038,803**
- Training precision: FP32
- Pretraining: none
- Verified local feasibility: batch size 2 at 512x256 on an RTX 3050 Laptop GPU with 4 GB VRAM

## Evaluation methodology

The primary metric is **mean intersection over union (mIoU)**. Secondary metrics are per-class IoU, mean/per-class Dice, and pixel accuracy.

Metrics are derived from one accumulated 19x19 confusion matrix across the full validation pass:

- rows: ground-truth classes;
- columns: predicted classes;
- target pixels equal to `255` are excluded;
- absent-class IoU and Dice are undefined and omitted from macro means.

The implementation accumulates pixels across batches before computing metrics; it does not average batch-level IoUs.

## Training infrastructure

The implemented engine provides:

- `train_one_epoch` and `validate_one_epoch`;
- sample-weighted epoch loss aggregation;
- accumulated validation metrics;
- finite-loss checks;
- `best.pt` selection by development-validation mIoU;
- `last.pt` checkpoints;
- model and optimizer restore/resume support;
- deterministic split validation and seeded experiment setup;
- incremental CSV epoch history and reproducible locked experiment runners.

Checkpoint loading includes compatibility coverage for the current format. Python, NumPy, PyTorch, and CUDA RNGs are seeded with 42 for the reported experiments; fully deterministic CUDA algorithms were not forced.

## Controlled experiment protocol

Both completed baselines used the same locked protocol except for the loss weights:

| Setting | Value |
|---|---|
| Model | Custom U-Net, `base_channels=64` |
| Resolution | 512x256 |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Batch size | 2 |
| Epochs | 30, no early stopping |
| Scheduler | CosineAnnealingLR, `T_max=30`, `eta_min=1e-5` |
| Seed | 42 |
| Precision | FP32 |
| Training augmentation | Synchronized horizontal flip, `p=0.5` |
| Model selection | Highest development-validation mIoU |

### Baseline 01 — plain cross-entropy

Baseline 01 used `CrossEntropyLoss(ignore_index=255)`. Its best checkpoint occurred at epoch 26:

- mIoU: **0.524634**
- Mean Dice: **0.641598**
- Pixel accuracy: **0.928686**

Road, sky, car, vegetation, and building were the strongest classes. Rider, motorcycle, train, truck, and bus were among the weakest. In particular:

- rider: IoU `0.0246`, recall `0.0256`, predicted/GT ratio `0.068x`;
- motorcycle: IoU `0.1793`, predicted/GT ratio `0.469x`.

#### Baseline 01 failure analysis

The failure analysis showed systematic rare-class under-prediction:

- 73.34% of rider pixels were predicted as person;
- motorcycle was most often confused with person (27.20%) and car (25.40%);
- train was frequently confused with building (32.57%), bus (14.64%), and truck (8.10%).

The evidence was consistent with both class imbalance and spatial/detail difficulty at 512x256. Pixel-level diagnostics alone cannot establish object-size causality. This analysis motivated weighted cross-entropy as the next **single** controlled change.

### Baseline 02 — sqrt-inverse-frequency weighted cross-entropy

Baseline 02 changed only the class weighting applied to cross-entropy. Weights were calculated from the 2,662-image development-training split only:

```text
weight[c] proportional to frequency[c]^-0.5
mean(weight) = 1
```

Development validation, official validation, and official test pixels were excluded from the weight calculation. The exact weights are regenerated by [`scripts/compute_class_weights.py`](scripts/compute_class_weights.py); the remaining model and training configuration was identical to Baseline 01.

The best checkpoint occurred at epoch 28:

- mIoU: **0.551079** (`+0.026445`)
- Mean Dice: **0.684990** (`+0.043392`)
- Pixel accuracy: **0.912909** (`-0.015777`)

Weighted and unweighted loss magnitudes are not directly comparable as model-quality measures.

#### Rare-class improvements

| Class | Baseline 01 IoU | Baseline 02 IoU | Baseline 01 recall | Baseline 02 recall |
|---|---:|---:|---:|---:|
| rider | 0.0246 | **0.3619** | 0.0256 | **0.6032** |
| motorcycle | 0.1793 | **0.3630** | 0.2233 | **0.5866** |
| train | 0.2260 | **0.3448** | 0.3113 | **0.5359** |
| truck | 0.2539 | **0.3427** | 0.3945 | **0.6135** |
| bus | 0.2757 | **0.4193** | 0.3836 | **0.6023** |

Precision also improved for all five classes, so these gains were not solely indiscriminate rare-class overprediction.

#### Tradeoffs and failure analysis

Traffic-light recall improved, but precision collapsed as predictions expanded well beyond the ground-truth volume:

| Traffic-light metric | Baseline 01 | Baseline 02 |
|---|---:|---:|
| Precision | 0.7209 | 0.3716 |
| Recall | 0.4107 | 0.6539 |
| IoU | 0.3544 | 0.3105 |
| Predicted/GT ratio | 0.570x | 1.760x |

The principal traffic-light false-positive sources were building (53.37%), vegetation (28.30%), and pole (10.68%). False positives appeared in 295 of 313 development-validation images, indicating a broad systematic effect rather than a handful of outlier scenes.

Common-class IoU regressions were:

| Class | IoU change |
|---|---:|
| sidewalk | -0.0341 |
| building | -0.0297 |
| sky | -0.0284 |
| vegetation | -0.0207 |
| car | -0.0155 |
| road | -0.0083 |

Confusion changes included strong increases in building to traffic-light predictions, person to rider predictions, and car to truck/motorcycle predictions. These patterns are **consistent with weighting-driven redistribution**, but the comparison does not by itself prove causality.

## Current conclusion and next hypothesis

**Current best development model:** Baseline 02, selected by the highest city-separated development-validation mIoU. It is not designated as the final model, and the official validation set has not yet been used.

The next evidence-based candidate is a milder weighting exponent:

```text
weight[c] proportional to frequency[c]^-0.25
mean(weight) = 1
```

The diagnostic candidate has a minimum weight of `0.386984`, maximum weight of `1.708051`, and max/min ratio of `4.413747x`, compared with approximately `19.48x` for Baseline 02. The hypothesis is that milder weighting may retain rare-class gains while reducing overprediction and common-class regression.

**Baseline 03 has not been run.** No candidate metrics are claimed.

## Project structure

```text
configs/
  splits/                         Deterministic city-level split manifest
reports/                          Tracked dataset-audit summaries
scripts/
  inspect_cityscapes.py           Dataset preflight utility
  compute_class_weights.py        Dev-train-only class-weight calculation
  run_baseline_01.py              Locked Baseline 01 runner
  analyze_baseline_01.py          Baseline 01 failure analysis
  run_baseline_02.py              Locked Baseline 02 runner
  analyze_baseline_02.py          Baseline 02 regression analysis
src/cityscapes_segmentation/
  data/                            Labels, splits, Dataset, paired transforms
  models/                          Custom U-Net
  engine/                          Train/validate and checkpoint handling
  analysis/                        Confusion-matrix segmentation metrics
tests/                             Unit and optional local integration tests
```

Raw data, experiment outputs, and checkpoints are intentionally excluded from Git.

## Environment setup

The verified environment uses Python 3.12.10 from a standalone installation. On Windows, confirm that the selected interpreter is not the Anaconda base interpreter before creating the virtual environment.

```powershell
py -3.12 -c "import sys; print(sys.executable)"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

Install the matching official PyTorch build before the general requirements so the version floors retain that CUDA wheel. Select a different official PyTorch build when required by the host GPU and driver.

Place `leftImg8bit/` and `gtFine/` under the project dataset root expected by the scripts. The tracked split manifest determines development membership; image-level split lists are not hard-coded.

Experiment runners are intentionally locked around their reported configurations. Existing output/checkpoint directories are guarded against accidental overwrite, and `--resume` should be used only after checking checkpoint and history integrity.

## Testing

**76 tests currently pass.** Coverage includes:

- official label mapping and unknown-ID behavior;
- split membership and leakage checks;
- synchronized transforms and Dataset contracts;
- confusion-matrix metrics;
- U-Net shapes and gradient flow;
- training/validation engine behavior;
- checkpoint save/load compatibility;
- dev-train class-weight calculation.

Some integration checks skip cleanly when raw Cityscapes data is unavailable.

## Hardware and runtime

The reported experiments ran locally on an **NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM** using PyTorch `2.13.0+cu130`, torchvision `0.28.0+cu130`, and a CUDA 13.0 build.

| Experiment | 30-epoch time | Peak CUDA reserved |
|---|---:|---:|
| Baseline 01 | approximately 6h 54m | 2,062 MiB |
| Baseline 02 | approximately 6h 20m | 2,062 MiB |

These are local workflow measurements, not hardware benchmarks.

## Dataset source and license

Cityscapes must be obtained separately from the [official Cityscapes website](https://www.cityscapes-dataset.com/). This repository does not redistribute images, annotations, or dataset archives. Local `leftImg8bit/`, `gtFine/`, `gtCoarse/`, and downloaded archives are Git-ignored.

Use of Cityscapes data remains subject to the dataset's official license and terms. The optional [`cityscapesscripts`](https://github.com/mcordts/cityscapesScripts) dependency supplies the authoritative label definitions and related official tooling.
