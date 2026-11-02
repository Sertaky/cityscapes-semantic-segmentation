"""Generate tracked README figures from verified, frozen summary metrics."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"

EXPERIMENTS = ("B01", "B02", "B03", "B04", "B05")
DEVELOPMENT_METRICS = {
    "mIoU": (0.524634, 0.551079, 0.561506, 0.569926, 0.582557),
    "Mean Dice": (0.641598, 0.684990, 0.689985, 0.698703, 0.711907),
    "Pixel accuracy": (0.928686, 0.912909, 0.926736, 0.927832, 0.929086),
}
FINAL_COMPARISON = {
    "Development validation\n313 images": (0.582557, 0.711907, 0.929086),
    "Official validation\n500 images": (0.529051, 0.655063, 0.919040),
}
CLASS_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)
OFFICIAL_VAL_IOU = (
    0.963774, 0.729227, 0.855885, 0.303813, 0.350273, 0.437005,
    0.359018, 0.486951, 0.879684, 0.492606, 0.910267, 0.549601,
    0.224792, 0.874574, 0.310675, 0.415808, 0.170618, 0.164473,
    0.572921,
)
EXAMPLE_NAMES = (
    "000_frankfurt_000000_000294_leftImg8bit.png",
    "213_frankfurt_000001_062016_leftImg8bit.png",
    "427_munster_000101_000019_leftImg8bit.png",
)


def save_experiment_metrics() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4), sharey=True)
    x = np.arange(len(EXPERIMENTS))
    for axis, (metric, values) in zip(axes, DEVELOPMENT_METRICS.items(), strict=True):
        axis.plot(x, values, marker="o", linewidth=2, color="tab:blue")
        axis.scatter([4], [values[4]], s=75, color="tab:orange", zorder=3, label="Frozen winner")
        for index, value in enumerate(values):
            axis.annotate(f"{value:.4f}", (index, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        axis.set_title(metric)
        axis.set_xticks(x, EXPERIMENTS)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("Controlled experiment")
    axes[0].set_ylabel("Score (0-1; full scale)")
    axes[-1].legend(loc="lower right", fontsize=8)
    figure.suptitle("Development-validation progression")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "experiment_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_dev_vs_official() -> None:
    metric_names = ("mIoU", "Mean Dice", "Pixel accuracy")
    x = np.arange(len(metric_names))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for offset, (label, values) in enumerate(FINAL_COMPARISON.items()):
        positions = x + (offset - 0.5) * width
        bars = axis.bar(positions, values, width, label=label)
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=9)
    axis.set_xticks(x, metric_names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score (0-1; full scale)")
    axis.set_title("Frozen B05: development and held-out evaluation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "dev_vs_official.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_final_per_class_iou() -> None:
    order = np.argsort(OFFICIAL_VAL_IOU)
    values = np.asarray(OFFICIAL_VAL_IOU)[order]
    labels = np.asarray(CLASS_NAMES)[order]
    colors = ["tab:orange" if value < 0.4 else "tab:blue" for value in values]
    figure, axis = plt.subplots(figsize=(9, 8))
    bars = axis.barh(labels, values, color=colors, alpha=0.88)
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Intersection over Union")
    axis.set_ylabel("Cityscapes train class")
    axis.set_title("Frozen B05 on official Cityscapes validation")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "final_per_class_iou.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_progression() -> None:
    figure, axis = plt.subplots(figsize=(16, 5))
    axis.set_xlim(-0.6, 6.6)
    axis.set_ylim(-1.25, 1.25)
    axis.axis("off")
    titles = ("B01", "B02", "B03", "B04", "B05", "Frozen", "Official val")
    changes = ("Plain CE", "Strong weights", "Mild weights", "640x320", "Dilation 2", "No tuning", "500 images")
    findings = (
        "Rare-class\nunderprediction",
        "Recovery;\novercorrection",
        "Better\nbalance",
        "Mixed resolution\ngains",
        "Best dev mIoU\n0.5826",
        "Checkpoint\nlocked",
        "Held-out mIoU\n0.5291",
    )
    for index, (title, change, finding) in enumerate(zip(titles, changes, findings, strict=True)):
        face = "#eaf2fb" if index < 5 else "#fff4e6"
        axis.text(index, 0.25, f"{title}\n{change}", ha="center", va="center", fontsize=9.5, fontweight="bold", bbox={"boxstyle": "round,pad=0.55", "facecolor": face, "edgecolor": "0.35"})
        axis.text(index, -0.72, finding, ha="center", va="center", fontsize=8.5, color="0.25")
        if index < len(titles) - 1:
            axis.annotate("", xy=(index + 0.67, 0.25), xytext=(index + 0.33, 0.25), arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 1.4})
    axis.set_title("Evidence-driven project progression", pad=18)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "experiment_progression.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def copy_deterministic_examples() -> None:
    source_dir = PROJECT_ROOT / "outputs" / "final_official_val" / "examples"
    destination = OUTPUT_DIR / "final_examples"
    destination.mkdir(parents=True, exist_ok=True)
    for name in EXAMPLE_NAMES:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)
        else:
            print(f"Skipped missing optional example: {source.relative_to(PROJECT_ROOT)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_metrics()
    save_dev_vs_official()
    save_final_per_class_iou()
    save_progression()
    copy_deterministic_examples()
    for path in sorted(OUTPUT_DIR.rglob("*.png")):
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
