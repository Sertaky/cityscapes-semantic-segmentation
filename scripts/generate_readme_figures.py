"""Generate the tracked README figures from reviewed experiment metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"

EXPERIMENTS = ("B01", "B02", "B03", "B04")
METRICS = {
    "mIoU": (0.524634, 0.551079, 0.561506, 0.569926),
    "Mean Dice": (0.641598, 0.684990, 0.689985, 0.698703),
    "Pixel accuracy": (0.928686, 0.912909, 0.926736, 0.927832),
}

CLASS_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)
B04_MINUS_B03_IOU = (
    0.00016856384271068947,
    0.0015752312783412492,
    -0.0009557741433381395,
    -0.0043907695801737034,
    0.02994257736615541,
    0.0005108369879080477,
    -0.032297149865376074,
    -0.016041125130081935,
    0.0020478342103021907,
    0.002345092096165069,
    0.0033764010475290096,
    0.011187553979100984,
    0.0130632912772628,
    -0.0009020254544408468,
    -0.07550798456739821,
    0.06954055747176652,
    0.06342664221419092,
    0.062219363074840495,
    0.030667002921315578,
)


def save_experiment_metrics() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    x = np.arange(len(EXPERIMENTS))
    for axis, (metric, values) in zip(axes, METRICS.items(), strict=True):
        axis.plot(x, values, marker="o", linewidth=2)
        axis.scatter([3], [values[3]], s=70, zorder=3)
        for index, value in enumerate(values):
            axis.annotate(
                f"{value:.4f}",
                (index, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_title(metric)
        axis.set_xticks(x, EXPERIMENTS)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("Experiment")
    axes[0].set_ylabel("Score (0-1; full scale)")
    figure.suptitle("Development-validation metrics across controlled experiments")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "experiment_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_per_class_delta() -> None:
    order = np.argsort(B04_MINUS_B03_IOU)
    values = np.asarray(B04_MINUS_B03_IOU)[order]
    labels = np.asarray(CLASS_NAMES)[order]
    colors = ["tab:red" if value < 0 else "tab:blue" for value in values]
    figure, axis = plt.subplots(figsize=(9, 8))
    bars = axis.barh(labels, values, color=colors, alpha=0.85)
    axis.axvline(0, color="0.2", linewidth=1)
    axis.set_xlabel("IoU change (Baseline 04 - Baseline 03)")
    axis.set_ylabel("Cityscapes train class")
    axis.set_title("Per-class effect of increasing resolution to 640x320")
    axis.grid(axis="x", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:+.3f}" for value in values], padding=3, fontsize=8)
    limit = max(abs(values.min()), abs(values.max())) * 1.25
    axis.set_xlim(-limit, limit)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "b04_per_class_iou_delta.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_progression() -> None:
    figure, axis = plt.subplots(figsize=(13, 4.8))
    axis.set_xlim(-0.6, 4.6)
    axis.set_ylim(-1.2, 1.2)
    axis.axis("off")
    titles = ("B01", "B02", "B03", "B04", "B05 proposed")
    changes = (
        "Plain CE",
        "freq^-0.5 weights",
        "freq^-0.25 weights",
        "640x320 input",
        "Dilated bottleneck",
    )
    findings = (
        "Rare-class\nunderprediction",
        "Rare recovery;\novercorrection",
        "Better class\nbalance",
        "Modest mIoU gain;\nmixed class effects",
        "NOT RUN",
    )
    for index, (title, change, finding) in enumerate(zip(titles, changes, findings, strict=True)):
        face = "0.96" if index < 4 else "white"
        style = "solid" if index < 4 else "dashed"
        axis.text(
            index,
            0.2,
            f"{title}\n{change}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": face, "edgecolor": "0.35", "linestyle": style},
        )
        axis.text(index, -0.72, finding, ha="center", va="center", fontsize=9, color="0.25")
        if index < 4:
            axis.annotate(
                "",
                xy=(index + 0.68, 0.2),
                xytext=(index + 0.32, 0.2),
                arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 1.5},
            )
    axis.set_title("Controlled experiment progression", pad=18)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "experiment_progression.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_metrics()
    save_per_class_delta()
    save_progression()
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
