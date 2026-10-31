"""Run locked Baseline 03: Baseline 02 with exponent 0.25 instead of 0.5.

The epoch loop, data pipeline, model, optimizer, scheduler, checkpointing, and
validation implementation are reused directly from the committed Baseline 02
runner. This module replaces only class-weight derivation, experiment identity,
and three-way result reporting.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

import run_baseline_02 as baseline


EXPERIMENT = "baseline_03_mild_weighted_ce"
OUTPUT_DIR = baseline.PROJECT_ROOT / "outputs" / EXPERIMENT
CHECKPOINT_DIR = baseline.PROJECT_ROOT / "checkpoints" / EXPERIMENT
BASELINE_01_DIR = baseline.PROJECT_ROOT / "outputs" / "baseline_01_unet_ce"
BASELINE_02_DIR = baseline.PROJECT_ROOT / "outputs" / "baseline_02_weighted_ce"
WEIGHTING_EXPONENT = 0.25

_base_configuration = baseline.configuration
_base_prepare_resume = baseline.prepare_resume
_base_clean_best_evaluation = baseline.clean_best_evaluation


def recompute_verified_weights(
    image_paths: tuple[Path, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count dev-train pixels and derive normalized frequency^-0.25 weights."""
    reviewed = json.loads((BASELINE_02_DIR / "config.json").read_text(encoding="utf-8"))
    if reviewed["weight_source"] != "dev_train_only" or reviewed["train_split"]["images"] != 2662:
        raise ValueError("Baseline 02 reference does not identify the locked dev-train source")

    counts, ignored_pixels, total_pixels = baseline.count_dev_train_pixels(image_paths)
    frequencies = counts.astype(np.float64) / float(counts.sum())
    reviewed_counts = np.asarray(reviewed["class_pixel_counts"], dtype=np.int64)
    reviewed_frequencies = np.asarray(reviewed["class_frequencies"], dtype=np.float64)
    if not np.array_equal(counts, reviewed_counts):
        raise RuntimeError("Recomputed dev-train counts differ from the reviewed Baseline 02 counts")
    if not np.array_equal(frequencies, reviewed_frequencies):
        raise RuntimeError("Recomputed dev-train frequencies differ from the reviewed calculation")
    if total_pixels <= 0 or ignored_pixels < 0 or ignored_pixels >= total_pixels:
        raise RuntimeError("Invalid dev-train pixel totals")
    if np.any(frequencies <= 0):
        raise ValueError("Every class must have non-zero dev-train frequency")

    raw_weights = np.power(frequencies, -WEIGHTING_EXPONENT, dtype=np.float64)
    weights = raw_weights / raw_weights.mean(dtype=np.float64)
    if not (
        weights.shape == (19,)
        and np.all(np.isfinite(weights))
        and np.all(weights > 0)
        and math.isclose(float(weights.mean()), 1.0, abs_tol=1e-12)
    ):
        raise RuntimeError("Locked Baseline 03 class weights failed sanity checks")
    expected_min, expected_max = 0.3869842578274517, 1.7080506639523498
    if not math.isclose(float(weights.min()), expected_min, abs_tol=1e-12):
        raise RuntimeError("Baseline 03 minimum weight differs from the reviewed diagnostic")
    if not math.isclose(float(weights.max()), expected_max, abs_tol=1e-12):
        raise RuntimeError("Baseline 03 maximum weight differs from the reviewed diagnostic")
    return counts, frequencies, weights


def configuration(split, train_transform, counts, frequencies, weights) -> dict:
    config = _base_configuration(split, train_transform, counts, frequencies, weights)
    config.update({
        "experiment_name": EXPERIMENT,
        "weighting_method": "normalized_inverse_frequency_power",
        "weighting_exponent": WEIGHTING_EXPONENT,
        "weight_source": "dev_train_only",
        "only_change_from_baseline_02": "Class-weight exponent 0.5 -> 0.25",
    })
    config.pop("only_change_from_baseline_01", None)
    return config


def prepare_fresh(config: dict) -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError("Baseline 03 output directory already exists; inspect before running")
    if CHECKPOINT_DIR.exists():
        raise FileExistsError("Baseline 03 checkpoint directory already exists; inspect before running")
    OUTPUT_DIR.mkdir(parents=True)
    CHECKPOINT_DIR.mkdir(parents=True)
    baseline.save_json(OUTPUT_DIR / "config.json", config)
    with (OUTPUT_DIR / "history.csv").open("x", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=baseline.HISTORY_FIELDS).writeheader()
        stream.flush()


def plot_histories(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    specifications = (
        ("loss_curve.png", "Weighted CE loss", (("Train", "train_loss"), ("Dev validation", "val_loss"))),
        ("miou_curve.png", "Dev-validation mIoU", (("mIoU", "val_miou"),)),
        ("dice_curve.png", "Dev-validation mean Dice", (("Mean Dice", "val_mean_dice"),)),
        ("lr_curve.png", "Learning rate", (("AdamW LR", "learning_rate"),)),
    )
    for filename, ylabel, series in specifications:
        figure, axis = plt.subplots(figsize=(7, 4))
        for label, key in series:
            axis.plot(epochs, [float(row[key]) for row in rows], label=label)
        axis.set(xlabel="Epoch", ylabel=ylabel, xlim=(1, 30))
        axis.grid(alpha=0.3)
        if len(series) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close(figure)

    histories = (
        (BASELINE_01_DIR / "history.csv", "Baseline 01 plain CE"),
        (BASELINE_02_DIR / "history.csv", "Baseline 02 exponent 0.5"),
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    for path, label in histories:
        prior = baseline.read_history(path)
        axis.plot(epochs, [float(row["val_miou"]) for row in prior], label=label)
    axis.plot(epochs, [float(row["val_miou"]) for row in rows], label="Baseline 03 exponent 0.25")
    axis.set(xlabel="Epoch", ylabel="Dev-validation mIoU", xlim=(1, 30))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "miou_comparison_3way.png", dpi=150)
    plt.close(figure)


def _load_diagnostics(directory: Path, names: dict[int, str]) -> tuple[dict, list[dict]]:
    metrics = json.loads((directory / "best_metrics.json").read_text(encoding="utf-8"))
    diagnostics = baseline.diagnostic_rows(np.asarray(metrics["confusion_matrix"], dtype=np.int64), names)
    return metrics, diagnostics


def write_comparison(best: dict, names: dict[int, str]) -> dict:
    metrics_01, diagnostics_01 = _load_diagnostics(BASELINE_01_DIR, names)
    metrics_02, diagnostics_02 = _load_diagnostics(BASELINE_02_DIR, names)
    diagnostics_03 = best["per_class"]
    fields = ["train_id", "class_name"]
    for metric in ("iou", "precision", "recall", "prediction_ratio"):
        fields.extend((f"b01_{metric}", f"b02_{metric}", f"b03_{metric}", f"b03_minus_b01_{metric}", f"b03_minus_b02_{metric}"))
    with (OUTPUT_DIR / "comparison_vs_baselines.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for class_id in range(19):
            first, second, third = diagnostics_01[class_id], diagnostics_02[class_id], diagnostics_03[class_id]
            row = {"train_id": class_id, "class_name": names[class_id]}
            for metric in ("iou", "precision", "recall", "prediction_ratio"):
                row.update({
                    f"b01_{metric}": first[metric], f"b02_{metric}": second[metric], f"b03_{metric}": third[metric],
                    f"b03_minus_b01_{metric}": third[metric] - first[metric],
                    f"b03_minus_b02_{metric}": third[metric] - second[metric],
                })
            writer.writerow(row)

    matrix_03 = np.asarray(best["confusion_matrix"], dtype=np.int64)
    traffic_fp_sources = sorted(
        (
            {"target_train_id": class_id, "target_class": names[class_id], "pixels": int(matrix_03[class_id, 6])}
            for class_id in range(19) if class_id != 6
        ),
        key=lambda row: (-row["pixels"], row["target_train_id"]),
    )
    comparison = {
        "baseline_01": {"best_epoch": metrics_01["best_epoch"], "miou": metrics_01["best_miou"], "mean_dice": metrics_01["mean_dice"], "pixel_accuracy": metrics_01["pixel_accuracy"]},
        "baseline_02": {"best_epoch": metrics_02["best_epoch"], "miou": metrics_02["best_miou"], "mean_dice": metrics_02["mean_dice"], "pixel_accuracy": metrics_02["pixel_accuracy"]},
        "baseline_03": {"best_epoch": best["best_epoch"], "miou": best["best_miou"], "mean_dice": best["mean_dice"], "pixel_accuracy": best["pixel_accuracy"]},
        "deltas": {
            "miou": best["best_miou"] - metrics_01["best_miou"],
            "mean_dice": best["mean_dice"] - metrics_01["mean_dice"],
            "pixel_accuracy": best["pixel_accuracy"] - metrics_01["pixel_accuracy"],
        },
        "deltas_vs_baseline_02": {
            "miou": best["best_miou"] - metrics_02["best_miou"],
            "mean_dice": best["mean_dice"] - metrics_02["mean_dice"],
            "pixel_accuracy": best["pixel_accuracy"] - metrics_02["pixel_accuracy"],
        },
        "traffic_light_fp_sources": traffic_fp_sources,
        "official_val_test_loaded": False,
    }
    baseline.save_json(OUTPUT_DIR / "comparison_summary.json", comparison)
    return comparison


def install_locked_overrides() -> None:
    baseline.EXPERIMENT = EXPERIMENT
    baseline.OUTPUT_DIR = OUTPUT_DIR
    baseline.CHECKPOINT_DIR = CHECKPOINT_DIR
    baseline.recompute_verified_weights = recompute_verified_weights
    baseline.configuration = configuration
    baseline.prepare_fresh = prepare_fresh
    baseline.prepare_resume = _base_prepare_resume
    baseline.clean_best_evaluation = _base_clean_best_evaluation
    baseline.write_comparison = write_comparison
    baseline.plot_histories = plot_histories


if __name__ == "__main__":
    install_locked_overrides()
    baseline.main()
