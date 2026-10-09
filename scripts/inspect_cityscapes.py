"""Entry point reserved for a later, real Cityscapes dataset audit.

The dataset has not been downloaded for this project. This script deliberately
does not emit audit statistics until the audit implementation is added.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SPLITS = ("train", "val", "test")
MODALITIES = ("leftImg8bit", "gtFine")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight check for a future Cityscapes audit")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root containing leftImg8bit/ and gtFine/")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_audit"))
    args = parser.parse_args()

    root = args.dataset_root.expanduser()
    if not root.is_dir():
        parser.error(f"Cityscapes dataset root is missing or is not a directory: {root}. Download leftImg8bit and gtFine from the official Cityscapes website first.")

    expected = [root / modality / split for modality in MODALITIES for split in SPLITS]
    missing = [path for path in expected if not path.is_dir()]
    if missing:
        parser.error("Cityscapes dataset layout is incomplete; missing directories:\n  " + "\n  ".join(str(path) for path in missing))

    parser.error(
        "Dataset audit is not implemented yet. Future checks will cover file and city counts, "
        "image/mask pairing and resolution, raw labels, train-ID distribution, ignore-pixel "
        "frequency, missing files, malformed masks, duplicate names, and split integrity. "
        f"No audit output was written to {args.output_dir}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
