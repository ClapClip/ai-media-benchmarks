"""Reproducible pixel-containment verifier for AI face-swap outputs.

Compares a swapped frame against a no-op control frame from the same pipeline
and reports changed pixels (max RGB channel delta > threshold) inside vs.
outside a bounding box [x1, y1, x2, y2].

Methodology: https://www.clapclip.net/blog/face-swap-containment-check
"""
import argparse
import json
import numpy as np
from PIL import Image


def measure_containment(control_path: str, swapped_path: str, box: tuple[int, int, int, int], threshold: int = 8) -> dict:
    ctrl = np.asarray(Image.open(control_path).convert("RGB"), dtype=np.float64)
    swap = np.asarray(Image.open(swapped_path).convert("RGB"), dtype=np.float64)
    if ctrl.shape != swap.shape:
        raise ValueError(f"Shape mismatch: control {ctrl.shape} vs swapped {swap.shape}")

    x1, y1, x2, y2 = box
    mask = np.zeros(ctrl.shape[:2], dtype=bool)
    mask[y1:y2, x1:x2] = True

    abs_diff = np.abs(swap - ctrl)
    mean_pixel_diff = abs_diff.mean(axis=-1)
    max_channel_diff = abs_diff.max(axis=-1)
    changed = max_channel_diff > threshold

    total_changed = int(changed.sum())
    inside_changed = int((changed & mask).sum())
    outside_changed = int((changed & ~mask).sum())

    return {
        "resolution": f"{ctrl.shape[1]}x{ctrl.shape[0]}",
        "threshold_rgb": threshold,
        "mad_inside_255": round(float(mean_pixel_diff[mask].mean()), 4),
        "mad_outside_255": round(float(mean_pixel_diff[~mask].mean()), 6),
        "pct_changed_inside": round(100.0 * inside_changed / max(1, int(mask.sum())), 2),
        "pct_changed_outside": round(100.0 * outside_changed / max(1, int((~mask).sum())), 4),
        "share_of_changes_inside_pct": round(100.0 * inside_changed / total_changed, 2) if total_changed else 100.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify face-swap pixel containment outside a face bounding box.")
    parser.add_argument("--control", required=True, help="Path to no-op control image")
    parser.add_argument("--swapped", required=True, help="Path to face-swapped image")
    parser.add_argument("--box", nargs=4, type=int, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--threshold", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(measure_containment(args.control, args.swapped, tuple(args.box), args.threshold), indent=2))
