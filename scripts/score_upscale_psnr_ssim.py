"""Full-frame RGB PSNR and Gaussian SSIM evaluator against OpenCV Lanczos4.

Methodology: https://www.clapclip.net/blog/2x-image-upscaling-ai-vs-lanczos
"""
import argparse
import json
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def score_pair(reference_rgb: np.ndarray, candidate_rgb: np.ndarray) -> dict:
    psnr = peak_signal_noise_ratio(reference_rgb, candidate_rgb, data_range=255)
    ssim = structural_similarity(
        reference_rgb,
        candidate_rgb,
        channel_axis=-1,
        data_range=255,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    return {"psnr_db": round(float(psnr), 3), "ssim": round(float(ssim), 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score an upscaled image against a reference and Lanczos4 baseline.")
    parser.add_argument("--reference", required=True, help="Untouched full-resolution reference image")
    parser.add_argument("--degraded", required=True, help="Half-resolution degraded JPEG input")
    parser.add_argument("--ai-output", required=True, help="2x AI upscaled output image")
    args = parser.parse_args()

    ref = cv2.cvtColor(cv2.imread(args.reference, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    deg = cv2.cvtColor(cv2.imread(args.degraded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ai = cv2.cvtColor(cv2.imread(args.ai_output, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    lanczos = cv2.resize(deg, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    print(json.dumps({
        "lanczos_2x": score_pair(ref, lanczos),
        "ai_2x": score_pair(ref, ai),
    }, indent=2))
