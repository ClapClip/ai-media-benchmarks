# AI Media Benchmarks & Verification Harness

Open-source scoring scripts and structured measurements behind the engineering studies published by **[ClapClip](https://www.clapclip.net/benchmarks)**:

1. **Face Swap Pixel Containment Check** — `scripts/check_face_swap_containment.py` ([Full report](https://www.clapclip.net/blog/face-swap-containment-check))
   - Verifies whether an AI face swap alters background, clothing, or hair pixels outside the target face region by diffing against a no-op pipeline control.
   - Published baseline on `skimage.data.astronaut` (512×512 and 256×256): **100% of changed pixels (>8/255 RGB delta) stayed inside the face region**; outside-region mean absolute difference was **0.0007 / 255** (512 px) and **0.0004 / 255** (256 px).
2. **2× Super-Resolution vs OpenCV `INTER_LANCZOS4`** — `scripts/score_upscale_psnr_ssim.py` ([Full report](https://www.clapclip.net/blog/2x-image-upscaling-ai-vs-lanczos))
   - Full-frame RGB PSNR (`data_range=255`) and Gaussian SSIM (`sigma=1.5`, population covariance) against `cv2.INTER_LANCZOS4` on Q50 JPEG inputs.

## Quick Start

```bash
pip install numpy pillow opencv-python scikit-image

# 1. Check pixel containment of any face-swap editor:
python scripts/check_face_swap_containment.py --control control.png --swapped swapped.png --box 140 60 360 310

# 2. Score 2x upscaling fidelity against Lanczos4:
python scripts/score_upscale_psnr_ssim.py --reference orig.png --degraded input_q50.jpg --ai-output ai_2x.png
```

## Related Tools

- **[ClapClip AI for Windows](https://www.clapclip.net/download)** — Local DirectML GPU video face swap, anchor-frame video enhancer & offline `whisper.cpp` transcription
- **[FaceSwap.free](https://faceswap.free)** — Browser photo & animated GIF face swap
- **[Enhance.place](https://enhance.place)** — Free 2× / 4× online image upscaler & [Print Size to Pixels calculator](https://enhance.place/print-size-to-pixels)

## License

MIT — see [LICENSE](./LICENSE).
