"""
Image-quality metrics.

Old notebooks only had a "BLACK" check (mean pixel < 5). Keep that as a cheap sanity
check (rename: is_degenerate), but add real metrics.

Two families — don't confuse them in the paper:
  * Fidelity to FP16 (same seed, same class): PSNR, SSIM, LPIPS.  Cheap, low variance, great for ablations.
  * Distribution quality: FID, sFID, IS, Precision/Recall vs ImageNet.  What papers report in the main table.
"""


def is_degenerate(images) -> list[bool]:
    """TODO: True for all-black / all-gray / NaN images. Run this on EVERY experiment's output."""
    raise NotImplementedError


def psnr(img_ref, img_test) -> float:
    raise NotImplementedError


def lpips_dist(img_ref, img_test, net: str = "alex") -> float:
    """TODO: `pip install lpips`; inputs must be in [-1, 1], NCHW. Load the net once, not per call."""
    raise NotImplementedError


def clip_score(images, class_labels) -> float:
    """
    Optional. TODO: torchmetrics CLIPScore with prompts like "a photo of a {imagenet class name}".
    Weak signal for class-conditional DiT — LPIPS-vs-FP is usually more useful.
    """
    raise NotImplementedError


def fid_from_folder(sample_dir: str, ref_stats: str) -> float:
    """
    TODO: FID against ImageNet-256 reference.
    Use the SAME protocol as the papers you compare against (PTQ4DiT / Q-DiT use the
    ADM / guided-diffusion evaluator with the VIRTUAL_imagenet256 reference batch;
    check each paper's sample count: 10K vs 50K, and steps/CFG). Otherwise numbers aren't comparable.
    torchmetrics / clean-fid are fine for quick internal comparisons.
    """
    raise NotImplementedError
