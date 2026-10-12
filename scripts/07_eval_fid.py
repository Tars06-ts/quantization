"""
FID for the final tables. Expensive — run only on the methods that survived the LPIPS ablations.

TODO:
  - generate K images (10K / 50K) per method, balanced over the 1000 ImageNet classes, save as .npz / PNGs
  - batch generation (e.g. 64-256 per batch on 80GB) — measure throughput first, estimate total hours
  - compute FID/sFID/IS with the same evaluator as the papers (see metrics.fid_from_folder)
  - make it resumable (skip already-generated batches) — long jobs WILL get interrupted
"""

if __name__ == "__main__":
    raise NotImplementedError
