"""
Milestone 3: W8A8 static PTQ end-to-end + error-vs-timestep plot.

Ports + replaces: the torchao runs in dit-pretrained / dit-quant, now with your own fake quant.

TODO:
  - FP reference images (same seeds as 00)
  - variants (each a fresh_transformer_copy):
      W8A16 per-tensor, W8A16 per-channel  (weight-only, sanity: should be ~lossless)
      W8A8 static per-tensor act, per-channel weight
      W8A8 dynamic (upper bound)
      W4A8 / W8A4 (where does it break?)
  - calibrate static ones (calibrate.calibrate with num_buckets=1)
  - metrics vs FP: PSNR, LPIPS, is_degenerate
  - OutputErrorProbe -> rel MSE per (layer, t) -> plot_error_vs_timestep
  - save a results table (csv) + grid
"""

if __name__ == "__main__":
    raise NotImplementedError
