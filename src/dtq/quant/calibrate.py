"""
Calibration: run sampling on calibration prompts, collect activation ranges, set scales.

The methods in your ablation table, all as one function with a `num_buckets` knob:
  Static PTQ            -> num_buckets = 1
  Time-bucket PTQ       -> num_buckets = 2 / 4 / 8 (uniform or clustered)
  Per-timestep PTQ      -> num_buckets = num_steps
  Dynamic PTQ           -> no calibration, QuantLinear mode="dynamic"
"""


def calibrate(pipe, qmodel, tracker, cfg, bucket_of_t: dict, range_fn="minmax"):
    """
    TODO:
      1. set every QuantLinear to mode="calibrate", give them bucket_of_t
      2. sample cfg.calib_num_samples images with cfg.calib_seed and random class labels
         (calibration data for class-conditional DiT = sampling trajectories; no dataset needed)
      3. per layer, per bucket: aggregate ranges (minmax / percentile / mse_search)
         Careful: averaging per-image maxima != max over everything. Pick one and document it.
      4. compute (scale, zp) per bucket, store in each QuantLinear, set mode="quant"
    Gotcha: if your quant model is used during calibration with quantized activations
    in earlier layers, later layers see quantized inputs (sequential calibration).
    Simpler first version: calibrate everything on FP activations (weights-only quantized or FP).
    """
    raise NotImplementedError
