"""
Fake (simulated) integer quantization. Pure PyTorch, no kernels.

Why fake quant instead of torchao (what the notebooks used)?
  - torchao picks the scheme + kernel for you; you can't plug in per-timestep scales.
  - on the T4 int8 weight-only was even SLOWER than fp16 (53s vs 48s) — kernel issue,
    not a quality signal.
  - For the paper, quality comes from fake quant; speed is a separate, later measurement.

Write these yourself — this is the part you should understand cold.
Test every function in tests/test_quantizers.py before using it anywhere.
"""


def compute_scale_zp(x_min, x_max, n_bits: int, symmetric: bool = True):
    """
    TODO: return (scale, zero_point) for the integer grid.
      symmetric: q in [-(2^(b-1)-1), 2^(b-1)-1], zp = 0, scale = absmax / qmax
      asymmetric: q in [0, 2^b - 1], scale = (max-min)/(2^b-1), zp = round(-min/scale)
    Guard against scale == 0 (dead channels).
    Works elementwise so it supports per-tensor AND per-channel min/max.
    """
    raise NotImplementedError


def fake_quant(x, scale, zero_point, n_bits: int, symmetric: bool = True):
    """TODO: quantize -> clamp -> dequantize. Return a tensor in x's dtype. (Do the math in fp32.)"""
    raise NotImplementedError


def quantize_weight(w, n_bits: int, granularity: str = "per_channel", group_size: int = 128):
    """
    TODO: fake-quantize a Linear weight [out_features, in_features].
      per_tensor : one scale
      per_channel: one scale per OUTPUT row (reduce over dim=1)
      per_group  : reshape in_features into groups of `group_size`, scale per (row, group)
    Return the fake-quantized weight (same shape).
    """
    raise NotImplementedError


def minmax_range(x, dims=None):
    """TODO: plain min/max range. Baseline observer."""
    raise NotImplementedError


def percentile_range(x, pct: float = 99.99, dims=None):
    """TODO: clip range from percentiles — robust to outliers."""
    raise NotImplementedError


def mse_search_range(x, n_bits: int, num_candidates: int = 100, dims=None):
    """
    TODO: grid-search a clipping ratio alpha in (0, 1] of absmax, keep the one with
    lowest ||x - fake_quant(x)||^2. This is the standard "MSE calibration" — note it is a
    different thing from your old MSE-based *layer selection*. Don't mix the two up in the paper.
    """
    raise NotImplementedError
