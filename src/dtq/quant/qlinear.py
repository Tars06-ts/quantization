"""
QuantLinear: drop-in replacement for nn.Linear that fake-quantizes weights and activations,
with activation scales that can depend on the current timestep.
"""
import torch.nn as nn


class QuantLinear(nn.Module):
    """
    TODO:
      __init__(linear, w_bits, a_bits, w_granularity, a_quantizer=..., tracker=None)
        - keep the original weight; precompute the fake-quantized weight ONCE (weights don't change)
        - hold activation quant params as a lookup: {bucket_id: (scale, zp)}
          bucket 0 only  -> static PTQ
          many buckets   -> time-aware PTQ
        - bucket_of_t: dict from timestep -> bucket id (from timestep.py)
      forward(x)
        - if a_bits is None/16: skip act quant
        - mode "calibrate": just observe x (record ranges per bucket), no quant
        - mode "quant": look up bucket from tracker.t, fake-quant x, F.linear(x_q, w_q, bias)
        - mode "dynamic": compute the scale from THIS x on the fly (upper bound baseline)
    Keep modes as a simple string attribute. Don't over-engineer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
