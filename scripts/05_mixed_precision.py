"""
Time-aware mixed precision — first publication-oriented contribution.

Ports + fixes: mse-based-quant.ipynb (MSE-based layer selection).
  old: MSE at 1 dummy timestep, raw thresholds, int4 justified by int8 error, NaN outputs.
  new: sensitivity measured along real trajectories, per (layer, bucket), under a bit budget.

TODO:
  - sensitivity per layer (and per layer x bucket): rel output error when ONLY that layer is
    quantized to low bits (one-at-a-time), from OutputErrorProbe.
    (Cheaper alternatives to compare later: activation range variance across t, Hessian/Fisher proxies.)
  - sensitivity_to_bits at budgets avg 8 / 6 / 5 / 4.5 bits
  - compare against: uniform at same avg bits, your old single-timestep-MSE allocation
  - quality vs avg-bits curve (the main figure if this direction pans out)
"""

if __name__ == "__main__":
    raise NotImplementedError
