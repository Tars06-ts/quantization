"""
Decide WHICH layers get WHICH precision, then swap nn.Linear -> QuantLinear.

Replaces: quantize_selective() (dit-quant) and get_quant_config(mse) (mse-based-quant).

Issues in the old MSE policy worth remembering:
  - MSE came from ONE dummy timestep with random latents -> doesn't reflect real sampling.
  - Thresholds (0.001 / 0.01) were absolute raw MSE, so they're layer-scale dependent.
  - Low int8 MSE was used to justify int4 — those are different error regimes; measure int4 directly.
  - The resulting images hit NaN ("invalid value encountered in cast") -> check outputs for NaN/inf.
"""


def apply_uniform(model, cfg, tracker, include=None):
    """TODO: replace every quantizable linear (layers.quantizable_linears) with QuantLinear(cfg.w_bits, cfg.a_bits)."""
    raise NotImplementedError


def apply_component(model, cfg, tracker, components: list[str]):
    """TODO: same, but only for the given components. This is your old component-wise sweep."""
    raise NotImplementedError


def apply_mixed_precision(model, tracker, bits_map: dict):
    """
    TODO: bits_map = {layer_name: (w_bits, a_bits)}. The general case.
    Later (time-aware mixed precision): bits could depend on bucket too —
    {(layer_name, bucket): bits}. Think about whether that's even deployable
    (switching weight precision per timestep means storing multiple weight copies!).
    Activation-bit switching per timestep is cheaper. Interesting paper discussion point.
    """
    raise NotImplementedError


def sensitivity_to_bits(sensitivity: dict, budget_avg_bits: float, choices=(4, 6, 8)) -> dict:
    """
    TODO: turn per-layer (or per-layer-per-bucket) sensitivity scores into a bits_map under an
    average-bit budget. Start greedy: everyone at min bits, upgrade the most sensitive until
    budget is spent. (ILP / knapsack later if needed.)
    """
    raise NotImplementedError
