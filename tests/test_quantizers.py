"""
Tests for the quant primitives. Write these BEFORE trusting any experiment.
Run: pytest -q
"""


def test_fake_quant_is_idempotent():
    """TODO: fake_quant(fake_quant(x)) == fake_quant(x) with the same scale."""
    raise NotImplementedError


def test_fake_quant_error_bound():
    """TODO: for in-range values, |x - fq(x)| <= scale / 2."""
    raise NotImplementedError


def test_per_channel_beats_per_tensor():
    """TODO: weight with one huge row -> per-channel MSE << per-tensor MSE."""
    raise NotImplementedError


def test_16_bits_is_nearly_lossless():
    raise NotImplementedError


def test_e4m3_matches_torch_dtype():
    """TODO: mx.round_to_fp(x, 4, 3) == x.to(torch.float8_e4m3fn).float() (within saturation rules)."""
    raise NotImplementedError


def test_nvfp4_grid_values():
    """TODO: every output / its block scale lies on {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}."""
    raise NotImplementedError
