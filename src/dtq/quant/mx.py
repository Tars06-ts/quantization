"""
Simulated microscaling formats: MXFP8 and NVFP4. (Secondary axis — do this LAST.)

Hardware note: an A3 VM = H100 (Hopper). Hopper has FP8 (E4M3/E5M2) tensor cores but
NO native MX / NVFP4 — that's Blackwell. So on H100 these are simulation-only
(quality numbers, not speed numbers). Say that explicitly in the paper.

Reference: OCP Microscaling Formats (MX) v1.0 spec; NVIDIA NVFP4 docs.
"""


def round_to_fp(x, exp_bits: int, man_bits: int):
    """
    TODO: round fp32 values to the nearest representable value of a small float format
    (E4M3, E5M2, E2M1), with saturation at the format max. Handle subnormals.
    Tip: torch has torch.float8_e4m3fn / float8_e5m2 dtypes — casting to them is a
    ground truth you can test your own implementation against. E2M1 (FP4) you must do yourself
    (it only has 8 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6).
    """
    raise NotImplementedError


def mxfp8_quant(x, block_size: int = 32, elem: str = "e4m3"):
    """
    TODO: MXFP8 along the last dim:
      - blocks of 32 elements
      - shared scale = power of two (E8M0), derived from block absmax
        (spec: scale exponent = floor(log2(absmax)) - emax_of_elem_format)
      - elements -> FP8 (E4M3 or E5M2)
    Return dequantized tensor.
    """
    raise NotImplementedError


def nvfp4_quant(x, block_size: int = 16):
    """
    TODO: NVFP4 along the last dim:
      - per-tensor FP32 global scale
      - blocks of 16 elements, per-block scale stored in FP8 E4M3 (not power-of-two!)
      - elements -> FP4 E2M1
    Return dequantized tensor.
    Interesting question for this project: does block scaling already absorb the
    timestep-dependent range shifts, making time-aware scales unnecessary? (Hypothesis to test.)
    """
    raise NotImplementedError
