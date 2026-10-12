"""
MXFP8 / NVFP4 simulation (secondary axis).

TODO:
  - replace the int fake quant in QuantLinear with mx.mxfp8_quant / mx.nvfp4_quant (weights and/or acts)
  - compare: INT8 static vs INT8 time-bucketed vs MXFP8 (no time-awareness) vs NVFP4 (+/- time-aware global scale)
  - key question: does block scaling remove the need for timestep-aware scales?
  - simulation only on H100 — no speed claims for MX/NVFP4.
"""

if __name__ == "__main__":
    raise NotImplementedError
