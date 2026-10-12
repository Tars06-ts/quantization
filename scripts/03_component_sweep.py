"""
Which components are most sensitive? (and WHEN?)

Ports: dit-quant.ipynb (the adaLN / attn / mlp sweep), but:
  - one script instead of 4 copy-pasted cells (loop over bit configs instead)
  - finer components: qkv / attn_out / fc1 / fc2 / adaln  (see layers.py)
  - measures LPIPS vs FP + per-timestep error, not just "BLACK/OK" and latency
  - optional: per-block sweep (quantize only block i) -> sensitivity vs depth

TODO: for each component set in a list, apply_component(...), calibrate, evaluate, append row.
"""

if __name__ == "__main__":
    raise NotImplementedError
