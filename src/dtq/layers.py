"""
Name the parts of DiT. Everything else (stats, policies, plots) groups by these.

Replaces: LAYER_TARGETS = {"attn": [...], "mlp": [...], "adaLN": [...]} in dit-quant.

Bug/gap in the old version: `"block" in name` also caught
  transformer_blocks.N.norm1.emb.timestep_embedder.linear_1/2
i.e. each block's own timestep/class embedder (diffusers' AdaLayerNormZero
owns one). Those were silently quantized and counted. Decide on purpose whether
they're in scope (most DiT PTQ papers keep embedders + final layer in FP).
"""

# TODO: fill these in after print(pipe.transformer). Suggested component names:
COMPONENTS = {
    "qkv":       [],   # attn1.to_q / to_k / to_v
    "attn_out":  [],   # attn1.to_out.0
    "fc1":       [],   # ff.net.0.proj
    "fc2":       [],   # ff.net.2      (post-GELU input -> very skewed activations, watch this one)
    "adaln":     [],   # norm1.linear  (outputs 6*hidden: shift/scale/gate for msa & mlp)
    "embedder":  [],   # norm1.emb.*, pos_embed, proj_out_*  -> usually kept FP
}


def component_of(name: str) -> str | None:
    """TODO: map a full module name -> component key above (or None)."""
    raise NotImplementedError


def block_index_of(name: str) -> int | None:
    """TODO: 'transformer_blocks.17.attn1.to_q' -> 17. Needed for depth-wise plots."""
    raise NotImplementedError


def quantizable_linears(model, include=("qkv", "attn_out", "fc1", "fc2", "adaln")):
    """
    TODO: yield (name, module) for nn.Linear modules whose component is in `include`.
    Sanity check: DiT-XL/2 has 28 blocks -> 28 * 7 = 196 linears for the default include.
    """
    raise NotImplementedError
