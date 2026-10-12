"""
Load DiT from diffusers and sample images. Do NOT reimplement DiT.

Replaces: load_pipeline() in dit-pretrained / dit-quant (duplicated 6+ times).

Things to (re)learn while writing this:
  - DiTPipeline.__call__: read its source once. Find
      * where CFG doubles the batch (latents concatenated, null class = 1000)
      * where `timestep` is passed into the transformer (you will hook this later)
      * what the transformer outputs (learned sigma -> it predicts 8 channels, 4 used)
  - pipe.transformer is a DiTTransformer2DModel. print(pipe.transformer) once and
    write down the module names in docs/model_map.md — layers.py depends on them.
"""


def load_pipeline(cfg):
    """
    TODO:
      - DiTPipeline.from_pretrained(cfg.model_id, torch_dtype=...)
      - swap scheduler according to cfg.scheduler
        (DDIMScheduler / DPMSolverMultistepScheduler / DDPMScheduler .from_config)
      - .to("cuda"), set eval mode, disable progress bar if you like
    Return the pipe.
    """
    raise NotImplementedError


def generate(pipe, cfg, class_labels=None, seed=None):
    """
    TODO: run the pipe with a fixed torch.Generator and return images.
    Decide: return PIL images, or output_type="np"/"pt" tensors?
    (Tensors are easier for LPIPS/PSNR; PIL is easier for saving grids. Maybe both.)
    """
    raise NotImplementedError


def fresh_transformer_copy(pipe):
    """
    TODO: return a deepcopy of pipe.transformer so you can quantize a copy
    and keep the FP16 reference around.
    Memory check: DiT-XL/2 is ~675M params -> ~1.35 GB in fp16. Two copies is fine on an 80GB card.
    (The old notebooks reloaded the whole pipe from disk instead — slow.)
    """
    raise NotImplementedError
