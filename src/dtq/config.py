"""
Experiment config. ONE place for every knob, so every run is reproducible.

Replaces: the SEED / CLASS_INDICES / NUM_STEPS / GUIDANCE_SCALE constants that
were copy-pasted at the top of every notebook cell.

Lesson from the notebooks: dit-pretrained used DPM-Solver++ @ 25 steps, while
mse-based-quant used the default scheduler @ 50 steps. Those results are NOT
comparable. Pin the scheduler + steps here and never change them silently.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # --- model / sampling ---
    model_id: str = "facebook/DiT-XL-2-256"
    dtype: str = "float16"            # TODO: try "bfloat16" too on the A3 (H100) and compare
    scheduler: str = "ddim"           # TODO: decide: "ddim" | "dpm++" | "ddpm". Check what PTQ4DiT / Q-DiT use
    num_steps: int = 50
    guidance_scale: float = 4.0
    seed: int = 42
    class_labels: list[int] = field(default_factory=lambda: [207, 281, 817, 963])

    # --- calibration ---
    calib_num_samples: int = 32       # how many images' trajectories to calibrate on
    calib_seed: int = 1234            # MUST differ from eval seed (no calibrating on test data)

    # --- quantization ---
    w_bits: int = 8
    a_bits: int = 8
    w_granularity: str = "per_channel"   # "per_tensor" | "per_channel" | "per_group"
    a_granularity: str = "per_tensor"
    num_time_buckets: int = 1            # 1 == static (all-timestep) calibration

    # --- io ---
    out_dir: str = "results/default"


def load_config(path: str | None = None, **overrides) -> Config:
    """
    TODO:
      - start from Config() defaults
      - if `path` is given, load a yaml/json and override fields
      - apply **overrides (e.g. from argparse)
    Keep it dumb. Don't build a config framework.
    """
    raise NotImplementedError


def save_config(cfg: Config, out_dir: str) -> None:
    """TODO: dump asdict(cfg) to <out_dir>/config.json so every result folder says how it was made."""
    raise NotImplementedError
