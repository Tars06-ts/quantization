"""
Forward hooks that record activation statistics per (layer, timestep).

Replaces: get_activation_hook() in mse-based-quant.

Problems with the old hook to fix:
  1. It stored the full output tensor (.cpu().float()) and OVERWROTE it every call,
     so after a 50-step run you only had the last step. Store reduced statistics,
     keyed by timestep, instead of raw tensors.
  2. It hooked the OUTPUT. For activation quantization of a Linear you care about
     its INPUT (that's what gets quantized before the matmul). Output error is
     still useful as a sensitivity signal — keep both, but know which is which.
  3. .cpu() every call is slow. Reduce on GPU, move a few scalars to CPU.
"""


class ActivationStats:
    """
    Accumulates per-(layer, timestep) statistics of Linear INPUTS.

    TODO: for each hooked layer and each timestep (from TimestepTracker), track
      - min, max, absmax
      - mean, std  (running, e.g. Welford, or sum / sumsq / count)
      - a high percentile (p99.9) — needs a sample or a histogram; torch.quantile on a
        random subset of elements is fine
      - outlier fraction: share of |x| > k * std (pick k, e.g. 6) — define it once, write it in docs
      - per-channel absmax (shape [in_features]) — needed later for per-channel / smoothing ideas
    Optionally: split cond vs uncond halves of the CFG batch — are they different?

    API sketch:
      stats = ActivationStats(tracker)
      stats.attach(model, layer_names)
      ... run sampling ...
      stats.detach()
      df = stats.to_dataframe()   # rows: layer, component, block, t, min, max, ...
      stats.save(path)
    """

    def __init__(self, tracker):
        raise NotImplementedError

    def attach(self, model, layer_names):
        raise NotImplementedError

    def detach(self):
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError


class OutputErrorProbe:
    """
    Per-(layer, timestep) error of a quantized model vs the FP reference.

    The upgrade of your old MSE experiment: same idea, but along the whole trajectory.
    TODO — think carefully about HOW to compare:
      (a) run FP and quant models on the SAME input at every step (teacher-forced):
          isolates per-step quantization error.
      (b) run each model on its own trajectory: includes error accumulation across steps.
      Both are interesting. (a) is easier to interpret; (b) is what the image sees.
      Implement (a) first: at each step feed the FP latent to both transformers.
    Metrics per layer per t: MSE, relative MSE (MSE / var(fp)), cosine similarity.
    Relative error matters — raw MSE is dominated by layers with big activations.
    """

    def __init__(self, tracker):
        raise NotImplementedError

    def attach(self, fp_model, q_model, layer_names):
        raise NotImplementedError

    def detach(self):
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError
