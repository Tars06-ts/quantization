"""
Know WHICH denoising step the model is in, from inside any layer.

This is the core plumbing for everything time-aware. Nothing like this existed
in the notebooks — mse-based-quant evaluated at a single dummy timestep (t=50)
with random latents, which is exactly the blind spot this project is about.
"""


class TimestepTracker:
    """
    TODO:
      - attach(transformer): register_forward_pre_hook(..., with_kwargs=True) on the
        top-level transformer, read kwargs["timestep"] and store it (+ a step counter).
      - expose .t (the scheduler timestep, e.g. 999..0) and .step (0..num_steps-1)
      - reset() between images/trajectories
      - detach()
    Questions:
      - With CFG the batch is 2x — is timestep a scalar or a vector? Same value for all?
      - Do you key things by .t or by .step? (.step only makes sense for a fixed scheduler+num_steps;
        .t generalizes across step counts — matters if you later vary num_steps.)
    """

    def attach(self, transformer):
        raise NotImplementedError

    def detach(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


def uniform_buckets(timesteps, num_buckets: int) -> dict:
    """
    TODO: split the ordered list of timesteps into `num_buckets` contiguous groups.
    Return {timestep: bucket_id}. num_buckets=1 -> static; =len(timesteps) -> per-timestep.
    """
    raise NotImplementedError


def clustered_buckets(per_t_features: dict, num_buckets: int, contiguous: bool = True) -> dict:
    """
    TODO (Day 2): group timesteps by activation similarity instead of uniformly.
      per_t_features: {t: feature vector} e.g. concat of per-layer absmax / p99.9.
      Options: k-means; or contiguous segmentation via DP minimizing within-group variance
      (contiguous = cheaper to implement at inference, just a range check).
    This is a candidate paper contribution — keep the code clean.
    """
    raise NotImplementedError
