# Problem statement (one page — write this yourself)

## 1. Why do DiT activations vary across timesteps?
<!-- TODO: input x_t changes from pure noise to clean latent; timestep embedding -> adaLN shift/scale/gate
     changes every block's input distribution. Back it with plots from scripts/01_collect_stats.py. -->

## 2. Why can static (all-timestep) quantization fail?
<!-- TODO: one scale must cover the union of ranges -> either clips at some t or wastes resolution at others.
     Which error matters more, and at which t? -->

## 3. What does time-aware quantization fix, and at what cost?
<!-- TODO: per-t / per-bucket scales, dynamic scales, mixed precision. Overhead: storage, lookups, kernel support. -->

## 4. What do PTQ4DiT / Q-DiT / TQ-DiT / TaQ-DiT (and TDQ / TFMQ-DM) already do?
<!-- TODO: 2–3 lines each + the gap you're going after. -->

| Paper | Key idea | Time-aware how? | Setting (bits, steps, FID protocol) |
|---|---|---|---|
| PTQ4DiT | | | |
| Q-DiT | | | |
| TQ-DiT | | | |
| TaQ-DiT | | | |
| TDQ (UNet) | | | |
| TFMQ-DM (UNet) | | | |

## 5. My hypothesis / contribution (1–2 sentences)
<!-- TODO -->
