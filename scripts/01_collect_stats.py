"""
Milestone 2: activation statistics across timesteps (FP16 model, no quantization).

New — the notebooks never did this.

TODO:
  - load pipe, TimestepTracker.attach(pipe.transformer)
  - ActivationStats on layers.quantizable_linears(...)
  - sample N images (e.g. 32, random classes, calib seed)
  - save stats dataframe -> results/01_stats/stats.parquet (or csv)
  - plots.plot_stat_vs_timestep for absmax / std / outlier_frac, per component
  - plots.heatmap_layer_timestep for absmax

Questions to answer in docs/problem_statement.md from these plots:
  - which components' ranges move most with t? (guess first, then check: adaLN outputs? fc2 input?)
  - does range grow/shrink toward t=0?
  - do outliers sit in fixed channels (channel-wise structure) or move around?
"""

if __name__ == "__main__":
    raise NotImplementedError
