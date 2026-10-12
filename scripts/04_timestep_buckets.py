"""
Milestone 5 / Day 2: static vs time-bucketed vs per-timestep vs dynamic.

TODO:
  - for num_buckets in [1, 2, 3 (early/mid/late), 4, 8, num_steps]:
        uniform_buckets -> calibrate -> evaluate
  - clustered_buckets (from 01's stats) for the same K -> calibrate -> evaluate
  - plot: quality (LPIPS vs FP, later FID) vs number of buckets, uniform vs clustered
  - plot: error vs timestep per method
  - report overhead: #scale params stored per method (trivial, but reviewers ask)
  - run at W8A8 AND a harder setting (W8A6 / W4A8): time-awareness probably only matters when bits are tight
"""

if __name__ == "__main__":
    raise NotImplementedError
