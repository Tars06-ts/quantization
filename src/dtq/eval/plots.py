"""
Plots + image grids. Every figure the paper needs should be produced by a function here
from a saved CSV/JSON — never only inside a script.

Replaces: the matplotlib blocks at the end of every notebook cell.
"""


def image_grid(rows: dict, col_titles=None, path=None):
    """TODO: rows = {method_name: [PIL images]} -> one comparison figure. (Your old fp16/int8/selective grid.)"""
    raise NotImplementedError


def plot_stat_vs_timestep(df, stat: str = "absmax", by: str = "component", path=None):
    """
    TODO: THE key motivating figure. x = timestep, y = activation stat, one line per component
    (or per block). Log-scale y probably. If the lines are flat, the time-aware story is weak —
    find that out early!
    """
    raise NotImplementedError


def plot_error_vs_timestep(df, metric: str = "rel_mse", by: str = "method", path=None):
    """TODO: quant error vs timestep for static vs bucketed vs dynamic. Second key figure."""
    raise NotImplementedError


def heatmap_layer_timestep(df, value: str = "rel_mse", path=None):
    """TODO: rows = layers (block x component), cols = timesteps. Shows where+when sensitivity lives."""
    raise NotImplementedError
