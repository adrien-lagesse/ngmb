"""
Module implementing random graph functions.
"""


__all__ = [
    "bernoulli_corruption",
    "bfs_sub_sampling",
    "erdos_renyi",
    "uniform_sub_sampling",
]

from ._random import (
    bernoulli_corruption,
    bfs_sub_sampling,
    erdos_renyi,
    uniform_sub_sampling,
)
