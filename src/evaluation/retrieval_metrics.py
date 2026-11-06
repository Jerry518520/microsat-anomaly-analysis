"""Retrieval metrics include misses in the denominator."""
from statistics import mean


def mean_reciprocal_rank(ranks):
    ranks = list(ranks)
    if any(rank is not None and (rank < 1 or int(rank) != rank) for rank in ranks):
        raise ValueError("ranks must be positive integers or None")
    return float(mean(1 / rank if rank is not None else 0. for rank in ranks)) if ranks else 0.


def rank_regressed(before, after):
    before = float("inf") if before is None else before
    after = float("inf") if after is None else after
    return after > before
