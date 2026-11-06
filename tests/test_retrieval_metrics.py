import pytest


@pytest.mark.parametrize("ranks,expected", [([1, None], .5), ([None, None], 0.), ([1, 5, None], .4)])
def test_mrr_includes_all_queries(ranks, expected):
    from src.evaluation.retrieval_metrics import mean_reciprocal_rank
    assert mean_reciprocal_rank(ranks) == expected


def test_rank_regression_direction():
    from src.evaluation.retrieval_metrics import rank_regressed
    assert rank_regressed(1, 5)
    assert rank_regressed(1, None)
    assert not rank_regressed(5, 1)
    assert not rank_regressed(None, 1)
    assert not rank_regressed(None, None)
