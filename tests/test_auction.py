import random

import pytest

from simulation.auction import draw_competitor_bids, settle_second_price


def test_winning_bid_pays_second_price():
    result = settle_second_price(our_bid=5.0, competitor_bids=[1.0, 3.0, 4.5])
    assert result.won is True
    assert result.price_paid == 4.5
    assert result.highest_competitor_bid == 4.5


def test_losing_bid_pays_nothing():
    result = settle_second_price(our_bid=2.0, competitor_bids=[1.0, 3.0, 4.5])
    assert result.won is False
    assert result.price_paid == 0.0


def test_tie_goes_to_competitor():
    result = settle_second_price(our_bid=4.5, competitor_bids=[1.0, 4.5])
    assert result.won is False
    assert result.price_paid == 0.0


def test_no_competitors_is_a_free_win():
    result = settle_second_price(our_bid=1.0, competitor_bids=[])
    assert result.won is True
    assert result.price_paid == 0.0
    assert result.highest_competitor_bid == 0.0


def test_price_paid_never_exceeds_our_bid():
    result = settle_second_price(our_bid=10.0, competitor_bids=[9.99])
    assert result.won is True
    assert result.price_paid == pytest.approx(9.99)
    assert result.price_paid <= result.our_bid


def test_draw_competitor_bids_returns_n_positive_bids():
    rng = random.Random(42)
    bids = draw_competitor_bids(n=10, mean_log_bid=0.0, sigma=0.5, rng=rng)
    assert len(bids) == 10
    assert all(b > 0 for b in bids)


def test_draw_competitor_bids_is_reproducible_with_seed():
    bids_a = draw_competitor_bids(n=5, mean_log_bid=0.5, sigma=0.3, rng=random.Random(7))
    bids_b = draw_competitor_bids(n=5, mean_log_bid=0.5, sigma=0.3, rng=random.Random(7))
    assert bids_a == bids_b
