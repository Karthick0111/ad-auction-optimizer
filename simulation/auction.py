"""
Second-price ("Vickrey") auction settlement logic - pure, no AWS/IO
dependencies so it's trivially unit-testable and reusable from both the
bid_consumer Lambda and local tests.
"""
from dataclasses import dataclass
import random


@dataclass
class AuctionResult:
    won: bool
    price_paid: float
    our_bid: float
    highest_competitor_bid: float


def draw_competitor_bids(n: int, mean_log_bid: float, sigma: float, rng: random.Random) -> list:
    """Draws n competitor bids from a lognormal distribution.

    mean_log_bid/sigma parameterize the underlying normal (location/scale),
    matching random.lognormvariate's signature - not the lognormal's own
    mean/stdev. Higher mean_log_bid/sigma = a more competitive auction.
    """
    return [rng.lognormvariate(mean_log_bid, sigma) for _ in range(n)]


def settle_second_price(our_bid: float, competitor_bids: list) -> AuctionResult:
    """Settles a second-price auction: we win if our bid exceeds every
    competitor bid, and if we win we pay the highest competitor bid (never
    more than our own bid) - the property that makes truthful bidding
    optimal in a Vickrey auction. Ties go to the competitor (conservative -
    never over-claims a win). No competitors means a free win.
    """
    highest_competitor = max(competitor_bids) if competitor_bids else 0.0
    won = our_bid > highest_competitor
    price_paid = highest_competitor if won else 0.0
    return AuctionResult(
        won=won,
        price_paid=price_paid,
        our_bid=our_bid,
        highest_competitor_bid=highest_competitor,
    )
