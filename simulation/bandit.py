"""
Thompson Sampling multi-armed bandit for the bid-multiplier decision.

Rewards here are continuous (net auction value: click value minus price
paid, which can be negative), not binary click/no-click - so this uses a
Gaussian reward model rather than the classic Beta-Bernoulli variant.

Each arm tracks a running mean/variance of its observed rewards (Welford's
online algorithm - numerically stable, O(1) per update, no need to keep
every past reward). Thompson Sampling draws one sample per arm from
Normal(empirical_mean, standard_error_of_the_mean) and picks the highest
sample. This is the common "empirical Bayes" simplification of Gaussian
Thompson Sampling (it approximates the true Normal-Inverse-Gamma conjugate
posterior); simpler to explain and implement while keeping the essential
explore/exploit behavior - arms with little evidence have a wide sampling
distribution (more exploration), arms with lots of evidence concentrate
tightly around their true mean (more exploitation).
"""
from dataclasses import dataclass
import math
import random

DEFAULT_ARMS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


@dataclass
class ArmStats:
    multiplier: float
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Welford's running sum of squared deviations

    def update(self, reward: float) -> None:
        self.n += 1
        delta = reward - self.mean
        self.mean += delta / self.n
        delta2 = reward - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 1.0  # wide prior until we have evidence
        return self.m2 / (self.n - 1)

    @property
    def std_error(self) -> float:
        return math.sqrt(self.variance / max(self.n, 1))


class ThompsonSamplingBandit:
    def __init__(self, arms=DEFAULT_ARMS, seed: int | None = None):
        self.arms = [ArmStats(multiplier=m) for m in arms]
        self._rng = random.Random(seed)

    def select_arm(self) -> ArmStats:
        """Draws one sample per arm from its current belief distribution
        and returns the arm with the highest sample."""
        sampled = [
            (self._rng.gauss(arm.mean, max(arm.std_error, 1e-6)), arm)
            for arm in self.arms
        ]
        return max(sampled, key=lambda pair: pair[0])[1]

    def update(self, multiplier: float, reward: float) -> None:
        for arm in self.arms:
            if arm.multiplier == multiplier:
                arm.update(reward)
                return
        raise ValueError(f"Unknown arm multiplier: {multiplier}")

    def best_arm_in_hindsight(self) -> ArmStats:
        """Highest empirical-mean-reward arm so far - the regret baseline
        (cumulative reward if we'd always played this one arm, knowable
        only in hindsight)."""
        return max(self.arms, key=lambda arm: arm.mean)

    def state(self) -> dict:
        """Serializable snapshot for persisting bandit state between
        stateless Lambda invocations (e.g. in DynamoDB)."""
        return {
            str(arm.multiplier): {"n": arm.n, "mean": arm.mean, "m2": arm.m2}
            for arm in self.arms
        }

    @classmethod
    def from_state(cls, state: dict, arms=DEFAULT_ARMS, seed: int | None = None) -> "ThompsonSamplingBandit":
        bandit = cls(arms=arms, seed=seed)
        for arm in bandit.arms:
            saved = state.get(str(arm.multiplier))
            if saved:
                arm.n = saved["n"]
                arm.mean = saved["mean"]
                arm.m2 = saved["m2"]
        return bandit
