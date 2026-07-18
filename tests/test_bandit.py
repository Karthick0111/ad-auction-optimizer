import random

import pytest

from simulation.bandit import ArmStats, ThompsonSamplingBandit


def test_arm_stats_running_mean_matches_direct_computation():
    arm = ArmStats(multiplier=1.0)
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    for r in rewards:
        arm.update(r)
    assert arm.n == len(rewards)
    assert arm.mean == pytest.approx(sum(rewards) / len(rewards))


def test_arm_stats_variance_matches_sample_variance():
    arm = ArmStats(multiplier=1.0)
    rewards = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    for r in rewards:
        arm.update(r)
    mean = sum(rewards) / len(rewards)
    expected_variance = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
    assert arm.variance == pytest.approx(expected_variance)


def test_bandit_state_round_trip_preserves_arm_stats():
    bandit = ThompsonSamplingBandit(seed=1)
    bandit.update(1.0, reward=3.0)
    bandit.update(1.0, reward=5.0)
    bandit.update(0.5, reward=-1.0)

    restored = ThompsonSamplingBandit.from_state(bandit.state(), seed=2)

    original_by_arm = {a.multiplier: a for a in bandit.arms}
    restored_by_arm = {a.multiplier: a for a in restored.arms}
    for multiplier, original_arm in original_by_arm.items():
        restored_arm = restored_by_arm[multiplier]
        assert restored_arm.n == original_arm.n
        assert restored_arm.mean == pytest.approx(original_arm.mean)
        assert restored_arm.m2 == pytest.approx(original_arm.m2)


def test_unknown_arm_multiplier_raises():
    bandit = ThompsonSamplingBandit(seed=1)
    with pytest.raises(ValueError):
        bandit.update(multiplier=99.0, reward=1.0)


def test_best_arm_in_hindsight_picks_highest_mean():
    bandit = ThompsonSamplingBandit(arms=(0.5, 1.0, 1.5), seed=1)
    for _ in range(20):
        bandit.update(0.5, reward=0.1)
        bandit.update(1.0, reward=5.0)
        bandit.update(1.5, reward=1.0)
    assert bandit.best_arm_in_hindsight().multiplier == 1.0


def test_thompson_sampling_converges_to_better_arm():
    """Statistical convergence check: given one clearly better arm, playing
    the bandit for many rounds should select the better arm substantially
    more often than a worse one, and cumulative regret vs the best arm in
    hindsight should shrink as a fraction of rounds played."""
    arms = (0.5, 1.0, 1.5)
    bandit = ThompsonSamplingBandit(arms=arms, seed=123)
    reward_rng = random.Random(456)

    true_means = {0.5: 0.0, 1.0: 2.0, 1.5: 0.5}
    n_rounds = 2000
    selections = {a: 0 for a in arms}
    cumulative_reward = 0.0

    for _ in range(n_rounds):
        arm = bandit.select_arm()
        selections[arm.multiplier] += 1
        reward = reward_rng.gauss(true_means[arm.multiplier], 1.0)
        bandit.update(arm.multiplier, reward)
        cumulative_reward += reward

    # The best true arm (1.0) should dominate selections after exploration.
    assert selections[1.0] > selections[0.5]
    assert selections[1.0] > selections[1.5]
    assert selections[1.0] > n_rounds * 0.5

    # Average regret per round vs the true best arm's mean should be small
    # relative to the reward scale - most rounds should have converged to
    # playing (close to) the best arm.
    best_possible = n_rounds * true_means[1.0]
    avg_regret_per_round = (best_possible - cumulative_reward) / n_rounds
    assert avg_regret_per_round < 0.5
