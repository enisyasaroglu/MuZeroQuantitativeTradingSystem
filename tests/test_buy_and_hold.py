import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.strategies.buy_and_hold import BuyAndHoldStrategy


def test_always_returns_long():
    strat = BuyAndHoldStrategy()
    for _ in range(10):
        action, info = strat.select_action(observation=None)
        assert action == 2


def test_ignores_observation_content():
    strat = BuyAndHoldStrategy()
    action_a, _ = strat.select_action(observation="anything")
    action_b, _ = strat.select_action(observation=12345)
    assert action_a == action_b == 2


def test_deterministic_flag_accepted_and_ignored():
    strat = BuyAndHoldStrategy()
    action_det, _ = strat.select_action(observation=None, deterministic=True)
    action_stoch, _ = strat.select_action(observation=None, deterministic=False)
    assert action_det == action_stoch == 2


def test_reset_does_not_raise():
    strat = BuyAndHoldStrategy()
    strat.reset()