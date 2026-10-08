import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.risk.risk_manager import RiskManager


def test_no_trigger_below_threshold():
    """Drawdown below the threshold never overrides the agent's action."""
    rm = RiskManager(max_drawdown=0.20, cooldown_steps=5)
    rm.reset(initial_value=100_000.0)
    action = rm.filter_action(proposed_action=2, current_portfolio_value=90_000.0)  # -10%
    assert action == 2
    assert rm.triggered_count == 0


def test_triggers_exactly_at_threshold():
    """Drawdown at or beyond the threshold forces Neutral."""
    rm = RiskManager(max_drawdown=0.20, cooldown_steps=5)
    rm.reset(initial_value=100_000.0)
    action = rm.filter_action(proposed_action=2, current_portfolio_value=80_000.0)  # exactly -20%
    assert action == RiskManager.NEUTRAL_ACTION
    assert rm.triggered_count == 1


def test_cooldown_forces_flat_for_configured_steps():
    """After a trigger, the next `cooldown_steps` calls stay forced-flat
    even if drawdown recovers, then release on the step after."""
    rm = RiskManager(max_drawdown=0.20, cooldown_steps=3)
    rm.reset(initial_value=100_000.0)

    rm.filter_action(proposed_action=2, current_portfolio_value=80_000.0)  # triggers
    assert rm.triggered_count == 1

    # Portfolio "recovers" immediately, but cooldown still forces flat.
    for _ in range(3):
        action = rm.filter_action(proposed_action=2, current_portfolio_value=100_000.0)
        assert action == RiskManager.NEUTRAL_ACTION

    # Cooldown exhausted -- agent's action passes through again.
    action = rm.filter_action(proposed_action=2, current_portfolio_value=100_000.0)
    assert action == 2


def test_peak_tracks_new_highs():
    """Drawdown is measured from the running peak, not the initial value
    -- a new high resets the reference point for future drawdown."""
    rm = RiskManager(max_drawdown=0.20, cooldown_steps=5)
    rm.reset(initial_value=100_000.0)

    rm.filter_action(proposed_action=2, current_portfolio_value=150_000.0)  # new peak
    # -19% from the NEW peak of 150k is 121,500 -- should not trigger.
    action = rm.filter_action(proposed_action=2, current_portfolio_value=121_500.0)
    assert action == 2
    assert rm.triggered_count == 0

    # -20%+ from the 150k peak should trigger.
    action = rm.filter_action(proposed_action=2, current_portfolio_value=119_000.0)
    assert action == RiskManager.NEUTRAL_ACTION


def test_reset_clears_state_between_episodes():
    rm = RiskManager(max_drawdown=0.20, cooldown_steps=5)
    rm.reset(initial_value=100_000.0)
    rm.filter_action(proposed_action=2, current_portfolio_value=80_000.0)  # trigger
    assert rm.triggered_count == 1

    rm.reset(initial_value=50_000.0)  # new episode, different starting capital
    assert rm.triggered_count == 0
    action = rm.filter_action(proposed_action=2, current_portfolio_value=45_000.0)  # -10%, not -20%
    assert action == 2


def test_filter_action_before_reset_raises():
    rm = RiskManager()
    try:
        rm.filter_action(proposed_action=2, current_portfolio_value=100_000.0)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    
def test_kill_switch_does_not_relatch_after_cooldown_when_flat():
    """While the portfolio is flat its value cannot rise, so the drawdown
    stays the same. After the cooldown the agent must get a fresh start
    and not be forced flat again immediately."""
    rm = RiskManager(max_drawdown=0.20, cooldown_steps=3)
    rm.reset(initial_value=100_000.0)

    rm.filter_action(proposed_action=2, current_portfolio_value=80_000.0)  # triggers
    for _ in range(3):
        rm.filter_action(proposed_action=2, current_portfolio_value=80_000.0)  # cooldown

    action = rm.filter_action(proposed_action=2, current_portfolio_value=80_000.0)
    assert action == 2
    assert rm.triggered_count == 1