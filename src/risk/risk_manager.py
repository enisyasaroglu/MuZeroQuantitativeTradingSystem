"""
Risk management overlay: sits between an agent's raw action and the
environment's step(), and can override that action based on realised
portfolio risk.

Deliberately a SEPARATE, stateful component from both the agent and the
environment -- matching the standard quant-systems separation between
"Alpha/Strategy" (what the model wants to do) and "Risk Engine" (whether
that's allowed to happen), rather than folding risk logic into the
network or the environment's reward function.

Two mechanisms, both simple, both standard:

1. Max-drawdown kill switch: once realised peak-to-trough drawdown
   breaches a threshold, force the position flat (Neutral) rather than
   letting the agent decide -- a hard capital-preservation floor that
   does not depend on the agent's judgement.
2. Cooldown: after a kill-switch trigger, stay flat for a fixed number
   of steps before allowing the agent's own actions through again,
   rather than immediately re-engaging the moment drawdown ticks back
   under the threshold.

Explicitly NOT included: volatility-scaled position sizing, per-trade
stop-losses, or dynamic leverage -- the action space here is
{short, neutral, long} with no notion of size, so risk management at
this stage means "when to override the discrete decision entirely", not
"how much to bet". Those are natural extensions once the action space
itself becomes continuous (see multi_asset_env.py's docstring for the
same scoping principle applied there).
"""


class RiskManager:
    NEUTRAL_ACTION = 1

    def __init__(self, max_drawdown: float = 0.20, cooldown_steps: int = 5):
        """
        Args:
            max_drawdown: positive fraction (e.g. 0.20 = 20%). Once
                realised drawdown from the running peak exceeds this,
                the kill switch engages. Not tuned/optimised -- a
                sensible starting default, same posture as
                temperature_schedule/entropy_coef_schedule's own
                documented defaults.
            cooldown_steps: number of steps to remain forced-flat after
                the kill switch engages, before the agent's own actions
                are allowed through again.
        """
        if max_drawdown <= 0:
            raise ValueError("max_drawdown must be a positive fraction, e.g. 0.20 for 20%.")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0.")
        self.max_drawdown = max_drawdown
        self.cooldown_steps = cooldown_steps

        self._peak_value = None
        self._cooldown_remaining = 0
        self._triggered_count = 0

    def reset(self, initial_value: float):
        """Call once per episode, alongside env.reset()."""
        self._peak_value = float(initial_value)
        self._cooldown_remaining = 0
        self._triggered_count = 0

    @property
    def triggered_count(self) -> int:
        """How many times the kill switch engaged this episode -- a
        diagnostic in its own right, not just a pass/fail flag."""
        return self._triggered_count

    def filter_action(self, proposed_action: int, current_portfolio_value: float) -> int:
        """
        Call BEFORE env.step(proposed_action). Returns the action that
        should actually be passed to the environment: either the agent's
        own choice, or NEUTRAL_ACTION if the kill switch is engaged.

        current_portfolio_value must be the value BEFORE this step is
        taken (i.e. env.portfolio_value as of the last step/reset) --
        drawdown is evaluated on realised history only, never on the
        outcome the proposed action would itself produce, so this cannot
        look ahead into the return of the decision being filtered.
        """
        if self._peak_value is None:
            raise RuntimeError("RiskManager.reset() must be called before filter_action().")

        self._peak_value = max(self._peak_value, current_portfolio_value)
        drawdown = (current_portfolio_value - self._peak_value) / self._peak_value

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return self.NEUTRAL_ACTION

        if drawdown <= -self.max_drawdown:
            self._triggered_count += 1
            self._cooldown_remaining = self.cooldown_steps
            # Restart the drawdown measurement from the current value, so
            # the agent gets a fresh start once the cooldown has ended.
            # Without this, a flat portfolio would stay below its old peak
            # for ever and the switch would fire again at once.
            self._peak_value = current_portfolio_value
            return self.NEUTRAL_ACTION

        return proposed_action