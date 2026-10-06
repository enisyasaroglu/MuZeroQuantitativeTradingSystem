"""
Buy-and-Hold: the trivial passive benchmark. Always returns Long
(action=2), every step, regardless of input. No internal state, no
parameters.

This mirrors evaluate.py's existing hardcoded Buy-and-Hold path
(`action_fn = lambda obs: 2` inside evaluate_buy_and_hold()) exactly --
wrapping it in this class doesn't change behaviour. It gives Buy-and-
Hold the same Strategy-shaped interface SMAStrategy uses, so both can be
dispatched through one shared code path (e.g. --agent sma / --agent
buy_and_hold) instead of Buy-and-Hold living as a special-cased lambda
in evaluate.py.

INTERFACE ASSUMPTION, NOT YET CONFIRMED: this follows
select_action(obs, deterministic=...) -> (action, info), matching
MuZeroAgent/PPOAgent's calling convention -- I haven't seen SMAStrategy
yet, so if it uses a different method name or return shape, this needs
to match it instead.
"""


class BuyAndHoldStrategy:
    def select_action(self, observation, deterministic: bool = True):
        """
        Always returns action=2 (Long). `observation` and
        `deterministic` are accepted and ignored -- present only so this
        strategy is a drop-in wherever MuZero/PPO are already called.

        Returns (action, info): info is an empty dict, not None, so
        callers that unpack extra fields from agents (e.g. probs/value)
        don't need a special case for this strategy -- it just has
        nothing meaningful to report there.
        """
        return 2, {}

    def reset(self):
        """No internal state to reset. Present for interface symmetry
        with any stateful strategy (e.g. SMAStrategy likely tracks a
        rolling window and needs this)."""
        pass