"""
Differential Sharpe Ratio (DSR) reward.

Moody & Wu (1997), "Optimisation of Trading Systems and Portfolios".

The Sharpe Ratio itself is a batch statistic (mean return / std of returns
over a whole episode) and can't be handed to an RL agent as a per-step
reward. The DSR is the first-order Taylor expansion of the Sharpe Ratio
with respect to a new return R_t, given exponential moving estimates of
the first and second moments of return (A_t, B_t). It gives a dense,
per-step signal that tells the agent whether THIS step's return improved
or worsened the running risk-adjusted performance, rather than just
raw PnL.

Update rule (with decay/adaptation rate eta):
    delta_A_t = R_t - A_{t-1}
    delta_B_t = R_t^2 - B_{t-1}
    D_t = (B_{t-1} * delta_A_t - 0.5 * A_{t-1} * delta_B_t) / (B_{t-1} - A_{t-1}^2)^(3/2)
    A_t = A_{t-1} + eta * delta_A_t
    B_t = B_{t-1} + eta * delta_B_t

D_t is the differential Sharpe ratio reward for step t.
"""

import numpy as np


class DifferentialSharpeRatio:
    """
    Stateful, per-episode reward shaper. Call reset() at the start of every
    episode and step(return) once per environment step.
    """

    def __init__(self, eta: float = 0.01, warmup_steps: int = 5, clip: float = 10.0):
        """
        Args:
            eta: EMA adaptation rate for the running moment estimates.
                 Smaller eta -> slower-adapting, smoother reward.
            warmup_steps: number of steps to return 0.0 while A, B have not
                 yet accumulated enough signal. The denominator
                 (B - A^2)^(3/2) is numerically unstable / meaningless
                 when B ~ A^2 (near-zero variance estimate), which is
                 exactly the condition at the start of an episode.
            clip: hard clip on the output reward. Financial return series
                 have fat tails; without a clip, a single large outlier
                 return can produce a reward magnitude that dominates the
                 training signal for the rest of the episode.
        """
        self.eta = eta
        self.warmup_steps = warmup_steps
        self.clip = clip
        self.reset()

    def reset(self):
        """Start a fresh episode. A, B are the running first/second moment
        estimates of the per-step return, both start at zero."""
        self.A = 0.0
        self.B = 0.0
        self._steps_seen = 0

    def step(self, return_t: float) -> float:
        """
        Consume one realised per-step return and produce the DSR reward.
        This mutates internal EMA state -- call exactly once per environment
        step. Calling it twice in one step (as the pre-fix version of this
        project's code did) silently corrupts A/B by double-advancing them.
        """
        self._steps_seen += 1

        delta_A = return_t - self.A
        delta_B = (return_t ** 2) - self.B

        denom = (self.B - self.A ** 2)
        if self._steps_seen <= self.warmup_steps or denom <= 1e-8:
            reward = 0.0
        else:
            denom_pow = denom ** 1.5
            reward = (self.B * delta_A - 0.5 * self.A * delta_B) / denom_pow
            reward = float(np.clip(reward, -self.clip, self.clip))

        # Update running moments AFTER computing the reward, using the
        # pre-update A/B (standard DSR recursion -- the reward at t uses
        # A_{t-1}, B_{t-1}, then we roll them forward for t+1).
        self.A = self.A + self.eta * delta_A
        self.B = self.B + self.eta * delta_B

        return reward