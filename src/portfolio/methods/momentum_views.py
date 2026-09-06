"""
Momentum-based view generation for Black-Litterman.

Resolves two backlog items simultaneously, since the project's own
answers converged on the same thing: the deferred "feed trailing returns
into the view matrix" piece from the Black-Litterman sequencing decision
IS the Factor-Based "Option 2" (momentum/low-volatility tilt) method --
"simple, intuitive rules... to feed into your Black-Litterman view
engine" is a description of exactly what this file builds.

WHAT THIS GENERATES: n ABSOLUTE views (P = identity matrix), one per
asset -- "asset i's expected return is Q_i" -- rather than a full set of
pairwise relative views (which would need n*(n-1)/2 views and does not
scale cleanly as the universe grows). Q_i is derived from each asset's
TRAILING CUMULATIVE log-return over the window (the standard
factor-investing definition of momentum -- distinct from the trailing
MEAN return already used elsewhere as expected_returns, e.g. in
Portfolio.propose_weights), cross-sectionally standardised, then mapped
back into return-space by scaling by the asset's own trailing volatility
-- so Q_i has genuine return units rather than being an arbitrary
unitless score.

RISK SCALING (the second thing this piece was built to test): Omega
(view uncertainty) is NOT a fixed confidence level. Each view's
uncertainty scales with the STANDARD ERROR OF THE MEAN for that asset's
returns (std / sqrt(T)) -- the exact same building block already used in
BoxUncertaintyEstimator.shrink_returns() for a different purpose
(shrinking mu directly rather than sizing a view's confidence). Reusing
it here is deliberate, not incidental: an asset with a noisy, short, or
highly volatile return history gets a WIDE Omega (a weak view -- the
Black-Litterman posterior stays close to the market-equilibrium prior
for that asset), while an asset with a stable, well-estimated signal
gets a TIGHT Omega (a strong view -- the posterior shifts toward it).
This is what makes the resulting views actually risk-aware, rather than
uniformly confident regardless of estimation quality.

LOW-VOLATILITY TILT (mentioned alongside momentum in the Factor-Based
sequencing answer) is NOT built here -- a natural next variant sharing
this exact same P/Omega machinery, just with a different SIGNAL function
(trailing volatility instead of cumulative return). Deferred as its own
small, separate piece rather than bundled in, consistent with this
project's one-piece-at-a-time practice.
"""
import numpy as np


class MomentumViewGenerator:
    def __init__(self, tilt_strength: float = 1.0, omega_scale: float = 1.0):
        """
        tilt_strength: multiplier converting a standardised momentum
            z-score into a return-space view magnitude. 1.0 means a
            one-standard-deviation momentum signal produces a view
            magnitude equal to one unit of the asset's own trailing
            volatility -- an illustrative default, not calibrated.
        omega_scale: multiplier on the standard-error-based view
            variance. Larger values make every view LESS confident
            (wider Omega) uniformly; the RELATIVE confidence between
            assets is still driven entirely by their own standard errors,
            not by this constant.
        """
        if tilt_strength <= 0:
            raise ValueError("tilt_strength must be positive.")
        if omega_scale <= 0:
            raise ValueError("omega_scale must be positive.")
        self.tilt_strength = tilt_strength
        self.omega_scale = omega_scale

    def generate_views(self, returns_sample: np.ndarray):
        """
        returns_sample: (T, n) trailing realised log-returns.
        Returns (P, Q, omega) ready to pass directly to
        BlackLittermanEstimator.estimate().
        """
        returns = np.asarray(returns_sample, dtype=np.float64)
        T, n = returns.shape
        if T < 2:
            raise ValueError("generate_views needs at least 2 observations to estimate volatility/standard error.")

        # Momentum signal: trailing CUMULATIVE log-return (sum, not
        # mean) -- log-returns compound additively, so this is exactly
        # the total realised return over the window. Deliberately
        # distinct from the trailing MEAN return already used elsewhere
        # as expected_returns (e.g. Portfolio.propose_weights) -- using
        # the same quantity under a different name would make this
        # "view generator" a relabelling exercise rather than a genuinely
        # different signal.
        cumulative_return = returns.sum(axis=0)  # (n,)

        # Cross-sectional standardisation: momentum is inherently a
        # RELATIVE signal (this asset vs. the rest of the universe right
        # now), not an absolute one -- a z-score across assets, not
        # across time.
        mean_mom = cumulative_return.mean()
        std_mom = cumulative_return.std(ddof=1) if n > 1 else 1.0
        if std_mom < 1e-12:
            momentum_z = np.zeros(n)  # no cross-sectional dispersion -- no informative signal
        else:
            momentum_z = (cumulative_return - mean_mom) / std_mom

        # Map the unitless z-score back into RETURN-SPACE by scaling by
        # each asset's own trailing per-period volatility -- Q_i then has
        # genuine return units, not an arbitrary score.
        asset_vol = returns.std(axis=0, ddof=1)
        Q = self.tilt_strength * momentum_z * asset_vol  # (n,)

        # P: n ABSOLUTE views, one per asset ("asset i's return is Q_i"),
        # not a full relative-view set (n*(n-1)/2 pairs) -- kept simple
        # and O(n), not combinatorial.
        P = np.eye(n)

        # Omega: RISK-SCALED view uncertainty. Each view's variance is
        # the standard error of the mean for that asset's OWN return
        # series (std / sqrt(T)), squared -- the same std/sqrt(T)
        # building block BoxUncertaintyEstimator uses for a different
        # purpose. A noisy/short/volatile history -> wide Omega -> weak
        # view; a stable, well-sampled history -> tight Omega -> strong
        # view.
        standard_error = asset_vol / np.sqrt(T)
        omega = np.diag(self.omega_scale * (standard_error ** 2) + 1e-12)  # +eps: guards a
                                                                            # zero-variance asset
                                                                            # from producing a
                                                                            # singular Omega

        return P, Q, omega