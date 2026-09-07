"""
Robust Portfolio Optimisation: two related, independently usable pieces,
per the project's own sequencing decision (build the simple baseline
first as a sanity/comparison point, then the resampling-based method as
the actual method used going forward).

1. BoxUncertaintyEstimator -- NOT a PortfolioOptimizerProtocol member,
   same category as BlackLittermanEstimator: it produces an ADJUSTED
   expected-returns vector, to feed into MeanVarianceOptimizer, not a
   weight vector directly. Implements the classical closed-form robust
   counterpart to Mean-Variance under BOX uncertainty on the mean
   estimate (Goldfarb & Iyengar, 2003; Ben-Tal & Nemirovski): for a
   long-only portfolio, the worst-case expected return over a box
   {mu : |mu_i - mu_hat_i| <= delta_i} is exactly mu_hat_i - delta_i,
   so the robust counterpart reduces to shrinking each asset's estimated
   mean return by delta_i before a standard Mean-Variance solve -- no
   new solver needed, and delta_i has a direct statistical
   interpretation (see class docstring). This is the "quick,
   zero-dependency baseline" from the sequencing decision.

2. RobustPortfolioOptimizer -- IS a PortfolioOptimizerProtocol member.
   Implements Michaud (1998) resampled efficient frontier: repeatedly
   bootstrap-resample the return history, solve Mean-Variance on each
   resample, and AVERAGE the resulting weight vectors. The averaging is
   the point -- a single Mean-Variance solve is highly sensitive to
   estimation error in mu/Sigma (a well-documented fragility); averaging
   many solves computed from perturbed versions of the same data damps
   that sensitivity. This is the project's CORE robust method going
   forward.

3. BoxUncertaintyOptimizerAdapter -- IS a PortfolioOptimizerProtocol
   member, added so BoxUncertaintyEstimator can actually participate in
   a Protocol-based method comparison (src/portfolio/comparison.py),
   the same reason BlackLittermanOptimizerAdapter exists for
   BlackLittermanEstimator. Chains: shrink expected returns via
   BoxUncertaintyEstimator, then solve Mean-Variance on the SHRUNK
   returns (not the raw sample mean).

Both (1) and (2) are independent, usable on their own -- this file does
NOT chain them together (e.g. applying the box-uncertainty shrinkage
inside each resample) since that would conflate two distinct,
separately-motivated robust techniques into one, undocumented hybrid.
Combining them deliberately is a reasonable future step, not assumed here.

DOCUMENTED SIMPLIFICATIONS (both honest, both explicit, matching this
project's practice elsewhere -- e.g. PSO's rolling_rebalance_weights):
  - RobustPortfolioOptimizer resamples via an i.i.d. ROW BOOTSTRAP of
    returns_sample, not Michaud's original parametric draw from a fitted
    multivariate normal, and not a BLOCK bootstrap. A plain row bootstrap
    ignores autocorrelation/volatility clustering in real return series;
    a block bootstrap would be the more correct choice for financial
    time series specifically. Kept simple deliberately -- a reasonable
    v1, with the more careful resampling scheme noted as future work.
  - BoxUncertaintyEstimator's delta_i is the STANDARD ERROR OF THE MEAN
    (std / sqrt(T)) scaled by a confidence z-score, treating each
    asset's returns as i.i.d. for that purpose -- the same simplifying
    assumption already made everywhere else sample covariance is
    estimated in this project (Mean-Variance, Risk Parity, CVaR).

scipy.stats is part of scipy, already required (Risk Parity's SLSQP
solve, HRP's clustering) -- no new requirements.txt entry needed.
"""
import numpy as np
from scipy.stats import norm

from src.portfolio.portfolio import OptimizationResult, OptimizationInputs
from src.portfolio.methods.mean_variance import MeanVarianceOptimizer


class BoxUncertaintyEstimator:
    """
    Produces a shrunk expected-returns vector to feed into
    MeanVarianceOptimizer -- the "quick, zero-dependency baseline"
    robust method. NOT a PortfolioOptimizerProtocol member (same
    category as BlackLittermanEstimator): its output is an INPUT to an
    optimizer, not a weight vector.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        confidence_level: e.g. 0.95 means each asset's expected return is
        shrunk down to (approximately) the lower bound of a one-sided 95%
        confidence interval on its true mean, given the sample in
        returns_sample. Higher confidence -> larger shrinkage -> more
        conservative resulting allocation.
        """
        if not (0.0 < confidence_level < 1.0):
            raise ValueError("confidence_level must be strictly between 0 and 1.")
        self.confidence_level = confidence_level

    def shrink_returns(self, returns_sample: np.ndarray) -> np.ndarray:
        """
        returns_sample: (T, n) realised returns.
        Returns: (n,) shrunk expected-returns vector, mu_hat_i - delta_i,
        where delta_i = z * (std_i / sqrt(T)) -- the standard error of
        the sample mean, scaled by the z-score for confidence_level.
        """
        returns = np.asarray(returns_sample, dtype=np.float64)
        T = returns.shape[0]

        mu_hat = returns.mean(axis=0)
        std = returns.std(axis=0, ddof=1)
        standard_error = std / np.sqrt(T)

        z = norm.ppf(self.confidence_level)
        delta = z * standard_error

        return mu_hat - delta


class RobustPortfolioOptimizer:
    """
    Michaud-style resampled efficient frontier. PortfolioOptimizerProtocol
    member -- produces weights directly, by averaging n_resamples
    independent Mean-Variance solves, each on a bootstrap resample of
    returns_sample.
    """

    def __init__(self, n_resamples: int = 200, risk_aversion: float = 1.0, random_state: int = 42):
        self.n_resamples = n_resamples
        self.risk_aversion = risk_aversion
        self.rng = np.random.RandomState(random_state)
        self._mvo = MeanVarianceOptimizer(risk_aversion=risk_aversion)

    def optimize(self, inputs, constraints) -> OptimizationResult:
        if inputs.returns_sample is None:
            raise ValueError("RobustPortfolioOptimizer requires returns_sample in OptimizationInputs.")

        returns = np.asarray(inputs.returns_sample, dtype=np.float64)
        T, n = returns.shape

        accumulated_weights = np.zeros(n)

        for _ in range(self.n_resamples):
            sample_idx = self.rng.randint(0, T, size=T)  # i.i.d. row bootstrap, with replacement
            resample = returns[sample_idx]

            mu_b = resample.mean(axis=0)
            cov_b = np.cov(resample, rowvar=False)

            # Each resample's Mean-Variance solve already respects
            # `constraints` individually -- but the AVERAGE of several
            # constraint-feasible weight vectors is not automatically
            # feasible itself under a cardinality limit (averaging can
            # reintroduce nonzero weight on an asset one resample zeroed
            # out and another kept), so the final average is re-projected
            # below rather than trusted as feasible on its own.
            result_b = self._mvo.optimize(
                OptimizationInputs(expected_returns=mu_b, cov_matrix=cov_b), constraints
            )
            accumulated_weights += result_b.weights

        averaged_weights = accumulated_weights / self.n_resamples
        weights = constraints.project(averaged_weights)

        return OptimizationResult(
            weights=weights,
            method_name="RobustResampled",
            converged=True,
            diagnostics={"n_resamples": self.n_resamples},
        )


class BoxUncertaintyOptimizerAdapter:
    """
    PortfolioOptimizerProtocol member wrapping BoxUncertaintyEstimator +
    MeanVarianceOptimizer -- see module docstring, item 3.
    """

    def __init__(self, confidence_level: float = 0.95, risk_aversion: float = 1.0):
        self._shrinker = BoxUncertaintyEstimator(confidence_level=confidence_level)
        self._mvo = MeanVarianceOptimizer(risk_aversion=risk_aversion)

    def optimize(self, inputs, constraints) -> OptimizationResult:
        if inputs.returns_sample is None:
            raise ValueError("BoxUncertaintyOptimizerAdapter requires returns_sample in OptimizationInputs.")
        if inputs.cov_matrix is None:
            raise ValueError("BoxUncertaintyOptimizerAdapter requires cov_matrix in OptimizationInputs.")

        shrunk_returns = self._shrinker.shrink_returns(inputs.returns_sample)
        mvo_result = self._mvo.optimize(
            OptimizationInputs(expected_returns=shrunk_returns, cov_matrix=inputs.cov_matrix),
            constraints,
        )
        return OptimizationResult(
            weights=mvo_result.weights,
            method_name="RobustBoxUncertainty",
            converged=mvo_result.converged,
            diagnostics={},
        )