"""
Regime detection: classifies each timestep into a market regime (e.g.
bear / sideways / bull) using a Gaussian Hidden Markov Model fit on
rolling return and volatility features.

This directly targets the "regime hysteresis" failure mode documented in
the dissertation: MuZero correctly learned a defensive/short-biased
policy during the 2022 bear market but had no mechanism to detect that
the regime had shifted back to a bull market in 2023-2024, and so
continued applying a policy that was no longer appropriate.

Design notes / scope:
  - This version detects regime from OBSERVABLE market features (rolling
    return, rolling volatility), not from MuZero's internal latent state.
    The dissertation's future-work section specifically proposed running
    the HMM on latent states; that is a natural next step once a trained
    MuZero checkpoint is available; this module is deliberately
    independent of any one agent so it's usable as: (a) an additional
    environment observation feature, (b) a diagnostic/evaluation tool
    ("what regime was the agent actually operating in when it made this
    decision"), and (c) an input to portfolio construction.
  - Like every other statistic in this pipeline, the HMM is fit on the
    TRAINING split only and then applied (via .predict(), not .fit()) to
    validation/test -- fitting it on val/test would leak future regime
    information into the features an agent sees.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class RegimeDetector:
    """
    Fits a Gaussian HMM with `n_states` hidden states on
    [rolling mean return, rolling volatility] and labels each state by its
    mean return (lowest -> "bear", highest -> "bull", middle -> "sideways")
    so labels are semantically stable across re-fits.
    """

    def __init__(self, n_states: int = 3, window: int = 20, random_state: int = 42):
        self.n_states = n_states
        self.window = window
        self.random_state = random_state
        self.model = None
        self._state_to_label = {}
        self._label_order = ["bear", "sideways", "bull"][:n_states] if n_states <= 3 else None
        self._feature_mean = None
        self._feature_std = None

    def _make_features(self, log_returns: pd.Series) -> pd.DataFrame:
        """Rolling mean return and rolling volatility, both computed only
        from information available up to and including the current step
        (no centered/forward-looking windows)."""
        roll_mean = log_returns.rolling(self.window, min_periods=self.window).mean()
        roll_std = log_returns.rolling(self.window, min_periods=self.window).std()
        feats = pd.concat([roll_mean, roll_std], axis=1).dropna()
        feats.columns = ["roll_mean", "roll_std"]
        return feats

    def _standardize(self, feats: pd.DataFrame, fit: bool) -> np.ndarray:
        """
        Z-score roll_mean/roll_std before fitting the HMM.

        roll_mean (~1e-4 scale) and roll_std (~1e-2 scale) differ by
        ~2 orders of magnitude. Left unstandardized, this repeatedly
        caused EM to collapse: components would initialise onto a scale
        that only fit one of the two features well, one component would
        absorb essentially all the responsibility, and the other
        components' covariances would blow up to an uninformative
        fallback value -- at which point Viterbi decoding just picks the
        single well-fit component for every timestep regardless of the
        true underlying regime. Standardizing both features to comparable
        scale before fitting is the standard fix.

        fit=True computes and stores mean/std (call only on the training
        split); fit=False reuses previously stored training statistics
        (call on val/test), matching DataProcessor.normalize()'s
        train-only pattern.
        """
        if fit:
            self._feature_mean = feats.mean()
            self._feature_std = feats.std()
        if self._feature_mean is None:
            raise RuntimeError("Detector must be fit() before standardizing new data.")
        return ((feats - self._feature_mean) / (self._feature_std + 1e-12)).values

    def fit(self, train_log_returns: pd.Series, n_init: int = 10):
        """
        Fit on TRAINING data only.

        HMM parameter estimation (EM / Baum-Welch) is only guaranteed to
        find a local optimum, and a single random initialisation can
        converge to a degenerate solution (e.g. two states collapsing onto
        nearly the same mean while a third, genuinely distinct regime is
        left unreachable in the transition matrix). n_init independent
        fits are run from different random seeds and the highest
        log-likelihood model is kept -- the same practice sklearn's
        GaussianMixture uses via its own n_init parameter.
        """
        feats = self._make_features(train_log_returns)
        X = self._standardize(feats, fit=True)

        best_model, best_score = None, -np.inf
        for i in range(n_init):
            candidate = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=200,
                random_state=self.random_state + i,
            )
            candidate.fit(X)
            score = candidate.score(X)
            if np.isfinite(score) and score > best_score:
                best_model, best_score = candidate, score

        self.model = best_model

        # Map raw HMM state indices -> semantic labels by mean-return rank
        # (in STANDARDIZED space, which preserves ordering since it's a
        # monotonic affine transform of the original mean-return feature),
        # so "state 2" isn't an arbitrary, re-shuffled label across refits.
        state_means = self.model.means_[:, 0]  # standardized mean-return dimension
        order = np.argsort(state_means)  # ascending: lowest return first
        if self._label_order is not None:
            self._state_to_label = {state: label for state, label in zip(order, self._label_order)}
        else:
            self._state_to_label = {state: f"regime_{rank}" for rank, state in enumerate(order)}
        return self

    def predict(self, log_returns: pd.Series) -> pd.DataFrame:
        """
        Predict regime labels for a (val/test or any) log-return series
        using the ALREADY-FITTED model. Returns a DataFrame aligned to the
        input's index (rows lost to the rolling-window warmup are dropped,
        matching how technical indicators are handled elsewhere in this
        pipeline).
        """
        if self.model is None:
            raise RuntimeError("RegimeDetector.fit() must be called (on training data) before predict().")

        feats = self._make_features(log_returns)
        X = self._standardize(feats, fit=False)
        state_seq = self.model.predict(X)
        labels = [self._state_to_label[s] for s in state_seq]

        # One-hot regime columns. Derived from the FULL fitted label set
        # (self._state_to_label.values()), not from labels observed in
        # THIS call's data -- using sorted(set(labels)) here previously
        # meant a split that happened not to visit every regime would
        # produce a different, narrower column set than a split that did,
        # a real schema mismatch for n_states > 3 (dormant at the
        # default n_states=3, which always uses the fixed _label_order).
        result = pd.DataFrame({"regime": labels}, index=feats.index)
        all_labels = self._label_order or sorted(set(self._state_to_label.values()))
        for label in all_labels:
            result[f"regime_{label}"] = (result["regime"] == label).astype(float)
        return result

    def fit_predict_pipeline(self, train_returns: pd.Series, *other_split_returns: pd.Series):
        """
        Convenience: fit on train_returns, then predict on train_returns
        itself plus every other split passed in (val, test, ...), all
        using the train-fitted model. Mirrors DataProcessor.normalize()'s
        train-only-statistics pattern.
        """
        self.fit(train_returns)
        return tuple(self.predict(r) for r in (train_returns,) + other_split_returns)