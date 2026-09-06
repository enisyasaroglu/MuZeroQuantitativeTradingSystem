"""
Hierarchical Risk Parity (Lopez de Prado, 2016): allocates using
CORRELATION STRUCTURE via hierarchical clustering, not mean-variance
optimisation. Group B, same interface as RiskParityOptimizer -- ignores
expected_returns entirely, needs only cov_matrix.

Three stages, each independently testable (see tests/test_hrp.py):
  1. Tree clustering: hierarchically cluster assets by correlation
     distance.
  2. Quasi-diagonalisation: reorder the covariance matrix so similar
     assets sit adjacent to each other (the tree's leaf order).
  3. Recursive bisection: walk DOWN the tree, splitting the allocation
     between each pair of sub-clusters in inverse proportion to their
     variance -- so a cluster with more internal risk gets a SMALLER
     share, not larger.

WHY THIS AVOIDS MEAN-VARIANCE'S KNOWN FRAGILITY: MVO and Risk Parity both
require inverting or otherwise using the FULL covariance matrix directly
-- Mean-Variance's closed-form solution is famously unstable when Sigma
is near-singular or poorly estimated (exactly the ridge-regularisation
concern already handled in mean_variance.py). HRP never inverts the full
matrix; it only ever uses SUB-matrices restricted to the two clusters
being split at each recursion step. This is HRP's specific, well-known
selling point in the portfolio-construction literature, not a marginal
detail.

Introduces scipy.cluster.hierarchy and scipy.spatial.distance as new
imports -- scipy itself was already added to requirements.txt for
risk_parity.py; no NEW dependency entry needed.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from src.portfolio.portfolio import OptimizationResult


def _cov_to_corr(cov_matrix: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov_matrix))
    corr = cov_matrix / np.outer(std, std)
    # Numerical clipping guards against |corr| slightly exceeding 1 from
    # floating-point error, which would otherwise break the distance
    # formula below (sqrt of a negative number).
    return np.clip(corr, -1.0, 1.0)


def _correlation_distance(corr_matrix: np.ndarray) -> np.ndarray:
    """Lopez de Prado's distance metric: d_ij = sqrt(0.5 * (1 - corr_ij)).
    Perfectly correlated assets (corr=1) have distance 0; perfectly
    anti-correlated (corr=-1) have maximum distance 1 -- a proper metric
    (satisfies the triangle inequality), unlike using (1 - corr) directly."""
    return np.sqrt(0.5 * (1.0 - corr_matrix))


def _quasi_diagonalize(link_matrix: np.ndarray) -> list:
    """Returns the leaf order (asset indices) from the linkage tree, via
    scipy's own dendrogram leaf-ordering -- assets that end up adjacent
    in this order are the most similar, by construction of the tree."""
    return dendrogram(link_matrix, no_plot=True)["leaves"]


def _cluster_variance(cov_matrix: np.ndarray, cluster_indices: list) -> float:
    """Variance of the INVERSE-VARIANCE-WEIGHTED portfolio restricted to
    `cluster_indices` -- NOT the raw sub-covariance's average variance.
    This is the specific quantity recursive bisection compares between
    sibling clusters; see _recursive_bisection below."""
    sub_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]
    inv_var = 1.0 / np.diag(sub_cov)
    weights = inv_var / inv_var.sum()
    return float(weights @ sub_cov @ weights)


def _recursive_bisection(cov_matrix: np.ndarray, sorted_indices: list) -> np.ndarray:
    """
    Walks DOWN the quasi-diagonalised order, splitting it into two
    contiguous halves at each step (mirroring the tree structure implied
    by the leaf order), and allocating the CURRENT weight budget between
    the two halves in INVERSE proportion to their cluster variance --
    the higher-variance half gets a SMALLER share.
    """
    n = len(cov_matrix)
    weights = np.ones(n)
    clusters = [sorted_indices]

    while clusters:
        clusters = [c for c in clusters if len(c) > 1]
        if not clusters:
            break
        new_clusters = []
        for cluster in clusters:
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]

            var_left = _cluster_variance(cov_matrix, left)
            var_right = _cluster_variance(cov_matrix, right)

            # Inverse-variance split: the riskier half gets LESS of the
            # current budget, not more.
            alloc_left = 1.0 - var_left / (var_left + var_right)
            alloc_right = 1.0 - alloc_left

            weights[left] *= alloc_left
            weights[right] *= alloc_right

            new_clusters.append(left)
            new_clusters.append(right)
        clusters = new_clusters

    return weights


class HierarchicalRiskParityOptimizer:
    def optimize(self, inputs, constraints) -> OptimizationResult:
        if inputs.cov_matrix is None:
            raise ValueError("HierarchicalRiskParityOptimizer requires cov_matrix in OptimizationInputs.")

        cov_matrix = np.asarray(inputs.cov_matrix, dtype=np.float64)
        n = cov_matrix.shape[0]

        if n < 2:
            # A single asset has no cluster structure to speak of --
            # trivially, all weight goes to it.
            weights = constraints.project(np.array([1.0]))
            return OptimizationResult(weights=weights, method_name="HRP", converged=True, diagnostics={"n_assets": n})

        corr_matrix = _cov_to_corr(cov_matrix)
        distance_matrix = _correlation_distance(corr_matrix)

        condensed = squareform(distance_matrix, checks=False)
        link_matrix = linkage(condensed, method="single")

        sorted_indices = _quasi_diagonalize(link_matrix)
        raw_weights = _recursive_bisection(cov_matrix, sorted_indices)

        # raw_weights already sums to ~1 by construction (each split
        # conserves the parent's budget), but explicit renormalisation
        # guards against small floating-point drift accumulating over
        # many recursion levels.
        raw_weights = raw_weights / raw_weights.sum()

        weights = constraints.project(raw_weights)

        return OptimizationResult(
            weights=weights,
            method_name="HRP",
            converged=True,  # deterministic given the covariance input -- no iterative search to fail to converge
            diagnostics={"leaf_order": sorted_indices},
        )