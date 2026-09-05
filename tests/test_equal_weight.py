import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.methods.equal_weight import EqualWeightOptimizer
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import OptimizationInputs


def test_equal_weight_sums_to_one_and_is_uniform():
    opt = EqualWeightOptimizer()
    inputs = OptimizationInputs(expected_returns=np.zeros(5))
    result = opt.optimize(inputs, PortfolioConstraints())
    assert np.allclose(result.weights, 0.2)
    assert abs(result.weights.sum() - 1.0) < 1e-9


def test_derives_n_assets_from_cov_matrix_if_expected_returns_missing():
    opt = EqualWeightOptimizer()
    inputs = OptimizationInputs(cov_matrix=np.eye(4))
    result = opt.optimize(inputs, PortfolioConstraints())
    assert len(result.weights) == 4


def test_raises_if_no_inputs_provided():
    opt = EqualWeightOptimizer()
    try:
        opt.optimize(OptimizationInputs(), PortfolioConstraints())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_respects_constraints_projection():
    opt = EqualWeightOptimizer()
    inputs = OptimizationInputs(expected_returns=np.zeros(4))
    constraints = PortfolioConstraints(max_assets=2)
    result = opt.optimize(inputs, constraints)
    assert constraints.is_feasible(result.weights)
    assert np.sum(result.weights > 1e-9) == 2