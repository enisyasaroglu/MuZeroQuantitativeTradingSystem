import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.methods.pso_optimizer import PSOPortfolioOptimizer, PSOOptimizerAdapter
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import OptimizationInputs


def _toy_inputs(n_assets=4, seed=0):
    rng = np.random.RandomState(seed)
    expected_returns = rng.uniform(0.0001, 0.001, size=n_assets)
    cov_matrix = np.cov(rng.normal(0, 0.01, size=(50, n_assets)), rowvar=False)
    return expected_returns, cov_matrix


def test_optimize_without_constraints_matches_prior_behaviour():
    er, cov = _toy_inputs()
    w_a = PSOPortfolioOptimizer(n_particles=20, n_iterations=15, random_state=1).optimize(er, cov)
    w_b = PSOPortfolioOptimizer(n_particles=20, n_iterations=15, random_state=1).optimize(er, cov, constraints=None)
    assert np.allclose(w_a, w_b)


def test_constraints_enforced_throughout_search():
    er, cov = _toy_inputs(n_assets=6)
    constraints = PortfolioConstraints(max_assets=2)
    pso = PSOPortfolioOptimizer(n_particles=20, n_iterations=15, random_state=1)
    w = pso.optimize(er, cov, constraints=constraints)
    assert constraints.is_feasible(w)


def test_adapter_satisfies_protocol_shape():
    er, cov = _toy_inputs()
    adapter = PSOOptimizerAdapter(pso=PSOPortfolioOptimizer(n_particles=10, n_iterations=10, random_state=1))
    result = adapter.optimize(OptimizationInputs(expected_returns=er, cov_matrix=cov), PortfolioConstraints())
    assert result.weights.shape == er.shape
    assert result.method_name == "PSO"